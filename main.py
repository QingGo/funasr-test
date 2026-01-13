import pyaudio
import numpy as np
from funasr import AutoModel
import logging
import time
import argparse
import signal
import os

# 屏蔽繁杂日志
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("funasr").setLevel(logging.ERROR)

# 全局变量
is_running = True
cleanup_called = False
all_audio_data = []

# 信号处理
def signal_handler(sig, frame):
    global is_running
    print("\n收到退出信号，正在停止...")
    is_running = False

# 命令行参数解析
parser = argparse.ArgumentParser(description="FunASR 实时语音识别系统")
parser.add_argument("--audio_file", type=str, help="录音文件路径，用于流式重放测试")
parser.add_argument("--benchmark", type=str, help="基准文本，用于自动验证")
parser.add_argument("--mic", action="store_true", help="使用麦克风实时输入")
parser.add_argument("--gain", type=float, default=3.0, help="音频增益调整，默认3.0")
parser.add_argument("--threshold", type=float, default=100.0, help="音量阈值，低于此值视为静音，默认100.0")
args = parser.parse_args()

print("=== FunASR 实时语音识别系统 ===")
print("支持实时麦克风输入、滑动窗口、录音保存和自动验证功能\n")

# 音频配置
SAMPLE_RATE = 16000
CHUNK = 960  # 60ms，每次读取的音频块大小
WINDOW_SIZE = 4800  # 300ms，滑动窗口大小
STEP_SIZE = 960     # 60ms，滑动步长
FORMAT = pyaudio.paInt16
CHANNELS = 1
gain = args.gain
volume_threshold = args.threshold

print(f"🎛️  配置参数:")
print(f"   CHUNK: {CHUNK} ({CHUNK/SAMPLE_RATE*1000:.0f}ms)")
print(f"   滑动窗口: {WINDOW_SIZE} ({WINDOW_SIZE/SAMPLE_RATE*1000:.0f}ms)")
print(f"   滑动步长: {STEP_SIZE} ({STEP_SIZE/SAMPLE_RATE*1000:.0f}ms)")
print(f"   采样率: {SAMPLE_RATE}Hz")
print(f"   音频增益: {gain}")
print(f"   音量阈值: {volume_threshold}")
print()

# 加载模型
print("正在加载模型...")
model = AutoModel(
    model="paraformer-zh-streaming", 
    model_revision="v2.0.4",
    disable_update=True,
    verbose=False
)
print("模型加载完成！\n")

# 音频预处理
def preprocess_audio(audio_chunk, gain=3.0):
    processed = audio_chunk.astype(np.float32) * gain
    processed = np.clip(processed, -32768, 32767)
    return processed.astype(np.int16)

# 保存录音
def save_recording(audio_data):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.npy"
    np.save(filename, audio_data)
    print(f"\n录音已保存为: {filename}")
    return filename

# 相似度计算
def calculate_similarity(text1, text2):
    """使用集合相似度计算文本相似度"""
    if not text1 or not text2:
        return 0.0
    set1 = set(text1)
    set2 = set(text2)
    common = set1.intersection(set2)
    if not set2:
        return 0.0
    return len(common) / len(set2)

def merge_stream_text(current_text, new_text):
    if not new_text:
        return current_text
    if not current_text:
        return new_text
    if new_text in current_text:
        return current_text

    max_overlap = 0
    max_len = min(len(current_text), len(new_text))
    for i in range(1, max_len + 1):
        if current_text[-i:] == new_text[:i]:
            max_overlap = i
    return current_text + new_text[max_overlap:]

def stream_recognition_from_samples(sample_iter, label=""):
    global all_audio_data

    cache = {}
    full_text = ""
    audio_cache = []
    chunk_size = [0, 10, 5]
    chunk_stride_samples = int(chunk_size[1] * 960)  # 600ms
    max_buffer = max(WINDOW_SIZE, chunk_stride_samples)

    if label:
        print(label)

    for audio_chunk in sample_iter:
        if not is_running:
            break

        # 保存到全局音频数据中
        all_audio_data.extend(audio_chunk)

        # 添加到滑动窗口缓存
        audio_cache.extend(audio_chunk)
        if len(audio_cache) > max_buffer:
            audio_cache = audio_cache[-max_buffer:]

        # 计算当前音量
        current_volume = np.abs(audio_chunk).mean()

        # 音量状态指示
        if current_volume < volume_threshold:
            volume_status = "🔇 静音"
        elif current_volume < volume_threshold * 2:
            volume_status = "🔉 小声"
        elif current_volume < volume_threshold * 5:
            volume_status = "🔊 正常"
        else:
            volume_status = "🔊🔊 大声"

        # 按 600ms 步长累积后送入模型，避免过短块导致输出卡在“嗯”
        if len(audio_cache) < chunk_stride_samples:
            debug_info = (
                f"\r{volume_status} | 音量: {current_volume:5.1f} | "
                f"缓存: {len(audio_cache):4d} | 识别结果: {full_text}"
            )
            print(debug_info, end="", flush=True)
            continue

        processed_audio = preprocess_audio(np.array(audio_cache[:chunk_stride_samples]), gain=gain)
        audio_cache = audio_cache[chunk_stride_samples:]

        recognize_start = time.time()
        res = model.generate(
            input=processed_audio,
            cache=cache,
            is_final=False,
            chunk_size=chunk_size,
            encoder_chunk_look_back=2,
            decoder_chunk_look_back=1,
            disable_pbar=True,
            disable_log=True
        )
        recognize_delay = (time.time() - recognize_start) * 1000

        partial_text = res[0]["text"] if (res and res[0]["text"]) else ""
        if partial_text:
            full_text = merge_stream_text(full_text, partial_text)

        debug_info = (
            f"\r{volume_status} | 音量: {current_volume:5.1f} | "
            f"延迟: {recognize_delay:4.1f}ms | 识别结果: {full_text}"
        )
        print(debug_info, end="", flush=True)

    # flush remaining cache
    if audio_cache:
        processed_audio = preprocess_audio(np.array(audio_cache), gain=gain)
        res = model.generate(
            input=processed_audio,
            cache=cache,
            is_final=True,
            chunk_size=chunk_size,
            encoder_chunk_look_back=2,
            decoder_chunk_look_back=1,
            disable_pbar=True,
            disable_log=True
        )
    else:
        res = model.generate(
            input=np.array([], dtype=np.int16),
            cache=cache,
            is_final=True,
            chunk_size=chunk_size,
            encoder_chunk_look_back=2,
            decoder_chunk_look_back=1,
            disable_pbar=True,
            disable_log=True
        )

    if res and res[0]["text"]:
        full_text = merge_stream_text(full_text, res[0]["text"])
        print(f"\n📝 最终流式识别结果: {full_text}")

    return full_text

# 麦克风实时录音和识别
def real_time_recognition():
    global is_running, all_audio_data

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎤 开始实时录音和识别...")
    print("   按 Ctrl+C 停止\n")

    def mic_iter():
        while is_running:
            data = stream.read(CHUNK, exception_on_overflow=False)
            yield np.frombuffer(data, dtype=np.int16)

    try:
        result = stream_recognition_from_samples(mic_iter())
    except Exception as e:
        print(f"\n录音出错: {e}")
        result = ""
    finally:
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存录音
        if all_audio_data:
            save_recording(np.array(all_audio_data))

        # 使用完整音频进行最终识别
        if all_audio_data:
            print("\n\n🔍 使用完整录音进行最终识别...")
            full_audio = np.array(all_audio_data)
            processed_full = preprocess_audio(full_audio)

            full_cache = {}
            res = model.generate(
                input=processed_full,
                cache=full_cache,
                is_final=True,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=2,
                decoder_chunk_look_back=1,
                disable_pbar=True,
                disable_log=True
            )

            if res and res[0]['text']:
                final_text = res[0]['text']
                print(f"📝 完整录音识别结果: {final_text}")
                if result:
                    print(f"🔄 实时识别结果: {result}")
                return final_text

    return result

# 音频文件流式重放
def file_streaming_recognition(audio_file):
    global all_audio_data
    
    print(f"📁 使用录音文件进行测试: {audio_file}")
    audio_data = np.load(audio_file)
    all_audio_data = audio_data.tolist()
    
    print(f"音频时长: {len(audio_data)/16000:.2f}秒")
    print(f"平均音量: {np.abs(audio_data).mean():.2f}")
    
    # 1. 按实时流式方式分块送入模型，复现实时行为
    def file_iter():
        for i in range(0, len(audio_data), CHUNK):
            yield audio_data[i:i + CHUNK]

    final_text = stream_recognition_from_samples(file_iter(), label="\n开始流式识别...")
    print(f"\n最终识别结果: {final_text}")
    return final_text

# 主流程
def main():
    global is_running
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    result = ""
    
    if args.audio_file:
        # 使用音频文件测试
        result = file_streaming_recognition(args.audio_file)
    elif args.mic:
        # 使用麦克风实时输入
        result = real_time_recognition()
    else:
        print("请指定 --mic 使用麦克风，或 --audio_file 指定音频文件")
        return
    
    # 自动验证
    if args.benchmark and result:
        print(f"\n✅ 自动验证:")
        print(f"   基准文本: {args.benchmark}")
        print(f"   识别结果: {result}")
        similarity = calculate_similarity(result, args.benchmark)
        print(f"   相似度: {similarity:.2f}")
        if similarity >= 0.7:
            print(f"   验证状态: 通过 ✅")
        else:
            print(f"   验证状态: 未通过 ❌")

if __name__ == "__main__":
    main()
    print("\n=== 程序结束 ===")
