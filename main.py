import argparse
import logging
import signal
import time
from dataclasses import dataclass

import numpy as np
import pyaudio
from funasr import AutoModel

# 屏蔽繁杂日志
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("funasr").setLevel(logging.ERROR)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    chunk: int = 960  # 60ms
    window_size: int = 4800  # 300ms
    step_size: int = 960  # 60ms
    format: int = pyaudio.paInt16
    channels: int = 1


@dataclass(frozen=True)
class StreamConfig:
    chunk_size: tuple = (0, 10, 5)
    encoder_chunk_look_back: int = 2
    decoder_chunk_look_back: int = 1


@dataclass
class RuntimeState:
    is_running: bool = True
    all_audio_data: list = None

    def __post_init__(self):
        if self.all_audio_data is None:
            self.all_audio_data = []


def signal_handler(sig, frame, state: RuntimeState):
    print("\n收到退出信号，正在停止...")
    state.is_running = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FunASR 实时语音识别系统")
    parser.add_argument("--audio_file", type=str, help="录音文件路径，用于流式重放测试")
    parser.add_argument("--benchmark", type=str, help="基准文本，用于自动验证")
    parser.add_argument("--mic", action="store_true", help="使用麦克风实时输入")
    parser.add_argument("--gain", type=float, default=3.0, help="音频增益调整，默认3.0")
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="音量阈值，低于此值视为静音，默认100.0",
    )
    return parser


def print_banner(audio_cfg: AudioConfig, gain: float, volume_threshold: float) -> None:
    print("=== FunASR 实时语音识别系统 ===")
    print("支持实时麦克风输入、滑动窗口、录音保存和自动验证功能\n")
    print("🎛️  配置参数:")
    print(
        f"   CHUNK: {audio_cfg.chunk} ({audio_cfg.chunk / audio_cfg.sample_rate * 1000:.0f}ms)"
    )
    print(
        f"   滑动窗口: {audio_cfg.window_size} "
        f"({audio_cfg.window_size / audio_cfg.sample_rate * 1000:.0f}ms)"
    )
    print(
        f"   滑动步长: {audio_cfg.step_size} "
        f"({audio_cfg.step_size / audio_cfg.sample_rate * 1000:.0f}ms)"
    )
    print(f"   采样率: {audio_cfg.sample_rate}Hz")
    print(f"   音频增益: {gain}")
    print(f"   音量阈值: {volume_threshold}")
    print()


def load_model() -> AutoModel:
    print("正在加载模型...")
    model = AutoModel(
        model="paraformer-zh-streaming",
        model_revision="v2.0.4",
        disable_update=True,
        verbose=False,
    )
    print("模型加载完成！\n")
    return model


def preprocess_audio(audio_chunk: np.ndarray, gain: float) -> np.ndarray:
    processed = audio_chunk.astype(np.float32) * gain
    processed = np.clip(processed, -32768, 32767)
    return processed.astype(np.int16)


def save_recording(audio_data: np.ndarray) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.npy"
    np.save(filename, audio_data)
    print(f"\n录音已保存为: {filename}")
    return filename


def calculate_similarity(text1: str, text2: str) -> float:
    """使用集合相似度计算文本相似度"""
    if not text1 or not text2:
        return 0.0
    set1 = set(text1)
    set2 = set(text2)
    common = set1.intersection(set2)
    if not set2:
        return 0.0
    return len(common) / len(set2)


def merge_stream_text(current_text: str, new_text: str) -> str:
    if not new_text:
        return current_text
    if not current_text:
        return new_text
    if new_text in current_text:
        return current_text

    # 处理流式输出的前后重叠，避免重复字
    max_overlap = 0
    max_len = min(len(current_text), len(new_text))
    for i in range(1, max_len + 1):
        if current_text[-i:] == new_text[:i]:
            max_overlap = i
    return current_text + new_text[max_overlap:]


def stream_recognition_from_samples(
    sample_iter,
    model: AutoModel,
    audio_cfg: AudioConfig,
    stream_cfg: StreamConfig,
    state: RuntimeState,
    gain: float,
    volume_threshold: float,
    label: str = "",
) -> str:
    cache = {}
    full_text = ""
    audio_cache = []
    # stream_cfg.chunk_size[1] 以 60ms 为单位，10 -> 600ms
    chunk_stride_samples = int(stream_cfg.chunk_size[1] * 960)
    max_buffer = max(audio_cfg.window_size, chunk_stride_samples)

    if label:
        print(label)

    for audio_chunk in sample_iter:
        if not state.is_running:
            break

        # 记录全部音频用于最终识别/回放
        state.all_audio_data.extend(audio_chunk)

        # 累积缓存，直到达到模型需要的步长
        audio_cache.extend(audio_chunk)
        if len(audio_cache) > max_buffer:
            audio_cache = audio_cache[-max_buffer:]

        current_volume = np.abs(audio_chunk).mean()
        if current_volume < volume_threshold:
            volume_status = "🔇 静音"
        elif current_volume < volume_threshold * 2:
            volume_status = "🔉 小声"
        elif current_volume < volume_threshold * 5:
            volume_status = "🔊 正常"
        else:
            volume_status = "🔊🔊 大声"

        if len(audio_cache) < chunk_stride_samples:
            debug_info = (
                f"\r{volume_status} | 音量: {current_volume:5.1f} | "
                f"缓存: {len(audio_cache):4d} | 识别结果: {full_text}"
            )
            print(debug_info, end="", flush=True)
            continue

        processed_audio = preprocess_audio(
            np.array(audio_cache[:chunk_stride_samples]),
            gain=gain,
        )
        audio_cache = audio_cache[chunk_stride_samples:]

        recognize_start = time.time()
        res = model.generate(
            input=processed_audio,
            cache=cache,
            is_final=False,
            chunk_size=list(stream_cfg.chunk_size),
            encoder_chunk_look_back=stream_cfg.encoder_chunk_look_back,
            decoder_chunk_look_back=stream_cfg.decoder_chunk_look_back,
            disable_pbar=True,
            disable_log=True,
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
            chunk_size=list(stream_cfg.chunk_size),
            encoder_chunk_look_back=stream_cfg.encoder_chunk_look_back,
            decoder_chunk_look_back=stream_cfg.decoder_chunk_look_back,
            disable_pbar=True,
            disable_log=True,
        )
    else:
        res = model.generate(
            input=np.array([], dtype=np.int16),
            cache=cache,
            is_final=True,
            chunk_size=list(stream_cfg.chunk_size),
            encoder_chunk_look_back=stream_cfg.encoder_chunk_look_back,
            decoder_chunk_look_back=stream_cfg.decoder_chunk_look_back,
            disable_pbar=True,
            disable_log=True,
        )

    if res and res[0]["text"]:
        full_text = merge_stream_text(full_text, res[0]["text"])
        print(f"\n📝 最终流式识别结果: {full_text}")

    return full_text


def final_full_recognition(
    model: AutoModel,
    stream_cfg: StreamConfig,
    audio_data: list,
    gain: float,
) -> str:
    if not audio_data:
        return ""

    print("\n\n🔍 使用完整录音进行最终识别...")
    full_audio = np.array(audio_data)
    processed_full = preprocess_audio(full_audio, gain=gain)

    res = model.generate(
        input=processed_full,
        cache={},
        is_final=True,
        chunk_size=list(stream_cfg.chunk_size),
        encoder_chunk_look_back=stream_cfg.encoder_chunk_look_back,
        decoder_chunk_look_back=stream_cfg.decoder_chunk_look_back,
        disable_pbar=True,
        disable_log=True,
    )

    if res and res[0]["text"]:
        final_text = res[0]["text"]
        print(f"📝 完整录音识别结果: {final_text}")
        return final_text
    return ""


def real_time_recognition(
    model: AutoModel,
    audio_cfg: AudioConfig,
    stream_cfg: StreamConfig,
    state: RuntimeState,
    gain: float,
    volume_threshold: float,
) -> str:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=audio_cfg.format,
        channels=audio_cfg.channels,
        rate=audio_cfg.sample_rate,
        input=True,
        frames_per_buffer=audio_cfg.chunk,
    )

    print("🎤 开始实时录音和识别...")
    print("   按 Ctrl+C 停止\n")

    def mic_iter():
        while state.is_running:
            data = stream.read(audio_cfg.chunk, exception_on_overflow=False)
            yield np.frombuffer(data, dtype=np.int16)

    try:
        result = stream_recognition_from_samples(
            mic_iter(),
            model=model,
            audio_cfg=audio_cfg,
            stream_cfg=stream_cfg,
            state=state,
            gain=gain,
            volume_threshold=volume_threshold,
        )
    except Exception as e:
        print(f"\n录音出错: {e}")
        result = ""
    finally:
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存录音
        if state.all_audio_data:
            save_recording(np.array(state.all_audio_data))

    return result


def file_streaming_recognition(
    audio_file: str,
    model: AutoModel,
    audio_cfg: AudioConfig,
    stream_cfg: StreamConfig,
    state: RuntimeState,
    gain: float,
    volume_threshold: float,
) -> str:
    print(f"📁 使用录音文件进行测试: {audio_file}")
    audio_data = np.load(audio_file)
    state.all_audio_data = audio_data.tolist()

    print(f"音频时长: {len(audio_data) / audio_cfg.sample_rate:.2f}秒")
    print(f"平均音量: {np.abs(audio_data).mean():.2f}")

    def file_iter():
        for i in range(0, len(audio_data), audio_cfg.chunk):
            yield audio_data[i : i + audio_cfg.chunk]

    final_text = stream_recognition_from_samples(
        file_iter(),
        model=model,
        audio_cfg=audio_cfg,
        stream_cfg=stream_cfg,
        state=state,
        gain=gain,
        volume_threshold=volume_threshold,
        label="\n开始流式识别...",
    )
    print(f"\n最终识别结果: {final_text}")
    return final_text


def run_benchmark(result: str, benchmark: str) -> None:
    if not benchmark or not result:
        return

    print("\n✅ 自动验证:")
    print(f"   基准文本: {benchmark}")
    print(f"   识别结果: {result}")
    similarity = calculate_similarity(result, benchmark)
    print(f"   相似度: {similarity:.2f}")
    if similarity >= 0.7:
        print("   验证状态: 通过 ✅")
    else:
        print("   验证状态: 未通过 ❌")


def main() -> None:
    args = build_arg_parser().parse_args()

    audio_cfg = AudioConfig()
    stream_cfg = StreamConfig()
    state = RuntimeState()

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, state))

    print_banner(audio_cfg, gain=args.gain, volume_threshold=args.threshold)
    model = load_model()

    if args.audio_file:
        result = file_streaming_recognition(
            args.audio_file,
            model=model,
            audio_cfg=audio_cfg,
            stream_cfg=stream_cfg,
            state=state,
            gain=args.gain,
            volume_threshold=args.threshold,
        )
    elif args.mic:
        result = real_time_recognition(
            model=model,
            audio_cfg=audio_cfg,
            stream_cfg=stream_cfg,
            state=state,
            gain=args.gain,
            volume_threshold=args.threshold,
        )
        final_text = final_full_recognition(
            model=model,
            stream_cfg=stream_cfg,
            audio_data=state.all_audio_data,
            gain=args.gain,
        )
        if final_text:
            print(f"🔄 实时识别结果: {result}")
            result = final_text
    else:
        print("请指定 --mic 使用麦克风，或 --audio_file 指定音频文件")
        return

    run_benchmark(result, args.benchmark)


if __name__ == "__main__":
    main()
    print("\n=== 程序结束 ===")
