import numpy as np
import argparse
import os
from scipy.io import wavfile

def npy_to_wav(npy_file, output_wav=None, sample_rate=16000):
    """将numpy数组格式的音频文件转换为WAV格式"""
    # 读取numpy数组
    audio_data = np.load(npy_file)
    
    # 检查数据类型，确保是int16格式
    if audio_data.dtype != np.int16:
        # 如果不是int16，转换为int16
        audio_data = audio_data.astype(np.int16)
    
    # 生成输出文件名
    if output_wav is None:
        output_wav = os.path.splitext(npy_file)[0] + '.wav'
    
    # 保存为WAV文件
    wavfile.write(output_wav, sample_rate, audio_data)
    
    print(f"✅ 转换完成！")
    print(f"   输入文件: {npy_file}")
    print(f"   输出文件: {output_wav}")
    print(f"   采样率: {sample_rate} Hz")
    print(f"   音频时长: {len(audio_data)/sample_rate:.2f}秒")
    print(f"   音量范围: {np.min(audio_data)} 到 {np.max(audio_data)}")
    print(f"   平均音量: {np.abs(audio_data).mean():.2f}")
    
    return output_wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将numpy数组格式的音频文件转换为WAV格式")
    parser.add_argument("audio_file", type=str, help="numpy音频文件路径")
    parser.add_argument("--output", type=str, help="输出WAV文件路径")
    parser.add_argument("--sample_rate", type=int, default=16000, help="采样率，默认16000 Hz")
    parser.add_argument("--play", action="store_true", help="转换后自动播放")
    
    args = parser.parse_args()
    
    print("=== numpy音频转WAV工具 ===")
    
    # 转换为WAV
    wav_file = npy_to_wav(args.audio_file, args.output, args.sample_rate)
    
    # 自动播放（如果系统支持）
    if args.play:
        print("\n🎵 正在播放音频...")
        try:
            # 使用系统默认播放器播放
            if os.name == 'nt':  # Windows
                os.startfile(wav_file)
            elif os.name == 'posix':  # macOS/Linux
                os.system(f'open "{wav_file}"')
            print("✅ 播放命令已发送，请查看系统播放器")
        except Exception as e:
            print(f"⚠️  自动播放失败: {e}")
            print(f"   请手动播放文件: {wav_file}")
    else:
        print(f"\n🎵 可以使用以下命令播放:")
        if os.name == 'nt':  # Windows
            print(f'   start "" "{wav_file}"')
        elif os.name == 'posix':  # macOS/Linux
            print(f'   open "{wav_file}"  # macOS')
            print(f'   aplay "{wav_file}"  # Linux')
