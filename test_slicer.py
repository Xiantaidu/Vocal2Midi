import os
import sys
import librosa
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.slicer2 import Slicer

def main():
    # ==========================
    # 在这里修改你的测试参数
    # ==========================
    INPUT_AUDIO = r"R:\新建文件夹31111\起风了 - 买辣椒也用券_vocals_noreverb.wav"          # 输入音频文件的路径
    OUTPUT_DIR = r"R:\autolyric_results (1)"   # 切片结果保存的文件夹
    
    # === 一级切片参数 (基础) ===
    THRESHOLD = -30.0                   # 静音判断阈值 (dB)
    MIN_LENGTH = 5000                   # 每个切片的最短长度 (ms)
    MIN_INTERVAL = 200                  # 允许切分的最小静音间隔 (ms)
    MAX_SIL_KEPT = 150                  # 切片两端最多保留的静音长度 (ms)
    
    # === 二级切片参数 (激进) ===
    # 当一级切片后的块长度仍大于 30 秒时触发
    AGG_THRESHOLD = -20.0               
    AGG_MIN_LENGTH = 4000
    AGG_MIN_INTERVAL = 200
    AGG_MAX_SIL_KEPT = 150
    # ==========================

    input_path = Path(INPUT_AUDIO)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading audio: {input_path}")
    # Load audio (mono)
    waveform, sr = librosa.load(input_path, sr=None, mono=True)
    duration = len(waveform) / sr
    print(f"Audio loaded. Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
    
    print("\n--- Slicer Parameters ---")
    print(f"Threshold:    {THRESHOLD} dB")
    print(f"Min Length:   {MIN_LENGTH} ms")
    print(f"Min Interval: {MIN_INTERVAL} ms")
    print(f"Max Sil Kept: {MAX_SIL_KEPT} ms")
    print("-------------------------\n")

    # Initialize Slicer
    slicer = Slicer(
        sr=sr,
        threshold=THRESHOLD,
        min_length=MIN_LENGTH,
        min_interval=MIN_INTERVAL,
        max_sil_kept=MAX_SIL_KEPT
    )

    print("Slicing audio (Level 1)...")
    initial_chunks = slicer.slice(waveform)
    
    final_chunks = []
    
    for chunk in initial_chunks:
        chunk_len = len(chunk['waveform'])
        chunk_dur = chunk_len / sr
        
        if chunk_dur <= 30.0:
            final_chunks.append(chunk)
            continue
            
        print(f"  [Warning] Chunk > 30s ({chunk_dur:.2f}s). Applying aggressive slicing (Level 2)...")
        
        slicer_agg = Slicer(
            sr=sr,
            threshold=AGG_THRESHOLD,
            min_length=AGG_MIN_LENGTH,
            min_interval=AGG_MIN_INTERVAL,
            max_sil_kept=AGG_MAX_SIL_KEPT
        )
        sub_chunks = slicer_agg.slice(chunk['waveform'])
        
        base_offset = chunk['offset']
        for sub in sub_chunks:
            sub['offset'] += base_offset
            
            sub_len = len(sub['waveform'])
            sub_dur = sub_len / sr
            
            if sub_dur <= 30.0:
                final_chunks.append(sub)
                continue
                
            print(f"  [Warning] Sub-chunk still > 30s ({sub_dur:.2f}s). Forcing split at min energy (Level 3)...")
            
            sub_wav = sub['waveform']
            if len(sub_wav.shape) > 1:
                sub_wav_mono = np.mean(sub_wav, axis=0)
            else:
                sub_wav_mono = sub_wav
                
            hop_len = 512
            frame_len = 2048
            
            rms = librosa.feature.rms(y=sub_wav_mono, frame_length=frame_len, hop_length=hop_len, center=True)
            if rms.ndim > 1:
                rms = rms[0]
                
            n_frames = len(rms)
            start_frame = int(n_frames * 0.2)
            end_frame = int(n_frames * 0.8)
            
            if end_frame > start_frame:
                min_idx = np.argmin(rms[start_frame:end_frame]) + start_frame
            else:
                min_idx = n_frames // 2
                
            split_sample = min_idx * hop_len
            
            if len(sub_wav.shape) > 1:
                part1_wav = sub_wav[:, :split_sample]
                part2_wav = sub_wav[:, split_sample:]
            else:
                part1_wav = sub_wav[:split_sample]
                part2_wav = sub_wav[split_sample:]
                
            final_chunks.append({
                'offset': sub['offset'],
                'waveform': part1_wav
            })
            final_chunks.append({
                'offset': sub['offset'] + split_sample / sr,
                'waveform': part2_wav
            })
    
    print(f"\nFinal result: Sliced into {len(final_chunks)} chunks. Exporting to '{out_dir}'...")
    
    for i, chunk in enumerate(final_chunks):
        offset = chunk['offset']
        chunk_wav = chunk['waveform']
        chunk_dur = len(chunk_wav) / sr
        
        print(f"  Chunk {i:03d}: offset = {offset:.2f}s, duration = {chunk_dur:.2f}s")
        
        out_filename = out_dir / f"{input_path.stem}_chunk_{i:03d}.wav"
        sf.write(out_filename, chunk_wav, sr)

    print(f"\nDone. Sliced audio saved in: {out_dir.absolute()}")

if __name__ == "__main__":
    main()
