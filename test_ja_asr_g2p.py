import os
import pathlib
import librosa
import sys
import torch
import soundfile as sf

# Add path
ROOT_DIR = pathlib.Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "inference" / "vendor"
sys.path.insert(0, str(VENDOR_DIR / "HubertFA"))

from inference.slicer_api import slice_audio
from inference.vendor.LyricFA.tools.JaG2p import JaG2p

def load_qwen_model(model_path, device="cuda"):
    print(f"Loading Qwen3-ASR PyTorch model from '{model_path}' on {device}...")
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    return model

def main():
    audio_path = r"R:\新建文件夹31111\風になる - つじあやの_vocals_noreverb_Vocals.wav"
    output_dir = pathlib.Path("temp_ja_test")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading audio: {audio_path}")
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
        
    sr = 44100
    waveform, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    print("Slicing audio...")
    chunks = slice_audio(waveform, sr, "默认切片")
    print(f"Got {len(chunks)} chunks.")
    
    g2p_model = JaG2p()
    
    asr_model_path = r"C:\Users\Xiantaidu\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
    asr_model = load_qwen_model(asr_model_path, device="cuda")
    
    asr_batch_size = 4
    
    audio_paths = []
    chunk_indices = []
    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx:03d}"
        chunk_path = output_dir / f"{stem}.wav"
        sf.write(chunk_path, chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)
        
    all_results = []
    for i in range(0, len(audio_paths), asr_batch_size):
        batch_audio_paths = audio_paths[i:i+asr_batch_size]
        print(f"  Processing ASR batch {i//asr_batch_size + 1}/{(len(audio_paths) - 1)//asr_batch_size + 1}...")
        
        with torch.cuda.amp.autocast():
            batch_results = asr_model.transcribe(audio=batch_audio_paths, language="Japanese") 
        all_results.extend(batch_results)
        
    for idx, res in enumerate(all_results):
        stem = f"chunk_{chunk_indices[idx]:03d}"
        if res is None or not getattr(res, 'text', '').strip():
            print(f"[{stem}] Empty ASR")
            continue
            
        text = res.text
        pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
        
        lab_path = output_dir / f"{stem}.lab"
        lab_path.write_text(pinyin_str, encoding="utf-8")
        print(f"[{stem}] ASR: {text} -> LAB: {pinyin_str}")
        
    print(f"\nDone. Results saved to {output_dir.absolute()}")

if __name__ == '__main__':
    main()
