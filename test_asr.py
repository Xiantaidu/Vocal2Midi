import librosa
import sys
import logging
import argparse
from funasr import AutoModel

def test_asr(audio_path, use_vad):
    print(f"Loading audio from {audio_path}...")
    # 读取音频
    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(waveform) / sr
    print(f"Audio duration: {duration:.2f} seconds")

    print("Resampling to 16000Hz (required by ASR)...")
    if sr != 16000:
        waveform_16k = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    else:
        waveform_16k = waveform

    print("Loading FunASR model...")
    logging.getLogger("funasr").setLevel(logging.ERROR)
    
    # 基础配置，与 auto_lyric.py 中一致
    model_kwargs = {
        "model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "model_revision": "v2.0.4",
        "disable_update": True,
        "disable_pbar": False
    }

    if use_vad:
        print("VAD is ENABLED. Long audio will be sliced internally by FunASR.")
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["punc_model"] = "ct-punc"
    else:
        print("VAD is DISABLED. This matches the exact behavior in auto_lyric.py.")

    asr_model = AutoModel(**model_kwargs)

    print("Running ASR inference...")
    res = asr_model.generate(input=[waveform_16k], cache={}, is_final=True)
    
    text = res[0].get('text', '') if isinstance(res[0], dict) else str(res[0])
    
    print("\n" + "="*30 + " ASR Output " + "="*30)
    print(text)
    print("="*72 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FunASR with long audio files.")
    parser.add_argument("audio_path", help="Path to the audio file to test.")
    parser.add_argument("--vad", action="store_true", help="Enable VAD (Voice Activity Detection) to handle long audio.")
    
    args = parser.parse_args()
    
    test_asr(args.audio_path, args.vad)
