import sys
import os
from pathlib import Path

# Add Qwen3-ASR-onnx-main to sys.path
qwen_asr_path = Path("experiments/Qwen3-ASR-1.7B-ONNX/Qwen3-ASR-onnx-main").resolve()
sys.path.insert(0, str(qwen_asr_path))

from onnx_asr_service import OnnxAsrRuntime

def main():
    # The ONNX model dir is probably "experiments/Qwen3-ASR-1.7B-ONNX" or "experiments/Qwen3-ASR-1.7B-ONNX/Qwen3-ASR-0.6B"
    # Let's assume the onnx-dir is the parent dir or we'll pass it explicitly
    # Wait, let's look for metadata.json in E:\Vocal2Midi\experiments\Qwen3-ASR-1.7B-ONNX
    # Let's pass the ONNX dir manually
    onnx_dir = Path("experiments/Qwen3-ASR-1.7B-ONNX").resolve()
    
    # We will test on a dummy audio file.
    import numpy as np
    import soundfile as sf
    
    dummy_audio_path = "dummy_test_audio.wav"
    sr = 16000
    # Generate 1 second of random noise
    noise = np.random.randn(sr).astype(np.float32)
    sf.write(dummy_audio_path, noise, sr)
    
    try:
        print(f"Loading model from: {onnx_dir}")
        runtime = OnnxAsrRuntime(
            onnx_dir=onnx_dir,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],  # Using DML for Windows/AMD/etc as GAME does
            max_new_tokens=100,
        )
        print("Model loaded successfully.")
        
        print("Running inference...")
        result = runtime.transcribe_input(dummy_audio_path)
        print("Inference successful. Result:")
        print(result)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)

if __name__ == "__main__":
    main()
