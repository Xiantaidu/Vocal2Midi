import torch
import soundfile as sf

def load_qwen_model(model_path, device="cuda"):
    """
    Loads the Qwen3-ASR model using PyTorch.
    """
    print(f"Loading Qwen3-ASR PyTorch model from '{model_path}' on {device}...")
    from qwen_asr import Qwen3ASRModel

    try:
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
    except Exception as e:
        raise RuntimeError(f"Error loading Qwen3-ASR model: {e}\nPlease ensure you have run 'pip install -U qwen-asr'.")
    return model

def batch_transcribe_asr(chunks, sr, asr_model, temp_dir_path, asr_batch_size, language, cancel_checker=None):
    """Saves chunks to temp_dir and runs batched ASR transcription."""
    asr_lang = "Japanese" if language == "ja" else "Chinese"
    print(f"[ASR API] Running ASR with PyTorch Qwen (Batch Size: {asr_batch_size}, Language: {asr_lang})...")

    audio_paths = []
    chunk_indices = []

    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        chunk_path = temp_dir_path / f"{stem}.wav"
        # 暂时先用temp_dir交换数据，保存wav供ASR和HFA读取
        sf.write(chunk_path, chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)

    all_results = []
    for i in range(0, len(audio_paths), asr_batch_size):
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR 任务已取消")
        batch_audio_paths = audio_paths[i:i+asr_batch_size]
        print(f"  Processing ASR batch {i//asr_batch_size + 1}/{(len(audio_paths) - 1)//asr_batch_size + 1}...")
        
        try:
            with torch.cuda.amp.autocast():
                batch_results = asr_model.transcribe(audio=batch_audio_paths, language=asr_lang)
            all_results.extend(batch_results)
        except Exception as e:
            print(f"Error during Qwen ASR transcription for batch starting at index {i}: {e}")
            all_results.extend([None] * len(batch_audio_paths))
            
    return all_results, chunk_indices
