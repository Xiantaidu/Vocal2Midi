import threading
import time
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import soundfile as sf


_QWEN_MODEL_CACHE = {}
_QWEN_MODEL_CACHE_LOCK = threading.Lock()

def load_qwen_model(model_path, device="cuda", use_cache=True):
    """
    Loads the Qwen3-ASR model using PyTorch.
    """
    cache_key = (str(model_path), str(device))
    if use_cache:
        with _QWEN_MODEL_CACHE_LOCK:
            cached_model = _QWEN_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            print(f"Reusing cached Qwen3-ASR model from '{model_path}' on {device}.")
            return cached_model

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
    if use_cache:
        with _QWEN_MODEL_CACHE_LOCK:
            _QWEN_MODEL_CACHE[cache_key] = model
    return model


def clear_qwen_model_cache():
    """Clears in-process Qwen ASR model cache."""
    with _QWEN_MODEL_CACHE_LOCK:
        _QWEN_MODEL_CACHE.clear()


def _run_transcribe_in_process(asr_model, paths, asr_lang):
    if torch.cuda.is_available():
        with torch.amp.autocast("cuda"):
            return asr_model.transcribe(audio=paths, language=asr_lang)
    return asr_model.transcribe(audio=paths, language=asr_lang)


def _run_transcribe_subprocess(paths, model_path, device, asr_lang, timeout_sec=180):
    """Run ASR in a subprocess to avoid in-process deadlocks/hangs."""
    with tempfile.TemporaryDirectory(prefix="v2m_asr_subproc_") as tmp:
        tmp_path = Path(tmp)
        input_json = tmp_path / "input.json"
        output_json = tmp_path / "output.json"

        input_payload = {
            "audio_paths": [str(p) for p in paths],
            "model_path": str(model_path),
            "device": str(device),
            "language": asr_lang,
        }
        input_json.write_text(json.dumps(input_payload, ensure_ascii=False), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "inference.asr_subprocess_worker",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"ASR subprocess timeout after {timeout_sec}s") from e

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-1000:]
            stdout_tail = (proc.stdout or "")[-1000:]
            raise RuntimeError(
                f"ASR subprocess failed (code={proc.returncode}). "
                f"stdout tail: {stdout_tail} | stderr tail: {stderr_tail}"
            )

        if not output_json.exists():
            raise RuntimeError("ASR subprocess finished but output json not found")

        try:
            result_payload = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse ASR subprocess output: {e}")

        if result_payload.get("error"):
            raise RuntimeError(f"ASR subprocess internal error: {result_payload.get('error')}")

        return result_payload.get("results", [])


def batch_transcribe_asr(
    chunks,
    sr,
    asr_model,
    temp_dir_path,
    asr_batch_size,
    language,
    cancel_checker=None,
    asr_model_path=None,
    device="cuda",
    force_subprocess=False,
    asr_timeout_sec=180,
):
    """Saves chunks to temp_dir and runs batched ASR transcription."""
    asr_lang = "Japanese" if language == "ja" else "Chinese"
    print(f"[ASR API] Running ASR with PyTorch Qwen (Batch Size: {asr_batch_size}, Language: {asr_lang})...")

    audio_paths = []
    chunk_indices = []

    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        chunk_path = temp_dir_path / f"{stem}.wav"
                                          
        sf.write(chunk_path, chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)

    all_results = []
    total_batches = (len(audio_paths) - 1) // asr_batch_size + 1 if audio_paths else 0

    def _run_transcribe(paths):
        if force_subprocess:
            if not asr_model_path:
                raise ValueError("asr_model_path is required when force_subprocess=True")
            return _run_transcribe_subprocess(
                paths,
                model_path=asr_model_path,
                device=device,
                asr_lang=asr_lang,
                timeout_sec=asr_timeout_sec,
            )
        if asr_model is None:
            raise ValueError("asr_model is required when force_subprocess=False")
        return _run_transcribe_in_process(asr_model, paths, asr_lang)

    for i in range(0, len(audio_paths), asr_batch_size):
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR 任务已取消")

        batch_audio_paths = audio_paths[i:i+asr_batch_size]
        batch_no = i // asr_batch_size + 1
        print(f"  Processing ASR batch {batch_no}/{total_batches} (size={len(batch_audio_paths)})...")
        
        try:
            batch_start = time.perf_counter()
            with torch.inference_mode():
                batch_results = _run_transcribe(batch_audio_paths)
            cost = time.perf_counter() - batch_start
            print(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
            all_results.extend(batch_results)
        except Exception as e:
            print(f"Error during Qwen ASR transcription for batch starting at index {i}: {e}")
                                  
            if len(batch_audio_paths) > 1:
                print("  Falling back to single-item ASR for this batch...")
                for single_path in batch_audio_paths:
                    if cancel_checker and cancel_checker():
                        raise InterruptedError("ASR 任务已取消")
                    try:
                        with torch.inference_mode():
                            single_res = _run_transcribe([single_path])
                        all_results.append(single_res[0] if single_res else None)
                    except Exception as single_e:
                        print(f"  Single-item ASR failed for '{single_path}': {single_e}")
                        all_results.append(None)
            else:
                all_results.extend([None] * len(batch_audio_paths))
            
    return all_results, chunk_indices
