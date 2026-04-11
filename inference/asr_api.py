import multiprocessing as mp
import threading
import time
from pathlib import Path

import torch
import soundfile as sf

# --- Qwen Model Loading ---
_QWEN_MODEL_CACHE = {}
_QWEN_MODEL_CACHE_LOCK = threading.Lock()


def load_qwen_model(model_path, device="cuda", use_cache=True):
    """
    Loads the Qwen ASR model using PyTorch.
    Caches model in-process to avoid repeated loading.
    """
    cache_key = (str(model_path), str(device))
    if use_cache:
        with _QWEN_MODEL_CACHE_LOCK:
            cached_model = _QWEN_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            print(f"Reusing cached Qwen ASR model from '{model_path}' on {device}.")
            return cached_model

    print(f"Loading Qwen ASR PyTorch model from '{model_path}' on {device}...")
    from qwen_asr import Qwen3ASRModel

    try:
        # Use fp16 for lower VRAM usage and better performance
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map=device,
        )
    except Exception as e:
        raise RuntimeError(f"Error loading Qwen ASR model: {e}\nPlease ensure you have run 'pip install -U qwen-asr'.")

    if use_cache:
        with _QWEN_MODEL_CACHE_LOCK:
            _QWEN_MODEL_CACHE[cache_key] = model
    return model


def clear_qwen_model_cache():
    """Clears in-process Qwen ASR model cache."""
    with _QWEN_MODEL_CACHE_LOCK:
        _QWEN_MODEL_CACHE.clear()


# --- Process Pool Worker ---
_WORKER_ASR_MODEL = None


def _init_worker(model_path, device):
    """Initializer for each worker process in the pool."""
    global _WORKER_ASR_MODEL
    proc_name = mp.current_process().name
    print(f"Initializing ASR worker ({proc_name}) with model '{model_path}' on {device}...")
    # Each worker loads its own copy of the model. `use_cache=False` is fine here
    # as this function is only called once per worker.
    _WORKER_ASR_MODEL = load_qwen_model(model_path, device, use_cache=False)
    print(f"ASR worker ({proc_name}) initialized.")


def _transcribe_task(paths, asr_lang):
    """The actual transcription task executed by a worker process."""
    if _WORKER_ASR_MODEL is None:
        return RuntimeError("ASR worker model not initialized.")

    try:
        # Use autocast for performance on CUDA devices
        use_cuda = torch.cuda.is_available() and "cuda" in str(_WORKER_ASR_MODEL.device)
        with torch.inference_mode():
            if use_cuda:
                with torch.amp.autocast("cuda"):
                    results = _WORKER_ASR_MODEL.transcribe(audio=paths, language=asr_lang)
            else:
                results = _WORKER_ASR_MODEL.transcribe(audio=paths, language=asr_lang)
        return results
    except Exception as e:
        # Propagate exceptions back to the main process
        return e


# --- Main ASR API ---
def batch_transcribe_asr(
        chunks,
        sr,
        asr_model,  # This is for in-process, will be None for pooled
        temp_dir_path,
        asr_batch_size,
        language,
        cancel_checker=None,
        asr_model_path=None,
        device="cuda",
        force_subprocess=False,  # If True, uses the new process pool
        asr_timeout_sec=180,
):
    """Saves chunks to temp_dir and runs batched ASR transcription."""
    asr_lang = "Japanese" if language == "ja" else "Chinese"
    print(f"[ASR API] Running ASR with PyTorch Qwen (Batch Size: {asr_batch_size}, Language: {asr_lang})...")

    # 1. Prepare audio files
    audio_paths = []
    chunk_indices = []
    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        chunk_path = temp_dir_path / f"{stem}.wav"
        sf.write(chunk_path, chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)

    if not audio_paths:
        return [], []

    # 2. Group paths into batches for processing
    batches = [audio_paths[i:i + asr_batch_size] for i in range(0, len(audio_paths), asr_batch_size)]
    total_batches = len(batches)
    all_results = []

    # 3. Choose execution strategy: Process Pool vs. In-Process
    if force_subprocess:
        if not asr_model_path:
            raise ValueError("asr_model_path is required when force_subprocess=True")

        # Use a managed process pool
        # Using 'spawn' is safer for CUDA applications
        ctx = mp.get_context("spawn")
        # Limit to 1 worker to ensure only one model is loaded into VRAM
        with ctx.Pool(processes=1, initializer=_init_worker, initargs=(asr_model_path, device)) as pool:
            print(f"ASR Process Pool created with 1 worker for {total_batches} batches.")
            # Map tasks to the pool
            async_results = [pool.apply_async(_transcribe_task, args=(batch, asr_lang)) for batch in batches]

            # Collect results with progress
            for i, res in enumerate(async_results):
                if cancel_checker and cancel_checker():
                    raise InterruptedError("ASR 任务已取消")

                batch_no = i + 1
                print(f"  Waiting for ASR batch {batch_no}/{total_batches}...")
                batch_start_time = time.perf_counter()
                try:
                    # Wait for the result with a timeout
                    batch_result = res.get(timeout=asr_timeout_sec)
                    cost = time.perf_counter() - batch_start_time

                    if isinstance(batch_result, Exception):
                        print(f"  ASR batch {batch_no}/{total_batches} failed with an error: {batch_result}")
                        all_results.extend([None] * len(batches[i]))
                    else:
                        print(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
                        all_results.extend(batch_result)
                except mp.TimeoutError:
                    print(f"  ASR batch {batch_no}/{total_batches} timed out after {asr_timeout_sec}s.")
                    all_results.extend([None] * len(batches[i]))

    else:
        # Fallback to the original in-process method
        if asr_model is None:
            raise ValueError("asr_model is required when force_subprocess=False")

        for i, batch in enumerate(batches):
            if cancel_checker and cancel_checker():
                raise InterruptedError("ASR 任务已取消")

            batch_no = i + 1
            print(f"  Processing ASR batch {batch_no}/{total_batches} (size={len(batch)})...")
            try:
                batch_start = time.perf_counter()
                results = _transcribe_task(batch, asr_lang)
                cost = time.perf_counter() - batch_start
                print(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
                all_results.extend(results)
            except Exception as e:
                print(f"Error during in-process ASR for batch {batch_no}: {e}")
                all_results.extend([None] * len(batch))

    return all_results, chunk_indices
