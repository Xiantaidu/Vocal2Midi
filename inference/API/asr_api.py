import multiprocessing as mp
import threading
import time
import json
from pathlib import Path

import torch
import soundfile as sf

# --- Qwen Model Loading ---
_QWEN_MODEL_CACHE = {}
_QWEN_MODEL_CACHE_LOCK = threading.Lock()
_PHONEME_MODEL_CACHE = {}
_PHONEME_MODEL_CACHE_LOCK = threading.Lock()


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


def load_phoneme_asr_model(ckpt_dir, device="cuda", use_cache=True):
    """Load HuBERT-CTC phoneme ASR model + vocab from checkpoint directory."""
    ckpt_dir = str(ckpt_dir)
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.is_file():
        # e.g. .../best/model.safetensors -> use .../best as from_pretrained directory
        ckpt_path = ckpt_path.parent
        ckpt_dir = str(ckpt_path)
    cache_key = (ckpt_dir, str(device))
    if use_cache:
        with _PHONEME_MODEL_CACHE_LOCK:
            cached = _PHONEME_MODEL_CACHE.get(cache_key)
        if cached is not None:
            print(f"Reusing cached phoneme ASR model from '{ckpt_dir}' on {device}.")
            return cached

    from transformers import HubertForCTC

    vocab_path = ckpt_path / "phoneme_vocab.json"
    if not vocab_path.exists():
        fallback_vocab = Path(__file__).resolve().parent.parent / "phonemeASR" / "data" / "phoneme_vocab.json"
        if fallback_vocab.exists():
            vocab_path = fallback_vocab
    if not vocab_path.exists():
        raise FileNotFoundError(f"phoneme_vocab.json not found in checkpoint dir: {ckpt_dir}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2phone = {int(v): k for k, v in vocab.items()}
    blank_id = vocab.get("<blank>", vocab.get("PAD", 0))

    print(f"Loading phoneme ASR model from '{ckpt_dir}' on {device}...")
    model = HubertForCTC.from_pretrained(ckpt_dir).to(device)
    model.eval()

    payload = {
        "model": model,
        "id2phone": id2phone,
        "blank_id": int(blank_id),
    }
    if use_cache:
        with _PHONEME_MODEL_CACHE_LOCK:
            _PHONEME_MODEL_CACHE[cache_key] = payload
    return payload


def clear_phoneme_model_cache():
    with _PHONEME_MODEL_CACHE_LOCK:
        _PHONEME_MODEL_CACHE.clear()


def _greedy_decode_ctc(logits, id2phone, blank=0):
    pred = logits.argmax(-1).tolist()
    out = []
    prev = -1
    for p in pred:
        if p != prev and p != blank:
            out.append(id2phone.get(p, "<unk>"))
        prev = p
    return out


def batch_transcribe_phoneme_asr(
        chunks,
        sr,
        temp_dir_path,
        phoneme_ckpt_dir,
        device="cuda",
        cancel_checker=None,
):
    """Run phoneme ASR directly and return token lists per chunk."""
    print("[ASR API] Running phoneme ASR (HuBERT-CTC) for Japanese romaji mode...")
    asr = load_phoneme_asr_model(phoneme_ckpt_dir, device=device, use_cache=True)
    model = asr["model"]
    id2phone = asr["id2phone"]
    blank_id = asr["blank_id"]

    all_results = []
    chunk_indices = []
    target_sr = 16000

    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR 任务已取消")

        stem = f"chunk_{chunk_idx}"
        chunk_path = temp_dir_path / f"{stem}.wav"
        sf.write(chunk_path, chunk["waveform"], sr)

        waveform, wav_sr = sf.read(str(chunk_path), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)  # mono

        wav_t = torch.from_numpy(waveform).unsqueeze(0)
        if wav_sr != target_sr:
            wav_t = torch.nn.functional.interpolate(
                wav_t.unsqueeze(1),
                size=int(wav_t.shape[-1] * target_sr / wav_sr),
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        wav_t = wav_t.to(device)
        with torch.inference_mode():
            logits = model(wav_t).logits[0]
        phones = _greedy_decode_ctc(logits, id2phone, blank=blank_id)

        all_results.append({"phonemes": phones, "text": " ".join(phones)})
        chunk_indices.append(chunk_idx)

    return all_results, chunk_indices


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
                batch_no = i + 1
                print(f"  Waiting for ASR batch {batch_no}/{total_batches}...")
                batch_start_time = time.perf_counter()
                try:
                    # Wait for result in short polling intervals so cancel can preempt quickly
                    deadline = time.perf_counter() + asr_timeout_sec
                    while not res.ready():
                        if cancel_checker and cancel_checker():
                            print("  ASR cancel requested. Terminating ASR pool...")
                            pool.terminate()
                            pool.join()
                            raise InterruptedError("ASR 任务已取消")
                        if time.perf_counter() >= deadline:
                            raise mp.TimeoutError
                        time.sleep(0.2)

                    batch_result = res.get(timeout=1)
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