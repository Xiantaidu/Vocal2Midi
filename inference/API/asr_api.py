import multiprocessing as mp
import queue
import re
import threading
import time

import soundfile as sf

from inference.device_utils import normalize_runtime_device
from inference.qwen3asr_dml.runtime import Qwen3ASRDmlModel
from inference.romaji_asr.runtime import RomajiASROnnxModel, resolve_model_dir

# --- Qwen Model Loading ---
_QWEN_MODEL_CACHE = {}
_QWEN_MODEL_CACHE_LOCK = threading.Lock()
_ROMAJI_MODEL_CACHE = {}
_ROMAJI_MODEL_CACHE_LOCK = threading.Lock()
_PHONEME_MODEL_CACHE = _ROMAJI_MODEL_CACHE
_PHONEME_MODEL_CACHE_LOCK = _ROMAJI_MODEL_CACHE_LOCK
DEFAULT_QWEN_ASR_PROMPT = (
    "你是一位专业的歌词转录助手，专注于从音频中准确提取歌词文本。"
    "请专注于识别歌曲中的歌词内容。"
    "避免乱猜歌词。"
)
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
_ASCII_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]+")
_CJK_KANA_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f])\s+(?=[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f])")


def _normalize_lyric_language(language: str | None) -> str:
    value = str(language or "").strip().lower()
    if value in {"ja", "japanese"}:
        return "ja"
    if value in {"zh", "cn", "chinese"}:
        return "zh"
    return value


def _filter_qwen_asr_text_for_lyric_flow(text: str, language: str | None) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if _normalize_lyric_language(language) not in {"zh", "ja"}:
        return cleaned

    filtered = _ASCII_WORD_RE.sub(" ", cleaned)
    filtered = _ASCII_PUNCT_RE.sub(" ", filtered)
    filtered = re.sub(r"\s+", " ", filtered).strip()
    filtered = _CJK_KANA_SPACE_RE.sub("", filtered)
    return filtered


def _sanitize_qwen_asr_result(result, language: str | None):
    if result is None:
        return None

    if isinstance(result, dict):
        sanitized = dict(result)
        for key in ("text", "transcript"):
            if key in sanitized and sanitized[key] is not None:
                sanitized[key] = _filter_qwen_asr_text_for_lyric_flow(sanitized[key], language)
        return sanitized

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        try:
            result.text = _filter_qwen_asr_text_for_lyric_flow(text_attr, language)
        except Exception:
            pass
        return result

    if isinstance(result, str):
        return _filter_qwen_asr_text_for_lyric_flow(result, language)
    return result


def _sanitize_qwen_asr_results(results, language: str | None):
    return [_sanitize_qwen_asr_result(result, language) for result in results]


def load_qwen_model(model_path, device="dml", use_cache=True):
    """
    Loads the Qwen ASR model using the unified ONNX + llama.cpp runtime.
    Caches model in-process to avoid repeated loading.
    """
    requested_device = normalize_runtime_device(device)
    cache_key = (str(model_path), requested_device)
    if use_cache:
        with _QWEN_MODEL_CACHE_LOCK:
            cached_model = _QWEN_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            print(f"Reusing cached Qwen ASR model from '{model_path}' on {requested_device}.")
            return cached_model

    try:
        print(f"Loading Qwen ASR runtime from '{model_path}' (requested device: {requested_device})...")
        model = Qwen3ASRDmlModel.from_model_path(
            model_path,
            device=requested_device,
            verbose=False,
        )
        print(
            "Qwen ASR runtime ready: "
            f"encoder={model.encoder_provider_name} {model.encoder_frontend_providers}, "
            f"decoder={model.decoder_backend}."
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading Qwen ASR DML runtime: {e}\n"
            "Please ensure the DML model files are present and required dependencies are installed."
        )

    if use_cache:
        with _QWEN_MODEL_CACHE_LOCK:
            _QWEN_MODEL_CACHE[cache_key] = model
    return model


def clear_qwen_model_cache():
    """Clears in-process Qwen ASR model cache."""
    with _QWEN_MODEL_CACHE_LOCK:
        cached_models = list(_QWEN_MODEL_CACHE.values())
        _QWEN_MODEL_CACHE.clear()
    for model in cached_models:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            shutdown()
    import gc

    gc.collect()


def load_romaji_asr_model(model_dir, device="dml", use_cache=True):
    """Load the Japanese romaji ASR ONNX runtime from a model directory."""
    resolved_dir = str(resolve_model_dir(model_dir))
    requested_device = normalize_runtime_device(device)
    cache_key = (resolved_dir, requested_device)
    cache_enabled = bool(use_cache) and requested_device == "cpu"
    if not cache_enabled:
        with _ROMAJI_MODEL_CACHE_LOCK:
            _ROMAJI_MODEL_CACHE.pop(cache_key, None)
        if use_cache and requested_device != "cpu":
            print("[Romaji ASR] In-process cache is disabled on DML for stability; creating a fresh session.")

    if cache_enabled:
        with _ROMAJI_MODEL_CACHE_LOCK:
            cached = _ROMAJI_MODEL_CACHE.get(cache_key)
        if cached is not None:
            print(f"Reusing cached romaji ASR model from '{resolved_dir}' on {requested_device}.")
            return cached

    print(f"Loading romaji ASR ONNX model from '{resolved_dir}' on {requested_device}...")
    model = RomajiASROnnxModel.from_model_path(resolved_dir, device=requested_device, verbose=True)

    payload = {
        "model": model,
        "sample_rate": int(model.sample_rate),
        "provider": model.provider,
    }
    if cache_enabled:
        with _ROMAJI_MODEL_CACHE_LOCK:
            _ROMAJI_MODEL_CACHE[cache_key] = payload
    return payload


def clear_romaji_model_cache():
    with _ROMAJI_MODEL_CACHE_LOCK:
        _ROMAJI_MODEL_CACHE.clear()
    import gc

    gc.collect()


def batch_transcribe_romaji_asr(
    chunks,
    sr,
    temp_dir_path,
    model_dir,
    device="dml",
    asr_batch_size=1,
    cancel_checker=None,
):
    """Run romaji ASR directly and return token lists per chunk."""
    print("[ASR API] Running romaji ASR (ONNX Runtime) for Japanese lyric mode...")
    asr = load_romaji_asr_model(model_dir, device=device, use_cache=True)
    model = asr["model"]

    audio_paths = []
    chunk_indices = []
    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR task cancelled")
        chunk_path = temp_dir_path / f"chunk_{chunk_idx}.wav"
        sf.write(chunk_path, chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)

    if not audio_paths:
        return [], []

    all_results = model.transcribe(audio_paths, batch_size=max(1, int(asr_batch_size)))
    return all_results, chunk_indices


# Compatibility aliases so higher layers can keep the old kwargs/field names
# while the implementation has switched from Torch phoneme ASR to romaji ASR.
def load_phoneme_asr_model(model_dir, device="dml", use_cache=True):
    return load_romaji_asr_model(model_dir, device=device, use_cache=use_cache)


def clear_phoneme_model_cache():
    clear_romaji_model_cache()


def batch_transcribe_phoneme_asr(
    chunks,
    sr,
    temp_dir_path,
    phoneme_ckpt_dir,
    device="dml",
    asr_batch_size=1,
    cancel_checker=None,
):
    results, chunk_indices = batch_transcribe_romaji_asr(
        chunks,
        sr,
        temp_dir_path=temp_dir_path,
        model_dir=phoneme_ckpt_dir,
        device=device,
        asr_batch_size=asr_batch_size,
        cancel_checker=cancel_checker,
    )
    return results, chunk_indices, {}


# --- Process Pool Worker ---
_WORKER_ASR_MODEL = None


def _init_worker(model_path, device):
    """Initializer for each worker process in the pool."""
    global _WORKER_ASR_MODEL
    proc_name = mp.current_process().name
    print(f"Initializing ASR worker ({proc_name}) with model '{model_path}' on {device}...")
    _WORKER_ASR_MODEL = load_qwen_model(model_path, device, use_cache=False)
    print(f"ASR worker ({proc_name}) initialized.")


def _transcribe_task(paths, asr_lang, context, model=None):
    """The actual transcription task executed by a worker process or in-process."""
    m = model if model is not None else _WORKER_ASR_MODEL
    if m is None:
        return RuntimeError("ASR worker model not initialized.")

    try:
        return m.transcribe(audio=paths, language=asr_lang, context=context)
    except Exception as e:
        return e


def _asr_worker_main(model_path, device, task_queue, result_queue):
    """Runs a single non-daemon ASR worker process for Qwen DML+CPU inference."""
    model = None
    try:
        proc_name = mp.current_process().name
        print(f"Initializing ASR worker ({proc_name}) with model '{model_path}' on {device}...")
        model = load_qwen_model(model_path, device, use_cache=False)
        print(f"ASR worker ({proc_name}) initialized.")
        result_queue.put({"type": "ready"})

        while True:
            message = task_queue.get()
            if message.get("type") == "stop":
                break
            if message.get("type") != "transcribe":
                continue

            task_id = int(message["task_id"])
            batch = list(message["paths"])
            asr_lang = str(message["asr_lang"])
            asr_prompt = str(message.get("asr_prompt") or DEFAULT_QWEN_ASR_PROMPT)
            batch_result = _transcribe_task(batch, asr_lang, asr_prompt, model=model)
            if isinstance(batch_result, Exception):
                result_queue.put(
                    {
                        "type": "result",
                        "task_id": task_id,
                        "error": str(batch_result),
                    }
                )
            else:
                result_queue.put(
                    {
                        "type": "result",
                        "task_id": task_id,
                        "result": batch_result,
                    }
                )
    except Exception as e:
        result_queue.put({"type": "startup_error", "error": str(e)})
    finally:
        if model is not None:
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                shutdown()


def _shutdown_asr_worker(worker, task_queue, *, terminate=False):
    if worker is None:
        return

    if terminate:
        if worker.is_alive():
            worker.terminate()
        worker.join()
        return

    if worker.is_alive():
        try:
            task_queue.put({"type": "stop"})
        except Exception:
            worker.terminate()
            worker.join()
            return
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join()


def _wait_for_worker_message(result_queue, worker, *, timeout_sec, cancel_checker=None, on_cancel=None):
    deadline = None if timeout_sec is None else time.perf_counter() + max(float(timeout_sec), 0.0)
    while True:
        if cancel_checker and cancel_checker():
            if on_cancel is not None:
                on_cancel()
            raise InterruptedError("ASR task cancelled")

        if deadline is not None and time.perf_counter() >= deadline:
            raise mp.TimeoutError

        poll_timeout = 0.2
        if deadline is not None:
            poll_timeout = max(0.01, min(0.2, deadline - time.perf_counter()))
        try:
            return result_queue.get(timeout=poll_timeout)
        except queue.Empty:
            if worker is not None and not worker.is_alive():
                raise RuntimeError("ASR worker exited unexpectedly.")
            continue


# --- Main ASR API ---
def batch_transcribe_asr(
    chunks,
    sr,
    asr_model,
    temp_dir_path,
    asr_batch_size,
    language,
    cancel_checker=None,
    asr_model_path=None,
    device="dml",
    force_subprocess=False,
    asr_timeout_sec=180,
    asr_prompt: str = DEFAULT_QWEN_ASR_PROMPT,
):
    """Saves chunks to temp_dir and runs batched ASR transcription."""
    asr_lang = "Japanese" if language == "ja" else "Chinese"
    print(f"[ASR API] Running ASR with Qwen DML+CPU runtime (Batch Size: {asr_batch_size}, Language: {asr_lang})...")

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

    batches = [audio_paths[i:i + asr_batch_size] for i in range(0, len(audio_paths), asr_batch_size)]
    total_batches = len(batches)
    all_results = []

    if force_subprocess:
        if not asr_model_path:
            raise ValueError("asr_model_path is required when force_subprocess=True")

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        worker = ctx.Process(
            target=_asr_worker_main,
            args=(asr_model_path, device, task_queue, result_queue),
            daemon=False,
        )
        worker.start()
        try:
            startup_message = _wait_for_worker_message(
                result_queue,
                worker,
                timeout_sec=asr_timeout_sec,
                cancel_checker=cancel_checker,
                on_cancel=lambda: _shutdown_asr_worker(worker, task_queue, terminate=True),
            )
            if startup_message.get("type") == "startup_error":
                raise RuntimeError(f"ASR worker failed to start: {startup_message.get('error', 'unknown error')}")
            if startup_message.get("type") != "ready":
                raise RuntimeError(f"Unexpected ASR worker startup message: {startup_message!r}")

            print(f"ASR subprocess worker created for {total_batches} batch(es).")
            for i, batch in enumerate(batches):
                batch_no = i + 1
                task_queue.put(
                    {
                        "type": "transcribe",
                        "task_id": i,
                        "paths": batch,
                        "asr_lang": asr_lang,
                        "asr_prompt": asr_prompt,
                    }
                )
                print(f"  Waiting for ASR batch {batch_no}/{total_batches}...")
                batch_start_time = time.perf_counter()
                try:
                    message = _wait_for_worker_message(
                        result_queue,
                        worker,
                        timeout_sec=asr_timeout_sec,
                        cancel_checker=cancel_checker,
                        on_cancel=lambda: _shutdown_asr_worker(worker, task_queue, terminate=True),
                    )
                    cost = time.perf_counter() - batch_start_time
                    if message.get("type") != "result" or int(message.get("task_id", -1)) != i:
                        raise RuntimeError(f"Unexpected ASR worker message: {message!r}")

                    if message.get("error"):
                        print(f"  ASR batch {batch_no}/{total_batches} failed with an error: {message['error']}")
                        all_results.extend([None] * len(batch))
                    else:
                        print(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
                        all_results.extend(message.get("result", []))
                except mp.TimeoutError:
                    print(f"  ASR batch {batch_no}/{total_batches} timed out after {asr_timeout_sec}s.")
                    _shutdown_asr_worker(worker, task_queue, terminate=True)
                    raise TimeoutError(f"ASR batch {batch_no}/{total_batches} timed out after {asr_timeout_sec}s")
        finally:
            _shutdown_asr_worker(worker, task_queue, terminate=False)

    else:
        if asr_model is None:
            raise ValueError("asr_model is required when force_subprocess=False")

        for i, batch in enumerate(batches):
            if cancel_checker and cancel_checker():
                raise InterruptedError("ASR task cancelled")

            batch_no = i + 1
            print(f"  Processing ASR batch {batch_no}/{total_batches} (size={len(batch)})...")
            try:
                batch_start = time.perf_counter()
                results = _transcribe_task(batch, asr_lang, asr_prompt, model=asr_model)
                cost = time.perf_counter() - batch_start
                print(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
                all_results.extend(results)
            except Exception as e:
                print(f"Error during in-process ASR for batch {batch_no}: {e}")
                all_results.extend([None] * len(batch))

    return _sanitize_qwen_asr_results(all_results, language), chunk_indices
