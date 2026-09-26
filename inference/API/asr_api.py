import logging
import multiprocessing as mp
import queue
import re
import sys
import threading
import time

import soundfile as sf

from inference.device_utils import normalize_runtime_device
from inference.pinyin_asr.runtime import PinyinASROnnxModel, resolve_model_dir as resolve_pinyin_model_dir
from inference.qwen3asr_dml.runtime import Qwen3ASRDmlModel
from inference.romaji_asr.runtime import RomajiASROnnxModel, resolve_model_dir

logger = logging.getLogger(__name__)

# --- Qwen Model Loading ---
_QWEN_MODEL_CACHE = {}
_QWEN_MODEL_CACHE_LOCK = threading.Lock()
_ROMAJI_MODEL_CACHE = {}
_ROMAJI_MODEL_CACHE_LOCK = threading.Lock()
_PINYIN_MODEL_CACHE = {}
_PINYIN_MODEL_CACHE_LOCK = threading.Lock()
DEFAULT_QWEN_ASR_PROMPT = (
    "你是一位专业的歌词转录助手，专注于从音频中准确提取歌词文本。"
    "请专注于识别歌曲中的歌词内容。"
    "避免乱猜歌词。"
)
DEFAULT_QWEN_ASR_PROMPT_EN = (
    "You are a professional lyrics transcription assistant focused on accurately "
    "extracting lyric text from audio. Transcribe the sung English lyrics exactly "
    "as heard. Avoid guessing or inventing lyrics."
)
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
_ASCII_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]+")
_CJK_KANA_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f])\s+(?=[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f])")
# English words for the HFA CMU dictionary: letters joined by apostrophes/hyphens
# (e.g. don't, world-class). Curly apostrophes are normalized to straight ones.
_ENGLISH_DICT_WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")


def _normalize_lyric_language(language: str | None) -> str:
    value = str(language or "").strip().lower()
    if value in {"ja", "japanese"}:
        return "ja"
    if value in {"zh", "cn", "chinese"}:
        return "zh"
    if value in {"en", "english"}:
        return "en"
    return value


# Language name passed to the Qwen ASR prompt for each pipeline language.
QWEN_ASR_LANGUAGE_NAMES = {
    "ja": "Japanese",
    "zh": "Chinese",
    "en": "English",
}


def _filter_qwen_asr_text_for_lyric_flow(text: str, language: str | None) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    normalized_language = _normalize_lyric_language(language)
    if normalized_language == "en":
        # Keep only dictionary-friendly English words: drops CJK bleed-over,
        # digits, and punctuation that would break CMU dict lookups.
        return " ".join(_ENGLISH_DICT_WORD_RE.findall(cleaned.replace("’", "'")))
    if normalized_language not in {"zh", "ja"}:
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


def load_qwen_model(model_path, device=None, use_cache=True):
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
            logger.info(f"Reusing cached Qwen ASR model from '{model_path}' on {requested_device}.")
            return cached_model

    try:
        logger.info(f"Loading Qwen ASR runtime from '{model_path}' (requested device: {requested_device})...")
        model = Qwen3ASRDmlModel.from_model_path(
            model_path,
            device=requested_device,
            verbose=False,
        )
        logger.info(
            "Qwen ASR runtime ready: "
            f"encoder={model.encoder_provider_name} {model.encoder_frontend_providers}, "
            f"decoder={model.decoder_backend}."
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading Qwen ASR runtime: {e}\n"
            "Please ensure the Qwen model files are present and required dependencies are installed."
        )

    if use_cache:
        # Re-check under the lock: a concurrent loader for the same key may
        # have finished first; release our duplicate instead of leaking it.
        with _QWEN_MODEL_CACHE_LOCK:
            existing = _QWEN_MODEL_CACHE.get(cache_key)
            if existing is None:
                _QWEN_MODEL_CACHE[cache_key] = model
                return model
        duplicate_shutdown = getattr(model, "shutdown", None)
        if callable(duplicate_shutdown):
            duplicate_shutdown()
        return existing
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


def load_romaji_asr_model(model_dir, device=None, use_cache=True):
    """Load the Japanese romaji ASR ONNX runtime from a model directory."""
    resolved_dir = str(resolve_model_dir(model_dir))
    requested_device = normalize_runtime_device(device)
    cache_key = (resolved_dir, requested_device)
    cache_enabled = bool(use_cache) and requested_device == "cpu"
    if not cache_enabled:
        with _ROMAJI_MODEL_CACHE_LOCK:
            _ROMAJI_MODEL_CACHE.pop(cache_key, None)
        if use_cache and requested_device != "cpu":
            logger.info("[Romaji ASR] In-process cache is disabled on DML for stability; creating a fresh session.")

    if cache_enabled:
        with _ROMAJI_MODEL_CACHE_LOCK:
            cached = _ROMAJI_MODEL_CACHE.get(cache_key)
        if cached is not None:
            logger.info(f"Reusing cached romaji ASR model from '{resolved_dir}' on {requested_device}.")
            return cached

    logger.info(f"Loading romaji ASR ONNX model from '{resolved_dir}' on {requested_device}...")
    model = RomajiASROnnxModel.from_model_path(resolved_dir, device=requested_device, verbose=True)

    payload = {
        "model": model,
        "sample_rate": int(model.sample_rate),
        "provider": model.provider,
    }
    if cache_enabled:
        with _ROMAJI_MODEL_CACHE_LOCK:
            existing = _ROMAJI_MODEL_CACHE.get(cache_key)
            if existing is None:
                _ROMAJI_MODEL_CACHE[cache_key] = payload
                return payload
        return existing
    return payload


def clear_romaji_model_cache():
    with _ROMAJI_MODEL_CACHE_LOCK:
        _ROMAJI_MODEL_CACHE.clear()
    import gc

    gc.collect()


def load_pinyin_asr_model(model_dir, device=None, use_cache=True):
    """Load the Chinese pinyin ASR ONNX runtime from a model directory."""
    resolved_dir = str(resolve_pinyin_model_dir(model_dir))
    requested_device = normalize_runtime_device(device)
    cache_key = (resolved_dir, requested_device)
    cache_enabled = bool(use_cache) and requested_device == "cpu"
    if not cache_enabled:
        with _PINYIN_MODEL_CACHE_LOCK:
            _PINYIN_MODEL_CACHE.pop(cache_key, None)
        if use_cache and requested_device != "cpu":
            logger.info("[Pinyin ASR] In-process cache is disabled on DML for stability; creating a fresh session.")

    if cache_enabled:
        with _PINYIN_MODEL_CACHE_LOCK:
            cached = _PINYIN_MODEL_CACHE.get(cache_key)
        if cached is not None:
            logger.info(f"Reusing cached pinyin ASR model from '{resolved_dir}' on {requested_device}.")
            return cached

    logger.info(f"Loading pinyin ASR ONNX model from '{resolved_dir}' on {requested_device}...")
    model = PinyinASROnnxModel.from_model_path(resolved_dir, device=requested_device, verbose=True)

    payload = {
        "model": model,
        "sample_rate": int(model.sample_rate),
        "provider": model.provider,
    }
    if cache_enabled:
        with _PINYIN_MODEL_CACHE_LOCK:
            existing = _PINYIN_MODEL_CACHE.get(cache_key)
            if existing is None:
                _PINYIN_MODEL_CACHE[cache_key] = payload
                return payload
        return existing
    return payload


def clear_pinyin_model_cache():
    with _PINYIN_MODEL_CACHE_LOCK:
        _PINYIN_MODEL_CACHE.clear()
    import gc

    gc.collect()


def batch_transcribe_pinyin_asr(
    chunks,
    sr,
    temp_dir_path,
    model_dir,
    device=None,
    asr_batch_size=1,
    cancel_checker=None,
):
    """Run pinyin ASR directly and return token lists per chunk."""
    logger.info("[ASR API] Running pinyin ASR (ONNX Runtime) for Chinese-pinyin lyric mode...")
    asr = load_pinyin_asr_model(model_dir, device=device, use_cache=True)
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


def batch_transcribe_romaji_asr(
    chunks,
    sr,
    temp_dir_path,
    model_dir,
    device=None,
    asr_batch_size=1,
    cancel_checker=None,
):
    """Run romaji ASR directly and return token lists per chunk."""
    logger.info("[ASR API] Running romaji ASR (ONNX Runtime) for Japanese lyric mode...")
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


# --- Process Pool Worker ---
_WORKER_ASR_MODEL = None


def _init_worker(model_path, device):
    """Initializer for each worker process in the pool."""
    global _WORKER_ASR_MODEL
    proc_name = mp.current_process().name
    logger.info(f"Initializing ASR worker ({proc_name}) with model '{model_path}' on {device}...")
    _WORKER_ASR_MODEL = load_qwen_model(model_path, device, use_cache=False)
    logger.info(f"ASR worker ({proc_name}) initialized.")


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
    """Runs a single non-daemon ASR worker process for Qwen inference."""
    # Child process: route module logging (including the qwen runtime) to the
    # inherited console stream, message-only like the old prints.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    model = None
    try:
        proc_name = mp.current_process().name
        logger.info(f"Initializing ASR worker ({proc_name}) with model '{model_path}' on {device}...")
        model = load_qwen_model(model_path, device, use_cache=False)
        logger.info(f"ASR worker ({proc_name}) initialized.")
        result_queue.put({"type": "ready"})

        while True:
            try:
                message = task_queue.get(timeout=5)
            except queue.Empty:
                # A non-daemon worker must not outlive the parent holding a
                # multi-GB model: exit when the parent disappears.
                parent = mp.parent_process()
                if parent is not None and not parent.is_alive():
                    break
                continue
            if message.get("type") == "stop":
                break
            if message.get("type") != "transcribe":
                continue

            try:
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
                # Per-task failures degrade this batch only; letting them
                # escape would abort every remaining batch (the parent treats
                # a dead worker as a startup failure).
                try:
                    fallback_id = int(message.get("task_id", -1))
                except Exception:
                    fallback_id = -1
                try:
                    result_queue.put({"type": "result", "task_id": fallback_id, "error": repr(e)})
                except Exception:
                    pass
    except Exception as e:
        # Startup-phase failures (model load) and truly unexpected errors:
        # report them so the parent can fail fast with a clear message.
        try:
            result_queue.put({"type": "startup_error", "error": str(e)})
        except Exception:
            pass
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


class AsrSubprocessSession:
    """A reusable Qwen ASR subprocess worker spanning multiple pipeline runs.

    The worker process (and the multi-GB model it loads) is spawned lazily on
    the first transcribe() call and reused across calls, so a batch of files
    pays the model load once instead of per file. If the bound model path or
    device changes, or the worker died, the next call respawns it. close()
    releases the worker; jobs that never reach the Qwen text-ASR stage never
    spawn one.
    """

    def __init__(self, *, startup_timeout_sec: float = 180, batch_timeout_sec: float = 180):
        self._startup_timeout_sec = startup_timeout_sec
        self._batch_timeout_sec = batch_timeout_sec
        self._worker = None
        self._task_queue = None
        self._result_queue = None
        self._model_path = None
        self._device = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _spawn_worker(self, model_path, device, cancel_checker=None):
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        worker = ctx.Process(
            target=_asr_worker_main,
            args=(model_path, device, task_queue, result_queue),
            daemon=False,
        )
        worker.start()
        self._worker = worker
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._model_path = model_path
        self._device = device
        try:
            startup_message = _wait_for_worker_message(
                result_queue,
                worker,
                timeout_sec=self._startup_timeout_sec,
                cancel_checker=cancel_checker,
                on_cancel=self.invalidate,
            )
        except mp.TimeoutError:
            # mp.TimeoutError is not a builtin TimeoutError subclass; give
            # multi-GB model loads a clear, catchable failure message.
            self.invalidate()
            raise TimeoutError(
                f"ASR worker failed to start within {self._startup_timeout_sec}s"
            ) from None
        if startup_message.get("type") == "startup_error":
            self.invalidate()
            raise RuntimeError(f"ASR worker failed to start: {startup_message.get('error', 'unknown error')}")
        if startup_message.get("type") != "ready":
            self.invalidate()
            raise RuntimeError(f"Unexpected ASR worker startup message: {startup_message!r}")

    def _ensure_worker(self, model_path, device, cancel_checker=None):
        if (
            self._worker is not None
            and self._worker.is_alive()
            and self._model_path == model_path
            and self._device == device
        ):
            return
        self.invalidate()
        logger.info(f"Starting ASR subprocess worker with model '{model_path}' on {device}...")
        self._spawn_worker(model_path, device, cancel_checker)

    def transcribe(
        self,
        audio_paths,
        *,
        model_path,
        device,
        asr_lang,
        asr_prompt,
        asr_batch_size,
        cancel_checker=None,
    ):
        """Transcribe the given wav paths in batches; one result entry per path."""
        if not audio_paths:
            return []
        batches = [audio_paths[i:i + asr_batch_size] for i in range(0, len(audio_paths), asr_batch_size)]
        total_batches = len(batches)
        all_results = []
        self._ensure_worker(model_path, device, cancel_checker)
        logger.info(f"ASR subprocess worker ready for {total_batches} batch(es).")
        for i, batch in enumerate(batches):
            batch_no = i + 1
            self._task_queue.put(
                {
                    "type": "transcribe",
                    "task_id": i,
                    "paths": batch,
                    "asr_lang": asr_lang,
                    "asr_prompt": asr_prompt,
                }
            )
            logger.info(f"  Waiting for ASR batch {batch_no}/{total_batches}...")
            batch_start_time = time.perf_counter()
            try:
                message = _wait_for_worker_message(
                    self._result_queue,
                    self._worker,
                    timeout_sec=self._batch_timeout_sec,
                    cancel_checker=cancel_checker,
                    on_cancel=self.invalidate,
                )
                cost = time.perf_counter() - batch_start_time
                if message.get("type") != "result" or int(message.get("task_id", -1)) != i:
                    raise RuntimeError(f"Unexpected ASR worker message: {message!r}")

                error = message.get("error")
                results = message.get("result", [])
                if error is None and len(results) == len(batch):
                    logger.info(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
                    all_results.extend(results)
                else:
                    # An empty error string or a short result list both
                    # count as failure; padding with None keeps
                    # all_results aligned with the chunk indices.
                    reason = error if error else f"incomplete result ({len(results)}/{len(batch)})"
                    logger.error(f"  ASR batch {batch_no}/{total_batches} failed with an error: {reason}")
                    all_results.extend([None] * len(batch))
            except mp.TimeoutError:
                logger.error(f"  ASR batch {batch_no}/{total_batches} timed out after {self._batch_timeout_sec}s.")
                self.invalidate()
                raise TimeoutError(
                    f"ASR batch {batch_no}/{total_batches} timed out after {self._batch_timeout_sec}s"
                ) from None
        return all_results

    def close(self):
        """Gracefully stop the worker (stop message first, terminate as fallback)."""
        if self._worker is not None:
            _shutdown_asr_worker(self._worker, self._task_queue, terminate=False)
            self._detach_worker()

    def invalidate(self):
        """Hard-terminate the worker (cancel, timeout, or unusable state)."""
        if self._worker is not None:
            _shutdown_asr_worker(self._worker, self._task_queue, terminate=True)
            self._detach_worker()

    def _detach_worker(self):
        self._worker = None
        self._task_queue = None
        self._result_queue = None
        self._model_path = None
        self._device = None


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
    device=None,
    force_subprocess=False,
    asr_timeout_sec=180,
    asr_prompt: str | None = None,
    session: "AsrSubprocessSession | None" = None,
):
    """Saves chunks to temp_dir and runs batched ASR transcription.

    With force_subprocess=True and a shared `session`, the spawned worker
    process (and its loaded model) is reused across calls; without a session
    a one-shot worker is spawned for this call only.
    """
    asr_batch_size = max(1, int(asr_batch_size))
    asr_lang = QWEN_ASR_LANGUAGE_NAMES.get(
        _normalize_lyric_language(language), QWEN_ASR_LANGUAGE_NAMES["zh"]
    )
    if not asr_prompt:
        asr_prompt = (
            DEFAULT_QWEN_ASR_PROMPT_EN
            if asr_lang == "English"
            else DEFAULT_QWEN_ASR_PROMPT
        )
    logger.info(f"[ASR API] Running ASR with Qwen runtime (Batch Size: {asr_batch_size}, Language: {asr_lang})...")

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

        own_session = session is None
        worker_session = session if session is not None else AsrSubprocessSession(
            startup_timeout_sec=asr_timeout_sec,
            batch_timeout_sec=asr_timeout_sec,
        )
        try:
            all_results = worker_session.transcribe(
                audio_paths,
                model_path=asr_model_path,
                device=device,
                asr_lang=asr_lang,
                asr_prompt=asr_prompt,
                asr_batch_size=asr_batch_size,
                cancel_checker=cancel_checker,
            )
        finally:
            if own_session:
                worker_session.close()

    else:
        if asr_model is None:
            raise ValueError("asr_model is required when force_subprocess=False")

        for i, batch in enumerate(batches):
            if cancel_checker and cancel_checker():
                raise InterruptedError("ASR task cancelled")

            batch_no = i + 1
            logger.info(f"  Processing ASR batch {batch_no}/{total_batches} (size={len(batch)})...")
            try:
                batch_start = time.perf_counter()
                results = _transcribe_task(batch, asr_lang, asr_prompt, model=asr_model)
                cost = time.perf_counter() - batch_start
                logger.info(f"  ASR batch {batch_no}/{total_batches} done in {cost:.2f}s")
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error during in-process ASR for batch {batch_no}: {e}")
                all_results.extend([None] * len(batch))

    return _sanitize_qwen_asr_results(all_results, language), chunk_indices
