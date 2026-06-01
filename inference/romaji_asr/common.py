import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly


DEFAULT_SAMPLE_RATE = 16000


def load_vocab(vocab_path: Path) -> tuple[dict[int, str], int]:
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2token = {int(v): k for k, v in vocab.items()}
    blank_id = int(vocab.get("<blank>", vocab.get("PAD", 0)))
    return id2token, blank_id


def load_audio(audio_path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32, copy=False)
    return np.ascontiguousarray(audio, dtype=np.float32)


def create_session(model_path: Path, provider: str = "dml") -> ort.InferenceSession:
    provider = provider.lower()
    available = ort.get_available_providers()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider == "dml":
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(
                "DmlExecutionProvider is not available. Install onnxruntime-directml and retry."
            )
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)


def get_fixed_batch_size(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    if not shape:
        return None
    dim0 = shape[0]
    return int(dim0) if isinstance(dim0, int) else None


def get_fixed_num_samples(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    if len(shape) < 2:
        return None
    dim1 = shape[1]
    return int(dim1) if isinstance(dim1, int) else None


def ort_type_to_numpy_dtype(ort_type: str) -> np.dtype:
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    return np.float32


def prepare_batch(
    session: ort.InferenceSession,
    audio_paths: list[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[dict[str, np.ndarray], list[int]]:
    if not audio_paths:
        raise ValueError("audio_paths must not be empty.")

    fixed_batch_size = get_fixed_batch_size(session)
    if fixed_batch_size is not None and fixed_batch_size != len(audio_paths):
        raise ValueError(f"Model expects batch_size={fixed_batch_size}, got {len(audio_paths)}.")

    waveforms = [load_audio(path, sample_rate=sample_rate) for path in audio_paths]
    lengths = [len(w) for w in waveforms]
    target_num_samples = get_fixed_num_samples(session) or max(lengths)

    batch_size = len(waveforms)
    input_values = np.zeros((batch_size, target_num_samples), dtype=np.float32)
    attention_mask = np.zeros((batch_size, target_num_samples), dtype=np.int64)
    used_lengths = []

    for idx, waveform in enumerate(waveforms):
        num = min(len(waveform), target_num_samples)
        if num > 0:
            input_values[idx, :num] = waveform[:num]
            attention_mask[idx, :num] = 1
        used_lengths.append(num)

    input_meta = {meta.name: meta for meta in session.get_inputs()}
    feeds = {
        "input_values": input_values.astype(
            ort_type_to_numpy_dtype(input_meta["input_values"].type),
            copy=False,
        )
    }
    if "attention_mask" in input_meta:
        feeds["attention_mask"] = attention_mask.astype(
            ort_type_to_numpy_dtype(input_meta["attention_mask"].type),
            copy=False,
        )
    return feeds, used_lengths


def decode_pred_ids(pred_ids: np.ndarray, id2token: dict[int, str], blank_id: int) -> list[str]:
    out: list[str] = []
    prev = -1
    for token_id in pred_ids.tolist():
        token_id = int(token_id)
        if token_id != prev and token_id != blank_id:
            out.append(id2token.get(token_id, "<unk>"))
        prev = token_id
    return out


def decode_logits(logits: np.ndarray, id2token: dict[int, str], blank_id: int) -> list[str]:
    return decode_pred_ids(np.argmax(logits, axis=-1), id2token, blank_id)


def decode_outputs(outputs: np.ndarray, id2token: dict[int, str], blank_id: int) -> list[list[str]]:
    preds: list[list[str]] = []
    for idx in range(outputs.shape[0]):
        item = outputs[idx]
        if np.issubdtype(item.dtype, np.integer):
            preds.append(decode_pred_ids(item, id2token, blank_id))
        else:
            preds.append(decode_logits(item, id2token, blank_id))
    return preds


def chunked(items: list, chunk_size: int):
    step = max(1, int(chunk_size))
    for idx in range(0, len(items), step):
        yield items[idx : idx + step]
