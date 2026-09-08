import json
from pathlib import Path

from inference.device_utils import normalize_runtime_device

# The ONNX CTC plumbing (session creation, 16 kHz batch prep, greedy CTC decode)
# is shared with the romaji ASR runtime; only the bundle file layout differs.
from inference.romaji_asr.common import (
    DEFAULT_SAMPLE_RATE,
    chunked,
    create_session,
    decode_outputs,
    get_fixed_batch_size,
    load_vocab,
    prepare_batch,
)

# model_fp16_dynb.onnx is the same graph as model_fp16.onnx with the batch axis
# exported as a symbolic dim, so real multi-chunk batches can run in one pass.
_MODEL_FILE_CANDIDATES = ("model_fp16_dynb.onnx", "model_fp16.onnx", "model.onnx")
_VOCAB_FILE_CANDIDATES = ("pinyin_vocab.json", "phoneme_vocab.json")
_META_FILE_CANDIDATES = ("model_fp16.meta.json", "model.meta.json")


def resolve_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_file():
        if path.name.lower().endswith(".onnx"):
            return path.parent
        raise ValueError(f"Unsupported pinyin ASR model file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Pinyin ASR model path does not exist: {path}")
    return path


def _find_first_existing(model_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


class PinyinASROnnxModel:
    def __init__(
        self,
        model_dir: Path,
        session,
        id2token: dict[int, str],
        blank_id: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        provider: str = "cpu",
    ):
        self.model_dir = model_dir
        self.session = session
        self.id2token = id2token
        self.blank_id = int(blank_id)
        self.sample_rate = int(sample_rate)
        self.provider = provider
        self.output_name = self.session.get_outputs()[0].name
        self.fixed_batch_size = get_fixed_batch_size(self.session)

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        device: str | None = None,
        provider: str | None = None,
        verbose: bool = False,
    ):
        model_dir = resolve_model_dir(model_path)
        model_file = _find_first_existing(model_dir, _MODEL_FILE_CANDIDATES)
        vocab_file = _find_first_existing(model_dir, _VOCAB_FILE_CANDIDATES)
        meta_file = _find_first_existing(model_dir, _META_FILE_CANDIDATES)

        if model_file is None:
            raise FileNotFoundError(f"Pinyin ASR ONNX model not found in: {model_dir}")
        if vocab_file is None:
            raise FileNotFoundError(f"Pinyin ASR vocab not found in: {model_dir}")

        sample_rate = DEFAULT_SAMPLE_RATE
        if meta_file is not None:
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            sample_rate = int(meta.get("sample_rate", sample_rate))

        requested_device = normalize_runtime_device(device)
        requested_provider = (provider or requested_device).lower()
        session = create_session(str(model_file), provider=requested_provider)
        active_provider = "dml" if "DmlExecutionProvider" in session.get_providers() else "cpu"
        if verbose and requested_provider == "dml" and active_provider == "cpu":
            print("[Pinyin ASR] DML provider unavailable, using CPUExecutionProvider.")

        id2token, blank_id = load_vocab(vocab_file)
        return cls(
            model_dir=model_dir,
            session=session,
            id2token=id2token,
            blank_id=blank_id,
            sample_rate=sample_rate,
            provider=active_provider,
        )

    def _prepare_audio_batch(self, audio_paths: list[str]) -> tuple[list[str], int]:
        if not audio_paths:
            return [], 0
        if self.fixed_batch_size is None or self.fixed_batch_size <= len(audio_paths):
            return list(audio_paths), len(audio_paths)
        padded = list(audio_paths)
        while len(padded) < self.fixed_batch_size:
            padded.append(audio_paths[-1])
        return padded, len(audio_paths)

    def transcribe_batch(self, audio_paths: list[str]) -> list[dict]:
        padded_paths, valid_size = self._prepare_audio_batch(audio_paths)
        if not padded_paths:
            return []
        feeds, _ = prepare_batch(self.session, padded_paths, sample_rate=self.sample_rate)
        outputs = self.session.run([self.output_name], feeds)[0]
        preds = decode_outputs(outputs, self.id2token, self.blank_id)
        return [
            {"text": " ".join(tokens), "phonemes": tokens}
            for tokens in preds[:valid_size]
        ]

    def transcribe(self, audio, batch_size: int = 1, language: str | None = None):
        del language
        if isinstance(audio, (str, Path)):
            audio_paths = [str(audio)]
        else:
            audio_paths = [str(path) for path in audio]

        results = []
        for batch_paths in chunked(audio_paths, batch_size):
            results.extend(self.transcribe_batch(batch_paths))
        return results
