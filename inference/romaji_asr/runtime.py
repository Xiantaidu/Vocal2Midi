import json
from pathlib import Path

from inference.device_utils import normalize_runtime_device

from .common import (
    DEFAULT_SAMPLE_RATE,
    chunked,
    create_session,
    decode_outputs,
    get_fixed_batch_size,
    load_vocab,
    prepare_batch,
)


def resolve_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_file():
        if path.name.lower().endswith(".onnx"):
            return path.parent
        raise ValueError(f"Unsupported romaji ASR model file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Romaji ASR model path does not exist: {path}")
    return path


class RomajiASROnnxModel:
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
        device: str = "dml",
        provider: str | None = None,
        verbose: bool = False,
    ):
        model_dir = resolve_model_dir(model_path)
        model_file = model_dir / "model.onnx"
        vocab_file = model_dir / "phoneme_vocab.json"
        meta_file = model_dir / "model.meta.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Romaji ASR ONNX model not found: {model_file}")
        if not vocab_file.exists():
            raise FileNotFoundError(f"Romaji ASR vocab not found: {vocab_file}")

        sample_rate = DEFAULT_SAMPLE_RATE
        if meta_file.exists():
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            sample_rate = int(meta.get("sample_rate", sample_rate))

        requested_device = normalize_runtime_device(device)
        requested_provider = (provider or requested_device).lower()
        try:
            session = create_session(model_file, provider=requested_provider)
            active_provider = requested_provider
        except RuntimeError:
            if requested_provider != "dml":
                raise
            if verbose:
                print("[Romaji ASR] DML provider unavailable, falling back to CPUExecutionProvider.")
            session = create_session(model_file, provider="cpu")
            active_provider = "cpu"

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
