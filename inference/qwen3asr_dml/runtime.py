from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from inference.device_utils import normalize_runtime_device, use_dml

from .asr import QwenASREngine
from .schema import ASREngineConfig


MODEL_FILENAMES = (
    "qwen3_asr_encoder_frontend.int4.onnx",
    "qwen3_asr_encoder_backend.int4.onnx",
    "qwen3_asr_llm.q4_k.gguf",
)

LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "en": "English",
    "english": "English",
}


def _has_model_files(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in MODEL_FILENAMES)


def resolve_model_dir(model_path: str | Path) -> Path:
    candidate = Path(model_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent

    if _has_model_files(candidate):
        return candidate

    nested = candidate / "dml_cpu"
    if _has_model_files(nested):
        return nested

    sibling_dml = candidate.with_name(candidate.name + "-dml")
    if _has_model_files(sibling_dml):
        return sibling_dml

    raise FileNotFoundError(
        "Qwen3-ASR DML model directory not found. "
        f"Tried: {candidate}, {nested}, {sibling_dml}"
    )


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    return LANGUAGE_ALIASES.get(value.lower(), value)


@dataclass
class Qwen3ASRDmlConfig:
    model_dir: Path
    use_dml: bool = True
    chunk_size: float = 40.0
    memory_chunks: int = 1
    verbose: bool = False


class Qwen3ASRDmlModel:
    def __init__(self, config: Qwen3ASRDmlConfig):
        self.config = config
        self.device = "dml+cpu" if config.use_dml else "cpu"
        self._engine = QwenASREngine(
            ASREngineConfig(
                model_dir=str(config.model_dir),
                use_dml=config.use_dml,
                chunk_size=config.chunk_size,
                memory_num=config.memory_chunks,
                verbose=config.verbose,
            )
        )

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        device: str = "dml",
        *,
        chunk_size: float = 40.0,
        memory_chunks: int = 1,
        verbose: bool = False,
    ) -> "Qwen3ASRDmlModel":
        resolved_dir = resolve_model_dir(model_path)
        requested_device = normalize_runtime_device(device)
        return cls(
            Qwen3ASRDmlConfig(
                model_dir=resolved_dir,
                use_dml=use_dml(requested_device),
                chunk_size=chunk_size,
                memory_chunks=memory_chunks,
                verbose=verbose,
            )
        )

    def transcribe(self, audio, language: str | None = None, context: str | None = None):
        normalized_language = normalize_language(language)
        if isinstance(audio, (str, Path)):
            return self._engine.transcribe(
                audio_file=str(audio),
                language=normalized_language,
                context=context,
            )
        if isinstance(audio, Iterable):
            return [
                self._engine.transcribe(
                    audio_file=str(path),
                    language=normalized_language,
                    context=context,
                )
                for path in audio
            ]
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
