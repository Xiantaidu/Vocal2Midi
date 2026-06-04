from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from inference.device_utils import normalize_runtime_device, use_dml

from .asr import QwenASREngine
from .schema import ASREngineConfig


LLM_VARIANTS = (
    "qwen3_asr_llm.f16.gguf",
    "qwen3_asr_llm.q4_k.gguf",
)
ENCODER_VARIANTS = (
    ("qwen3_asr_encoder_frontend.fp16.onnx", "qwen3_asr_encoder_backend.fp16.onnx"),
    ("qwen3_asr_encoder_frontend.int4.onnx", "qwen3_asr_encoder_backend.int4.onnx"),
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
    if not any((model_dir / llm_name).is_file() for llm_name in LLM_VARIANTS):
        return False
    return any(
        (model_dir / frontend_name).is_file() and (model_dir / backend_name).is_file()
        for frontend_name, backend_name in ENCODER_VARIANTS
    )


def resolve_encoder_filenames(model_dir: Path) -> tuple[str, str]:
    for frontend_name, backend_name in ENCODER_VARIANTS:
        if (model_dir / frontend_name).is_file() and (model_dir / backend_name).is_file():
            return frontend_name, backend_name
    raise FileNotFoundError(f"No supported encoder model pair found in: {model_dir}")


def resolve_llm_filename(model_dir: Path) -> str:
    for llm_name in LLM_VARIANTS:
        if (model_dir / llm_name).is_file():
            return llm_name
    raise FileNotFoundError(f"No supported LLM GGUF found in: {model_dir}")


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


def resolve_llama_backend(device: str | None, llama_backend: str = "auto") -> str:
    requested_device = normalize_runtime_device(device)
    if requested_device == "cpu" and str(llama_backend or "auto").strip().lower() == "auto":
        return "cpu"
    return llama_backend


@dataclass
class Qwen3ASRDmlConfig:
    model_dir: Path
    llm_fn: str
    encoder_frontend_fn: str
    encoder_backend_fn: str
    use_dml: bool = True
    chunk_size: float = 40.0
    memory_chunks: int = 1
    max_decode_tokens: int = 512
    llama_backend: str = "auto"
    verbose: bool = False


class Qwen3ASRDmlModel:
    def __init__(self, config: Qwen3ASRDmlConfig):
        self.config = config
        self.device = "dml+cpu" if config.use_dml else "cpu"
        self._engine = QwenASREngine(
            ASREngineConfig(
                model_dir=str(config.model_dir),
                llm_fn=config.llm_fn,
                encoder_frontend_fn=config.encoder_frontend_fn,
                encoder_backend_fn=config.encoder_backend_fn,
                use_dml=config.use_dml,
                chunk_size=config.chunk_size,
                memory_num=config.memory_chunks,
                max_decode_tokens=config.max_decode_tokens,
                llama_backend=config.llama_backend,
                verbose=config.verbose,
            )
        )

    @property
    def encoder_provider_name(self) -> str:
        return getattr(self._engine, "encoder_runtime", {}).get("encoder_provider", "unknown")

    @property
    def encoder_frontend_providers(self) -> list[str]:
        return list(getattr(self._engine, "encoder_runtime", {}).get("frontend_providers", []))

    @property
    def encoder_backend_providers(self) -> list[str]:
        return list(getattr(self._engine, "encoder_runtime", {}).get("backend_providers", []))

    @property
    def decoder_backend(self) -> str:
        return getattr(self._engine, "decoder_backend", "unknown")

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        device: str = "dml",
        *,
        chunk_size: float = 40.0,
        memory_chunks: int = 1,
        max_decode_tokens: int = 512,
        llama_backend: str = "auto",
        verbose: bool = False,
    ) -> "Qwen3ASRDmlModel":
        resolved_dir = resolve_model_dir(model_path)
        llm_fn = resolve_llm_filename(resolved_dir)
        encoder_frontend_fn, encoder_backend_fn = resolve_encoder_filenames(resolved_dir)
        requested_device = normalize_runtime_device(device)
        resolved_llama_backend = resolve_llama_backend(requested_device, llama_backend)
        return cls(
            Qwen3ASRDmlConfig(
                model_dir=resolved_dir,
                llm_fn=llm_fn,
                encoder_frontend_fn=encoder_frontend_fn,
                encoder_backend_fn=encoder_backend_fn,
                use_dml=use_dml(requested_device),
                chunk_size=chunk_size,
                memory_chunks=memory_chunks,
                max_decode_tokens=max_decode_tokens,
                llama_backend=resolved_llama_backend,
                verbose=verbose,
            )
        )

    def transcribe(self, audio, language: str | None = None, context: str | None = None, batch_size: int | None = None):
        normalized_language = normalize_language(language)
        if isinstance(audio, (str, Path)):
            return self._engine.transcribe(
                audio_file=str(audio),
                language=normalized_language,
                context=context,
            )
        if isinstance(audio, Iterable):
            audio_files = [str(path) for path in audio]
            if batch_size is not None and batch_size > 0:
                results = []
                for start in range(0, len(audio_files), batch_size):
                    results.extend(
                        self._engine.transcribe_batch(
                            audio_files=audio_files[start : start + batch_size],
                            language=normalized_language,
                            context=context,
                        )
                    )
                return results
            return self._engine.transcribe_batch(
                audio_files=audio_files,
                language=normalized_language,
                context=context,
            )
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
