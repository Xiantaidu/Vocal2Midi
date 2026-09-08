import os

from inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline
from application.config import PipelineConfig
from application.exceptions import (
    Vocal2MidiError,
    ModelNotFoundError,
    CancellationError,
)


def _uses_pinyin_asr(cfg: PipelineConfig) -> bool:
    return (cfg.language or "").strip().lower() in {"zh-pinyin", "中文-拼音", "中文拼音"}


def _validate_model_paths(cfg: PipelineConfig) -> None:
    """Validate that required model paths exist before starting the pipeline."""
    required_paths = [("GAME 模型目录", cfg.game_model_dir)]
    if cfg.output_lyrics:
        required_paths.append(("HubertFA 模型目录", cfg.hfa_model_dir))
        if _uses_pinyin_asr(cfg):
            # 中文-拼音 routes to the pinyin ASR; the Qwen model is not used.
            if cfg.pinyin_asr_model_path:
                required_paths.append(("拼音ASR模型路径", cfg.pinyin_asr_model_path))
        else:
            required_paths.append(("ASR 模型路径", cfg.asr_model_path))

    errors = []
    for label, path in required_paths:
        if not path or not os.path.exists(path):
            errors.append(f"{label}不存在或无效: {path}")
    if errors:
        raise ModelNotFoundError(
            "模型路径验证失败",
            details="; ".join(errors),
        )


def run_auto_lyric_job(cfg: PipelineConfig):
    """Application-layer entry for the primary auto lyric extraction use-case.

    GUI should call this function with a PipelineConfig instead of importing
    the inference pipeline directly.

    Args:
        cfg: PipelineConfig with all parameters for the hybrid pipeline.

    Raises:
        ModelNotFoundError: If required model paths do not exist.
        CancellationError: If the user cancels the pipeline.
        Vocal2MidiError: Base exception for other pipeline errors.
    """
    _validate_model_paths(cfg)

    # Check cancellation before starting
    if cfg.cancel_checker and cfg.cancel_checker():
        raise CancellationError("Pipeline was cancelled before starting.")

    try:
        auto_lyric_hybrid_pipeline(**cfg.to_kwargs())
    except InterruptedError:
        raise CancellationError("Pipeline was interrupted by user.") from None
    except Vocal2MidiError:
        raise
    except Exception as e:
        raise Vocal2MidiError(
            f"Pipeline execution failed: {e}",
            details=str(e),
        ) from e
