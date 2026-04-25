import os

from inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline
from application.config import PipelineConfig
from application.exceptions import (
    Vocal2MidiError,
    ModelNotFoundError,
    CancellationError,
)


def _validate_model_paths(cfg: PipelineConfig) -> None:
    """Validate that required model paths exist before starting the pipeline."""
    errors = []
    for label, path in [
        ("GAME 模型目录", cfg.game_model_dir),
        ("HubertFA 模型目录", cfg.hfa_model_dir),
        ("ASR 模型路径", cfg.asr_model_path),
    ]:
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
