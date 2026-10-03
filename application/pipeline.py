import os

from inference.API.asr_api import AsrSubprocessSession
from inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline
from application.config import PipelineConfig
from application.exceptions import (
    Vocal2MidiError,
    ModelNotFoundError,
    CancellationError,
)


def open_asr_session() -> AsrSubprocessSession:
    """Open a reusable Qwen ASR subprocess session for a batch of jobs.

    The worker process (and the multi-GB model inside it) is spawned lazily
    on first use and shared by every job that receives the session via
    ``PipelineConfig.asr_session``; ``close()`` releases it. Jobs that never
    reach the Qwen text-ASR stage never spawn a worker.
    """
    return AsrSubprocessSession()


def _uses_pinyin_asr(cfg: PipelineConfig) -> bool:
    if (cfg.language or "").strip().lower() == "zh-pinyin":
        return True
    lang = (cfg.language or "").strip().lower()
    return lang in {"zh", "zh-pinyin"} and str(cfg.chinese_asr_engine or "").strip().lower() == "pinyin"


def _ja_uses_romaji_asr(cfg: PipelineConfig) -> bool:
    return (cfg.language or "").strip().lower() == "ja" and str(
        cfg.japanese_asr_engine or "romaji"
    ).strip().lower() != "qwen"


def _validate_model_paths(cfg: PipelineConfig) -> None:
    """Validate that required model paths exist before starting the pipeline."""
    required_paths = [("GAME model dir", cfg.game_model_dir)]
    if cfg.output_lyrics:
        # The forced-alignment engine decides which aligner model is required.
        if str(cfg.alignment_engine or "").strip().lower() == "tifa":
            required_paths.append(("TiFA model dir", cfg.tifa_model_path))
        else:
            required_paths.append(("HubertFA model dir", cfg.hfa_model_dir))
        if _uses_pinyin_asr(cfg):
            # Chinese + pinyin ASR routes to the pinyin model; Qwen is not used.
            if cfg.pinyin_asr_model_path:
                required_paths.append(("Pinyin ASR model path", cfg.pinyin_asr_model_path))
        else:
            required_paths.append(("ASR model path", cfg.asr_model_path))
            # A provided romaji ASR path is used as-is by the ja pipeline;
            # an empty one degrades gracefully, a broken one must fail here.
            # With the Qwen engine selected for Japanese it is bypassed entirely.
            if _ja_uses_romaji_asr(cfg) and cfg.phoneme_asr_model_path:
                required_paths.append(("Phoneme ASR model path", cfg.phoneme_asr_model_path))

    # Pitch curves require RMVPE; fail fast with a clear path instead of a
    # mid-run error inside RmvpeTranscriber.
    wants_pitch_curve = cfg.output_pitch_curve and any(
        fmt in (cfg.output_formats or []) for fmt in ("ustx", "vsqx")
    )
    if wants_pitch_curve and cfg.rmvpe_model_path:
        required_paths.append(("RMVPE model path", cfg.rmvpe_model_path))

    errors = []
    for label, path in required_paths:
        if not path or not os.path.exists(path):
            errors.append(f"{label} does not exist or is invalid: {path}")
    if errors:
        raise ModelNotFoundError(
            "Model path validation failed",
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
