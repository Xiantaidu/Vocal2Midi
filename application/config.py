"""Pipeline configuration dataclass for Vocal2Midi.

Replaces the 30+ parameter dict/kwargs pattern used between
GUI -> application -> inference layers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch


@dataclass
class PipelineConfig:
    """Configuration for the auto-lyric hybrid pipeline.

    Required fields have no default value.
    Optional fields have sensible defaults matching the GUI.
    """

    # ── Required ──────────────────────────────────────────────
    audio_path: str
    output_filename: str
    output_dir: Path
    game_model_dir: str
    hfa_model_dir: str
    asr_model_path: str
    device: str
    language: str
    ts: torch.Tensor

    # ── Optional with defaults ────────────────────────────────
    lyric_output_mode: str = "auto"
    original_lyrics: str = ""
    output_formats: list = field(default_factory=lambda: ["mid"])
    slicing_method: str = "auto"
    tempo: float = 120.0
    quantization_step: int = 16
    quantization_mode: str = "simple"
    pitch_format: str = "midi"
    round_pitch: bool = True
    seg_threshold: float = -40.0
    seg_radius: float = 0.2
    est_threshold: float = 0.5
    batch_size: int = 8
    asr_batch_size: int = 4
    output_lyrics: bool = True
    rmvpe_model_path: str = ""
    phoneme_asr_model_path: str = ""
    use_phoneme_asr_for_ja_without_lyrics: bool = False
    output_pitch_curve: bool = False
    debug_mode: bool = False
    cancel_checker: Optional[Callable[[], bool]] = None

    def to_kwargs(self) -> dict:
        """Convert to dict for passing to auto_lyric_hybrid_pipeline."""
        return {
            "audio_path": self.audio_path,
            "output_filename": self.output_filename,
            "output_dir": self.output_dir,
            "game_model_dir": self.game_model_dir,
            "hfa_model_dir": self.hfa_model_dir,
            "asr_model_path": self.asr_model_path,
            "device": self.device,
            "language": self.language,
            "ts": self.ts,
            "lyric_output_mode": self.lyric_output_mode,
            "original_lyrics": self.original_lyrics,
            "output_formats": self.output_formats,
            "slicing_method": self.slicing_method,
            "tempo": self.tempo,
            "quantization_step": self.quantization_step,
            "quantization_mode": self.quantization_mode,
            "pitch_format": self.pitch_format,
            "round_pitch": self.round_pitch,
            "seg_threshold": self.seg_threshold,
            "seg_radius": self.seg_radius,
            "est_threshold": self.est_threshold,
            "batch_size": self.batch_size,
            "asr_batch_size": self.asr_batch_size,
            "output_lyrics": self.output_lyrics,
            "rmvpe_model_path": self.rmvpe_model_path,
            "phoneme_asr_model_path": self.phoneme_asr_model_path,
            "use_phoneme_asr_for_ja_without_lyrics": self.use_phoneme_asr_for_ja_without_lyrics,
            "output_pitch_curve": self.output_pitch_curve,
            "debug_mode": self.debug_mode,
            "cancel_checker": self.cancel_checker,
        }