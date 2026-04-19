from pathlib import Path
from typing import Callable, Optional

import torch

from inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline


def run_auto_lyric_job(
    *,
    audio_path: str,
    output_filename: str,
    output_dir: Path,
    game_model_dir: str,
    hfa_model_dir: str,
    asr_model_path: str,
    device: str,
    ts: torch.Tensor,
    language: str,
    lyric_output_mode: str,
    original_lyrics: str,
    output_formats: list,
    slicing_method: str,
    tempo: float,
    quantization_step: int,
    quantization_mode: str,
    pitch_format: str,
    round_pitch: bool,
    seg_threshold: float,
    seg_radius: float,
    est_threshold: float,
    batch_size: int,
    asr_batch_size: int,
    output_lyrics: bool,
    output_pitch_curve: bool,
    debug_mode: bool,
    cancel_checker: Optional[Callable[[], bool]] = None,
):
    """Application-layer entry for the primary auto lyric extraction use-case.

    GUI should call this function instead of importing inference pipeline directly.
    """
    auto_lyric_hybrid_pipeline(
        audio_path=audio_path,
        output_filename=output_filename,
        game_model_dir=game_model_dir,
        device=device,
        hfa_model_dir=hfa_model_dir,
        asr_model_path=asr_model_path,
        ts=ts,
        language=language,
        lyric_output_mode=lyric_output_mode,
        original_lyrics=original_lyrics,
        output_dir=output_dir,
        output_formats=output_formats,
        slicing_method=slicing_method,
        tempo=tempo,
        quantization_step=quantization_step,
        quantization_mode=quantization_mode,
        pitch_format=pitch_format,
        round_pitch=round_pitch,
        seg_threshold=seg_threshold,
        seg_radius=seg_radius,
        est_threshold=est_threshold,
        batch_size=batch_size,
        asr_batch_size=asr_batch_size,
        output_lyrics=output_lyrics,
        output_pitch_curve=output_pitch_curve,
        debug_mode=debug_mode,
        cancel_checker=cancel_checker,
    )
