"""Audio loading through the bundled minimal ffmpeg.

Every user-facing audio input takes one uniform decode path: the audio-only
``ffmpeg.exe`` at the repository root decodes any supported container straight
to raw float32 PCM on stdout. Nothing is written to disk and there is no
dependence on libsndfile's per-format read support (librosa 0.11 itself has no
ffmpeg fallback, so formats like M4A would be unreadable without this).

Waveforms are always returned as mono float32 at the requested sample rate;
resampling and channel mixdown happen inside ffmpeg (swresample).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import numpy as np

_FFMPEG_PATH = pathlib.Path(__file__).resolve().parents[2] / "ffmpeg.exe"
_DECODE_TIMEOUT_SEC = 600


class AudioLoadError(RuntimeError):
    """Raised when the bundled ffmpeg fails to decode an audio file."""


def _find_ffmpeg() -> str:
    if _FFMPEG_PATH.is_file():
        return str(_FFMPEG_PATH)
    found = shutil.which("ffmpeg")
    if found is None:
        raise AudioLoadError(
            f"ffmpeg not found: expected the bundled copy at '{_FFMPEG_PATH}' "
            "or an 'ffmpeg' on PATH."
        )
    return found


def load_audio(path: str | pathlib.Path, sr: int, mono: bool = True):
    """Decode an audio file to float32 PCM with the bundled ffmpeg.

    Args:
        path: Audio file path (any format the bundled ffmpeg decodes:
              wav/flac/mp3/ogg/opus/m4a/aac/wma/webm/aiff/...).
        sr: Target sample rate; ffmpeg resamples via swresample.
        mono: Downmix to one channel (all current callers want mono).

    Returns:
        Tuple of ``(waveform, sr)`` with waveform as 1-D float32.
    """
    cmd = [
        _find_ffmpeg(),
        "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(int(sr)),
        "-f", "f32le",
        "-",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=_DECODE_TIMEOUT_SEC,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise AudioLoadError(
            f"Failed to decode '{path}' with the bundled ffmpeg: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()[:500]}"
        )

    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if waveform.size == 0:
        raise AudioLoadError(f"ffmpeg decoded zero samples from '{path}'")
    if not mono:
        waveform = waveform.reshape(-1, 2).T
    return waveform, int(sr)
