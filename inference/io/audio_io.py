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
import threading
import time

import numpy as np

_FFMPEG_PATH = pathlib.Path(__file__).resolve().parents[2] / "ffmpeg.exe"
_DECODE_TIMEOUT_SEC = 600
_STDOUT_BLOCK = 1 << 20


class AudioLoadError(RuntimeError):
    """Raised when the bundled ffmpeg fails to decode an audio file."""


def _find_ffmpeg() -> str:
    if _FFMPEG_PATH.is_file():
        return str(_FFMPEG_PATH)
    found = shutil.which("ffmpeg")
    if found is None:
        raise AudioLoadError(
            f"ffmpeg not found: expected the bundled copy at '{_FFFMPEG_PATH}' "
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
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    # Stream stdout into a growable buffer instead of communicate(): decoding
    # a long file into memory once (~1x PCM) beats capture_output's copy
    # (~2x PCM peak). stderr is drained on a thread to avoid pipe deadlock.
    stderr_bytes = bytearray()
    pcm = bytearray()

    def _drain_stderr():
        try:
            stderr_bytes.extend(process.stderr.read())
        except Exception:
            pass

    def _pump_stdout():
        while True:
            block = process.stdout.read(_STDOUT_BLOCK)
            if not block:
                break
            pcm.extend(block)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    stdout_thread = threading.Thread(target=_pump_stdout, daemon=True)
    stdout_thread.start()

    timed_out = False
    try:
        process.wait(timeout=_DECODE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        try:
            process.wait(timeout=10)
        except Exception:
            pass
    stdout_thread.join(timeout=10)
    stderr_thread.join(timeout=5)

    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
    if timed_out:
        raise AudioLoadError(
            f"Decoding '{path}' timed out after {_DECODE_TIMEOUT_SEC}s"
        )
    if process.returncode != 0:
        raise AudioLoadError(
            f"Failed to decode '{path}' with the bundled ffmpeg: {stderr_text[:500]}"
        )

    waveform = np.frombuffer(pcm, dtype="<f4")
    if waveform.size == 0:
        raise AudioLoadError(f"ffmpeg decoded zero samples from '{path}'")
    if not mono:
        waveform = waveform.reshape(-1, 2).T
    return waveform, int(sr)
