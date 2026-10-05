"""BPM input box with drag-and-drop audio BPM detection support."""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from PySide6.QtCore import QEvent, QThread, Qt, Signal
from qfluentwidgets import DoubleSpinBox

from gui.audio_file_list import AudioFileList
from gui.i18n import tr


def round_bpm(val: float) -> float:
    """Round BPM to one decimal place, with the 2nd decimal place rounded and set to 0.

    Mathematically:
        120.46 -> 120.5 (displayed as 120.50 in a 2-decimal spin box)
        120.44 -> 120.4 (displayed as 120.40)
        120.45 -> 120.5 (displayed as 120.50)
        120.00 -> 120.0 (displayed as 120.00)
    """
    if val is None or math.isnan(val) or math.isinf(val) or val <= 0:
        return 120.0
    d = Decimal(f"{float(val):.6f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(d)


def detect_audio_bpm(audio_path: str | Path) -> float:
    """Detect BPM of an audio file and return the rounded BPM value.

    Args:
        audio_path: Path to the target audio file.

    Returns:
        Estimated BPM float value rounded to the first decimal place.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sr = 22050
    waveform = None

    # First attempt decoding through the bundled ffmpeg pipeline
    try:
        from inference.io.audio_io import load_audio
        waveform, sr = load_audio(path, sr=sr, mono=True)
    except Exception:
        waveform = None

    # Fallback to librosa.load if ffmpeg decoding failed
    if waveform is None or len(waveform) == 0:
        import librosa
        waveform, sr = librosa.load(str(path), sr=sr, mono=True)

    if len(waveform) < sr * 0.5:
        raise ValueError(f"Audio file '{path.name}' is too short for BPM detection")

    # Limit analysis window to the first 240s for responsiveness on long audio
    if len(waveform) > sr * 240:
        waveform = waveform[: sr * 240]

    import librosa
    import numpy as np

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
    if len(onset_env) < 10 or np.all(onset_env == 0):
        raise ValueError(f"Could not extract rhythmic features from '{path.name}'")

    fps = sr / hop_length
    max_lag = min(len(onset_env) - 1, int(fps * 60.0 / 30.0))  # down to 30 BPM
    ac = librosa.autocorrelate(onset_env, max_size=max_lag + 1)

    min_bpm, max_bpm = 40.0, 260.0
    min_lag = max(1, int(fps * 60.0 / max_bpm))
    max_lag = min(len(ac) - 1, int(fps * 60.0 / min_bpm))

    if len(ac) < min_lag + 2 or np.max(ac) <= 0:
        t = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        return round_bpm(float(np.atleast_1d(t)[0]))

    bpms = np.zeros(len(ac))
    bpms[1:] = 60.0 * fps / np.arange(1, len(ac))

    # Log-normal prior centered at 120 BPM
    logprior = -0.5 * ((np.log2(np.maximum(1e-5, bpms)) - np.log2(120.0)) / 1.0) ** 2
    norm_ac = ac / (np.max(ac) + 1e-9)
    curve = np.log1p(1e6 * norm_ac) + logprior
    curve[:min_lag] = -np.inf
    if max_lag + 1 < len(curve):
        curve[max_lag + 1 :] = -np.inf

    best_idx = int(np.argmax(curve))
    if best_idx <= 0 or best_idx >= len(curve) - 1 or np.isneginf(curve[best_idx]):
        t = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        return round_bpm(float(np.atleast_1d(t)[0]))

    # Quadratic/parabolic peak interpolation for sub-frame accuracy
    alpha = curve[best_idx - 1]
    beta = curve[best_idx]
    gamma = curve[best_idx + 1]
    denom = alpha - 2 * beta + gamma
    delta = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
    delta = max(-0.5, min(0.5, delta))
    sub_lag = best_idx + delta

    if sub_lag <= 0:
        raw_bpm = 120.0
    else:
        raw_bpm = float(60.0 * fps / sub_lag)

    return round_bpm(raw_bpm)


class BpmDetectorWorker(QThread):
    """Background worker thread to perform audio decoding and BPM detection."""

    detected = Signal(float, str)
    failed = Signal(str, str)

    def __init__(self, audio_path: str, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path

    def run(self):
        try:
            bpm = detect_audio_bpm(self.audio_path)
            self.detected.emit(bpm, self.audio_path)
        except Exception as exc:
            self.failed.emit(str(exc), self.audio_path)


class BpmSpinBox(DoubleSpinBox):
    """DoubleSpinBox tailored for BPM input with drag-and-drop file detection."""

    bpmDetectStarted = Signal(str)
    bpmDetected = Signal(float, str)
    bpmDetectFailed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAcceptDrops(True)
        self.setDecimals(2)
        self.setToolTip(tr("tempo_bpm_tooltip"))

        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setAcceptDrops(True)
            line_edit.installEventFilter(self)

        self._active_worker: BpmDetectorWorker | None = None

    def eventFilter(self, watched, event):
        if watched == self.lineEdit():
            if event.type() == QEvent.Type.DragEnter:
                self.dragEnterEvent(event)
                return True
            elif event.type() == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
                return True
            elif event.type() == QEvent.Type.Drop:
                self.dropEvent(event)
                return True
        return super().eventFilter(watched, event)

    def _has_supported_audio(self, event) -> bool:
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            return False
        for url in mime.urls():
            if url.isLocalFile():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in AudioFileList.AUDIO_EXTENSIONS:
                    return True
        return False

    def dragEnterEvent(self, event):
        if self._has_supported_audio(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._has_supported_audio(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            event.ignore()
            return

        target_file = None
        for url in mime.urls():
            if url.isLocalFile():
                p = Path(url.toLocalFile())
                if p.suffix.lower() in AudioFileList.AUDIO_EXTENSIONS and p.is_file():
                    target_file = str(p.resolve())
                    break

        if not target_file:
            event.ignore()
            return

        event.acceptProposedAction()
        self.recognize_bpm(target_file)

    def recognize_bpm(self, file_path: str):
        """Trigger background BPM detection for the given audio file."""
        if self._active_worker is not None and self._active_worker.isRunning():
            try:
                self._active_worker.detected.disconnect()
                self._active_worker.failed.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._active_worker.quit()
            self._active_worker.wait(100)

        self.bpmDetectStarted.emit(file_path)
        worker = BpmDetectorWorker(file_path, self)
        self._active_worker = worker
        worker.detected.connect(self._on_detected)
        worker.failed.connect(self._on_failed)
        worker.start()

    def _on_detected(self, bpm: float, path: str):
        clamped = max(self.minimum(), min(self.maximum(), bpm))
        self.setValue(clamped)
        self.bpmDetected.emit(clamped, path)

    def _on_failed(self, error: str, path: str):
        self.bpmDetectFailed.emit(error, path)
