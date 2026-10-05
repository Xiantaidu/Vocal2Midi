"""Unit and integration tests for BpmSpinBox and BPM detection."""
import os
import pathlib
import wave
import struct

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt, QUrl, QMimeData, QEvent
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from gui.bpm_spinbox import BpmSpinBox, round_bpm, detect_audio_bpm


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def create_synthetic_wav(path: pathlib.Path, bpm: float = 120.0, duration: float = 10.0, sr: int = 22050):
    """Generate a clean click/beat WAV file with a given BPM."""
    total_samples = int(duration * sr)
    beat_interval = 60.0 / bpm
    samples = np.zeros(total_samples, dtype=np.int16)

    for beat_time in np.arange(0, duration, beat_interval):
        idx = int(beat_time * sr)
        # click burst: 100 samples
        burst_len = min(100, total_samples - idx)
        if burst_len > 0:
            samples[idx : idx + burst_len] = 20000

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def test_round_bpm_accuracy():
    """Verify that the 2nd decimal place is rounded and set to 0 (precision = 0.1)."""
    assert round_bpm(120.46) == 120.5
    assert f"{round_bpm(120.46):.2f}" == "120.50"

    assert round_bpm(120.44) == 120.4
    assert f"{round_bpm(120.44):.2f}" == "120.40"

    assert round_bpm(120.45) == 120.5
    assert f"{round_bpm(120.45):.2f}" == "120.50"

    assert round_bpm(120.0) == 120.0
    assert f"{round_bpm(120.0):.2f}" == "120.00"

    assert round_bpm(135.96) == 136.0
    assert f"{round_bpm(135.96):.2f}" == "136.00"

    # Edge cases
    assert round_bpm(None) == 120.0
    assert round_bpm(float("nan")) == 120.0
    assert round_bpm(float("inf")) == 120.0
    assert round_bpm(-50.0) == 120.0


def test_detect_audio_bpm_synthetic(tmp_path):
    """Test BPM detection on a generated wav file with 120 BPM."""
    wav_path = tmp_path / "test_120bpm.wav"
    create_synthetic_wav(wav_path, bpm=120.0, duration=15.0)

    bpm = detect_audio_bpm(wav_path)
    # Target is 120.0; accurate to 1 decimal place with 2nd place 0
    assert isinstance(bpm, float)
    assert abs(bpm - 120.0) <= 1.0
    # Must be rounded to 1 decimal place
    assert round(bpm, 1) == bpm
    assert f"{bpm:.2f}".endswith("0")


def test_detect_audio_bpm_invalid_file(tmp_path):
    """Test error handling on missing or empty files."""
    with pytest.raises(FileNotFoundError):
        detect_audio_bpm(tmp_path / "nonexistent.wav")

    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(Exception):
        detect_audio_bpm(empty_wav)


def test_bpm_spinbox_properties(qapp):
    box = BpmSpinBox()
    box.setRange(10, 300)
    assert box.decimals() == 2
    assert box.toolTip() != ""

    box.setValue(round_bpm(120.46))
    assert box.value() == 120.5
    assert box.text() == "120.50"

    box.setValue(round_bpm(120.44))
    assert box.value() == 120.4
    assert box.text() == "120.40"


def test_bpm_spinbox_drag_filter(qapp):
    box = BpmSpinBox()

    # Drag non-audio file
    mime_txt = QMimeData()
    mime_txt.setUrls([QUrl.fromLocalFile("test.txt")])
    event_enter_txt = QDragEnterEvent(QPointF(5, 5).toPoint(), Qt.CopyAction, mime_txt, Qt.LeftButton, Qt.NoModifier)
    box.dragEnterEvent(event_enter_txt)
    assert not event_enter_txt.isAccepted()

    # Drag audio file
    mime_wav = QMimeData()
    mime_wav.setUrls([QUrl.fromLocalFile("test.wav")])
    event_enter_wav = QDragEnterEvent(QPointF(5, 5).toPoint(), Qt.CopyAction, mime_wav, Qt.LeftButton, Qt.NoModifier)
    box.dragEnterEvent(event_enter_wav)
    assert event_enter_wav.isAccepted()


def test_bpm_spinbox_drop_flow(qapp, tmp_path):
    wav_path = tmp_path / "song.wav"
    create_synthetic_wav(wav_path, bpm=128.0, duration=15.0)

    box = BpmSpinBox()
    box.setRange(10, 300)

    started_events = []
    detected_events = []
    box.bpmDetectStarted.connect(started_events.append)
    box.bpmDetected.connect(lambda b, p: detected_events.append((b, p)))

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(wav_path))])
    drop_ev = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)

    box.dropEvent(drop_ev)
    assert drop_ev.isAccepted()
    assert len(started_events) == 1
    assert started_events[0] == str(wav_path.resolve())

    # Wait for the worker thread to finish and dispatch queued signals
    if box._active_worker is not None:
        box._active_worker.wait(10000)
    qapp.processEvents()

    assert len(detected_events) == 1
    bpm, path = detected_events[0]
    assert abs(bpm - 128.0) <= 1.5
    assert box.value() == bpm
    assert f"{box.value():.2f}".endswith("0")


def test_bpm_spinbox_lineedit_event_filter(qapp, tmp_path):
    wav_path = tmp_path / "lineedit_song.wav"
    create_synthetic_wav(wav_path, bpm=100.0, duration=15.0)

    box = BpmSpinBox()
    box.setRange(10, 300)

    detected = []
    box.bpmDetected.connect(lambda b, p: detected.append(b))

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(wav_path))])
    drop_ev = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)

    handled = box.eventFilter(box.lineEdit(), drop_ev)
    assert handled is True

    if box._active_worker is not None:
        box._active_worker.wait(10000)
    qapp.processEvents()

    assert len(detected) == 1
    assert abs(detected[0] - 100.0) <= 1.5
    assert box.value() == detected[0]
