"""Tests for the per-file settings dialog and the shared option tables.

The dialog must round-trip backend values only (no display-text transport)
and honor the zh + PinyinASR lyric output lock.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from gui import option_tables
from gui.file_settings_dialog import FileSettingsDialog
from gui.i18n import tr


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def parent_widget(qapp):
    widget = QWidget()
    widget.resize(800, 600)
    yield widget


def _base_values(**overrides) -> dict:
    values = {
        "slicing_method": "smart",
        "language": "zh",
        "lyric_output": "hanzi",
        "device": "cpu",
        "match_lyrics": False,
        "original_lyrics": "",
        "export_format": "mid",
        "output_lyrics": True,
        "pitch_curve": True,
        "tempo": 120.0,
        "quantization_step": 0,
        "quantization_mode": "smart",
        "batch_size": 1,
        "asr_batch_size": 2,
        "output_dir": "out",
        "devices": ["cpu", "dml"],
        "chinese_asr_engine": "qwen",
        "japanese_asr_engine": "romaji",
    }
    values.update(overrides)
    return values


def test_values_round_trip_backend_values(parent_widget):
    dialog = FileSettingsDialog("a.wav", _base_values(), parent_widget)
    values = dialog.values()
    assert values["slicing_method"] == "smart"
    assert values["language"] == "zh"
    assert values["lyric_output"] == "hanzi"
    assert values["export_format"] == "mid"
    assert values["quantization_step"] == 0
    assert values["quantization_mode"] == "smart"


def test_dialog_edits_return_backend_values(parent_widget):
    dialog = FileSettingsDialog("a.wav", _base_values(), parent_widget)
    dialog.quantize_combo.setCurrentIndex(dialog.quantize_combo.findData(480))
    dialog.export_format_combo.setCurrentIndex(dialog.export_format_combo.findData("ustx"))
    dialog.lang_combo.setCurrentIndex(dialog.lang_combo.findData("ja"))
    values = dialog.values()
    assert values["quantization_step"] == 480
    assert values["export_format"] == "ustx"
    assert values["language"] == "ja"
    # switching to ja auto-selects that language's default lyric format
    assert values["lyric_output"] == "romaji"


def test_pinyin_engine_locks_lyric_output(parent_widget):
    dialog = FileSettingsDialog(
        "a.wav", _base_values(chinese_asr_engine="pinyin", lyric_output="hanzi"), parent_widget
    )
    assert dialog.lyric_output_combo.count() == 1
    assert dialog.lyric_output_combo.currentData() == "pinyin"
    assert not dialog.lyric_output_combo.isEnabled()
    assert dialog.values()["lyric_output"] == "pinyin"


def test_unknown_language_value_falls_back_to_zh(parent_widget):
    dialog = FileSettingsDialog("a.wav", _base_values(language="invalid_language"), parent_widget)
    assert dialog.lang_combo.currentData() == "zh"
    assert dialog.values()["language"] == "zh"


def test_slicing_values_are_slicer_api_canonical():
    from inference.API.slicer_api import SLICE_METHOD_CHOICES as CANONICAL

    for value, _key in option_tables.SLICE_METHOD_CHOICES:
        assert value in CANONICAL


def test_option_tables_have_translations():
    tables = (
        option_tables.TARGET_LANGUAGE_CHOICES,
        option_tables.SLICE_METHOD_CHOICES,
        option_tables.EXPORT_FORMAT_CHOICES,
        option_tables.QUANT_STEP_CHOICES,
        option_tables.QUANT_MODE_CHOICES,
        *option_tables.LYRIC_OUTPUT_BY_LANGUAGE.values(),
    )
    for table in tables:
        for _value, key in table:
            # tr() returns the key itself when the entry is missing
            assert tr(key) != key, f"missing i18n key: {key}"
