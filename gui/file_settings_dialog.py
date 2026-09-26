"""Per-file settings dialog for batch mode.

Presents the full parameter set that the main UI controls, seeded from the
main UI's current values, so each file in a batch can be customized
individually. Combos are filled from the shared option tables and every
values() entry is a backend value — display text is never transported.
"""
from __future__ import annotations

from qfluentwidgets import (
    MessageBoxBase,
    BodyLabel,
    ComboBox,
    DoubleSpinBox,
    LineEdit,
    SwitchButton,
    SpinBox,
)
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QLabel

from gui.i18n import tr
from gui.option_tables import (
    DEFAULT_LYRIC_OUTPUT,
    EXPORT_FORMAT_CHOICES,
    LYRIC_OUTPUT_BY_LANGUAGE,
    QUANT_MODE_CHOICES,
    QUANT_STEP_CHOICES,
    SLICE_METHOD_CHOICES,
    TARGET_LANGUAGE_CHOICES,
    fill_combo,
)
from gui.settings_utils import default_output_dir


def _add_pair(grid, row, col, label_text, widget, dialog):
    label = BodyLabel(label_text, dialog)
    grid.addWidget(label, row, col)
    grid.addWidget(widget, row, col + 1)


class FileSettingsDialog(MessageBoxBase):
    """Custom MessageBox with the full per-file parameter set."""

    def __init__(self, filename: str, base_values: dict, parent=None):
        super().__init__(parent)
        self.yesButton.setText(tr("apply"))
        self.cancelButton.setText(tr("cancel"))

        title = QLabel(f"{filename}", self)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.viewLayout.addWidget(title)
        hint = BodyLabel(tr("file_settings_hint"), self)
        hint.setStyleSheet("font-size: 12px;")
        self.viewLayout.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)

        # ── recognition ─────────────────────────────────────────────
        # The Chinese ASR engine is a global choice; when PinyinASR is active
        # the lyric output for zh is locked to pinyin here as well.
        self._pinyin_locked = str(base_values.get("chinese_asr_engine", "qwen")).strip().lower() == "pinyin"

        self.slicing_combo = ComboBox(self)
        fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES)
        self.slicing_combo.setCurrentIndex(max(0, self.slicing_combo.findData(base_values["slicing_method"])))
        _add_pair(grid, 0, 0, tr("slicing_method"), self.slicing_combo, self)

        self.lang_combo = ComboBox(self)
        fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(base_values["language"])))
        _add_pair(grid, 0, 2, tr("target_lang"), self.lang_combo, self)

        self.lyric_output_combo = ComboBox(self)
        self._fill_lyric_output_options(base_values["language"], base_values["lyric_output"])
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        _add_pair(grid, 1, 0, tr("lyric_output_format"), self.lyric_output_combo, self)

        self.device_combo = ComboBox(self)
        self.device_combo.addItems(list(base_values["devices"]))
        self.device_combo.setCurrentText(base_values["device"])
        _add_pair(grid, 1, 2, tr("device"), self.device_combo, self)

        self.cb_match_lyrics = SwitchButton("On", self)
        self.cb_match_lyrics.setOffText("Off")
        self.cb_match_lyrics.setChecked(base_values["match_lyrics"])
        _add_pair(grid, 2, 0, tr("match_lyrics"), self.cb_match_lyrics, self)

        self.lyrics_edit = LineEdit(self)
        self.lyrics_edit.setPlaceholderText(tr("ref_lyrics_hint"))
        self.lyrics_edit.setText(base_values["original_lyrics"])
        _add_pair(grid, 2, 2, tr("ref_lyrics"), self.lyrics_edit, self)

        self.export_format_combo = ComboBox(self)
        fill_combo(self.export_format_combo, EXPORT_FORMAT_CHOICES)
        self.export_format_combo.setCurrentIndex(max(0, self.export_format_combo.findData(base_values["export_format"])))
        _add_pair(grid, 3, 0, tr("export_format"), self.export_format_combo, self)

        self.cb_output_lyrics = SwitchButton("On", self)
        self.cb_output_lyrics.setOffText("Off")
        self.cb_output_lyrics.setChecked(base_values["output_lyrics"])
        _add_pair(grid, 3, 2, tr("output_lyrics"), self.cb_output_lyrics, self)

        self.cb_pitch_curve = SwitchButton("On", self)
        self.cb_pitch_curve.setOffText("Off")
        self.cb_pitch_curve.setChecked(base_values["pitch_curve"])
        _add_pair(grid, 4, 0, tr("pitch_curve"), self.cb_pitch_curve, self)

        # ── output ──────────────────────────────────────────────────
        self.tempo_spin = DoubleSpinBox(self)
        self.tempo_spin.setRange(10, 300)
        self.tempo_spin.setValue(float(base_values["tempo"]))
        _add_pair(grid, 5, 0, tr("tempo_bpm"), self.tempo_spin, self)

        self.quantize_combo = ComboBox(self)
        fill_combo(self.quantize_combo, QUANT_STEP_CHOICES)
        self.quantize_combo.setCurrentIndex(max(0, self.quantize_combo.findData(base_values["quantization_step"])))
        _add_pair(grid, 5, 2, tr("quant_step"), self.quantize_combo, self)

        self.quantize_mode_combo = ComboBox(self)
        fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES)
        self.quantize_mode_combo.setCurrentIndex(max(0, self.quantize_mode_combo.findData(base_values["quantization_mode"])))
        _add_pair(grid, 6, 0, tr("quant_mode"), self.quantize_mode_combo, self)

        self.batch_spin = SpinBox(self)
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(int(base_values["batch_size"]))
        _add_pair(grid, 6, 2, tr("game_batch"), self.batch_spin, self)

        self.asr_batch_spin = SpinBox(self)
        self.asr_batch_spin.setRange(1, 32)
        self.asr_batch_spin.setValue(int(base_values["asr_batch_size"]))
        _add_pair(grid, 7, 0, tr("asr_batch"), self.asr_batch_spin, self)

        self.save_dir_edit = LineEdit(self)
        self.save_dir_edit.setText(base_values["output_dir"])
        _add_pair(grid, 7, 2, tr("save_dir"), self.save_dir_edit, self)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.viewLayout.addLayout(grid)

    # ── language-linked lyric output options ────────────────────────
    def _fill_lyric_output_options(self, language: str, selected: str | None):
        """Rebuild the lyric output options for the given language.

        `selected` is a backend value; the combo keeps it when still
        available. Under zh with PinyinASR the format is locked to pinyin
        and the combo is disabled.
        """
        locked = language == "zh" and self._pinyin_locked
        choices = [
            (value, key)
            for value, key in LYRIC_OUTPUT_BY_LANGUAGE.get(language, LYRIC_OUTPUT_BY_LANGUAGE["zh"])
            if not locked or value == "pinyin"
        ]
        fill_combo(self.lyric_output_combo, choices)
        values = [value for value, _ in choices]
        if selected not in values:
            selected = "pinyin" if locked else DEFAULT_LYRIC_OUTPUT.get(language, "hanzi")
        self.lyric_output_combo.setCurrentIndex(max(0, self.lyric_output_combo.findData(selected)))
        self.lyric_output_combo.setEnabled(not locked)
        self.lyric_output_combo.setToolTip(tr("lyric_output_locked_hint") if locked else "")

    def _on_language_changed(self):
        self._fill_lyric_output_options(self.lang_combo.currentData(), self.lyric_output_combo.currentData())

    def values(self) -> dict:
        """Return the edited values as a plain dict keyed by config field names."""
        return {
            "slicing_method": self.slicing_combo.currentData(),
            "language": self.lang_combo.currentData(),
            "lyric_output": self.lyric_output_combo.currentData(),
            "device": self.device_combo.currentText(),
            "match_lyrics": self.cb_match_lyrics.isChecked(),
            "original_lyrics": self.lyrics_edit.text().strip(),
            "export_format": self.export_format_combo.currentData(),
            "output_lyrics": self.cb_output_lyrics.isChecked(),
            "pitch_curve": self.cb_pitch_curve.isChecked(),
            "tempo": self.tempo_spin.value(),
            "quantization_step": self.quantize_combo.currentData(),
            "quantization_mode": self.quantize_mode_combo.currentData(),
            "batch_size": self.batch_spin.value(),
            "asr_batch_size": self.asr_batch_spin.value(),
            "output_dir": self.save_dir_edit.text().strip() or str(default_output_dir()),
        }
