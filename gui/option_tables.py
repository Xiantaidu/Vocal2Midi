"""Shared option tables for the main UI and the per-file settings dialog.

Each table maps backend values to i18n keys; widgets render the label with
tr() and carry the backend value as userData. Settings, per-file overrides
and PipelineConfig always transport backend values — display text is never
used as data.
"""
from __future__ import annotations

from qfluentwidgets import ComboBox

from gui.i18n import tr

TARGET_LANGUAGE_CHOICES = [
    ("zh", "lang_name_zh"),
    ("ja", "lang_name_ja"),
    ("en", "lang_name_en"),
]
SLICE_METHOD_CHOICES = [
    # Canonical slicer_api values (its normalize_slicing_method also accepts
    # the legacy Chinese aliases, but the GUI sends the canonical ones).
    ("smart", "slice_smart"),
    ("heuristic", "slice_heuristic"),
    ("default", "slice_default"),
    ("grid", "slice_grid"),
]
EXPORT_FORMAT_CHOICES = [
    ("mid", "export_fmt_mid"),
    ("ustx", "export_fmt_ustx"),
    ("vsqx", "export_fmt_vsqx"),
]
QUANT_STEP_CHOICES = [
    (0, "quant_off"),
    (480, "quant_1_4"),
    (240, "quant_1_8"),
    (120, "quant_1_16"),
    (60, "quant_1_32"),
    (30, "quant_1_64"),
]
QUANT_MODE_CHOICES = [
    ("smart", "quant_smart"),
    ("simple", "quant_simple"),
]
LYRIC_OUTPUT_BY_LANGUAGE = {
    "zh": [("pinyin", "opt_pinyin"), ("hanzi", "opt_hanzi")],
    "ja": [("romaji", "opt_romaji"), ("kana", "opt_kana")],
    "en": [("word", "opt_word")],
}
DEFAULT_LYRIC_OUTPUT = {"zh": "hanzi", "ja": "romaji", "en": "word"}


def fill_combo(combo: ComboBox, choices: list[tuple], keep_value: bool = False) -> None:
    """Fill a ComboBox from a (backend value, i18n key) table.

    The backend value becomes the item's userData; with keep_value the
    current selection survives a refill (used for live retranslation).
    """
    current = combo.currentData() if keep_value else None
    combo.blockSignals(True)
    combo.clear()
    for value, key in choices:
        combo.addItem(tr(key), userData=value)
    if current is not None:
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)
    combo.blockSignals(False)
