"""Model configuration interface — ASR engine choices and model paths."""

import pathlib

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog

from qfluentwidgets import (
    ScrollArea,
    PushButton,
    CardWidget,
    BodyLabel,
    LineEdit,
    ComboBox,
    FluentIcon,
    SubtitleLabel,
)
from gui.i18n import tr

MODEL_ROWS = [
    ("game_model", "game_path"),
    ("hfa_model", "hfa_path"),
    ("tifa_model", "tifa_path"),
    ("kashi_g2p_model", "kashi_g2p_path"),
    ("asr_model", "asr_path"),
    ("phoneme_asr_model", "phoneme_path"),
    ("pinyin_asr_model", "pinyin_path"),
    ("rmvpe_model", "rmvpe_path"),
]

# settings key, label key, (value, tr key) choices, default value
ENGINE_ROWS = [
    ("chinese_asr_engine", "zh_asr_choice", [("pinyin", "asr_pinyin"), ("qwen", "asr_qwen")], "qwen"),
    ("japanese_asr_engine", "ja_asr_choice", [("romaji", "asr_romaji"), ("qwen", "asr_qwen")], "romaji"),
    ("alignment_engine", "aligner_choice", [("tifa", "aligner_tifa"), ("hfa", "aligner_hfa")], "tifa"),
    ("japanese_g2p_engine", "ja_g2p_choice", [("kashi-g2p-onnx", "g2p_kashi"), ("pyopenjtalk", "g2p_pyopenjtalk")], "kashi-g2p-onnx"),
]

# Backward-compatible alias for the ASR subset of ENGINE_ROWS.
ASR_ENGINE_ROWS = ENGINE_ROWS[:2]


class ModelConfigInterface(ScrollArea):
    # Emitted when the user switches the Chinese ASR engine; the auto lyric
    # page listens to re-evaluate its pinyin output lock.
    chinese_asr_engine_changed = Signal(str)
    # Emitted when the user switches the alignment engine; the auto lyric
    # page listens to enable/disable Cantonese ('yue') which HFA does not support.
    alignment_engine_changed = Signal(str)

    def __init__(self, settings, project_root, parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.project_root = pathlib.Path(project_root)
        self._tr_bindings: list = []

        self.default_values = {
            "game_model": "models/GAME-1.0.3-medium-onnx",
            "hfa_model": "models/1218_hfa_model_new_dict",
            "tifa_model": "models/tifa-1.0-onnx",
            "kashi_g2p_model": "models/kashi-g2p-onnx",
            "asr_model": "models/Qwen3-ASR-1.7B-dml",
            "phoneme_asr_model": "models/romajiASR",
            "pinyin_asr_model": "models/pinyinASR",
            "rmvpe_model": "models/RMVPE",
        }

        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName("view")
        self.setObjectName("modelConfigInterface")

        title_layout = QHBoxLayout()
        title = SubtitleLabel(self)
        self._bind_tr(lambda: title.setText(tr("model_title")))
        title_layout.addWidget(title)
        title_layout.addStretch(1)
        btn_reset = PushButton(tr("reset_defaults"), self, FluentIcon.SYNC)
        self._bind_tr(lambda: btn_reset.setText(tr("reset_defaults")))
        btn_reset.clicked.connect(self.reset_to_default)
        title_layout.addWidget(btn_reset)
        self.vBoxLayout.addLayout(title_layout)

        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setSpacing(14)

        engine_grid = QGridLayout()
        engine_grid.setHorizontalSpacing(24)
        engine_grid.setVerticalSpacing(12)
        engine_grid.setContentsMargins(0, 0, 0, 8)

        for i, (settings_key, label_key, choices, default) in enumerate(ENGINE_ROWS):
            r = i // 2
            c = i % 2
            cell_layout = QHBoxLayout()
            label = BodyLabel(self)
            self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
            combo = ComboBox(self)
            for value, tr_key in choices:
                combo.addItem(tr(tr_key), userData=value)
            valid = [value for value, _ in choices]
            saved = str(self.settings.value(settings_key, default) or default)
            index = combo.findData(saved if saved in valid else default)
            combo.setCurrentIndex(max(0, index))
            self.settings.setValue(settings_key, combo.currentData())
            combo.currentIndexChanged.connect(
                lambda _i, k=settings_key, c=combo: self._on_engine_choice_changed(k, c.currentData())
            )
            setattr(self, f"{settings_key}_combo", combo)
            cell_layout.addWidget(label)
            cell_layout.addWidget(combo, 1)
            engine_grid.addLayout(cell_layout, r, c)

        layout.addLayout(engine_grid)

        for settings_key, label_key in MODEL_ROWS:
            self._add_model_row(layout, label_key, settings_key)

        self.vBoxLayout.addWidget(card)
        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

    # ── widget helpers ──────────────────────────────────────────────

    def _bind_tr(self, fn):
        self._tr_bindings.append(fn)
        fn()

    def retranslate_ui(self):
        for fn in self._tr_bindings:
            fn()

    # ── read-only accessors for other pages ─────────────────────────
    # Other pages read model paths and engine choices through these instead
    # of reaching into the row widgets directly.
    def model_path(self, settings_key: str) -> str:
        edit = getattr(self, f"{settings_key}_edit", None)
        return edit.text() if edit is not None else ""

    def chinese_asr_engine(self) -> str:
        return self._engine_choice("chinese_asr_engine", "qwen")

    def japanese_asr_engine(self) -> str:
        return self._engine_choice("japanese_asr_engine", "romaji")

    def alignment_engine(self) -> str:
        return self._engine_choice("alignment_engine", "tifa")

    def japanese_g2p_engine(self) -> str:
        return self._engine_choice("japanese_g2p_engine", "kashi-g2p-onnx")

    def _engine_choice(self, settings_key: str, fallback: str) -> str:
        combo = getattr(self, f"{settings_key}_combo", None)
        if combo is None:
            return fallback
        return combo.currentData() or fallback

    def _on_engine_choice_changed(self, settings_key: str, value):
        self.settings.setValue(settings_key, value)
        if settings_key == "chinese_asr_engine":
            self.chinese_asr_engine_changed.emit(str(value))
        elif settings_key == "alignment_engine":
            self.alignment_engine_changed.emit(str(value))

    def _add_asr_choice_row(self, parent_layout, label_key: str, settings_key: str, choices: list, default: str):
        row = QHBoxLayout()
        label = BodyLabel(self)
        self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
        row.addWidget(label)
        combo = ComboBox(self)
        for value, tr_key in choices:
            combo.addItem(tr(tr_key), userData=value)
        valid = [value for value, _ in choices]
        saved = str(self.settings.value(settings_key, default) or default)
        index = combo.findData(saved if saved in valid else default)
        combo.setCurrentIndex(max(0, index))
        self.settings.setValue(settings_key, combo.currentData())
        combo.currentIndexChanged.connect(
            lambda _i, k=settings_key, c=combo: self._on_engine_choice_changed(k, c.currentData())
        )
        setattr(self, f"{settings_key}_combo", combo)
        row.addWidget(combo, 1)
        parent_layout.addLayout(row)

    def _add_model_row(self, parent_layout, label_key: str, settings_key: str):
        row = QHBoxLayout()
        label = BodyLabel(self)
        self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
        row.addWidget(label)
        edit = LineEdit(self)
        edit.setText(self._normalize_model_path(settings_key, self.default_values[settings_key]))
        edit.textChanged.connect(lambda t, k=settings_key: self.settings.setValue(k, t))
        setattr(self, f"{settings_key}_edit", edit)
        row.addWidget(edit, 1)
        btn = PushButton(tr("browse"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda b=btn: b.setText(tr("browse")))
        btn.clicked.connect(lambda checked, e=edit: self._browse_dir(e))
        row.addWidget(btn)
        parent_layout.addLayout(row)

    # ── path helpers ─────────────────────────────────────────────────

    def _to_project_relative(self, path_str: str) -> str:
        p = pathlib.Path(path_str).resolve()
        try:
            return str(p.relative_to(self.project_root)).replace("\\", "/")
        except ValueError:
            return str(p)

    def _normalize_model_path(self, key: str, fallback_relative: str) -> str:
        if key == "kashi_g2p_model" and not self.settings.contains("kashi_g2p_model"):
            for legacy_key in ("ja_g2p_model", "ja_g2p_onnx_model", "ja_g2p"):
                if self.settings.contains(legacy_key):
                    val = str(self.settings.value(legacy_key))
                    if "ja_g2p" in val and (self.project_root / "models" / "kashi-g2p-onnx").exists():
                        val = fallback_relative
                    self.settings.setValue("kashi_g2p_model", val)
                    break
        raw_value = self.settings.value(key, fallback_relative)
        value = str(raw_value) if raw_value is not None else fallback_relative
        if key == "kashi_g2p_model" and ("ja_g2p" in value) and (self.project_root / "models" / "kashi-g2p-onnx").exists():
            value = fallback_relative
            self.settings.setValue(key, value)
        p = pathlib.Path(value)
        if p.is_absolute():
            try:
                value = str(p.resolve().relative_to(self.project_root)).replace("\\", "/")
            except ValueError:
                # Path outside the project (e.g. models on another drive):
                # keep the user's absolute path; never silently reset it.
                value = str(p.resolve())
        self.settings.setValue(key, value)
        return value

    def _browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, tr("choose_folder_dialog"), line_edit.text())
        if dir_path:
            line_edit.setText(self._to_project_relative(dir_path))

    def reset_to_default(self):
        for settings_key, _label_key, choices, default in ENGINE_ROWS:
            combo = getattr(self, f"{settings_key}_combo", None)
            if combo is not None:
                index = combo.findData(default)
                if index >= 0 and index != combo.currentIndex():
                    combo.setCurrentIndex(index)  # persists via currentIndexChanged
        for key, default in self.default_values.items():
            edit = getattr(self, f"{key}_edit", None)
            if edit is not None:
                edit.setText(default)
