"""Model configuration interface — model paths only, moved out of global settings."""

import pathlib

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog

from qfluentwidgets import (
    ScrollArea,
    PushButton,
    CardWidget,
    BodyLabel,
    LineEdit,
    FluentIcon,
    SubtitleLabel,
)
from gui.i18n import tr

MODEL_ROWS = [
    ("game_model", "game_path"),
    ("hfa_model", "hfa_path"),
    ("asr_model", "asr_path"),
    ("phoneme_asr_model", "phoneme_path"),
    ("pinyin_asr_model", "pinyin_path"),
    ("rmvpe_model", "rmvpe_path"),
]


class ModelConfigInterface(ScrollArea):
    def __init__(self, settings, project_root, parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.project_root = pathlib.Path(project_root)
        self._tr_bindings: list = []

        self.default_values = {
            "game_model": "models/GAME-1.0.3-medium-onnx",
            "hfa_model": "models/1218_hfa_model_new_dict",
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
        raw_value = self.settings.value(key, fallback_relative)
        value = str(raw_value) if raw_value is not None else fallback_relative
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
        for key, default in self.default_values.items():
            edit = getattr(self, f"{key}_edit", None)
            if edit is not None:
                edit.setText(default)
