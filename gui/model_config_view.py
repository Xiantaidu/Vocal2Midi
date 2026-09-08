"""Model configuration interface — model paths only, moved out of global settings."""

import pathlib

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog

from qfluentwidgets import (
    ScrollArea,
    PushButton,
    CardWidget,
    BodyLabel,
    LineEdit,
    FluentIcon,
    SubtitleLabel,
)


class ModelConfigInterface(ScrollArea):
    def __init__(self, settings, project_root, parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.project_root = pathlib.Path(project_root)

        self.default_values = {
            "game_model": "experiments/GAME-1.0.3-medium-onnx",
            "hfa_model": "experiments/1218_hfa_model_new_dict",
            "asr_model": "experiments/Qwen3-ASR-1.7B-dml",
            "phoneme_asr_model": "experiments/romajiASR",
            "pinyin_asr_model": "experiments/pinyinASR",
            "rmvpe_model": "experiments/RMVPE/rmvpe.onnx",
        }

        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName("view")
        self.setObjectName("modelConfigInterface")

        title_layout = QHBoxLayout()
        title = SubtitleLabel("模型配置", self)
        title_layout.addWidget(title)
        title_layout.addStretch(1)
        btn_reset = PushButton("恢复默认", self, FluentIcon.SYNC)
        btn_reset.clicked.connect(self.reset_to_default)
        title_layout.addWidget(btn_reset)
        self.vBoxLayout.addLayout(title_layout)

        card = CardWidget(self)
        layout = QVBoxLayout(card)

        self._add_model_row(layout, "GAME 模型路径:", "game_model", browse_dir=True)
        self._add_model_row(layout, "HubertFA模型路径:", "hfa_model", browse_dir=True)
        self._add_model_row(layout, "Qwen3-ASR模型路径:", "asr_model", browse_dir=True)
        self._add_model_row(layout, "音素ASR模型路径:", "phoneme_asr_model", browse_dir=True)
        self._add_model_row(layout, "拼音ASR模型路径:", "pinyin_asr_model", browse_dir=True)
        self._add_model_row(layout, "RMVPE模型文件:", "rmvpe_model", browse_dir=False)

        self.vBoxLayout.addWidget(card)
        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

    # ── widget helpers ──────────────────────────────────────────────

    def _add_model_row(self, parent_layout, label_text: str, settings_key: str, browse_dir: bool):
        row = QHBoxLayout()
        row.addWidget(BodyLabel(label_text, self))
        edit = LineEdit(self)
        edit.setText(self._normalize_model_path(settings_key, self.default_values[settings_key]))
        edit.textChanged.connect(lambda t, k=settings_key: self.settings.setValue(k, t))
        setattr(self, f"{settings_key}_edit", edit)
        row.addWidget(edit, 1)
        btn = PushButton("浏览", self, FluentIcon.FOLDER)
        if browse_dir:
            btn.clicked.connect(lambda checked, e=edit: self._browse_dir(e))
        else:
            btn.clicked.connect(lambda checked, e=edit: self._browse_file(e))
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
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if dir_path:
            line_edit.setText(self._to_project_relative(dir_path))

    def _browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", line_edit.text(),
            "Model Files (*.pt *.pth *.bin *.onnx);;All Files (*)",
        )
        if file_path:
            line_edit.setText(self._to_project_relative(file_path))

    def reset_to_default(self):
        for key, default in self.default_values.items():
            edit = getattr(self, f"{key}_edit", None)
            if edit is not None:
                edit.setText(default)