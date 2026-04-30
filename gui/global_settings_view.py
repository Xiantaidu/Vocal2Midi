import pathlib

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QSettings

from qfluentwidgets import (
    ScrollArea,
    PushButton,
    CardWidget,
    BodyLabel,
    LineEdit,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    SwitchButton,
    FluentIcon,
    SubtitleLabel,
)


class GlobalSettingsInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.settings = QSettings("GAME_Extractor", "Vocal2Midi")
        self.default_values = {
            "game_model": "experiments/GAME-1.0-medium",
            "hfa_model": "experiments/1218_hfa_model_new_dict",
            "asr_model": "experiments/Qwen3-ASR-1.7B",
            "phoneme_asr_model": "inference/phonemeASR/checkpoints/exp1/best",
            "rmvpe_model": "experiments/RMVPE/rmvpe.pt",
            "seg_thresh": 0.2,
            "seg_rad": 0.02,
            "est_thresh": 0.2,
            "t0": 0.0,
            "nsteps": 8,
            "batch_size": 2,
            "asr_batch": 2,
            "debug_txt": False,
            "debug_csv": False,
            "debug_chunks": False,
            "output_lyrics": True,
            "enable_lyrics_match": False,
            "pitch_format": "name",
            "round_pitch": True,
        }

        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName('view')
        self.setObjectName('globalSettingsInterface')

        title_layout = QHBoxLayout()
        title = SubtitleLabel("全局设置", self)
        title_layout.addWidget(title)
        title_layout.addStretch(1)
        btn_reset = PushButton("恢复默认", self, FluentIcon.SYNC)
        btn_reset.clicked.connect(self.reset_to_default)
        title_layout.addWidget(btn_reset)
        self.vBoxLayout.addLayout(title_layout)

        model_card = CardWidget(self)
        model_layout = QVBoxLayout(model_card)
        model_title = BodyLabel("模型配置", self)
        model_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        model_layout.addWidget(model_title)

        game_layout = QHBoxLayout()
        game_layout.addWidget(BodyLabel("GAME 模型路径:", self))
        self.game_model_edit = LineEdit(self)
        self.game_model_edit.setText(self._normalize_model_path("game_model", self.default_values["game_model"]))
        self.game_model_edit.textChanged.connect(lambda t: self.settings.setValue("game_model", t))
        game_layout.addWidget(self.game_model_edit, 1)
        btn_browse_game = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_browse_game.clicked.connect(lambda: self.browse_dir(self.game_model_edit))
        game_layout.addWidget(btn_browse_game)
        model_layout.addLayout(game_layout)

        hfa_layout = QHBoxLayout()
        hfa_layout.addWidget(BodyLabel("HubertFA模型路径:", self))
        self.hfa_model_edit = LineEdit(self)
        self.hfa_model_edit.setText(self._normalize_model_path("hfa_model", self.default_values["hfa_model"]))
        self.hfa_model_edit.textChanged.connect(lambda t: self.settings.setValue("hfa_model", t))
        hfa_layout.addWidget(self.hfa_model_edit, 1)
        btn_hfa = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_hfa.clicked.connect(lambda: self.browse_dir(self.hfa_model_edit))
        hfa_layout.addWidget(btn_hfa)
        model_layout.addLayout(hfa_layout)

        asr_layout = QHBoxLayout()
        asr_layout.addWidget(BodyLabel("Qwen3-ASR模型路径:", self))
        self.asr_model_edit = LineEdit(self)
        self.asr_model_edit.setText(self._normalize_model_path("asr_model", self.default_values["asr_model"]))
        self.asr_model_edit.textChanged.connect(lambda t: self.settings.setValue("asr_model", t))
        asr_layout.addWidget(self.asr_model_edit, 1)
        btn_asr = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_asr.clicked.connect(lambda: self.browse_dir(self.asr_model_edit))
        asr_layout.addWidget(btn_asr)
        model_layout.addLayout(asr_layout)

        phoneme_asr_layout = QHBoxLayout()
        phoneme_asr_layout.addWidget(BodyLabel("音素ASR模型路径:", self))
        self.phoneme_asr_model_edit = LineEdit(self)
        self.phoneme_asr_model_edit.setText(
            self._normalize_model_path("phoneme_asr_model", self.default_values["phoneme_asr_model"])
        )
        self.phoneme_asr_model_edit.textChanged.connect(lambda t: self.settings.setValue("phoneme_asr_model", t))
        phoneme_asr_layout.addWidget(self.phoneme_asr_model_edit, 1)
        btn_phoneme_asr = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_phoneme_asr.clicked.connect(lambda: self.browse_dir(self.phoneme_asr_model_edit))
        phoneme_asr_layout.addWidget(btn_phoneme_asr)
        model_layout.addLayout(phoneme_asr_layout)

        rmvpe_layout = QHBoxLayout()
        rmvpe_layout.addWidget(BodyLabel("RMVPE模型文件路径:", self))
        self.rmvpe_model_edit = LineEdit(self)
        self.rmvpe_model_edit.setText(self._normalize_model_path("rmvpe_model", self.default_values["rmvpe_model"]))
        self.rmvpe_model_edit.textChanged.connect(lambda t: self.settings.setValue("rmvpe_model", t))
        rmvpe_layout.addWidget(self.rmvpe_model_edit, 1)
        btn_rmvpe = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_rmvpe.clicked.connect(lambda: self.browse_file(self.rmvpe_model_edit))
        rmvpe_layout.addWidget(btn_rmvpe)
        model_layout.addLayout(rmvpe_layout)
        self.vBoxLayout.addWidget(model_card)

        adv_card = CardWidget(self)
        adv_layout = QVBoxLayout(adv_card)
        adv_title = BodyLabel("高级处理参数", self)
        adv_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        adv_layout.addWidget(adv_title)

        from PyQt5.QtWidgets import QGridLayout

        adv_grid = QGridLayout()
        adv_grid.setVerticalSpacing(15)
        adv_grid.setHorizontalSpacing(20)

        lbl1 = BodyLabel("边界解码阈值:", self)
        self.seg_thresh_spin = DoubleSpinBox(self)
        self.seg_thresh_spin.setRange(0.01, 0.99)
        self.seg_thresh_spin.setSingleStep(0.01)
        self.seg_thresh_spin.setValue(float(self.settings.value("seg_thresh", self.default_values["seg_thresh"])))
        self.seg_thresh_spin.valueChanged.connect(lambda v: self.settings.setValue("seg_thresh", v))
        adv_grid.addWidget(lbl1, 0, 0)
        adv_grid.addWidget(self.seg_thresh_spin, 0, 1)

        lbl2 = BodyLabel("边界解码半径/秒:", self)
        self.seg_rad_spin = DoubleSpinBox(self)
        self.seg_rad_spin.setRange(0.01, 0.1)
        self.seg_rad_spin.setSingleStep(0.005)
        self.seg_rad_spin.setValue(float(self.settings.value("seg_rad", self.default_values["seg_rad"])))
        self.seg_rad_spin.valueChanged.connect(lambda v: self.settings.setValue("seg_rad", v))
        adv_grid.addWidget(lbl2, 0, 2)
        adv_grid.addWidget(self.seg_rad_spin, 0, 3)

        lbl3 = BodyLabel("音符存在阈值:", self)
        self.est_thresh_spin = DoubleSpinBox(self)
        self.est_thresh_spin.setRange(0.01, 0.99)
        self.est_thresh_spin.setSingleStep(0.01)
        self.est_thresh_spin.setValue(float(self.settings.value("est_thresh", self.default_values["est_thresh"])))
        self.est_thresh_spin.valueChanged.connect(lambda v: self.settings.setValue("est_thresh", v))
        adv_grid.addWidget(lbl3, 0, 4)
        adv_grid.addWidget(self.est_thresh_spin, 0, 5)

        lbl4 = BodyLabel("D3PM 起始 T 值:", self)
        self.t0_spin = DoubleSpinBox(self)
        self.t0_spin.setRange(0.0, 0.99)
        self.t0_spin.setSingleStep(0.01)
        self.t0_spin.setValue(float(self.settings.value("t0", self.default_values["t0"])))
        self.t0_spin.valueChanged.connect(lambda v: self.settings.setValue("t0", v))
        adv_grid.addWidget(lbl4, 1, 0)
        adv_grid.addWidget(self.t0_spin, 1, 1)

        lbl5 = BodyLabel("D3PM 采样步数:", self)
        self.nsteps_spin = SpinBox(self)
        self.nsteps_spin.setRange(1, 20)
        self.nsteps_spin.setValue(int(self.settings.value("nsteps", self.default_values["nsteps"])))
        self.nsteps_spin.valueChanged.connect(lambda v: self.settings.setValue("nsteps", v))
        adv_grid.addWidget(lbl5, 1, 2)
        adv_grid.addWidget(self.nsteps_spin, 1, 3)

        lbl6 = BodyLabel("GAME Batch:", self)
        self.batch_spin = SpinBox(self)
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(int(self.settings.value("batch_size", self.default_values["batch_size"])))
        self.batch_spin.valueChanged.connect(lambda v: self.settings.setValue("batch_size", v))
        adv_grid.addWidget(lbl6, 1, 4)
        adv_grid.addWidget(self.batch_spin, 1, 5)

        lbl7 = BodyLabel("ASR Batch:", self)
        self.asr_batch_spin = SpinBox(self)
        self.asr_batch_spin.setRange(1, 32)
        self.asr_batch_spin.setValue(int(self.settings.value("asr_batch", self.default_values["asr_batch"])))
        self.asr_batch_spin.valueChanged.connect(lambda v: self.settings.setValue("asr_batch", v))
        adv_grid.addWidget(lbl7, 2, 0)
        adv_grid.addWidget(self.asr_batch_spin, 2, 1)

        lbl8 = BodyLabel("启用歌词匹配:", self)
        self.cb_match_lyrics = SwitchButton("On", self, self)
        self.cb_match_lyrics.setOffText("Off")
        self.cb_match_lyrics.setFixedWidth(75)
        self.cb_match_lyrics.setChecked(self.settings.value("enable_lyrics_match", self.default_values["enable_lyrics_match"], type=bool))
        self.cb_match_lyrics.checkedChanged.connect(lambda v: self.settings.setValue("enable_lyrics_match", v))
        adv_grid.addWidget(lbl8, 2, 2)
        adv_grid.addWidget(self.cb_match_lyrics, 2, 3, alignment=Qt.AlignLeft)

        adv_grid.setColumnStretch(6, 1)
        adv_layout.addLayout(adv_grid)
        self.vBoxLayout.addWidget(adv_card)

        debug_card = CardWidget(self)
        debug_layout = QVBoxLayout(debug_card)
        debug_title = BodyLabel("Debug", self)
        debug_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        debug_layout.addWidget(debug_title)

        debug_grid = QHBoxLayout()
        debug_grid.addWidget(BodyLabel("导出 Text (.txt):", self))
        self.cb_txt = SwitchButton("On", self, self)
        self.cb_txt.setOffText("Off")
        self.cb_txt.setChecked(self.settings.value("debug_txt", self.default_values["debug_txt"], type=bool))
        self.cb_txt.checkedChanged.connect(lambda v: self.settings.setValue("debug_txt", v))
        debug_grid.addWidget(self.cb_txt)
        debug_grid.addSpacing(20)

        debug_grid.addWidget(BodyLabel("导出 CSV (.csv):", self))
        self.cb_csv = SwitchButton("On", self, self)
        self.cb_csv.setOffText("Off")
        self.cb_csv.setChecked(self.settings.value("debug_csv", self.default_values["debug_csv"], type=bool))
        self.cb_csv.checkedChanged.connect(lambda v: self.settings.setValue("debug_csv", v))
        debug_grid.addWidget(self.cb_csv)
        debug_grid.addSpacing(20)

        debug_grid.addWidget(BodyLabel("导出切片:", self))
        self.cb_chunks = SwitchButton("On", self, self)
        self.cb_chunks.setOffText("Off")
        self.cb_chunks.setChecked(self.settings.value("debug_chunks", self.default_values["debug_chunks"], type=bool))
        self.cb_chunks.checkedChanged.connect(lambda v: self.settings.setValue("debug_chunks", v))
        debug_grid.addWidget(self.cb_chunks)
        debug_grid.addSpacing(20)

        debug_grid.addWidget(BodyLabel("音高格式:", self))
        self.pitch_combo = ComboBox(self)
        self.pitch_combo.addItems(["name", "number"])
        self.pitch_combo.setCurrentText(self.settings.value("pitch_format", self.default_values["pitch_format"]))
        self.pitch_combo.currentTextChanged.connect(lambda t: self.settings.setValue("pitch_format", t))
        debug_grid.addWidget(self.pitch_combo)
        debug_grid.addSpacing(20)

        debug_grid.addWidget(BodyLabel("音高取整:", self))
        self.cb_round = SwitchButton("On", self, self)
        self.cb_round.setOffText("Off")
        self.cb_round.setChecked(self.settings.value("round_pitch", self.default_values["round_pitch"], type=bool))
        self.cb_round.checkedChanged.connect(lambda v: self.settings.setValue("round_pitch", v))
        debug_grid.addWidget(self.cb_round)
        debug_grid.addStretch(1)

        debug_layout.addLayout(debug_grid)
        self.vBoxLayout.addWidget(debug_card)

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

    def reset_to_default(self):
        self.game_model_edit.setText(self.default_values["game_model"])
        self.hfa_model_edit.setText(self.default_values["hfa_model"])
        self.asr_model_edit.setText(self.default_values["asr_model"])
        self.phoneme_asr_model_edit.setText(self.default_values["phoneme_asr_model"])
        self.rmvpe_model_edit.setText(self.default_values["rmvpe_model"])
        self.seg_thresh_spin.setValue(self.default_values["seg_thresh"])
        self.seg_rad_spin.setValue(self.default_values["seg_rad"])
        self.est_thresh_spin.setValue(self.default_values["est_thresh"])
        self.t0_spin.setValue(self.default_values["t0"])
        self.nsteps_spin.setValue(self.default_values["nsteps"])
        self.batch_spin.setValue(self.default_values["batch_size"])
        self.asr_batch_spin.setValue(self.default_values["asr_batch"])
        self.cb_txt.setChecked(self.default_values["debug_txt"])
        self.cb_csv.setChecked(self.default_values["debug_csv"])
        self.cb_chunks.setChecked(self.default_values["debug_chunks"])
        self.cb_match_lyrics.setChecked(self.default_values["enable_lyrics_match"])
        self.pitch_combo.setCurrentText(self.default_values["pitch_format"])
        self.cb_round.setChecked(self.default_values["round_pitch"])

    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if dir_path:
            line_edit.setText(self._to_project_relative(dir_path))

    def browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", line_edit.text(), "Model Files (*.pt *.pth *.bin *.onnx);;All Files (*)")
        if file_path:
            line_edit.setText(self._to_project_relative(file_path))

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
                value = fallback_relative

        self.settings.setValue(key, value)
        return value
