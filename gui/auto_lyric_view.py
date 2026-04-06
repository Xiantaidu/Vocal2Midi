import os
import pathlib

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog

from qfluentwidgets import (
    ScrollArea,
    PushButton,
    PrimaryPushButton,
    CardWidget,
    IconWidget,
    BodyLabel,
    LineEdit,
    ComboBox,
    DoubleSpinBox,
    TextEdit,
    ListWidget,
    FluentIcon,
    SubtitleLabel,
)

from gui.fluent_utils import parse_quantization
from gui.fluent_worker import WorkerThread, HYBRID_AVAILABLE


class AutoLyricInterface(ScrollArea):
    def __init__(self, global_settings, parent=None):
        super().__init__(parent=parent)
        self.global_settings = global_settings
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName('view')
        self.setObjectName('autoLyricInterface')

        title = SubtitleLabel("自动提取与歌词灌注", self)
        self.vBoxLayout.addWidget(title)

        audio_card = CardWidget(self)
        audio_layout = QVBoxLayout(audio_card)
        header_layout = QHBoxLayout()

        music_icon = IconWidget(FluentIcon.MUSIC, self)
        music_icon.setFixedSize(16, 16)
        header_layout.addWidget(music_icon)

        title_label = BodyLabel("上传音频文件", self)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        btn_add = PushButton("选择文件", self, FluentIcon.FOLDER)
        btn_add.clicked.connect(self.add_audio_files)
        btn_clear = PushButton("清空选择", self, FluentIcon.DELETE)
        btn_clear.clicked.connect(self.clear_audio_files)
        header_layout.addWidget(btn_add)
        header_layout.addWidget(btn_clear)
        audio_layout.addLayout(header_layout)

        self.audio_list = ListWidget(self)
        self.audio_list.setMaximumHeight(40)
        audio_layout.addWidget(self.audio_list)
        self.vBoxLayout.addWidget(audio_card)

        self.lyric_card = CardWidget(self)
        lyric_layout = QVBoxLayout(self.lyric_card)
        self.lyric_title = BodyLabel("参考歌词 (可选)", self)
        self.lyric_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lyric_layout.addWidget(self.lyric_title)

        self.lyrics_edit = TextEdit(self)
        self.lyrics_edit.setPlaceholderText("如果有确切的歌词，请在此输入（纯文本）以提高对齐准确率...")
        self.lyrics_edit.setMaximumHeight(80)
        lyric_layout.addWidget(self.lyrics_edit)
        self.vBoxLayout.addWidget(self.lyric_card)

        combo_card = CardWidget(self)
        combo_layout = QHBoxLayout(combo_card)
        slice_icon = IconWidget(FluentIcon.CUT, self)
        slice_icon.setFixedSize(16, 16)
        combo_layout.addWidget(slice_icon)
        combo_layout.addWidget(BodyLabel("音频切片方法", self))
        self.slicing_combo = ComboBox(self)
        self.slicing_combo.addItems(["默认切片", "启发式切片", "网格搜索切片"])
        combo_layout.addWidget(self.slicing_combo)

        lang_icon = IconWidget(FluentIcon.LANGUAGE, self)
        lang_icon.setFixedSize(16, 16)
        combo_layout.addSpacing(28)
        combo_layout.addWidget(lang_icon)
        combo_layout.addWidget(BodyLabel("目标语言", self))
        self.lang_combo = ComboBox(self)
        self.lang_combo.addItems(["zh", "ja"])
        combo_layout.addWidget(self.lang_combo)

        combo_layout.addSpacing(28)
        dev_icon = IconWidget(FluentIcon.SETTING, self)
        dev_icon.setFixedSize(16, 16)
        combo_layout.addWidget(dev_icon)
        combo_layout.addWidget(BodyLabel("计算设备", self))
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(["cuda", "cpu"])
        combo_layout.addWidget(self.device_combo)
        combo_layout.addStretch(1)
        self.vBoxLayout.addWidget(combo_card)

        output_card = CardWidget(self)
        output_layout = QVBoxLayout(output_card)
        output_title = BodyLabel("输出设置", self)
        output_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        output_layout.addWidget(output_title)

        opts_layout = QHBoxLayout()
        opts_layout.addWidget(BodyLabel("曲速 (Tempo BPM):", self))
        self.tempo_spin = DoubleSpinBox(self)
        self.tempo_spin.setRange(10, 300)
        self.tempo_spin.setValue(120)
        opts_layout.addWidget(self.tempo_spin)
        opts_layout.addSpacing(20)
        opts_layout.addWidget(BodyLabel("MIDI 量化精度:", self))
        self.quantize_combo = ComboBox(self)
        self.quantize_combo.addItems(["不量化", "1/4 音符 (1拍)", "1/8 音符 (1/2拍)", "1/16 音符 (1/4拍)", "1/32 音符 (1/8拍)", "1/64 音符 (1/16拍)"])
        opts_layout.addWidget(self.quantize_combo)
        opts_layout.addStretch(1)
        output_layout.addLayout(opts_layout)

        save_layout = QHBoxLayout()
        save_layout.addWidget(BodyLabel("保存目录:", self))
        self.save_dir_edit = LineEdit(self)
        self.save_dir_edit.setText(self.global_settings.settings.value("save_dir", str(pathlib.Path.home() / "Desktop")))
        self.save_dir_edit.textChanged.connect(lambda t: self.global_settings.settings.setValue("save_dir", t))
        save_layout.addWidget(self.save_dir_edit, 1)
        btn_browse_save = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_browse_save.clicked.connect(lambda: self.browse_dir(self.save_dir_edit))
        save_layout.addWidget(btn_browse_save)
        output_layout.addLayout(save_layout)
        self.vBoxLayout.addWidget(output_card)

        action_layout = QHBoxLayout()
        self.btn_run = PrimaryPushButton(FluentIcon.PLAY, "开始全自动提取", self)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_stop = PushButton(FluentIcon.PAUSE, "强制停止", self)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_stop)
        self.vBoxLayout.addLayout(action_layout)

        self.log_edit = TextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        self.vBoxLayout.addWidget(self.log_edit)

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.worker = None
        self.update_lyrics_visibility()

    def update_lyrics_visibility(self):
        enabled = self.global_settings.cb_match_lyrics.isChecked()
        self.lyric_card.setVisible(enabled)
        self.lyric_title.setVisible(enabled)
        self.lyrics_edit.setVisible(enabled)

    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def add_audio_files(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "Audio Files (*.wav *.flac *.mp3 *.ogg)")
        if file:
            self.audio_list.clear()
            self.audio_list.addItem(file)

    def clear_audio_files(self):
        self.audio_list.clear()

    def log_msg(self, msg):
        self.log_edit.append(msg)
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def run_pipeline(self):
        if not HYBRID_AVAILABLE:
            self.log_msg("错误: 混合管线未能正确加载，请检查环境。")
            return

        audio_files = [self.audio_list.item(i).text() for i in range(self.audio_list.count())]
        if not audio_files:
            self.log_msg("错误: 请至少上传一个音频文件。")
            return

        output_formats = ["mid"]
        if self.global_settings.cb_txt.isChecked():
            output_formats.append("txt")
        if self.global_settings.cb_csv.isChecked():
            output_formats.append("csv")
        if self.global_settings.cb_chunks.isChecked():
            output_formats.append("chunks")

        save_dir = self.save_dir_edit.text()
        if not save_dir or not os.path.isdir(save_dir):
            self.log_msg("错误: 请选择一个有效的保存目录。")
            return

        kwargs = {
            'audio_files': audio_files,
            'save_dir': save_dir,
            'game_model_path_str': self.global_settings.game_model_edit.text(),
            'device': self.device_combo.currentText(),
            'hfa_model_path_str': self.global_settings.hfa_model_edit.text(),
            'asr_model_path_str': self.global_settings.asr_model_edit.text(),
            'language': self.lang_combo.currentText(),
            'original_lyrics': self.lyrics_edit.toPlainText().strip() if self.global_settings.cb_match_lyrics.isChecked() else "",
            'output_formats': output_formats,
            'slicing_method': self.slicing_combo.currentText(),
            'tempo': self.tempo_spin.value(),
            'quantization_step': parse_quantization(self.quantize_combo.currentText()),
            'pitch_format': self.global_settings.pitch_combo.currentText(),
            'round_pitch': self.global_settings.cb_round.isChecked(),
            'seg_threshold': self.global_settings.seg_thresh_spin.value(),
            'seg_radius': self.global_settings.seg_rad_spin.value(),
            't0': self.global_settings.t0_spin.value(),
            'nsteps': self.global_settings.nsteps_spin.value(),
            'est_threshold': self.global_settings.est_thresh_spin.value(),
            'batch_size': self.global_settings.batch_spin.value(),
            'asr_batch_size': self.global_settings.asr_batch_spin.value(),
        }

        self.log_edit.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker = WorkerThread(kwargs)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def stop_pipeline(self):
        if self.worker:
            self.worker.stop()
            self.log_msg("已请求强制停止，正在尽快中断当前流程...")
            self.btn_stop.setEnabled(False)
            if self.worker.isRunning():
                self.log_msg("提示: 若正在执行底层模型调用，需等待当前批次返回后停止。")

    def on_finished(self, msg):
        self.log_msg(msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        try:
            save_dir_path = self.save_dir_edit.text()
            if os.path.exists(save_dir_path):
                os.startfile(save_dir_path)
        except Exception as e:
            self.log_msg(f"无法自动打开文件夹: {e}")

    def on_error(self, msg):
        self.log_msg(msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
