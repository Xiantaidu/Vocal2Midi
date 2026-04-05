import sys
import os
import pathlib
import tempfile
import traceback
import zipfile
import torch

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QSettings
from PyQt5.QtGui import QIcon

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, SubtitleLabel, setTheme, Theme,
    ScrollArea, VBoxLayout, PushButton, PrimaryPushButton,
    CardWidget, IconWidget, BodyLabel, LineEdit, ComboBox,
    SpinBox, DoubleSpinBox, SwitchButton, TextEdit, ListWidget, FluentIcon
)

# Import the hybrid pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from inference.auto_lyric_hybrid import auto_lyric_hybrid_pipeline
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Hybrid pipeline not available. Error: {e}")
    HYBRID_AVAILABLE = False

def _parse_quantization(quantize_option: str) -> int:
    if "1/4 音符" in quantize_option: return 480
    elif "1/8 音符" in quantize_option: return 240
    elif "1/16 音符" in quantize_option: return 120
    elif "1/32 音符" in quantize_option: return 60
    elif "1/64 音符" in quantize_option: return 30
    return 0

def _t0_nstep_to_ts(t0: float, nsteps: int) -> list:
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]

class StreamRedirector:
    def __init__(self, stream, signal):
        self.stream = stream
        self.signal = signal

    def write(self, text):
        if text.strip():
            self.signal.emit(text.strip())
        self.stream.write(text)

    def flush(self):
        self.stream.flush()

class WorkerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._is_running = True
        
    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(sys.stdout, self.log_signal)
        sys.stderr = StreamRedirector(sys.stderr, self.log_signal)
        
        try:
            audio_files = self.kwargs['audio_files']
            save_dir = pathlib.Path(self.kwargs['save_dir'])
            
            for audio_path in audio_files:
                if not self._is_running:
                    break
                original_path = pathlib.Path(audio_path)
                filename = original_path.name
                self.log_signal.emit(f"========== 正在处理: {filename} ==========")
                
                ts_list = _t0_nstep_to_ts(self.kwargs['t0'], int(self.kwargs['nsteps']))
                ts_tensor = torch.tensor(ts_list, device=self.kwargs['device'])
                
                auto_lyric_hybrid_pipeline(
                    audio_path=str(original_path),
                    output_filename=filename,
                    game_model_dir=self.kwargs['game_model_path_str'],
                    device=self.kwargs['device'],
                    hfa_model_dir=self.kwargs['hfa_model_path_str'],
                    asr_model_path=self.kwargs['asr_model_path_str'],
                    ts=ts_tensor,
                    language=self.kwargs['language'],
                    original_lyrics=self.kwargs['original_lyrics'],
                    output_dir=save_dir,
                    output_formats=self.kwargs['output_formats'],
                    slicing_method=self.kwargs['slicing_method'],
                    tempo=self.kwargs['tempo'],
                    quantization_step=self.kwargs['quantization_step'],
                    pitch_format=self.kwargs['pitch_format'],
                    round_pitch=self.kwargs['round_pitch'],
                    seg_threshold=self.kwargs['seg_threshold'],
                    seg_radius=self.kwargs['seg_radius'],
                    est_threshold=self.kwargs['est_threshold'],
                    batch_size=self.kwargs['batch_size'],
                    asr_batch_size=self.kwargs['asr_batch_size'],
                    debug_mode=True
                )
            
            if self._is_running:
                self.finished_signal.emit(f"提取成功！文件已保存至: {save_dir}")
            else:
                self.error_signal.emit("任务已被取消。")
                
        except Exception as e:
            self.error_signal.emit(f"发生错误:\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        self._is_running = False

class GlobalSettingsInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.settings = QSettings("GAME_Extractor", "Vocal2Midi")
        self.default_values = {
            "game_model": r"E:\Vocal2Midi\experiments\GAME-1.0-medium",
            "hfa_model": r"E:\Vocal2Midi\experiments\1218_hfa_model_new_dict",
            "asr_model": r"C:\Users\Xiantaidu\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B",
            "seg_thresh": 0.2,
            "seg_rad": 0.02,
            "est_thresh": 0.2,
            "t0": 0.0,
            "nsteps": 8,
            "batch_size": 4,
            "asr_batch": 2,
            "debug_txt": False,
            "debug_csv": False,
            "debug_chunks": False,
            "pitch_format": "name",
            "round_pitch": True
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
        
        # Model Configuration
        model_card = CardWidget(self)
        model_layout = QVBoxLayout(model_card)
        model_title = BodyLabel("模型配置", self)
        model_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        model_layout.addWidget(model_title)
        
        game_layout = QHBoxLayout()
        game_layout.addWidget(BodyLabel("GAME 模型路径:", self))
        self.game_model_edit = LineEdit(self)
        self.game_model_edit.setText(self.settings.value("game_model", self.default_values["game_model"]))
        self.game_model_edit.textChanged.connect(lambda t: self.settings.setValue("game_model", t))
        game_layout.addWidget(self.game_model_edit, 1)
        btn_browse_game = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_browse_game.clicked.connect(lambda: self.browse_dir(self.game_model_edit))
        game_layout.addWidget(btn_browse_game)
        model_layout.addLayout(game_layout)
        
        hfa_layout = QHBoxLayout()
        hfa_layout.addWidget(BodyLabel("HubertFA模型路径:", self))
        self.hfa_model_edit = LineEdit(self)
        self.hfa_model_edit.setText(self.settings.value("hfa_model", self.default_values["hfa_model"]))
        self.hfa_model_edit.textChanged.connect(lambda t: self.settings.setValue("hfa_model", t))
        hfa_layout.addWidget(self.hfa_model_edit, 1)
        btn_hfa = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_hfa.clicked.connect(lambda: self.browse_dir(self.hfa_model_edit))
        hfa_layout.addWidget(btn_hfa)
        model_layout.addLayout(hfa_layout)
        
        asr_layout = QHBoxLayout()
        asr_layout.addWidget(BodyLabel("Qwen3-ASR模型路径:", self))
        self.asr_model_edit = LineEdit(self)
        self.asr_model_edit.setText(self.settings.value("asr_model", self.default_values["asr_model"]))
        self.asr_model_edit.textChanged.connect(lambda t: self.settings.setValue("asr_model", t))
        asr_layout.addWidget(self.asr_model_edit, 1)
        btn_asr = PushButton("浏览", self, FluentIcon.FOLDER)
        btn_asr.clicked.connect(lambda: self.browse_dir(self.asr_model_edit))
        asr_layout.addWidget(btn_asr)
        model_layout.addLayout(asr_layout)

        self.vBoxLayout.addWidget(model_card)
        
        # Advanced Processing
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
        
        adv_grid.setColumnStretch(6, 1)

        adv_layout.addLayout(adv_grid)
        self.vBoxLayout.addWidget(adv_card)

        # Debug Settings
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
        self.pitch_combo.setCurrentText(self.default_values["pitch_format"])
        self.cb_round.setChecked(self.default_values["round_pitch"])

    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

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
        
        # Audio Files
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
        
        # Lyrics and Slicing
        lyric_card = CardWidget(self)
        lyric_layout = QVBoxLayout(lyric_card)
        
        lyric_title = BodyLabel("参考歌词 (可选)", self)
        lyric_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lyric_layout.addWidget(lyric_title)
        
        self.lyrics_edit = TextEdit(self)
        self.lyrics_edit.setPlaceholderText("如果有确切的歌词，请在此输入（纯文本）以提高对齐准确率...")
        self.lyrics_edit.setMaximumHeight(80)
        lyric_layout.addWidget(self.lyrics_edit)
        
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(BodyLabel("音频切片方法:", self))
        self.slicing_combo = ComboBox(self)
        self.slicing_combo.addItems(["默认切片", "启发式切片", "网格搜索切片"])
        self.slicing_combo.setMinimumWidth(150)
        slice_layout.addWidget(self.slicing_combo)
        slice_layout.addStretch(1)
        lyric_layout.addLayout(slice_layout)
        self.vBoxLayout.addWidget(lyric_card)
        
        # Language & Device
        combo_card = CardWidget(self)
        combo_layout = QHBoxLayout(combo_card)
        
        lang_icon = IconWidget(FluentIcon.LANGUAGE, self)
        lang_icon.setFixedSize(16, 16)
        combo_layout.addWidget(lang_icon)
        combo_layout.addWidget(BodyLabel("目标语言", self))
        self.lang_combo = ComboBox(self)
        self.lang_combo.addItems(["zh", "ja"])
        combo_layout.addWidget(self.lang_combo)
        
        combo_layout.addSpacing(40)
        
        dev_icon = IconWidget(FluentIcon.SETTING, self)
        dev_icon.setFixedSize(16, 16)
        combo_layout.addWidget(dev_icon)
        combo_layout.addWidget(BodyLabel("计算设备", self))
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(["cuda", "cpu"])
        combo_layout.addWidget(self.device_combo)
        
        combo_layout.addStretch(1)
        self.vBoxLayout.addWidget(combo_card)
        
        # Output Options
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
        
        # Execution & Logs
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
        if self.global_settings.cb_txt.isChecked(): output_formats.append("txt")
        if self.global_settings.cb_csv.isChecked(): output_formats.append("csv")
        if self.global_settings.cb_chunks.isChecked(): output_formats.append("chunks")

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
            'original_lyrics': self.lyrics_edit.toPlainText().strip(),
            'output_formats': output_formats,
            'slicing_method': self.slicing_combo.currentText(),
            'tempo': self.tempo_spin.value(),
            'quantization_step': _parse_quantization(self.quantize_combo.currentText()),
            'pitch_format': self.global_settings.pitch_combo.currentText(),
            'round_pitch': self.global_settings.cb_round.isChecked(),
            'seg_threshold': self.global_settings.seg_thresh_spin.value(),
            'seg_radius': self.global_settings.seg_rad_spin.value(),
            't0': self.global_settings.t0_spin.value(),
            'nsteps': self.global_settings.nsteps_spin.value(),
            'est_threshold': self.global_settings.est_thresh_spin.value(),
            'batch_size': self.global_settings.batch_spin.value(),
            'asr_batch_size': self.global_settings.asr_batch_spin.value()
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
            self.log_msg("正在停止任务，请稍候...")
            self.btn_stop.setEnabled(False)

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


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GAME: 生成式自适应 MIDI 提取器")
        self.resize(1000, 800)
        
        setTheme(Theme.LIGHT)

        # Create sub interfaces
        self.globalSettingsInterface = GlobalSettingsInterface(self)
        self.autoLyricInterface = AutoLyricInterface(self.globalSettingsInterface, self)

        self.initNavigation()

    def initNavigation(self):
        self.addSubInterface(self.autoLyricInterface, FluentIcon.MUSIC, "自动提取与灌注")
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, "全局设置", position=NavigationItemPosition.BOTTOM)
        
        self.navigationInterface.setCurrentItem(self.autoLyricInterface.objectName())
        self.stackedWidget.setCurrentWidget(self.autoLyricInterface)


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
