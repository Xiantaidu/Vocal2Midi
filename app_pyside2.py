import sys
import os
import pathlib
import tempfile
import traceback
import zipfile
import torch

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, 
                               QGroupBox, QFileDialog, QTextEdit, QListWidget, QTabWidget, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal

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
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(sys.stdout, self.log_signal)
        sys.stderr = StreamRedirector(sys.stderr, self.log_signal)
        
        try:
            audio_files = self.kwargs['audio_files']
            output_dir = pathlib.Path(tempfile.mkdtemp(prefix="game_qt_autolyric_"))
            
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
                    output_dir=output_dir,
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
                generated_files = list(output_dir.glob("*"))
                if not generated_files:
                    self.error_signal.emit("推理完成，但未生成任何输出文件。")
                else:
                    if len(generated_files) == 1:
                        result_path = str(generated_files[0])
                        self.finished_signal.emit(f"提取成功！文件位于: {result_path}")
                    else:
                        zip_path = output_dir.parent / "autolyric_results.zip"
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in generated_files:
                                zipf.write(file, file.name)
                        self.finished_signal.emit(f"提取成功！请查看 ZIP 压缩包: {zip_path}")
            else:
                self.error_signal.emit("任务已被取消。")
                
        except Exception as e:
            self.error_signal.emit(f"发生错误:\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAME: 生成式自适应 MIDI 提取器 (PySide2)")
        self.resize(1000, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- Global Settings ---
        global_group = QGroupBox("全局设置")
        global_layout = QGridLayout()
        
        global_layout.addWidget(QLabel("GAME 模型路径:"), 0, 0)
        self.game_model_edit = QLineEdit(r"E:\Vocal2Midi\experiments\GAME-1.0-medium")
        global_layout.addWidget(self.game_model_edit, 0, 1)
        btn_browse_game = QPushButton("浏览")
        btn_browse_game.clicked.connect(lambda: self.browse_dir(self.game_model_edit))
        global_layout.addWidget(btn_browse_game, 0, 2)
        
        global_layout.addWidget(QLabel("目标语言:"), 0, 3)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["zh", "ja"])
        global_layout.addWidget(self.lang_combo, 0, 4)
        
        global_layout.addWidget(QLabel("计算设备:"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        global_layout.addWidget(self.device_combo, 1, 1)
        
        global_group.setLayout(global_layout)
        main_layout.addWidget(global_group)
        
        # --- Tabs ---
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # --- Auto Lyric Tab ---
        tab_auto_lyric = QWidget()
        al_layout = QVBoxLayout(tab_auto_lyric)
        tabs.addTab(tab_auto_lyric, "🎤 自动提取与歌词灌注 (Auto Lyric)")
        
        top_split = QHBoxLayout()
        al_layout.addLayout(top_split)
        
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        top_split.addLayout(left_panel, 1)
        top_split.addLayout(right_panel, 1)
        
        # Left Panel Components
        audio_group = QGroupBox("上传音频文件")
        audio_layout = QVBoxLayout()
        self.audio_list = QListWidget()
        audio_layout.addWidget(self.audio_list)
        btn_audio_add = QPushButton("添加文件")
        btn_audio_add.clicked.connect(self.add_audio_files)
        btn_audio_clear = QPushButton("清空列表")
        btn_audio_clear.clicked.connect(self.audio_list.clear)
        audio_btn_layout = QHBoxLayout()
        audio_btn_layout.addWidget(btn_audio_add)
        audio_btn_layout.addWidget(btn_audio_clear)
        audio_layout.addLayout(audio_btn_layout)
        audio_group.setLayout(audio_layout)
        left_panel.addWidget(audio_group)
        
        model_group = QGroupBox("辅助模型配置")
        model_layout = QGridLayout()
        model_layout.addWidget(QLabel("HubertFA:"), 0, 0)
        self.hfa_model_edit = QLineEdit(r"E:\Vocal2Midi\experiments\1218_hfa_model_new_dict")
        model_layout.addWidget(self.hfa_model_edit, 0, 1)
        btn_hfa = QPushButton("浏览")
        btn_hfa.clicked.connect(lambda: self.browse_dir(self.hfa_model_edit))
        model_layout.addWidget(btn_hfa, 0, 2)
        
        model_layout.addWidget(QLabel("Qwen3-ASR:"), 1, 0)
        self.asr_model_edit = QLineEdit(r"C:\Users\Xiantaidu\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B")
        model_layout.addWidget(self.asr_model_edit, 1, 1)
        btn_asr = QPushButton("浏览")
        btn_asr.clicked.connect(lambda: self.browse_dir(self.asr_model_edit))
        model_layout.addWidget(btn_asr, 1, 2)
        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)
        
        lyrics_group = QGroupBox("参考歌词 (可选)")
        lyrics_layout = QVBoxLayout()
        self.lyrics_edit = QTextEdit()
        self.lyrics_edit.setPlaceholderText("如果有确切的歌词，请在此输入（纯文本）以提高对齐准确率...")
        self.lyrics_edit.setMaximumHeight(80)
        lyrics_layout.addWidget(self.lyrics_edit)
        
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("音频切片方法:"))
        self.slicing_combo = QComboBox()
        self.slicing_combo.addItems(["默认切片", "启发式切片", "网格搜索切片"])
        slice_layout.addWidget(self.slicing_combo)
        lyrics_layout.addLayout(slice_layout)
        lyrics_group.setLayout(lyrics_layout)
        left_panel.addWidget(lyrics_group)
        
        output_group = QGroupBox("输出设置")
        output_layout = QGridLayout()
        self.cb_mid = QCheckBox("导出 MIDI (.mid)"); self.cb_mid.setChecked(True)
        self.cb_txt = QCheckBox("导出 Text (.txt)"); self.cb_txt.setChecked(True)
        self.cb_csv = QCheckBox("导出 CSV (.csv)")
        self.cb_chunks = QCheckBox("导出切片与 TextGrid")
        output_layout.addWidget(self.cb_mid, 0, 0); output_layout.addWidget(self.cb_txt, 0, 1)
        output_layout.addWidget(self.cb_csv, 1, 0); output_layout.addWidget(self.cb_chunks, 1, 1)
        
        output_layout.addWidget(QLabel("曲速 (Tempo BPM):"), 2, 0)
        self.tempo_spin = QDoubleSpinBox(); self.tempo_spin.setRange(10, 300); self.tempo_spin.setValue(120)
        output_layout.addWidget(self.tempo_spin, 2, 1)
        
        output_layout.addWidget(QLabel("MIDI 量化精度:"), 3, 0)
        self.quantize_combo = QComboBox()
        self.quantize_combo.addItems(["不量化", "1/4 音符 (1拍)", "1/8 音符 (1/2拍)", "1/16 音符 (1/4拍)", "1/32 音符 (1/8拍)", "1/64 音符 (1/16拍)"])
        output_layout.addWidget(self.quantize_combo, 3, 1)
        
        output_layout.addWidget(QLabel("音高格式:"), 4, 0)
        self.pitch_combo = QComboBox(); self.pitch_combo.addItems(["name", "number"])
        output_layout.addWidget(self.pitch_combo, 4, 1)
        
        self.cb_round = QCheckBox("音高取整")
        output_layout.addWidget(self.cb_round, 5, 0)
        output_group.setLayout(output_layout)
        left_panel.addWidget(output_group)
        
        # Right Panel Components
        adv_group = QGroupBox("高级处理参数")
        adv_layout = QGridLayout()
        
        adv_layout.addWidget(QLabel("边界解码阈值:"), 0, 0)
        self.seg_thresh_spin = QDoubleSpinBox(); self.seg_thresh_spin.setRange(0.01, 0.99); self.seg_thresh_spin.setSingleStep(0.01); self.seg_thresh_spin.setValue(0.2)
        adv_layout.addWidget(self.seg_thresh_spin, 0, 1)
        
        adv_layout.addWidget(QLabel("边界解码半径/秒:"), 1, 0)
        self.seg_rad_spin = QDoubleSpinBox(); self.seg_rad_spin.setRange(0.01, 0.1); self.seg_rad_spin.setSingleStep(0.005); self.seg_rad_spin.setValue(0.02)
        adv_layout.addWidget(self.seg_rad_spin, 1, 1)
        
        adv_layout.addWidget(QLabel("D3PM 起始 T 值:"), 2, 0)
        self.t0_spin = QDoubleSpinBox(); self.t0_spin.setRange(0.0, 0.99); self.t0_spin.setSingleStep(0.01); self.t0_spin.setValue(0.0)
        adv_layout.addWidget(self.t0_spin, 2, 1)
        
        adv_layout.addWidget(QLabel("D3PM 采样步数:"), 3, 0)
        self.nsteps_spin = QSpinBox(); self.nsteps_spin.setRange(1, 20); self.nsteps_spin.setValue(8)
        adv_layout.addWidget(self.nsteps_spin, 3, 1)
        
        adv_layout.addWidget(QLabel("音符存在阈值:"), 4, 0)
        self.est_thresh_spin = QDoubleSpinBox(); self.est_thresh_spin.setRange(0.01, 0.99); self.est_thresh_spin.setSingleStep(0.01); self.est_thresh_spin.setValue(0.2)
        adv_layout.addWidget(self.est_thresh_spin, 4, 1)
        
        adv_layout.addWidget(QLabel("GAME 批处理大小:"), 5, 0)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 32); self.batch_spin.setValue(4)
        adv_layout.addWidget(self.batch_spin, 5, 1)
        
        adv_layout.addWidget(QLabel("ASR 批处理大小:"), 6, 0)
        self.asr_batch_spin = QSpinBox(); self.asr_batch_spin.setRange(1, 32); self.asr_batch_spin.setValue(2)
        adv_layout.addWidget(self.asr_batch_spin, 6, 1)
        
        adv_group.setLayout(adv_layout)
        right_panel.addWidget(adv_group)
        
        log_group = QGroupBox("运行日志与状态")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("🚀 开始全自动提取")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px;")
        self.btn_run.clicked.connect(self.run_pipeline)
        
        self.btn_stop = QPushButton("🛑 强制停止")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 16px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)
        
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        al_layout.addLayout(btn_layout)
        
        self.worker = None

    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def add_audio_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择音频文件", "", "Audio Files (*.wav *.flac *.mp3 *.ogg)")
        if files:
            for f in files:
                # Add only if not already in list
                items = self.audio_list.findItems(f, Qt.MatchExactly)
                if not items:
                    self.audio_list.addItem(f)

    def log_msg(self, msg):
        self.log_edit.append(msg)
        # Scroll to bottom
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
            
        output_formats = []
        if self.cb_mid.isChecked(): output_formats.append("mid")
        if self.cb_txt.isChecked(): output_formats.append("txt")
        if self.cb_csv.isChecked(): output_formats.append("csv")
        if self.cb_chunks.isChecked(): output_formats.append("chunks")
        if not output_formats:
            self.log_msg("错误: 请至少选择一种输出格式。")
            return

        kwargs = {
            'audio_files': audio_files,
            'game_model_path_str': self.game_model_edit.text(),
            'device': self.device_combo.currentText(),
            'hfa_model_path_str': self.hfa_model_edit.text(),
            'asr_model_path_str': self.asr_model_edit.text(),
            'language': self.lang_combo.currentText(),
            'original_lyrics': self.lyrics_edit.toPlainText().strip(),
            'output_formats': output_formats,
            'slicing_method': self.slicing_combo.currentText(),
            'tempo': self.tempo_spin.value(),
            'quantization_step': _parse_quantization(self.quantize_combo.currentText()),
            'pitch_format': self.pitch_combo.currentText(),
            'round_pitch': self.cb_round.isChecked(),
            'seg_threshold': self.seg_thresh_spin.value(),
            'seg_radius': self.seg_rad_spin.value(),
            't0': self.t0_spin.value(),
            'nsteps': self.nsteps_spin.value(),
            'est_threshold': self.est_thresh_spin.value(),
            'batch_size': self.batch_spin.value(),
            'asr_batch_size': self.asr_batch_spin.value()
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

    def on_error(self, msg):
        self.log_msg(msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply modern styling
    app.setStyle("Fusion")
    
    modern_qss = """
    QWidget {
        background-color: #2b2b2b;
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-size: 13px;
    }
    QGroupBox {
        border: 1px solid #444444;
        border-radius: 8px;
        margin-top: 1.5ex;
        padding-top: 15px;
        background-color: #323232;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: #4CAF50;
        font-weight: bold;
        left: 10px;
    }
    QPushButton {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 12px;
        color: #ffffff;
    }
    QPushButton:hover {
        background-color: #505050;
        border: 1px solid #777777;
    }
    QPushButton:pressed {
        background-color: #2d2d30;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListWidget {
        background-color: #1e1e1e;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 4px;
        color: #ffffff;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
        border: 1px solid #4CAF50;
    }
    QComboBox::drop-down {
        border: none;
    }
    QTabWidget::pane {
        border: 1px solid #444444;
        border-radius: 4px;
        background-color: #2b2b2b;
    }
    QTabBar::tab {
        background: #3e3e42;
        color: #aaaaaa;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: #2b2b2b;
        color: #ffffff;
        border: 1px solid #444444;
        border-bottom-color: #2b2b2b;
        font-weight: bold;
    }
    QTabBar::tab:hover:!selected {
        background: #505050;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid #555555;
        background: #1e1e1e;
    }
    QCheckBox::indicator:checked {
        background: #4CAF50;
        border: 1px solid #4CAF50;
    }
    QScrollBar:vertical {
        border: none;
        background: #2b2b2b;
        width: 12px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:vertical {
        background: #555555;
        min-height: 20px;
        border-radius: 6px;
        margin: 2px;
    }
    QScrollBar::handle:vertical:hover {
        background: #777777;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """
    app.setStyleSheet(modern_qss)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
