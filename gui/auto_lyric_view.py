import html
import os
import pathlib
import re

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from PySide6.QtCore import Qt

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
    SwitchButton,
    InfoBar,
    InfoBarPosition,
    IndeterminateProgressRing,
    isDarkTheme,
    qconfig,
)

from application.config import PipelineConfig, validate_slice_bounds
from gui.fluent_utils import t0_nstep_to_ts
from gui.fluent_worker import WorkerThread, HYBRID_AVAILABLE
from gui.i18n import tr
from gui.settings_utils import default_output_dir
from inference.device_utils import VISIBLE_RUNTIME_DEVICE_CHOICES, normalize_runtime_device

# value -> backend value; the display text comes from tr() at build/refresh time
SLICE_METHOD_CHOICES = [
    ("智能切片", "slice_smart"),
    ("启发式切片", "slice_heuristic"),
    ("默认切片", "slice_default"),
    ("网格搜索切片", "slice_grid"),
]
TARGET_LANGUAGE_CHOICES = [
    ("zh", "lang_name_zh"),
    ("zh-pinyin", "lang_name_zh_pinyin"),
    ("ja", "lang_name_ja"),
    ("en", "lang_name_en"),
]
EXPORT_FORMAT_CHOICES = [
    ("mid", "MIDI"),
    ("ustx", "USTX"),
    ("vsqx", "VSQX"),
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
    ("repair", "quant_repair"),
    ("bayes", "quant_bayes"),
    ("dp", "quant_dp"),
    ("simple", "quant_simple"),
]
LYRIC_OUTPUT_BY_LANGUAGE = {
    "zh": [("pinyin", "opt_pinyin"), ("hanzi", "opt_hanzi")],
    "zh-pinyin": [("pinyin", "opt_pinyin")],
    "ja": [("romaji", "opt_romaji"), ("kana", "opt_kana")],
    "en": [("word", "opt_word")],
}
DEFAULT_LYRIC_OUTPUT = {"zh": "hanzi", "zh-pinyin": "pinyin", "ja": "romaji", "en": "word"}
# Saved preferences used to store display texts; migrate them to values.
LEGACY_LYRIC_OUTPUT_VALUES = {"拼音": "pinyin", "汉字": "hanzi", "罗马音": "romaji", "假名": "kana", "单词": "word"}


class AutoLyricInterface(ScrollArea):
    AUDIO_EXTENSIONS = {".wav", ".m4a", ".flac", ".mp3", ".ogg", ".opus", ".wma", ".webm", ".aif", ".aiff"}

    # key: True for the dark theme; the light palette uses darker shades that
    # keep sufficient contrast on a light background
    _LOG_PALETTES = {
        True: {
            "bg": "#161b22",
            "border": "rgba(255, 255, 255, 0.08)",
            "text": "#d4d4d4",
            "error": "#f14c4c",
            "warn": "#e5c07b",
            "success": "#98c379",
        },
        False: {
            "bg": "#ffffff",
            "border": "rgba(0, 0, 0, 0.10)",
            "text": "#24292f",
            "error": "#cf222e",
            "warn": "#9a6700",
            "success": "#1a7f37",
        },
    }

    _LOG_COLOR_RULES = [
        ("error", re.compile(r"错误|error|traceback|failed|exception|失败", re.I)),
        ("warn", re.compile(r"警告|warning|取消|停止|跳过|重试|retry", re.I)),
        ("success", re.compile(r"成功|完成|finished|done", re.I)),
    ]

    def __init__(self, global_settings, model_config, parent=None):
        super().__init__(parent=parent)
        self.global_settings = global_settings
        self.model_config = model_config
        self._tr_bindings: list = []
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName('view')
        self.setObjectName('autoLyricInterface')

        title = SubtitleLabel(self)
        self._bind_tr(lambda: title.setText(tr("app_title")))
        self.vBoxLayout.addWidget(title)

        audio_card = CardWidget(self)
        audio_layout = QVBoxLayout(audio_card)
        header_layout = QHBoxLayout()

        music_icon = IconWidget(FluentIcon.MUSIC, self)
        music_icon.setFixedSize(16, 16)
        header_layout.addWidget(music_icon)

        title_label = BodyLabel(self)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: title_label.setText(tr("upload_audio")))
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        btn_add = PushButton(tr("pick_files"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda: btn_add.setText(tr("pick_files")))
        btn_add.clicked.connect(self.add_audio_files)
        btn_clear = PushButton(tr("clear_files"), self, FluentIcon.DELETE)
        self._bind_tr(lambda: btn_clear.setText(tr("clear_files")))
        btn_clear.clicked.connect(self.clear_audio_files)
        header_layout.addWidget(btn_add)
        header_layout.addWidget(btn_clear)
        audio_layout.addLayout(header_layout)

        self.audio_list = ListWidget(self)
        self.audio_list.setMaximumHeight(40)
        audio_layout.addWidget(self.audio_list)
        self.vBoxLayout.addWidget(audio_card)
        self.setAcceptDrops(True)

        self.lyric_card = CardWidget(self)
        lyric_layout = QVBoxLayout(self.lyric_card)
        self.lyric_title = BodyLabel(self)
        self.lyric_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: self.lyric_title.setText(tr("ref_lyrics")))
        lyric_layout.addWidget(self.lyric_title)

        self.lyrics_edit = TextEdit(self)
        self.lyrics_edit.setPlaceholderText(tr("ref_lyrics_hint"))
        self._bind_tr(lambda: self.lyrics_edit.setPlaceholderText(tr("ref_lyrics_hint")))
        self.lyrics_edit.setMaximumHeight(80)
        self.lyrics_edit.setAcceptDrops(False)  # let drops fall through to the file list
        lyric_layout.addWidget(self.lyrics_edit)
        self.vBoxLayout.addWidget(self.lyric_card)

        combo_card = CardWidget(self)
        combo_layout = QVBoxLayout(combo_card)

        combo_row1 = QHBoxLayout()
        self.slicing_combo = ComboBox(self)
        self._fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES)
        self._bind_tr(lambda: self._fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES, keep_value=True))
        self._add_flow_pair(combo_row1, "slicing_method", self.slicing_combo)
        combo_row1.addSpacing(28)
        self.lang_combo = ComboBox(self)
        self._fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES)
        self._bind_tr(lambda: self._fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES, keep_value=True))
        self.lang_combo.currentIndexChanged.connect(self.update_lyric_output_options)
        self._add_flow_pair(combo_row1, "target_lang", self.lang_combo)
        combo_row1.addSpacing(28)
        self.lyric_output_label = BodyLabel(self)
        self.lyric_output_combo = ComboBox(self)
        self.lyric_output_combo.currentIndexChanged.connect(self.save_lyric_output_preference)
        self._add_flow_pair(combo_row1, "lyric_output_format", self.lyric_output_combo, label=self.lyric_output_label)
        combo_row1.addSpacing(28)
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(list(VISIBLE_RUNTIME_DEVICE_CHOICES))
        self.device_combo.currentTextChanged.connect(self.apply_device_batch_defaults)
        self._add_flow_pair(combo_row1, "device", self.device_combo)
        combo_row1.addStretch(1)
        combo_layout.addLayout(combo_row1)

        combo_row2 = QHBoxLayout()
        self.cb_match_lyrics = SwitchButton("On", self)
        self.cb_match_lyrics.setOffText("Off")
        self.cb_match_lyrics.setChecked(self.global_settings.settings.value("enable_lyrics_match", False, type=bool))
        self.cb_match_lyrics.checkedChanged.connect(self.on_match_lyrics_changed)
        self._add_flow_pair(combo_row2, "match_lyrics", self.cb_match_lyrics)
        combo_row2.addSpacing(28)
        self.cb_output_lyrics = SwitchButton("On", self)
        self.cb_output_lyrics.setOffText("Off")
        self.cb_output_lyrics.setChecked(self.global_settings.settings.value("output_lyrics", True, type=bool))
        self.cb_output_lyrics.checkedChanged.connect(self.on_output_lyrics_changed)
        self._add_flow_pair(combo_row2, "output_lyrics", self.cb_output_lyrics)
        combo_row2.addSpacing(28)
        self.export_format_combo = ComboBox(self)
        self._fill_combo(self.export_format_combo, EXPORT_FORMAT_CHOICES)
        self.export_format_combo.setCurrentIndex(max(0, self.export_format_combo.findData(self._initial_export_format_value())))
        self.export_format_combo.currentIndexChanged.connect(self.on_export_format_changed)
        self._add_flow_pair(combo_row2, "export_format", self.export_format_combo)
        combo_row2.addSpacing(28)
        self.pitch_curve_label = BodyLabel(self)
        self.cb_pitch_curve = SwitchButton("On", self)
        self.cb_pitch_curve.setOffText("Off")
        self.cb_pitch_curve.setChecked(self.global_settings.settings.value("output_pitch_curve", True, type=bool))
        self.cb_pitch_curve.checkedChanged.connect(lambda v: self.global_settings.settings.setValue("output_pitch_curve", v))
        self._add_flow_pair(combo_row2, "pitch_curve", self.cb_pitch_curve, label=self.pitch_curve_label)
        combo_row2.addStretch(1)
        combo_layout.addLayout(combo_row2)

        self.vBoxLayout.addWidget(combo_card)

        output_card = CardWidget(self)
        output_layout = QVBoxLayout(output_card)
        output_title = BodyLabel(self)
        output_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: output_title.setText(tr("output_settings")))
        output_layout.addWidget(output_title)

        opts_layout = QHBoxLayout()
        self.tempo_spin = DoubleSpinBox(self)
        self.tempo_spin.setRange(10, 300)
        self.tempo_spin.setValue(120)
        self._add_flow_pair(opts_layout, "tempo_bpm", self.tempo_spin)
        opts_layout.addSpacing(28)
        self.quantize_combo = ComboBox(self)
        self._fill_combo(self.quantize_combo, QUANT_STEP_CHOICES)
        self._bind_tr(lambda: self._fill_combo(self.quantize_combo, QUANT_STEP_CHOICES, keep_value=True))
        self.quantize_combo.setCurrentIndex(0)
        self._add_flow_pair(opts_layout, "quant_step", self.quantize_combo)
        opts_layout.addSpacing(28)
        self.quantize_mode_combo = ComboBox(self)
        self._fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES)
        self._bind_tr(lambda: self._fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES, keep_value=True))
        self.quantize_mode_combo.setCurrentIndex(0)
        self._add_flow_pair(opts_layout, "quant_mode", self.quantize_mode_combo)
        opts_layout.addStretch(1)
        output_layout.addLayout(opts_layout)

        save_layout = QHBoxLayout()
        save_dir_label = BodyLabel(self)
        self._bind_tr(lambda: save_dir_label.setText(tr("save_dir")))
        save_layout.addWidget(save_dir_label)
        self.save_dir_edit = LineEdit(self)
        self.save_dir_edit.setText(
            self.global_settings.settings.value("save_dir", str(default_output_dir(self.global_settings.project_root)))
        )
        self.save_dir_edit.textChanged.connect(lambda t: self.global_settings.settings.setValue("save_dir", t))
        save_layout.addWidget(self.save_dir_edit, 1)
        btn_browse_save = PushButton(tr("browse"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda: btn_browse_save.setText(tr("browse")))
        btn_browse_save.clicked.connect(lambda: self.browse_dir(self.save_dir_edit))
        save_layout.addWidget(btn_browse_save)
        output_layout.addLayout(save_layout)
        self.vBoxLayout.addWidget(output_card)

        action_layout = QHBoxLayout()
        self.btn_run = PrimaryPushButton(tr("run"), self, FluentIcon.PLAY)
        self._bind_tr(lambda: self.btn_run.setText(tr("run")))
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_stop = PushButton(tr("stop"), self, FluentIcon.PAUSE)
        self._bind_tr(lambda: self.btn_stop.setText(tr("stop")))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)
        self.progress_ring = IndeterminateProgressRing(self)
        self.progress_ring.setFixedSize(28, 28)
        self.progress_ring.setVisible(False)
        self.run_status_label = BodyLabel("", self)
        action_layout.addWidget(self.btn_run, 1)
        action_layout.addWidget(self.btn_stop, 1)
        action_layout.addWidget(self.progress_ring)
        action_layout.addWidget(self.run_status_label)
        self.vBoxLayout.addLayout(action_layout)

        self.log_edit = TextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        self.log_edit.setObjectName("logTerminal")
        self._log_lines = []
        self._apply_log_terminal_style()
        # themeChangedFinished fires after qfluentwidgets reapplies its widget
        # stylesheets; otherwise our terminal style gets overwritten by the
        # library's TextEdit stylesheet
        qconfig.themeChangedFinished.connect(self._on_log_theme_changed)
        self.vBoxLayout.addWidget(self.log_edit)

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        self.worker = None
        self._last_device = None
        self.update_lyrics_visibility()
        self.update_lyric_output_options()
        self._last_device = self.device_combo.currentText()  # baseline; don't wipe saved batches on startup
        self.on_export_format_changed()

    # ── i18n helpers ────────────────────────────────────────────────
    def _bind_tr(self, fn):
        self._tr_bindings.append(fn)
        fn()

    def retranslate_ui(self):
        for fn in self._tr_bindings:
            fn()

    @staticmethod
    def _fill_combo(combo, choices: list[tuple], keep_value: bool = False):
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

    def _add_flow_pair(self, row, label_key, widget, label=None):
        label = label or BodyLabel(self)
        self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
        row.addWidget(label)
        row.addWidget(widget)

    # ── drag & drop ─────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [
            url.toLocalFile() for url in event.mimeData().urls()
            if url.isLocalFile() and pathlib.Path(url.toLocalFile()).suffix.lower() in self.AUDIO_EXTENSIONS
        ]
        if paths:
            event.acceptProposedAction()
            self.add_audio_paths(paths)
        else:
            event.ignore()

    def add_audio_paths(self, paths):
        existing = {self.audio_list.item(i).text() for i in range(self.audio_list.count())}
        added = 0
        for path in paths:
            path = str(path)
            if pathlib.Path(path).suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue
            if path in existing:
                continue
            self.audio_list.addItem(path)
            existing.add(path)
            added += 1
        if added:
            self.log_msg(tr("files_added", n=added))

    # ── log terminal ────────────────────────────────────────────────
    def _log_palette(self):
        return self._LOG_PALETTES[isDarkTheme()]

    def _apply_log_terminal_style(self):
        p = self._log_palette()
        self.log_edit.setStyleSheet(
            "#logTerminal{"
            f"background-color: {p['bg']};"
            f"color: {p['text']};"
            f"border: 1px solid {p['border']};"
            "border-radius: 6px;"
            "padding: 6px;"
            "font-family: 'Cascadia Mono', 'Consolas', 'Courier New', monospace;"
            "font-size: 12px;"
            "}"
        )

    def _line_html(self, msg):
        p = self._log_palette()
        kind = "text"
        for rule_kind, pattern in self._LOG_COLOR_RULES:
            if pattern.search(msg):
                kind = rule_kind
                break
        text = html.escape(msg).replace("\n", "<br>")
        return f'<span style="color:{p[kind]};">{text}</span>'

    def _on_log_theme_changed(self):
        # rebuild every log line on theme change so stale colors never sit on
        # the new background
        self._apply_log_terminal_style()
        self.log_edit.setHtml("<br>".join(self._line_html(m) for m in self._log_lines))
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def log_msg(self, msg):
        self._log_lines.append(msg)
        self.log_edit.append(self._line_html(msg))
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_error(self, title: str, content: str):
        self.log_msg(f"{tr('error_prefix')}: {content}")
        InfoBar.error(
            title=title, content=content, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=5000, parent=self,
        )

    def _set_running_ui(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.progress_ring.setVisible(running)
        if not running:
            self.run_status_label.setText("")

    def on_progress(self, current: int, total: int, filename: str):
        self.run_status_label.setText(tr("processing", i=current, n=total, f=filename))

    # ── option helpers ──────────────────────────────────────────────
    def apply_device_batch_defaults(self, device: str):
        # Apply conservative batch defaults only when the device actually
        # changes; never wipe a value the user persisted in global settings.
        if device == self._last_device:
            return
        self._last_device = device
        self.global_settings.batch_spin.setValue(1)
        self.global_settings.asr_batch_spin.setValue(2)

    def update_lyrics_visibility(self):
        enabled = self.cb_match_lyrics.isChecked()
        self.lyric_card.setVisible(enabled)
        self.lyric_title.setVisible(enabled)
        self.lyrics_edit.setVisible(enabled)

    def _lyric_output_setting_key(self, language: str):
        return f"lyric_output_mode_{language}"

    def _selected_language(self) -> str:
        return self.lang_combo.currentData() or "zh"

    def save_lyric_output_preference(self, *_args):
        language = self._selected_language()
        value = self.lyric_output_combo.currentData()
        if value:
            self.global_settings.settings.setValue(self._lyric_output_setting_key(language), value)

    def update_lyric_output_enabled_state(self):
        enabled = self.cb_output_lyrics.isChecked()
        self.lyric_output_label.setEnabled(enabled)
        self.lyric_output_combo.setEnabled(enabled)

    def on_output_lyrics_changed(self, enabled: bool):
        self.global_settings.settings.setValue("output_lyrics", enabled)
        self.update_lyric_output_enabled_state()

    def on_match_lyrics_changed(self, enabled: bool):
        self.global_settings.settings.setValue("enable_lyrics_match", enabled)
        self.update_lyrics_visibility()

    def get_export_format(self) -> str:
        return self.export_format_combo.currentData() or "mid"

    def _initial_export_format_value(self) -> str:
        """Load the selected format, migrating the former export switches."""
        saved_format = str(self.global_settings.settings.value("export_format", "")).strip().lower()
        return saved_format if saved_format in {"mid", "ustx", "vsqx"} else "mid"

    def on_export_format_changed(self, *_args):
        self.global_settings.settings.setValue("export_format", self.get_export_format())
        self._update_pitch_curve_enabled()

    def _update_pitch_curve_enabled(self):
        enabled = self.get_export_format() in {"ustx", "vsqx"}
        self.pitch_curve_label.setEnabled(enabled)
        self.cb_pitch_curve.setEnabled(enabled)

    def update_lyric_output_options(self, *_args):
        language = self._selected_language()
        choices = LYRIC_OUTPUT_BY_LANGUAGE.get(language, LYRIC_OUTPUT_BY_LANGUAGE["zh"])
        saved_value = str(self.global_settings.settings.value(
            self._lyric_output_setting_key(language),
            DEFAULT_LYRIC_OUTPUT.get(language, "hanzi"),
        ))
        saved_value = LEGACY_LYRIC_OUTPUT_VALUES.get(saved_value, saved_value)

        self._fill_combo(self.lyric_output_combo, choices)
        values = [value for value, _ in choices]
        index = self.lyric_output_combo.findData(saved_value if saved_value in values else DEFAULT_LYRIC_OUTPUT.get(language, "hanzi"))
        self.lyric_output_combo.setCurrentIndex(max(0, index))

        self.save_lyric_output_preference()
        self.update_lyric_output_enabled_state()

    def get_lyric_output_mode(self):
        return self.lyric_output_combo.currentData() or DEFAULT_LYRIC_OUTPUT.get(self._selected_language(), "hanzi")

    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, tr("choose_folder_dialog"), line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def add_audio_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("pick_files_dialog"), "",
            "Audio Files (*.wav *.m4a *.flac *.mp3 *.ogg *.opus *.wma *.webm *.aif *.aiff)"
        )
        self.add_audio_paths(files)

    def clear_audio_files(self):
        self.audio_list.clear()

    def run_pipeline(self):
        if not HYBRID_AVAILABLE:
            self.log_msg(tr("err_hybrid"))
            return

        audio_files = [self.audio_list.item(i).text() for i in range(self.audio_list.count())]
        if not audio_files:
            self._show_error(tr("err_cannot_start"), tr("err_no_audio"))
            return

        selected_export_format = self.get_export_format()
        output_formats = [selected_export_format]
        if self.global_settings.cb_txt.isChecked():
            output_formats.append("txt")
        if self.global_settings.cb_csv.isChecked():
            output_formats.append("csv")
        if self.global_settings.cb_chunks.isChecked():
            output_formats.append("chunks")
        save_dir = self.save_dir_edit.text()
        if not save_dir:
            self._show_error(tr("err_cannot_start"), tr("err_no_dir"))
            return

        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                self.log_msg(tr("info_dir_created", dir=save_dir))
            except Exception as e:
                self._show_error(tr("err_dir_create"), str(e))
                return
        elif not os.path.isdir(save_dir):
            self._show_error(tr("err_cannot_start"), tr("err_dir_not_dir"))
            return

        ts_list = t0_nstep_to_ts(
            self.global_settings.t0_spin.value(),
            int(self.global_settings.nsteps_spin.value()),
        )
        device = normalize_runtime_device(self.device_combo.currentText())
        slice_min_sec = float(self.global_settings.slice_min_spin.value())
        slice_max_sec = float(self.global_settings.slice_max_spin.value())
        try:
            validate_slice_bounds(slice_min_sec, slice_max_sec)
        except ValueError as exc:
            self.log_msg(f"Error: invalid slice duration settings: {exc}")
            return

        config = PipelineConfig(
            audio_path="",  # set per-file in worker
            output_filename="",  # set per-file in worker
            output_dir=pathlib.Path(save_dir),
            game_model_dir=self.model_config.game_model_edit.text(),
            hfa_model_dir=self.model_config.hfa_model_edit.text(),
            asr_model_path=self.model_config.asr_model_edit.text(),
            device=device,
            language=self._selected_language(),
            ts=ts_list,
            lyric_output_mode=self.get_lyric_output_mode(),
            original_lyrics=self.lyrics_edit.toPlainText().strip() if self.cb_match_lyrics.isChecked() else "",
            output_formats=output_formats,
            output_lyrics=self.cb_output_lyrics.isChecked(),
            output_pitch_curve=self.cb_pitch_curve.isChecked() if selected_export_format in {"ustx", "vsqx"} else False,
            slicing_method=self.slicing_combo.currentData(),
            slice_min_sec=slice_min_sec,
            slice_max_sec=slice_max_sec,
            tempo=self.tempo_spin.value(),
            quantization_step=self.quantize_combo.currentData(),
            quantization_mode=self.quantize_mode_combo.currentData(),
            pitch_format=self.global_settings.pitch_combo.currentText(),
            round_pitch=self.global_settings.cb_round.isChecked(),
            seg_threshold=self.global_settings.seg_thresh_spin.value(),
            seg_radius=self.global_settings.seg_rad_spin.value(),
            est_threshold=self.global_settings.est_thresh_spin.value(),
            batch_size=self.global_settings.batch_spin.value(),
            asr_batch_size=self.global_settings.asr_batch_spin.value(),
            rmvpe_model_path=self.model_config.rmvpe_model_edit.text(),
            phoneme_asr_model_path=self.model_config.phoneme_asr_model_edit.text(),
            pinyin_asr_model_path=self.model_config.pinyin_asr_model_edit.text(),
        )

        self.log_edit.clear()
        self._log_lines.clear()
        self._set_running_ui(True)
        self.run_status_label.setText(tr("preparing", n=len(audio_files)))
        self.worker = WorkerThread(config, audio_files)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def stop_pipeline(self):
        if self.worker:
            self.worker.stop()
            self.run_status_label.setText(tr("stopping"))
            InfoBar.warning(
                title=tr("stop_requested_title"), content=tr("stop_requested_body"), orient=Qt.Horizontal,
                isClosable=True, position=InfoBarPosition.TOP, duration=3000, parent=self,
            )
            if self.worker.isRunning():
                self.log_msg(tr("stop_hint"))

    def on_finished(self, msg):
        self.log_msg(msg)
        self._set_running_ui(False)
        InfoBar.success(
            title=tr("done_title"), content=msg, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=6000, parent=self,
        )
        try:
            save_dir_path = self.save_dir_edit.text()
            if os.path.exists(save_dir_path):
                os.startfile(save_dir_path)
        except Exception as e:
            self.log_msg(f"Cannot open output folder: {e}")

    def on_error(self, msg):
        self.log_msg(msg)
        self._set_running_ui(False)
        InfoBar.error(
            title=tr("fail_title"), content=tr("fail_body"), orient=Qt.Horizontal,
            isClosable=True, position=InfoBarPosition.TOP, duration=6000, parent=self,
        )
