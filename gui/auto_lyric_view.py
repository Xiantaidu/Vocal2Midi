import os
import pathlib

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
    FluentIcon,
    SubtitleLabel,
    SwitchButton,
    InfoBar,
    InfoBarPosition,
    IndeterminateProgressRing,
)

from application.config import PipelineConfig, validate_slice_bounds
from gui.audio_file_list import AudioFileList
from gui.fluent_utils import t0_nstep_to_ts
from gui.fluent_worker import WorkerThread, HYBRID_AVAILABLE
from gui.i18n import tr
from gui.log_terminal import LogTerminal
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
from inference.device_utils import VISIBLE_RUNTIME_DEVICE_CHOICES, normalize_runtime_device


class AutoLyricInterface(ScrollArea):
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

        self.audio_list = AudioFileList(self)
        self.audio_list.setMaximumHeight(40)
        audio_layout.addWidget(self.audio_list, 1)  # stretch so batch mode can grow the list
        self._audio_card_stretch = audio_layout
        self.vBoxLayout.addWidget(audio_card)
        self.setAcceptDrops(True)
        self._is_running = False
        self.audio_list.filesAdded.connect(lambda n: self.log_msg(tr("files_added", n=n)))
        self.audio_list.filesChanged.connect(self._update_batch_mode)
        self.audio_list.settingsRequested.connect(self._open_file_settings)
        self._audio_card = audio_card
        self._combo_card = None  # set after cards are built
        self._output_card = None

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
        fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES)
        self._bind_tr(lambda: fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES, keep_value=True))
        self._add_flow_pair(combo_row1, "slicing_method", self.slicing_combo)
        combo_row1.addSpacing(28)
        self.lang_combo = ComboBox(self)
        fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES)
        self._bind_tr(lambda: fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES, keep_value=True))
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
        fill_combo(self.export_format_combo, EXPORT_FORMAT_CHOICES)
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
        self._combo_card = combo_card

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
        fill_combo(self.quantize_combo, QUANT_STEP_CHOICES)
        self._bind_tr(lambda: fill_combo(self.quantize_combo, QUANT_STEP_CHOICES, keep_value=True))
        self.quantize_combo.setCurrentIndex(0)
        self._add_flow_pair(opts_layout, "quant_step", self.quantize_combo)
        opts_layout.addSpacing(28)
        self.quantize_mode_combo = ComboBox(self)
        fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES)
        self._bind_tr(lambda: fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES, keep_value=True))
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
        self._output_card = output_card

        # selection changes drive batch mode; the retranslate refresh runs
        # after __init__ finishes (batch refresh is invoked there too)
        self.audio_list.itemSelectionChanged.connect(self._update_batch_mode)

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

        self.log_edit = LogTerminal(self)
        self.vBoxLayout.addWidget(self.log_edit)

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        self.worker = None
        self._last_device = None
        self.update_lyrics_visibility()
        self.update_lyric_output_options()
        # Switching the Chinese ASR engine re-evaluates the pinyin output lock.
        self.model_config.chinese_asr_engine_changed.connect(
            lambda _value: self.update_lyric_output_options()
        )
        self._last_device = self.device_combo.currentText()  # baseline; don't wipe saved batches on startup
        self.on_export_format_changed()
        self._update_batch_mode()  # all widgets exist now

    # ── i18n helpers ────────────────────────────────────────────────
    def _bind_tr(self, fn):
        self._tr_bindings.append(fn)
        fn()

    def retranslate_ui(self):
        for fn in self._tr_bindings:
            fn()

    @staticmethod
    def _fill_combo(combo, choices: list[tuple], keep_value: bool = False):
        fill_combo(combo, choices, keep_value=keep_value)

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
            if url.isLocalFile() and pathlib.Path(url.toLocalFile()).suffix.lower() in AudioFileList.AUDIO_EXTENSIONS
        ]
        if paths:
            event.acceptProposedAction()
            self.audio_list.add_paths(paths)
        else:
            event.ignore()

    # ── log terminal ────────────────────────────────────────────────
    def log_msg(self, msg):
        self.log_edit.log(msg)

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
        self.global_settings.apply_batch_defaults(batch=1, asr_batch=2)

    def update_lyrics_visibility(self):
        enabled = self.cb_match_lyrics.isChecked()
        self.lyric_card.setVisible(enabled)
        self.lyric_title.setVisible(enabled)
        self.lyrics_edit.setVisible(enabled)

    def _lyric_output_setting_key(self, language: str):
        return f"lyric_output_mode_{language}"

    def _selected_language(self) -> str:
        return self.lang_combo.currentData() or "zh"

    def _chinese_asr_engine(self) -> str:
        return self.model_config.chinese_asr_engine()

    def _japanese_asr_engine(self) -> str:
        return self.model_config.japanese_asr_engine()

    def _pinyin_output_locked(self) -> bool:
        """PinyinASR can only emit pinyin, so hanzi output is unavailable."""
        return self._selected_language() == "zh" and self._chinese_asr_engine() == "pinyin"

    def save_lyric_output_preference(self, *_args):
        language = self._selected_language()
        value = self.lyric_output_combo.currentData()
        if value:
            self.global_settings.settings.setValue(self._lyric_output_setting_key(language), value)

    def update_lyric_output_enabled_state(self):
        enabled = self.cb_output_lyrics.isChecked()
        locked = self._pinyin_output_locked()
        self.lyric_output_label.setEnabled(enabled)
        self.lyric_output_combo.setEnabled(enabled and not locked)
        if locked:
            hint = tr("lyric_output_locked_hint")
            self.lyric_output_combo.setToolTip(hint)
            self.lyric_output_label.setToolTip(hint)
        else:
            self.lyric_output_combo.setToolTip("")
            self.lyric_output_label.setToolTip("")

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
        locked = self._pinyin_output_locked()
        choices = [
            (value, key)
            for value, key in LYRIC_OUTPUT_BY_LANGUAGE.get(language, LYRIC_OUTPUT_BY_LANGUAGE["zh"])
            if not locked or value == "pinyin"
        ]
        saved_value = str(self.global_settings.settings.value(
            self._lyric_output_setting_key(language),
            DEFAULT_LYRIC_OUTPUT.get(language, "hanzi"),
        ))

        self._fill_combo(self.lyric_output_combo, choices)
        values = [value for value, _ in choices]
        if saved_value not in values:
            saved_value = "pinyin" if locked else DEFAULT_LYRIC_OUTPUT.get(language, "hanzi")
        index = self.lyric_output_combo.findData(saved_value)
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
        self.audio_list.add_paths(files)

    def clear_audio_files(self):
        self.audio_list.clear_all()

    # ── batch mode ──────────────────────────────────────────────────
    def _update_batch_mode(self):
        """Single file keeps the panels; more than one file in the list enters
        batch mode: panels hide and the list grows to fill the freed space."""
        file_count = self.audio_list.count()
        batch = file_count > 1
        self._combo_card.setVisible(not batch)
        self._output_card.setVisible(not batch)
        self.audio_list.setMaximumHeight(16777215 if batch else 40)  # unclamped in batch mode
        self.audio_list.setMinimumHeight(160 if batch else 40)
        if batch:
            self.run_status_label.setText(tr("batch_mode", n=file_count))
        elif not self._is_running:
            self.run_status_label.setText("")

    def _current_base_values(self) -> dict:
        """Snapshot every parameter the main UI currently controls.

        All entries are backend values; the per-file dialog round-trips them
        without any display-text conversion.
        """
        return {
            "slicing_method": self.slicing_combo.currentData(),
            "language": self.lang_combo.currentData() or "zh",
            "lyric_output": self.lyric_output_combo.currentData() or "hanzi",
            "device": self.device_combo.currentText(),
            "match_lyrics": self.cb_match_lyrics.isChecked(),
            "original_lyrics": self.lyrics_edit.toPlainText().strip(),
            "export_format": self.get_export_format(),
            "output_lyrics": self.cb_output_lyrics.isChecked(),
            "pitch_curve": self.cb_pitch_curve.isChecked(),
            "tempo": self.tempo_spin.value(),
            "quantization_step": self.quantize_combo.currentData(),
            "quantization_mode": self.quantize_mode_combo.currentData(),
            "batch_size": self.global_settings.batch_size(),
            "asr_batch_size": self.global_settings.asr_batch_size(),
            "output_dir": self.save_dir_edit.text(),
            "chinese_asr_engine": self._chinese_asr_engine(),
            "japanese_asr_engine": self._japanese_asr_engine(),
            "devices": VISIBLE_RUNTIME_DEVICE_CHOICES,
            "slice_min_sec": self.global_settings.slice_min_sec(),
            "slice_max_sec": self.global_settings.slice_max_sec(),
            "output_formats_extra": self.global_settings.extra_output_formats(),
            "pitch_format": self.global_settings.pitch_format(),
            "round_pitch": self.global_settings.round_pitch(),
            "seg_threshold": self.global_settings.seg_threshold(),
            "seg_radius": self.global_settings.seg_radius(),
            "est_threshold": self.global_settings.est_threshold(),
        }

    def _open_file_settings(self, filename: str):
        from gui.file_settings_dialog import FileSettingsDialog

        base = dict(self._current_base_values())
        saved = self.audio_list.file_settings.get(filename)
        if saved:
            base.update({k: v for k, v in saved.items() if k in base})
        dialog = FileSettingsDialog(filename, base, self)
        if dialog.exec():
            self.audio_list.file_settings[filename] = dialog.values()
            self.log_msg(f"{filename}: {tr('file_settings')} {tr('apply')}")

    def _build_file_config(self, filename: str, base: dict, ts_list: list, overrides: dict | None) -> PipelineConfig:
        """Build a PipelineConfig for one file, applying its saved overrides."""
        values = dict(base)
        if overrides:
            values.update(overrides)
        export_format = values["export_format"]
        output_formats = [export_format, *values["output_formats_extra"]]
        save_dir = values["output_dir"]
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                self.log_msg(tr("info_dir_created", dir=save_dir))
            except Exception as e:
                self._show_error(tr("err_dir_create"), str(e))
                raise ValueError(f"cannot create save dir: {save_dir}") from e
        elif not os.path.isdir(save_dir):
            self._show_error(tr("err_cannot_start"), tr("err_dir_not_dir"))
            raise ValueError(f"save path is not a directory: {save_dir}")

        return PipelineConfig(
            audio_path="",  # set per-file in worker
            output_filename="",  # set per-file in worker
            output_dir=pathlib.Path(save_dir),
            game_model_dir=self.model_config.model_path("game_model"),
            hfa_model_dir=self.model_config.model_path("hfa_model"),
            asr_model_path=self.model_config.model_path("asr_model"),
            device=normalize_runtime_device(values["device"]),
            language=values["language"],
            ts=ts_list,
            lyric_output_mode=values["lyric_output"],
            original_lyrics=values["original_lyrics"] if values["match_lyrics"] else "",
            output_formats=output_formats,
            output_lyrics=values["output_lyrics"],
            output_pitch_curve=values["pitch_curve"] if export_format in {"ustx", "vsqx"} else False,
            slicing_method=values["slicing_method"],
            slice_min_sec=values["slice_min_sec"],
            slice_max_sec=values["slice_max_sec"],
            tempo=float(values["tempo"]),
            quantization_step=values["quantization_step"],
            quantization_mode=values["quantization_mode"],
            pitch_format=values["pitch_format"],
            round_pitch=values["round_pitch"],
            seg_threshold=values["seg_threshold"],
            seg_radius=values["seg_radius"],
            est_threshold=values["est_threshold"],
            batch_size=int(values["batch_size"]),
            asr_batch_size=int(values["asr_batch_size"]),
            rmvpe_model_path=self.model_config.model_path("rmvpe_model"),
            phoneme_asr_model_path=self.model_config.model_path("phoneme_asr_model"),
            pinyin_asr_model_path=self.model_config.model_path("pinyin_asr_model"),
            chinese_asr_engine=values.get("chinese_asr_engine", "qwen"),
            japanese_asr_engine=values.get("japanese_asr_engine", "romaji"),
        )

    def run_pipeline(self):
        if not HYBRID_AVAILABLE:
            self.log_msg(tr("err_hybrid"))
            return

        # Batch mode runs the current selection; otherwise every file in the list.
        audio_files = self.audio_list.selected_paths() or self.audio_list.all_paths()
        if not audio_files:
            self._show_error(tr("err_cannot_start"), tr("err_no_audio"))
            return

        base = self._current_base_values()
        ts_list = t0_nstep_to_ts(
            self.global_settings.t0(),
            self.global_settings.nsteps(),
        )
        try:
            validate_slice_bounds(base["slice_min_sec"], base["slice_max_sec"])
        except ValueError as exc:
            self.log_msg(f"Error: invalid slice duration settings: {exc}")
            return

        tasks: list[tuple[PipelineConfig, str]] = []
        try:
            for filename in audio_files:
                config = self._build_file_config(
                    filename, base, ts_list, self.audio_list.file_settings.get(filename)
                )
                tasks.append((config, filename))
        except ValueError:
            return  # error already reported

        self.log_edit.clear()
        self._set_running_ui(True)
        self.run_status_label.setText(tr("preparing", n=len(tasks)))
        self.worker = WorkerThread(tasks)
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

    # ── worker lifecycle accessors (used by the main window) ────────
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def request_stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()

    def wait_for_worker(self, timeout_ms: int) -> bool:
        """True when the worker thread finished within the timeout."""
        return self.worker is not None and self.worker.wait(timeout_ms)

    def on_worker_settled(self, callback) -> None:
        """Invoke callback once the current worker thread has finished."""
        if self.worker is not None:
            self.worker.finished.connect(callback)

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
