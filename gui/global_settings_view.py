import pathlib

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout

from application.config import (
    DEFAULT_SLICE_MAX_SEC,
    DEFAULT_SLICE_MIN_SEC,
    SLICE_DURATION_MAX_SEC,
    SLICE_DURATION_MIN_SEC,
    validate_slice_bounds,
)
from qfluentwidgets import (
    ScrollArea,
    PushButton,
    CardWidget,
    BodyLabel,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    SwitchButton,
    FluentIcon,
    SubtitleLabel,
    setTheme,
    Theme,
)
from gui.i18n import tr, set_language
from gui.settings_utils import create_app_settings

THEME_CHOICES = [("light", "theme_light"), ("dark", "theme_dark"), ("auto", "theme_auto")]
LANGUAGE_CHOICES = [("zh", "language_zh"), ("en", "language_en")]


class GlobalSettingsInterface(ScrollArea):
    languageChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.settings = create_app_settings(self.project_root)
        self._tr_bindings: list = []
        self.default_values = {
            "seg_thresh": 0.2,
            "seg_rad": 0.02,
            "est_thresh": 0.2,
            "t0": 0.0,
            "nsteps": 8,
            "batch_size": 1,
            "asr_batch": 2,
            "slice_min_sec": DEFAULT_SLICE_MIN_SEC,
            "slice_max_sec": DEFAULT_SLICE_MAX_SEC,
            "debug_txt": False,
            "debug_csv": False,
            "debug_chunks": False,
            "output_lyrics": True,
            "enable_lyrics_match": False,
            "pitch_format": "name",
            "round_pitch": True,
        }
        initial_slice_min = float(self.settings.value("slice_min_sec", self.default_values["slice_min_sec"]))
        initial_slice_max = float(self.settings.value("slice_max_sec", self.default_values["slice_max_sec"]))
        try:
            validate_slice_bounds(initial_slice_min, initial_slice_max)
        except ValueError:
            initial_slice_min = self.default_values["slice_min_sec"]
            initial_slice_max = self.default_values["slice_max_sec"]
        self._slice_bounds_updating = False
        self._last_valid_slice_bounds = (initial_slice_min, initial_slice_max)

        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName('view')
        self.setObjectName('globalSettingsInterface')

        title_layout = QHBoxLayout()
        title = SubtitleLabel(self)
        self._bind_tr(lambda: title.setText(tr("settings_title")))
        title_layout.addWidget(title)
        title_layout.addStretch(1)
        btn_reset = PushButton(tr("reset_defaults"), self, FluentIcon.SYNC)
        self._bind_tr(lambda: btn_reset.setText(tr("reset_defaults")))
        btn_reset.clicked.connect(self.reset_to_default)
        title_layout.addWidget(btn_reset)
        self.vBoxLayout.addLayout(title_layout)

        # ── appearance ──────────────────────────────────────────────
        appearance_card = CardWidget(self)
        appearance_layout = QVBoxLayout(appearance_card)
        appearance_title = BodyLabel(self)
        self._bind_tr(lambda: appearance_title.setText(tr("appearance")))
        appearance_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        appearance_layout.addWidget(appearance_title)

        appearance_grid = QGridLayout()
        appearance_grid.setHorizontalSpacing(20)
        appearance_grid.setVerticalSpacing(14)

        theme_label = BodyLabel(self)
        self._bind_tr(lambda: theme_label.setText(tr("theme")))
        self.theme_combo = ComboBox(self)
        self._fill_combo(self.theme_combo, THEME_CHOICES)
        self._bind_tr(lambda: self._fill_combo(self.theme_combo, THEME_CHOICES, keep_value=True))
        self.theme_combo.setCurrentIndex(self._index_by_value(self.theme_combo, self._saved_theme()))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        appearance_grid.addWidget(theme_label, 0, 0)
        appearance_grid.addWidget(self.theme_combo, 0, 1)

        language_label = BodyLabel(self)
        self._bind_tr(lambda: language_label.setText(tr("language")))
        self.language_combo = ComboBox(self)
        self._fill_combo(self.language_combo, LANGUAGE_CHOICES)
        self.language_combo.setCurrentIndex(self._index_by_value(self.language_combo, self._saved_language()))
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        appearance_grid.addWidget(language_label, 0, 2)
        appearance_grid.addWidget(self.language_combo, 0, 3)
        appearance_grid.setColumnStretch(4, 1)
        appearance_layout.addLayout(appearance_grid)
        self.vBoxLayout.addWidget(appearance_card)

        # ── advanced processing parameters ──────────────────────────
        adv_card = CardWidget(self)
        adv_layout = QVBoxLayout(adv_card)
        adv_title = BodyLabel(self)
        self._bind_tr(lambda: adv_title.setText(tr("adv_params")))
        adv_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        adv_layout.addWidget(adv_title)

        adv_grid = QGridLayout()
        adv_grid.setVerticalSpacing(15)
        adv_grid.setHorizontalSpacing(20)

        def add_pair(label_key, row, col, widget):
            label = BodyLabel(self)
            self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
            adv_grid.addWidget(label, row, col)
            adv_grid.addWidget(widget, row, col + 1)

        self.seg_thresh_spin = DoubleSpinBox(self)
        self.seg_thresh_spin.setRange(0.01, 0.99)
        self.seg_thresh_spin.setSingleStep(0.01)
        self.seg_thresh_spin.setValue(float(self.settings.value("seg_thresh", self.default_values["seg_thresh"])))
        self.seg_thresh_spin.valueChanged.connect(lambda v: self.settings.setValue("seg_thresh", v))
        add_pair("seg_thresh", 0, 0, self.seg_thresh_spin)

        self.seg_rad_spin = DoubleSpinBox(self)
        self.seg_rad_spin.setRange(0.01, 0.1)
        self.seg_rad_spin.setSingleStep(0.005)
        self.seg_rad_spin.setValue(float(self.settings.value("seg_rad", self.default_values["seg_rad"])))
        self.seg_rad_spin.valueChanged.connect(lambda v: self.settings.setValue("seg_rad", v))
        add_pair("seg_rad", 0, 2, self.seg_rad_spin)

        self.est_thresh_spin = DoubleSpinBox(self)
        self.est_thresh_spin.setRange(0.01, 0.99)
        self.est_thresh_spin.setSingleStep(0.01)
        self.est_thresh_spin.setValue(float(self.settings.value("est_thresh", self.default_values["est_thresh"])))
        self.est_thresh_spin.valueChanged.connect(lambda v: self.settings.setValue("est_thresh", v))
        add_pair("est_thresh", 0, 4, self.est_thresh_spin)

        self.t0_spin = DoubleSpinBox(self)
        self.t0_spin.setRange(0.0, 0.99)
        self.t0_spin.setSingleStep(0.01)
        self.t0_spin.setValue(float(self.settings.value("t0", self.default_values["t0"])))
        self.t0_spin.valueChanged.connect(lambda v: self.settings.setValue("t0", v))
        add_pair("d3pm_t0", 1, 0, self.t0_spin)

        self.nsteps_spin = SpinBox(self)
        self.nsteps_spin.setRange(1, 20)
        self.nsteps_spin.setValue(int(self.settings.value("nsteps", self.default_values["nsteps"])))
        self.nsteps_spin.valueChanged.connect(lambda v: self.settings.setValue("nsteps", v))
        add_pair("d3pm_nsteps", 1, 2, self.nsteps_spin)

        self.batch_spin = SpinBox(self)
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(int(self.settings.value("batch_size", self.default_values["batch_size"])))
        self.batch_spin.valueChanged.connect(lambda v: self.settings.setValue("batch_size", v))
        add_pair("game_batch", 1, 4, self.batch_spin)

        self.asr_batch_spin = SpinBox(self)
        self.asr_batch_spin.setRange(1, 32)
        self.asr_batch_spin.setValue(int(self.settings.value("asr_batch", self.default_values["asr_batch"])))
        self.asr_batch_spin.valueChanged.connect(lambda v: self.settings.setValue("asr_batch", v))
        add_pair("asr_batch", 2, 0, self.asr_batch_spin)

        self.slice_min_spin = DoubleSpinBox(self)
        self.slice_min_spin.setRange(SLICE_DURATION_MIN_SEC, SLICE_DURATION_MAX_SEC)
        self.slice_min_spin.setDecimals(1)
        self.slice_min_spin.setSingleStep(0.5)
        self.slice_min_spin.setValue(initial_slice_min)
        add_pair("slice_min", 2, 2, self.slice_min_spin)

        self.slice_max_spin = DoubleSpinBox(self)
        self.slice_max_spin.setRange(SLICE_DURATION_MIN_SEC, SLICE_DURATION_MAX_SEC)
        self.slice_max_spin.setDecimals(1)
        self.slice_max_spin.setSingleStep(0.5)
        self.slice_max_spin.setValue(initial_slice_max)
        add_pair("slice_max", 2, 4, self.slice_max_spin)

        self.slice_min_spin.valueChanged.connect(self._on_slice_bounds_changed)
        self.slice_max_spin.valueChanged.connect(self._on_slice_bounds_changed)
        self._store_slice_bounds(initial_slice_min, initial_slice_max)

        adv_grid.setColumnStretch(6, 1)
        adv_layout.addLayout(adv_grid)
        self.vBoxLayout.addWidget(adv_card)

        # ── debug ───────────────────────────────────────────────────
        debug_card = CardWidget(self)
        debug_layout = QVBoxLayout(debug_card)
        debug_title = BodyLabel(self)
        self._bind_tr(lambda: debug_title.setText(tr("debug")))
        debug_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        debug_layout.addWidget(debug_title)

        debug_grid = QHBoxLayout()
        self._add_debug_switch(debug_grid, "export_txt", "debug_txt", self.default_values["debug_txt"])
        debug_grid.addSpacing(20)
        self._add_debug_switch(debug_grid, "export_csv", "debug_csv", self.default_values["debug_csv"])
        debug_grid.addSpacing(20)
        self._add_debug_switch(debug_grid, "export_chunks", "debug_chunks", self.default_values["debug_chunks"])
        debug_grid.addSpacing(20)

        self.pitch_combo = ComboBox(self)
        self.pitch_combo.addItems(["name", "number"])
        self.pitch_combo.setCurrentText(self.settings.value("pitch_format", self.default_values["pitch_format"]))
        self.pitch_combo.currentTextChanged.connect(lambda t: self.settings.setValue("pitch_format", t))
        pitch_label = BodyLabel(self)
        self._bind_tr(lambda: pitch_label.setText(tr("pitch_format")))
        debug_grid.addWidget(pitch_label)
        debug_grid.addWidget(self.pitch_combo)
        debug_grid.addSpacing(20)

        self.cb_round = SwitchButton("On", self)
        self.cb_round.setOffText("Off")
        self.cb_round.setChecked(self.settings.value("round_pitch", self.default_values["round_pitch"], type=bool))
        self.cb_round.checkedChanged.connect(lambda v: self.settings.setValue("round_pitch", v))
        round_label = BodyLabel(self)
        self._bind_tr(lambda: round_label.setText(tr("round_pitch")))
        debug_grid.addWidget(round_label)
        debug_grid.addWidget(self.cb_round)
        debug_grid.addStretch(1)

        debug_layout.addLayout(debug_grid)
        self.vBoxLayout.addWidget(debug_card)

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

    # ── i18n helpers ────────────────────────────────────────────────
    def _bind_tr(self, fn):
        self._tr_bindings.append(fn)
        fn()

    def retranslate_ui(self):
        for fn in self._tr_bindings:
            fn()

    @staticmethod
    def _fill_combo(combo, choices: list[tuple[str, str]], keep_value: bool = False):
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

    @staticmethod
    def _index_by_value(combo, value: str) -> int:
        return max(0, combo.findData(value))

    def _saved_theme(self) -> str:
        return str(self.settings.value("theme", "light")).strip().lower()

    def _saved_language(self) -> str:
        return "en" if str(self.settings.value("language", "zh")).strip().lower() in {"en", "english"} else "zh"

    def _add_debug_switch(self, layout, label_key, settings_key, default):
        switch = SwitchButton("On", self)
        switch.setOffText("Off")
        switch.setChecked(self.settings.value(settings_key, default, type=bool))
        switch.checkedChanged.connect(lambda v, k=settings_key: self.settings.setValue(k, v))
        label = BodyLabel(self)
        self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
        layout.addWidget(label)
        layout.addWidget(switch)
        attr = {"debug_txt": "cb_txt", "debug_csv": "cb_csv", "debug_chunks": "cb_chunks"}[settings_key]
        setattr(self, attr, switch)

    def _on_theme_changed(self):
        value = self.theme_combo.currentData() or "light"
        self.settings.setValue("theme", value)
        setTheme({"dark": Theme.DARK, "auto": Theme.AUTO}.get(value, Theme.LIGHT))

    def _on_language_changed(self):
        value = self.language_combo.currentData() or "zh"
        self.settings.setValue("language", value)
        set_language(value)
        self.retranslate_ui()
        self.languageChanged.emit(value)

    # ── existing behaviour ──────────────────────────────────────────
    def reset_to_default(self):
        self.seg_thresh_spin.setValue(self.default_values["seg_thresh"])
        self.seg_rad_spin.setValue(self.default_values["seg_rad"])
        self.est_thresh_spin.setValue(self.default_values["est_thresh"])
        self.t0_spin.setValue(self.default_values["t0"])
        self.nsteps_spin.setValue(self.default_values["nsteps"])
        self.batch_spin.setValue(self.default_values["batch_size"])
        self.asr_batch_spin.setValue(self.default_values["asr_batch"])
        self._set_slice_bounds(self.default_values["slice_min_sec"], self.default_values["slice_max_sec"])
        self._store_slice_bounds(self.default_values["slice_min_sec"], self.default_values["slice_max_sec"])
        self.cb_debug_txt.setChecked(self.default_values["debug_txt"])
        self.cb_debug_csv.setChecked(self.default_values["debug_csv"])
        self.cb_debug_chunks.setChecked(self.default_values["debug_chunks"])
        self.pitch_combo.setCurrentText(self.default_values["pitch_format"])
        self.cb_round.setChecked(self.default_values["round_pitch"])
        # Note: enable_lyrics_match / output_lyrics live in AutoLyricInterface;
        # this view deliberately does not reset them (UI would desync).

    def get_slice_bounds(self) -> tuple[float, float]:
        slice_min_sec = float(self.slice_min_spin.value())
        slice_max_sec = float(self.slice_max_spin.value())
        validate_slice_bounds(slice_min_sec, slice_max_sec)
        return slice_min_sec, slice_max_sec

    def _on_slice_bounds_changed(self, _value: float) -> None:
        if self._slice_bounds_updating:
            return

        slice_min_sec = float(self.slice_min_spin.value())
        slice_max_sec = float(self.slice_max_spin.value())
        try:
            validate_slice_bounds(slice_min_sec, slice_max_sec)
        except ValueError:
            self._set_slice_bounds(*self._last_valid_slice_bounds)
            return

        self._store_slice_bounds(slice_min_sec, slice_max_sec)

    def _set_slice_bounds(self, slice_min_sec: float, slice_max_sec: float) -> None:
        self._slice_bounds_updating = True
        try:
            self.slice_min_spin.setValue(slice_min_sec)
            self.slice_max_spin.setValue(slice_max_sec)
        finally:
            self._slice_bounds_updating = False

    def _store_slice_bounds(self, slice_min_sec: float, slice_max_sec: float) -> None:
        self._last_valid_slice_bounds = (slice_min_sec, slice_max_sec)
        self.settings.setValue("slice_min_sec", slice_min_sec)
        self.settings.setValue("slice_max_sec", slice_max_sec)
