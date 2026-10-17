from __future__ import annotations

import enum
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "BodyLabel",
    "CardWidget",
    "ComboBox",
    "DoubleSpinBox",
    "FluentIcon",
    "FluentWindow",
    "IconWidget",
    "IndeterminateProgressRing",
    "InfoBar",
    "InfoBarPosition",
    "LineEdit",
    "ListWidget",
    "MessageBoxBase",
    "NavigationItemPosition",
    "PrimaryPushButton",
    "PushButton",
    "ScrollArea",
    "SpinBox",
    "SubtitleLabel",
    "SwitchButton",
    "TextEdit",
    "Theme",
    "TransparentToolButton",
    "isDarkTheme",
    "qconfig",
    "setTheme",
]

_ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"

# Control height the GUI was laid out with (the original library pins text
# inputs and spin boxes to this; the height lives in code, not in QSS).
_CONTROL_HEIGHT = 33

# Accent palette, matching the Fluent theme colour the GUI shipped with.
_ACCENT = "#009faa"
_ACCENT_HOVER = "#00a7b3"
_ACCENT_PRESSED = "#3eabb3"

# UI font stack, matching the original library's default families and order.
_FONT_FAMILIES = ("Segoe UI", "Microsoft YaHei", "PingFang SC")
_FONT_CACHE: QFont | None = None


def _base_font() -> QFont:
    global _FONT_CACHE
    if _FONT_CACHE is None:
        font = QFont()
        font.setFamilies(list(_FONT_FAMILIES))
        font.setPixelSize(14)
        _FONT_CACHE = font
    return QFont(_FONT_CACHE)


def _rgba(r: int, g: int, b: int, a: float) -> str:
    return f"rgba({r}, {g}, {b}, {a:.4f})"


def _tokens() -> dict:
    if _resolve_dark():
        return {
            "window": "#202020",
            "text": "#ffffff",
            "control_bg": _rgba(255, 255, 255, 0.0605),
            "control_hover": _rgba(255, 255, 255, 0.0837),
            "control_pressed": _rgba(255, 255, 255, 0.0326),
            # Text inputs (LineEdit/TextEdit/SpinBox) draw a strong bottom
            # edge; combo boxes use a milder pair with the emphasised edge on
            # top in dark mode, so the two sets are kept separate.
            "field_border": _rgba(255, 255, 255, 0.08),
            "field_border_bottom": _rgba(255, 255, 255, 0.5442),
            "combo_border": _rgba(255, 255, 255, 0.053),
            "combo_edge": _rgba(255, 255, 255, 0.08),
            "combo_edge_side": "top",
            "control_disabled_bg": _rgba(255, 255, 255, 0.0419),
            "text_secondary": _rgba(255, 255, 255, 0.786),
            "text_disabled": _rgba(255, 255, 255, 0.3628),
            "text_placeholder": _rgba(255, 255, 255, 0.6063),
            "card_bg": _rgba(255, 255, 255, 0.0512),
            "card_border": _rgba(255, 255, 255, 0.0706),
            "hover_overlay": _rgba(255, 255, 255, 0.0392),
            "accent_text": "#000000",
            "infobar": {
                "info": (39, 39, 39),
                "success": (57, 61, 27),
                "warning": (67, 53, 25),
                "error": (68, 39, 38),
            },
            "infobar_border": "rgb(29, 29, 29)",
        }
    return {
        "window": "#efefef",
        "text": "#000000",
        "control_bg": _rgba(255, 255, 255, 0.7),
        "control_hover": _rgba(249, 249, 249, 0.5),
        "control_pressed": _rgba(249, 249, 249, 0.3),
        "field_border": "rgba(0, 0, 0, 13)",
        "field_border_bottom": "rgba(0, 0, 0, 100)",
        "combo_border": _rgba(0, 0, 0, 0.073),
        "combo_edge": _rgba(0, 0, 0, 0.183),
        "combo_edge_side": "bottom",
        "control_disabled_bg": _rgba(249, 249, 249, 0.3),
        "text_secondary": _rgba(0, 0, 0, 0.6063),
        "text_disabled": _rgba(0, 0, 0, 0.3612),
        "text_placeholder": _rgba(0, 0, 0, 0.6063),
        "card_bg": _rgba(255, 255, 255, 0.6667),
        "card_border": _rgba(0, 0, 0, 0.0588),
        "hover_overlay": _rgba(0, 0, 0, 0.0353),
        "accent_text": "#ffffff",
        "infobar": {
            "info": (244, 244, 244),
            "success": (223, 246, 221),
            "warning": (255, 244, 206),
            "error": (253, 231, 233),
        },
        "infobar_border": "rgb(229, 229, 229)",
    }


class Theme(enum.Enum):

    LIGHT = "Light"
    DARK = "Dark"
    AUTO = "Auto"


class _Config(QObject):

    themeChangedFinished = Signal()


qconfig = _Config()

_current_theme = Theme.LIGHT


def _resolve_dark() -> bool:
    if _current_theme is Theme.DARK:
        return True
    if _current_theme is Theme.LIGHT:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def isDarkTheme() -> bool:
    return _resolve_dark()


def _light_palette() -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#efefef"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#f7f7f7"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#efefef"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(_ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffdc"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(0, 0, 0, 128))
    pal.setColor(QPalette.ColorRole.Mid, QColor(0, 0, 0, 156))
    return pal


def _dark_palette() -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#202020"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#2b2b2b"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#272727"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#2b2b2b"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(_ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2b2b2b"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(255, 255, 255, 155))
    pal.setColor(QPalette.ColorRole.Mid, QColor(255, 255, 255, 201))
    return pal


def setTheme(theme: Theme) -> None:
    global _current_theme
    _current_theme = theme if isinstance(theme, Theme) else Theme.LIGHT

    app = QApplication.instance()
    if app is not None:
        app.setPalette(_dark_palette() if _resolve_dark() else _light_palette())
        app.setFont(_base_font())

    qconfig.themeChangedFinished.emit()


class FluentIcon(enum.Enum):

    MUSIC = "Music"
    SETTING = "Setting"
    FOLDER = "Folder"
    DEVELOPER_TOOLS = "DeveloperTools"
    DELETE = "Delete"
    SYNC = "Sync"
    PLAY = "Play"
    PAUSE = "Pause"

    def path(self, theme: Theme = Theme.AUTO) -> str:
        variant = "white" if _theme_is_dark(theme) else "black"
        return str(_ICON_DIR / f"{self.value}_{variant}.svg")

    def icon(self, theme: Theme = Theme.AUTO, color: QColor | None = None) -> QIcon:
        return QIcon(self.path(theme))


def _theme_is_dark(theme: Theme) -> bool:
    if theme is Theme.AUTO:
        return _resolve_dark()
    return theme is Theme.DARK


def toQIcon(icon) -> QIcon:
    if icon is None:
        return QIcon()
    if isinstance(icon, QIcon):
        return icon
    if isinstance(icon, FluentIcon):
        return icon.icon()
    if isinstance(icon, str):
        return QIcon(icon)
    if hasattr(icon, "icon"):
        return icon.icon()
    return QIcon()


class _CompatLabel(QLabel):

    _font_size = 14
    _font_weight = "normal"

    def __init__(self, text=None, parent=None):
        if isinstance(text, QWidget) and parent is None:
            parent, text = text, None
        super().__init__(parent)
        if text is not None:
            self.setText(text)
        # Only set the metrics here: call sites layer their own styleSheet
        # (e.g. bold headings) on top, and any colour set here would win
        # over the application palette they inherit.
        self.setStyleSheet(
            f"font-size: {self._font_size}px; font-weight: {self._font_weight};"
        )


class BodyLabel(_CompatLabel):
    pass


class SubtitleLabel(_CompatLabel):

    _font_size = 20
    _font_weight = "600"


class CardWidget(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("compatCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        t = _tokens()
        # The original draws no hard border, only a faint top/bottom edge on
        # top of the translucent fill.
        top = _rgba(255, 255, 255, 0.0706) if _resolve_dark() else _rgba(0, 0, 0, 0.0431)
        self.setStyleSheet(
            "CardWidget#compatCard {"
            f"background-color: {t['card_bg']};"
            f"border: 1px solid {top};"
            "border-radius: 5px;"
            "}"
        )


class ScrollArea(QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)

    def enableTransparentBackground(self):
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        view = self.widget()
        if view is not None:
            view.setStyleSheet("background: transparent;")


class ComboItem:

    def __init__(self, text: str, icon=None, userData=None, isEnabled: bool = True):
        self.text = text
        self.icon = icon
        self.userData = userData
        self.isEnabled = isEnabled


class ComboBox(QComboBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[ComboItem] = []
        self.setFont(_base_font())
        # The original combo is a flat button, so it sits slightly shorter
        # than the 33 px text fields.
        self.setFixedHeight(27)
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        t = _tokens()
        dark = _resolve_dark()
        popup_bg = "#2b2b2b" if dark else "#ffffff"
        popup_border = _rgba(255, 255, 255, 0.08) if dark else _rgba(0, 0, 0, 0.073)
        # A 10 px chevron sits at the right edge, as in the original; the
        # native indicator arrow is suppressed so only ours is drawn.
        chevron = str(_ICON_DIR / ("chevron_down_white.svg" if dark else "chevron_down_black.svg")).replace("\\", "/")
        self.setStyleSheet(
            "QComboBox {"
            f" color: {t['text']};"
            f" background-color: {t['control_bg']};"
            f" border: 1px solid {t['combo_border']};"
            f" border-{t['combo_edge_side']}: 1px solid {t['combo_edge']};"
            " border-radius: 5px;"
            " padding: 5px 31px 6px 11px;"
            " text-align: left;"
            "}"
            f"QComboBox:hover {{ background-color: {t['control_hover']}; }}"
            f"QComboBox:disabled {{ color: {t['text_disabled']};"
            f" background-color: {t['control_disabled_bg']}; }}"
            "QComboBox::drop-down { border: none; width: 26px; }"
            "QComboBox::down-arrow {"
            f" image: url({chevron}); width: 10px; height: 10px;"
            "}"
            "QComboBox QAbstractItemView {"
            f" background-color: {popup_bg};"
            f" border: 1px solid {popup_border};"
            f" color: {t['text']};"
            " border-radius: 6px; outline: none;"
            " padding: 4px;"
            f" selection-background-color: {t['hover_overlay']};"
            f" selection-color: {t['text']};"
            "}"
        )

    def addItem(self, text, icon=None, userData=None):
        self.items.append(ComboItem(text, icon, userData))
        super().addItem(text)
        index = super().count() - 1
        if icon is not None:
            super().setItemIcon(index, toQIcon(icon))
        if userData is not None:
            super().setItemData(index, userData, Qt.ItemDataRole.UserRole)

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def clear(self):
        self.items.clear()
        super().clear()

    def setItemEnabled(self, index: int, isEnabled: bool):
        if 0 <= index < len(self.items):
            self.items[index].isEnabled = isEnabled
        model = self.model()
        if model is not None:
            item = model.item(index)
            if item is not None:
                item.setEnabled(isEnabled)

    def setItemText(self, index: int, text: str):
        if 0 <= index < len(self.items):
            self.items[index].text = text
        super().setItemText(index, text)


def _field_qss(selector: str) -> str:
    t = _tokens()
    return (
        f"{selector} {{"
        f" color: {t['text']};"
        f" background-color: {t['control_bg']};"
        f" border: 1px solid {t['field_border']};"
        f" border-bottom: 1px solid {t['field_border_bottom']};"
        " border-radius: 5px;"
        f" selection-background-color: {_ACCENT};"
        f" selection-color: {t['accent_text']};"
        "}"
        f"{selector}:hover {{ background-color: {t['control_hover']}; }}"
        f"{selector}:focus {{ background-color: {t['control_hover']}; }}"
        f"{selector}:disabled {{"
        f" color: {t['text_disabled']};"
        f" background-color: {t['control_disabled_bg']};"
        "}"
    )


class _FieldMixin:

    _field_selector = ""
    _field_padding = "0px 10px"

    def _init_field(self, padding: str):
        self._field_padding = padding
        # Explicit 14 px font: relying on the application font lets some
        # controls measure 2 px taller than the original.
        self.setFont(_base_font())
        self._restyle_field()
        qconfig.themeChangedFinished.connect(self._restyle_field)

    def _restyle_field(self):
        # Padding is not part of the colour tokens, so it is re-applied here
        # rather than living in _field_qss.
        self.setStyleSheet(
            f"{self._field_selector} {{ padding: {self._field_padding}; }}"
            + _field_qss(self._field_selector)
        )


class _SpinArrowButton(QToolButton):

    def __init__(self, direction: str, parent=None):
        super().__init__(parent)
        self._direction = direction
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoRaise(True)
        self.setIconSize(QSize(10, 10))
        self.setFixedSize(31, 23)
        self._refresh_icon()
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _refresh_icon(self):
        variant = "white" if _resolve_dark() else "black"
        self.setIcon(QIcon(str(_ICON_DIR / f"chevron_{self._direction}_{variant}.svg")))

    def _restyle(self):
        self._refresh_icon()
        dark = _resolve_dark()
        hover = _rgba(255, 255, 255, 0.0353) if dark else _rgba(0, 0, 0, 0.0353)
        pressed = _rgba(255, 255, 255, 0.0235) if dark else _rgba(0, 0, 0, 0.0235)
        self.setStyleSheet(
            "QToolButton { background-color: transparent; border: none;"
            " border-radius: 4px; margin: 0; }"
            f"QToolButton:hover {{ background-color: {hover}; }}"
            f"QToolButton:pressed {{ background-color: {pressed}; }}"
            f"QToolButton:disabled {{ opacity: 0.36; }}"
        )


class _SpinBoxMixin:

    def _init_spin_arrows(self):
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFixedHeight(_CONTROL_HEIGHT)

        self.upButton = _SpinArrowButton("up", self)
        self.downButton = _SpinArrowButton("down", self)
        self.upButton.clicked.connect(self.stepUp)
        self.downButton.clicked.connect(self.stepDown)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin, spacing = 4, 5
        h = self.height()
        y = (h - 23) // 2
        right = self.width() - margin
        self.downButton.move(right - self.downButton.width(), y)
        self.upButton.move(right - self.downButton.width() - spacing - self.upButton.width(), y)


class LineEdit(_FieldMixin, QLineEdit):

    _field_selector = "QLineEdit"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_CONTROL_HEIGHT)
        self._init_field("0px 10px")


class ListWidget(QListWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        t = _tokens()
        self.setStyleSheet(
            "QListWidget {"
            " background: transparent; border: none; outline: none;"
            f" selection-background-color: {t['hover_overlay']};"
            f" color: {t['text']};"
            "}"
        )


class TextEdit(_FieldMixin, QTextEdit):

    _field_selector = "QTextEdit"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_field("2px 3px 2px 8px")


class SpinBox(_SpinBoxMixin, _FieldMixin, QSpinBox):

    _field_selector = "QSpinBox"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_field("0px 74px 0px 10px")
        self._init_spin_arrows()


class DoubleSpinBox(_SpinBoxMixin, _FieldMixin, QDoubleSpinBox):

    _field_selector = "QDoubleSpinBox"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_field("0px 74px 0px 10px")
        self._init_spin_arrows()


class SwitchButton(QWidget):

    checkedChanged = Signal(bool)

    def __init__(self, text: str = "On", parent=None):
        if isinstance(text, QWidget) and parent is None:
            parent, text = text, "On"
        super().__init__(parent)
        self._onText = "On"
        self._offText = text
        self._text = text

        self._indicator = _SwitchIndicator(self)
        self._label = QLabel(self._text, self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._indicator)
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._indicator.toggled.connect(self._on_toggled)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def isChecked(self) -> bool:
        return self._indicator.isChecked()

    def setChecked(self, checked: bool):
        if bool(checked) != self._indicator.isChecked():
            self._indicator.setChecked(bool(checked))

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        self._text = text
        self._label.setText(text)
        self.adjustSize()

    def getOnText(self) -> str:
        return self._onText

    def setOnText(self, text: str):
        self._onText = text
        self._refresh_text()

    def getOffText(self) -> str:
        return self._offText

    def setOffText(self, text: str):
        self._offText = text
        self._refresh_text()

    def _refresh_text(self):
        # Only the built-in On/Off captions follow the state; a custom text
        # (as passed to the constructor) stays fixed.
        if self._text in (self._onText, self._offText):
            self.setText(self._onText if self.isChecked() else self._offText)

    def _on_toggled(self, checked: bool):
        if self._text in (self._onText, self._offText):
            self._label.setText(self._onText if checked else self._offText)
        self.checkedChanged.emit(checked)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._indicator.setEnabled(enabled)
        self._label.setEnabled(enabled)

    onText = Property(str, getOnText, setOnText)
    offText = Property(str, getOffText, setOffText)


class _SwitchIndicator(QWidget):

    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(42, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._checked = False
        self._hover = False
        self._knob_x = 5.0
        self._anim = QPropertyAnimation(self, b"knobX", self)
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def getKnobX(self) -> float:
        return self._knob_x

    def setKnobX(self, value: float):
        self._knob_x = value
        self.update()

    knobX = Property(float, getKnobX, setKnobX)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, animate: bool = True):
        if checked == self._checked:
            return
        self._checked = checked
        self._animate(animate)
        self.toggled.emit(checked)

    def toggle(self):
        self.setChecked(not self._checked)

    def _animate(self, animate: bool):
        target = 25.0 if self._checked else 5.0
        if animate:
            self._anim.stop()
            self._anim.setStartValue(self._knob_x)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self.setKnobX(target)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.toggle()
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        dark = _resolve_dark()
        enabled = self.isEnabled()

        if self._checked:
            track = QColor(_ACCENT)
            if not enabled:
                track = QColor(255, 255, 255, 41) if dark else QColor(0, 0, 0, 56)
            elif self._hover:
                track = QColor(_ACCENT_HOVER)
        else:
            if not enabled:
                track = QColor(0, 0, 0, 0)
            elif self._hover:
                track = QColor(255, 255, 255, 10) if dark else QColor(0, 0, 0, 15)
            else:
                track = QColor(0, 0, 0, 0)

        # Unchecked and enabled draws only an outline, as in the original.
        if not self._checked and enabled:
            border = QColor(255, 255, 255, 153) if dark else QColor(0, 0, 0, 133)
            pen = QPen(border)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(track)

        radius = self.height() / 2
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)

        if self._checked:
            knob = QColor(Qt.GlobalColor.black if dark else Qt.GlobalColor.white) if enabled \
                else (QColor(255, 255, 255, 77) if dark else QColor(255, 255, 255))
        else:
            knob = (QColor(255, 255, 255, 201) if dark else QColor(0, 0, 0, 156)) if enabled \
                else (QColor(255, 255, 255, 96) if dark else QColor(0, 0, 0, 91))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(knob)
        painter.drawEllipse(int(self._knob_x), 5, 12, 12)


def _apply_button_style(button: QPushButton, primary: bool):
    t = _tokens()
    if primary:
        # The accent is a light teal, so the label flips to black on dark.
        disabled_bg = "rgb(52, 52, 52)" if _resolve_dark() else "rgb(205, 205, 205)"
        button.setStyleSheet(
            "QPushButton {"
            f" color: {t['accent_text']};"
            f" background-color: {_ACCENT};"
            f" border: 1px solid {_ACCENT_HOVER};"
            f" border-bottom: 1px solid {_ACCENT_HOVER};"
            " border-radius: 5px; padding: 5px 12px 6px 12px; outline: none;"
            "}"
            f"QPushButton:hover {{ background-color: {_ACCENT_HOVER}; }}"
            f"QPushButton:pressed {{ background-color: {_ACCENT_PRESSED}; }}"
            "QPushButton:disabled {"
            f" color: rgba(255, 255, 255, 0.9); background-color: {disabled_bg};"
            f" border: 1px solid {disabled_bg};"
            "}"
        )
    else:
        button.setStyleSheet(
            "QPushButton {"
            f" color: {t['text']};"
            f" background-color: {t['control_bg']};"
            f" border: 1px solid {t['combo_border']};"
            f" border-bottom: 1px solid {t['combo_edge']};"
            " border-radius: 5px; padding: 5px 12px 6px 12px; outline: none;"
            "}"
            f"QPushButton:hover {{ background-color: {t['control_hover']}; }}"
            "QPushButton:pressed {"
            f" color: {t['text_secondary']};"
            f" background-color: {t['control_pressed']};"
            f" border-bottom: 1px solid {t['combo_border']};"
            "}"
            f"QPushButton:disabled {{ color: {t['text_disabled']};"
            f" background-color: {t['control_disabled_bg']}; }}"
        )


class PushButton(QPushButton):

    def __init__(self, text=None, parent=None, icon=None):
        if isinstance(text, QWidget) and parent is None:
            parent, text = text, ""
        super().__init__(parent)
        self.setIconSize(QSize(16, 16))
        if isinstance(text, FluentIcon):
            icon, text = text, ""
        if text:
            self.setText(text)
        if icon is not None:
            self.setIcon(toQIcon(icon))
        self._primary = False
        _apply_button_style(self, primary=False)
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        _apply_button_style(self, primary=self._primary)

    def setIcon(self, icon):
        super().setIcon(toQIcon(icon))


class PrimaryPushButton(PushButton):

    def __init__(self, text=None, parent=None, icon=None):
        super().__init__(text, parent, icon)
        self._primary = True
        _apply_button_style(self, primary=True)


class TransparentToolButton(QToolButton):

    def __init__(self, icon=None, parent=None):
        if isinstance(icon, QWidget) and parent is None:
            parent, icon = icon, None
        super().__init__(parent)
        self.setIconSize(QSize(16, 16))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if icon is not None:
            self.setIcon(toQIcon(icon))
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        t = _tokens()
        self.setStyleSheet(
            "QToolButton { border: none; background: transparent; border-radius: 5px; }"
            f"QToolButton:hover {{ background-color: {t['hover_overlay']}; }}"
            f"QToolButton:pressed {{ background-color: {t['control_pressed']}; }}"
        )

    def setIcon(self, icon):
        super().setIcon(toQIcon(icon))


class IconWidget(QWidget):

    def __init__(self, icon=None, parent=None):
        if isinstance(icon, QWidget) and parent is None:
            parent, icon = icon, None
        super().__init__(parent)
        self._icon = toQIcon(icon)
        qconfig.themeChangedFinished.connect(self.update)

    def setIcon(self, icon):
        self._icon = toQIcon(icon)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        size = self.size()
        if not self._icon.isNull() and size.width() > 0 and size.height() > 0:
            painter.drawPixmap(0, 0, self._icon.pixmap(size))


class IndeterminateProgressRing(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance)

    def _advance(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def showEvent(self, event):
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        margin = 3
        rect = QRectF(margin, margin, side - 2 * margin, side - 2 * margin)

        track_pen = QPen(QColor(128, 128, 128, 60))
        track_pen.setWidth(3)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0, 360 * 16)

        arc_pen = QPen(QColor(_ACCENT))
        arc_pen.setWidth(3)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        painter.drawArc(rect, -self._angle * 16, 90 * 16)


class InfoBarPosition(enum.Enum):

    TOP = 0
    BOTTOM = 1
    TOP_LEFT = 2
    TOP_RIGHT = 3
    BOTTOM_LEFT = 4
    BOTTOM_RIGHT = 5
    NONE = 6


class InfoBar(QFrame):

    _ICON_COLORS = {
        "info": {"light": "#00b7c3", "dark": "#29f7ff", "char": "\u2139"},
        "success": {"light": "#0f7b0f", "dark": "#6ccb5f", "char": "\u2714"},
        "warning": {"light": "#9d5d00", "dark": "#fce100", "char": "\u26a0"},
        "error": {"light": "#c42b1c", "dark": "#ff99a4", "char": "\u2715"},
    }

    def __init__(self, kind: str, title: str, content: str, *, orient=Qt.Orientation.Horizontal,
                 is_closable: bool = True, duration: int = 1000,
                 position: InfoBarPosition = InfoBarPosition.TOP_RIGHT, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._duration = duration
        self._position = position
        self.setObjectName("compatInfoBar")
        self._build(title or "", content or "", is_closable)
        self._style()

    def _build(self, title: str, content: str, is_closable: bool):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 12, 12)
        layout.setSpacing(12)

        icon_spec = self._ICON_COLORS.get(self._kind, self._ICON_COLORS["info"])
        self._icon = QLabel(icon_spec["char"], self)
        self._icon.setStyleSheet("font-size: 16px; background: transparent;")
        self._icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        if title:
            title_label = QLabel(title, self)
            title_label.setStyleSheet("font-weight: bold; background: transparent;")
            text_box.addWidget(title_label)
        if content:
            content_label = QLabel(content, self)
            content_label.setWordWrap(True)
            content_label.setStyleSheet("background: transparent;")
            text_box.addWidget(content_label)
        layout.addLayout(text_box, 1)

        if is_closable:
            close_btn = QToolButton(self)
            close_btn.setText("\u2715")
            close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            close_btn.setStyleSheet(
                "QToolButton { border: none; background: transparent; border-radius: 5px; }"
            )
            close_btn.clicked.connect(self._dismiss)
            layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)

    def _style(self):
        t = _tokens()
        r, g, b = t["infobar"].get(self._kind, t["infobar"]["info"])
        icon_spec = self._ICON_COLORS.get(self._kind, self._ICON_COLORS["info"])
        icon_color = icon_spec["dark"] if _resolve_dark() else icon_spec["light"]
        self._icon.setStyleSheet(
            f"font-size: 16px; color: {icon_color}; background: transparent;"
        )
        self.setStyleSheet(
            "InfoBar#compatInfoBar {"
            f"background-color: rgb({r}, {g}, {b});"
            f"border: 1px solid {t['infobar_border']};"
            "border-radius: 6px;"
            f"color: {t['text']};"
            "}"
            "InfoBar#compatInfoBar QLabel {"
            f"color: {t['text']};"
            "}"
        )

    def _dismiss(self):
        self.hide()
        self.deleteLater()

    def show_over_parent(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        width = min(400, max(240, parent.width() - 48))
        self.setFixedWidth(width)
        self.adjustSize()

        top = self._position in (
            InfoBarPosition.TOP,
            InfoBarPosition.TOP_LEFT,
            InfoBarPosition.TOP_RIGHT,
        )
        y = 24 if top else max(24, parent.height() - self.height() - 24)
        if self._position in (InfoBarPosition.TOP_LEFT, InfoBarPosition.BOTTOM_LEFT):
            x = 24
        elif self._position in (InfoBarPosition.TOP_RIGHT, InfoBarPosition.BOTTOM_RIGHT):
            x = max(24, parent.width() - width - 24)
        else:
            x = max(24, (parent.width() - width) // 2)

        self.move(x, y)
        self.show()
        self.raise_()

        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        fade = QPropertyAnimation(effect, b"opacity", self)
        fade.setDuration(180)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self._fade = fade

        if self._duration and self._duration > 0:
            QTimer.singleShot(self._duration, self._dismiss)

    @classmethod
    def _create(cls, kind, title, content, orient, is_closable, duration, position, parent):
        bar = cls(
            kind, title, content, orient=orient, is_closable=is_closable,
            duration=duration, position=position, parent=parent,
        )
        bar.show_over_parent()
        return bar

    @classmethod
    def info(cls, title, content, orient=Qt.Orientation.Horizontal, isClosable=True,
             duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        return cls._create("info", title, content, orient, isClosable, duration, position, parent)

    @classmethod
    def success(cls, title, content, orient=Qt.Orientation.Horizontal, isClosable=True,
                duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        return cls._create("success", title, content, orient, isClosable, duration, position, parent)

    @classmethod
    def warning(cls, title, content, orient=Qt.Orientation.Horizontal, isClosable=True,
                duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        return cls._create("warning", title, content, orient, isClosable, duration, position, parent)

    @classmethod
    def error(cls, title, content, orient=Qt.Orientation.Horizontal, isClosable=True,
              duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        return cls._create("error", title, content, orient, isClosable, duration, position, parent)


class MessageBoxBase(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setObjectName("compatMessageBox")

        self.widget = QWidget(self)
        self.widget.setObjectName("compatDialogBody")
        self.viewLayout = QVBoxLayout()
        self.viewLayout.setContentsMargins(24, 24, 24, 24)
        self.viewLayout.setSpacing(12)

        self.buttonGroup = QFrame(self.widget)
        self.buttonGroup.setObjectName("compatButtonGroup")
        self.yesButton = PrimaryPushButton(self.tr("OK"), self.buttonGroup)
        self.cancelButton = PushButton(self.tr("Cancel"), self.buttonGroup)

        button_layout = QHBoxLayout(self.buttonGroup)
        button_layout.setContentsMargins(24, 16, 24, 16)
        button_layout.setSpacing(12)
        button_layout.addWidget(self.yesButton, 1)
        button_layout.addWidget(self.cancelButton, 1)

        outer = QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(self.viewLayout, 1)
        outer.addWidget(self.buttonGroup, 0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.widget)

        self.yesButton.clicked.connect(self._on_yes)
        self.cancelButton.clicked.connect(self.reject)
        self.setMinimumWidth(560)
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        dark = _resolve_dark()
        body = "#272727" if dark else "#ffffff"
        footer = "#202020" if dark else "rgb(243, 243, 243)"
        divider = "#3a3a3a" if dark else "rgb(229, 229, 229)"
        self.setStyleSheet(
            "QDialog#compatMessageBox { background-color: %s; }" % body
            + "QWidget#compatDialogBody { background-color: %s; }" % body
            + "QFrame#compatButtonGroup { background-color: %s;"
              " border-top: 1px solid %s; border-left: none;"
              " border-right: none; border-bottom: none; }" % (footer, divider)
        )

    def _on_yes(self):
        if self.validate():
            self.accept()

    def validate(self) -> bool:
        return True


class NavigationItemPosition(enum.Enum):

    TOP = 0
    SCROLL_TOP = 1
    BOTTOM = 2


class _NavItemHandle:

    def __init__(self, navigation, name: str):
        self._navigation = navigation
        self._name = name

    def setText(self, text: str):
        self._navigation._set_item_text(self._name, text)


class NavigationInterface(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, QListWidgetItem] = {}
        self._labels: dict[str, str] = {}
        self._icon_only = False
        self._menu_button_visible = True

        self._list = QListWidget(self)
        self._list.setObjectName("compatNavigation")
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setFixedWidth(50)
        self._list.setIconSize(QSize(18, 18))
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)

    def _restyle(self):
        t = _tokens()
        self.setStyleSheet(
            "QListWidget#compatNavigation {"
            " border: 0; outline: none; padding: 6px 4px; background: transparent;"
            "}"
            "QListWidget#compatNavigation::item {"
            " padding: 8px 6px; border-radius: 6px; background: transparent;"
            "}"
            "QListWidget#compatNavigation::item:hover {"
            f" background-color: {t['hover_overlay']};"
            "}"
            "QListWidget#compatNavigation::item:selected {"
            f" background-color: {t['control_hover']};"
            "}"
        )

    def add_item(self, name: str, icon, text: str, position: NavigationItemPosition):
        item = QListWidgetItem(toQIcon(icon), "" if self._icon_only else text)
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setSizeHint(QSize(40, 36))
        self._list.addItem(item)
        self._items[name] = item
        self._labels[name] = text
        item.setToolTip(text)
        if self._list.count() == 1:
            self._list.setCurrentRow(0)

    def _rebuild_texts(self):
        for name, item in self._items.items():
            label = self._labels.get(name, "")
            item.setText("" if self._icon_only else label)
            item.setToolTip(label)

    def _set_item_text(self, name: str, text: str):
        self._labels[name] = text
        item = self._items.get(name)
        if item is not None:
            item.setText("" if self._icon_only else text)
            item.setToolTip(text)

    def widget(self, name: str):
        return _NavItemHandle(self, name) if name in self._items else None

    def setMenuButtonVisible(self, visible: bool):
        # This shell has no hamburger button; the call is accepted for
        # compatibility with the previous widget library.
        self._menu_button_visible = bool(visible)

    def setExpandWidth(self, width: int):
        self._icon_only = width <= 60
        self._list.setFixedWidth(max(36, width))
        self._rebuild_texts()

    def setMinimumExpandWidth(self, width: int):
        self.setMinimumWidth(width)

    def setCurrentItem(self, name: str):
        item = self._items.get(name)
        if item is not None:
            self._list.setCurrentItem(item)

    def current_name(self):
        item = self._list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    @property
    def currentItemChanged(self):
        return self._list.currentItemChanged


class FluentWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._interfaces: dict[str, QWidget] = {}

        self.navigationInterface = NavigationInterface(self)
        self.stackedWidget = QStackedWidget(self)

        central = QWidget(self)
        central.setObjectName("compatCentral")
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.navigationInterface)
        layout.addWidget(self.stackedWidget, 1)
        self.setCentralWidget(central)

        self.navigationInterface.currentItemChanged.connect(self._on_nav_changed)
        self._restyle()
        qconfig.themeChangedFinished.connect(self._restyle)

    def _restyle(self):
        # Paint the window in the theme background so the sidebar's selection
        # highlight and the cards read against the same surface.
        t = _tokens()
        self.setStyleSheet(
            f"QWidget#compatCentral {{ background-color: {t['window']}; }}"
            f"QStackedWidget {{ background-color: {t['window']}; }}"
        )

    def addSubInterface(self, interface: QWidget, icon, text: str,
                        position: NavigationItemPosition = NavigationItemPosition.TOP):
        name = interface.objectName()
        self._interfaces[name] = interface
        self.stackedWidget.addWidget(interface)
        self.navigationInterface.add_item(name, icon, text, position)

    def _on_nav_changed(self, current, previous):
        name = self.navigationInterface.current_name()
        interface = self._interfaces.get(name) if name else None
        if interface is not None:
            self.stackedWidget.setCurrentWidget(interface)

    def setMicaEffectEnabled(self, enabled: bool):
        return None
