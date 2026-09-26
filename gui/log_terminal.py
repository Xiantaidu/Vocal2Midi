"""Log terminal widget: monospaced output colored by severity, theme-aware."""
import re

from qfluentwidgets import TextEdit, isDarkTheme, qconfig


class LogTerminal(TextEdit):
    """Read-only log view that colors lines by severity and follows the theme."""

    # key: True for the dark theme; the light palette uses darker shades that
    # keep sufficient contrast on a light background
    _PALETTES = {
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

    _COLOR_RULES = [
        ("error", re.compile(r"错误|error|traceback|failed|exception|失败", re.I)),
        ("warn", re.compile(r"警告|warning|取消|停止|跳过|重试|retry", re.I)),
        ("success", re.compile(r"成功|完成|finished|done", re.I)),
    ]

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setMinimumHeight(150)
        self.setObjectName("logTerminal")
        self._lines: list[str] = []
        self._apply_style()
        # themeChangedFinished fires after qfluentwidgets reapplies its widget
        # stylesheets; otherwise our terminal style gets overwritten by the
        # library's TextEdit stylesheet
        qconfig.themeChangedFinished.connect(self._on_theme_changed)

    def log(self, msg: str):
        self._lines.append(msg)
        self.append(self._line_html(msg))
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        self._lines.clear()
        super().clear()

    def _palette(self):
        return self._PALETTES[isDarkTheme()]

    def _apply_style(self):
        p = self._palette()
        self.setStyleSheet(
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
        import html

        p = self._palette()
        kind = "text"
        for rule_kind, pattern in self._COLOR_RULES:
            if pattern.search(msg):
                kind = rule_kind
                break
        text = html.escape(msg).replace("\n", "<br>")
        return f'<span style="color:{p[kind]};">{text}</span>'

    def _on_theme_changed(self):
        # rebuild every log line on theme change so stale colors never sit on
        # the new background
        self._apply_style()
        self.setHtml("<br>".join(self._line_html(m) for m in self._lines))
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
