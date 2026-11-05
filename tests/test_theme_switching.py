import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from gui import fluent_compat as compat


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_auto_theme_tracks_qt_system_color_scheme(qapp, monkeypatch):
    # Keep the app palette deliberately opposite to the reported OS scheme;
    # AUTO must not infer the scheme from the palette it controls itself.
    monkeypatch.setattr(compat, "_system_color_scheme", lambda: True)
    compat.setTheme(compat.Theme.AUTO)
    assert compat.isDarkTheme()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == "#202020"

    monkeypatch.setattr(compat, "_system_color_scheme", lambda: False)
    compat._on_system_color_scheme_changed()
    assert not compat.isDarkTheme()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == "#efefef"

    compat.setTheme(compat.Theme.LIGHT)


def test_theme_change_restyles_controls_and_refreshes_fluent_icons(qapp):
    compat.setTheme(compat.Theme.LIGHT)
    button = compat.PushButton("Restore Defaults", icon=compat.FluentIcon.SYNC)
    light_style = button.styleSheet()
    light_icon = button.icon().pixmap(button.iconSize()).toImage()

    compat.setTheme(compat.Theme.DARK)
    dark_icon = button.icon().pixmap(button.iconSize()).toImage()
    assert button.styleSheet() != light_style
    assert "rgba(255, 255, 255" in button.styleSheet()
    assert dark_icon != light_icon

    button.close()
    compat.setTheme(compat.Theme.LIGHT)
