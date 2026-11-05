import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from gui import fluent_compat as compat


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_light_theme():
    yield
    compat.setTheme(compat.Theme.LIGHT)


@pytest.fixture
def styled_page(qapp):
    """The hierarchy the real pages use: a scroll area holding a text edit.

    ScrollArea is styled with ``background: transparent``, which is what made an
    unstyled QMenu inherit a black window colour.
    """
    area = compat.ScrollArea()
    holder = QWidget()
    layout = QVBoxLayout(holder)
    editor = compat.TextEdit()
    layout.addWidget(editor)
    area.setWidget(holder)
    # The application calls setTheme() once at start-up; that is what installs
    # the popup styler.
    compat.setTheme(compat.Theme.LIGHT)
    editors = [area, holder, editor]  # keep the C++ parents alive
    yield editor, editors
    area.deleteLater()


def _open_menu(editor):
    menu = editor.createStandardContextMenu()
    menu.popup(QPoint(0, 0))
    QApplication.processEvents()
    return menu


def test_context_menu_from_scroll_page_is_styled(styled_page):
    editor, _parents = styled_page
    menu = _open_menu(editor)
    assert menu.styleSheet(), "context menu must not rely on the inherited palette"
    assert compat._tokens()["menu_bg"] in menu.styleSheet()
    menu.close()


@pytest.mark.parametrize("theme", [compat.Theme.LIGHT, compat.Theme.DARK])
def test_context_menu_follows_the_theme(styled_page, theme):
    editor, _parents = styled_page
    compat.setTheme(theme)
    menu = _open_menu(editor)
    assert compat._tokens()["menu_bg"] in menu.styleSheet()
    assert compat._tokens()["text"] in menu.styleSheet()
    menu.close()


def test_menu_stylesheet_is_rebuilt_after_a_theme_change():
    compat.setTheme(compat.Theme.LIGHT)
    light_sheet = compat._menu_qss()
    compat.setTheme(compat.Theme.DARK)
    dark_sheet = compat._menu_qss()
    assert light_sheet != dark_sheet
    assert "#f9f9f9" in light_sheet
    assert "#2b2b2b" in dark_sheet


def test_menu_keeps_its_own_stylesheet(styled_page):
    editor, _parents = styled_page
    menu = _open_menu(editor)
    menu.setStyleSheet("QMenu { background-color: #123456; }")
    menu.hide()
    menu.popup(QPoint(0, 0))
    QApplication.processEvents()
    assert "#123456" in menu.styleSheet()
    menu.close()
