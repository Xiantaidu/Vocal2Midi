"""Native macOS window shell for the Vocal2Midi desktop UI.

The Windows build uses ``FluentWindow`` from ``gui/fluent_compat.py``, whose
frameless-window native integration is not a good fit for macOS. This module
keeps the same pages and settings in a plain QMainWindow with a native sidebar.
"""

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.fluent_compat import FluentIcon, setTheme, Theme, qconfig, isDarkTheme

from gui import __version__
from gui.auto_lyric_view import AutoLyricInterface
from gui.global_settings_view import GlobalSettingsInterface
from gui.i18n import set_language, tr
from gui.model_config_view import ModelConfigInterface
from gui.settings_utils import create_app_settings


def _load_app_icon() -> QIcon:
    icon_path = Path(__file__).resolve().parents[1] / "icon.png"
    return QIcon(str(icon_path)) if icon_path.is_file() else QIcon()


class MacMainWindow(QMainWindow):
    pageChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Vocal2Midi v{__version__}")
        icon = _load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setMinimumSize(980, 700)
        self.resize(1260, 860)

        self.app_settings = create_app_settings()
        set_language(str(self.app_settings.value("language", "zh")))

        self.globalSettingsInterface = GlobalSettingsInterface(self)
        setTheme(self._load_theme())
        self.modelConfigInterface = ModelConfigInterface(
            self.globalSettingsInterface.settings,
            self.globalSettingsInterface.project_root,
            self,
        )
        self.autoLyricInterface = AutoLyricInterface(
            self.globalSettingsInterface,
            self.modelConfigInterface,
            self,
        )

        self._pages = [
            self.autoLyricInterface,
            self.modelConfigInterface,
            self.globalSettingsInterface,
        ]
        self._nav_keys = ["nav_auto", "nav_model", "nav_settings"]
        # (setter, i18n key) pairs for texts that must follow language changes.
        self._translated_texts = []
        self._build_layout()
        self._build_macos_menus()
        qconfig.themeChangedFinished.connect(self._restyle_theme)
        self.globalSettingsInterface.languageChanged.connect(self._on_language_changed)
        self._refresh_model_status()
        # Finder can launch the app with one or more audio files when the
        # bundle is registered as an audio document handler.
        QTimer.singleShot(0, self._open_document_arguments)

    def _load_theme(self):
        raw = str(self.globalSettingsInterface.settings.value("theme", "light")).strip().lower()
        return {"dark": Theme.DARK, "auto": Theme.AUTO}.get(raw, Theme.LIGHT)

    def _build_layout(self):
        root = QWidget(self)
        self._mac_root = root
        root.setObjectName("macRoot")
        self._mac_root_stylesheet = (
            "QWidget#macRoot { background: palette(window); }"
            "QListWidget#macNavigation { border: 0; padding: 10px 6px; "
            "background: rgba(127,127,127,0.08); font-size: 13px; }"
            "QListWidget#macNavigation::item { padding: 10px 8px; border-radius: 7px; }"
            "QListWidget#macNavigation::item:selected { background: palette(highlight); "
            "color: palette(highlighted-text); }"
        )
        root.setStyleSheet(self._mac_root_stylesheet)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.navigationList = QListWidget(root)
        self.navigationList.setObjectName("macNavigation")
        self.navigationList.setFixedWidth(190)
        self.navigationList.setFrameShape(QListWidget.NoFrame)
        self.navigationList.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.navigationList.setIconSize(QSize(18, 18))
        for page, key, icon in zip(
            self._pages,
            self._nav_keys,
            (FluentIcon.MUSIC, FluentIcon.DEVELOPER_TOOLS, FluentIcon.SETTING),
        ):
            item = QListWidgetItem(icon.icon(), tr(key))
            item.setData(Qt.UserRole, page.objectName())
            self.navigationList.addItem(item)
        self.navigationList.currentRowChanged.connect(self._switch_page)
        sidebar = QWidget(root)
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 18, 12, 14)
        sidebar_layout.setSpacing(10)
        brand = QLabel("Vocal2Midi")
        brand.setStyleSheet("font-size: 20px; font-weight: 700; padding: 2px 7px;")
        subtitle = QLabel()
        self._bind_text(subtitle.setText, "mac_subtitle")
        self._mac_subtitle = subtitle
        self._mac_subtitle_stylesheet = "color: palette(mid); font-size: 11px; padding: 0 7px 8px;"
        subtitle.setStyleSheet(self._mac_subtitle_stylesheet)
        subtitle.setWordWrap(True)
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addWidget(self.navigationList, 1)
        self.modelStatusLabel = QLabel()
        self.modelStatusLabel.setWordWrap(True)
        self._model_status_stylesheet = "color: palette(mid); font-size: 11px; padding: 8px 7px;"
        self.modelStatusLabel.setStyleSheet(self._model_status_stylesheet)
        sidebar_layout.addWidget(self.modelStatusLabel)
        layout.addWidget(sidebar)

        self.stackedWidget = QStackedWidget(root)
        for page in self._pages:
            self.stackedWidget.addWidget(page)
        layout.addWidget(self.stackedWidget, 1)

        self.setCentralWidget(root)
        self.navigationList.setCurrentRow(0)

    def _restyle_theme(self):
        """Reapply palette based QSS and icons after the application theme changes."""
        self._mac_root.setStyleSheet(self._mac_root_stylesheet)
        self._mac_subtitle.setStyleSheet(self._mac_subtitle_stylesheet)
        self.modelStatusLabel.setStyleSheet(self._model_status_stylesheet)
        dark = isDarkTheme()
        for row, icon in enumerate(
            (FluentIcon.MUSIC, FluentIcon.DEVELOPER_TOOLS, FluentIcon.SETTING)
        ):
            item = self.navigationList.item(row)
            if item is not None:
                item.setIcon(icon.icon(Theme.DARK if dark else Theme.LIGHT))

    def _build_macos_menus(self):
        """Expose the common actions through macOS' native menu bar and shortcuts."""
        file_menu = self.menuBar().addMenu("")
        self._bind_text(file_menu.setTitle, "menu_file")
        open_action = self._translated_action("menu_open_audio")
        open_action.setShortcut(QKeySequence("Meta+O"))
        open_action.triggered.connect(self._open_audio)
        file_menu.addAction(open_action)
        convert_action = self._translated_action("menu_start")
        convert_action.setShortcut(QKeySequence("Meta+Return"))
        convert_action.triggered.connect(self._start_conversion)
        file_menu.addAction(convert_action)
        stop_action = self._translated_action("menu_stop")
        stop_action.triggered.connect(self.autoLyricInterface.request_stop)
        file_menu.addAction(stop_action)
        file_menu.addSeparator()
        quit_action = self._translated_action("menu_quit")
        quit_action.setShortcut(QKeySequence("Meta+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("")
        self._bind_text(view_menu.setTitle, "menu_view")
        for index, key in enumerate(self._nav_keys):
            action = self._translated_action(key)
            action.triggered.connect(lambda _checked=False, i=index: self.navigationList.setCurrentRow(i))
            view_menu.addAction(action)
        help_menu = self.menuBar().addMenu("")
        self._bind_text(help_menu.setTitle, "menu_help")
        about_action = self._translated_action("menu_about")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _bind_text(self, setter, key: str) -> None:
        setter(tr(key))
        self._translated_texts.append((setter, key))

    def _translated_action(self, key: str) -> QAction:
        action = QAction(self)
        self._bind_text(action.setText, key)
        return action

    def _open_audio(self):
        self.navigationList.setCurrentRow(0)
        self.autoLyricInterface.add_audio_files()

    def _open_document_arguments(self):
        paths = [p for p in sys.argv[1:] if os.path.isfile(p)]
        if paths and hasattr(self.autoLyricInterface, "audio_list"):
            self.autoLyricInterface.audio_list.add_paths(paths)

    def _start_conversion(self):
        self.navigationList.setCurrentRow(0)
        self.autoLyricInterface.run_pipeline()

    def _show_about(self):
        QMessageBox.about(
            self,
            tr("menu_about"),
            f"<b>Vocal2Midi v{__version__}</b><br>{tr('about_body')}<br><br>"
            f"{tr('about_models')} {self._model_status_text()}",
        )

    def _model_status_text(self):
        paths = {
            "TiFA": self.modelConfigInterface.model_path("tifa_model"),
            "GAME": self.modelConfigInterface.model_path("game_model"),
            "RMVPE": self.modelConfigInterface.model_path("rmvpe_model"),
        }
        return " · ".join(f"{name} {'✓' if os.path.exists(path) else '⚠'}" for name, path in paths.items())

    def _refresh_model_status(self):
        self.modelStatusLabel.setText(tr("model_status", status=self._model_status_text()))

    def _switch_page(self, row: int):
        if 0 <= row < self.stackedWidget.count():
            self.stackedWidget.setCurrentIndex(row)
            self.pageChanged.emit(row)

    def _on_language_changed(self, language: str):
        set_language(language)
        for row, key in enumerate(self._nav_keys):
            item = self.navigationList.item(row)
            if item is not None:
                item.setText(tr(key))
        self.modelConfigInterface.retranslate_ui()
        self.autoLyricInterface.retranslate_ui()
        self.globalSettingsInterface.retranslate_ui()
        for setter, key in self._translated_texts:
            setter(tr(key))
        self._refresh_model_status()

    def _finish_close(self, event):
        if not self.autoLyricInterface.is_running():
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            tr("close_running_title"),
            tr("close_running_body"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            event.ignore()
            return
        self.autoLyricInterface.request_stop()
        if self.autoLyricInterface.wait_for_worker(5000):
            event.accept()
            return
        self.hide()
        self.autoLyricInterface.on_worker_settled(self.close)
        event.ignore()

    def closeEvent(self, event):
        self._finish_close(event)
