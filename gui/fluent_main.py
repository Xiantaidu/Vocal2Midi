import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon

from qfluentwidgets import FluentWindow, NavigationItemPosition, setTheme, Theme, FluentIcon

from gui.global_settings_view import GlobalSettingsInterface
from gui.i18n import tr, set_language
from gui.model_config_view import ModelConfigInterface
from gui.auto_lyric_view import AutoLyricInterface
from gui.settings_utils import create_app_settings


def _load_app_icon() -> QIcon:
    icon_path = Path(__file__).resolve().parents[1] / "icon.png"
    if not icon_path.is_file():
        return QIcon()
    return QIcon(str(icon_path))


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vocal2Midi")
        app_icon = _load_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.resize(1000, 800)
        self._apply_fixed_physical_size(1260, 1000)

        # Apply the saved language before building any widget so every page
        # constructs in the right language.
        self.app_settings = create_app_settings()
        set_language(str(self.app_settings.value("language", "zh")))

        self.globalSettingsInterface = GlobalSettingsInterface(self)
        setTheme(self._load_theme())
        self._enable_mica()

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

        self.initNavigation()
        self.globalSettingsInterface.languageChanged.connect(self._on_language_changed)

    def _on_language_changed(self, language: str):
        set_language(language)
        self._retranslate_navigation()
        self.modelConfigInterface.retranslate_ui()
        self.autoLyricInterface.retranslate_ui()
        self.globalSettingsInterface.retranslate_ui()

    def _retranslate_navigation(self):
        nav = self.navigationInterface
        for page, key in (
            (self.autoLyricInterface, "nav_auto"),
            (self.modelConfigInterface, "nav_model"),
            (self.globalSettingsInterface, "nav_settings"),
        ):
            item = nav.widget(page.objectName())
            if item is not None:
                item.setText(tr(key))

    def _apply_fixed_physical_size(self, width: int, height: int):
        """Lock the window size in physical pixels: logical = physical / scale factor.

        Qt's resize/setFixedSize take logical pixels, so passing 1260x1000
        verbatim at 125% scaling would produce a 1575x1250 physical window.
        The size is not re-adjusted after moving to a display with a
        different scale factor.
        """
        screen = self.screen() or QApplication.primaryScreen()
        dpr = screen.devicePixelRatio() if screen is not None else 1.0
        size = QSize(max(1, round(width / dpr)), max(1, round(height / dpr)))
        self.resize(size)
        self.setFixedSize(size)

    def _load_theme(self):
        raw = str(self.globalSettingsInterface.settings.value("theme", "light")).strip().lower()
        return {"dark": Theme.DARK, "auto": Theme.AUTO}.get(raw, Theme.LIGHT)

    def _enable_mica(self):
        # Mica only exists on Windows 11; qfluentwidgets degrades gracefully
        # elsewhere, and older builds may not expose the API at all.
        try:
            self.setMicaEffectEnabled(True)
        except Exception:
            pass

    def initNavigation(self):
        self.addSubInterface(self.autoLyricInterface, FluentIcon.MUSIC, tr("nav_auto"))
        self.addSubInterface(self.modelConfigInterface, FluentIcon.DEVELOPER_TOOLS, tr("nav_model"))
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, tr("nav_settings"), position=NavigationItemPosition.BOTTOM)

        # Hide the hamburger/menu button at the top of the sidebar.
        self.navigationInterface.setMenuButtonVisible(False)
        
        # Keep the sidebar width fixed.
        self.navigationInterface.setExpandWidth(50)
        self.navigationInterface.setMinimumExpandWidth(50)
        self.navigationInterface.setMaximumWidth(50)

        self.navigationInterface.setCurrentItem(self.autoLyricInterface.objectName())
        self.stackedWidget.setCurrentWidget(self.autoLyricInterface)

    def closeEvent(self, event):
        # Destroying a running QThread crashes at exit ("QThread: Destroyed
        # while thread is still running"); stop the worker first.
        worker = getattr(self.autoLyricInterface, "worker", None)
        if worker is not None and worker.isRunning():
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
            worker.stop()
            worker.wait(5000)
        event.accept()


def run_app():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
