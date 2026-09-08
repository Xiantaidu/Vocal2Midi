import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from qfluentwidgets import FluentWindow, NavigationItemPosition, setTheme, Theme, FluentIcon

from gui.global_settings_view import GlobalSettingsInterface
from gui.model_config_view import ModelConfigInterface
from gui.auto_lyric_view import AutoLyricInterface


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

        setTheme(Theme.LIGHT)

        self.globalSettingsInterface = GlobalSettingsInterface(self)
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

    def initNavigation(self):
        self.addSubInterface(self.autoLyricInterface, FluentIcon.MUSIC, "自动提取与灌注")
        self.addSubInterface(self.modelConfigInterface, FluentIcon.DEVELOPER_TOOLS, "模型配置")
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, "全局设置", position=NavigationItemPosition.BOTTOM)

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
                "任务仍在运行",
                "提取任务尚未结束，关闭窗口将强制停止任务。确定关闭吗？",
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
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
