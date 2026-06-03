import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from qfluentwidgets import FluentWindow, NavigationItemPosition, setTheme, Theme, FluentIcon

from gui.global_settings_view import GlobalSettingsInterface
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
        self.autoLyricInterface = AutoLyricInterface(self.globalSettingsInterface, self)

        self.initNavigation()

    def initNavigation(self):
        self.addSubInterface(self.autoLyricInterface, FluentIcon.MUSIC, "自动提取与灌注")
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, "全局设置", position=NavigationItemPosition.BOTTOM)

        # 隐藏侧边栏顶部三条杠（菜单）按钮
        self.navigationInterface.setMenuButtonVisible(False)
        
        # 固定侧边栏宽度
        self.navigationInterface.setExpandWidth(50)
        self.navigationInterface.setMinimumExpandWidth(50)
        self.navigationInterface.setMaximumWidth(50)

        self.navigationInterface.setCurrentItem(self.autoLyricInterface.objectName())
        self.stackedWidget.setCurrentWidget(self.autoLyricInterface)


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
