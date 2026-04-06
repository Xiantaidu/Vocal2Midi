import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from qfluentwidgets import FluentWindow, NavigationItemPosition, setTheme, Theme, FluentIcon

from gui.global_settings_view import GlobalSettingsInterface
from gui.auto_lyric_view import AutoLyricInterface


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GAME: 生成式自适应 MIDI 提取器")
        self.resize(1000, 800)

        setTheme(Theme.LIGHT)

        self.globalSettingsInterface = GlobalSettingsInterface(self)
        self.autoLyricInterface = AutoLyricInterface(self.globalSettingsInterface, self)
        self.globalSettingsInterface.cb_match_lyrics.checkedChanged.connect(self.autoLyricInterface.update_lyrics_visibility)

        self.initNavigation()

    def initNavigation(self):
        self.addSubInterface(self.autoLyricInterface, FluentIcon.MUSIC, "自动提取与灌注")
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, "全局设置", position=NavigationItemPosition.BOTTOM)

        self.navigationInterface.setCurrentItem(self.autoLyricInterface.objectName())
        self.stackedWidget.setCurrentWidget(self.autoLyricInterface)


def run_app():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
