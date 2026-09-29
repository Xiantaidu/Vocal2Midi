"""Audio file list with per-row actions and per-file settings storage."""
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QAbstractItemView, QWidget

from qfluentwidgets import BodyLabel, FluentIcon, ListWidget, TransparentToolButton

from gui.i18n import tr


class AudioFileList(ListWidget):
    """List of audio files; each row carries settings & delete buttons.

    The display text of items stays empty — the row widget draws the file
    name itself; the real path lives in Qt.UserRole.
    """

    AUDIO_EXTENSIONS = {".wav", ".m4a", ".flac", ".mp3", ".ogg", ".opus", ".wma", ".webm", ".aif", ".aiff"}

    filesAdded = Signal(int)  # number of newly added files
    filesChanged = Signal()  # any add/remove/clear; drives batch mode
    settingsRequested = Signal(str)  # per-file settings button clicked

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # per-file settings in batch mode: filename -> dict of overrides
        self.file_settings: dict[str, dict] = {}
        # path -> gear button; visibility is managed batch-wide
        self._gear_buttons: dict[str, TransparentToolButton] = {}

    def add_paths(self, paths) -> int:
        """Add audio files (deduplicated); returns the number added."""
        existing = {self.item_path(self.item(i)) for i in range(self.count())}
        added = 0
        for path in paths:
            path = str(path)
            if Path(path).suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue
            if path in existing:
                continue
            self.addItem(path)
            self._attach_gear_row(path)
            existing.add(path)
            added += 1
        self._update_gear_visibility()
        if added:
            self.filesAdded.emit(added)
            self.filesChanged.emit()
        return added

    def clear_all(self):
        self.file_settings.clear()
        self._gear_buttons.clear()
        self.clear()
        self.filesChanged.emit()

    def remove_item(self, item):
        """Remove a single row and its per-file settings."""
        path = self.item_path(item)
        self.file_settings.pop(path, None)
        # drop the gear reference before the row widget is destroyed
        self._gear_buttons.pop(path, None)
        row_widget = self.itemWidget(item)
        if row_widget is not None:
            self.removeItemWidget(item)
        self.takeItem(self.row(item))
        self._update_gear_visibility()
        self.filesChanged.emit()

    def _update_gear_visibility(self):
        """Per-file settings only exist in batch mode (two or more files)."""
        visible = self.count() >= 2
        for button in self._gear_buttons.values():
            button.setVisible(visible)

    def item_path(self, item) -> str:
        return str(item.data(Qt.UserRole) or item.text())

    def selected_paths(self) -> list[str]:
        return [self.item_path(item) for item in self.selectedItems()]

    def all_paths(self) -> list[str]:
        return [self.item_path(self.item(i)) for i in range(self.count())]

    def _attach_gear_row(self, path: str):
        """Attach the filename + settings/delete button row to the last item."""
        item = self.item(self.count() - 1)
        if item is None or self.item_path(item) != path:
            return
        item.setText("")
        item.setData(Qt.UserRole, path)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 0, 8, 0)
        name_label = BodyLabel(path, row)
        row_layout.addWidget(name_label)
        row_layout.addStretch(1)
        btn = TransparentToolButton(FluentIcon.SETTING, row)
        btn.setToolTip(tr("file_settings"))
        btn.setFixedSize(28, 28)
        btn.clicked.connect(lambda checked, f=path: self.settingsRequested.emit(f))
        btn.setVisible(self.count() >= 2)  # batch mode only
        self._gear_buttons[path] = btn
        row_layout.addWidget(btn)
        btn_del = TransparentToolButton(FluentIcon.DELETE, row)
        btn_del.setToolTip(tr("clear_files"))
        btn_del.setFixedSize(28, 28)
        btn_del.clicked.connect(lambda checked, it=item: self.remove_item(it))
        row_layout.addWidget(btn_del)

        hint = row.sizeHint()
        hint.setHeight(36)  # fixed row height; QWidget.sizeHint ignores setFixedHeight
        item.setSizeHint(hint)
        self.setItemWidget(item, row)
