"""Tests for the extracted GUI components: LogTerminal and AudioFileList."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.audio_file_list import AudioFileList
from gui.log_terminal import LogTerminal


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_log_terminal_accumulates_and_clears(qapp):
    terminal = LogTerminal()
    terminal.log("hello")
    terminal.log("错误: boom")
    assert terminal._lines == ["hello", "错误: boom"]
    # severity changes the color of the rendered line
    assert terminal._line_html("错误: boom") != terminal._line_html("hello")
    assert "<span" in terminal._line_html("错误: boom")
    terminal.clear()
    assert terminal._lines == []


def test_audio_file_list_filters_and_dedupes(qapp):
    file_list = AudioFileList()
    added_counts = []
    changes = []
    file_list.filesAdded.connect(added_counts.append)
    file_list.filesChanged.connect(lambda: changes.append(True))

    added = file_list.add_paths(["a.wav", "b.mp3", "notes.txt", "a.wav"])
    assert added == 2
    assert added_counts == [2]
    assert len(changes) == 1
    assert file_list.all_paths() == ["a.wav", "b.mp3"]
    assert file_list.item(file_list.count() - 1).data(Qt.UserRole) == "b.mp3"

    # re-adding an existing file changes nothing and emits no signals
    assert file_list.add_paths(["a.wav"]) == 0
    assert added_counts == [2]


def test_audio_file_list_remove_clears_file_settings(qapp):
    file_list = AudioFileList()
    file_list.add_paths(["a.wav", "b.wav"])
    file_list.file_settings["a.wav"] = {"tempo": 90.0}

    item = file_list.item(0)
    file_list.remove_item(item)

    assert file_list.all_paths() == ["b.wav"]
    assert "a.wav" not in file_list.file_settings

    file_list.clear_all()
    assert file_list.count() == 0
    assert file_list.file_settings == {}


def test_audio_file_list_settings_request_signal(qapp):
    file_list = AudioFileList()
    requested = []
    file_list.settingsRequested.connect(requested.append)
    file_list.add_paths(["a.wav"])
    # the gear row's settings button targets the file path
    file_list.settingsRequested.emit("a.wav")
    assert requested == ["a.wav"]
