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


@pytest.fixture(autouse=True)
def clean_batch_mode_setting():
    from gui.global_settings_view import GlobalSettingsInterface
    ui = GlobalSettingsInterface()
    ui.settings.remove("enable_batch_mode")
    ui.settings.remove("quantization_mode")
    ui.settings.remove("quantization_step")
    ui.settings.remove("alignment_engine")
    yield
    ui.settings.remove("enable_batch_mode")
    ui.settings.remove("quantization_mode")
    ui.settings.remove("quantization_step")
    ui.settings.remove("alignment_engine")


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


def test_gear_button_hidden_for_single_file_visible_in_batch(qapp):
    file_list = AudioFileList()

    file_list.add_paths(["a.wav"])
    assert file_list.count() == 1
    # isHidden reflects the explicit visibility flag even when unshown
    assert all(btn.isHidden() for btn in file_list._gear_buttons.values())

    file_list.add_paths(["b.wav"])
    assert file_list.count() == 2
    assert len(file_list._gear_buttons) == 2
    assert all(not btn.isHidden() for btn in file_list._gear_buttons.values())

    # back down to one file: the remaining gear hides again
    file_list.remove_item(file_list.item(0))
    assert file_list.count() == 1
    assert list(file_list._gear_buttons) == ["b.wav"]
    assert all(btn.isHidden() for btn in file_list._gear_buttons.values())


def test_batch_mode_hides_lyric_matching_and_invalidates_lyrics(qapp):
    from gui.auto_lyric_view import AutoLyricInterface
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    model_cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)
    view = AutoLyricInterface(settings_ui, model_cfg)

    # 1. Single file mode
    view.audio_list.add_paths(["song1.wav"])
    assert view.audio_list.count() == 1

    view.cb_match_lyrics.setChecked(True)
    view.lyrics_edit.setText("test lyrics 123")

    assert not view._combo_card.isHidden()
    assert not view._output_card.isHidden()
    assert not view.lyric_card.isHidden()

    base = view._current_base_values()
    assert base["match_lyrics"] is True
    assert base["original_lyrics"] == "test lyrics 123"

    # 2. Add second file -> batch mode
    view.audio_list.add_paths(["song2.wav"])
    assert view.audio_list.count() == 2

    # Cards must be hidden, including lyric_card
    assert view._combo_card.isHidden()
    assert view._output_card.isHidden()
    assert view.lyric_card.isHidden()

    # Lyrics on main window must be invalidated in batch mode
    batch_base = view._current_base_values()
    assert batch_base["match_lyrics"] is False
    assert batch_base["original_lyrics"] == ""

    # 3. Remove second file -> back to single file mode
    view.audio_list.remove_item(view.audio_list.item(1))
    assert view.audio_list.count() == 1

    assert not view._combo_card.isHidden()
    assert not view._output_card.isHidden()
    assert not view.lyric_card.isHidden()

    restored_base = view._current_base_values()
    assert restored_base["match_lyrics"] is True
    assert restored_base["original_lyrics"] == "test lyrics 123"


def test_batch_mode_setting_toggle_and_defaults(qapp):
    from gui.global_settings_view import GlobalSettingsInterface

    settings_ui = GlobalSettingsInterface()
    assert settings_ui.enable_batch_mode() is True

    signals = []
    settings_ui.batchModeChanged.connect(signals.append)

    settings_ui.cb_batch_mode.setChecked(False)
    assert settings_ui.enable_batch_mode() is False
    assert signals == [False]

    settings_ui.reset_to_default()
    assert settings_ui.enable_batch_mode() is True
    assert signals == [False, True]


def test_audio_file_list_batch_mode_off_restricts_to_one(qapp):
    file_list = AudioFileList()
    file_list.set_batch_mode(False)
    assert file_list.batch_mode is False

    # Adding multiple files when batch_mode is False only adds the first file
    added = file_list.add_paths(["song1.wav", "song2.mp3", "song3.flac"])
    assert added == 1
    assert file_list.count() == 1
    assert file_list.all_paths() == ["song1.wav"]

    # Adding another file replaces the existing file in single-file mode
    added = file_list.add_paths(["song4.wav"])
    assert added == 1
    assert file_list.count() == 1
    assert file_list.all_paths() == ["song4.wav"]

    # Re-adding the exact same file is a no-op
    added = file_list.add_paths(["song4.wav"])
    assert added == 0
    assert file_list.count() == 1
    assert file_list.all_paths() == ["song4.wav"]


def test_audio_file_list_dynamic_disable_trims_to_first(qapp):
    file_list = AudioFileList()
    file_list.add_paths(["a.wav", "b.wav", "c.wav"])
    assert file_list.count() == 3

    changes = []
    file_list.filesChanged.connect(lambda: changes.append(True))

    file_list.set_batch_mode(False)
    assert file_list.count() == 1
    assert file_list.all_paths() == ["a.wav"]
    assert len(changes) == 1
    assert all(btn.isHidden() for btn in file_list._gear_buttons.values())


def test_auto_lyric_view_coordinates_with_global_batch_mode(qapp):
    from gui.auto_lyric_view import AutoLyricInterface
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    settings_ui.cb_batch_mode.setChecked(True)
    model_cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)
    view = AutoLyricInterface(settings_ui, model_cfg)

    # 1. Add 2 files while batch mode is on
    view.audio_list.add_paths(["song1.wav", "song2.wav"])
    assert view.audio_list.count() == 2
    assert view._combo_card.isHidden()

    # 2. Turn off batch mode in global settings
    settings_ui.cb_batch_mode.setChecked(False)
    # The audio list must automatically trim down to 1 file and restore panels
    assert view.audio_list.count() == 1
    assert view.audio_list.all_paths() == ["song1.wav"]
    assert not view._combo_card.isHidden()
    assert not view._output_card.isHidden()

    # 3. In single file mode, attempting to add multiple files adds only 1 (replacing)
    view.audio_list.add_paths(["song3.wav", "song4.wav"])
    assert view.audio_list.count() == 1
    assert view.audio_list.all_paths() == ["song3.wav"]


def test_quantization_settings_persisted(qapp):
    from gui.auto_lyric_view import AutoLyricInterface
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    model_cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)

    # 1. Defaults should be "smart" and 0 (off)
    view1 = AutoLyricInterface(settings_ui, model_cfg)
    assert view1.quantize_mode_combo.currentData() == "smart"
    assert view1.quantize_combo.currentData() == 0

    # 2. Change mode to "simple" and step to 480 (1/4 note)
    idx_simple = view1.quantize_mode_combo.findData("simple")
    assert idx_simple >= 0
    view1.quantize_mode_combo.setCurrentIndex(idx_simple)
    assert settings_ui.settings.value("quantization_mode") == "simple"

    idx_480 = view1.quantize_combo.findData(480)
    assert idx_480 >= 0
    view1.quantize_combo.setCurrentIndex(idx_480)
    assert int(settings_ui.settings.value("quantization_step")) == 480

    # 3. New view instance should load "simple" and 480 from settings
    view2 = AutoLyricInterface(settings_ui, model_cfg)
    assert view2.quantize_mode_combo.currentData() == "simple"
    assert view2.quantize_combo.currentData() == 480


def test_gui_version(qapp):
    from gui import __version__
    from gui.fluent_main import MainWindow

    assert __version__ == "2.0.0"
    window = MainWindow()
    assert "2.0.0" in window.windowTitle()
    assert window.windowTitle() == f"Vocal2Midi v{__version__}"


def test_model_config_engine_choices_and_kashi_g2p(qapp):
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)

    # Verify all 4 engines exist
    assert hasattr(cfg, "chinese_asr_engine_combo")
    assert hasattr(cfg, "japanese_asr_engine_combo")
    assert hasattr(cfg, "alignment_engine_combo")
    assert hasattr(cfg, "japanese_g2p_engine_combo")
    assert hasattr(cfg, "kashi_g2p_model_edit")

    # Default value for japanese_g2p_engine
    assert cfg.japanese_g2p_engine() in ("kashi-g2p-onnx", "pyopenjtalk")
    assert cfg.model_path("kashi_g2p_model") == "models/kashi-g2p-onnx"

    # Switching Japanese G2P engine
    idx_py = cfg.japanese_g2p_engine_combo.findData("pyopenjtalk")
    assert idx_py >= 0
    cfg.japanese_g2p_engine_combo.setCurrentIndex(idx_py)
    assert cfg.japanese_g2p_engine() == "pyopenjtalk"
    assert settings_ui.settings.value("japanese_g2p_engine") == "pyopenjtalk"


def test_hfa_disables_cantonese_in_gui(qapp):
    from gui.auto_lyric_view import AutoLyricInterface
    from gui.file_settings_dialog import FileSettingsDialog
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)

    # 1. Default alignment engine is tifa: Cantonese ('yue') is enabled by default
    assert cfg.alignment_engine() == "tifa"

    view = AutoLyricInterface(settings_ui, cfg)
    yue_idx = view.lang_combo.findData("yue")
    assert yue_idx >= 0
    assert view.lang_combo.items[yue_idx].isEnabled
    assert view.lang_combo.itemText(yue_idx) == "yue"

    # 2. Switch aligner to hfa: Cantonese ('yue') becomes disabled
    idx_hfa = cfg.alignment_engine_combo.findData("hfa")
    assert idx_hfa >= 0
    cfg.alignment_engine_combo.setCurrentIndex(idx_hfa)
    assert cfg.alignment_engine() == "hfa"
    assert not view.lang_combo.items[yue_idx].isEnabled
    assert "(需 TiFA)" in view.lang_combo.itemText(yue_idx) or "(requires TiFA)" in view.lang_combo.itemText(yue_idx)

    # If yue is somehow attempted to be selected when hfa is active, it snaps back to zh
    view.lang_combo.setCurrentIndex(yue_idx)
    assert view.lang_combo.currentData() == "zh"

    # 3. Switch back to tifa: Cantonese ('yue') becomes enabled again
    idx_tifa = cfg.alignment_engine_combo.findData("tifa")
    assert idx_tifa >= 0
    cfg.alignment_engine_combo.setCurrentIndex(idx_tifa)
    assert cfg.alignment_engine() == "tifa"
    assert view.lang_combo.items[yue_idx].isEnabled
    assert view.lang_combo.itemText(yue_idx) == "yue"

    # Can now select yue
    view.lang_combo.setCurrentIndex(yue_idx)
    assert view.lang_combo.currentData() == "yue"

    # 4. In FileSettingsDialog, hfa base value disables yue
    base_hfa = dict(view._current_base_values())
    base_hfa["alignment_engine"] = "hfa"
    base_hfa["language"] = "yue"
    dlg_hfa = FileSettingsDialog("test.wav", base_hfa, parent=view)
    dlg_yue_idx = dlg_hfa.lang_combo.findData("yue")
    assert not dlg_hfa.lang_combo.items[dlg_yue_idx].isEnabled
    assert dlg_hfa.lang_combo.currentData() == "zh"




