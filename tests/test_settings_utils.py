from __future__ import annotations

from pathlib import Path

from gui import settings_utils


def test_resolve_settings_path_uses_portable_root(monkeypatch, tmp_path):
    monkeypatch.setenv(settings_utils.PORTABLE_ROOT_ENV, str(tmp_path))

    settings_path = settings_utils.resolve_settings_path()

    assert settings_path == tmp_path / "settings" / "vocal2midi.ini"


def test_default_output_dir_uses_portable_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv(settings_utils.PORTABLE_ROOT_ENV, str(tmp_path))

    output_dir = settings_utils.default_output_dir()

    assert output_dir == tmp_path / "outputs"
