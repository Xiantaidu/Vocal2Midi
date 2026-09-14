from __future__ import annotations

import os
from pathlib import Path


PORTABLE_ROOT_ENV = "V2M_PORTABLE_ROOT"


def get_portable_root(project_root: Path | None = None) -> Path | None:
    raw = os.environ.get(PORTABLE_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).resolve()
    if project_root is None:
        return None
    return None


def resolve_settings_path(project_root: Path | None = None) -> Path | None:
    portable_root = get_portable_root(project_root)
    if portable_root is None:
        return None
    return portable_root / "settings" / "vocal2midi.ini"


def create_app_settings(project_root: Path | None = None):
    """Create the shared application QSettings (ini in portable mode, registry otherwise).

    QSettings is imported lazily so this module stays importable in
    Qt-free environments (headless test runners, CLI tools).
    """
    from PySide6.QtCore import QSettings

    settings_path = resolve_settings_path(project_root)
    if settings_path is None:
        return QSettings("GAME_Extractor", "Vocal2Midi")
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings = QSettings(str(settings_path), QSettings.IniFormat)
    settings.setFallbacksEnabled(False)
    return settings


def default_output_dir(project_root: Path | None = None) -> Path:
    portable_root = get_portable_root(project_root)
    if portable_root is not None:
        return portable_root / "outputs"
    return Path.home() / "Desktop"
