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


def default_output_dir(project_root: Path | None = None) -> Path:
    portable_root = get_portable_root(project_root)
    if portable_root is not None:
        return portable_root / "outputs"
    return Path.home() / "Desktop"
