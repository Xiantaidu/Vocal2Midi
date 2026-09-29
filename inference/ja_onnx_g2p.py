"""Shared access to the ja_g2p_onnx Japanese G2P model (LAKE-G2P v5).

The model folder (``models/ja_g2p_onnx``) ships a self-contained runtime
(``g2p_onnx_runtime.py``) that resolves its resources relative to its own
file, so it is loaded in place via importlib instead of being vendored.
Both alignment engines consume its output: kanji text -> kana reading
(contextually disambiguated polyphones) -> romaji -> phonemes via the
DiffSinger Japanese dictionary.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "ja_g2p_onnx"

_lock = threading.Lock()
_runtime = None
_runtime_dir: Path | None = None


def _load_runtime_module(model_dir: Path) -> ModuleType:
    script = model_dir / "g2p_onnx_runtime.py"
    spec = importlib.util.spec_from_file_location("ja_g2p_onnx_runtime", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_ja_onnx_runtime(model_dir: str | Path | None = None):
    """Return the cached ``G2POnnxRuntime`` for *model_dir*, or None if absent.

    Loading the bundle takes a few seconds (one-time per process); failures
    are logged once and the caller falls back to its legacy G2P path.
    """
    global _runtime, _runtime_dir
    resolved = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if _runtime is not None and _runtime_dir == resolved:
        return _runtime
    with _lock:
        if _runtime is not None and _runtime_dir == resolved:
            return _runtime
        if not (resolved / "g2p_onnx_runtime.py").is_file():
            return None
        try:
            module = _load_runtime_module(resolved)
            runtime = module.G2POnnxRuntime(prefer_dml=True)
        except Exception as e:
            logger.warning(f"ja_g2p_onnx runtime unavailable ({e}); using legacy Japanese G2P")
            return None
        logger.info(f"[JaG2P-ONNX] loaded from {resolved} ({runtime.provider_name})")
        _runtime = runtime
        _runtime_dir = resolved
        return _runtime


# The model emits particle readings orthographically as standalone word edges
# (surface exactly "は"/"へ"); their pronunciation is わ/え.
_PARTICLE_READINGS = {"は": "わ", "へ": "え"}


def normalize_edge_reading(surface: str, reading: str) -> str:
    return _PARTICLE_READINGS.get(surface, reading) if len(surface) == 1 else reading
