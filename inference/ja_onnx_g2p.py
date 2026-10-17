"""Japanese G2P via ja_g2p_onnx (LAKE-G2P v5): kanji -> kana -> romaji ->
DiffSinger phonemes. The model folder's runtime resolves resources relative to
its own file, so it is loaded in place via importlib, not vendored.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "kashi-g2p-onnx"
LEGACY_MODEL_DIR = PROJECT_ROOT / "models" / "ja_g2p_onnx"

_lock = threading.Lock()
_runtime = None
_runtime_module: ModuleType | None = None
_runtime_dir: Path | None = None


def _load_runtime_module(model_dir: Path | None = None) -> ModuleType:
    try:
        import inference.kashi_g2p_ja.g2p_onnx_runtime as module
        return module
    except ImportError:
        if model_dir:
            script = model_dir / "g2p_onnx_runtime.py"
            if script.is_file():
                spec = importlib.util.spec_from_file_location("ja_g2p_onnx_runtime", script)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
        raise ImportError("Cannot load g2p_onnx_runtime from inference.kashi_g2p_ja")


def get_ja_onnx_runtime_module(model_dir: str | Path | None = None) -> ModuleType | None:
    """Return the loaded ``g2p_onnx_runtime`` module, or None if absent.

    Adapters reuse its candidate-graph building blocks (build_edges,
    decode_viterbi, ...) instead of reimplementing them.
    """
    if get_ja_onnx_runtime(model_dir) is None:
        return None
    return _runtime_module


def get_ja_onnx_runtime(model_dir: str | Path | None = None):
    """Return the cached ``G2POnnxRuntime`` for *model_dir*, or None if absent.

    Loading is a one-time per-process cost; on failure the caller falls back
    to its legacy G2P path.
    """
    global _runtime, _runtime_module, _runtime_dir
    if model_dir:
        resolved = Path(model_dir)
    elif DEFAULT_MODEL_DIR.exists():
        resolved = DEFAULT_MODEL_DIR
    elif LEGACY_MODEL_DIR.exists():
        resolved = LEGACY_MODEL_DIR
    else:
        resolved = DEFAULT_MODEL_DIR
    if _runtime is not None and _runtime_dir == resolved:
        return _runtime
    with _lock:
        if _runtime is not None and _runtime_dir == resolved:
            return _runtime
        if not (resolved / "model.onnx").is_file():
            return None
        try:
            module = _load_runtime_module(resolved)
            runtime = module.G2POnnxRuntime(model_dir=resolved, prefer_dml=True)
        except Exception as e:
            logger.warning(f"kashi-g2p-onnx runtime unavailable ({e}); using legacy Japanese G2P")
            return None
        logger.info(f"[JaG2P-ONNX] loaded from {resolved} ({runtime.provider_name})")
        _runtime = runtime
        _runtime_module = module
        _runtime_dir = resolved
        return _runtime


# The model emits particle readings orthographically as standalone word edges
# (surface exactly "は"/"へ"); their pronunciation is わ/え.
_PARTICLE_READINGS = {"は": "わ", "へ": "え"}


def _is_word_boundary_surface(surface: str) -> bool:
    # Multi-char edges and single kanji end a word; a lone kana carries no
    # boundary signal (unspaced kana text is read char-by-char).
    if surface is None:
        return False
    if len(surface) > 1:
        return True
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in surface)


def normalize_edge_reading(surface: str, reading: str, prev_surface: str | None = None) -> str:
    """Rewrite particle は/へ to わ/え when the edge context allows it.

    Only when the previous edge marks a word boundary; in char-by-char kana
    runs a standalone は is more often word-internal (はんせん) than a particle.
    """
    if len(surface) != 1 or surface not in _PARTICLE_READINGS:
        return reading
    if not _is_word_boundary_surface(prev_surface):
        return reading
    return _PARTICLE_READINGS[surface]
