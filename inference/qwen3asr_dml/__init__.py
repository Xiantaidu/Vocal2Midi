import logging
import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
BIN_DIR = PACKAGE_DIR / "bin"

if hasattr(os, "add_dll_directory") and BIN_DIR.is_dir():
    os.add_dll_directory(str(BIN_DIR))

logger = logging.getLogger("qwen3asr_dml")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

from .asr import QwenASREngine
from .chinese_itn import chinese_to_num as itn
from .schema import ASREngineConfig, DecodeResult, TranscribeResult
from .utils import load_audio

__all__ = [
    "logger",
    "QwenASREngine",
    "ASREngineConfig",
    "DecodeResult",
    "TranscribeResult",
    "itn",
    "load_audio",
]
