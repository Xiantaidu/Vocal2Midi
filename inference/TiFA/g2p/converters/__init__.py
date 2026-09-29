import importlib
from pathlib import Path

from .base import Converter, G2PConversionError, G2PGroup, G2PPath, G2PWord, G2PReading

_dir = Path(__file__).parent
for _f in _dir.iterdir():
    if _f.suffix == ".py" and _f.stem not in ("__init__", "base"):
        importlib.import_module(f".{_f.stem}", __package__)
