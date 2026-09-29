"""Faithful port of cpp-pinyin G2P engine.

This subpackage is NOT auto-discovered by ``g2p/converters/__init__.py``.
Instead, wrapper modules at the parent level (``mandarin.py``,
``cantonese.py``) import and delegate to ``PinyinEngine``.
"""

from .engine import PinyinEngine
from .tones import apply_tone, STYLE_TONE3, STYLE_NORMAL, STYLE_TONE2, STYLE_FIRST_LETTER, STYLE_BOPOMOFO

__all__ = [
    "PinyinEngine",
    "apply_tone",
    "STYLE_TONE3",
    "STYLE_NORMAL",
    "STYLE_TONE2",
    "STYLE_FIRST_LETTER",
    "STYLE_BOPOMOFO",
]
