import threading
from typing import Optional

_OPENCC_INSTANCE = None
_OPENCC_LOCK = threading.Lock()


def get_opencc():
    """Lazily initialize and return the global OpenCC instance for t2s conversion."""
    global _OPENCC_INSTANCE
    if _OPENCC_INSTANCE is not None:
        return _OPENCC_INSTANCE

    with _OPENCC_LOCK:
        if _OPENCC_INSTANCE is not None:
            return _OPENCC_INSTANCE
        import opencc
        _OPENCC_INSTANCE = opencc.OpenCC("t2s")
        return _OPENCC_INSTANCE


def traditional_to_simplified(text: str) -> str:
    """Convert Traditional Chinese text to Simplified Chinese using OpenCC (t2s).

    Non-Chinese text, punctuation, and already-simplified characters are preserved.
    Safe on empty/None inputs.
    """
    if not text:
        return ""
    try:
        converter = get_opencc()
        return converter.convert(str(text))
    except Exception:
        return str(text)
