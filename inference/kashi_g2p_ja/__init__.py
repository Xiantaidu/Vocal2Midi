"""kashi_g2p_ja: Standalone ONNX runtime for the LAKE-G2P Japanese G2P model."""
from inference.kashi_g2p_ja.g2p_onnx_runtime import (
    Edge,
    G2POnnxRuntime,
    UnifiedLexicon,
    decode_viterbi,
    build_edges,
    safe_windows,
    local_edges,
    normalize_surface,
    hira,
    is_kanji,
)

__all__ = [
    "Edge",
    "G2POnnxRuntime",
    "UnifiedLexicon",
    "decode_viterbi",
    "build_edges",
    "safe_windows",
    "local_edges",
    "normalize_surface",
    "hira",
    "is_kanji",
]
