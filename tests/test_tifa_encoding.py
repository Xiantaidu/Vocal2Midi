"""Encoding gate: the four text shapes must encode against the real TiFA vocabulary.

These are the acceptance gate for "TiFA full-text mode covers every ASR path's
text": hanzi (Qwen zh), kana (Qwen ja), pinyin tokens, romaji mora tokens and
English words. Skipped when the tifa-1.0-onnx bundle is absent.
"""
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "tifa-1.0-onnx"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "vocabulary.json").is_file(),
    reason="tifa-1.0-onnx model bundle not present",
)


@pytest.fixture(scope="module")
def pipeline():
    from inference.TiFA.g2p_pipeline import build_g2p_pipeline

    return build_g2p_pipeline(MODEL_DIR)


@pytest.fixture(scope="module")
def vocabulary():
    from inference.TiFA.lib.vocabulary import Vocabulary

    return Vocabulary.from_file(MODEL_DIR / "vocabulary.json")


def _encode(pipeline, vocabulary, text, language):
    from inference.TiFA.g2p.encoding import encode_paths
    from inference.TiFA.aligner import GLOBAL_SYMBOLS, STOP_SYMBOLS

    words = pipeline.convert(text, languages=[language])
    return encode_paths(
        words, vocabulary, "discard",
        languages=[language],
        global_symbols=GLOBAL_SYMBOLS,
        stop_symbols=STOP_SYMBOLS,
    )


def test_hanzi_encodes(pipeline, vocabulary):
    data, lexicon, texts = _encode(pipeline, vocabulary, "你好世界", "zh")
    assert data["paths"].shape[0] > 0
    assert data["candidates"].sum() >= 4  # one valid candidate per char
    assert data["paths"].shape[1] > 0 and texts == list("你好世界")


def test_pinyin_tokens_encode(pipeline, vocabulary):
    data, lexicon, texts = _encode(pipeline, vocabulary, "ni hao", "zh")
    assert data["paths"].any()
    assert texts == ["ni", "hao"]


def test_romaji_mora_tokens_encode(pipeline, vocabulary):
    data, lexicon, texts = _encode(pipeline, vocabulary, "a i shi te ru", "ja")
    assert data["paths"].any(), "romaji mora tokens must encode into the ja vocabulary"


def test_kana_encodes(pipeline, vocabulary):
    data, lexicon, texts = _encode(pipeline, vocabulary, "あいしてる", "ja")
    assert data["paths"].any()


def test_english_encodes(pipeline, vocabulary):
    data, lexicon, texts = _encode(pipeline, vocabulary, "hello world", "en")
    assert data["paths"].any()
    assert texts == ["hello", "world"]


def test_encoded_tokens_are_not_reserved(pipeline, vocabulary):
    from inference.TiFA.lib.vocabulary import NUM_RESERVED_TOKENS

    words = pipeline.convert("你好世界 hello", languages=["zh", "en"])
    from inference.TiFA.g2p.encoding import encode_paths
    from inference.TiFA.aligner import GLOBAL_SYMBOLS, STOP_SYMBOLS

    data, _, _ = encode_paths(
        words, vocabulary, "discard",
        languages=["zh", "en"],
        global_symbols=GLOBAL_SYMBOLS,
        stop_symbols=STOP_SYMBOLS,
    )
    tokens = data["paths"]
    assert tokens[tokens > 0].min() >= NUM_RESERVED_TOKENS
    assert np.isin(tokens, [0]).sum() > 0  # alignment gaps exist between words


def test_japanese_lexicon_kanji_encodes_with_candidates(pipeline, vocabulary):
    """The ja_g2p lexicon converter emits multiple candidate readings per kanji
    word; selection is deferred to the scoring DP."""
    import importlib.util

    if not (MODEL_DIR / "dictionaries" / "ja_lexicon.txz").is_file():
        pytest.skip("ja_lexicon.txz not present")
    assert importlib.util.find_spec("fugashi") is None or True  # MeCab optional

    words = pipeline.convert("普通の世界", languages=["ja"])
    by_text = {word.text: word for word in words}
    assert "普通" in by_text
    assert len(by_text["普通"].readings) >= 2, "polyphone candidates must survive"

    data, lexicon, texts = _encode(pipeline, vocabulary, "普通の世界", "ja")
    assert data["paths"].any()
    assert "普通" in texts and "世界" in texts


def test_ja_dakuten_digraphs_encode(pipeline, vocabulary):
    """ぢ-row digraphs are missing from the upstream kana->romaji map (the
    chunk_11/chunk_2 production failure); the additions must cover them."""
    for text in ("ぢょう", "ちぢむ", "ぢゃ無い"):
        data, lexicon, texts = _encode(pipeline, vocabulary, text, "ja")
        assert data["paths"].any(), text


def test_ja_nonstandard_kana_readings_are_faithful():
    """A handful of lexicon typos were mapped to the wrong sound. Pin the
    corrected readings so they never regress to the collapsed palatalized form:
    だョ (東京だョ = da-yo) must stay two mora, and づ-onset foreign spellings
    (zain/zopfli) must keep the z consonant, not gain a spurious 'du'."""
    from inference.TiFA.g2p.converters.japanese import JapaneseKanaConverter

    conv = JapaneseKanaConverter(
        dict_path=str(MODEL_DIR / "dictionaries" / "japanese_dict_full.txt")
    )

    def phonemes(kana):
        return [
            tuple(g.phonemes)
            for w in conv.convert(kana)
            for r in w.readings
            for path in r.paths
            for g in path
        ]

    assert phonemes("だょ") == [("d", "a", "y", "o")]
    assert phonemes("づぁ") == [("z", "a")]
    assert phonemes("づぉ") == [("z", "o")]

