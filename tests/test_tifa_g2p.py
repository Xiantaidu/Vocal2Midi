"""Real-data smoke tests for the vendored TiFA G2P pipeline.

Covers the four text shapes the alignment stage can receive per ASR path:
hanzi (Qwen zh), kana (Qwen ja), pinyin tokens, romaji mora tokens, plus
English words. Skipped when the tifa-1.0-onnx bundle is absent.
"""
from pathlib import Path

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


def _word_phonemes(words):
    phonemes = []
    for word in words:
        for reading in word.readings:
            for path in reading.paths:
                for group in path:
                    phonemes.extend(group.phonemes)
    return phonemes


def test_hanzi_text_converts(pipeline):
    words = pipeline.convert("你好世界", languages=["zh"])
    assert words, "hanzi must convert"
    phonemes = _word_phonemes(words)
    assert phonemes, "hanzi words must carry phoneme readings"


def test_hanzi_polyphone_produces_candidates(pipeline):
    # 重 is a classic polyphone (zhong4/zhong5); the pipeline must offer
    # multiple candidate readings for disambiguation.
    words = pipeline.convert("重庆", languages=["zh"])
    assert words
    reading_counts = [len(word.readings) for word in words]
    assert max(reading_counts) >= 2, f"expected polyphone candidates, got {reading_counts}"


def test_kana_text_converts(pipeline):
    words = pipeline.convert("あいしてる", languages=["ja"])
    assert words
    assert _word_phonemes(words)


def test_pinyin_tokens_convert_via_dictionary(pipeline):
    words = pipeline.convert("ni hao", languages=["zh"])
    assert words
    assert _word_phonemes(words)


def test_romaji_mora_tokens_convert(pipeline):
    words = pipeline.convert("a i shi te ru", languages=["ja"])
    assert words
    assert _word_phonemes(words)


def test_english_words_convert(pipeline):
    words = pipeline.convert("hello world", languages=["en"])
    assert words
    assert _word_phonemes(words)


def test_mixed_language_text_converts(pipeline):
    words = pipeline.convert("こんにちは hello", languages=["zh", "ja", "en"])
    assert len(words) >= 2


def test_cantonese_text_converts(pipeline):
    words = pipeline.convert("海阔天空", languages=["yue"])
    assert len(words) == 4
    phonemes = _word_phonemes(words)
    assert phonemes
    assert "h" in phonemes and "oi" in phonemes


def test_cantonese_jyutping_tokens_convert(pipeline):
    words = pipeline.convert("hoi fut tin hung", languages=["yue"])
    assert len(words) == 4
    phonemes = _word_phonemes(words)
    assert phonemes
    assert "h" in phonemes and "oi" in phonemes
