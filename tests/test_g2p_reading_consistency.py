"""Reading-consistency gate: display G2P (LyricFA ZhG2p) and alignment G2P
(TiFA cpp-pinyin) must agree on the cantonese/mandarin defaults they were
given. Regression for the reported 只 zi2->zek3 misreading and the phrase-table
sharing that keeps both engines on one data set.

JyutpingConverter pins are skipped when the model bundle (jyutping_dict.txt)
is absent; the ZhG2p pins run anywhere (LyricFA dicts are vendored).
"""
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "tifa-1.0-onnx"
JYUTPING_DICT = MODEL_DIR / "dictionaries" / "jyutping_dict.txt"


def _zhg2p(lang):
    from inference.LyricFA.tools.ZhG2p import ZhG2p

    return ZhG2p(lang)


def _jyutping_converter():
    from inference.TiFA.g2p.converters.chinese import JyutpingConverter

    return JyutpingConverter(dict_path=str(JYUTPING_DICT))


def test_zhg2p_cantonese_zhi_default_is_zi():
    g2p = _zhg2p("cantonese")
    assert g2p.convert("只", include_tone=False).split() == ["zi"]
    assert g2p.convert("只剩", include_tone=False).split() == ["zi", "sing"]


def test_zhg2p_cantonese_measure_word_phrases():
    g2p = _zhg2p("cantonese")
    assert g2p.convert("一只", include_tone=False).split() == ["jat", "zek"]
    assert g2p.convert("一隻", include_tone=False).split() == ["jat", "zek"]


def test_zhg2p_cantonese_traditional_phrase_reaches_matching():
    g2p = _zhg2p("cantonese")
    # 壓歲錢 rides per-char defaults; 一隻 must hit the shared phrase entry
    assert g2p.convert("一隻", include_tone=False).split() == ["jat", "zek"]


@pytest.mark.skipif(not JYUTPING_DICT.is_file(),
                    reason="tifa-1.0-onnx model bundle not present")
def test_jyutping_converter_agrees_with_display_g2p():
    conv = _jyutping_converter()
    assert conv.text_to_scripts(["只"]) == [["zi"]]
    # measure word: phrase primary zek, per-char zi stays as a DP candidate
    assert conv.text_to_scripts(list("一只")) == [["jat"], ["zek", "zi"]]
    assert conv.text_to_scripts(list("一隻")) == [["jat"], ["zek"]]
    # phrase table revival: simplified-keyed entries must match post-simplify
    assert conv.text_to_scripts(list("压岁钱")) == [
        ["aat", "ngaat", "ngaat"], ["seoi"], ["cin", "zin"]]


@pytest.mark.skipif(not (MODEL_DIR / "dictionaries" / "ds-zh-pinyin-lite.txt").is_file(),
                    reason="tifa-1.0-onnx model bundle not present")
def test_mandarin_first_readings_agree():
    """The mandarin audit found zero first-reading disagreements between the
    two engines; pin a few common polyphones so it stays that way."""
    g2p = _zhg2p("mandarin")
    from inference.TiFA.g2p.converters.chinese import PinyinConverter
    converter = PinyinConverter(dict_path=str(
        MODEL_DIR / "dictionaries" / "ds-zh-pinyin-lite.txt"))
    for ch in ["了", "还", "重", "地", "得"]:
        display = g2p.convert(ch, include_tone=True).split()
        aligned = converter.text_to_scripts([ch])[0]
        assert display, f"ZhG2p mandarin gave no reading for {ch}"
        assert aligned, f"cpp-pinyin mandarin gave no reading for {ch}"
        base = lambda s: s.rstrip("012345")
        assert base(display[0]) == base(aligned[0]), (
            f"mandarin default disagreement for {ch}: "
            f"display={display[0]} alignment={aligned[0]}")
