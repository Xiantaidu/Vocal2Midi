"""ja_g2p_onnx integration: shared loader, TiFA converter, HFA JaG2p backend.

Real-model tests skip when models/ja_g2p_onnx is absent. The model is the
primary Japanese G2P for both engines: kanji -> contextually disambiguated
kana -> romaji -> phonemes via japanese_dict_full.txt.
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "ja_g2p_onnx"
TIFA_MODEL_DIR = PROJECT_ROOT / "models" / "tifa-1.0-onnx"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "g2p_onnx_runtime.py").is_file(),
    reason="ja_g2p_onnx model bundle not present",
)


@pytest.fixture(scope="module")
def runtime():
    from inference.ja_onnx_g2p import get_ja_onnx_runtime

    rt = get_ja_onnx_runtime()
    assert rt is not None
    return rt


def test_runtime_disambiguates_polyphones(runtime):
    out = runtime.predict("今日はいい天気ですね")
    assert out["reading"].startswith("きょう")


def test_runtime_reads_digit_strings(runtime):
    assert runtime.predict("12年")["reading"].startswith("じゅうに")


def test_particle_edge_reading_is_normalized():
    from inference.ja_onnx_g2p import normalize_edge_reading

    assert normalize_edge_reading("は", "は") == "わ"
    assert normalize_edge_reading("へ", "へ") == "え"
    assert normalize_edge_reading("はな", "はな") == "はな"
    assert normalize_edge_reading("歯", "は") == "は"


def test_tifa_converter_claims_kanji_runs():
    from inference.TiFA.japanese_onnx import JapaneseOnnxConverter

    conv = JapaneseOnnxConverter(
        dict_path=str(TIFA_MODEL_DIR / "dictionaries" / "japanese_dict_full.txt")
    )
    assert conv.find("普通の世界") == (0, 5)
    assert conv.find("お母さん") == (1, 4)
    assert conv.find("あいしてる") is None
    assert conv.find("hello") is None


def test_tifa_converter_emits_word_per_edge_with_kana_scripts():
    from inference.TiFA.g2p.encoding import encode_paths
    from inference.TiFA.japanese_onnx import JapaneseOnnxConverter

    conv = JapaneseOnnxConverter(
        dict_path=str(TIFA_MODEL_DIR / "dictionaries" / "japanese_dict_full.txt")
    )
    words = conv.convert("普通の世界")
    assert [word.text for word in words] == ["普通", "の", "世界"]
    for word in words:
        assert len(word.readings) == 1
        groups = word.readings[0].paths[0]
        assert groups
        for group in groups:
            assert group.script and group.phonemes
    # scripts carry kana for per-mora display; phonemes are in-vocab CV pairs
    texts = [word.text for word in words]
    ordinary = next(word for word in words if word.text == "普通")
    assert [group.script for group in ordinary.readings[0].paths[0]] == ["ふ", "つ", "う"]

    from inference.TiFA.lib.vocabulary import Vocabulary
    from inference.TiFA.aligner import GLOBAL_SYMBOLS, STOP_SYMBOLS

    for word in words:
        word.language = "ja"
    vocabulary = Vocabulary.from_file(TIFA_MODEL_DIR / "vocabulary.json")
    data, lexicon, texts = encode_paths(
        words, vocabulary, "discard", languages=["ja"],
        global_symbols=GLOBAL_SYMBOLS, stop_symbols=STOP_SYMBOLS,
    )
    assert data["paths"].any()
    assert texts == ["普通", "の", "世界"]


def test_tifa_converter_expands_katakana_long_vowels():
    from inference.TiFA.japanese_onnx import JapaneseOnnxConverter

    conv = JapaneseOnnxConverter(
        dict_path=str(TIFA_MODEL_DIR / "dictionaries" / "japanese_dict_full.txt")
    )
    words = conv.convert("東京タワー")
    by_text = {word.text: word for word in words}
    # the runtime splits unknown katakana per character (タ/ワ/ー); the ー
    # edge must become the preceding mora's vowel, never dropped
    tower_moras = []
    for kana in ("タ", "ワ", "ー"):
        word = by_text.get(kana)
        if word is None:
            continue
        tower_moras.extend(group.script for group in word.readings[0].paths[0])
    assert tower_moras == ["タ", "ワ", "あ"]


def test_factory_prefers_onnx_converter_for_kanji():
    from inference.TiFA.g2p_pipeline import build_g2p_pipeline

    pipeline = build_g2p_pipeline(TIFA_MODEL_DIR)
    words = pipeline.convert("普通の世界", languages=["ja"])
    texts = [word.text for word in words]
    assert "普通" in texts and "世界" in texts
    # ONNX readings are single-candidate: the model itself disambiguates
    ordinary = next(word for word in words if word.text == "普通")
    assert len(ordinary.readings) == 1


def test_jag2p_uses_onnx_backend_for_kanji():
    from inference.LyricFA.tools.JaG2p import JaG2p

    g2p = JaG2p()
    # particle は must read as わ (wa), not orthographic ha
    assert g2p.convert("私は") == "wa ta shi wa"
    moras = g2p.convert("今日はいい天気ですね").split()
    assert moras[:3] == ["kyo", "u", "wa"]
    kana = g2p.split_kana_no_regex("東京タワー")
    assert kana[:4] == ["と", "う", "きょ", "う"]


def test_jag2p_falls_back_to_pyopenjtalk_without_onnx(monkeypatch):
    import importlib.util

    import inference.LyricFA.tools.JaG2p as jag2p_module

    monkeypatch.setattr(jag2p_module, "_get_onnx_runtime", lambda: None)
    g2p = jag2p_module.JaG2p()
    moras = g2p.convert("今日はいい天気ですね").split()
    assert moras, "pyopenjtalk fallback must still produce moras"
    if importlib.util.find_spec("pyopenjtalk") is not None:
        assert moras[:1] == ["kyo"]
