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

    # particle edges after a word boundary read as pronunciation
    assert normalize_edge_reading("は", "は", prev_surface="私") == "わ"
    assert normalize_edge_reading("へ", "へ", prev_surface="東京") == "え"
    assert normalize_edge_reading("はな", "はな") == "はな"
    assert normalize_edge_reading("歯", "は") == "は"
    # char-level kana runs carry no word-boundary signal: a standalone kana
    # は is more often word-internal (はんせん) than a particle, and the mora
    # ASR emits orthographic ha anyway
    assert normalize_edge_reading("は", "は") == "は"
    assert normalize_edge_reading("は", "は", prev_surface="に") == "は"


def test_analyze_lyric_text_keeps_kana_and_romaji_aligned(runtime):
    from inference.LyricFA.tools.JaG2p import JaG2p

    g2p = JaG2p()
    kana, romaji = g2p.analyze_lyric_text("感情線を抜けて 反戦国 ICBM ラーメン")

    # the lyric matcher consumes these as parallel arrays and indexes the
    # kana list by phonetic position: equal length is the core invariant
    assert len(kana) == len(romaji)
    joined = " ".join(romaji)
    # digraph halves re-merged: じょ is one mora "jo", never ji+yo
    assert "jo" in romaji and "ji yo" not in joined
    # は in 反戦国 keeps the orthographic reading (per-mora conversion used to
    # rewrite every standalone は to わ, giving "wa n se n")
    assert "ha n se n" in joined and "wa n se n" not in joined
    # latin resolves through the model's alnum table instead of leaking the
    # raw non-phoneme token into the phonetics
    assert "icbm" not in romaji and "bi" in romaji
    # prolonged mark repeats the preceding vowel (ラーメン -> ra a me n)
    assert "ra a me n" in joined


def test_particle_rewrite_applies_only_in_word_context(runtime):
    from inference.LyricFA.tools.JaG2p import JaG2p

    g2p = JaG2p()
    # full-text conversion: the kanji word edge before は marks the boundary,
    # so the particle still reads as わ
    assert g2p.convert("私は") == "wa ta shi wa"
    # unspaced kana text is read char-by-char: orthographic は throughout
    kana, romaji = g2p.analyze_lyric_text("くらぶはんせん")
    assert "ha n se n" in " ".join(romaji)


def test_ja_reference_lyric_lists_stay_aligned():
    from inference.LyricFA.tools.lyric_matcher import LyricMatcher

    matcher = LyricMatcher("ja")
    data = matcher.process_lyric_text("感情線を抜けて 反戦国 ICBM")
    assert len(data.text_list) == len(data.phonetic_list)
    assert "icbm" not in data.phonetic_list
    assert "wa n se n" not in " ".join(data.phonetic_list)


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
        assert word.readings
        groups = word.readings[0].paths[0]
        assert groups
        for group in groups:
            assert group.script and group.phonemes
    # scripts carry kana for per-mora display; phonemes are in-vocab CV pairs
    ordinary = next(word for word in words if word.text == "普通")
    assert [group.script for group in ordinary.readings[0].paths[0]] == ["ふ", "つ", "う"]

    for word in words:
        word.language = "ja"
    from inference.TiFA.lib.vocabulary import Vocabulary
    from inference.TiFA.aligner import GLOBAL_SYMBOLS, STOP_SYMBOLS

    vocabulary = Vocabulary.from_file(TIFA_MODEL_DIR / "vocabulary.json")
    data, lexicon, texts = encode_paths(
        words, vocabulary, "discard", languages=["ja"],
        global_symbols=GLOBAL_SYMBOLS, stop_symbols=STOP_SYMBOLS,
    )
    assert data["paths"].any()
    assert texts == ["普通", "の", "世界"]


def test_tifa_converter_beams_reading_candidates():
    """Beam N-best over the scored candidate graph feeds TiFA's polling DP:
    polyphone words carry multiple ranked readings, model-best first."""
    from inference.TiFA.japanese_onnx import JapaneseOnnxConverter

    conv = JapaneseOnnxConverter(
        dict_path=str(TIFA_MODEL_DIR / "dictionaries" / "japanese_dict_full.txt")
    )
    words = conv.convert("私は東京へ行く")
    by_text = {word.text: word for word in words}
    assert len(by_text["私"].readings) >= 2, "polyphone must carry candidates"
    assert [group.script for group in by_text["私"].readings[0].paths[0]] == ["わ", "た", "し"]
    assert len(by_text["行"].readings) >= 2, "okurigana stem must carry candidates"
    assert [group.script for group in by_text["行"].readings[0].paths[0]] == ["い"]
    # particle edges normalize to pronunciation
    assert [group.script for group in by_text["は"].readings[0].paths[0]] == ["わ"]
    assert [group.script for group in by_text["へ"].readings[0].paths[0]] == ["え"]


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
    # ONNX readings are beam-ranked candidates: the model's best first,
    # TiFA's scoring DP polls among them against the audio
    ordinary = next(word for word in words if word.text == "普通")
    assert ordinary.readings
    assert [group.script for group in ordinary.readings[0].paths[0]] == ["ふ", "つ", "う"]


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
