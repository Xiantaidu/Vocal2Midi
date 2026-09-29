"""Real-model tests for the TiFA aligner (skipped without the ONNX bundle)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "tifa-1.0-onnx"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="tifa-1.0-onnx model bundle not present",
)


@pytest.fixture(scope="module")
def model():
    from inference.TiFA.runtime import load_tifa_model

    return load_tifa_model(MODEL_DIR, device="cpu")


def _write_tone(path: Path, seconds: float = 1.0, sr: int = 44100):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * 220 * t), sr)


def test_run_tifa_fa_end_to_end(tmp_path, model):
    _write_tone(tmp_path / "chunk_0.wav")
    (tmp_path / "chunk_0.txt").write_text("你好世界", encoding="utf-8")

    from inference.TiFA.aligner import run_tifa_fa

    pred_dict, display = run_tifa_fa(model, tmp_path, language="zh")

    assert "chunk_0" in pred_dict
    assert display == {}  # zh keeps the lfa display tokens
    wav_path, duration, words = pred_dict["chunk_0"]
    assert abs(duration - 1.0) < 0.05
    assert len(words) >= 1
    starts = [word.start for word in words]
    assert starts == sorted(starts), "word onsets must be monotonic"
    for word in words:
        assert 0 <= word.start < word.end <= duration + 0.2
        assert word.text
        for phoneme in word.phonemes:
            assert phoneme.start < phoneme.end


def test_run_tifa_fa_skips_unencodable_text(tmp_path, model):
    fugashi_missing = importlib.util.find_spec("fugashi") is None
    if not fugashi_missing:
        pytest.skip("fugashi installed; the unencodable fallback path cannot be exercised")
    _write_tone(tmp_path / "chunk_0.wav")
    # 们 is a Chinese-only character: absent from the ja lexicon pack, so no
    # converter can resolve it and the chunk must fall back to pitch-only.
    (tmp_path / "chunk_0.txt").write_text("们", encoding="utf-8")

    from inference.TiFA.aligner import run_tifa_fa

    pred_dict, display = run_tifa_fa(model, tmp_path, language="ja")
    assert pred_dict == {} and display == {}


def test_run_tifa_fa_respects_cancellation(tmp_path, model):
    _write_tone(tmp_path / "chunk_0.wav")
    (tmp_path / "chunk_0.txt").write_text("你好", encoding="utf-8")

    from inference.TiFA.aligner import run_tifa_fa

    with pytest.raises(InterruptedError):
        run_tifa_fa(model, tmp_path, language="zh", cancel_checker=lambda: True)


def test_textgrid_export_matches_hfa_layout(tmp_path, model):
    _write_tone(tmp_path / "chunk_0.wav")
    (tmp_path / "chunk_0.txt").write_text("你好", encoding="utf-8")

    from inference.TiFA.aligner import run_tifa_fa
    from inference.TiFA.textgrid import save_textgrids

    pred_dict, _display = run_tifa_fa(model, tmp_path, language="zh")
    predictions = [pred_dict["chunk_0"]]
    save_textgrids(predictions, tmp_path, "Song")

    tg_file = tmp_path / "Song_000.TextGrid"
    assert tg_file.is_file()
    content = tg_file.read_text(encoding="utf-8")
    assert 'name = "words"' in content
    assert 'name = "phones"' in content


def test_run_tifa_fa_kanji_lexicon_text(tmp_path, model):
    if not (MODEL_DIR / "dictionaries" / "ja_lexicon.txz").is_file():
        pytest.skip("ja_lexicon.txz not present")
    _write_tone(tmp_path / "chunk_0.wav", seconds=1.6)
    (tmp_path / "chunk_0.txt").write_text("普通の世界", encoding="utf-8")

    from inference.TiFA.aligner import run_tifa_fa

    pred_dict, display = run_tifa_fa(model, tmp_path, language="ja")

    assert "chunk_0" in pred_dict
    _, duration, words = pred_dict["chunk_0"]
    # mora-level romaji words paired with kana display tokens; the SP word
    # appended by add_SP consumes no lyric and has no display entry
    romaji = [word.text for word in words if word.text != "SP"]
    kana = [kana for _romaji, kana in display["chunk_0"]]
    assert "fu" in romaji and "tsu" in romaji, romaji
    assert "ふ" in kana and "つ" in kana, kana
    assert len(romaji) == len(kana), (romaji, kana)
    starts = [word.start for word in words]
    assert starts == sorted(starts)
    for word in words:
        assert 0 <= word.start < word.end <= duration + 0.2
