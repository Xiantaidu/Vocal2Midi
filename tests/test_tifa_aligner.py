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

    pred_dict = run_tifa_fa(model, tmp_path, language="zh")

    assert "chunk_0" in pred_dict
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
        pytest.skip("fugashi installed; the kanji fallback path cannot be exercised")
    _write_tone(tmp_path / "chunk_0.wav")
    # kanji without the optional MeCab stack: no converter can resolve it
    (tmp_path / "chunk_0.txt").write_text("薔薇", encoding="utf-8")

    from inference.TiFA.aligner import run_tifa_fa

    pred_dict = run_tifa_fa(model, tmp_path, language="ja")
    assert pred_dict == {}


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

    pred_dict = run_tifa_fa(model, tmp_path, language="zh")
    predictions = [pred_dict["chunk_0"]]
    save_textgrids(predictions, tmp_path, "Song")

    tg_file = tmp_path / "Song_000.TextGrid"
    assert tg_file.is_file()
    content = tg_file.read_text(encoding="utf-8")
    assert 'name = "words"' in content
    assert 'name = "phones"' in content
