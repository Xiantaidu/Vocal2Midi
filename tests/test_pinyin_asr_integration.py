"""Tests for wiring the direct pinyin ASR into the Chinese lyric flow.

Convention: language '中文-拼音' / 'zh-pinyin' routes lyric extraction through
the pinyin ONNX CTC model; every other language keeps the existing engines
(ja romaji ASR, otherwise Qwen text ASR).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.API import asr_api
from inference.API.lfa_api import (
    _normalize_lyric_output_mode,
    create_lyric_matcher,
    process_asr_to_phonemes,
)
from inference.io.note_io import NoteInfo
from inference.pinyin_asr.runtime import PinyinASROnnxModel, resolve_model_dir
from inference.pipeline import auto_lyric_hybrid as pipeline


# --- language routing helpers ---

def test_normalize_pipeline_language_maps_chinese_pinyin():
    assert pipeline.normalize_pipeline_language("中文-拼音") == ("zh-pinyin", True)
    assert pipeline.normalize_pipeline_language("zh-pinyin") == ("zh-pinyin", True)
    assert pipeline.normalize_pipeline_language("zh") == ("zh", False)
    assert pipeline.normalize_pipeline_language("ja") == ("ja", False)
    assert pipeline.normalize_pipeline_language("") == ("zh", False)
    assert pipeline.normalize_pipeline_language(None) == ("zh", False)


def test_zh_pinyin_output_mode_is_forced_to_pinyin():
    assert _normalize_lyric_output_mode("zh-pinyin", "hanzi") == "pinyin"
    assert _normalize_lyric_output_mode("zh-pinyin", "拼音") == "pinyin"
    assert _normalize_lyric_output_mode("zh-pinyin", None) == "pinyin"
    # plain zh keeps both modes
    assert _normalize_lyric_output_mode("zh", "hanzi") == "hanzi"


def test_pinyin_asr_path_resolution(tmp_path):
    assert pipeline._select_pinyin_asr_path("custom/dir") == "custom/dir"
    default = pipeline._select_pinyin_asr_path("")
    if pipeline.PINYIN_ASR_DEFAULT_DIR.exists():
        assert default == str(pipeline.PINYIN_ASR_DEFAULT_DIR)
    else:
        assert default is None


# --- pipeline engine routing ---

def _base_kwargs(tmp_path: Path) -> dict:
    return {
        "audio_path": "input.wav",
        "output_filename": "Song.WAV",
        "game_model_dir": "game",
        "device": "cpu",
        "hfa_model_dir": "hfa",
        "asr_model_path": "asr",
        "ts": [0.0],
        "language": "zh",
        "lyric_output_mode": "auto",
        "original_lyrics": "",
        "output_dir": tmp_path / "out",
        "output_formats": [],
        "slice_min_sec": 8.0,
        "slice_max_sec": 22.0,
        "slicing_method": "default",
        "tempo": 120.0,
        "quantization_step": 0,
        "pitch_format": "name",
        "round_pitch": True,
        "quantization_mode": "simple",
        "seg_threshold": 0.2,
        "seg_radius": 0.02,
        "est_threshold": 0.2,
        "batch_size": 2,
        "asr_batch_size": 2,
    }


def _patch_common(monkeypatch):
    chunks = [{"waveform": np.zeros(1600, dtype=np.float32), "offset": 0.0}]
    monkeypatch.setattr(
        pipeline, "load_audio", lambda *a, **k: (np.zeros(4410, dtype=np.float32), 44100)
    )
    monkeypatch.setattr(pipeline, "slice_audio", lambda *a, **k: chunks)
    monkeypatch.setattr(pipeline, "free_memory", lambda: None)
    monkeypatch.setattr(pipeline, "load_game_model", lambda *a, **k: MagicMock())
    return chunks


def _patch_alignment_success(monkeypatch):
    monkeypatch.setattr(pipeline, "load_hfa_model", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        pipeline, "run_hubert_fa", lambda *a, **k: {"chunk_0": (None, None, [MagicMock()])}
    )
    monkeypatch.setattr(pipeline, "export_hfa_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline,
        "extract_pitches_and_align",
        lambda *a, **k: ([NoteInfo(0.0, 0.5, 60.0, "a")], {0}),
    )


def test_zh_pinyin_uses_pinyin_asr_and_zh_hfa(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_select_pinyin_asr_path", lambda path: "pinyin")
    run_pinyin = MagicMock(return_value=({"chunk_0": ["ni"]}, ["log"]))
    run_romaji = MagicMock()
    run_qwen = MagicMock()
    monkeypatch.setattr(pipeline, "run_pinyin_asr", run_pinyin)
    monkeypatch.setattr(pipeline, "run_romaji_asr", run_romaji)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", run_qwen)
    _patch_alignment_success(monkeypatch)
    run_hfa = MagicMock(
        side_effect=lambda *a, **k: {"chunk_0": (None, None, [MagicMock()])}
    )
    monkeypatch.setattr(pipeline, "run_hubert_fa", run_hfa)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "中文-拼音"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    run_pinyin.assert_called_once()
    run_qwen.assert_not_called()
    run_romaji.assert_not_called()
    assert run_pinyin.call_args.kwargs["asr_model_path"] == "pinyin"
    assert run_pinyin.call_args.kwargs["language"] == "zh-pinyin"
    assert run_pinyin.call_args.kwargs["lyric_output_mode"] == "pinyin"
    # HFA/GAME downstream must see the base language.
    assert run_hfa.call_args.kwargs["language"] == "zh"


def test_zh_pinyin_falls_back_to_qwen_when_model_missing(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_select_pinyin_asr_path", lambda path: None)
    run_pinyin = MagicMock()
    run_qwen = MagicMock(return_value=({"chunk_0": ["你"]}, ["log"]))
    monkeypatch.setattr(pipeline, "run_pinyin_asr", run_pinyin)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", run_qwen)
    _patch_alignment_success(monkeypatch)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "zh-pinyin"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    run_pinyin.assert_not_called()
    run_qwen.assert_called_once()
    assert run_qwen.call_args.kwargs["language"] == "zh-pinyin"


def test_plain_zh_still_uses_qwen(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *a, **k: None)
    run_pinyin = MagicMock()
    run_qwen = MagicMock(return_value=({"chunk_0": ["你"]}, ["log"]))
    monkeypatch.setattr(pipeline, "run_pinyin_asr", run_pinyin)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", run_qwen)
    _patch_alignment_success(monkeypatch)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "zh"
    kwargs["lyric_output_mode"] = "hanzi"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    run_pinyin.assert_not_called()
    run_qwen.assert_called_once()


# --- pinyin token processing (lfa) ---

def test_direct_pinyin_tokens_without_matcher(tmp_path):
    results = [{"text": "ni hao", "phonemes": ["ni", "hao"]}]
    chars, logs = process_asr_to_phonemes(
        results,
        [0],
        tmp_path,
        "zh-pinyin",
        None,
        lyric_output_mode="pinyin",
        use_asr_phonemes=True,
    )

    assert chars == {"chunk_0": ["ni", "hao"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "ni hao"
    assert "Direct pinyin ASR" in logs[0]


def test_direct_pinyin_tokens_are_not_mora_joined(tmp_path):
    # A bare nasal followed by a vowel must stay two syllable tokens
    # (the Japanese mora-join pass would merge them into "na").
    results = [{"text": "n a", "phonemes": ["n", "a"]}]
    process_asr_to_phonemes(
        results,
        [0],
        tmp_path,
        "zh-pinyin",
        None,
        lyric_output_mode="pinyin",
        use_asr_phonemes=True,
    )

    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "n a"


def test_direct_pinyin_tokens_match_reference_lyrics(tmp_path):
    matcher = create_lyric_matcher("zh", "你好世界")
    results = [{"text": "ni hao", "phonemes": ["ni", "hao"]}]
    chars, logs = process_asr_to_phonemes(
        results,
        [0],
        tmp_path,
        "zh-pinyin",
        matcher,
        lyric_output_mode="pinyin",
        use_asr_phonemes=True,
    )

    assert chars == {"chunk_0": ["ni", "hao"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "ni hao"
    assert "Matched original lyrics" in logs[0]
    assert "你 好" in logs[0]


def test_empty_pinyin_result_is_skipped(tmp_path):
    chars, logs = process_asr_to_phonemes(
        [None],
        [0],
        tmp_path,
        "zh-pinyin",
        None,
        lyric_output_mode="pinyin",
        use_asr_phonemes=True,
    )
    assert chars == {}
    assert "Ignored" in logs[0]


# --- asr_api batching ---

def test_batch_transcribe_pinyin_asr_writes_chunks_and_returns_results(monkeypatch, tmp_path):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [
        {"text": "ni hao", "phonemes": ["ni", "hao"]},
        {"text": "shi jie", "phonemes": ["shi", "jie"]},
    ]
    monkeypatch.setattr(
        asr_api, "load_pinyin_asr_model", lambda *a, **k: {"model": fake_model}
    )

    chunks = [
        {"waveform": np.zeros(100, dtype=np.float32)},
        {"waveform": np.zeros(100, dtype=np.float32)},
    ]
    results, indices = asr_api.batch_transcribe_pinyin_asr(
        chunks, 16000, tmp_path, "model_dir", asr_batch_size=2
    )

    assert indices == [0, 1]
    assert results[0]["phonemes"] == ["ni", "hao"]
    assert (tmp_path / "chunk_0.wav").exists()
    assert (tmp_path / "chunk_1.wav").exists()
    fake_model.transcribe.assert_called_once()


# --- runtime model file resolution / real ONNX smoke ---

def test_resolve_model_dir_accepts_dir_and_onnx_file(tmp_path):
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    (model_dir / "model_fp16.onnx").write_bytes(b"x")
    assert resolve_model_dir(model_dir) == model_dir
    assert resolve_model_dir(model_dir / "model_fp16.onnx") == model_dir

    with pytest.raises(FileNotFoundError):
        resolve_model_dir(tmp_path / "missing")


def test_from_model_path_reports_missing_files(tmp_path):
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        PinyinASROnnxModel.from_model_path(model_dir)

    (model_dir / "model_fp16.onnx").write_bytes(b"not a real model")
    with pytest.raises(FileNotFoundError):
        PinyinASROnnxModel.from_model_path(model_dir)


def test_real_pinyin_onnx_transcribe_structure():
    """Loads the real exp26b bundle and checks the transcription result shape."""
    default_dir = Path(__file__).resolve().parents[1] / "experiments" / "pinyinASR"
    model_file = default_dir / "model_fp16_dynb.onnx"
    if not model_file.exists():
        pytest.skip("pinyinASR dynamic-batch model bundle not present")

    model = PinyinASROnnxModel.from_model_path(default_dir, device="cpu")
    assert model.sample_rate == 16000
    assert model.id2token[model.blank_id] in {"<blank>", "PAD"}
    # The dynb export carries a symbolic batch axis; no fixed-batch fallback.
    assert model.fixed_batch_size is None

    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    wave = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav_a = default_dir.parent / "_pinyin_asr_test_tone_a.wav"
    wav_b = default_dir.parent / "_pinyin_asr_test_tone_b.wav"
    try:
        sf.write(wav_a, wave, sr)
        sf.write(wav_b, wave * 0.5, sr)

        batched = model.transcribe([str(wav_a), str(wav_b)], batch_size=2)
        single = model.transcribe([str(wav_a)], batch_size=1)
    finally:
        wav_a.unlink(missing_ok=True)
        wav_b.unlink(missing_ok=True)

    assert len(batched) == 2
    assert isinstance(batched[0]["text"], str)
    assert isinstance(batched[0]["phonemes"], list)
    # Identical audio must decode identically whether batched or not.
    assert batched[0]["phonemes"] == single[0]["phonemes"]
