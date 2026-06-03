from pathlib import Path
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.io.note_io import NoteInfo
from inference.pipeline import auto_lyric_hybrid as pipeline


def _patch_librosa_load(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "librosa",
        types.SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(4410, dtype=np.float32), 44100)),
    )


def _base_kwargs(tmp_path: Path) -> dict:
    return {
        "audio_path": "input.wav",
        "output_filename": "Song.WAV",
        "game_model_dir": "game",
        "device": "cpu",
        "hfa_model_dir": "hfa",
        "asr_model_path": "asr",
        "ts": [0.0],
        "language": "ja",
        "lyric_output_mode": "auto",
        "original_lyrics": "",
        "output_dir": tmp_path / "out",
        "output_formats": [],
        "slice_min_sec": 8.0,
        "slice_max_sec": 22.0,
        "slicing_method": "默认切片",
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
    _patch_librosa_load(monkeypatch)
    monkeypatch.setattr(pipeline, "slice_audio", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(pipeline, "free_memory", lambda: None)
    monkeypatch.setattr(pipeline, "load_game_model", lambda *args, **kwargs: MagicMock())
    return chunks


def test_ja_romaji_mode_uses_mora_asr(monkeypatch, tmp_path):
    chunks = _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_select_romaji_asr_path", lambda path: "phoneme")
    run_phoneme = MagicMock(return_value=({"chunk_0": ["a"]}, ["log"]))
    run_qwen = MagicMock(return_value=({"chunk_0": ["a"]}, ["log"]))
    monkeypatch.setattr(pipeline, "run_romaji_asr", run_phoneme)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", run_qwen)
    monkeypatch.setattr(pipeline, "load_hfa_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "run_hubert_fa", lambda *args, **kwargs: {"chunk_0": (None, None, [MagicMock()])})
    monkeypatch.setattr(pipeline, "export_hfa_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "extract_pitches_and_align_torch",
        lambda *args, **kwargs: ([NoteInfo(0.0, 0.5, 60.0, "a")], {0}),
    )

    kwargs = _base_kwargs(tmp_path)

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    run_phoneme.assert_called_once()
    run_qwen.assert_not_called()
    assert run_phoneme.call_args.kwargs["lyric_output_mode"] == "romaji"
    assert run_phoneme.call_args.args[0] is chunks


def test_ja_kana_mode_uses_mora_asr_and_dictionary_hfa(monkeypatch, tmp_path):
    chunks = _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_select_romaji_asr_path", lambda path: "phoneme")

    run_phoneme = MagicMock(return_value=({"chunk_0": ["a"]}, ["log"]))
    load_hfa = MagicMock(return_value=MagicMock())
    run_hfa = MagicMock(return_value={"chunk_0": (None, None, [MagicMock()])})

    monkeypatch.setattr(pipeline, "run_romaji_asr", run_phoneme)
    monkeypatch.setattr(pipeline, "load_hfa_model", load_hfa)
    monkeypatch.setattr(pipeline, "run_hubert_fa", run_hfa)
    monkeypatch.setattr(pipeline, "export_hfa_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "extract_pitches_and_align_torch",
        lambda *args, **kwargs: ([NoteInfo(0.0, 0.5, 60.0, "a")], {0}),
    )

    kwargs = _base_kwargs(tmp_path)
    kwargs["lyric_output_mode"] = "kana"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    run_phoneme.assert_called_once()
    load_hfa.assert_called_once()
    run_hfa.assert_called_once()
    assert run_hfa.call_args.kwargs.get("use_phoneme_g2p", False) is False


def test_no_lyrics_mode_skips_asr_and_hfa(monkeypatch, tmp_path):
    chunks = _patch_common(monkeypatch)
    run_qwen = MagicMock()
    run_phoneme = MagicMock()
    load_hfa = MagicMock()
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", run_qwen)
    monkeypatch.setattr(pipeline, "run_romaji_asr", run_phoneme)
    monkeypatch.setattr(pipeline, "load_hfa_model", load_hfa)
    extract_only = MagicMock(return_value=[NoteInfo(0.0, 0.5, 60.0, "")])
    monkeypatch.setattr(pipeline, "extract_pitches_only_torch", extract_only)

    kwargs = _base_kwargs(tmp_path)
    kwargs["output_lyrics"] = False
    kwargs["output_formats"] = ["TXT"]
    with patch.object(pipeline, "_save_text") as save_text:
        pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    run_qwen.assert_not_called()
    run_phoneme.assert_not_called()
    load_hfa.assert_not_called()
    extract_only.assert_called_once()
    assert extract_only.call_args.args[0] is chunks
    save_text.assert_called_once()
    assert (tmp_path / "out").is_dir()


def test_invalid_batch_sizes_fail_fast(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["batch_size"] = 0

    with pytest.raises(ValueError, match="batch_size"):
        pipeline.auto_lyric_hybrid_pipeline(**kwargs)


def test_invalid_slice_bounds_fail_fast(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["slice_min_sec"] = 12.0
    kwargs["slice_max_sec"] = 6.0

    with pytest.raises(ValueError, match="slice_min_sec"):
        pipeline.auto_lyric_hybrid_pipeline(**kwargs)


def test_empty_chunks_fail_before_models(monkeypatch, tmp_path):
    _patch_librosa_load(monkeypatch)
    monkeypatch.setattr(pipeline, "slice_audio", lambda *args, **kwargs: [])
    load_game = MagicMock()
    monkeypatch.setattr(pipeline, "load_game_model", load_game)

    kwargs = _base_kwargs(tmp_path)

    with pytest.raises(RuntimeError, match="切片阶段"):
        pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    load_game.assert_not_called()


def test_empty_hfa_predictions_fall_back_to_pitch_only(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", lambda *args, **kwargs: ({"chunk_0": ["a"]}, ["log"]))
    monkeypatch.setattr(pipeline, "load_hfa_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "run_hubert_fa", lambda *args, **kwargs: {})
    extract_aligned = MagicMock(return_value=([NoteInfo(0.0, 0.5, 60.0, "a")], {0}))
    extract_only = MagicMock(return_value=[NoteInfo(0.0, 0.5, 60.0, "")])
    monkeypatch.setattr(pipeline, "extract_pitches_and_align_torch", extract_aligned)
    monkeypatch.setattr(pipeline, "extract_pitches_only_torch", extract_only)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "zh"
    kwargs["lyric_output_mode"] = "hanzi"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    extract_aligned.assert_not_called()
    extract_only.assert_called_once()


def test_empty_asr_results_fall_back_to_pitch_only(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", lambda *args, **kwargs: ({}, ["empty log"]))
    load_hfa = MagicMock()
    extract_only = MagicMock(return_value=[NoteInfo(0.0, 0.5, 60.0, "")])
    monkeypatch.setattr(pipeline, "load_hfa_model", load_hfa)
    monkeypatch.setattr(pipeline, "extract_pitches_only_torch", extract_only)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "zh"
    kwargs["lyric_output_mode"] = "hanzi"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    load_hfa.assert_not_called()
    extract_only.assert_called_once()


def test_missing_hfa_chunk_uses_pitch_only_fallback(monkeypatch, tmp_path):
    chunks = [
        {"waveform": np.zeros(1600, dtype=np.float32), "offset": 0.0},
        {"waveform": np.zeros(1600, dtype=np.float32), "offset": 1.0},
    ]
    _patch_librosa_load(monkeypatch)
    monkeypatch.setattr(pipeline, "slice_audio", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(pipeline, "free_memory", lambda: None)
    monkeypatch.setattr(pipeline, "load_game_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_qwen_asr_and_fa",
        lambda *args, **kwargs: ({"chunk_0": ["a"], "chunk_1": ["i"]}, ["log0", "log1"]),
    )
    monkeypatch.setattr(pipeline, "load_hfa_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "run_hubert_fa", lambda *args, **kwargs: {"chunk_0": (None, None, [MagicMock()])})
    monkeypatch.setattr(pipeline, "export_hfa_artifacts", lambda *args, **kwargs: None)
    extract_aligned = MagicMock(return_value=([NoteInfo(0.0, 0.5, 60.0, "a")], {0}))
    extract_only = MagicMock(return_value=[NoteInfo(1.0, 1.5, 62.0, "")])
    monkeypatch.setattr(pipeline, "extract_pitches_and_align_torch", extract_aligned)
    monkeypatch.setattr(pipeline, "extract_pitches_only_torch", extract_only)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "zh"
    kwargs["lyric_output_mode"] = "hanzi"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    extract_aligned.assert_called_once()
    extract_only.assert_called_once()
    assert extract_only.call_args.args[0] == [chunks[1]]


def test_unproductive_aligned_chunk_uses_pitch_only_fallback(monkeypatch, tmp_path):
    chunks = [{"waveform": np.zeros(1600, dtype=np.float32), "offset": 0.0}]
    _patch_librosa_load(monkeypatch)
    monkeypatch.setattr(pipeline, "slice_audio", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(pipeline, "free_memory", lambda: None)
    monkeypatch.setattr(pipeline, "load_game_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "create_lyric_matcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "run_qwen_asr_and_fa", lambda *args, **kwargs: ({"chunk_0": ["a"]}, ["log"]))
    monkeypatch.setattr(pipeline, "load_hfa_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(pipeline, "run_hubert_fa", lambda *args, **kwargs: {"chunk_0": (None, None, [MagicMock()])})
    monkeypatch.setattr(pipeline, "export_hfa_artifacts", lambda *args, **kwargs: None)
    extract_aligned = MagicMock(return_value=([], set()))
    extract_only = MagicMock(return_value=[NoteInfo(0.0, 0.5, 62.0, "")])
    monkeypatch.setattr(pipeline, "extract_pitches_and_align_torch", extract_aligned)
    monkeypatch.setattr(pipeline, "extract_pitches_only_torch", extract_only)

    kwargs = _base_kwargs(tmp_path)
    kwargs["language"] = "zh"
    kwargs["lyric_output_mode"] = "hanzi"

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    extract_aligned.assert_called_once()
    extract_only.assert_called_once()
    assert extract_only.call_args.args[0] == chunks


def test_slice_bounds_are_forwarded_to_slicer(monkeypatch, tmp_path):
    chunks = [{"waveform": np.zeros(1600, dtype=np.float32), "offset": 0.0}]
    slice_audio = MagicMock(return_value=chunks)
    _patch_librosa_load(monkeypatch)
    monkeypatch.setattr(pipeline, "slice_audio", slice_audio)
    monkeypatch.setattr(pipeline, "free_memory", lambda: None)
    monkeypatch.setattr(pipeline, "load_game_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(
        pipeline,
        "extract_pitches_only_torch",
        lambda *args, **kwargs: [NoteInfo(0.0, 0.5, 60.0, "")],
    )

    kwargs = _base_kwargs(tmp_path)
    kwargs["output_lyrics"] = False
    kwargs["slice_min_sec"] = 5.0
    kwargs["slice_max_sec"] = 17.5

    pipeline.auto_lyric_hybrid_pipeline(**kwargs)

    assert slice_audio.call_args.kwargs["min_len_sec"] == 5.0
    assert slice_audio.call_args.kwargs["max_len_sec"] == 17.5
