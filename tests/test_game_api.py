import numpy as np
import pytest

from inference.API.game_api import extract_pitches_only_torch, extract_vowel_boundaries
from inference.HubertFA.tools.align_word import Phoneme, Word


class _FailingGameModel:
    timestep = 0.01

    def infer_batch(self, **kwargs):
        raise RuntimeError("GAME exploded")


class _CapturingGameModel:
    timestep = 0.01

    def __init__(self):
        self.calls = []

    def infer_batch(self, **kwargs):
        self.calls.append(kwargs)
        return [
            (
                np.array([0.5], dtype=np.float32),
                np.array([1], dtype=np.int64),
                np.array([60.0], dtype=np.float32),
            )
        ]


def _make_word(start, end, text, phones):
    word = Word(start, end, text)
    word.phonemes = [Phoneme(ph_start, ph_end, ph_text) for ph_start, ph_end, ph_text in phones]
    return word


def test_no_lyrics_game_inference_error_is_not_swallowed():
    chunks = [{"waveform": np.zeros(32, dtype=np.float32), "offset": 0.0}]

    with pytest.raises(RuntimeError, match="GAME exploded"):
        extract_pitches_only_torch(
            chunks,
            sr=16000,
            game_model=_FailingGameModel(),
            device="cpu",
            ts=[0.0],
            seg_threshold=0.2,
            seg_radius=0.02,
            est_threshold=0.2,
            batch_size=1,
        )


def test_game_language_is_forced_empty_semantics():
    chunks = [{"waveform": np.zeros(32, dtype=np.float32), "offset": 0.0}]
    model = _CapturingGameModel()

    notes = extract_pitches_only_torch(
        chunks,
        sr=16000,
        game_model=model,
        device="cpu",
        ts=[0.0],
        seg_threshold=0.2,
        seg_radius=0.02,
        est_threshold=0.2,
        batch_size=1,
        language="ja",
    )

    assert len(notes) == 1
    assert len(model.calls) == 1
    assert model.calls[0]["language"] == 0


def test_extract_vowel_boundaries_ja_uses_first_singable_nucleus():
    words = [
        _make_word(0.0, 0.25, "ka", [(0.0, 0.08, "k"), (0.08, 0.25, "a")]),
        _make_word(0.25, 0.50, "ni", [(0.25, 0.33, "n"), (0.33, 0.50, "i")]),
    ]

    word_durs, word_vuvs, lyrics = extract_vowel_boundaries(words, ["か", "に"], language="ja")

    assert word_durs == pytest.approx([0.08, 0.25, 0.17])
    assert word_vuvs == [0, 1, 1]
    assert lyrics == ["", "か", "に"]


def test_extract_vowel_boundaries_zh_does_not_use_coda_as_vowel_start():
    words = [
        _make_word(0.0, 0.30, "ang", [(0.0, 0.22, "a"), (0.22, 0.30, "ng")]),
        _make_word(0.30, 0.60, "ni", [(0.30, 0.36, "n"), (0.36, 0.60, "i")]),
    ]

    word_durs, word_vuvs, lyrics = extract_vowel_boundaries(words, ["昂", "你"], language="zh")

    assert word_durs == pytest.approx([0.36, 0.24])
    assert word_vuvs == [1, 1]
    assert lyrics == ["昂", "你"]
