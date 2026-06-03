import numpy as np
import pytest

from inference.API.game_api import extract_pitches_only_torch


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
