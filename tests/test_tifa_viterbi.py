"""Known-answer tests for the vendored TiFA Viterbi decoder."""
import numpy as np

from inference.TiFA.decoding import decode_alignment_flat


def test_clear_separation_produces_contiguous_spans():
    T, N = 20, 2
    sim = np.zeros((1, T, N), dtype=np.float32)
    sim[0, :10, 0] = 1.0
    sim[0, 10:, 1] = 1.0

    spans = decode_alignment_flat(
        sim,
        np.array([T], dtype=np.int64),
        np.array([N], dtype=np.int64),
        skip_penalty=0.5,
    )

    assert spans[0, 0].tolist() == [0, 10]
    assert spans[0, 1].tolist() == [10, 20]


def test_unmatched_token_is_skipped_zero_width():
    T, N = 30, 3
    sim = np.full((1, T, N), -1.0, dtype=np.float32)
    sim[0, :10, 0] = 1.0
    sim[0, 20:, 2] = 1.0
    # token 1 has no matching frames anywhere: emitting costs more than skipping

    spans = decode_alignment_flat(
        sim,
        np.array([T], dtype=np.int64),
        np.array([N], dtype=np.int64),
        skip_penalty=0.5,
    )

    assert spans[0, 0].tolist() == [0, 10]
    assert spans[0, 1, 0] == spans[0, 1, 1]  # zero-width skip
    assert spans[0, 2].tolist() == [20, 30]


def test_group_forces_no_gap_between_tokens():
    T, N = 20, 2
    sim = np.zeros((1, T, N), dtype=np.float32)
    sim[0, :5, 0] = 1.0
    sim[0, 15:, 1] = 1.0
    groups = np.array([[1, 1]], dtype=np.int64)  # same group: gap forbidden

    spans = decode_alignment_flat(
        sim,
        np.array([T], dtype=np.int64),
        np.array([N], dtype=np.int64),
        groups=groups,
        skip_penalty=0.5,
    )

    assert spans[0, 0, 1] == spans[0, 1, 0]  # token 1 starts exactly at token 0's end


def test_batch_with_varying_lengths():
    sim = np.zeros((2, 20, 2), dtype=np.float32)
    sim[0, :10, 0] = 1.0
    sim[0, 10:, 1] = 1.0
    sim[1, :4, 0] = 1.0
    sim[1, 4:8, 1] = 1.0

    spans = decode_alignment_flat(
        sim,
        np.array([20, 8], dtype=np.int64),
        np.array([2, 2], dtype=np.int64),
        skip_penalty=0.5,
    )

    assert spans[0, 0].tolist() == [0, 10]
    assert spans[1, 0].tolist() == [0, 4]
    assert spans[1, 1].tolist() == [4, 8]
