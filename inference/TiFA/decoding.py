"""Viterbi alignment decoding for TiFA.

Vendored from openvpi/TIFA modules/decoding.py (numba core kept verbatim,
torch wrappers replaced with numpy-level functions). numba is already a
runtime dependency via librosa.
"""
from __future__ import annotations

import numba
import numpy as np

MAX_PREDS = 3  # max predecessors per state across all variants


# ---------------------------------------------------------------------------
# Generic Viterbi decoder
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def viterbi(
    emission: np.ndarray,
    predecessors: np.ndarray,
    num_preds: np.ndarray,
    p_init: np.ndarray,
) -> np.ndarray:
    """Generic single-sequence Viterbi decoder with sparse predecessors.

    emission[s, t]: additive score for state s at time t (higher = better).
    predecessors[s, :]: valid prior states for state s (padded with -1).
    num_preds[s]: number of valid entries in predecessors[s, :].
    p_init[s]: initial additive score for state s.

    Returns states[t]: best state at each time step.
    """
    S, T_ = emission.shape
    NEG_INF = np.float32(-1e9)

    dp_prev = p_init.copy()
    dp_cur = np.empty(S, dtype=np.float32)
    back = np.empty((T_, S), dtype=np.int32)

    for t in range(T_):
        if t > 0:
            dp_prev[:] = dp_cur
        for j in range(S):
            best_score = NEG_INF
            best_i = -1
            for k in range(num_preds[j]):
                i = predecessors[j, k]
                score = dp_prev[i]
                if score > best_score:
                    best_score = score
                    best_i = i
            dp_cur[j] = best_score + emission[j, t]
            back[t, j] = best_i

    states = np.empty(T_, dtype=np.int32)
    best_s = np.argmax(dp_cur)
    for t_ in range(T_ - 1, -1, -1):
        states[t_] = best_s
        best_s = back[t_, best_s]

    return states


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _extract_spans(states: np.ndarray, T: int, N: int) -> np.ndarray:
    """Extract per-token [onset, offset) spans from a state sequence.

    P_i is at index N+1+i. Unreached tokens get degenerate span (T-1, T-1).
    """
    spans = np.empty((N, 2), dtype=np.int64)
    for i in range(N):
        p_idx = N + 1 + i
        first = -1
        last = -1
        for t in range(T):
            if states[t] == p_idx:
                if first < 0:
                    first = t
                last = t
        if first >= 0:
            spans[i, 0] = first
            spans[i, 1] = last + 1
        else:
            spans[i, 0] = T
            spans[i, 1] = T
    return spans


@numba.njit(cache=True)
def _add_predecessor(
    predecessors: np.ndarray, num_preds: np.ndarray, state: int, pred: int,
) -> None:
    """Append a predecessor to a state's predecessor list."""
    k = num_preds[state]
    predecessors[state, k] = pred
    num_preds[state] = k + 1


@numba.njit(cache=True)
def _remove_predecessor(
    predecessors: np.ndarray, num_preds: np.ndarray, state: int, pred: int,
) -> None:
    """Remove a predecessor from a state's predecessor list."""
    k = num_preds[state]
    for idx in range(k):
        if predecessors[state, idx] == pred:
            for j in range(idx, k - 1):
                predecessors[state, j] = predecessors[state, j + 1]
            num_preds[state] = k - 1
            return


# ---------------------------------------------------------------------------
# State machine builder (flat or spaced)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _make_fa(sim: np.ndarray, spaced: bool):
    """Build FA state machine.

    State space (S = 2N + 1):
      G_i (0 <= i <= N):  indices 0 .. N       (gap states, emission 0)
      P_i (0 <= i < N):   indices N+1 .. 2N    (token states, emission sim[:, i])

    Transitions (all cost 0):
      G_i -> G_i       for all i           (stay in gap)
      G_i -> P_i       for i < N           (enter token i)
      P_i -> P_i       for all i           (stay in token i)
      P_i -> G_{i+1}   for i < N           (exit token i to next gap)
      P_{i-1} -> P_i   for i > 0           (advance to next token, skip G_i)

    When spaced=True, additionally for each even i (space token):
      G_i -> G_{i+1}  (skip space token i)
    """
    T_, N = sim.shape
    S = 2 * N + 1
    NEG_INF = np.float32(-1e9)

    emission = np.zeros((S, T_), dtype=np.float32)
    for i in range(N):
        emission[N + 1 + i, :] = sim[:, i]

    predecessors = np.full((S, MAX_PREDS), -1, dtype=np.int32)
    num_preds = np.zeros(S, dtype=np.int32)

    # G_0: from G_0 only
    _add_predecessor(predecessors, num_preds, 0, 0)

    # G_i (1 <= i <= N): from G_i, P_{i-1}, and optionally G_{i-1} for skip
    for i in range(1, N + 1):
        _add_predecessor(predecessors, num_preds, i, i)
        _add_predecessor(predecessors, num_preds, i, N + 1 + (i - 1))
        if spaced and (i - 1) % 2 == 0:
            _add_predecessor(predecessors, num_preds, i, i - 1)

    # P_0: from P_0 and G_0
    p0 = N + 1
    _add_predecessor(predecessors, num_preds, p0, p0)
    _add_predecessor(predecessors, num_preds, p0, 0)

    # P_i (1 <= i < N): from P_i, G_i, P_{i-1}
    for i in range(1, N):
        p_idx = N + 1 + i
        _add_predecessor(predecessors, num_preds, p_idx, p_idx)
        _add_predecessor(predecessors, num_preds, p_idx, i)
        _add_predecessor(predecessors, num_preds, p_idx, N + 1 + (i - 1))

    p_init = np.full(S, NEG_INF, dtype=np.float32)
    p_init[0] = np.float32(0.0)

    return emission, predecessors, num_preds, p_init


# ---------------------------------------------------------------------------
# Grouped modifiers
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_groups_flat(
    predecessors: np.ndarray, num_preds: np.ndarray, groups: np.ndarray, N: int,
) -> None:
    """Modify flat state machine for grouped variant.

    Deactivates G_i for 0 < i < N where groups[i-1] == groups[i].
    G_0 and G_N always remain active.
    """
    for i in range(1, N):
        if groups[i - 1] == groups[i]:
            num_preds[i] = 0
            _remove_predecessor(predecessors, num_preds, N + 1 + i, i)


@numba.njit(cache=True)
def _apply_groups_spaced(
    predecessors: np.ndarray,
    num_preds: np.ndarray,
    groups: np.ndarray,
    N: int,
) -> None:
    """Modify spaced state machine for grouped variant.

    For consecutive real tokens in the same group:
      - Deactivates the space token between them and its two gap states.
      - Adds a direct jump P_{real_prev} -> P_{real_next}.
      - Cleans up dead predecessor entries.
    """
    M = len(groups)
    for k in range(M - 1):
        if groups[k] == groups[k + 1]:
            space_col = 2 * k + 2
            g1 = space_col
            g2 = space_col + 1
            p_space = N + 1 + space_col
            p_prev = N + 1 + (space_col - 1)
            p_next = N + 1 + (space_col + 1)

            num_preds[g1] = 0
            num_preds[g2] = 0
            num_preds[p_space] = 0
            _add_predecessor(predecessors, num_preds, p_next, p_prev)
            _remove_predecessor(predecessors, num_preds, p_next, g2)
            _remove_predecessor(predecessors, num_preds, p_next, p_space)


# ---------------------------------------------------------------------------
# Batched decode
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _decode_fa_batch(
    sim: np.ndarray,
    T_all: np.ndarray,
    N_all: np.ndarray,
    max_N: int,
    groups: np.ndarray | None,
    spaced: bool,
) -> np.ndarray:
    """Batched Viterbi decode, dispatching on spaced flag and groups presence."""
    B = sim.shape[0]
    spans_out = np.zeros((B, max_N, 2), dtype=np.int64)

    for b in numba.prange(B):
        Ti = int(T_all[b])
        Ni = int(N_all[b])
        if Ni == 0 or Ti == 0:
            continue

        sim_i = sim[b, :Ti, :Ni]
        emission, preds, n_preds, p_init = _make_fa(sim_i, spaced)
        if groups is not None:
            if spaced:
                _apply_groups_spaced(preds, n_preds, groups[b, :Ni // 2], Ni)
            else:
                _apply_groups_flat(preds, n_preds, groups[b, :Ni], Ni)
        states = viterbi(emission, preds, n_preds, p_init)
        spans_i = _extract_spans(states, Ti, Ni)
        spans_out[b, :Ni] = spans_i

    return spans_out


# ---------------------------------------------------------------------------
# Flat decode (used by the ONNX host pipeline)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _canonicalize_skipped_spans(
    spans: np.ndarray, T: int, gap_allowed: np.ndarray,
) -> None:
    """Place silence at the first permitted gap in each skipped-token run."""
    N = len(spans)
    lo = 0
    while lo < N:
        if spans[lo, 0] != spans[lo, 1]:
            lo += 1
            continue
        hi = lo
        while hi + 1 < N and spans[hi + 1, 0] == spans[hi + 1, 1]:
            hi += 1
        left = spans[lo - 1, 1] if lo > 0 else 0
        right = spans[hi + 1, 0] if hi + 1 < N else T
        gap_index = lo
        while gap_index <= hi + 1 and not gap_allowed[gap_index]:
            gap_index += 1
        if gap_index > hi + 1:
            assert left == right
        for i in range(lo, hi + 1):
            anchor = left if i < gap_index else right
            spans[i, 0] = anchor
            spans[i, 1] = anchor
        lo = hi + 1


@numba.njit(cache=True)
def _decode_flat_single(
    sim: np.ndarray, skip_penalty: float, gap_allowed: np.ndarray,
) -> np.ndarray:
    """Decode all tokens with zero-time exits and equally priced skips."""
    T, N = sim.shape
    penalty = np.float32(skip_penalty)
    gap = np.full(N + 1, -np.inf, dtype=np.float32)
    token = np.full(N, -np.inf, dtype=np.float32)
    gap_back = np.zeros((T + 1, N + 1), dtype=np.int8)
    token_back = np.zeros((T + 1, N), dtype=np.int8)
    gap[0] = 0.0
    for i in range(N):
        gap[i + 1] = gap[i] - penalty
        gap_back[0, i + 1] = 2  # skip

    for t in range(1, T + 1):
        next_token = np.empty(N, dtype=np.float32)
        next_gap = np.full(N + 1, -np.inf, dtype=np.float32)
        for i in range(N):
            best = token[i]
            if gap[i] > best:
                best = gap[i]
                token_back[t, i] = 1  # enter from gap
            next_token[i] = best + sim[t - 1, i]
        for i in range(N + 1):
            if gap_allowed[i]:
                next_gap[i] = gap[i]  # wait consumes one frame
        for i in range(N):
            if next_token[i] > next_gap[i + 1]:
                next_gap[i + 1] = next_token[i]
                gap_back[t, i + 1] = 1  # exit without consuming a frame
            skipped = next_gap[i] - penalty
            if skipped > next_gap[i + 1]:
                next_gap[i + 1] = skipped
                gap_back[t, i + 1] = 2
        gap = next_gap
        token = next_token

    spans = np.full((N, 2), -1, dtype=np.int64)
    t, i = T, N
    in_token = False
    while t > 0 or i > 0 or in_token:
        if in_token:
            if spans[i, 1] < 0:
                spans[i, 1] = t
            spans[i, 0] = t - 1
            entered = token_back[t, i]
            t -= 1
            if entered == 1:
                in_token = False
        else:
            source = gap_back[t, i]
            if source == 0:
                t -= 1
            elif source == 1:
                i -= 1
                in_token = True
            else:
                i -= 1
                spans[i, 0] = t
                spans[i, 1] = t
    _canonicalize_skipped_spans(spans, T, gap_allowed)
    return spans


@numba.njit(parallel=True, cache=True)
def _decode_flat_batch(
    sim: np.ndarray,
    T_all: np.ndarray,
    N_all: np.ndarray,
    max_N: int,
    groups: np.ndarray | None,
    skip_penalty: float,
) -> np.ndarray:
    spans_out = np.zeros((sim.shape[0], max_N, 2), dtype=np.int64)
    for b in numba.prange(sim.shape[0]):
        T, N = int(T_all[b]), int(N_all[b])
        if N == 0:
            continue
        gap_allowed = np.ones(N + 1, dtype=np.bool_)
        if groups is not None:
            for i in range(1, N):
                gap_allowed[i] = groups[b, i - 1] != groups[b, i]
        spans_out[b, :N] = _decode_flat_single(sim[b, :T, :N], skip_penalty, gap_allowed)
    return spans_out


def decode_alignment_flat(
    sim: np.ndarray,
    frame_lengths: np.ndarray,
    token_lengths: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    skip_penalty: float = 0.5,
) -> np.ndarray:
    """Maximize summed raw cosine similarity minus a cost per skipped token.

    Every token emits at least one frame or pays skip_penalty, including
    prefixes and suffixes. Exits and skips consume no frames. Groups restrict
    gap waiting, not zero-time transitions. Skipped spans are right-anchored
    where group constraints allow, with all token positions retained.

    Args:
        sim: [B, T, N] float32 similarity between frame and token features.
        frame_lengths: [B] int, number of valid frames per sample.
        token_lengths: [B] int, number of valid tokens per sample.
        groups: optional [B, N] int, group label per token.
        skip_penalty: nonnegative raw cosine-score cost per skip.

    Returns:
        spans [B, N, 2] int64, (onset, offset) in frames.
    """
    T_all = np.asarray(frame_lengths, dtype=np.int64)
    N_all = np.asarray(token_lengths, dtype=np.int64)
    max_N = int(N_all.max()) if N_all.size else 0
    sim_np = np.ascontiguousarray(sim, dtype=np.float32)
    groups_np = np.ascontiguousarray(groups, dtype=np.int64) if groups is not None else None

    return _decode_flat_batch(sim_np, T_all, N_all, max_N, groups_np, float(skip_penalty))
