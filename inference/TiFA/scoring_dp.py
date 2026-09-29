"""Whole-word candidate DP for TiFA pronunciation scoring.

Vendored from openvpi/TIFA inference/scoring.py (``_advance`` and
``_select_sample``; already pure numpy/math, only renamed for module use).
Feeds on the outputs of score.onnx per ONNX.md.
"""
from __future__ import annotations

import math

import numpy as np


def _advance(state, pieces, choice, descriptors, lengths, costs, tails, capacity):
    """Process every fragment of a complete candidate before returning a state."""
    segment, used = state
    score = 0.0
    for fragment in pieces:
        current = int(descriptors[fragment, 1])
        if current != segment:
            score += float(tails[segment, used])
            segment, used = current, 0
        length = int(lengths[fragment, choice])
        if used + length > capacity[segment]:
            return None
        score += float(costs[fragment, choice, used])
        used += length
    return (segment, used), score


def select_sample(valid, descriptors, lengths, costs, tails, capacity):
    """Whole-word Viterbi and conditional scores with deterministic ties.

    F[i+1,y] = max(F[i,x] + A[i,c,x]) over complete-candidate transitions.
    H[W,(s,n)] = R[s,n]; H[i,x] = max(A[i,c,x] + H[i+1,y]).
    Q[i,c] = max_x(F[i,x] + A[i,c,x] + H[i+1,y]).
    Closing a segment adds its SPACE suffix exactly once.

    Returns (choices [W] int64, scores [W,C] float); choices are 1-based
    candidate IDs with 0 for absent words.
    """
    forward = [{(0, 0): 0.0}]
    ranks = {(0, 0): 0}
    parents, layers = [], []
    for w, row in enumerate(valid):
        pieces = np.flatnonzero(descriptors[:, 0] == w + 1)
        candidates = (np.flatnonzero(row) + 1).tolist() or [0]
        following, parent, order, edges = {}, {}, {}, []
        for source, value in forward[-1].items():
            for c in candidates:
                transition = (
                    _advance(source, pieces, c - 1, descriptors, lengths, costs, tails, capacity)
                    if c > 0 else (source, 0.0)
                )
                if transition is None:
                    continue
                target, cost = transition
                edges.append((source, target, c, cost))
                total = value + cost
                key = (ranks[source], c)
                if (target not in following or total > following[target]
                        or (total == following[target] and key < order[target])):
                    following[target] = total
                    parent[target] = (source, c)
                    order[target] = key
        if not following:
            raise ValueError("No complete pronunciation path fits the scoring template.")
        ranks = {state: rank for rank, state in enumerate(sorted(order, key=order.get))}
        forward.append(following)
        parents.append(parent)
        layers.append(edges)

    terminal = {state: float(tails[state[0], state[1]]) for state in forward[-1]}
    state = min(forward[-1], key=lambda x: (-(forward[-1][x] + terminal[x]), ranks[x]))
    choices = np.zeros(len(valid), dtype=np.int64)
    for i in range(len(valid) - 1, -1, -1):
        state, choices[i] = parents[i][state]

    scores = np.full(valid.shape, -np.inf)
    backward = terminal
    for i in range(len(valid) - 1, -1, -1):
        previous = {state: -math.inf for state in forward[i]}
        for source, target, c, cost in layers[i]:
            suffix = cost + backward[target]
            previous[source] = max(previous[source], suffix)
            if c > 0:
                scores[i, c - 1] = max(scores[i, c - 1], forward[i][source] + suffix)
        backward = previous
    return choices, scores
