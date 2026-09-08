"""Rhythm-repair quantization (mode: "repair").

Assumes the user-provided BPM is exact and constant and that the beat grid is
anchored at tick 0. Every note onset is snapped *exactly* onto the user's
grid lines (``{k * step}`` ticks, the lines a MIDI editor draws), while a
Viterbi pass keeps the performed inter-onset rhythm pattern and a metrical
preference breaks ties toward stronger beats. Legato/tie notes stay glued to
their successor and durations snap to musical note values (grid multiples and
dotted values).

Coordinate system matches quantization.py: PPQ 480, one beat = 480 ticks.
"""

from __future__ import annotations

from typing import Any

from inference.quant.quantization import _TIE_LYRICS, _ticks_from_sec

_BEAT_TICKS = 480
# Rests shorter than this fraction of the grid are absorbed (legato).
_LEGATO_GAP_FACTOR = 0.4
# Notes shorter than this fraction of the grid are grace notes: they keep a
# half-grid duration instead of being stretched to a full grid value.
_GRACE_DUR_FACTOR = 0.375
_MIN_NOTE_TICKS = 30
# Onset candidates are searched within raw +- (step + this slack).
_ONSET_WINDOW_SLACK = 30
# Cost weights (normalized so 1.0 == one grid step of movement).
_ONSET_ERROR_WEIGHT = 0.5
_METRICAL_COST_WEIGHT = 0.3
_INTERVAL_WEIGHT = 0.35
_DUR_MULTIPLIERS = (1, 1.5, 2, 3, 4, 6, 8, 12, 16)


def _metrical_strength(tick: int) -> float:
    """How strong the metrical position of ``tick`` is (beats every 480)."""
    r = tick % _BEAT_TICKS
    if r == 0:
        return 1.0
    if r % 240 == 0:
        return 0.55
    if r % 120 == 0:
        return 0.35
    if r % 60 == 0:
        return 0.2
    return 0.1


def _onset_candidates(raw: int, step: int) -> list[int]:
    """Grid lines within raw +- (step + slack), i.e. every plausible line."""
    window = step + _ONSET_WINDOW_SLACK
    lo = -((-(raw - window)) // step)  # ceil division
    hi = (raw + window) // step  # floor division
    return [k * step for k in range(int(lo), int(hi) + 1)]


def _repair_onsets(raw_onsets: list[int], step: int) -> list[int]:
    """Viterbi over per-note grid lines.

    Local cost pulls each onset to its nearest grid line with a preference
    for stronger metrical positions; the transition cost keeps the repaired
    inter-onset intervals close to the performed ones, so the rhythm pattern
    survives the snapping.
    """
    n = len(raw_onsets)
    cand_lists = [_onset_candidates(raw, step) for raw in raw_onsets]

    dp: list[dict[int, float]] = []
    back: list[dict[int, int | None]] = []
    for i in range(n):
        cur_cost: dict[int, float] = {}
        cur_back: dict[int, int | None] = {}
        raw_prev = raw_onsets[i - 1] if i > 0 else None
        for point in cand_lists[i]:
            local = _ONSET_ERROR_WEIGHT * ((point - raw_onsets[i]) / step) ** 2
            local += _METRICAL_COST_WEIGHT * (1.0 - _metrical_strength(point))
            if i == 0:
                cur_cost[point] = local
                cur_back[point] = None
                continue

            best: float | None = None
            best_prev: int | None = None
            for prev_point, prev_cost in dp[-1].items():
                if point < prev_point + _MIN_NOTE_TICKS:
                    continue
                interval_err = abs(
                    (point - prev_point) - (raw_onsets[i] - raw_prev)
                ) / step
                total = prev_cost + local + _INTERVAL_WEIGHT * interval_err
                if best is None or total < best:
                    best = total
                    best_prev = prev_point
            if best is not None:
                cur_cost[point] = best
                cur_back[point] = best_prev
        if not cur_cost:
            # Dense degenerate input (e.g. several notes on the same tick):
            # every candidate violates the minimum gap. Force the first grid
            # line that keeps the gap from the previous onset so the output
            # stays strictly monotone and on-grid.
            prev_min = min(dp[-1], key=dp[-1].get)
            forced = ((prev_min + _MIN_NOTE_TICKS + step - 1) // step) * step
            cur_cost[forced] = dp[-1][prev_min] + 5.0
            cur_back[forced] = prev_min
        dp.append(cur_cost)
        back.append(cur_back)

    seq = [min(dp[-1], key=dp[-1].get)]
    for i in range(n - 1, 0, -1):
        prev = back[i][seq[-1]]
        seq.append(int(prev) if prev is not None else seq[-1])
    seq.reverse()
    return seq


def _snap_duration(raw_dur: int, step: int) -> int:
    """Snap a raw duration to a musical note value (grid multiples + dotted)."""
    if raw_dur <= max(1, round(step * _GRACE_DUR_FACTOR)):
        # Grace note: half-grid value, never below the minimum note length.
        return max(_MIN_NOTE_TICKS, step // 2) if step >= 2 * _MIN_NOTE_TICKS else step
    values = [m * step for m in _DUR_MULTIPLIERS]
    return min(values, key=lambda d: abs(d - raw_dur))


def repair_rhythm(notes: list[Any], tempo: float, quantization_step: int) -> None:
    """Repair note rhythm in place against the (exact) user tempo.

    Every onset lands exactly on a grid line ``k * step`` of the *user
    selected* grid — the grid is never auto-refined, so what the user picks
    in the GUI is what the notes snap to. Dense passages (e.g. 1/8 pairs
    sung inside a 1/16 grid) necessarily stretch to the grid; that is the
    accepted trade-off for a strict on-grid guarantee. See the module
    docstring for what is repaired. ``quantization_step`` is the grid in
    ticks (480 = quarter ... 30 = 64th); ``step <= 0`` is a no-op.
    """
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)
    step = int(quantization_step)

    raw_onsets = [_ticks_from_sec(n.onset, tempo) for n in notes]
    raw_ends = [_ticks_from_sec(n.offset, tempo) for n in notes]
    raw_durs = [max(1, end - onset) for onset, end in zip(raw_onsets, raw_ends)]
    lyrics = [(n.lyric or "") for n in notes]

    onsets = _repair_onsets(raw_onsets, step)

    # Stage 2: durations. Legato (tie lyrics / tiny raw gap) stretches the
    # note to the next repaired onset; real rests keep a musical note value.
    n = len(notes)
    ends: list[int] = []
    for i in range(n):
        raw_gap_next = None if i + 1 >= n else raw_onsets[i + 1] - raw_ends[i]
        legato = raw_gap_next is not None and raw_gap_next < step * _LEGATO_GAP_FACTOR
        tie = lyrics[i] in _TIE_LYRICS
        if tie and raw_gap_next is not None and raw_gap_next < step:
            legato = True

        if legato:
            ends.append(max(onsets[i] + _MIN_NOTE_TICKS, onsets[i + 1]))
            continue

        end = onsets[i] + _snap_duration(raw_durs[i], step)
        if i + 1 < n:
            end = min(end, onsets[i + 1])
        ends.append(max(end, onsets[i] + _MIN_NOTE_TICKS))

    # Final safety pass: strict ordering and positive durations. An overlapped
    # onset is pushed to the first grid line at/after the previous end so the
    # "every onset on a grid line" guarantee survives the repair.
    fixed: list[tuple[int, int]] = []
    for i in range(n):
        start = onsets[i]
        end = ends[i]
        if i > 0 and start < fixed[-1][1]:
            start = ((fixed[-1][1] + step - 1) // step) * step
        if end <= start:
            end = start + _MIN_NOTE_TICKS
        fixed.append((start, end))

    for note, (start_tick, end_tick) in zip(notes, fixed):
        note.onset = start_tick / (tempo * 8)
        note.offset = end_tick / (tempo * 8)

    moved = sum(1 for s, r in zip(onsets, raw_onsets) if abs(s - r) > 5)
    max_move = max((abs(s - r) for s, r in zip(onsets, raw_onsets)), default=0)
    glued = sum(
        1 for i in range(n - 1)
        if fixed[i][1] == fixed[i + 1][0]
        and raw_onsets[i + 1] - raw_ends[i] >= step * _LEGATO_GAP_FACTOR
    )
    print(
        f"[RhythmRepair] {n} notes @ {step}-tick grid: {moved} onsets snapped "
        f"(max {max_move} ticks), {glued} rests absorbed"
    )
