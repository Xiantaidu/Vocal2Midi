from __future__ import annotations

from typing import Any

import numpy as np


def _ticks_from_sec(t: float, tempo: float) -> int:
    return int(round(t * tempo * 8))


_DPQ_DEFAULT_ASYM = {
    "start_early": 2.0,
    "start_late": 0.5,
    "end_early": 1.5,
    "end_late": 0.75,
    "gap": 0.75,
    "dur": 1.0,
    "grid120": 1.0,
    "grid480": 1.0,
}
_BAYES_ASYM = {
    **_DPQ_DEFAULT_ASYM,
    "start_early": 1.4,
    "start_late": 0.65,
    "end_early": 1.3,
    "end_late": 0.9,
}
_DPQ_WEIGHTS = [0.08971, 0.025064, 0.235416, 0.038973, 0.580321, 0.030516]
_DPQ_INTERNAL_GRID = 30
_DPQ_SEGMENT_SHIFT_CANDIDATES = (0, 60, 120)
_DPQ_SEGMENT_CENTER_WEIGHT = 0.02
_DPQ_SEGMENT_SWITCH_PENALTY = 1.0
_DPQ_SEGMENT_TIE_DELAY_BONUS = 0.4
_DPQ_SEGMENT_ZERO_GAP_BONUS = 0.3
_DPQ_SEGMENT_LONG_DELAY_BONUS = 0.2
_DPQ_SEGMENT_LONG_THRESHOLD = 240
_DPQ_SEGMENT_MAX_NOTES = 16
_BEAT_TICKS = 480
_BAYES_CANDIDATE_RADIUS = 3
_BAYES_MOTIF_MIN_COUNT = 6
_BAYES_MOTIF_PHASE_WEIGHT = 0.0
_BAYES_MOTIF_DURATION_WEIGHT = 0.05
_BAYES_MOTIF_GAP_WEIGHT = 0.0
_BAYES_SEGMENT_CENTER_WEIGHT = 0.03
_BAYES_SEGMENT_MAX_NOTES = 48
_BAYES_LATE_PULLBACK_MIN_PHASE_FACTOR = 0.5
_BAYES_LATE_PULLBACK_MAX_SPREAD_FACTOR = 0.5
_BAYES_LATE_PULLBACK_MIN_NOTES = 4
_BAYES_LATE_PULLBACK_WEIGHT = 0.08
_BAYES_MAX_START_SHIFT_FACTOR = 1.0
_BAYES_MAX_START_SHIFT_FLOOR = 60
_BAYES_MAX_START_SHIFT_CAP = 180
_BAYES_MAX_END_SHIFT_FACTOR = 1.0
_BAYES_MAX_END_SHIFT_FLOOR = 90
_BAYES_MAX_END_SHIFT_CAP = 240
_BAYES_HALF_GRID_MAX_DUR_FACTOR = 0.375
_BAYES_DURATION_PRIOR_WEIGHT = 0.015


def _resolve_dp_grid_step(quantization_step: int) -> int:
    return int(quantization_step) if quantization_step and quantization_step > 0 else _DPQ_INTERNAL_GRID


def _resolve_segment_shift_candidates(grid_step: int) -> tuple[int, ...]:
    if grid_step <= _DPQ_INTERNAL_GRID:
        return _DPQ_SEGMENT_SHIFT_CANDIDATES

    vals = {0, grid_step // 4, grid_step // 2}
    return tuple(sorted(v for v in vals if v >= 0))


def _nearest_candidate(value: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda cand: abs(cand - value))


def _mod_distance(a: int, b: int, modulo: int) -> int:
    diff = abs(a - b) % modulo
    return min(diff, modulo - diff)


def _dist_grid(x: int, step: int) -> int:
    r = x % step
    return min(r, step - r if r else 0)


def _candidate_values(raw_tick: int, radius: int = 3, step: int = 30) -> list[int]:
    center = round(raw_tick / step)
    vals = sorted({(center + k) * step for k in range(-radius, radius + 1)})
    return vals


def _build_note_pair(raw_note: Any, tick_onset: int, tick_offset: int) -> dict[str, Any]:
    return {
        "raw_start": tick_onset,
        "raw_end": tick_offset,
        "raw_dur": max(1, tick_offset - tick_onset),
        "lyrics": getattr(raw_note, "lyric", "") or "",
    }


def _annotate_pairs_with_gap(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    prev_end = None
    for pair in pairs:
        cur = dict(pair)
        cur["raw_gap"] = 0 if prev_end is None else max(0, cur["raw_start"] - prev_end)
        annotated.append(cur)
        prev_end = cur["raw_end"]
    return annotated


def _build_candidate_pairs(pair: dict[str, Any], radius: int = 3, step: int = 30) -> list[tuple[int, int]]:
    starts = _candidate_values(pair["raw_start"], radius, step)
    ends = _candidate_values(pair["raw_end"], radius, step)
    cands = []
    for s in starts:
        for e in ends:
            if e > s:
                cands.append((s, e))
    return sorted(set(cands))


def _resolve_bayes_shift_limit(step: int, *, factor: float, floor: int, cap: int) -> int:
    return int(min(cap, max(floor, round(step * factor))))


def _filter_bayes_candidate_pairs(
    pair: dict[str, Any],
    candidates: list[tuple[int, int]],
    step: int,
) -> list[tuple[int, int]]:
    start_limit = _resolve_bayes_shift_limit(
        step,
        factor=_BAYES_MAX_START_SHIFT_FACTOR,
        floor=_BAYES_MAX_START_SHIFT_FLOOR,
        cap=_BAYES_MAX_START_SHIFT_CAP,
    )
    end_limit = _resolve_bayes_shift_limit(
        step,
        factor=_BAYES_MAX_END_SHIFT_FACTOR,
        floor=_BAYES_MAX_END_SHIFT_FLOOR,
        cap=_BAYES_MAX_END_SHIFT_CAP,
    )
    filtered = [
        cand
        for cand in candidates
        if abs(cand[0] - pair["raw_start"]) <= start_limit
        and abs(cand[1] - pair["raw_end"]) <= end_limit
    ]
    if filtered:
        return filtered

    if not candidates:
        return []

    best = min(
        candidates,
        key=lambda cand: (
            abs(cand[0] - pair["raw_start"]) + abs(cand[1] - pair["raw_end"]),
            abs((cand[1] - cand[0]) - pair["raw_dur"]),
        ),
    )
    return [best]


def _build_bayes_candidate_pairs(
    pair: dict[str, Any],
    step: int,
) -> list[tuple[int, int]]:
    fine_step = max(30, step // 2) if step > 30 else step
    max_half_grid_dur = max(1, int(round(step * _BAYES_HALF_GRID_MAX_DUR_FACTOR)))
    candidate_step = fine_step if fine_step < step and pair["raw_dur"] <= max_half_grid_dur else step
    candidates = _build_candidate_pairs(pair, radius=_BAYES_CANDIDATE_RADIUS, step=candidate_step)
    return _filter_bayes_candidate_pairs(pair, candidates, step)


def _estimate_segment_phase_center(
    pairs: list[dict[str, Any]],
    step: int,
) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0

    raw_shifts = np.array(
        [round(pair["raw_start"] / step) * step - pair["raw_start"] for pair in pairs],
        dtype=np.float64,
    )
    center = float(np.median(raw_shifts))
    spread = float(np.mean(np.abs(raw_shifts - center)))
    spread_scale = max(step * 0.4, 1.0)
    strength = max(0.0, 1.0 - min(spread / spread_scale, 1.0))

    phases = np.array([pair["raw_start"] % step for pair in pairs], dtype=np.float64)
    phase_center = float(np.median(phases))
    phase_spread = float(np.mean(np.abs(phases - phase_center)))
    if (
        len(pairs) >= _BAYES_LATE_PULLBACK_MIN_NOTES
        and phase_center >= step * _BAYES_LATE_PULLBACK_MIN_PHASE_FACTOR
        and phase_spread <= step * _BAYES_LATE_PULLBACK_MAX_SPREAD_FACTOR
    ):
        pullback_strength = max(0.0, 1.0 - min(phase_spread / max(step * 0.35, 1.0), 1.0))
        return -phase_center, _BAYES_LATE_PULLBACK_WEIGHT * max(strength, pullback_strength)

    return center, _BAYES_SEGMENT_CENTER_WEIGHT * strength


def _local_cost_asym(
    cand_start: int,
    cand_end: int,
    pair: dict[str, Any],
    prev_end: int | None = None,
    prev_raw_end: int | None = None,
    asym: dict[str, float] | None = None,
):
    asym = asym or _DPQ_DEFAULT_ASYM
    early_start = max(pair["raw_start"] - cand_start, 0)
    late_start = max(cand_start - pair["raw_start"], 0)
    early_end = max(pair["raw_end"] - cand_end, 0)
    late_end = max(cand_end - pair["raw_end"], 0)
    raw_gap = 0 if prev_raw_end is None else pair["raw_start"] - prev_raw_end
    pred_gap = 0 if prev_end is None else cand_start - prev_end

    f1 = _DPQ_WEIGHTS[0] * (asym["start_early"] * early_start + asym["start_late"] * late_start)
    f2 = _DPQ_WEIGHTS[1] * (asym["end_early"] * early_end + asym["end_late"] * late_end)
    f3 = _DPQ_WEIGHTS[2] * asym["gap"] * abs(pred_gap - raw_gap)
    f4 = _DPQ_WEIGHTS[3] * asym["dur"] * abs((cand_end - cand_start) - pair["raw_dur"])
    f5 = _DPQ_WEIGHTS[4] * asym["grid120"] * _dist_grid(cand_start, max(1, 120))
    f6 = _DPQ_WEIGHTS[5] * asym["grid480"] * _dist_grid(cand_start, max(1, 480))
    return f1 + f2 + f3 + f4 + f5 + f6


def _segment_split_indices(pairs: list[dict[str, Any]], step: int) -> list[tuple[int, int]]:
    if not pairs:
        return []

    segments = []
    start = 0
    split_gap = max(step, _DPQ_INTERNAL_GRID)
    hard_max = _DPQ_SEGMENT_MAX_NOTES

    for i in range(1, len(pairs)):
        prev = pairs[i - 1]
        cur = pairs[i]
        raw_gap = cur["raw_start"] - prev["raw_end"]
        prev_tie = prev.get("lyrics") == "-"
        cur_tie = cur.get("lyrics") == "-"

        should_split = False
        if raw_gap >= split_gap:
            should_split = True
        elif raw_gap > 0 and not prev_tie and not cur_tie:
            should_split = True
        elif (i - start) >= hard_max and raw_gap >= 0:
            should_split = True

        if should_split:
            segments.append((start, i))
            start = i

    segments.append((start, len(pairs)))
    return segments


def _segment_split_indices_bayesian(pairs: list[dict[str, Any]], step: int) -> list[tuple[int, int]]:
    if not pairs:
        return []

    segments = []
    start = 0
    split_gap = max(step * 3, 240)
    hard_max = _BAYES_SEGMENT_MAX_NOTES

    for i in range(1, len(pairs)):
        prev = pairs[i - 1]
        cur = pairs[i]
        raw_gap = cur["raw_start"] - prev["raw_end"]
        should_split = raw_gap >= split_gap or (i - start) >= hard_max
        if should_split:
            segments.append((start, i))
            start = i

    segments.append((start, len(pairs)))
    return segments


def _center_adjustment(
    pair: dict[str, Any],
    cand_start: int,
    prev_raw_end: int | None,
    center: int,
) -> float:
    shift_now = cand_start - pair["raw_start"]
    penalty = _DPQ_SEGMENT_CENTER_WEIGHT * abs(shift_now - center)
    if center <= 0:
        return penalty

    bonus = 0.0
    if pair.get("lyrics") == "-":
        bonus += _DPQ_SEGMENT_TIE_DELAY_BONUS
    raw_gap = 0 if prev_raw_end is None else pair["raw_start"] - prev_raw_end
    if raw_gap == 0:
        bonus += _DPQ_SEGMENT_ZERO_GAP_BONUS
    if pair["raw_dur"] >= _DPQ_SEGMENT_LONG_THRESHOLD:
        bonus += _DPQ_SEGMENT_LONG_DELAY_BONUS
    if center >= 120:
        bonus *= 1.15
    return penalty - bonus


def _decode_segment_with_center(
    pairs: list[dict[str, Any]],
    center: int,
    grid_step: int,
    asym: dict[str, float] | None = None,
) -> tuple[list[tuple[int, int]], float]:
    cand_lists = [_build_candidate_pairs(p, radius=3, step=grid_step) for p in pairs]
    dp = []
    back = []

    for i, p in enumerate(pairs):
        cur_cost = {}
        cur_back = {}
        prev_raw_end = None if i == 0 else pairs[i - 1]["raw_end"]
        for c in cand_lists[i]:
            base = _local_cost_asym(c[0], c[1], p, None, None, asym) if i == 0 else None
            center_cost = _center_adjustment(p, c[0], prev_raw_end, center)
            if i == 0:
                cur_cost[c] = base + center_cost
                cur_back[c] = None
                continue

            best = None
            best_prev = None
            for pc, pcost in dp[-1].items():
                total = pcost + _local_cost_asym(c[0], c[1], p, pc[1], prev_raw_end, asym) + center_cost
                if best is None or total < best:
                    best = total
                    best_prev = pc
            cur_cost[c] = best
            cur_back[c] = best_prev
        dp.append(cur_cost)
        back.append(cur_back)

    last = min(dp[-1], key=dp[-1].get)
    total_cost = float(dp[-1][last])
    seq = [last]
    for i in range(len(pairs) - 1, 0, -1):
        seq.append(back[i][seq[-1]])
    seq.reverse()
    return seq, total_cost


def _quantize_notes_phrase_hybrid(notes: list[Any], tempo: float, quantization_step: int):
    if not notes:
        return

    notes.sort(key=lambda n: n.onset)
    grid_step = _resolve_dp_grid_step(quantization_step)
    center_candidates = _resolve_segment_shift_candidates(grid_step)

    orig_onsets = [_ticks_from_sec(n.onset, tempo) for n in notes]
    orig_offsets = [_ticks_from_sec(n.offset, tempo) for n in notes]
    pairs = [
        _build_note_pair(note, onset, max(onset + 1, offset))
        for note, onset, offset in zip(notes, orig_onsets, orig_offsets)
    ]
    segments = _segment_split_indices(pairs, grid_step)

    segment_candidates = []
    for start, end in segments:
        seg_pairs = pairs[start:end]
        center_options = []
        for center in center_candidates:
            seq, cost = _decode_segment_with_center(
                seg_pairs,
                center=center,
                grid_step=grid_step,
                asym=_DPQ_DEFAULT_ASYM,
            )
            center_options.append({"center": center, "seq": seq, "cost": cost})
        segment_candidates.append(center_options)

    seg_dp = []
    seg_back = []
    for i, options in enumerate(segment_candidates):
        cur_cost = {}
        cur_back = {}
        for opt_idx, opt in enumerate(options):
            if i == 0:
                cur_cost[opt_idx] = opt["cost"]
                cur_back[opt_idx] = None
                continue
            best = None
            best_prev = None
            for prev_idx, prev_cost in seg_dp[-1].items():
                prev_center = segment_candidates[i - 1][prev_idx]["center"]
                trans = 0.0 if prev_center == opt["center"] else _DPQ_SEGMENT_SWITCH_PENALTY
                
                # Second order shifting consistency penalty
                if i >= 2:
                    prevprev_idx = seg_back[-1][prev_idx]
                    if prevprev_idx is not None:
                        prevprev_center = segment_candidates[i - 2][prevprev_idx]["center"]
                        # We penalize changes in shift (consistency)
                        trans += 0.08 * abs(opt["center"] - prev_center)

                total = prev_cost + opt["cost"] + trans
                if best is None or total < best:
                    best = total
                    best_prev = prev_idx
            cur_cost[opt_idx] = best
            cur_back[opt_idx] = best_prev
        seg_dp.append(cur_cost)
        seg_back.append(cur_back)

    last_idx = min(seg_dp[-1], key=seg_dp[-1].get)
    chosen = [last_idx]
    for i in range(len(segment_candidates) - 1, 0, -1):
        chosen.append(seg_back[i][chosen[-1]])
    chosen.reverse()

    final_ticks = []
    for seg_idx, opt_idx in enumerate(chosen):
        final_ticks.extend(segment_candidates[seg_idx][opt_idx]["seq"])

    fixed = []
    for i, (start_tick, end_tick) in enumerate(final_ticks):
        if i > 0 and start_tick < fixed[-1][1]:
            start_tick = fixed[-1][1]
        if end_tick <= start_tick:
            end_tick = start_tick + grid_step
        fixed.append((start_tick, end_tick))

    for note, (start_tick, end_tick) in zip(notes, fixed):
        note.onset = start_tick / (tempo * 8)
        note.offset = end_tick / (tempo * 8)


def _build_duration_candidates(step: int, max_raw_tick: int) -> list[int]:
    multipliers = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    cap = max(step, int(np.ceil(max_raw_tick / step)) * step + 2 * step)
    vals = [m * step for m in multipliers if m * step <= cap]
    if not vals:
        vals = [step]
    return sorted(set(vals))


def _build_gap_candidates(step: int, max_raw_gap: int) -> list[int]:
    multipliers = [0, 1, 2, 3, 4, 6, 8]
    cap = max(step * 2, int(np.ceil(max_raw_gap / step)) * step + step)
    vals = [m * step for m in multipliers if m * step <= cap]
    if 0 not in vals:
        vals.insert(0, 0)
    return sorted(set(vals))


def _metrical_position_penalty(tick: int, step: int) -> float:
    pos = tick % _BEAT_TICKS
    if pos == 0:
        return 0.0
    if pos % 240 == 0:
        return 0.04 * step
    if pos % 120 == 0:
        return 0.1 * step
    if step <= 60 and pos % 60 == 0:
        return 0.2 * step
    if step <= 30 and pos % 30 == 0:
        return 0.3 * step
    return 0.42 * step


def _note_value_penalty(duration_tick: int, step: int) -> float:
    multiple = max(1, int(round(duration_tick / max(step, 1))))
    if multiple in {1, 2, 4, 8, 16, 32}:
        return 0.0
    if multiple in {3, 6, 12, 24}:
        return 0.05 * step
    return 0.16 * step


def _preferred_sv_duration(raw_duration_tick: int, step: int) -> int | None:
    if raw_duration_tick <= max(1, round(step * _BAYES_HALF_GRID_MAX_DUR_FACTOR)):
        return None

    ratio = raw_duration_tick / max(step, 1)
    thresholds = (
        (1.5, 1),
        (2.625, 2),
        (3.375, 3),
        (4.625, 4),
        (5.625, 5),
        (7.0, 6),
    )
    for upper_ratio, multiple in thresholds:
        if ratio <= upper_ratio:
            return int(multiple * step)
    return max(step, int(round(ratio)) * step)


def _build_piece_specific_priors(
    pairs: list[dict[str, Any]],
    step: int,
    duration_candidates: list[int],
    gap_candidates: list[int],
) -> list[dict[str, float | int]]:
    grouped: dict[tuple[int, int, bool], list[dict[str, Any]]] = {}
    for pair in pairs:
        key = (
            max(1, int(round(pair["raw_dur"] / max(step, 1)))),
            int(round(pair.get("raw_gap", 0) / max(step, 1))),
            pair.get("lyrics") == "-",
        )
        grouped.setdefault(key, []).append(pair)

    prior_by_key: dict[tuple[int, int, bool], dict[str, float | int]] = {}
    for key, items in grouped.items():
        count = len(items)
        if count < _BAYES_MOTIF_MIN_COUNT:
            continue
        dur_values = [item["raw_dur"] for item in items]
        gap_values = [item.get("raw_gap", 0) for item in items]
        phase_values = [item["raw_start"] % _BEAT_TICKS for item in items]
        phase_candidates = list(range(0, _BEAT_TICKS, step))
        prior_by_key[key] = {
            "count": count,
            "strength": min(0.65, 0.18 + 0.12 * (count - _BAYES_MOTIF_MIN_COUNT)),
            "preferred_dur": _nearest_candidate(float(np.median(dur_values)), duration_candidates),
            "preferred_gap": _nearest_candidate(float(np.median(gap_values)), gap_candidates),
            "preferred_phase": _nearest_candidate(float(np.median(phase_values)), phase_candidates),
        }

    priors = []
    for pair in pairs:
        key = (
            max(1, int(round(pair["raw_dur"] / max(step, 1)))),
            int(round(pair.get("raw_gap", 0) / max(step, 1))),
            pair.get("lyrics") == "-",
        )
        priors.append(prior_by_key.get(key, {"count": 1, "strength": 0.0}))
    return priors


def _bayes_local_cost(
    cand_start: int,
    cand_end: int,
    pair: dict[str, Any],
    prior: dict[str, float | int],
    step: int,
    prev_end: int | None = None,
    prev_raw_end: int | None = None,
    segment_center: float = 0.0,
    segment_center_weight: float = 0.0,
) -> float:
    base = _local_cost_asym(cand_start, cand_end, pair, prev_end, prev_raw_end, _BAYES_ASYM)
    duration_tick = cand_end - cand_start
    cost = base
    cost += _metrical_position_penalty(cand_start, step)
    cost += 0.3 * _note_value_penalty(duration_tick, step)
    preferred_sv_dur = _preferred_sv_duration(pair["raw_dur"], step)
    if preferred_sv_dur is not None:
        cost += _BAYES_DURATION_PRIOR_WEIGHT * abs(duration_tick - preferred_sv_dur)
    if segment_center_weight > 0.0:
        cost += segment_center_weight * abs((cand_start - pair["raw_start"]) - segment_center)

    strength = float(prior.get("strength", 0.0))
    if strength > 0.0:
        preferred_dur = int(prior["preferred_dur"])
        preferred_gap = int(prior["preferred_gap"])
        preferred_phase = int(prior["preferred_phase"])
        cand_gap = 0 if prev_end is None else max(0, cand_start - prev_end)
        cost += strength * _BAYES_MOTIF_DURATION_WEIGHT * abs(duration_tick - preferred_dur)
        cost += strength * _BAYES_MOTIF_GAP_WEIGHT * abs(cand_gap - preferred_gap)
        cost += strength * _BAYES_MOTIF_PHASE_WEIGHT * _mod_distance(
            cand_start % _BEAT_TICKS,
            preferred_phase,
            _BEAT_TICKS,
        )
    return cost


def _decode_segment_bayesian(
    pairs: list[dict[str, Any]],
    priors: list[dict[str, float | int]],
    grid_step: int,
) -> tuple[list[tuple[int, int]], float]:
    cand_lists = [_build_bayes_candidate_pairs(p, grid_step) for p in pairs]
    segment_center, segment_center_weight = _estimate_segment_phase_center(pairs, grid_step)
    dp = []
    back = []

    for i, pair in enumerate(pairs):
        cur_cost = {}
        cur_back = {}
        prev_raw_end = None if i == 0 else pairs[i - 1]["raw_end"]
        for cand in cand_lists[i]:
            if i == 0:
                cur_cost[cand] = _bayes_local_cost(
                    cand[0],
                    cand[1],
                    pair,
                    priors[i],
                    grid_step,
                    segment_center=segment_center,
                    segment_center_weight=segment_center_weight,
                )
                cur_back[cand] = None
                continue

            best = None
            best_prev = None
            prev_candidates = [
                (prev_cand, prev_cost)
                for prev_cand, prev_cost in dp[-1].items()
                if cand[0] >= prev_cand[1]
            ]
            if not prev_candidates:
                prev_candidates = list(dp[-1].items())
            for prev_cand, prev_cost in prev_candidates:
                total = prev_cost + _bayes_local_cost(
                    cand[0],
                    cand[1],
                    pair,
                    priors[i],
                    grid_step,
                    prev_end=prev_cand[1],
                    prev_raw_end=prev_raw_end,
                    segment_center=segment_center,
                    segment_center_weight=segment_center_weight,
                )
                if best is None or total < best:
                    best = total
                    best_prev = prev_cand
            cur_cost[cand] = best
            cur_back[cand] = best_prev
        dp.append(cur_cost)
        back.append(cur_back)

    last = min(dp[-1], key=dp[-1].get)
    total_cost = float(dp[-1][last])
    seq = [last]
    for i in range(len(pairs) - 1, 0, -1):
        seq.append(back[i][seq[-1]])
    seq.reverse()
    return seq, total_cost


def _quantize_notes_bayesian(notes: list[Any], tempo: float, quantization_step: int):
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)
    grid_step = quantization_step
    orig_onsets = [_ticks_from_sec(n.onset, tempo) for n in notes]
    orig_offsets = [_ticks_from_sec(n.offset, tempo) for n in notes]
    pairs = [
        _build_note_pair(note, onset, max(onset + 1, offset))
        for note, onset, offset in zip(notes, orig_onsets, orig_offsets)
    ]
    pairs = _annotate_pairs_with_gap(pairs)

    max_raw = max(pair["raw_dur"] for pair in pairs)
    max_gap = max(pair.get("raw_gap", 0) for pair in pairs)
    duration_candidates = _build_duration_candidates(grid_step, max_raw)
    gap_candidates = _build_gap_candidates(grid_step, max_gap)
    priors = _build_piece_specific_priors(pairs, grid_step, duration_candidates, gap_candidates)
    segments = _segment_split_indices_bayesian(pairs, grid_step)

    final_ticks = []
    for start, end in segments:
        seq, _ = _decode_segment_bayesian(pairs[start:end], priors[start:end], grid_step)
        final_ticks.extend(seq)

    fixed = []
    for i, (start_tick, end_tick) in enumerate(final_ticks):
        if i > 0 and start_tick < fixed[-1][1]:
            start_tick = fixed[-1][1]
        if end_tick <= start_tick:
            end_tick = start_tick + grid_step
        fixed.append((start_tick, end_tick))

    for note, (start_tick, end_tick) in zip(notes, fixed):
        note.onset = start_tick / (tempo * 8)
        note.offset = end_tick / (tempo * 8)


def _quantize_notes_simple(notes: list[Any], tempo: float, quantization_step: int):
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)

    orig_onsets = [n.onset for n in notes]
    orig_offsets = [n.offset for n in notes]

    q_onsets = []
    for onset in orig_onsets:
        ticks = _ticks_from_sec(onset, tempo)
        q_ticks = round(ticks / quantization_step) * quantization_step
        q_onsets.append(q_ticks)

    for i in range(1, len(q_onsets)):
        if q_onsets[i] <= q_onsets[i - 1]:
            q_onsets[i] = q_onsets[i - 1] + quantization_step

    q_offsets = []
    for i in range(len(notes)):
        ticks = _ticks_from_sec(orig_offsets[i], tempo)
        q_ticks = round(ticks / quantization_step) * quantization_step

        if i < len(notes) - 1 and abs(orig_offsets[i] - orig_onsets[i + 1]) < 1e-3:
            q_ticks = q_onsets[i + 1]

        if q_ticks <= q_onsets[i]:
            q_ticks = q_onsets[i] + quantization_step

        if i < len(notes) - 1 and q_ticks > q_onsets[i + 1]:
            q_ticks = q_onsets[i + 1]

        q_offsets.append(q_ticks)

    for i in range(len(notes)):
        notes[i].onset = q_onsets[i] / (tempo * 8)
        notes[i].offset = q_offsets[i] / (tempo * 8)


def _quantize_notes_smart(notes: list[Any], tempo: float, quantization_step: int):
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)
    n = len(notes)

    orig_onsets = [_ticks_from_sec(n_.onset, tempo) for n_ in notes]
    orig_offsets = [_ticks_from_sec(n_.offset, tempo) for n_ in notes]
    raw_durs = [max(1, off - on) for on, off in zip(orig_onsets, orig_offsets)]
    max_raw = max(raw_durs) if raw_durs else quantization_step

    candidates = _build_duration_candidates(quantization_step, max_raw)
    m = len(candidates)

    dp = np.full((n, m), np.inf, dtype=np.float64)
    prev = np.full((n, m), -1, dtype=np.int32)

    preferred = {1, 2, 4, 8, 16}
    dur_pref_penalty = np.array([
        0.0 if (c // quantization_step) in preferred else 0.08 * quantization_step
        for c in candidates
    ], dtype=np.float64)

    for k, c in enumerate(candidates):
        dp[0, k] = abs(raw_durs[0] - c) + dur_pref_penalty[k]

    for i in range(1, n):
        for k, c in enumerate(candidates):
            local_cost = abs(raw_durs[i] - c) + dur_pref_penalty[k]
            jump_cost = np.abs(np.array(candidates, dtype=np.float64) - c) * 0.08
            trans = dp[i - 1] + jump_cost + local_cost
            best_prev = int(np.argmin(trans))
            dp[i, k] = trans[best_prev]
            prev[i, k] = best_prev

    best_last = int(np.argmin(dp[-1]))
    q_durs = [0] * n
    q_durs[-1] = candidates[best_last]
    idx = best_last
    for i in range(n - 1, 0, -1):
        idx = prev[i, idx]
        if idx < 0:
            idx = 0
        q_durs[i - 1] = candidates[idx]

    q_onsets = [0] * n
    q_offsets = [0] * n
    q_onsets[0] = round(orig_onsets[0] / quantization_step) * quantization_step

    for i in range(n):
        q_offsets[i] = q_onsets[i] + max(quantization_step, int(q_durs[i]))
        if i < n - 1:
            raw_rest = max(0, orig_onsets[i + 1] - orig_offsets[i])
            q_rest = 0 if raw_rest < quantization_step * 0.5 else round(raw_rest / quantization_step) * quantization_step
            q_onsets[i + 1] = q_offsets[i] + q_rest

    for i in range(n):
        notes[i].onset = q_onsets[i] / (tempo * 8)
        notes[i].offset = q_offsets[i] / (tempo * 8)


def _quantize_notes_dp_asym(notes: list[Any], tempo: float, quantization_step: int):
    _quantize_notes_phrase_hybrid(notes, tempo, quantization_step)


def quantize_notes(notes: list[Any], tempo: float, quantization_step: int, mode: str = "simple"):
    mode = (mode or "simple").lower()
    if mode == "smart":
        _quantize_notes_smart(notes, tempo, quantization_step)
    elif mode == "bayes":
        _quantize_notes_bayesian(notes, tempo, quantization_step)
    elif mode == "dp":
        # Keep SVP-style phrase DP, but honor the requested grid when one is provided.
        _quantize_notes_dp_asym(notes, tempo, quantization_step)
    else:
        _quantize_notes_simple(notes, tempo, quantization_step)


def should_apply_quantization(mode: str, quantization_step: int) -> bool:
    mode = (mode or "simple").lower()
    if mode == "dp":
        return True
    return quantization_step > 0
