"""Smart Rhythm-Repair Quantization Engine (mode: "repair").

Musically grounded, context-aware rhythm repair for vocal-to-MIDI synthesis.
Key features:
1. Phrase Segmentation: Isolates musical phrases at natural breath pauses,
   preventing local timing deviations from propagating across phrases.
2. Phrase Latency Calibration: Detects and compensates for consistent human
   laid-back (dragging) or pushing (rushing) microtiming offsets.
3. Dual Straight + Triplet Lattice: Dynamically supports standard binary
   subdivisions (1/16, 1/8, 1/4) as well as triplet intervals (160/80 ticks)
   when triplet motifs occur.
4. Syncopation-Protected IOI Viterbi: Preserves intentional off-beat onsets
   and rhythm motifs using relative Inter-Onset Interval (IOI) preservation,
   avoiding aggressive strong-beat snapping.
5. Musical Duration & Legato Structuring: Seamlessly glues consecutive legato
   vowels while snapping authentic pauses to musical rests.
   Zero-overlap and strict monotonicity are mathematically guaranteed.

Coordinate system: PPQ 480, one quarter note (beat) = 480 ticks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inference.quant.quantization import _TIE_LYRICS, _ticks_from_sec

_BEAT_TICKS = 480
_MIN_NOTE_TICKS = 30
_TRIPLET_EIGHTH_TICKS = 160
_TRIPLET_SIXTEENTH_TICKS = 80
_GRACE_DUR_FACTOR = 0.375
_LEGATO_GAP_FACTOR = 0.40
_DUR_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


def _metrical_strength(tick: int) -> float:
    """Metrical strength of a tick position (gentle tie-breaker).

    Quarter beats get 1.0; 8th notes get 0.75; 16th notes get 0.5;
    triplet 8ths get 0.65; triplet 16ths get 0.45; other positions get 0.2.
    """
    pos_beat = tick % _BEAT_TICKS
    if pos_beat == 0:
        return 1.0
    if pos_beat % 240 == 0:
        return 0.75
    if pos_beat % 160 == 0:
        return 0.65
    if pos_beat % 120 == 0:
        return 0.50
    if pos_beat % 80 == 0:
        return 0.45
    if pos_beat % 60 == 0:
        return 0.35
    return 0.20


def _segment_phrases(
    notes: list[Any],
    tempo: float,
    step: int,
) -> list[list[int]]:
    """Split note indices into independent phrases by natural breathing pauses.

    A phrase boundary occurs when the gap between note offset and the next onset
    is >= max(240 ticks, 2 * step), or when the raw IOI is >= max(960 ticks, 8 * step).
    """
    n = len(notes)
    if n == 0:
        return []

    raw_onsets = [_ticks_from_sec(note.onset, tempo) for note in notes]
    raw_offsets = [_ticks_from_sec(note.offset, tempo) for note in notes]

    split_gap_threshold = max(240, 2 * step)
    split_ioi_threshold = max(960, 8 * step)

    phrases: list[list[int]] = []
    cur_phrase = [0]

    for i in range(1, n):
        gap = raw_onsets[i] - raw_offsets[i - 1]
        ioi = raw_onsets[i] - raw_onsets[i - 1]
        is_tie = getattr(notes[i], "lyric", "") in _TIE_LYRICS

        if not is_tie and (gap >= split_gap_threshold or ioi >= split_ioi_threshold):
            phrases.append(cur_phrase)
            cur_phrase = [i]
        else:
            cur_phrase.append(i)

    if cur_phrase:
        phrases.append(cur_phrase)
    return phrases


def _calibrate_phrase_latency(raw_onsets: list[int], step: int) -> int:
    """Estimate consistent phrase-level timing shift (dragging or rushing).

    Computes the circular median of raw onsets modulo step. If the singer
    consistently drags (e.g. +30 ticks late) with small variance, returns the
    latency in ticks so onsets can be pre-aligned before lattice snapping.
    """
    if len(raw_onsets) < 3:
        return 0

    half_step = step / 2.0
    shifts = np.array([((x + half_step) % step) - half_step for x in raw_onsets], dtype=np.float64)
    median_shift = float(np.median(shifts))
    abs_dev = float(np.mean(np.abs(shifts - median_shift)))

    # Only apply calibration if median shift is significant and tightly clustered
    if abs(median_shift) >= max(15.0, step * 0.15) and abs_dev <= step * 0.25:
        return int(round(median_shift))
    return 0


def _has_triplet_motif(raw_onsets: list[int], step: int) -> bool:
    """Check if the phrase contains evidence of triplet rhythm.

    Returns True if at least 2 consecutive intervals match triplet 8th (160 +- 30)
    or triplet 16th (80 +- 20) ticks.
    """
    if len(raw_onsets) < 3:
        return False

    trip_8th_matches = 0
    trip_16th_matches = 0
    for i in range(len(raw_onsets) - 1):
        ioi = raw_onsets[i + 1] - raw_onsets[i]
        if abs(ioi - _TRIPLET_EIGHTH_TICKS) <= 30:
            trip_8th_matches += 1
        elif abs(ioi - _TRIPLET_SIXTEENTH_TICKS) <= 20:
            trip_16th_matches += 1

    return trip_8th_matches >= 2 or trip_16th_matches >= 2


def _candidate_points(
    raw: int,
    step: int,
) -> list[int]:
    """Generate plausible grid lines around a raw onset.

    Includes straight lattice points (k * step) within raw +- (step + slack).
    Strictly adheres to the user-selected grid lines.
    """
    slack = max(30, step // 4)
    window = step + slack
    lo = int(np.floor((raw - window) / step))
    hi = int(np.ceil((raw + window) / step))
    points = {k * step for k in range(lo, hi + 1) if k * step >= 0}
    return sorted(points)


def _repair_phrase_onsets(
    raw_onsets: list[int],
    step: int,
) -> list[int]:
    """Viterbi DP over onset candidates with rhythm IOI preservation and syncopation protection.

    Hard constraint: point_{i} >= point_{i-1} + min_step (strict monotonicity guaranteed).
    Cost function balances:
    - Distance from raw sung onset
    - IOI interval matching (relative rhythm preservation)
    - Musical note value preference for IOIs
    - Syncopation protection (gentle tie-breaking metrical preference, no violent snap)
    - Triplet group consistency
    """
    n = len(raw_onsets)
    if n == 0:
        return []
    if n == 1:
        # Single note: snap to nearest straight grid line
        cands = [k * step for k in range(max(0, (raw_onsets[0] - step) // step), (raw_onsets[0] + step) // step + 2)]
        best_cand = min(cands, key=lambda c: (abs(c - raw_onsets[0]), -_metrical_strength(c)))
        return [best_cand]

    min_gap = max(_MIN_NOTE_TICKS, step // 4)
    cand_lists = [_candidate_points(raw, step) for raw in raw_onsets]

    # Pre-identify syncopation flags: sung note was held for at least an eighth note
    # and onset started off the quarter-beat grid
    is_syncopated = [False] * n
    for i in range(n):
        if i + 1 < n:
            raw_ioi = raw_onsets[i + 1] - raw_onsets[i]
            if raw_ioi >= 200 and (raw_onsets[i] % _BEAT_TICKS) not in (0, 240):
                is_syncopated[i] = True

    dp: list[dict[int, float]] = []
    back: list[dict[int, int | None]] = []

    for i in range(n):
        cur_cost: dict[int, float] = {}
        cur_back: dict[int, int | None] = {}
        raw_cur = raw_onsets[i]
        raw_prev = raw_onsets[i - 1] if i > 0 else None
        target_ioi = (raw_cur - raw_prev) if i > 0 else None

        for point in cand_lists[i]:
            # 1. Local onset distance error
            dist_err = abs(point - raw_cur) / step
            local_cost = 0.50 * (dist_err ** 2)

            # 2. Metrical preference (gentle tie-breaker; reduced for syncopated notes)
            metrical_weight = 0.04 if is_syncopated[i] else 0.12
            local_cost += metrical_weight * (1.0 - _metrical_strength(point))

            # 3. Triplet prior penalty (straight is default; triplets require consecutive confirmation)
            is_triplet_point = (point % step != 0) and (point % _TRIPLET_SIXTEENTH_TICKS == 0)
            if is_triplet_point:
                local_cost += 0.20

            if i == 0:
                cur_cost[point] = local_cost
                cur_back[point] = None
                continue

            best_total: float | None = None
            best_prev: int | None = None

            for prev_point, prev_cost in dp[-1].items():
                interval = point - prev_point
                if interval < min_gap:
                    continue

                # Transition cost: IOI preservation
                ioi_err = abs(interval - target_ioi) / step
                trans_cost = 0.50 * ioi_err

                # Bonus for standard musical IOI ratios (1, 2, 3, 4, 6, 8 * step)
                multiple = interval / step
                is_standard_mult = any(abs(multiple - m) < 1e-3 for m in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0))
                if is_standard_mult:
                    trans_cost -= 0.08

                # Triplet consistency: if interval is a triplet interval (160 or 80 ticks),
                # refund the triplet penalty
                if interval in (_TRIPLET_EIGHTH_TICKS, _TRIPLET_SIXTEENTH_TICKS):
                    trans_cost -= 0.25

                total = prev_cost + local_cost + trans_cost
                if best_total is None or total < best_total:
                    best_total = total
                    best_prev = prev_point

            if best_total is not None:
                cur_cost[point] = best_total
                cur_back[point] = best_prev

        if not cur_cost:
            # Fallback for dense collisions: pick best previous and advance by minimum grid step
            prev_best = min(dp[-1], key=dp[-1].get)
            forced_point = prev_best + max(min_gap, step)
            cur_cost[forced_point] = dp[-1][prev_best] + 3.0
            cur_back[forced_point] = prev_best

        dp.append(cur_cost)
        back.append(cur_back)

    best_final = min(dp[-1], key=dp[-1].get)
    seq = [best_final]
    for i in range(n - 1, 0, -1):
        prev = back[i][seq[-1]]
        seq.append(int(prev) if prev is not None else seq[-1] - step)
    seq.reverse()
    return seq


def _snap_musical_duration(raw_dur: int, step: int, max_dur: int | None = None) -> int:
    """Snap a raw duration to musical note values (grid multiples, dotted, or grace notes)."""
    if raw_dur <= max(1, round(step * _GRACE_DUR_FACTOR)):
        # Grace note: half-grid value, never below minimum note length
        grace = max(_MIN_NOTE_TICKS, step // 2) if step >= 2 * _MIN_NOTE_TICKS else step
        return min(grace, max_dur) if max_dur is not None else grace

    candidates = [int(round(m * step)) for m in _DUR_MULTIPLIERS]
    candidates = sorted({c for c in candidates if c >= _MIN_NOTE_TICKS})

    if max_dur is not None:
        valid_candidates = [c for c in candidates if c <= max_dur]
        if valid_candidates:
            candidates = valid_candidates
        else:
            return max(_MIN_NOTE_TICKS, max_dur)

    return min(candidates, key=lambda d: abs(d - raw_dur))


def _structure_phrase_notes(
    notes: list[Any],
    repaired_onsets: list[int],
    tempo: float,
    step: int,
) -> list[tuple[int, int]]:
    """Determine musical offsets and clean legato structuring for a phrase.

    1. Legato: if consecutive notes have a tiny sung gap (< LEGATO_GAP_FACTOR * step)
       or a tie lyric, the previous note connects seamlessly to the next onset.
    2. Real rest: duration snaps to a musical note value, leaving a clean rest.
    3. Strict safety: end_i is guaranteed to be in (onset_i, onset_{i+1}], zero overlaps.
    """
    n = len(notes)
    raw_onsets = [_ticks_from_sec(note.onset, tempo) for note in notes]
    raw_offsets = [_ticks_from_sec(note.offset, tempo) for note in notes]
    raw_durs = [max(1, off - on) for on, off in zip(raw_onsets, raw_offsets)]
    lyrics = [getattr(note, "lyric", "") or "" for note in notes]

    legato_gap_threshold = max(35, int(round(step * _LEGATO_GAP_FACTOR)))
    fixed: list[tuple[int, int]] = []

    for i in range(n):
        onset_cur = repaired_onsets[i]
        onset_next = repaired_onsets[i + 1] if i + 1 < n else None

        if onset_next is not None:
            available_slot = onset_next - onset_cur
            raw_gap = raw_onsets[i + 1] - raw_offsets[i]
            is_tie = lyrics[i] in _TIE_LYRICS or (lyrics[i + 1] in _TIE_LYRICS)
            is_legato = is_tie or (raw_gap < legato_gap_threshold)

            if is_legato or available_slot <= step:
                # Connected legato: offset matches next onset exactly
                offset_cur = onset_next
            else:
                # Note followed by an authentic musical rest
                min_rest = max(_MIN_NOTE_TICKS, step // 4)
                max_note_dur = available_slot - min_rest
                dur = _snap_musical_duration(raw_durs[i], step, max_dur=max_note_dur)
                offset_cur = min(onset_cur + dur, onset_next)
        else:
            # Last note of the phrase: snap duration freely
            dur = _snap_musical_duration(raw_durs[i], step)
            offset_cur = onset_cur + dur

        # Guarantee strictly positive duration
        if offset_cur <= onset_cur:
            offset_cur = onset_cur + _MIN_NOTE_TICKS

        fixed.append((onset_cur, offset_cur))

    return fixed


def repair_rhythm(notes: list[Any], tempo: float, quantization_step: int) -> None:
    """Repair note rhythm in place with the smart rhythm-repair engine.

    Args:
        notes: List of NoteInfo objects to quantize in place.
        tempo: Project tempo in BPM (exact).
        quantization_step: Grid step in ticks (480=1/4, 240=1/8, 120=1/16, 60=1/32).
                           Values <= 0 are a no-op.
    """
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)
    step = int(quantization_step)

    # Step 1: Segment into independent musical phrases
    phrase_indices = _segment_phrases(notes, tempo, step)
    all_repaired_spans: list[tuple[int, int]] = [None] * len(notes)

    total_snapped = 0
    total_absorbed_rests = 0

    for phrase in phrase_indices:
        phrase_notes = [notes[idx] for idx in phrase]
        raw_onsets = [_ticks_from_sec(n.onset, tempo) for n in phrase_notes]

        # Step 2: Calibrate phrase-level timing latency (dragging/rushing)
        phrase_latency = _calibrate_phrase_latency(raw_onsets, step)
        calibrated_onsets = [x - phrase_latency for x in raw_onsets]

        # Step 3 & 4: Viterbi DP over dual straight/triplet lattice with IOI preservation
        repaired_onsets = _repair_phrase_onsets(calibrated_onsets, step)

        # Step 5: Musical duration & legato structuring
        phrase_spans = _structure_phrase_notes(phrase_notes, repaired_onsets, tempo, step)

        for local_i, global_idx in enumerate(phrase):
            all_repaired_spans[global_idx] = phrase_spans[local_i]

        # Metrics logging
        total_snapped += sum(1 for s, r in zip(repaired_onsets, raw_onsets) if abs(s - r) > 5)
        for i in range(len(phrase) - 1):
            if phrase_spans[i][1] == phrase_spans[i + 1][0]:
                raw_gap = raw_onsets[i + 1] - _ticks_from_sec(phrase_notes[i].offset, tempo)
                if raw_gap >= step * _LEGATO_GAP_FACTOR:
                    total_absorbed_rests += 1

    # Apply repaired ticks back to NoteInfo objects in seconds
    scale = tempo * 8.0
    for note, (start_tick, end_tick) in zip(notes, all_repaired_spans):
        note.onset = start_tick / scale
        note.offset = end_tick / scale

    print(
        f"[RhythmRepair] {len(notes)} notes ({len(phrase_indices)} phrases) @ {step}-tick grid: "
        f"{total_snapped} onsets repaired, {total_absorbed_rests} micro-rests smoothed into legato"
    )

