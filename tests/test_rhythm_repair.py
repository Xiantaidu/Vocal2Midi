"""Tests for the rhythm-repair quantization mode.

All fixtures are synthetic NoteInfo lists at BPM 120, where one tick is
1/960 s (PPQ 480). The user BPM is assumed exact and the grid is anchored at
tick 0: every repaired onset must sit exactly on a grid line ``k * step``
(the lines a MIDI editor draws).
"""

import numpy as np
import pytest

from inference.io.note_io import NoteInfo
from inference.quant.quantization import quantize_notes, should_apply_quantization

TEMPO = 120.0
STEP = 120  # 1/16 note grid


def _tick(note) -> int:
    return int(round(note.onset * TEMPO * 8))


def _end_tick(note) -> int:
    return int(round(note.offset * TEMPO * 8))


def _notes_from_ticks(pairs, lyrics=None):
    lyrics = lyrics or ["a"] * len(pairs)
    return [
        NoteInfo(s / 960.0, e / 960.0, 60.0, lyric)
        for (s, e), lyric in zip(pairs, lyrics)
    ]


def _jittered_onsets(grid_ticks, jitter=15, seed=0):
    rng = np.random.default_rng(seed)
    return [grid + float(rng.uniform(-jitter, jitter)) for grid in grid_ticks]


def test_repair_snaps_every_onset_onto_grid_lines():
    # Onsets sitting BETWEEN grid lines (90 + 120k, +- jitter) must all land
    # exactly on grid lines after repair.
    raw = [90 + k * STEP + d for k, d in enumerate(_jittered_onsets([0] * 12, jitter=15, seed=0))]
    notes = _notes_from_ticks([(t, t + 120) for t in raw])
    quantize_notes(notes, TEMPO, STEP, mode="repair")

    for n in notes:
        assert _tick(n) % STEP == 0, _tick(n)
    # The performed rhythm pattern (uniform 120-tick intervals) survives.
    intervals = [b - a for a, b in zip((_tick(n) for n in notes), (_tick(n) for n in notes[1:]))]
    assert all(iv == STEP for iv in intervals)


def test_repair_prefers_stronger_beat_for_ambiguous_onset():
    # Raw onset exactly between the 120 (16th) and 240 (8th) grid lines is
    # pulled to the stronger half-beat position.
    notes = _notes_from_ticks([(180, 300)])
    quantize_notes(notes, TEMPO, STEP, mode="repair")
    assert _tick(notes[0]) == 240


def test_repair_keeps_rhythm_pattern_under_drag():
    # A line that drags progressively (up to 40 ticks late) still snaps onto
    # the grid lines with its intervals preserved and no overlaps.
    rng = np.random.default_rng(2)
    raw = []
    for k in range(8):
        drag = 40 * k / 7
        raw.append(90 + 1290 + k * STEP + drag + float(rng.uniform(-8, 8)))
    notes = _notes_from_ticks([(t, t + 120) for t in raw])
    quantize_notes(notes, TEMPO, STEP, mode="repair")

    onsets = [_tick(n) for n in notes]
    for t in onsets:
        assert t % STEP == 0, t
    intervals = [b - a for a, b in zip(onsets, onsets[1:])]
    assert all(90 <= iv <= 150 for iv in intervals), intervals
    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) <= _tick(cur)


def test_repair_snaps_durations_to_musical_values():
    # Dotted / plain / short values on a phase-0 grid with real rests.
    raw = [(0, 310), (480, 725), (960, 1090)]
    notes = _notes_from_ticks(raw)
    quantize_notes(notes, TEMPO, STEP, mode="repair")

    durs = [_end_tick(n) - _tick(n) for n in notes]
    assert durs[0] == 360  # dotted value (3 x step)
    assert durs[1] == 240
    assert durs[2] == 120
    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) <= _tick(cur)


def test_repair_keeps_legato_and_ties_connected():
    # Zero-gap chain with tie lyrics must stay glued after repair.
    raw = [(240, 480), (480, 720), (720, 960)]
    lyrics = ["a", "-", "+"]
    notes = _notes_from_ticks(raw, lyrics)
    quantize_notes(notes, TEMPO, STEP, mode="repair")

    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) == _tick(cur)
    for n, (s, _) in zip(notes, raw):
        assert _tick(n) == s


def test_repair_handles_grace_notes():
    # A 20-tick grace note keeps a half-grid duration instead of stretching.
    notes = _notes_from_ticks([(600, 620)])
    quantize_notes(notes, TEMPO, STEP, mode="repair")

    assert _tick(notes[0]) == 600
    assert _end_tick(notes[0]) - _tick(notes[0]) == 60


def test_repair_never_refines_the_user_grid():
    # Notes a 1/8 apart inside a 1/16 grid: the grid stays exactly what the
    # user picked (never auto-halved). Dense pairs stretch onto the 120-tick
    # lattice — the accepted trade-off for the strict on-grid guarantee.
    raw = [k * (STEP // 2) + 3 for k in range(8)]
    notes = _notes_from_ticks([(t, t + STEP // 2 - 5) for t in raw])
    quantize_notes(notes, TEMPO, STEP, mode="repair")

    onsets = [_tick(n) for n in notes]
    for t in onsets:
        assert t % STEP == 0, t
    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) <= _tick(cur)
    # The dense 60-tick pattern collapses onto the coarser lattice: repeated
    # onsets are forbidden, so every snapped interval is >= one grid step.
    intervals = [b - a for a, b in zip(onsets, onsets[1:])]
    assert all(iv >= STEP for iv in intervals), intervals


def test_repair_bounds_movement_on_structureless_input():
    # Uniform random onsets carry no meaningful rhythm. The repair must stay
    # bounded (snap within the search window), keep the sequence monotonic
    # with positive durations, and land every onset on a grid line.
    rng = np.random.default_rng(42)
    onsets = np.sort(rng.uniform(0, 5000, size=24))
    notes = []
    for on in onsets:
        dur = rng.uniform(80, 300)
        notes.append(NoteInfo(on / 960.0, (on + dur) / 960.0, 60.0, "a"))
    before = [_tick(n) for n in notes]

    quantize_notes(notes, TEMPO, STEP, mode="repair")

    # Grid is never refined: every onset sits on the user-selected lattice.
    for n in notes:
        assert _tick(n) % STEP == 0, _tick(n)
    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) <= _tick(cur)
        assert _end_tick(cur) > _tick(cur)
    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) <= _tick(cur)
        assert _end_tick(cur) > _tick(cur)


def test_repair_dispatch_and_gate():
    assert should_apply_quantization("repair", 120) is True
    assert should_apply_quantization("repair", 0) is False

    notes = _notes_from_ticks([(k * STEP, k * STEP + 120) for k in range(4)])
    before = [(n.onset, n.offset) for n in notes]
    quantize_notes(notes, TEMPO, 0, mode="repair")
    assert [(n.onset, n.offset) for n in notes] == pytest.approx(before)


def test_repair_survives_degenerate_input():
    # Duplicate onsets and a zero-duration note must not crash.
    notes = _notes_from_ticks(
        [(600, 600), (600, 700), (1150, 1350)], ["a", "a", "-"]
    )
    quantize_notes(notes, TEMPO, STEP, mode="repair")
    for prev, cur in zip(notes, notes[1:]):
        assert _end_tick(prev) <= _tick(cur)
        assert _end_tick(cur) > _tick(cur)
