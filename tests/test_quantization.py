"""Tests for the quantization dispatch: simple mode + the smart rhythmic
alignment engine, strictly per the reference smart_quantize.py.

The smart engine runs on its own fixed 128th-note correction grid (15 ticks
at PPQ 480) with no bar/measure awareness: ``quantization_step`` only
toggles quantization on/off, and a failed chain (infeasible overlaps) leaves
the notes untouched — the reference semantics, intentionally not "fixed".

All fixtures are synthetic NoteInfo lists at BPM 120, where one tick is
1/960 s (PPQ 480).
"""

import logging

from fractions import Fraction

import pytest

from inference.io.note_io import NoteInfo
from inference.quant.quantization import quantize_notes, should_apply_quantization
from inference.quant.smart_quantize import (
    GRID,
    QUARTER,
    TICKS_PER_QUARTER,
    smart_quantize_musical,
)

TEMPO = 120.0
STEP = 120  # 1/16 note grid


def _tick(note) -> int:
    return int(round(note.onset * TEMPO * 8))


def _end_tick(note) -> int:
    return int(round(note.offset * TEMPO * 8))


def _spans(notes):
    return [(_tick(n), _end_tick(n)) for n in notes]


def _notes_from_ticks(pairs, lyrics=None):
    lyrics = lyrics or ["a"] * len(pairs)
    return [
        NoteInfo(s / 960.0, e / 960.0, 60.0, lyric)
        for (s, e), lyric in zip(pairs, lyrics)
    ]


# ── simple mode ──────────────────────────────────────────────────────


def test_simple_mode_snaps_to_grid():
    notes = _notes_from_ticks([(73, 221), (350, 407)])
    quantize_notes(notes, TEMPO, STEP, mode="simple")
    assert _spans(notes) == [(120, 240), (360, 480)]


def test_simple_mode_keeps_contiguous_notes_glued():
    notes = _notes_from_ticks([(115, 245), (245, 355)])
    quantize_notes(notes, TEMPO, STEP, mode="simple")
    # the joint follows the next onset so the legato chain survives
    assert _spans(notes) == [(120, 240), (240, 360)]


def test_legacy_mode_names_dispatch_to_simple():
    # Modes removed from the UI must stay quantizing (not crash) for old
    # settings: unknown modes fall back to the simple quantizer.
    notes = _notes_from_ticks([(73, 221)])
    quantize_notes(notes, TEMPO, STEP, mode="repair")
    assert _spans(notes) == [(120, 240)]


def test_zero_step_disables_every_mode():
    notes = _notes_from_ticks([(113, 291), (350, 407)])
    before = [(n.onset, n.offset) for n in notes]
    for mode in ("simple", "smart", "repair"):
        current = _notes_from_ticks([(113, 291), (350, 407)])
        quantize_notes(current, TEMPO, 0, mode=mode)
        assert [(n.onset, n.offset) for n in current] == pytest.approx(before)


def test_should_apply_quantization_gate():
    assert should_apply_quantization("smart", 120) is True
    assert should_apply_quantization("simple", 0) is False


# ── smart mode ───────────────────────────────────────────────────────


def test_engine_quarter_is_the_beat_and_grid_is_a_128th():
    # Regression pin for the reference's historical bug: TICKS_PER_QUARTER
    # used to be QUARTER/4, so its MIDI path read every position as 1/4
    # while the engine-scale path stayed correct. The corrected engine
    # defines one quarter note = one beat = 32 correction-grid columns, and
    # the ladder is relative to the beat (no bar/measure concept).
    assert TICKS_PER_QUARTER == QUARTER
    assert QUARTER == GRID * 32
    assert Fraction(QUARTER, 480) * 480 == QUARTER


def test_engine_maps_one_ppq_quarter_to_one_beat():
    # End-to-end: a clean quarter-long note must be recognized as exactly
    # one on-beat span and stay where it is.
    res = smart_quantize_musical([(0, 480)], simplicity=0.0, ticks_per_quarter=480)
    assert res.ok
    assert res.notes == [(0, 480)]


def test_smart_mode_keeps_on_grid_phrases_untouched():
    # A phrase already sitting on strong positions is optimal for every
    # simplicity level: the engine must not move it.
    raw = [(0, 480), (480, 960), (960, 1440)]
    for simplicity in (0.0, 2.5, 5.0):
        notes = _notes_from_ticks(raw)
        quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=simplicity)
        assert _spans(notes) == raw


def test_smart_mode_lands_every_boundary_on_the_correction_grid():
    notes = _notes_from_ticks([(85, 205), (205, 325), (325, 445)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)

    spans = _spans(notes)
    assert spans == [(90, 120), (120, 240), (240, 360)]
    for start, end in spans:
        assert start % 15 == 0
        assert end % 15 == 0
        assert end > start
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start >= prev_end


def test_smart_mode_straightens_a_dragged_line():
    # A line dragging progressively (up to 40 ticks late): every boundary
    # lands on the engine's correction grid while the performed rhythm
    # survives; the final boundary is chosen freely by the DP.
    raw = [int(1380 + k * STEP + 40 * k / 7) for k in range(8)]
    notes = _notes_from_ticks([(t, t + STEP) for t in raw])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)

    spans = _spans(notes)
    assert spans == [
        (1380, 1440), (1560, 1620), (1680, 1740), (1800, 1860),
        (1920, 1980), (2040, 2100), (2160, 2220), (2280, 2340),
    ]
    for start, end in spans:
        assert start % 15 == 0 and end % 15 == 0
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start >= prev_end


def test_smart_mode_inflates_micro_rests_to_grid_rests():
    # Reference behavior: every rest becomes at least one correction-grid
    # column, so 15-tick gaps turn into full rests. Not "fixed" on purpose.
    notes = _notes_from_ticks([(0, 225), (240, 465), (480, 705)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)
    assert _spans(notes) == [(0, 120), (240, 360), (480, 600)]


def test_smart_mode_preserves_authentic_rests():
    notes = _notes_from_ticks([(0, 310), (480, 725), (960, 1090)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)

    spans = _spans(notes)
    assert spans == [(0, 240), (480, 720), (960, 1080)]
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start > prev_end  # the rests survive


def test_smart_mode_ignores_the_grid_step_choice():
    # The engine's grid is its own fixed correction grid; the user's step
    # value only toggles quantization on/off.
    a = _notes_from_ticks([(37, 880), (1415, 2010)])
    quantize_notes(a, TEMPO, 480, mode="smart", simplicity=2.5)
    b = _notes_from_ticks([(37, 880), (1415, 2010)])
    quantize_notes(b, TEMPO, 60, mode="smart", simplicity=2.5)
    assert _spans(a) == _spans(b) == [(30, 960), (1440, 1920)]


def test_smart_mode_simplicity_controls_aggressiveness():
    # The reference demo phrase: conservative barely moves durations, higher
    # simplicity pulls the same material onto stronger positions.
    phrase = [
        (0 * 480 + 37, 2 * 480 - 80),
        (2 * 480 + 65, 3 * 480 + 40),
        (3 * 480 - 25, 4 * 480 + 90),
        (4 * 480 + 120, 6 * 480 - 60),
        (6 * 480 + 30, 7 * 480 + 15),
        (7 * 480 - 70, 9 * 480 + 150),
        (9 * 480 + 55, 10 * 480 - 30),
        (10 * 480 + 95, 12 * 480 - 110),
    ]
    results = {}
    for simplicity in (0.0, 2.5):
        notes = _notes_from_ticks(phrase)
        quantize_notes(notes, TEMPO, 60, mode="smart", simplicity=simplicity)
        results[simplicity] = _spans(notes)
        for start, end in results[simplicity]:
            assert start % 15 == 0 and end % 15 == 0
    assert results[0.0] != results[2.5]
    assert results[0.0][-1] == (4920, 5640)
    assert results[2.5][-1] == (4800, 5520)


def test_smart_mode_absorbs_small_stacked_overlaps():
    # Reference behavior: the ±16-column band lets the engine squeeze small
    # overlaps onto the grid instead of failing.
    notes = _notes_from_ticks([(0, 240), (0, 240), (0, 240)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)
    assert _spans(notes) == [(0, 120), (120, 180), (180, 240)]


def test_smart_mode_leaves_notes_unchanged_when_chain_is_infeasible(caplog):
    # A note fully containing a later one collapses the chain span; the
    # engine reports failure and the notes stay exactly as sung — the
    # upstream "left unchanged" semantics, intentionally not "fixed".
    notes = _notes_from_ticks([(0, 1920), (60, 120)])
    before = [(n.onset, n.offset) for n in notes]
    with caplog.at_level(logging.WARNING):
        quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)

    assert any("left notes unchanged" in r.message for r in caplog.records)
    assert [(n.onset, n.offset) for n in notes] == before


def test_smart_mode_quantizes_single_sub_grid_note():
    # A single note shorter than one correction-grid column still forms a
    # one-column chain and snaps onto the grid.
    notes = _notes_from_ticks([(62401, 62413)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)
    assert _spans(notes) == [(62400, 62415)]
