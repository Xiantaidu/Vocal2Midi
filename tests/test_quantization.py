"""Tests for the quantization dispatch: simple mode + the smart rhythmic
alignment engine, strictly per the reference smart_quantize.py.

The smart engine aligns to the correction grid the caller selects
(``quantization_step``: 480 = 1/4, 240 = 1/8, 120 = 1/16, 60 = 1/32,
30 = 1/64) with no bar/measure awareness: the importance ladder is
relative to the beat, and a failed chain (infeasible overlaps) leaves
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
    assert spans == [(120, 240), (240, 360), (360, 480)]
    for start, end in spans:
        assert start % STEP == 0
        assert end % STEP == 0
        assert end > start
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start >= prev_end


def test_smart_mode_straightens_a_dragged_line():
    # A line dragging progressively (up to 40 ticks late): every boundary
    # lands on the 1/16 grid while the performed rhythm survives.
    raw = [int(1380 + k * STEP + 40 * k / 7) for k in range(8)]
    notes = _notes_from_ticks([(t, t + STEP) for t in raw])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)

    spans = _spans(notes)
    assert len(spans) == len(raw)
    for start, end in spans:
        assert start % STEP == 0 and end % STEP == 0
        assert end > start
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start >= prev_end


def test_smart_mode_absorbs_rests_shorter_than_half_a_grid_cell():
    # A 15-tick rest is well below the 1/16 grid, so it rounds to zero: the
    # notes become contiguous instead of each gap inflating to a full cell.
    notes = _notes_from_ticks([(0, 225), (240, 465), (480, 705)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)
    assert _spans(notes) == [(0, 240), (240, 480), (480, 720)]


def test_smart_mode_preserves_authentic_rests():
    notes = _notes_from_ticks([(0, 310), (480, 725), (960, 1090)])
    quantize_notes(notes, TEMPO, STEP, mode="smart", simplicity=2.5)

    spans = _spans(notes)
    assert spans == [(0, 240), (480, 720), (960, 1080)]
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start > prev_end  # the rests survive


def test_smart_mode_honors_the_grid_step_choice():
    # The step is the correction grid the engine aligns to: a finer step
    # resolves rhythmic detail a coarser one rounds away.
    raw = [(0, 240), (240, 480), (495, 510), (510, 720), (720, 960)]

    fine = _notes_from_ticks(raw)
    quantize_notes(fine, TEMPO, 30, mode="smart", simplicity=0.0)
    coarse = _notes_from_ticks(raw)
    quantize_notes(coarse, TEMPO, 60, mode="smart", simplicity=0.0)

    assert _spans(fine) != _spans(coarse)
    for start, end in _spans(fine):
        assert start % 30 == 0 and end % 30 == 0
    for start, end in _spans(coarse):
        assert start % 60 == 0 and end % 60 == 0


def test_smart_mode_one_sixty_fourth_keeps_sixty_fourth_detail():
    # Regression: a 1/64 grid must place boundaries on 30-tick columns
    # instead of collapsing them onto the coarser 1/16 positions.
    notes = _notes_from_ticks([(0, 240), (240, 480), (720, 960)])
    quantize_notes(notes, TEMPO, 30, mode="smart", simplicity=0.0)
    for start, end in _spans(notes):
        assert start % 30 == 0 and end % 30 == 0


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
    # The ±band window lets the engine squeeze small overlaps onto the grid
    # instead of failing when the grid is fine enough for them.
    notes = _notes_from_ticks([(0, 240), (0, 240), (0, 240)])
    quantize_notes(notes, TEMPO, 60, mode="smart", simplicity=2.5)
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
    # A single note shorter than one grid column still forms a one-column
    # chain and snaps onto the grid.
    notes = _notes_from_ticks([(62401, 62413)])
    quantize_notes(notes, TEMPO, 30, mode="smart", simplicity=2.5)
    assert _spans(notes) == [(62400, 62430)]
