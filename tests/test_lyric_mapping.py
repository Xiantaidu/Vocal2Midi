"""Lyric-to-note mapping: DP matching, splits, melisma and symbol safety."""
import numpy as np
import pytest

from inference.HubertFA.tools.align_word import Phoneme, Word
from inference.io.note_io import NoteInfo
from inference.pipeline.lyric_mapping import (
    MappingStats,
    assign_chunk_lyrics,
    build_lyric_units,
    map_lyrics_to_notes,
    sanitize_display_tokens,
)


def _word(start, end, text, phones):
    word = Word(start, end, text)
    word.phonemes = [Phoneme(s, e, t) for s, e, t in phones]
    return word


def _unit(start, end, lyric, group=0, head=None):
    from inference.pipeline.lyric_mapping import LyricUnit

    return LyricUnit(start, end, lyric, group, head if head is not None else lyric)


def _note(start, end, pitch=60.0):
    return NoteInfo(start, end, pitch, "")


def _map(notes, units):
    stats = MappingStats()
    out = map_lyrics_to_notes(notes, units, stats)
    return out, stats


def test_onset_jitter_matches_each_syllable_to_its_note():
    notes = [_note(0.03, 0.5), _note(0.5, 1.0, 62), _note(1.0, 1.5, 64)]
    units = [_unit(0.0, 0.5, "a"), _unit(0.48, 1.0, "b"), _unit(1.05, 1.5, "c")]

    out, stats = _map(notes, units)

    assert [(n.onset, n.offset, n.lyric) for n in out] == [
        (0.03, 0.5, "a"), (0.5, 1.0, "b"), (1.0, 1.5, "c"),
    ]
    assert stats.dropped == []


def test_extra_notes_within_one_syllable_become_melisma():
    notes = [_note(0.0, 0.3), _note(0.3, 0.6, 62), _note(0.6, 0.9, 64), _note(1.0, 1.5)]
    units = [_unit(0.0, 1.0, "a"), _unit(1.0, 1.5, "b")]

    out, stats = _map(notes, units)

    assert [n.lyric for n in out] == ["a", "-", "-", "b"]
    assert stats.dropped == []


def test_legato_two_syllables_on_one_note_are_split():
    notes = [_note(0.0, 1.0)]
    units = [_unit(0.0, 0.5, "a"), _unit(0.5, 1.0, "b")]

    out, stats = _map(notes, units)

    assert [(n.onset, n.offset, n.pitch, n.lyric) for n in out] == [
        (0.0, 0.5, 60.0, "a"), (0.5, 1.0, 60.0, "b"),
    ]
    assert stats.splits == 1


def test_hallucinated_syllable_is_dropped_not_forced():
    notes = [_note(0.0, 0.5), _note(2.0, 2.5)]
    units = [_unit(0.0, 0.5, "a", 0), _unit(1.0, 1.5, "x", 1), _unit(2.0, 2.5, "b", 2)]

    out, stats = _map(notes, units)

    assert [n.lyric for n in out] == ["a", "b"]
    assert stats.dropped == ["x"]


def test_note_after_a_rest_without_lyric_stays_unlabelled():
    notes = [_note(0.0, 0.5), _note(1.5, 2.0)]
    units = [_unit(0.0, 0.5, "a")]

    out, stats = _map(notes, units)

    assert [n.lyric for n in out] == ["a", ""]
    assert stats.unlabeled == 1


def test_early_note_onset_joins_the_syllable_inside_it():
    # GAME started the note before the aligner's nucleus; the note must not
    # become an orphan "-" ahead of its own syllable.
    notes = [_note(0.0, 0.3), _note(0.6, 1.2)]
    units = [_unit(0.0, 0.3, "a"), _unit(0.85, 1.2, "b")]

    out, stats = _map(notes, units)

    assert [(n.onset, n.lyric) for n in out] == [(0.0, "a"), (0.6, "b")]
    assert stats.dropped == []


def test_english_word_syllables_keep_plus_marker():
    notes = [_note(0.06, 0.3), _note(0.3, 0.5, 62), _note(0.52, 0.86, 64)]
    units = [
        _unit(0.06, 0.30, "possible", 0, "possible"),
        _unit(0.30, 0.52, "+", 0, "possible"),
        _unit(0.52, 0.86, "+", 0, "possible"),
    ]

    out, _ = _map(notes, units)

    assert [n.lyric for n in out] == ["possible", "+", "+"]


def test_dropped_head_syllable_hands_word_to_survivor():
    notes = [_note(0.5, 1.0)]
    units = [_unit(0.0, 0.1, "possible", 0, "possible"), _unit(0.5, 1.0, "+", 0, "possible")]

    out, stats = _map(notes, units)

    assert [n.lyric for n in out] == ["possible"]
    # Only the syllable position was lost, not the word.
    assert stats.dropped == ["+"]


def test_too_many_syllables_on_a_tiny_note_drop_the_middle_one():
    notes = [_note(0.0, 0.2)]
    units = [_unit(0.0, 0.05, "a", 0), _unit(0.05, 0.1, "b", 1), _unit(0.1, 0.2, "c", 2)]

    out, stats = _map(notes, units)

    assert [n.lyric for n in out] == ["a", "c"]
    assert stats.dropped == ["b"]


def test_unsorted_or_invalid_notes_are_tolerated():
    notes = [_note(0.5, 1.0), _note(0.0, 0.3), NoteInfo(float("nan"), 1.0, 60.0)]
    units = [_unit(0.0, 0.3, "a"), _unit(0.5, 1.0, "b")]

    out, _ = _map(notes, units)

    assert [(n.onset, n.lyric) for n in out] == [(0.0, "a"), (0.5, "b")]


def test_sanitize_display_tokens_drops_breaths_and_bare_symbols():
    # Breath marks, the aligner's "-" sentinel and a bare prolonged mark are
    # removed outright so they can never land on a note as a lyric.
    tokens = ["AP", "EP", "SP", "-", "+", "ー", "", None, "你"]
    assert sanitize_display_tokens(tokens) == ["你"]
    # Multi-char real lyrics with punctuation inside survive.
    assert sanitize_display_tokens(["world-class"]) == ["world-class"]


def test_build_lyric_units_zh_anchors_at_vowels_and_pairs_tokens():
    words = [
        _word(0.0, 0.3, "ni", [(0.0, 0.1, "n"), (0.1, 0.3, "i")]),
        _word(0.3, 0.6, "hao", [(0.3, 0.4, "h"), (0.4, 0.6, "au")]),
    ]

    units = build_lyric_units(words, ["你", "好"], "zh", offset=1.0)

    # The onset consonants (n, h) belong to no unit; anchors are the vowels,
    # shifted by the chunk offset.
    assert [(round(u.start, 3), round(u.end, 3), u.lyric) for u in units] == [
        (1.1, 1.4, "你"), (1.4, 1.6, "好"),
    ]


def test_build_lyric_units_drops_non_lyric_tokens_without_desync():
    words = [
        _word(0.0, 0.3, "ni", [(0.0, 0.1, "n"), (0.1, 0.3, "i")]),
        _word(0.3, 0.6, "hao", [(0.3, 0.4, "h"), (0.4, 0.6, "au")]),
    ]

    units = build_lyric_units(words, ["AP", "-", "好"], "zh")

    # The noise tokens are removed, so the first singable word takes the real
    # lyric; the second has no token left and falls back to its word text.
    assert [u.lyric for u in units] == ["好", "hao"]


def test_assign_chunk_lyrics_mixes_aligned_and_unaligned_chunks():
    words = [_word(0.1, 0.4, "ni", [(0.1, 0.4, "i")])]

    class _Outcome:
        aligned = True
        pred_dict = {"chunk_0": (None, 0.5, words)}
        chars_dict = {"chunk_0": ["你"]}

    chunk_notes = [
        [_note(0.1, 0.4), _note(0.4, 0.6, 62)],
        [_note(1.0, 1.5)],
    ]

    out = assign_chunk_lyrics(chunk_notes, [0.0, 1.0], _Outcome(), "zh")

    assert [(round(n.onset, 2), n.lyric) for n in out] == [(0.1, "你"), (0.4, "-"), (1.0, "")]


def test_assign_chunk_lyrics_without_alignment_keeps_notes_unlabelled():
    class _Outcome:
        aligned = False
        pred_dict = {}
        chars_dict = {}

    out = assign_chunk_lyrics([[_note(0.0, 0.5)]], [0.0], _Outcome(), "zh")

    assert [(n.onset, n.lyric) for n in out] == [(0.0, "")]


def test_assign_chunk_lyrics_survives_malformed_words():
    class _Outcome:
        aligned = True
        pred_dict = {"chunk_0": (None, 0.5, [object()])}
        chars_dict = {"chunk_0": ["你"]}

    out = assign_chunk_lyrics([[_note(0.0, 0.5)]], [0.0], _Outcome(), "zh")

    assert [(n.onset, n.lyric) for n in out] == [(0.0, "")]


def test_assign_chunk_lyrics_output_is_onset_sorted_across_chunks():
    class _Outcome:
        aligned = False
        pred_dict = {}
        chars_dict = {}

    out = assign_chunk_lyrics([[_note(2.0, 2.5)], [_note(1.0, 1.5)]], [0.0, 0.0], _Outcome(), "zh")

    assert [n.onset for n in out] == [1.0, 2.0]


def test_assign_chunk_lyrics_keeps_notes_when_an_offset_is_missing(caplog):
    """Regression: zip(chunk_notes, chunk_offsets) silently dropped the tail.

    A note list without a slicer offset must stay in the output (unlabelled),
    never vanish from the export.
    """
    import logging

    class _Outcome:
        aligned = False
        pred_dict = {}
        chars_dict = {}

    with caplog.at_level(logging.WARNING):
        out = assign_chunk_lyrics([[_note(0.0, 0.5)], [_note(1.0, 1.5)]], [0.0], _Outcome(), "zh")

    assert [(n.onset, n.lyric) for n in out] == [(0.0, ""), (1.0, "")]
    assert any("slicer offset" in record.message for record in caplog.records)
