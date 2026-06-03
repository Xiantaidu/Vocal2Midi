from inference.io.note_io import NoteInfo
from inference.quant.quantization import quantize_notes


def _ticks(sec: float, tempo: float) -> int:
    return round(sec * tempo * 8)


def _note_from_ticks(onset_tick: int, offset_tick: int, *, pitch: float = 60.0, lyric: str = "a") -> NoteInfo:
    tempo = 120.0
    scale = tempo * 8
    return NoteInfo(onset=onset_tick / scale, offset=offset_tick / scale, pitch=pitch, lyric=lyric)


def test_dp_quantization_respects_requested_grid():
    notes = [
        _note_from_ticks(73, 221, lyric="la"),
        _note_from_ticks(226, 407, pitch=62.0, lyric="-"),
        _note_from_ticks(430, 701, pitch=64.0, lyric="li"),
    ]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="dp")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks
    assert all(start % 120 == 0 for start, _ in quantized_ticks)
    assert all(end % 120 == 0 for _, end in quantized_ticks)
    assert all(end > start for start, end in quantized_ticks)
    assert all(
        quantized_ticks[i][0] >= quantized_ticks[i - 1][1]
        for i in range(1, len(quantized_ticks))
    )


def test_dp_quantization_changes_with_grid_size():
    notes_16 = [_note_from_ticks(120, 240, lyric="la")]
    notes_8 = [_note_from_ticks(120, 240, lyric="la")]

    quantize_notes(notes_16, tempo=120.0, quantization_step=120, mode="dp")
    quantize_notes(notes_8, tempo=120.0, quantization_step=240, mode="dp")

    ticks_16 = (_ticks(notes_16[0].onset, 120.0), _ticks(notes_16[0].offset, 120.0))
    ticks_8 = (_ticks(notes_8[0].onset, 120.0), _ticks(notes_8[0].offset, 120.0))

    assert ticks_16 == (120, 240)
    assert ticks_8 != ticks_16
    assert ticks_8[0] % 240 == 0
    assert ticks_8[1] % 240 == 0


def test_dp_quantization_keeps_auto_mode_for_zero_step():
    notes = [_note_from_ticks(113, 291, lyric="la")]

    quantize_notes(notes, tempo=120.0, quantization_step=0, mode="dp")

    onset_tick = _ticks(notes[0].onset, 120.0)
    offset_tick = _ticks(notes[0].offset, 120.0)
    assert onset_tick % 30 == 0
    assert offset_tick % 30 == 0
    assert (onset_tick, offset_tick) != (113, 291)


def test_bayes_quantization_prefers_stronger_beat_anchor():
    notes = [
        _note_from_ticks(70, 180, lyric="la"),
        _note_from_ticks(270, 380, pitch=62.0, lyric="li"),
        _note_from_ticks(470, 640, pitch=64.0, lyric="lu"),
    ]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks == [(0, 120), (240, 360), (480, 600)]


def test_bayes_quantization_preserves_repeated_short_note_pattern():
    notes = [
        _note_from_ticks(0, 110, lyric="la"),
        _note_from_ticks(125, 235, pitch=62.0, lyric="li"),
        _note_from_ticks(250, 420, pitch=64.0, lyric="lu"),
    ]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    durations = [_ticks(note.offset - note.onset, 120.0) for note in notes]
    assert durations == [120, 120, 120]


def test_bayes_quantization_keeps_consistent_phrase_phase_without_overstretching():
    notes = [
        _note_from_ticks(85, 205, lyric="la"),
        _note_from_ticks(205, 325, pitch=62.0, lyric="li"),
        _note_from_ticks(325, 445, pitch=64.0, lyric="lu"),
    ]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks == [(120, 240), (240, 360), (360, 480)]


def test_bayes_quantization_allows_sv_style_half_grid_grace_notes():
    notes = [_note_from_ticks(62401, 62413, lyric="pang")]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks == [(62400, 62460)]


def test_bayes_quantization_allows_full_grid_start_shift():
    notes = [
        _note_from_ticks(142313, 142464, lyric="he"),
        _note_from_ticks(142464, 142776, pitch=62.0, lyric="ran"),
    ]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks == [(142320, 142560), (142560, 142800)]


def test_bayes_quantization_prefers_sv_midpoint_duration_prior():
    notes = [_note_from_ticks(8166, 8480, lyric="la")]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks == [(8160, 8400)]


def test_bayes_quantization_pulls_back_consistently_late_phrase():
    spans = [
        (16635, 17127),
        (17127, 17598),
        (17598, 17684),
        (17684, 17823),
        (17823, 18076),
        (18076, 18719),
    ]
    notes = [_note_from_ticks(start, end, pitch=60.0 + i, lyric="la") for i, (start, end) in enumerate(spans)]

    quantize_notes(notes, tempo=120.0, quantization_step=120, mode="bayes")

    quantized_ticks = [(_ticks(note.onset, 120.0), _ticks(note.offset, 120.0)) for note in notes]
    assert quantized_ticks == [
        (16560, 17040),
        (17040, 17520),
        (17520, 17640),
        (17640, 17760),
        (17760, 18000),
        (18000, 18720),
    ]
