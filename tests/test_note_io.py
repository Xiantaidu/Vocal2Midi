import math

import mido

from inference.io.note_io import NoteInfo, _save_midi, _save_text


def test_text_export_skips_invalid_notes_and_clamps_note_names(tmp_path):
    notes = [
        NoteInfo(0.0, 0.5, 200.0, "hi"),
        NoteInfo(0.6, 0.4, 60.0, "bad"),
        NoteInfo(0.7, 1.0, math.nan, "nan"),
    ]
    path = tmp_path / "notes.txt"

    _save_text(notes, path, "txt", "name", round_pitch=True)

    text = path.read_text(encoding="utf8")
    assert "G9" in text
    assert "bad" not in text
    assert "nan" not in text


def test_midi_export_skips_invalid_notes_and_clamps_pitch(tmp_path):
    notes = [
        NoteInfo(0.0, 0.5, -12.0, "low"),
        NoteInfo(0.6, 0.4, 60.0, "bad"),
    ]
    path = tmp_path / "notes.mid"

    _save_midi(notes, path, tempo=120)

    midi = mido.MidiFile(path)
    note_ons = [msg for track in midi.tracks for msg in track if msg.type == "note_on"]
    assert len(note_ons) == 1
    assert note_ons[0].note == 0
