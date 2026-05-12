import math

import yaml

from inference.API.ustx_api import save_ustx
from inference.io.note_io import NoteInfo


def test_ustx_export_skips_invalid_notes_and_clamps_tone(tmp_path):
    path = tmp_path / "notes.ustx"
    notes = [
        NoteInfo(0.0, 0.5, 200.0, "a"),
        NoteInfo(0.6, 0.4, 60.0, "bad"),
        NoteInfo(0.7, 1.0, math.nan, "nan"),
    ]

    save_ustx(notes, path, tempo=120.0)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    exported = data["voice_parts"][0]["notes"]
    assert len(exported) == 1
    assert exported[0]["tone"] == 127
    assert exported[0]["lyric"] == "a"
