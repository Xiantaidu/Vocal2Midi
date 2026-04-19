import csv
import pathlib
from dataclasses import dataclass
from typing import Literal

import librosa
import mido
import numpy as np


@dataclass
class NoteInfo:
    onset: float
    offset: float
    pitch: float
    lyric: str = ""


def pad_1d_arrays(arrays: list[np.ndarray], pad_value=0.0) -> np.ndarray:
    """Pad a list of 1D numpy arrays to the maximum length and stack them."""
    if not arrays:
        return np.array([])
    max_len = max(len(arr) for arr in arrays)
    if max_len == 0:
        return np.zeros((len(arrays), 1), dtype=arrays[0].dtype)

    padded = []
    for arr in arrays:
        pad_width = max_len - len(arr)
        padded.append(np.pad(arr, (0, pad_width), constant_values=pad_value))
    return np.stack(padded)


def _save_midi(notes: list[NoteInfo], filepath: pathlib.Path, tempo: int = 120):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))

    sorted_notes = sorted(notes, key=lambda n: n.onset)

    last_abs_ticks = 0
    for note in sorted_notes:
        abs_onset_ticks = round(note.onset * tempo * 8)
        abs_offset_ticks = round(note.offset * tempo * 8)

        if abs_onset_ticks < last_abs_ticks:
            abs_onset_ticks = last_abs_ticks

        if abs_offset_ticks <= abs_onset_ticks:
            abs_offset_ticks = abs_onset_ticks + 1

        midi_pitch = round(note.pitch)

        delta_onset_ticks = abs_onset_ticks - last_abs_ticks
        note_duration_ticks = abs_offset_ticks - abs_onset_ticks

        if getattr(note, "lyric", ""):
            track.append(mido.MetaMessage('lyrics', text=note.lyric, time=delta_onset_ticks))
            track.append(mido.Message("note_on", note=midi_pitch, velocity=100, time=0))
        else:
            track.append(mido.Message("note_on", note=midi_pitch, velocity=100, time=delta_onset_ticks))

        track.append(mido.Message("note_off", note=midi_pitch, velocity=100, time=note_duration_ticks))

        last_abs_ticks = abs_offset_ticks

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with mido.MidiFile(charset="utf8") as midi_file:
        midi_file.tracks.append(track)
        midi_file.save(filepath)
    print(f"Saved MIDI file: {filepath}")


def _save_text(
    notes: list[NoteInfo],
    filepath: pathlib.Path,
    file_format: Literal["txt", "csv"],
    pitch_format: Literal["number", "name"],
    round_pitch: bool,
):
    onset_list = [f"{note.onset:.3f}" for note in notes]
    offset_list = [f"{note.offset:.3f}" for note in notes]
    pitch_list = []
    for note in notes:
        pitch = note.pitch
        if round_pitch:
            pitch = round(pitch)
        pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not round_pitch) if pitch_format == "name" else f"{pitch:.3f}"
        pitch_list.append(pitch_txt)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    lyric_list = [getattr(note, "lyric", "") for note in notes]
    has_lyrics = any(lyric_list)

    if file_format == "txt":
        with filepath.open(encoding="utf8", mode="w") as f:
            for onset, offset, pitch, lyric in zip(onset_list, offset_list, pitch_list, lyric_list):
                if has_lyrics:
                    f.write(f"{onset}\t{offset}\t{pitch}\t{lyric}\n")
                else:
                    f.write(f"{onset}\t{offset}\t{pitch}\n")
    elif file_format == "csv":
        with filepath.open(encoding="utf8", mode="w", newline="") as f:
            fieldnames = ["onset", "offset", "pitch"]
            if has_lyrics:
                fieldnames.append("lyric")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for onset, offset, pitch, lyric in zip(onset_list, offset_list, pitch_list, lyric_list):
                row = {"onset": onset, "offset": offset, "pitch": pitch}
                if has_lyrics:
                    row["lyric"] = lyric
                writer.writerow(row)
    print(f"Saved {file_format.upper()} file: {filepath}")
