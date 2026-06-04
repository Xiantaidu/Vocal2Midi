from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inference.API.rmvpe_api import RmvpeResult


UCurveInterval = 5
# Match the current OpenUtau USTX schema more closely for external tools such
# as UtaFormatix, which expect complete expression descriptors.
USTX_RESOLUTION = 480
USTX_VERSION = "0.7"


@dataclass
class _PitchPoint:
    x: int
    y: int


def _to_ticks(seconds: float, tempo: float) -> int:
    return int(round(seconds * tempo * 8.0))


def _finite_notes(notes: list[Any]) -> list[Any]:
    valid_notes = []
    skipped = 0
    for note in notes:
        vals = (getattr(note, "onset", np.nan), getattr(note, "offset", np.nan), getattr(note, "pitch", np.nan))
        if not all(np.isfinite(v) for v in vals):
            skipped += 1
            continue
        if note.offset <= note.onset:
            skipped += 1
            continue
        valid_notes.append(note)
    if skipped:
        print(f"[Warning] Skipped {skipped} invalid note(s) during USTX export.")
    return valid_notes


def _median_filter(values: list[int], radius: int = 2) -> list[int]:
    if not values:
        return values
    result: list[int] = []
    for i in range(len(values)):
        left = max(0, i - radius)
        right = min(len(values), i + radius + 1)
        window = sorted(values[left:right])
        result.append(window[len(window) // 2])
    return result


def _adaptive_smooth(values: list[int], threshold_cents: float = 75.0, blend: float = 0.7) -> list[int]:
    if len(values) <= 2:
        return values
    out = values.copy()
    for i in range(1, len(out) - 1):
        neighbor_avg = (out[i - 1] + out[i + 1]) / 2.0
        delta = out[i] - neighbor_avg
        if abs(delta) > threshold_cents:
            out[i] = int(round(out[i] - delta * blend))
    return out


def _fill_short_gaps(points: list[_PitchPoint], max_gap_steps: int = 12) -> list[_PitchPoint]:
    if not points:
        return points
    expanded: list[_PitchPoint] = [points[0]]
    for p in points[1:]:
        prev = expanded[-1]
        gap_steps = max(0, (p.x - prev.x) // UCurveInterval - 1)
        if 0 < gap_steps <= max_gap_steps:
            for step in range(1, gap_steps + 1):
                ratio = step / (gap_steps + 1)
                expanded.append(
                    _PitchPoint(
                        x=prev.x + step * UCurveInterval,
                        y=int(round(prev.y + (p.y - prev.y) * ratio)),
                    )
                )
        expanded.append(p)
    return expanded


def _append_smoothed_points(xs: list[int], ys: list[int], points: list[_PitchPoint]):
    if not points:
        return
    processed = _fill_short_gaps(points)
    smoothed = _adaptive_smooth(_median_filter([p.y for p in processed]))
    for i, p in enumerate(processed):
        if xs and xs[-1] == p.x:
            ys[-1] = smoothed[i]
        else:
            xs.append(p.x)
            ys.append(smoothed[i])


def _build_pitd_curve(notes: list[Any], rmvpe: RmvpeResult, tempo: float) -> tuple[list[int], list[int]]:
    if not notes or rmvpe.midi_pitch.size == 0:
        return [], []

    notes = sorted(notes, key=lambda n: n.onset)
    xs: list[int] = []
    ys: list[int] = []
    pending: list[_PitchPoint] = []
    pending_note_idx = -1
    note_idx = 0
    for i, mp in enumerate(rmvpe.midi_pitch):
        t = i * rmvpe.time_step_seconds
        while note_idx + 1 < len(notes) and notes[note_idx].offset <= t:
            note_idx += 1
        if note_idx >= len(notes):
            break
        note = notes[note_idx]
        if pending and pending_note_idx != note_idx:
            _append_smoothed_points(xs, ys, pending)
            pending.clear()
            pending_note_idx = -1

        if not (note.onset <= t < note.offset):
            continue
        if np.isnan(mp):
            continue

        dur = max(0.0, note.offset - note.onset)
        note_offset = t - note.onset
        edge_trim = min(0.025, dur * 0.15)
        if dur > edge_trim * 2 and (note_offset < edge_trim or dur - note_offset <= edge_trim):
            continue

        x = int(round(_to_ticks(t, tempo) / UCurveInterval) * UCurveInterval)
        # Convert to cents offset relative to the note's semitone pitch.
        y = int(round(np.clip((float(mp) - float(note.pitch)) * 100.0, -1200, 1200)))
        pending_note_idx = note_idx
        if pending and pending[-1].x == x:
            pending[-1] = _PitchPoint(x=x, y=y)
        else:
            pending.append(_PitchPoint(x=x, y=y))

    _append_smoothed_points(xs, ys, pending)
    return xs, ys


def _default_expressions() -> dict:
    return {
        "dyn": {
            "name": "dynamics",
            "abbr": "dyn",
            "type": "Curve",
            "min": -240,
            "max": 120,
            "default_value": 0,
            "is_flag": False,
        },
        "pitd": {
            "name": "pitch deviation",
            "abbr": "pitd",
            "type": "Curve",
            "min": -1200,
            "max": 1200,
            "default_value": 0,
            "is_flag": False,
        },
        "clr": {
            "name": "voice color",
            "abbr": "clr",
            "type": "Options",
            "min": 0,
            "max": -1,
            "default_value": 0,
            "is_flag": False,
            "options": [],
        },
        "eng": {
            "name": "resampler engine",
            "abbr": "eng",
            "type": "Options",
            "min": 0,
            "max": 1,
            "default_value": 0,
            "is_flag": False,
            "options": ["", "worldline"],
        },
        "vel": {
            "name": "velocity",
            "abbr": "vel",
            "type": "Numerical",
            "min": 0,
            "max": 200,
            "default_value": 100,
            "is_flag": False,
        },
        "vol": {
            "name": "volume",
            "abbr": "vol",
            "type": "Numerical",
            "min": 0,
            "max": 200,
            "default_value": 100,
            "is_flag": False,
        },
        "atk": {
            "name": "attack",
            "abbr": "atk",
            "type": "Numerical",
            "min": 0,
            "max": 200,
            "default_value": 100,
            "is_flag": False,
        },
        "dec": {
            "name": "decay",
            "abbr": "dec",
            "type": "Numerical",
            "min": 0,
            "max": 100,
            "default_value": 0,
            "is_flag": False,
        },
        "gen": {
            "name": "gender",
            "abbr": "gen",
            "type": "Numerical",
            "min": -100,
            "max": 100,
            "default_value": 0,
            "is_flag": True,
            "flag": "g",
        },
        "genc": {
            "name": "gender (curve)",
            "abbr": "genc",
            "type": "Curve",
            "min": -100,
            "max": 100,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "bre": {
            "name": "breath",
            "abbr": "bre",
            "type": "Numerical",
            "min": 0,
            "max": 100,
            "default_value": 0,
            "is_flag": True,
            "flag": "B",
        },
        "brec": {
            "name": "breathiness (curve)",
            "abbr": "brec",
            "type": "Curve",
            "min": -100,
            "max": 100,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "lpf": {
            "name": "lowpass",
            "abbr": "lpf",
            "type": "Numerical",
            "min": 0,
            "max": 100,
            "default_value": 0,
            "is_flag": True,
            "flag": "H",
        },
        "norm": {
            "name": "normalize",
            "abbr": "norm",
            "type": "Numerical",
            "min": 0,
            "max": 100,
            "default_value": 86,
            "is_flag": True,
            "flag": "P",
        },
        "mod": {
            "name": "modulation",
            "abbr": "mod",
            "type": "Numerical",
            "min": 0,
            "max": 100,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "mod+": {
            "name": "modulation plus",
            "abbr": "mod+",
            "type": "Numerical",
            "min": 0,
            "max": 100,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "alt": {
            "name": "alternate",
            "abbr": "alt",
            "type": "Numerical",
            "min": 0,
            "max": 16,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "dir": {
            "name": "direct",
            "abbr": "dir",
            "type": "Options",
            "min": 0,
            "max": 1,
            "default_value": 0,
            "is_flag": False,
            "options": ["off", "on"],
        },
        "shft": {
            "name": "tone shift",
            "abbr": "shft",
            "type": "Numerical",
            "min": -36,
            "max": 36,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "shfc": {
            "name": "tone shift (curve)",
            "abbr": "shfc",
            "type": "Curve",
            "min": -1200,
            "max": 1200,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "tenc": {
            "name": "tension (curve)",
            "abbr": "tenc",
            "type": "Curve",
            "min": -100,
            "max": 100,
            "default_value": 0,
            "is_flag": False,
            "flag": "",
        },
        "voic": {
            "name": "voicing (curve)",
            "abbr": "voic",
            "type": "Curve",
            "min": 0,
            "max": 100,
            "default_value": 100,
            "is_flag": False,
            "flag": "",
        },
    }


def save_ustx(notes: list[Any], filepath: Path, tempo: float, rmvpe_result: RmvpeResult | None = None):
    notes = sorted(_finite_notes(notes), key=lambda n: n.onset)
    ustx_notes = []
    max_end_tick = 0
    for note in notes:
        pos = _to_ticks(note.onset, tempo)
        end = _to_ticks(note.offset, tempo)
        dur = max(10, end - pos)
        tone = int(np.clip(round(note.pitch), 0, 127))
        max_end_tick = max(max_end_tick, pos + dur)
        ustx_notes.append(
            {
                "position": pos,
                "duration": dur,
                "tone": tone,
                "lyric": note.lyric or "a",
                "pitch": {
                    "data": [
                        {"x": -40.0, "y": 0.0, "shape": "io"},
                        {"x": 15.0, "y": 0.0, "shape": "io"},
                    ],
                    "snap_first": True,
                },
                "vibrato": {
                    "length": 0,
                    "period": 175,
                    "depth": 25,
                    "in": 10,
                    "out": 10,
                    "shift": 0,
                    "drift": 0,
                    "vol_link": 0,
                },
                "phoneme_expressions": [],
                "phoneme_overrides": [],
            }
        )

    curves = []
    if rmvpe_result is not None:
        xs, ys = _build_pitd_curve(notes, rmvpe_result, tempo)
        if xs:
            curves.append({"abbr": "pitd", "xs": xs, "ys": ys})

    project = {
        "name": filepath.stem,
        "comment": "Generated by Vocal2Midi",
        "output_dir": "Vocal",
        "cache_dir": "UCache",
        "ustx_version": USTX_VERSION,
        "resolution": USTX_RESOLUTION,
        "bpm": float(tempo),
        "beat_per_bar": 4,
        "beat_unit": 4,
        "expressions": _default_expressions(),
        "exp_selectors": ["dyn", "pitd", "clr", "eng", "vel", "vol", "atk", "dec", "gen", "bre"],
        "exp_primary": 0,
        "exp_secondary": 1,
        "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
        "tempos": [{"position": 0, "bpm": float(tempo)}],
        "tracks": [
            {
                "phonemizer": "OpenUtau.Core.DefaultPhonemizer",
                "renderer_settings": {},
                "track_name": "Track1",
                "track_color": "Blue",
                "mute": False,
                "solo": False,
                "volume": 0,
                "pan": 0,
                "track_expressions": [],
                "voice_color_names": [""],
            }
        ],
        "voice_parts": [
            {
                "duration": max(480, max_end_tick),
                "name": filepath.stem,
                "comment": "",
                "track_no": 0,
                "position": 0,
                "notes": ustx_notes,
                "curves": curves,
            }
        ],
        "wave_parts": [],
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(project, f, allow_unicode=True, sort_keys=False)
