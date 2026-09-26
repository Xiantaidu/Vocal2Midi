from __future__ import annotations

import logging
from typing import Any

from inference.quant.smart_quantize import smart_quantize_musical

logger = logging.getLogger(__name__)


def _ticks_from_sec(t: float, tempo: float) -> int:
    return int(round(t * tempo * 8))


def _quantize_notes_simple(notes: list[Any], tempo: float, quantization_step: int):
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)

    orig_onsets = [n.onset for n in notes]
    orig_offsets = [n.offset for n in notes]

    q_onsets = []
    for onset in orig_onsets:
        ticks = _ticks_from_sec(onset, tempo)
        q_ticks = round(ticks / quantization_step) * quantization_step
        q_onsets.append(q_ticks)

    for i in range(1, len(q_onsets)):
        if q_onsets[i] <= q_onsets[i - 1]:
            q_onsets[i] = q_onsets[i - 1] + quantization_step

    q_offsets = []
    for i in range(len(notes)):
        ticks = _ticks_from_sec(orig_offsets[i], tempo)
        q_ticks = round(ticks / quantization_step) * quantization_step

        if i < len(notes) - 1 and abs(orig_offsets[i] - orig_onsets[i + 1]) < 1e-3:
            q_ticks = q_onsets[i + 1]

        if q_ticks <= q_onsets[i]:
            q_ticks = q_onsets[i] + quantization_step

        if i < len(notes) - 1 and q_ticks > q_onsets[i + 1]:
            q_ticks = q_onsets[i + 1]

        q_offsets.append(q_ticks)

    for i in range(len(notes)):
        notes[i].onset = q_onsets[i] / (tempo * 8)
        notes[i].offset = q_offsets[i] / (tempo * 8)


def _quantize_notes_smart(notes: list[Any], tempo: float, quantization_step: int, simplicity: float):
    """Smart rhythmic alignment, strictly per the reference smart_quantize.py.

    The notes live in PPQ-480 ticks (one quarter note = tempo*8 seconds-ticks),
    so the engine's own ``smart_quantize_musical`` wrapper applies unchanged:
    the grid is the engine's fixed 32nd note (60 ticks), ``quantization_step``
    only toggles quantization on/off, and overlapping notes that make the
    rhythmic chain infeasible report failure — the notes are then left
    unchanged, exactly like the upstream MIDI path.
    """
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)
    tick_pairs = [(_ticks_from_sec(n.onset, tempo), _ticks_from_sec(n.offset, tempo)) for n in notes]
    result = smart_quantize_musical(tick_pairs, simplicity=simplicity, ticks_per_quarter=480)
    if not result.ok:
        logger.warning("smart quantize left notes unchanged: %s", result.message)
        return

    for note, (start_tick, end_tick) in zip(notes, result.notes):
        note.onset = start_tick / (tempo * 8)
        note.offset = end_tick / (tempo * 8)
    logger.info("smart quantize: %s", result.message)


def quantize_notes(
    notes: list[Any],
    tempo: float,
    quantization_step: int,
    mode: str = "simple",
    simplicity: float = 2.5,
):
    mode = (mode or "simple").lower()
    # "不量化" ("no quantization"; step <= 0) disables every mode at the public entrypoint.
    if quantization_step <= 0:
        return
    if mode == "smart":
        _quantize_notes_smart(notes, tempo, quantization_step, simplicity)
    else:
        _quantize_notes_simple(notes, tempo, quantization_step)


def should_apply_quantization(mode: str, quantization_step: int) -> bool:
    # "不量化" ("no quantization"; step <= 0) must disable every mode.
    del mode
    return quantization_step > 0
