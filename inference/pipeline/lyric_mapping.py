"""Map aligned lyrics onto notes that GAME transcribed on its own.

GAME runs without lyric constraints, so its notes are the authority on pitch
and timing. The forced aligner (TiFA / HubertFA) only decides which syllable
starts on which note. Each singable syllable becomes a :class:`LyricUnit`
anchored at its vowel nucleus (the same anchor the old constrained path used),
and a monotonic DP matches units to note onsets:

* a unit matches a note onset, or a split point inside a long note when two
  syllables were sung legato on one pitch and GAME merged them;
* a unit that no note can reasonably take is dropped (logged, never shifted
  onto a distant note, which would push every later lyric off by one);
* a note that no unit takes is a melisma "-" when it continues the previous
  syllable, and stays unlabelled ("") otherwise, e.g. a hum after a rest.

Lyric symbols on the produced notes:

* first note of a syllable: the display token (hanzi / kana / romaji / word);
* later syllables of one English word: "+";
* further notes of the same syllable: "-";
* no lyric known: "" (exporters apply their own default).

Literal "-" / "+" / breath tokens in the display text are removed before
pairing, so they can never be confused with those markers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from inference.API.game_api import extract_vowel_boundaries
from inference.io.note_io import NoteInfo

logger = logging.getLogger(__name__)

SUSTAIN_LYRIC = "-"
SYLLABLE_LYRIC = "+"

_NON_LYRIC_TOKENS = {"SP", "AP", "EP", "GS", "br", "sil", "pau", "cl"}

# Shortest note piece a split may leave behind; below this a split would
# create an unsingable sliver instead of fixing a merged note.
MIN_FRAGMENT_SEC = 0.06
# Within this distance an onset/anchor match is plain aligner jitter.
MAX_SHIFT_SEC = 0.35
# Using a split point instead of a real GAME onset costs this much extra, so
# an onset slightly further away still wins over cutting a note.
SPLIT_PENALTY = 0.08
# Cost of discarding a syllable: higher than any jitter-sized shift.
DROP_COST = 0.5
# Cost of a note that starts outside every syllable span and gets no lyric.
ORPHAN_COST = 0.3
# A note starting this close after its syllable's span still continues it.
TERRITORY_TOLERANCE_SEC = 0.08


@dataclass(frozen=True)
class LyricUnit:
    """One singable syllable: where its note should start and how far it may run."""

    start: float
    end: float
    lyric: str
    group: int
    head_lyric: str


@dataclass
class MappingStats:
    units: int = 0
    notes: int = 0
    matched: int = 0
    splits: int = 0
    dropped: list = field(default_factory=list)
    unlabeled: int = 0


def sanitize_display_tokens(tokens) -> list[str]:
    """Remove tokens that are not lyrics: breath marks and bare symbols.

    Matching ASR or reference text can carry "AP"/"EP" or a literal "-" (the
    sequence aligner's own gap sentinel); attached to a note those would read
    as breath or melisma markers. Removing them keeps the remaining lyrics
    glued to their words; a syllable left without a token falls back to the
    aligner's word text inside extract_vowel_boundaries.
    """
    cleaned = []
    for token in tokens or []:
        text = str(token or "").strip()
        if text in _NON_LYRIC_TOKENS or not any(ch.isalnum() and ch != "ー" for ch in text):
            continue
        cleaned.append(text)
    return cleaned


def _valid_words(result_word) -> list:
    """Aligner words with finite, increasing times, in time order."""
    words = []
    for word in result_word or []:
        try:
            start = float(word.start)
            end = float(word.end)
        except (TypeError, ValueError, AttributeError):
            continue
        if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
            continue
        words.append(word)
    words.sort(key=lambda w: float(w.start))
    return words


def build_lyric_units(result_word, display_tokens, language, offset: float = 0.0) -> list[LyricUnit]:
    """Turn one chunk's aligned words into lyric units with absolute times.

    The nucleus/territory rules are those of extract_vowel_boundaries: a unit
    starts at its vowel and runs to the next syllable's vowel, or to its word
    end before a pause. English words yield one unit per syllable, the later
    ones labelled "+".
    """
    words = _valid_words(result_word)
    if not words:
        return []
    durs, vuvs, lyrics = extract_vowel_boundaries(
        words, sanitize_display_tokens(display_tokens), language=language
    )

    units = []
    cursor = float(offset)
    group = -1
    head_lyric = ""
    for dur, vuv, lyric in zip(durs, vuvs, lyrics):
        start, cursor = cursor, cursor + float(dur)
        if not vuv or dur <= 0:
            continue
        lyric = str(lyric or "").strip()
        if lyric != SYLLABLE_LYRIC or group < 0:
            group += 1
            head_lyric = lyric if lyric != SYLLABLE_LYRIC else ""
        units.append(LyricUnit(start=start, end=cursor, lyric=lyric, group=group, head_lyric=head_lyric))
    return units


def _match_cost(unit: LyricUnit, time: float, is_split: bool) -> float:
    shift = abs(time - unit.start)
    inside = unit.start - TERRITORY_TOLERANCE_SEC <= time < unit.end
    if shift > MAX_SHIFT_SEC and not inside:
        return np.inf
    return shift + (SPLIT_PENALTY if is_split else 0.0)


def _skip_cost(prev_unit: LyricUnit | None, candidate) -> float:
    time, _note_idx, is_split = candidate
    if is_split:
        return 0.0  # an unused split point simply leaves the note whole
    if _in_territory(prev_unit, time):
        return 0.0  # melisma of the syllable sung before it
    return ORPHAN_COST


def _in_territory(unit: LyricUnit | None, time: float) -> bool:
    return unit is not None and unit.start - TERRITORY_TOLERANCE_SEC <= time < unit.end + TERRITORY_TOLERANCE_SEC


def _candidates(notes: list[NoteInfo], units: list[LyricUnit]) -> list[tuple[float, int, bool]]:
    """Note onsets plus split points where a syllable starts inside a note."""
    candidates = [(note.onset, idx, False) for idx, note in enumerate(notes)]
    onsets = np.asarray([note.onset for note in notes], dtype=np.float64)
    for unit in units:
        idx = int(np.searchsorted(onsets, unit.start, side="right")) - 1
        if idx < 0:
            continue
        note = notes[idx]
        if unit.start - note.onset >= MIN_FRAGMENT_SEC and note.offset - unit.start >= MIN_FRAGMENT_SEC:
            candidates.append((unit.start, idx, True))
    candidates.sort(key=lambda c: (c[0], c[2]))

    kept = []
    last_cut: dict[int, float] = {}
    for time, idx, is_split in candidates:
        if is_split and time - last_cut.get(idx, notes[idx].onset) < MIN_FRAGMENT_SEC:
            continue
        last_cut[idx] = time
        kept.append((time, idx, is_split))
    return kept


def _solve(units: list[LyricUnit], candidates) -> tuple[dict[int, int], list[int]]:
    """Monotonic unit/candidate matching; returns ({candidate: unit}, dropped units)."""
    n_units, n_cands = len(units), len(candidates)
    inf = float("inf")
    cost = [[inf] * (n_cands + 1) for _ in range(n_units + 1)]
    back = [[0] * (n_cands + 1) for _ in range(n_units + 1)]
    cost[0][0] = 0.0
    # Row-major order visits every predecessor first; on equal cost the first
    # relaxation wins, so match beats drop beats skip.
    for i in range(n_units + 1):
        prev_unit = units[i - 1] if i > 0 else None
        for k in range(n_cands + 1):
            base = cost[i][k]
            if base == inf:
                continue
            if i < n_units and k < n_cands:
                value = base + _match_cost(units[i], candidates[k][0], candidates[k][2])
                if value < cost[i + 1][k + 1]:
                    cost[i + 1][k + 1] = value
                    back[i + 1][k + 1] = 1
            if i < n_units:
                value = base + DROP_COST
                if value < cost[i + 1][k]:
                    cost[i + 1][k] = value
                    back[i + 1][k] = 2
            if k < n_cands:
                value = base + _skip_cost(prev_unit, candidates[k])
                if value < cost[i][k + 1]:
                    cost[i][k + 1] = value
                    back[i][k + 1] = 3

    matched: dict[int, int] = {}
    dropped: list[int] = []
    i, k = n_units, n_cands
    while i > 0 or k > 0:
        step = back[i][k]
        if step == 1:
            i, k = i - 1, k - 1
            matched[k] = i
        elif step == 2:
            i -= 1
            dropped.append(i)
        else:
            k -= 1
    dropped.reverse()
    return matched, dropped


def map_lyrics_to_notes(notes, units: list[LyricUnit], stats: MappingStats | None = None) -> list[NoteInfo]:
    """Label (and where needed split) one chunk's GAME notes with its lyric units.

    Pitches and note boundaries come from GAME; a note is only cut where a
    syllable demonstrably starts inside it. Returns new NoteInfo objects.
    """
    stats = stats if stats is not None else MappingStats()
    notes = sorted(
        (n for n in notes if np.isfinite(n.onset) and np.isfinite(n.offset) and n.offset > n.onset),
        key=lambda n: n.onset,
    )
    stats.units += len(units)
    stats.notes += len(notes)
    if not units or not notes:
        stats.dropped.extend(unit.lyric for unit in units)
        stats.unlabeled += len(notes)
        return [NoteInfo(n.onset, n.offset, n.pitch, "") for n in notes]

    candidates = _candidates(notes, units)
    matched, dropped = _solve(units, candidates)
    stats.matched += len(matched)
    surviving_groups = {units[idx].group for idx in matched.values()}
    for idx in dropped:
        # A word whose other syllable survived only lost a syllable.
        lost = units[idx].group not in surviving_groups
        stats.dropped.append(units[idx].head_lyric or units[idx].lyric if lost else SYLLABLE_LYRIC)

    cuts: list[list[tuple[float, int | None]]] = [[] for _ in notes]
    for k, (time, note_idx, is_split) in enumerate(candidates):
        if is_split and k not in matched:
            continue
        if is_split:
            stats.splits += 1
        cuts[note_idx].append((time, matched.get(k)))

    fragments = []  # [start, end, pitch, note_idx, unit_idx]
    for note_idx, note in enumerate(notes):
        points = sorted(cuts[note_idx], key=lambda c: c[0])
        for pos, (start, unit_idx) in enumerate(points):
            end = points[pos + 1][0] if pos + 1 < len(points) else note.offset
            fragments.append([start, end, note.pitch, note_idx, unit_idx])

    result: list[NoteInfo] = []
    seen_groups: set[int] = set()
    last_unit: LyricUnit | None = None
    for pos, (start, end, pitch, note_idx, unit_idx) in enumerate(fragments):
        if unit_idx is None:
            nxt = fragments[pos + 1] if pos + 1 < len(fragments) else None
            if _in_territory(last_unit, start):
                lyric = SUSTAIN_LYRIC
            elif nxt is not None and nxt[3] == note_idx and nxt[4] is not None:
                # The note starts before the syllable the aligner placed in
                # it (an early onset): the whole note belongs to that syllable.
                nxt[0] = start
                continue
            else:
                lyric = ""
                stats.unlabeled += 1
            result.append(NoteInfo(start, end, pitch, lyric))
            continue

        unit = units[unit_idx]
        if unit.group not in seen_groups:
            seen_groups.add(unit.group)
            # A dropped head syllable hands its word to the first survivor.
            lyric = unit.head_lyric if unit.lyric == SYLLABLE_LYRIC and unit.head_lyric else unit.lyric
        else:
            lyric = unit.lyric
        last_unit = unit
        result.append(NoteInfo(start, end, pitch, lyric))
    return result


def _unlabeled(notes) -> list[NoteInfo]:
    return [NoteInfo(n.onset, n.offset, n.pitch, "") for n in notes]


def assign_chunk_lyrics(chunk_notes, chunk_offsets, outcome, language) -> list[NoteInfo]:
    """Label every chunk's notes from the alignment outcome; flat, onset-sorted.

    Chunks without a usable alignment keep their GAME notes unlabelled, which
    is exactly the pitch-only output.
    """
    aligned = outcome is not None and bool(getattr(outcome, "aligned", False))
    pred_dict = (getattr(outcome, "pred_dict", None) or {}) if aligned else {}
    chars_dict = (getattr(outcome, "chars_dict", None) or {}) if aligned else {}

    all_notes: list[NoteInfo] = []
    total = MappingStats()
    unaligned_chunks = []
    for chunk_idx, (notes, offset) in enumerate(zip(chunk_notes, chunk_offsets)):
        stem = f"chunk_{chunk_idx}"
        entry = pred_dict.get(stem)
        words = entry[2] if isinstance(entry, (tuple, list)) and len(entry) >= 3 else None
        if not words:
            if aligned and notes:
                unaligned_chunks.append(stem)
            all_notes.extend(_unlabeled(notes))
            continue

        stats = MappingStats()
        try:
            units = build_lyric_units(words, chars_dict.get(stem, []), language, offset=float(offset))
            mapped = map_lyrics_to_notes(notes, units, stats)
        except (TypeError, ValueError, AttributeError):
            # Malformed aligner output for one chunk must not cost the song.
            logger.warning(f"[Warning] {stem}: unusable alignment; its notes keep no lyrics.", exc_info=True)
            unaligned_chunks.append(stem)
            all_notes.extend(_unlabeled(notes))
            continue

        if stats.dropped:
            preview = " ".join(stats.dropped[:12]) + (" ..." if len(stats.dropped) > 12 else "")
            logger.warning(
                f"[Lyric Mapping] {stem}: {len(stats.dropped)} lyric(s) found no note and were dropped: {preview}"
            )
        total.units += stats.units
        total.notes += stats.notes
        total.matched += stats.matched
        total.splits += stats.splits
        total.dropped.extend(stats.dropped)
        total.unlabeled += stats.unlabeled
        all_notes.extend(mapped)

    if unaligned_chunks:
        preview = ", ".join(unaligned_chunks[:8]) + (" ..." if len(unaligned_chunks) > 8 else "")
        logger.warning(
            f"[Warning] {len(unaligned_chunks)} chunk(s) without usable alignment ({preview}) keep pitch-only notes."
        )
    if aligned:
        logger.info(
            f"[Lyric Mapping] {total.matched}/{total.units} lyric(s) placed on {total.notes} GAME note(s); "
            f"{total.splits} merged note(s) split, {len(total.dropped)} lyric(s) dropped, "
            f"{total.unlabeled} note(s) left without lyric."
        )
    all_notes.sort(key=lambda n: n.onset)
    return all_notes
