"""ja_g2p_onnx converter for TiFA's Japanese chain.

Claims kanji-bearing runs and reads them with the LAKE-G2P v5 ONNX model
(``models/ja_g2p_onnx``). The adapter drives the base runtime's own
building blocks (``build_edges`` candidate graph, ``score`` edge scoring,
``decode_viterbi`` exact best path) plus a beam search over the same
scored graph, so every canonical word span carries a ranked pool of
candidate readings. Per the TiFA design the per-word candidates are then
polled by the pronunciation-scoring DP against the actual audio — the
model's contextual scores order the candidates, the audio picks one.

Each reading converts through the standard two-phase (kana -> romaji ->
DiffSinger Japanese dictionary) with group scripts carrying kana
characters for the per-mora romaji/kana display tokens. Long-vowel marks
(ー) inside a reading expand to the preceding mora's vowel (タワー ->
ta wa a); a standalone ー edge repeats the previous word's vowel.

Particle edges (surface exactly は/へ) are normalized to わ/え — the model
segments particles into standalone word edges, so the surface match is
reliable.
"""
from __future__ import annotations

import logging
import math

from inference.ja_onnx_g2p import (
    get_ja_onnx_runtime,
    get_ja_onnx_runtime_module,
    normalize_edge_reading,
)
from inference.TiFA.g2p.converters.base import Converter, G2PGroup, G2PReading, G2PWord
from inference.TiFA.g2p.converters.japanese import (
    _KANA_TO_ROMAJI,
    JapaneseKanaConverter,
    _kata_to_hira,
)
from inference.TiFA.g2p.converters.text import split_words
from inference.TiFA.g2p.registry import converter
from inference.TiFA.japanese_lexicon import _KANJI_RE, kana_reading_to_path

logger = logging.getLogger(__name__)

# Beam-16 is the audited width: the correct reading lies within the beam-16
# candidate grid (oracle CER 0.776% vs 2.462% top-1, ja_g2p EXPERIMENTS.md
# §18.1). The per-word cap matches the base runtime's MAX_SPAN_READINGS.
BEAM_WIDTH = 16  # N-best complete-cover paths kept by the beam search
MAX_READING_CANDIDATES = 16  # per-word readings handed to TiFA's scoring DP

_VOWEL_KANA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}
_VOWELS = frozenset("aiueo")


def _is_claim_char(ch: str) -> bool:
    """Characters a claimed run may span: kanji, kana, long mark, digits."""
    return bool(
        _KANJI_RE.match(ch)
        or ch == "ー"
        or ch.isdigit()
        or (0x3041 <= ord(ch) <= 0x30FF)
    )


def _expand_long_vowel_moras(moras: list[str]) -> list[str]:
    """Replace ー moras with the preceding mora's vowel kana (タワー -> タワあ).

    っ/ん and a leading ー cannot be extended; those marks are dropped.
    """
    out: list[str] = []
    last_vowel: str | None = None
    for mora in moras:
        if mora == "ー":
            if last_vowel:
                out.append(_VOWEL_KANA[last_vowel])
            continue
        out.append(mora)
        romaji = _KANA_TO_ROMAJI.get(_kata_to_hira(mora))
        if romaji and romaji[-1] in _VOWELS:
            last_vowel = romaji[-1]
        else:
            last_vowel = None
    return out


def _beam_nbest(module, local, scores, length: int, beam_width: int):
    """N-best complete-cover edge paths over the scored candidate graph.

    Mirrors the base runtime's ``decode_viterbi`` (allowed-edge filtering,
    morphosyntactic penalties) but keeps the top *beam_width* hypotheses
    per boundary position, yielding alternative segmentation/reading paths
    for the candidate pool. Returns ``[(score, (edge_index, ...)), ...]``
    best-first; empty when no complete path survives pruning.
    """
    allowed = module._allowed_edges(local, length)
    values = [float(s) for s in scores]
    by_start: list[list[int]] = [[] for _ in range(length + 1)]
    for index in allowed:
        edge = local[index]
        if 0 <= edge.start < edge.end <= length:
            by_start[edge.start].append(index)

    beams: dict[int, list[tuple[float, tuple[int, ...]]]] = {0: [(0.0, ())]}
    for position in range(length):
        hypotheses = beams.pop(position, None)
        if not hypotheses:
            continue
        hypotheses.sort(key=lambda item: -item[0])
        for score, path in hypotheses[:beam_width]:
            for index in by_start[position]:
                edge = local[index]
                penalty = 0.0
                if path:
                    penalty = module._morpho_penalty(local[path[-1]], edge)
                beams.setdefault(edge.end, []).append(
                    (score + values[index] + penalty, path + (index,)))
    complete = beams.get(length, [])
    complete.sort(key=lambda item: -item[0])
    return complete[:beam_width]


class JapaneseOnnxConverter(Converter):
    """Claims kanji-bearing runs and reads them with ja_g2p_onnx."""

    def __init__(self, dict_path: str, model_dir: str | None = None):
        self._kana = JapaneseKanaConverter(dict_path=dict_path, double_written_sokuon=False)
        self._model_dir = model_dir

    def find(self, text: str) -> tuple[int, int] | None:
        match = _KANJI_RE.search(text)
        if match is None:
            return None
        start = match.start()
        end = start + 1
        while end < len(text) and _is_claim_char(text[end]):
            end += 1
        return start, end

    def _reading_paths(self, readings: list[str]) -> list[G2PReading]:
        out: list[G2PReading] = []
        seen: set = set()
        for reading in readings:
            expanded = "".join(_expand_long_vowel_moras(split_words(reading)))
            path = kana_reading_to_path(self._kana, expanded)
            if path is None:
                continue
            key = tuple((group.script, tuple(group.phonemes)) for group in path)
            if key in seen:
                continue
            seen.add(key)
            out.append(G2PReading(paths=[path]))
        return out

    def convert(self, text: str) -> list[G2PWord]:
        module = get_ja_onnx_runtime_module(self._model_dir)
        runtime = get_ja_onnx_runtime(self._model_dir)
        if module is None or runtime is None:
            logger.warning("[JaG2P-ONNX] runtime unavailable; span dropped")
            return []
        try:
            norm = module.normalize_surface(text)
            edges = module.build_edges(norm, runtime.lexicon)
        except Exception as e:
            logger.warning(f"[JaG2P-ONNX] candidate graph failed for {text!r}: {e}; span dropped")
            return []

        words: list[G2PWord] = []
        for start, end in module.safe_windows(norm, edges):
            local = module.local_edges(norm, edges, start, end)
            local_text = norm[start:end]
            try:
                scores = runtime.score(local_text, local)
            except Exception as e:
                logger.warning(f"[JaG2P-ONNX] scoring failed for {local_text!r}: {e}; span dropped")
                continue

            try:
                best_indices, _score = module.decode_viterbi(scores, local, end - start)
            except ValueError:
                best_indices = []
            beam_paths = _beam_nbest(module, local, scores, end - start, BEAM_WIDTH)
            if not best_indices and beam_paths:
                best_indices = list(beam_paths[0][1])
            if not best_indices:
                continue

            # Candidate pool per span from the exact best path plus the beam
            # N-best: {span -> {reading -> edge score}}.
            pool: dict[tuple[int, int], dict[str, float]] = {}

            def _add(path) -> None:
                for index in path:
                    edge = local[index]
                    reading = normalize_edge_reading(edge.surface, edge.reading)
                    span_pool = pool.setdefault((edge.start, edge.end), {})
                    span_pool[reading] = max(
                        span_pool.get(reading, -math.inf), float(scores[index]))

            _add(best_indices)
            for _path_score, path in beam_paths:
                _add(path)

            entries: list[tuple[str, list[G2PReading]]] = []
            for index in best_indices:
                edge = local[index]
                best_reading = normalize_edge_reading(edge.surface, edge.reading)
                span_pool = pool.get((edge.start, edge.end), {})
                ranked = sorted(span_pool.items(), key=lambda item: -item[1])
                ordered = [reading for reading, _score in ranked]
                if best_reading in ordered:
                    ordered.remove(best_reading)
                ordered.insert(0, best_reading)
                readings = self._reading_paths(ordered[:MAX_READING_CANDIDATES])
                entries.append((edge.surface, readings))

            words.extend(self._materialize(entries))
        return words

    def _materialize(self, entries: list[tuple[str, list[G2PReading]]]) -> list[G2PWord]:
        """Attach vowel repeats to standalone ー spans and drop empty words."""
        out: list[G2PWord] = []
        previous: G2PWord | None = None
        for surface, readings in entries:
            if not readings and surface == "ー" and previous is not None:
                vowel = _last_vowel_phoneme(previous)
                if vowel:
                    path = [G2PGroup(script=_VOWEL_KANA[vowel], phonemes=[vowel])]
                    readings = [G2PReading(paths=[path])]
            if not readings:
                logger.debug(f"[JaG2P-ONNX] no readable candidate for {surface!r}; word dropped")
                continue
            word = G2PWord(text=surface, readings=readings)
            out.append(word)
            previous = word
        return out


def _last_vowel_phoneme(word: G2PWord) -> str | None:
    """Final vowel phoneme of a word's top reading (for ー repeats)."""
    for reading in word.readings:
        for path in reading.paths:
            if path and path[-1].phonemes:
                phoneme = path[-1].phonemes[-1]
                if phoneme in _VOWELS:
                    return phoneme
        break
    return None


@converter(id="japanese-onnx", language="ja")
class RegisteredJapaneseOnnxConverter(JapaneseOnnxConverter):
    """Registry-visible subclass so the factory can build it by id."""
