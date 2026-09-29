"""Kanji lexicon converter for TiFA's Japanese chain.

Draws candidate readings from the ja_g2p (LAKE-G2P v5) lexicon pack
(``lexicon.txz``: surface -> kana readings with source/bonus/weight). Per the
TiFA design, the converter emits **all** candidate readings as G2PWord
readings — selection is deferred to the pronunciation-scoring DP, which picks
in the context of the actual audio (mirroring how cpp_pinyin feeds multiple
readings for Chinese polyphones). The lexicon's weight/bonus columns only
order and cap the candidates (top-k per surface).
"""
from __future__ import annotations

import lzma
import pathlib
import re

from inference.TiFA.g2p.converters.base import Converter, G2PGroup, G2PReading, G2PWord
from inference.TiFA.g2p.converters.japanese import JapaneseKanaConverter
from inference.TiFA.g2p.converters.text import split_words
from inference.TiFA.g2p.registry import converter

# Same source ranking as the ja_g2p runtime's UnifiedLexiconProvider.
_PRIORITY_ORDER = (
    "gold", "lyric_memory", "alnum", "rules", "haqumei", "pyopenjtalk",
    "jmdict", "unidic", "yomogi_dict", "kanjidic", "copy",
)
_SOURCE_PRIORITY = {
    name: len(_PRIORITY_ORDER) - index
    for index, name in enumerate(_PRIORITY_ORDER)
}

_KANJI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f々〆ヶ]")
_MAX_SURFACE_LEN = 16
_MAX_READINGS = 16

_lexicon_cache: dict[str, dict[str, list[str]]] = {}


def _contains_kanji(text: str) -> bool:
    return _KANJI_RE.search(text) is not None


class JapaneseLexiconConverter(Converter):
    """Claims kanji-bearing spans and emits every lexicon reading as a candidate.

    Readings are kana strings; they are converted to phoneme paths through the
    standard JapaneseKanaConverter two-phase (kana -> romaji -> dictionary).
    """

    def __init__(self, lexicon_path: str, dict_path: str):
        self._lexicon_path = pathlib.Path(lexicon_path)
        self._kana = JapaneseKanaConverter(dict_path=dict_path, double_written_sokuon=False)
        self._surfaces: dict[str, list[str]] | None = None

    def _load(self) -> dict[str, list[str]]:
        if self._surfaces is not None:
            return self._surfaces
        cache_key = str(self._lexicon_path.resolve())
        cached = _lexicon_cache.get(cache_key)
        if cached is not None:
            self._surfaces = cached
            return cached

        entries: dict[str, list[tuple[str, tuple[int, float, float]]]] = {}
        with lzma.open(self._lexicon_path, "rt", encoding="utf-8") as fh:
            section = None
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("##"):
                    if line == "##end":
                        break
                    if line.startswith("##section"):
                        section = line.split(None, 1)[1].strip()
                    continue
                if section != "pack":
                    continue
                parts = line.split("\t")
                if len(parts) != 5:
                    continue
                surface, reading, source, bonus, weight = parts
                if not (surface and reading and _contains_kanji(surface)):
                    continue
                if len(surface) > _MAX_SURFACE_LEN:
                    continue
                try:
                    order_key = (
                        -_SOURCE_PRIORITY.get(source, -1),
                        -float(weight),
                        -float(bonus),
                    )
                except ValueError:
                    continue
                entries.setdefault(surface, []).append((reading, order_key))

        surfaces: dict[str, list[str]] = {}
        for surface, candidates in entries.items():
            candidates.sort(key=lambda item: item[1])
            seen: set[str] = set()
            readings: list[str] = []
            for reading, _key in candidates:
                if reading not in seen:
                    seen.add(reading)
                    readings.append(reading)
            surfaces[surface] = readings[:_MAX_READINGS]

        _lexicon_cache[cache_key] = surfaces
        self._surfaces = surfaces
        return surfaces

    def _reading_to_path(self, reading: str):
        """Kana reading -> phoneme path via the kana converter's two phases.

        Each group's script carries its kana character (not romaji) so the
        aligner can emit per-mora kana display tokens; romaji is recovered by
        merging the group's CV phonemes.
        """
        words = self._kana.convert(reading)
        kana_tokens = split_words(reading)
        if len(words) != len(kana_tokens):
            return None
        path = []
        for word, kana in zip(words, kana_tokens):
            if not word.readings or not word.readings[0].paths:
                return None
            for group in word.readings[0].paths[0]:
                path.append(G2PGroup(script=kana, phonemes=list(group.phonemes)))
        return path or None

    def find(self, text: str) -> tuple[int, int] | None:
        surfaces = self._load()
        n = len(text)
        i = 0
        while i < n:
            limit = min(_MAX_SURFACE_LEN, n - i)
            for k in range(limit, 0, -1):
                sub = text[i : i + k]
                if sub in surfaces:
                    return i, i + k
            i += 1
        return None

    def convert(self, text: str) -> list[G2PWord]:
        surfaces = self._load()
        readings = surfaces.get(text)
        if not readings:
            return []
        # ONE word carrying ALL candidate readings; the scoring DP picks.
        word_readings: list[G2PReading] = []
        for reading in readings:
            path = self._reading_to_path(reading)
            if path is None:
                continue
            word_readings.append(G2PReading(paths=[path]))
        if not word_readings:
            return []
        return [G2PWord(text=text, readings=word_readings)]


@converter(id="japanese-lexicon", language="ja")
class RegisteredJapaneseLexiconConverter(JapaneseLexiconConverter):
    """Registry-visible subclass so the factory can build it by id."""
