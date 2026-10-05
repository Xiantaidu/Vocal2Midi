"""pyopenjtalk converter for TiFA's Japanese chain."""
from __future__ import annotations

import logging
from inference.TiFA.g2p.converters.base import Converter, G2PGroup, G2PReading, G2PWord
from inference.TiFA.g2p.converters.japanese import JapaneseKanaConverter, _kata_to_hira
from inference.TiFA.g2p.converters.text import split_words
from inference.TiFA.g2p.registry import converter
from inference.TiFA.japanese_lexicon import _KANJI_RE, kana_reading_to_path
from inference.TiFA.japanese_onnx import _is_claim_char, _expand_long_vowel_moras, _last_vowel_phoneme, _VOWEL_KANA

logger = logging.getLogger(__name__)


class JapanesePyopenjtalkConverter(Converter):
    """Claims kanji-bearing runs and reads them with pyopenjtalk."""

    def __init__(self, dict_path: str):
        self._kana = JapaneseKanaConverter(dict_path=dict_path, double_written_sokuon=False)

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
        try:
            import pyopenjtalk
            words_info = pyopenjtalk.run_frontend(text)
        except Exception as e:
            logger.warning(f"[JaG2P-pyopenjtalk] frontend failed for {text!r}: {e}; span dropped")
            return []

        entries: list[tuple[str, list[G2PReading]]] = []
        for word in words_info:
            surface = word.get("string", "")
            pron = word.get("pron", "")
            if not surface or not pron:
                continue
            pron_clean = pron.replace("’", "").replace("、", "").replace("。", "")
            reading_hira = _kata_to_hira(pron_clean)
            readings = self._reading_paths([reading_hira])
            entries.append((surface, readings))

        out: list[G2PWord] = []
        previous: G2PWord | None = None
        for surface, readings in entries:
            if not readings and surface == "ー" and previous is not None:
                vowel = _last_vowel_phoneme(previous)
                if vowel:
                    path = [G2PGroup(script=_VOWEL_KANA[vowel], phonemes=[vowel])]
                    readings = [G2PReading(paths=[path])]
            if not readings:
                continue
            word = G2PWord(text=surface, readings=readings)
            out.append(word)
            previous = word
        return out


@converter(id="japanese-pyopenjtalk", language="ja")
class RegisteredJapanesePyopenjtalkConverter(JapanesePyopenjtalkConverter):
    """Registry-visible subclass so the factory can build it by id."""
