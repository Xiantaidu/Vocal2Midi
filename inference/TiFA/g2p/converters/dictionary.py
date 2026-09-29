"""Pronunciation-dictionary G2P converter and shared loader."""

import re
from abc import ABC
from pathlib import Path

from ..registry import converter
from .base import Converter, G2PGroup, G2PPath, G2PWord, G2PReading
from .paradigm import PronunciationScriptConverter
from .text import split_words, word_spans

_PRON_UNSAFE_RE = re.compile(r"\s*\(\d+\)$")


def load_pronunciation_dict(path: str | Path) -> dict[str, list[list[str]]]:
    """Load a tab-separated pronunciation dictionary.

    Format: ``<key>\\t<ph1> <ph2> ...``
    Duplicate keys accumulate pronunciations.

    Returns ``{key: [[ph, ...], ...]}``  --  a mapping from lookup keys to
    lists of alternative phoneme sequences.
    """
    result: dict[str, list[list[str]]] = {}
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, phoneme_str = line.partition("\t")
            if not phoneme_str:
                continue
            result.setdefault(key, []).append(phoneme_str.split())
    return result


@converter(id="dictionary", language=None)
class DictionaryConverter(Converter):
    """Pronunciation dictionary lookup. Loads a tab-separated file:
    ``<word>\\t<ph1> <ph2> ...``. Duplicate words accumulate pronunciations;
    ``word(N)`` and ``word (N)`` suffixes are variant forms of the same word.
    """

    def __init__(self, dict_path: str) -> None:
        raw = load_pronunciation_dict(dict_path)
        self._dict: dict[str, list[list[str]]] = {}
        for key, prons in raw.items():
            base = _PRON_UNSAFE_RE.sub("", key).lower()
            self._dict.setdefault(base, []).extend(prons)

    def find(self, text: str) -> tuple[int, int] | None:
        for begin, end in word_spans(text):
            if text[begin:end].lower() in self._dict:
                return begin, end
        return None

    def convert(self, text: str) -> list[G2PWord]:
        result: list[G2PWord] = []
        for token in split_words(text):
            pronunciations = self._dict.get(token.lower())
            if pronunciations is None:
                raise KeyError(
                    f"DictionaryConverter: token '{token}' not in dictionary. "
                    f"find should have filtered it."
                )
            paths = [
                [G2PGroup(script=token, phonemes=list(p))] if p else []
                for p in pronunciations
            ]
            result.append(G2PWord(text=token, readings=[G2PReading(paths=paths)]))
        return result


class PronunciationScriptDictionaryConverter(PronunciationScriptConverter, ABC):
    """``PronunciationScriptConverter`` whose *script_to_paths* step is a
    dictionary lookup loaded from *dict_path*.

    When *dict_path* is omitted each script token passes through unchanged.
    Subclasses implement ``text_to_scripts``.
    """

    def __init__(self, dict_path: str) -> None:
        super().__init__()
        self._script_dict = load_pronunciation_dict(dict_path)

    def script_to_paths(self, script: str) -> list[G2PPath]:
        pronunciations = self._script_dict.get(script)
        if pronunciations is None:
            raise KeyError(
                f"Script token {script!r} not found in script-to-phoneme dict."
            )
        return [
            [G2PGroup(script=script, phonemes=list(p))] if p else []
            for p in pronunciations
        ]
