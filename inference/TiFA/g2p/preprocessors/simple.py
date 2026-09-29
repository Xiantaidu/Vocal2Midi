import string
import unicodedata

from ..registry import preprocessor
from .base import Preprocessor


@preprocessor(id="filter-punctuation")
class FilterPunctuation(Preprocessor):
    """Split tokens on punctuation and discard the punctuation characters."""

    _punctuation_set = frozenset(
        string.punctuation
        + "，。；：“”‘’（）【】《》…—～、·"
        + "！？"
    ) - {"'", "-"}  # Keep apostrophes and hyphens, which are common in phonetic transcriptions.

    def process(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for t in tokens:
            part: list[str] = []
            for c in t:
                if c in self._punctuation_set:
                    if part:
                        result.append("".join(part))
                        part = []
                else:
                    part.append(c)
            if part:
                result.append("".join(part))
        return result


@preprocessor(id="lowercase")
class LowercasePreprocessor(Preprocessor):
    """Lowercase all tokens."""

    def process(self, tokens: list[str]) -> list[str]:
        return [t.lower() for t in tokens]


@preprocessor(id="strip-whitespace")
class StripWhitespacePreprocessor(Preprocessor):
    """Strip leading and trailing whitespace from tokens, removing empty ones."""

    def process(self, tokens: list[str]) -> list[str]:
        return [s for t in tokens if (s := t.strip())]


@preprocessor(id="remove-accents")
class RemoveAccentsPreprocessor(Preprocessor):
    """Decompose accented characters and strip combining marks.

    ``café résumé naïve`` -> ``cafe resume naive``.
    """

    def process(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for t in tokens:
            decomposed = unicodedata.normalize("NFKD", t)
            stripped = "".join(
                c for c in decomposed
                if not unicodedata.combining(c)
            )
            result.append(stripped)
        return result
