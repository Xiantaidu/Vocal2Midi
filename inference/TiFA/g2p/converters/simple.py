from ..registry import Language, converter
from .base import Converter, G2PGroup, G2PWord, G2PReading
from .text import split_words, word_spans


@converter(id="passthrough", language=Language.ANY)
class PassthroughConverter(Converter):
    """Catch-all converter that returns each token as its own phoneme.
    Typically placed last in a chain as a fallback for unconverted tokens."""

    def find(self, text: str) -> tuple[int, int] | None:
        return next(word_spans(text), None)

    def convert(self, text: str) -> list[G2PWord]:
        return [G2PWord(text=t, readings=[G2PReading(paths=[
            [G2PGroup(script=t, phonemes=[t])],
        ])]) for t in split_words(text)]


@converter(id="characters", language=Language.ANY)
class CharPhonemeConverter(Converter):
    """One-to-one character-to-phoneme mapping.
    Each character in a token is mapped to one or more phonemes."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def find(self, text: str) -> tuple[int, int] | None:
        for begin, end in word_spans(text):
            if all(c in self._mapping for c in text[begin:end]):
                return begin, end
        return None

    def convert(self, text: str) -> list[G2PWord]:
        result: list[G2PWord] = []
        for token in split_words(text):
            phonemes: list[str] = []
            for c in token:
                phonemes.extend(self._mapping[c])
            path = [G2PGroup(script=token, phonemes=phonemes)] if phonemes else []
            result.append(G2PWord(text=token, readings=[G2PReading(paths=[path])]))
        return result
