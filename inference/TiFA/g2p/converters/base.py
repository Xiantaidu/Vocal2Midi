from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeAlias

from ..preprocessors.base import Preprocessor
from ..registry import Language


@dataclass
class G2PGroup:
    """A contiguous pronunciation group and its pronunciation script."""

    script: str
    phonemes: list[str]


G2PPath: TypeAlias = list[G2PGroup]


@dataclass
class G2PReading:
    """Complete legal phoneme realizations of one reading."""

    paths: list[G2PPath] = field(default_factory=list)


@dataclass
class G2PWord:
    """One converter-defined output word, with its alternative readings.

    Its text and boundaries are determined by the converter.
    """

    text: str
    language: str | Language | None = None
    readings: list[G2PReading] = field(default_factory=list)


class Converter(ABC):
    language: tuple[str, ...] | Language | None = None

    @abstractmethod
    def find(self, text: str) -> tuple[int, int] | None:
        """Return the earliest accepted non-empty slice, or None.

        Offsets use Python string indices in the supplied text. Matching
        does not perform pronunciation inference.
        """
        ...

    # noinspection PyMethodMayBeStatic
    def preprocessors(self) -> list[Preprocessor]:
        """Transform a claimed input run; word text and count may change."""
        return []

    @abstractmethod
    def convert(self, text: str) -> list[G2PWord]:
        """Convert a claimed run into zero or more output words.

        The pipeline preserves output text and order and sets each word's
        language. Implementations may merge, split, rewrite, or omit inputs.
        """
        ...


class G2PConversionError(Exception):
    def __init__(self, unconverted_tokens: list[str]) -> None:
        self.unconverted_tokens = unconverted_tokens
        super().__init__(
            f"The following tokens could not be converted "
            f"by any converter in the chain: {unconverted_tokens}"
        )


def resolve_language(
    language: tuple[str, ...] | Language | None,
    languages: list[str] | None,
) -> str | Language | None:
    """Resolve explicit tags, preserving ANY for vocabulary-based resolution."""
    if language is None:
        return None
    if language is Language.ANY:
        return Language.ANY
    if not languages:
        return language[0]
    for tag in language:
        if tag in languages:
            return tag
    return None
