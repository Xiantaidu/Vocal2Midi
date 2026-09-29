"""Shared character ranges and word boundaries for converters."""

from collections.abc import Callable, Iterator


SMALL_KANA = frozenset("ゃゅょャュョぁぃぅぇぉァィゥェォ")


def is_kana(char: str) -> bool:
    cp = ord(char)
    return 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x2E80 <= cp <= 0x2EFF
        or 0x2F00 <= cp <= 0x2FDF
        or 0x3000 <= cp <= 0x303F
        or is_kana(char)
        or 0x31F0 <= cp <= 0x31FF
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xAC00 <= cp <= 0xD7AF
        or 0xF900 <= cp <= 0xFAFF
        or 0xFF00 <= cp <= 0xFFEF
    )


def word_spans(text: str) -> Iterator[tuple[int, int]]:
    """Split on whitespace and CJK characters, retaining kana digraphs."""
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        begin = i
        if is_kana(text[i]) and i + 1 < len(text) and text[i + 1] in SMALL_KANA:
            i += 2
        elif _is_cjk(text[i]):
            i += 1
        else:
            i += 1
            while i < len(text) and not text[i].isspace() and not _is_cjk(text[i]):
                i += 1
        yield begin, i


def split_words(text: str) -> list[str]:
    return [text[begin:end] for begin, end in word_spans(text)]


def find_run(text: str, accepts: Callable[[str], bool]) -> tuple[int, int] | None:
    """Find the first contiguous run of accepted characters."""
    for begin, char in enumerate(text):
        if accepts(char):
            end = begin + 1
            while end < len(text) and accepts(text[end]):
                end += 1
            return begin, end
    return None
