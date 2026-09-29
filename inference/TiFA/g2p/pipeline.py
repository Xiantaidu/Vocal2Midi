from .converters.base import Converter, G2PConversionError, G2PWord, resolve_language
from .preprocessors.base import Preprocessor
from .registry import Language


class G2PPipeline:
    def __init__(
        self,
        preprocessors: list[Preprocessor] | None = None,
        converters: list[Converter] | None = None,
    ) -> None:
        self._preprocessors = preprocessors or []
        self._converters = converters or []

    def convert(
        self, text: str, *, languages: list[str] | None = None,
    ) -> list[G2PWord]:
        language_set = set(languages) if languages else None
        active = [
            c for c in self._converters
            if language_set is None
            or c.language is None
            or c.language is Language.ANY
            or any(ln in language_set for ln in c.language)
        ]
        if not active:
            raise ValueError("No converter matches the requested languages.")

        fragments = [text]
        for processor in self._preprocessors:
            fragments = processor.process(fragments)

        claimed: list[tuple[int, str]] = []
        unrecognized: list[str] = []
        for fragment in fragments:
            ranges: list[tuple[int, int, int]] = []
            pending = [(0, len(fragment), 0)]
            while pending:
                begin, end, priority = pending.pop()
                if begin == end:
                    continue
                part = fragment[begin:end]
                for index in range(priority, len(active)):
                    match = active[index].find(part)
                    if match is None:
                        continue
                    left, right = match
                    if not 0 <= left < right <= len(part):
                        raise ValueError(f"{type(active[index]).__name__}.find returned invalid range {match}.")
                    ranges.append((begin + left, begin + right, index))
                    # LIFO order routes the left remainder before the right.
                    pending.append((begin + right, end, index))
                    pending.append((begin, begin + left, index + 1))
                    break
                else:
                    if part.strip():
                        unrecognized.append(part)
            claimed.extend((index, fragment[begin:end]) for begin, end, index in sorted(ranges))

        if unrecognized:
            raise G2PConversionError(unrecognized)

        result: list[G2PWord] = []
        for index, part in claimed:
            converter = active[index]
            parts = [part]
            for processor in converter.preprocessors():
                parts = processor.process(parts)
            language = resolve_language(converter.language, languages)
            for part in parts:
                if part:
                    words = converter.convert(part)
                    for word in words:
                        word.language = language
                    result.extend(words)
        return result
