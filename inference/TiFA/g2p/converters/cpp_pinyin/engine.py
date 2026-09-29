"""Faithful port of cpp-pinyin EngineImpl core G2P algorithm.

Contains the exact queryRaw() sliding-window phrase-disambiguation logic,
including all four search directions per length.
"""


from pathlib import Path

from .dict_loader import (
    encode_phrase_key,
    load_word_dict,
    load_phrase_dict,
    load_phrase_map,
    load_trans_dict,
    load_user_dict,
)
from .tones import apply_tone, STYLE_TONE3


class PinyinEngine:
    """Port of cpp-pinyin EngineImpl.

    Loads the original colon-separated dictionary format and runs the
    identical queryRaw() sliding-window disambiguation algorithm.
    """

    def __init__(self, dict_dir: str | Path | None = None) -> None:
        if dict_dir is None:
            dict_dir = Path(__file__).parent / "dicts" / "mandarin"
        self._dict_dir = Path(dict_dir)
        self._words: dict[str, list[list[str]]] = {}
        self._phrases: dict[int, list[list[str]]] = {}
        self._polyphonic: set[str] = set()
        self._trans: dict[str, str] = {}
        # word.txt -> char -> [[pron1], [pron2], ...]
        self._words: dict[str, list[list[str]]] = {}
        # phrases_dict.txt + user_dict.txt -> int_key -> [[p1,p2,...], ...]
        self._phrases: dict[int, list[list[str]]] = {}
        # phrases_map.txt -> set of polyphonic chars
        self._polyphonic: set[str] = set()
        # trans_word.txt -> trad -> simp
        self._trans: dict[str, str] = {}

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        d = self._dict_dir
        self._words = load_word_dict(d / "word.txt")
        self._phrases = load_phrase_dict(d / "phrases_dict.txt")
        usr = load_user_dict(d / "user_dict.txt")
        for k, v in usr.items():
            self._phrases.setdefault(k, []).extend(v)
        self._polyphonic = load_phrase_map(d / "phrases_map.txt")
        self._trans = load_trans_dict(d / "trans_word.txt")

    def is_ready(self) -> bool:
        return bool(self._words)

    # ------------------------------------------------------------------
    # Core: faithful port of EngineImpl::queryRaw()
    # ------------------------------------------------------------------

    def query_raw(self, chars: list[str], *,
                  style: int = STYLE_TONE3,
                  v_to_u: bool = False,
                  neutral_tone5: bool = True) -> list[list[str]]:
        """Port of EngineImpl::queryRaw().

        Args:
            chars: List of CJK character strings (may include non-CJK).
            style: Output tone style (default STYLE_TONE3).
            v_to_u: Whether to convert 'v' back to 'ü'.
            neutral_tone5: Whether to append '5' for neutral tone.

        Returns:
            list[list[str]]  --  one phoneme list per input char, in order.
        """
        result: list[list[str]] = []
        cursor = 0

        while cursor < len(chars):
            ch = chars[cursor]

            # ---- Not in dictionary (passthrough) ----
            candidates = self._words.get(ch)
            if not candidates:
                result.append([ch])
                cursor += 1
                continue

            # ---- Not polyphonic  --  fast path (first pronunciation) ----
            if ch not in self._polyphonic:
                pron = apply_tone(
                    candidates[0][0], style,
                    v_to_u=v_to_u, neutral_tone5=neutral_tone5,
                )
                result.append([pron])
                cursor += 1
                continue

            # ---- Polyphonic  --  sliding-window phrase matching ----
            found = False

            # Closure matching cpp-pinyin's emitPhrase lambda
            def _emit(key_str: str, result_idx: int,
                      advance: int, pop_last: bool) -> bool:
                nonlocal found, cursor
                key = encode_phrase_key(list(key_str))
                it = self._phrases.get(key)
                if it is None:
                    return False
                phrase_prons = [
                    apply_tone(p, style, v_to_u=v_to_u, neutral_tone5=neutral_tone5)
                    for p in it[0]
                ]
                if pop_last and result:
                    result.pop()
                for i, p in enumerate(phrase_prons):
                    idx = result_idx + i
                    if idx < len(result):
                        result[idx] = [p]
                    else:
                        result.append([p])
                cursor += advance
                found = True
                return True

            for length in range(4, 1, -1):
                if found:
                    break

                # Direction 1: Forward  chars[cursor..cursor+length)
                if cursor + length <= len(chars):
                    _emit(
                        "".join(chars[cursor:cursor + length]),
                        len(result), length, False,
                    )

                # Direction 2: 1-backward  chars[cursor-1..cursor-1+length)
                if not found and cursor >= 1:
                    _emit(
                        "".join(chars[cursor - 1:cursor - 1 + length]),
                        len(result) - 1, length - 1, True,
                    )

                # Direction 3: 1-forward  chars[cursor+1-length..cursor+1)
                if not found and cursor + 1 >= length and cursor + 1 <= len(chars):
                    back_start = cursor + 1 - length
                    _emit(
                        "".join(chars[back_start:back_start + length]),
                        back_start, 1, False,
                    )

                # Direction 4: 2-forward  chars[cursor+2-length..cursor+2)
                if not found and cursor + 2 >= length and cursor + 2 <= len(chars):
                    back_start = cursor + 2 - length
                    _emit(
                        "".join(chars[back_start:back_start + length]),
                        back_start, 2, False,
                    )

            # ---- No phrase found  --  default pronunciation ----
            if not found:
                default = apply_tone(
                    candidates[0][0], style,
                    v_to_u=v_to_u, neutral_tone5=neutral_tone5,
                )
                result.append([default])
                cursor += 1

        return result

    # ------------------------------------------------------------------
    # Helpers mirroring cpp-pinyin's EngineImpl
    # ------------------------------------------------------------------

    def _to_simplified(self, ch: str) -> str:
        """Simplify a single character (trad -> simp)."""
        return self._trans.get(ch, ch)

    def simplify(self, chars: list[str]) -> list[str]:
        """Apply traditional->simplified conversion to every char."""
        return [self._trans.get(ch, ch) for ch in chars]

    def has_polyphonic(self, ch: str) -> bool:
        return ch in self._polyphonic

    def readings(self, ch: str, *, style: int = STYLE_TONE3,
                 v_to_u: bool = False,
                 neutral_tone5: bool = True) -> list[str]:
        """All possible readings for a character, tone-converted."""
        candidates = self._words.get(self._to_simplified(ch))
        if not candidates:
            return [ch]
        return [
            apply_tone(c[0], style, v_to_u=v_to_u, neutral_tone5=neutral_tone5)
            for c in candidates
        ]

    def default_reading(self, ch: str, *, style: int = STYLE_TONE3,
                        v_to_u: bool = False,
                        neutral_tone5: bool = True) -> str:
        """First (default) reading for a character."""
        candidates = self._words.get(self._to_simplified(ch))
        if not candidates:
            return ch
        return apply_tone(
            candidates[0][0], style,
            v_to_u=v_to_u, neutral_tone5=neutral_tone5,
        )
