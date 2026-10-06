"""Mandarin and Cantonese G2P converters  --  delegate to the cpp-pinyin engine.

Both derive from ``PronunciationScriptDictionaryConverter``: hanzi -> pinyin/jyutping
(text-to-script, via the cpp-pinyin engine) then pinyin/jyutping -> phonemes
(script-to-phonemes, via dictionary lookup when *dict_path* is given).
"""


from pathlib import Path

from .cpp_pinyin import PinyinEngine
from .cpp_pinyin.constants import STYLE_NORMAL
from ..registry import converter
from .base import G2PGroup, G2PReading, G2PWord
from .dictionary import DictionaryConverter, PronunciationScriptDictionaryConverter
from .text import find_run, word_spans, split_words

_CPP_PINYIN_DIR = Path(__file__).parent / "cpp_pinyin" / "dicts"

_TONE_DIGITS = "012345"

# Non-standard explicit-glide spellings -> the vowel rest of the standard
# syllable. Keys are never standard syllables themselves. Contracted finals
# expand too: "yiu" (y + iu, iu = iou) folds to "you", "wun" (w + un,
# un = uen) to "wen".
_Y_GLYPH_FORMS = {"ia": "a", "iao": "ao", "ie": "e", "iu": "ou", "iang": "ang", "iong": "ong"}
_W_GLYPH_FORMS = {"ua": "a", "uai": "ai", "uan": "an", "uang": "ang", "uo": "o", "ui": "ei", "un": "en"}


def _fold_explicit_glide(syllable: str) -> str:
    """Fold a non-standard explicit-glide spelling to the standard syllable.

    Direct pinyin ASR occasionally spells the y/w glide out with its medial
    ("yiang" for yang, "yiu" for you, "wuang" for wang); standard pinyin
    writes those syllables with the glide fused into a single vowel letter.
    """
    for glide, forms in (("y", _Y_GLYPH_FORMS), ("w", _W_GLYPH_FORMS)):
        if syllable.startswith(glide):
            rest = forms.get(syllable[1:])
            if rest is not None:
                return glide + rest
    return syllable


class _ChineseScriptConverter(PronunciationScriptDictionaryConverter):
    """Shared base for cpp-pinyin-backed Chinese converters.

    Subclasses are decorated with ``@converter`` to register language and id.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        _bundled_dict: str,
    ) -> None:
        super().__init__(dict_path=dict_path)
        self._engine = PinyinEngine(_bundled_dict)

    @staticmethod
    def _is_hanzi(token: str) -> bool:
        if len(token) != 1:
            return False
        return 0x4E00 <= ord(token) <= 0x9FA5

    def find(self, text: str) -> tuple[int, int] | None:
        return find_run(text, self._is_hanzi)

    def text_to_scripts(self, words: list[str]) -> list[list[str]]:
        simplified = self._engine.simplify(words)
        best = self._engine.query_raw(simplified, style=STYLE_NORMAL)
        result: list[list[str]] = []
        for ch, best_list in zip(simplified, best):
            primary = best_list[0]
            scripts = [primary]
            for reading in self._engine.readings(ch, style=STYLE_NORMAL):
                if reading != primary:
                    scripts.append(reading)
            supported = [script for script in scripts if script in self._script_dict]
            # Preserve the lookup error when none of the readings is supported.
            result.append(supported or scripts)
        return result


@converter(id="chinese-pinyin", language="zh,zho,cmn")
class PinyinConverter(_ChineseScriptConverter):
    """Mandarin Chinese pinyin converter."""

    def __init__(self, dict_path: str) -> None:
        super().__init__(
            dict_path=dict_path,
            _bundled_dict=str(_CPP_PINYIN_DIR / "mandarin"),
        )


@converter(id="yue-jyutping", language="yue")
class JyutpingConverter(_ChineseScriptConverter):
    """Yue (Jyutping) converter."""

    def __init__(self, dict_path: str) -> None:
        super().__init__(
            dict_path=dict_path,
            _bundled_dict=str(_CPP_PINYIN_DIR / "cantonese"),
        )


@converter(id="pinyin-lax", language="zh,zho,cmn")
class PinyinLaxConverter(DictionaryConverter):
    """Tolerant pinyin fallback for direct-ASR spellings the dictionary misses.

    Claims a token only when some standard spelling of it is in the
    pronunciation dictionary: the exact form, the tone digits stripped, or
    the explicit-glide fold. Anything else stays unclaimed so the chain's
    passthrough converter reports it as unpronounceable.
    """

    def _dict_key(self, token: str) -> str | None:
        token = token.lower()
        candidates = [token]
        stripped = token.rstrip(_TONE_DIGITS)
        if stripped and stripped != token:
            candidates.append(stripped)
        folded = _fold_explicit_glide(stripped)
        if folded != stripped and folded not in candidates:
            candidates.append(folded)
        for candidate in candidates:
            if candidate in self._dict:
                return candidate
        return None

    def find(self, text: str) -> tuple[int, int] | None:
        for begin, end in word_spans(text):
            if self._dict_key(text[begin:end]) is not None:
                return begin, end
        return None

    def convert(self, text: str) -> list[G2PWord]:
        result: list[G2PWord] = []
        for token in split_words(text):
            key = self._dict_key(token)
            if key is None:
                raise KeyError(
                    f"PinyinLaxConverter: token '{token}' not pronounceable. "
                    f"find should have filtered it."
                )
            paths = [
                [G2PGroup(script=key, phonemes=list(p))] if p else []
                for p in self._dict[key]
            ]
            result.append(G2PWord(text=token, readings=[G2PReading(paths=paths)]))
        return result
