"""Mandarin and Cantonese G2P converters  --  delegate to the cpp-pinyin engine.

Both derive from ``PronunciationScriptDictionaryConverter``: hanzi -> pinyin/jyutping
(text-to-script, via the cpp-pinyin engine) then pinyin/jyutping -> phonemes
(script-to-phonemes, via dictionary lookup when *dict_path* is given).
"""


from pathlib import Path

from .cpp_pinyin import PinyinEngine
from .cpp_pinyin.constants import STYLE_NORMAL
from ..registry import converter
from .dictionary import PronunciationScriptDictionaryConverter
from .text import find_run

_CPP_PINYIN_DIR = Path(__file__).parent / "cpp_pinyin" / "dicts"


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
