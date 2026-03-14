import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Type

from .ZhG2p import ZhG2p, split_string as zh_split_string


class LanguageProcessor(ABC):
    def __init__(self, language_code: str, allowed_chars: str) -> None:
        self.language_code = language_code.lower()
        self._allowed_chars = allowed_chars

    def clean_text(self, text: str) -> str:
        cleaned = re.sub(rf'[^{self._allowed_chars}]', '', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def get_phonetic_list(self, text_list: List[str]) -> List[str]:
        pass


class ChineseProcessor(LanguageProcessor):
    _CHINESE_CHAR_RANGE: str = r'[\u4e00-\u9fa5]'

    def __init__(self) -> None:
        super().__init__('zh', self._CHINESE_CHAR_RANGE)
        self.g2p: ZhG2p = ZhG2p('mandarin')

    def split_text(self, text: str) -> List[str]:
        return zh_split_string(text)

    def get_phonetic_list(self, text_list: List[str]) -> List[str]:
        return self.g2p.convert_list(text_list).split(' ')


class EnglishProcessor(LanguageProcessor):
    _ALLOWED_CHARS: str = r'a-zA-Z0-9\s,.!?;:"\'-'

    def __init__(self) -> None:
        super().__init__('en', self._ALLOWED_CHARS)

    def split_text(self, text: str) -> List[str]:
        return zh_split_string(text.lower())

    def get_phonetic_list(self, text_list: List[str]) -> List[str]:
        return text_list


@dataclass(frozen=True)
class LyricData:
    text_list: List[str]
    phonetic_list: List[str]
    raw_text: str


class ProcessorFactory:
    _PROCESSOR_MAP: Dict[str, Type[LanguageProcessor]] = {
        'zh': ChineseProcessor,
        'en': EnglishProcessor,
    }

    @classmethod
    def create_processor(cls, language_code: str) -> LanguageProcessor:
        code: str = language_code.lower()
        if code not in cls._PROCESSOR_MAP:
            raise ValueError(f"Unsupported language: {language_code}")
        return cls._PROCESSOR_MAP[code]()

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        return list(cls._PROCESSOR_MAP.keys())
