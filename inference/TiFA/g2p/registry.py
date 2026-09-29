from enum import Enum, auto
from typing import Callable


class Language(Enum):
    """Special converter language markers."""

    ANY = auto()


_preprocessor_registry: dict[str, type] = {}
_converter_registry: dict[str, type] = {}


def parse_language(language: str | Language | None) -> tuple[str, ...] | Language | None:
    if language is None or language is Language.ANY:
        return language
    return tuple(t.strip() for t in language.split(","))


def preprocessor(*, id: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if id in _preprocessor_registry:
            raise ValueError(f"Preprocessor '{id}' is already registered.")
        _preprocessor_registry[id] = cls
        return cls
    return decorator


def converter(*, id: str, language: str | Language | None = None) -> Callable[[type], type]:
    tags = parse_language(language)

    def decorator(cls: type) -> type:
        if id in _converter_registry:
            raise ValueError(f"Converter '{id}' is already registered.")
        _converter_registry[id] = cls
        cls.language = tags
        return cls
    return decorator


def get_preprocessor(id: str) -> type:
    return _preprocessor_registry[id]


def get_converter(id: str) -> type:
    return _converter_registry[id]


def list_preprocessors() -> list[str]:
    return list(_preprocessor_registry.keys())


def list_converters() -> list[str]:
    return list(_converter_registry.keys())
