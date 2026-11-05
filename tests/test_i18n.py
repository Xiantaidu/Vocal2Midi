import string

import pytest

from gui import i18n


@pytest.fixture(autouse=True)
def restore_language():
    previous = i18n.current_language
    yield
    i18n.set_language(previous)


def _placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_key_has_all_languages():
    for key, entry in i18n._STRINGS.items():
        for language in ("zh", "en", "ja"):
            assert entry.get(language), f"{key} is missing {language}"


def test_translations_keep_the_same_placeholders():
    for key, entry in i18n._STRINGS.items():
        expected = _placeholders(entry["zh"])
        for language in ("en", "ja"):
            assert _placeholders(entry[language]) == expected, f"{key} ({language})"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ja", "ja"),
        ("Japanese", "ja"),
        (" EN ", "en"),
        ("english", "en"),
        ("zh", "zh"),
        ("", "zh"),
        (None, "zh"),
        ("fr", "zh"),
    ],
)
def test_normalize_language(value, expected):
    assert i18n.normalize_language(value) == expected


def test_set_language_switches_tr_output():
    i18n.set_language("ja")
    assert i18n.tr("menu_file") == "ファイル"
    assert i18n.tr("files_added", n=3) == "3 個の音声ファイルを追加しました"
    i18n.set_language("en")
    assert i18n.tr("menu_file") == "File"
