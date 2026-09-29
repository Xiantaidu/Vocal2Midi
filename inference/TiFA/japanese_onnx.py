"""ja_g2p_onnx converter for TiFA's Japanese chain.

Claims kanji-bearing runs and reads them with the LAKE-G2P v5 ONNX model
(``models/ja_g2p_onnx``), whose transformer disambiguates polyphones in
sentence context (今日は -> きょうは) and applies number rules. The model
emits one kana reading per word edge; each edge becomes a G2PWord whose
phonemes come from the standard two-phase conversion (kana -> romaji ->
DiffSinger Japanese dictionary), so both alignment engines consume the
same phoneme vocabulary. Group scripts carry kana characters for the
per-mora romaji/kana display tokens.

Long-vowel marks (ー) in katakana readings are expanded to the preceding
mora's vowel (タワー -> ta wa a) before dictionary lookup, matching the
HFA side's mora semantics.
"""
from __future__ import annotations

import logging

from inference.ja_onnx_g2p import get_ja_onnx_runtime, normalize_edge_reading
from inference.TiFA.g2p.converters.base import Converter, G2PReading, G2PWord
from inference.TiFA.g2p.converters.japanese import (
    _KANA_TO_ROMAJI,
    JapaneseKanaConverter,
    _kata_to_hira,
)
from inference.TiFA.g2p.converters.text import split_words
from inference.TiFA.g2p.registry import converter
from inference.TiFA.japanese_lexicon import _KANJI_RE, kana_reading_to_path

logger = logging.getLogger(__name__)

_VOWEL_KANA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}


def _is_claim_char(ch: str) -> bool:
    """Characters a claimed run may span: kanji, kana, long mark, digits."""
    return bool(
        _KANJI_RE.match(ch)
        or ch == "ー"
        or ch.isdigit()
        or (0x3041 <= ord(ch) <= 0x30FF)
    )


def _expand_long_vowel_moras(moras: list[str]) -> list[str]:
    """Replace ー moras with the preceding mora's vowel kana (タワー -> タワあ).

    っ/ん and a leading ー cannot be extended; those marks are dropped.
    """
    out: list[str] = []
    last_vowel: str | None = None
    for mora in moras:
        if mora == "ー":
            if last_vowel:
                out.append(_VOWEL_KANA[last_vowel])
            continue
        out.append(mora)
        romaji = _KANA_TO_ROMAJI.get(_kata_to_hira(mora))
        if romaji and romaji[-1] in "aiueo":
            last_vowel = romaji[-1]
        else:
            last_vowel = None
    return out


class JapaneseOnnxConverter(Converter):
    """Claims kanji-bearing runs and reads them with ja_g2p_onnx."""

    def __init__(self, dict_path: str, model_dir: str | None = None):
        self._kana = JapaneseKanaConverter(dict_path=dict_path, double_written_sokuon=False)
        self._model_dir = model_dir

    def find(self, text: str) -> tuple[int, int] | None:
        match = _KANJI_RE.search(text)
        if match is None:
            return None
        start = match.start()
        end = start + 1
        while end < len(text) and _is_claim_char(text[end]):
            end += 1
        return start, end

    def convert(self, text: str) -> list[G2PWord]:
        runtime = get_ja_onnx_runtime(self._model_dir)
        if runtime is None:
            logger.warning("[JaG2P-ONNX] runtime unavailable; span dropped")
            return []
        try:
            predicted = runtime.predict(text)
        except Exception as e:
            logger.warning(f"[JaG2P-ONNX] predict failed for {text!r}: {e}; span dropped")
            return []

        # The runtime may split one katakana word into per-character edges
        # (ラーメン -> ラ/ー/メ/ン), so long-vowel marks are expanded over the
        # whole-line reading and re-aligned to edges by mora count.
        moras = _expand_long_vowel_moras(split_words(predicted["reading"]))
        words: list[G2PWord] = []
        index = 0
        for edge in predicted["edges"]:
            surface, reading = edge["surface"], edge["reading"]
            if not reading:
                continue
            reading = normalize_edge_reading(surface, reading)
            count = len(split_words(reading))
            chunk_moras = moras[index : index + count]
            index += count
            path = kana_reading_to_path(self._kana, "".join(chunk_moras))
            if path is None:
                logger.warning(
                    f"[JaG2P-ONNX] reading {reading!r} for {surface!r} not convertible; word dropped"
                )
                continue
            words.append(G2PWord(text=surface, readings=[G2PReading(paths=[path])]))
        return words


@converter(id="japanese-onnx", language="ja")
class RegisteredJapaneseOnnxConverter(JapaneseOnnxConverter):
    """Registry-visible subclass so the factory can build it by id."""
