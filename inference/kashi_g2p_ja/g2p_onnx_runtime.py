"""Standalone ONNX runtime for the LAKE-G2P v4 span model (DirectML/CPU).

Self-contained: needs only `onnxruntime-directml` (falls back to CPU) and
numpy. Reproduces the production inference path byte-for-byte:

  1. candidate graph  = unified lexicon pack (alnum + number rules + prefix
     scan) + COPY, then the CompositeProvider chain (rendaku variants,
     dedup, per-span cap 16 / total cap 256, okurigana / rendaku filters,
     path repair, kun-stem derivation)
  2. features         = BERT char ids, IDS components, char types, per-edge
     reading tokens, source ids, priors, prior-pack ids
  3. scoring          = model.onnx (edge scores; L=64/R=64 static, edges dynamic)
  4. decoding         = locked-edge filter + morphosyntactic penalties +
     exact semi-Markov Viterbi (0th order), windowed for long lines

CLI (mirrors scripts/transcribe_txt.py):
    python g2p_onnx_runtime.py -i input.txt -o output.json [--cpu]
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install onnxruntime-directml numpy") from exc

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "kashi-g2p-onnx"
RESOURCES = DEFAULT_MODEL_DIR / "resources" if (DEFAULT_MODEL_DIR / "resources").exists() else (HERE / "resources")
SEQ_LEN = 64
READING_LEN = 64
MAX_COMPONENTS = 8
MAX_EDGES = 256
MAX_SPAN_READINGS = 16
MAX_SURFACE_LEN = 16  # unified pack prefix-scan limit

# ---------------------------------------------------------------- normalization

KANJI_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002fa1f々〆ヶ]"
)


def normalize_surface(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("|", "")


def hira(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text
    )


def is_kanji(ch: str) -> bool:
    return bool(ch and KANJI_RE.fullmatch(ch))


def _char_type(char: str) -> int:
    if is_kanji(char):
        return 1
    code = ord(char)
    if 0x3040 <= code <= 0x309F:
        return 2
    if 0x30A0 <= code <= 0x30FF:
        return 3
    if char.isascii() and char.isalpha():
        return 4
    if char.isdigit():
        return 5
    return 6


# ---------------------------------------------------------------- edges


class Edge:
    __slots__ = ("start", "end", "surface", "reading", "source",
                 "prior", "confidence", "locked")

    def __init__(self, start, end, surface, reading, source="unknown",
                 prior=0.0, confidence=0.0, locked=False):
        self.start = start
        self.end = end
        self.surface = surface
        self.reading = reading
        self.source = source
        self.prior = prior
        self.confidence = confidence
        self.locked = locked

    def key(self):
        return (self.start, self.end, self.reading,
                "copy" if self.source == "copy" else "")


SOURCE_TO_ID = {
    "rules": 0, "haqumei": 1, "lyric_memory": 2, "kanjidic": 3,
    "pyopenjtalk": 4, "gold": 5, "copy": 6, "unknown": 7, "jmdict": 8,
    "unidic": 9, "generated": 10, "alnum": 11, "yomogi_dict": 12,
}
_PRIORITY_ORDER = (
    "gold", "lyric_memory", "alnum", "rules", "haqumei", "pyopenjtalk",
    "jmdict", "unidic", "yomogi_dict", "kanjidic", "copy")
SOURCE_PRIORITY = {
    name: len(_PRIORITY_ORDER) - index
    for index, name in enumerate(_PRIORITY_ORDER)
}

_DAKUTEN = dict(zip("かきくけこさしすせそたちつてとはひふへほう",
                    "がぎぐげござじずぜぞだぢづでどばびぶべぼゔ"))
_HANDAKU = dict(zip("はひふへほ", "ぱぴぷぺぽ"))
_WORD_SOURCES = frozenset({"jmdict", "unidic", "yomogi_dict", "pyopenjtalk"})
_PUNCT_OR_SPACE = frozenset(" \t\r\n、。！？!?…・~～-—")


# ---------------------------------------------------------------- number rules


class NumberRules:
    _COUNTER_RE = re.compile(
        r"(?P<number>\d+)(?P<counter>時間|人|日|秒|分|回|歩|歳|才|年|月|個|本|枚|台|番|曲|時|目|匹|通|件|点|度|階|杯|冊|倍|段|つ)")
    _KANJI_NUM_RE = re.compile(
        r"(?P<number>[一二三四五六七八九十百千]+)(?P<counter>時間|日|秒|分|回|歳|年|月|個|本|枚|台|番|曲|時|匹|件|点|度|階|杯|冊|倍|段|つ)")
    _KANJI_DIGIT_VALUE = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                          "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000}
    _NUMBER_RE = re.compile(r"\d+")
    _WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    _DIGITS = ("ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう")
    _DIGIT_READINGS = {0: ("ぜろ", "れい"), 1: ("いち", "わん"), 2: ("に", "つー"),
                       3: ("さん", "すりー"), 4: ("よん", "し", "ふぉー"),
                       5: ("ご", "ふぁいぶ"), 6: ("ろく", "しっくす"),
                       7: ("なな", "しち", "せぶん"), 8: ("はち", "えいと"),
                       9: ("きゅう", "く", "ないん"), 10: ("じゅう", "てん")}
    _LETTER_NAMES = {"a": "えー", "b": "びー", "c": "しー", "d": "でぃー",
                     "e": "いー", "f": "えふ", "g": "じー", "h": "えいち",
                     "i": "あい", "j": "じぇー", "k": "けー", "l": "える",
                     "m": "えむ", "n": "えぬ", "o": "おー", "p": "ぴー",
                     "q": "きゅー", "r": "あーる", "s": "えす", "t": "てぃー",
                     "u": "ゆー", "v": "ぶい", "w": "だぶりゅー", "x": "えっくす",
                     "y": "わい", "z": "ぜっと"}
    _WORDS = {"a": "えー", "an": "あん", "are": "あー", "baby": "べいびー",
              "bye": "ばい", "come": "かむ", "day": "でい", "good": "ぐっど",
              "hello": "はろー", "how": "はう", "i": "あい", "just": "じゃすと",
              "like": "らいく", "love": "らぶ", "merry": "めりー", "me": "みー",
              "my": "まい", "only": "おんりー", "ready": "れでぃ", "sf": "えすえふ",
              "spot": "すぽっと", "the": "ざ", "to": "とぅ", "type": "たいぷ",
              "want": "うぉんと", "what": "ほわっと", "you": "ゆー",
              "your": "ゆあー", "you're": "ゆあー", "we": "うぃー",
              "with": "うぃず", "up": "あっぷ", "la": "ら", "yah": "やー",
              "yeah": "いぇー", "oh": "おー", "ah": "あー", "no": "のー",
              "go": "ごー", "all": "おーる", "one": "わん", "two": "つー",
              "three": "すりー", "four": "ふぉー", "five": "ふぁいぶ",
              "six": "しっくす", "seven": "せぶん", "eight": "えいと",
              "nine": "ないん", "ten": "てん", "be": "びー", "do": "どぅー",
              "so": "そー", "in": "いん", "on": "おん", "it": "いっと",
              "is": "いず", "of": "おぶ", "for": "ふぉー", "and": "あんど",
              "night": "ないと", "time": "たいむ", "music": "みゅーじっく",
              "song": "そんぐ", "heart": "はーと", "dream": "どりーむ",
              "star": "すたー", "world": "わーるど", "dance": "だんす",
              "party": "ぱーてぃー", "girl": "がーる", "boy": "ぼーい",
              "never": "ねばー", "always": "おーるうぇいず",
              "forever": "ふぉーえばー", "together": "とぅげざー",
              "tonight": "とぅないと", "stop": "すとっぷ",
              "joyful": "じょいふる", "again": "あげいん"}
    _SPECIAL_COUNTERS = {
        "人": {1: "ひとり", 2: "ふたり", 4: "よにん"},
        "日": {1: "ついたち", 2: "ふつか", 3: "みっか", 4: "よっか",
               5: "いつか", 6: "むいか", 7: "なのか", 8: "ようか",
               9: "ここのか", 10: "とおか", 14: "じゅうよっか",
               20: "はつか", 24: "にじゅうよっか"},
        "つ": {1: "ひとつ", 2: "ふたつ", 3: "みっつ", 4: "よっつ",
               5: "いつつ", 6: "むっつ", 7: "ななつ", 8: "やっつ",
               9: "ここのつ", 10: "とお"},
        "月": {4: "しがつ", 7: "しちがつ", 9: "くがつ"},
        "時": {4: "よじ", 9: "くじ", 14: "じゅうよじ", 24: "にじゅうよじ"},
        "時間": {4: "よじかん", 9: "くじかん", 14: "じゅうよじかん",
                 24: "にじゅうよじかん"},
        "年": {4: "よねん"},
    }
    _COUNTER_SUFFIX = {
        "秒": ("びょう", {}),
        "分": ("ふん", {1: "いっぷん", 3: "さんぷん", 4: "よんぷん",
                       6: "ろっぷん", 8: "はっぷん", 10: "じゅっぷん"}),
        "回": ("かい", {1: "いっかい", 6: "ろっかい", 8: "はっかい",
                       10: "じゅっかい"}),
        "歩": ("ほ", {1: "いっぽ", 3: "さんぽ", 6: "ろっぽ", 8: "はっぽ",
                     10: "じゅっぽ"}),
        "日": ("か", {1: "いちにち", 2: "ふつか", 3: "さんか", 4: "よんか",
                     5: "ごか", 6: "ろっか", 7: "しちか", 8: "はちか",
                     9: "きゅうか", 10: "じゅっか"}),
        "歳": ("さい", {1: "いっさい", 8: "はっさい", 10: "じゅっさい",
                       20: "はたち"}),
        "才": ("さい", {1: "いっさい", 8: "はっさい", 10: "じゅっさい",
                       20: "はたち"}),
        "人": ("にん", {1: "ひとり", 2: "ふたり", 4: "よにん"}),
        "年": ("ねん", {4: "よねん"}),
        "月": ("がつ", {4: "しがつ", 7: "しちがつ", 9: "くがつ"}),
        "個": ("こ", {1: "いっこ", 6: "ろっこ", 8: "はっこ", 10: "じゅっこ"}),
        "本": ("ほん", {1: "いっぽん", 2: "にほん", 3: "さんぼん",
                       6: "ろっぽん", 8: "はっぽん", 10: "じゅっぽん"}),
        "枚": ("まい", {}), "台": ("だい", {}), "番": ("ばん", {}),
        "曲": ("きょく", {1: "いっきょく", 6: "ろっきょく", 8: "はっきょく",
                         10: "じゅっきょく"}),
        "時": ("じ", {4: "よじ", 9: "くじ", 14: "じゅうよじ",
                     24: "にじゅうよじ"}),
        "時間": ("じかん", {4: "よじかん", 9: "くじかん",
                           14: "じゅうよじかん", 24: "にじゅうよじかん"}),
        "目": ("め", {}),
        "匹": ("ひき", {1: "いっぴき", 3: "さんびき", 6: "ろっぴき",
                       8: "はっぴき", 10: "じゅっぴき"}),
        "通": ("つう", {1: "いっつう", 8: "はっつう", 10: "じゅっつう"}),
        "件": ("けん", {1: "いっけん", 3: "さんげん", 6: "ろっけん",
                       8: "はっけん", 10: "じゅっけん"}),
        "点": ("てん", {1: "いってん", 10: "じゅってん"}),
        "度": ("ど", {1: "いちど", 2: "にど", 3: "さんど"}),
        "階": ("かい", {1: "いっかい", 3: "さんがい", 6: "ろっかい",
                       8: "はっかい", 10: "じゅっかい"}),
        "杯": ("はい", {1: "いっぱい", 2: "にはい", 3: "さんばい",
                       6: "ろっぱい", 8: "はっぱい", 10: "じゅっぱい"}),
        "冊": ("さつ", {1: "いっさつ", 8: "はっさつ", 10: "じゅっさつ"}),
        "倍": ("ばい", {}), "段": ("だん", {}), "つ": ("", {}),
    }

    @classmethod
    def _under_1000(cls, value: int) -> str:
        result = ""
        hundreds, value = divmod(value, 100)
        if hundreds:
            result += {1: "ひゃく", 2: "にひゃく", 3: "さんびゃく",
                       4: "よんひゃく", 5: "ごひゃく", 6: "ろっぴゃく",
                       7: "ななひゃく", 8: "はっぴゃく",
                       9: "きゅうひゃく"}[hundreds]
        tens, ones = divmod(value, 10)
        if tens:
            result += "じゅう" if tens == 1 else cls._DIGITS[tens] + "じゅう"
        if ones:
            result += cls._DIGITS[ones]
        return result or cls._DIGITS[0]

    @classmethod
    def _number(cls, value: int) -> str:
        if value < 0:
            raise ValueError("number must be non-negative")
        if value == 0:
            return cls._DIGITS[0]
        if value < 1000:
            return cls._under_1000(value)
        if value < 10000:
            thousands, rest = divmod(value, 1000)
            prefix = {1: "せん", 2: "にせん", 3: "さんぜん", 4: "よんせん",
                      5: "ごせん", 6: "ろくせん", 7: "ななせん", 8: "はっせん",
                      9: "きゅうせん"}[thousands]
            return prefix + (cls._under_1000(rest) if rest else "")
        for divisor, name in ((10**12, "ちょう"), (10**8, "おく"),
                              (10**4, "まん")):
            if value >= divisor:
                high, low = divmod(value, divisor)
                return cls._number(high) + name + (
                    cls._number(low) if low else "")
        raise ValueError("number is outside Japanese cardinal range")

    @classmethod
    def _number_readings(cls, value: int) -> list[str]:
        readings = [cls._number(value)]
        for alt in cls._DIGIT_READINGS.get(value, ()):
            if alt not in readings:
                readings.append(alt)
        return readings

    @classmethod
    def _number_counter(cls, value: int, counter: str) -> str:
        special = cls._SPECIAL_COUNTERS.get(counter, {}).get(value)
        if special:
            return special
        suffix, values = cls._COUNTER_SUFFIX.get(counter, (counter, {}))
        if value in values:
            return values[value]
        return cls._number(value) + suffix

    @classmethod
    def _word_reading(cls, word: str) -> tuple[str, float]:
        lowered = word.lower()
        if lowered in cls._WORDS:
            return hira(cls._WORDS[lowered]), 0.99
        if (word.isupper() and len(word) <= 4
                and all(ch in cls._LETTER_NAMES for ch in lowered)):
            return "".join(cls._LETTER_NAMES[ch] for ch in lowered), 0.93
        return "", 0.0

    @classmethod
    def build(cls, text: str) -> list[Edge]:
        result: list[Edge] = []
        consumed: set[int] = set()
        for match in cls._COUNTER_RE.finditer(text):
            value = int(match.group("number"))
            reading = cls._number_counter(value, match.group("counter"))
            start, end = match.span()
            if reading and not any(is_kanji(ch) for ch in reading):
                result.append(Edge(start, end, text[start:end], reading,
                                   "rules", prior=2.5, confidence=1.0))
            consumed.update(range(start, end))
        for match in cls._KANJI_NUM_RE.finditer(text):
            value = cls._KANJI_DIGIT_VALUE.get(match.group("number"))
            if value is None or value > 10:
                continue
            counter = match.group("counter")
            start, end = match.span()
            rule_reading = cls._number_counter(value, counter)
            readings = []
            for reading in (rule_reading,
                            cls._number(value) + cls._COUNTER_SUFFIX[counter][0]):
                if (reading and reading not in readings
                        and not any(is_kanji(ch) for ch in reading)):
                    readings.append(reading)
            for reading in readings:
                result.append(Edge(start, end, text[start:end], reading,
                                   "rules", prior=1.5, confidence=0.9))
            consumed.update(range(start, end))
        for match in cls._NUMBER_RE.finditer(text):
            if any(index in consumed for index in range(*match.span())):
                continue
            start, end = match.span()
            for reading in cls._number_readings(int(match.group())):
                result.append(Edge(start, end, text[start:end], reading,
                                   "rules", prior=2.0, confidence=1.0))
        for match in cls._WORD_RE.finditer(text):
            start, end = match.span()
            reading, confidence = cls._word_reading(match.group())
            if reading:
                result.append(Edge(start, end, text[start:end], reading,
                                   "rules",
                                   prior=1.8 if confidence > 0.95 else 0.5,
                                   confidence=confidence))
        return result


# ---------------------------------------------------------------- lexicon pack


class UnifiedLexicon:
    """Port of UnifiedLexiconProvider (pack + rules + prefix scan)."""

    def __init__(self, table: dict, *, use_rules: bool = True, use_copy: bool = False):
        self.table = {k: tuple(v) for k, v in table.items()}
        self._lower_table: dict[str, tuple] = {}
        for k, v in self.table.items():
            low = k.lower()
            if low not in self._lower_table:
                self._lower_table[low] = v
        self.use_rules = use_rules
        self.use_copy = use_copy

    @staticmethod
    def _entries(item) -> list[tuple[str, str, float, float]]:
        r, src, p, c = item if isinstance(item, tuple) and len(item) == 4 \
            else (item, "alnum", 1.2, 0.9)
        return [(r, src, p, c)]

    def build(self, text: str) -> list[Edge]:
        text = normalize_surface(text)
        n = len(text)
        edges: list[Edge] = []
        seen: set[tuple[int, int, str]] = set()

        for match in re.finditer(r"[A-Za-z0-9]+", text):
            span_start, span_end = match.span()
            word = match.group()
            entries = self.table.get(word) or self._lower_table.get(word.lower())
            if entries:
                for item in entries:
                    for r, src, p, c in self._entries(item):
                        if (span_start, span_end, r) not in seen:
                            seen.add((span_start, span_end, r))
                            edges.append(Edge(span_start, span_end, word, r,
                                              src, p, c))
            else:
                for k in range(len(word) - 1, 1, -1):
                    prefix = word[:k]
                    sub_entries = (self.table.get(prefix)
                                   or self._lower_table.get(prefix.lower()))
                    if sub_entries:
                        for item in sub_entries:
                            for r, src, p, c in self._entries(item):
                                if (span_start, span_start + k, r) not in seen:
                                    seen.add((span_start, span_start + k, r))
                                    edges.append(Edge(span_start, span_start + k,
                                                      prefix, r, src, p, c))
                        break

        if self.use_rules:
            for e in NumberRules.build(text):
                if (e.start, e.end, e.reading) not in seen:
                    seen.add((e.start, e.end, e.reading))
                    edges.append(e)

        for i in range(n):
            limit = min(MAX_SURFACE_LEN, n - i)
            for k in range(1, limit + 1):
                sub = text[i:i + k]
                if sub.isascii() and sub.isalnum():
                    continue
                entries = self.table.get(sub)
                if entries is not None:
                    for item in entries:
                        r, src, p, c = item if isinstance(item, tuple) and len(item) == 4 \
                            else (item, "jmdict", 0.85, 0.82)
                        if (i, i + k, r) not in seen:
                            seen.add((i, i + k, r))
                            edges.append(Edge(i, i + k, sub, r, src, p, c))

        if self.use_copy:
            edges.extend(copy_edges(text))
        return filter_truncated_okurigana(edges, text)


def copy_edges(text: str) -> list[Edge]:
    return [Edge(i, i + 1, ch, ch, "copy", prior=-1.0)
            for i, ch in enumerate(text)]


# ------------------------------------------------- composite provider chain


def _rank(edge: Edge) -> tuple[int, float, float]:
    return (SOURCE_PRIORITY.get(edge.source, -1), edge.confidence, edge.prior)


def expand_word_variants(edges: list[Edge], text: str) -> list[Edge]:
    out = list(edges)
    seen = {(e.start, e.end, e.reading) for e in edges}
    for edge in edges:
        if (edge.source not in _WORD_SOURCES or edge.locked
                or edge.end - edge.start < 2 or len(edge.reading) < 2):
            continue
        if edge.start == 0:
            continue
        if text and text[edge.start - 1] in _PUNCT_OR_SPACE:
            continue
        if edge.surface and edge.surface[0].isdigit():
            continue
        first = edge.reading[0]
        for mapped in (_DAKUTEN.get(first), _HANDAKU.get(first)):
            if not mapped:
                continue
            reading = mapped + edge.reading[1:]
            marker = (edge.start, edge.end, reading)
            if marker not in seen:
                seen.add(marker)
                out.append(Edge(edge.start, edge.end, edge.surface, reading,
                                edge.source, max(0.0, edge.prior - 0.4),
                                edge.confidence * 0.85))
    return out


def filter_truncated_okurigana(edges: list[Edge], text: str) -> list[Edge]:
    longer_spans: dict[tuple[int, str], int] = {}
    for e in edges:
        key = (e.start, e.reading)
        if key not in longer_spans or e.end > longer_spans[key]:
            longer_spans[key] = e.end
    filtered = []
    for e in edges:
        max_end = longer_spans.get((e.start, e.reading), e.end)
        if max_end > e.end:
            ext = text[e.end:max_end]
            if ext and all("\u3040" <= ch <= "\u30ff" for ch in ext):
                continue
        filtered.append(e)
    return filtered


def filter_rendaku_traps(edges: list[Edge]) -> list[Edge]:
    by_span: dict[tuple[int, int], list[int]] = {}
    for index, edge in enumerate(edges):
        by_span.setdefault((edge.start, edge.end), []).append(index)
    drop: set[int] = set()
    for indices in by_span.values():
        readings = {edges[i].reading for i in indices
                    if edges[i].source in _WORD_SOURCES}
        if len(readings) < 2:
            continue
        for i in indices:
            edge = edges[i]
            if edge.source not in _WORD_SOURCES or edge.locked:
                continue
            reading = edge.reading
            for k in range(1, len(reading)):
                voiced = _DAKUTEN.get(reading[k]) or _HANDAKU.get(reading[k])
                if voiced and reading[:k] + voiced + reading[k + 1:] in readings:
                    drop.add(i)
                    break
    if not drop:
        return edges
    return [edge for index, edge in enumerate(edges) if index not in drop]


def _has_complete_path(edges: list[Edge], length: int) -> bool:
    reachable = [False] * (length + 1)
    reachable[0] = True
    by_start: dict[int, list[int]] = {}
    for edge in edges:
        if 0 <= edge.start < edge.end <= length:
            by_start.setdefault(edge.start, []).append(edge.end)
    for start in range(length):
        if reachable[start]:
            for end in by_start.get(start, ()):
                reachable[end] = True
    return reachable[length]


def repair_complete_path(edges: list[Edge], text: str) -> list[Edge]:
    if _has_complete_path(edges, len(text)):
        return edges
    present = {(e.start, e.end, e.reading, e.source) for e in edges}
    for edge in copy_edges(text):
        key = (edge.start, edge.end, edge.reading, edge.source)
        if key not in present:
            edges.append(edge)
            present.add(key)
    if not _has_complete_path(edges, len(text)):
        raise ValueError("candidate graph has no complete 0..L path")
    return edges


def _hira_char(ch: str) -> bool:
    return "ぁ" <= ch <= "ゖ"


def derive_okurigana_stems(edges: list[Edge], text: str) -> list[Edge]:
    existing = {(e.start, e.end, e.reading) for e in edges}
    derived: list[Edge] = []
    for e in edges:
        surf = text[e.start:e.end]
        if len(surf) < 2:
            continue
        k = len(surf)
        while k > 0 and _hira_char(surf[k - 1]):
            k -= 1
        okuri = surf[k:]
        stem = surf[:k]
        if not okuri or not stem or not is_kanji(stem[0]):
            continue
        if any(_hira_char(ch) for ch in stem):
            continue
        if not e.reading.endswith(okuri):
            continue
        stem_reading = e.reading[:-len(okuri)]
        if not stem_reading:
            continue
        key = (e.start, e.start + k, stem_reading)
        if key in existing:
            continue
        existing.add(key)
        derived.append(Edge(e.start, e.start + k, stem, stem_reading,
                            e.source, e.prior, e.confidence, False))
    return edges + derived


def _prune_key(edge: Edge):
    priority, confidence, prior = _rank(edge)
    return (-priority, -confidence, -prior, edge.start, edge.end,
            edge.reading, edge.source)


def prune(edges: list[Edge], text: str) -> list[Edge]:
    mandatory = {(e.start, e.end, e.reading, e.source)
                 for e in copy_edges(text)}
    keep = list(edges)
    spans: dict[tuple[int, int], list[Edge]] = {}
    for edge in keep:
        spans.setdefault((edge.start, edge.end), []).append(edge)
    keep = []
    for span in sorted(spans):
        ranked = sorted(spans[span], key=_prune_key)
        copies = [e for e in ranked
                  if (e.start, e.end, e.reading, e.source) in mandatory]
        others = [e for e in ranked if e not in copies]
        selected = others[:MAX_SPAN_READINGS]
        reserve = max(1, MAX_SPAN_READINGS // 2)
        fallback = [e for e in others
                    if e.source == "kanjidic" and e not in selected]
        for edge in fallback:
            if sum(1 for e in selected if e.source == "kanjidic") >= reserve:
                break
            replaceable = [e for e in selected if e.source != "kanjidic"]
            if not replaceable:
                break
            selected[selected.index(replaceable[-1])] = edge
        keep.extend(copies)
        keep.extend(selected)
    copies = [e for e in keep
              if (e.start, e.end, e.reading, e.source) in mandatory]
    others = [e for e in keep if e not in copies]
    keep = copies + sorted(others, key=_prune_key)[:MAX_EDGES]
    copy_keys = {(e.start, e.end, e.reading, e.source)
                 for e in keep if e.source == "copy"}
    for edge in copy_edges(text):
        key = (edge.start, edge.end, edge.reading, edge.source)
        if key not in copy_keys:
            keep.append(edge)
            copy_keys.add(key)
    return keep


def build_edges(text: str, lexicon: UnifiedLexicon) -> list[Edge]:
    text = normalize_surface(text)
    edges = lexicon.build(text)
    edges = expand_word_variants(edges, text)
    unique: dict[tuple, Edge] = {}
    for edge in edges:
        old = unique.get(edge.key())
        if old is None or _rank(edge) > _rank(old):
            unique[edge.key()] = edge
    result = list(unique.values())
    result = prune(result, text)
    result = filter_truncated_okurigana(result, text)
    result = filter_rendaku_traps(result)
    result = repair_complete_path(result, text)
    result = derive_okurigana_stems(result, text)
    return sorted(result,
                  key=lambda e: (e.start, e.end, -e.prior, e.reading, e.source))


# ---------------------------------------------------------------- decoder

_COUNTER_SURFACES = frozenset((
    "つ", "時", "時間", "分", "秒", "年", "月", "日", "人", "回", "曲",
    "目", "匹", "個", "本", "枚", "台", "番", "度", "階", "杯", "冊", "倍",
    "段", "才", "歳", "歩"))
_SINO_DIGIT_CHARS = frozenset(
    "ぜろいちにさんよんしごろくななしちはちきゅうくじゅうひゃくせんまんおくちょう")


def _morpho_penalty(prev: Edge, curr: Edge) -> float:
    if prev.surface.isdigit():
        is_counter = curr.surface in _COUNTER_SURFACES or (
            curr.surface and is_kanji(curr.surface[0]))
        if is_counter:
            if prev.reading == prev.surface:
                return -10.0
            if prev.reading == "みっ":
                if curr.surface not in ("つ", "日", "か"):
                    return -10.0
            else:
                if any(ch not in _SINO_DIGIT_CHARS for ch in prev.reading):
                    return -10.0
                if curr.surface == "つ":
                    return -10.0
    if prev.surface == "微笑" and prev.reading == "びしょう":
        if curr.surface in ("ん", "んで", "む", "まない", "み", "めば", "もう",
                            "んだ", "た", "て"):
            return -10.0
    elif prev.surface == "気付" and prev.reading in ("きつけ", "ぎづけ"):
        if curr.surface in ("い", "いて", "いた", "く", "かず", "けば", "き"):
            return -10.0
    elif prev.surface == "失" and prev.reading in ("しつ", "うしな"):
        if curr.surface in ("く", "くし", "くして", "くした", "くさ"):
            return -10.0
    elif prev.surface == "逃" and prev.reading in ("とう", "に"):
        if curr.surface in ("し", "した", "して", "す", "せば"):
            return -10.0
    elif prev.surface == "止" and prev.reading == "し":
        if curr.surface in ("ま", "まる", "まり", "まった", "まない", "め",
                            "める", "めて"):
            return -10.0
    return 0.0


def _forced_edges(edges: list[Edge], length: int) -> set[int]:
    locked = [(i, e) for i, e in enumerate(edges) if e.locked]
    locked.sort(key=lambda item: (item[1].start,
                                  -(item[1].end - item[1].start),
                                  -item[1].confidence))
    selected: set[int] = set()
    occupied: list[tuple[int, int]] = []
    for index, edge in locked:
        if any(edge.start < end and start < edge.end
               for start, end in occupied):
            continue
        selected.add(index)
        occupied.append((edge.start, edge.end))
    if any(start < 0 or end > length for start, end in occupied):
        raise ValueError("locked edge is outside the input")
    return selected


def _allowed_edges(edges: list[Edge], length: int) -> set[int]:
    forced = _forced_edges(edges, length)
    if not forced:
        return set(range(len(edges)))
    occupied = [(edges[i].start, edges[i].end) for i in forced]
    allowed = set(forced)
    for index, edge in enumerate(edges):
        if index in forced:
            continue
        if any(edge.start < end and start < edge.end
               for start, end in occupied):
            continue
        allowed.add(index)
    return allowed


def decode_viterbi(scores, edges: list[Edge], length: int):
    allowed = _allowed_edges(edges, length)
    values = [float(s) for s in scores]
    num_edges = len(edges)
    v_edge = [-math.inf] * num_edges
    back_edge = [-1] * num_edges
    by_end: list[list[int]] = [[] for _ in range(length + 1)]
    by_start: list[list[int]] = [[] for _ in range(length + 1)]
    for index in allowed:
        edge = edges[index]
        if 0 <= edge.start < edge.end <= length:
            by_start[edge.start].append(index)
            by_end[edge.end].append(index)
    for start in range(length):
        incoming = by_end[start]
        for curr in by_start[start]:
            val = values[curr]
            if start == 0:
                if val > v_edge[curr]:
                    v_edge[curr] = val
                    back_edge[curr] = -1
            elif incoming:
                best_score = -math.inf
                best_prev = -1
                for prev in incoming:
                    if not math.isfinite(v_edge[prev]):
                        continue
                    cand = (v_edge[prev] + val
                            + _morpho_penalty(edges[prev], edges[curr]))
                    if cand > best_score:
                        best_score = cand
                        best_prev = prev
                if best_score > v_edge[curr]:
                    v_edge[curr] = best_score
                    back_edge[curr] = best_prev
    terminal = by_end[length]
    best_terminal = -1
    best_total = -math.inf
    for edge_idx in terminal:
        if v_edge[edge_idx] > best_total:
            best_total = v_edge[edge_idx]
            best_terminal = edge_idx
    if best_terminal < 0 or not math.isfinite(best_total):
        raise ValueError("candidate graph has no path covering the input")
    indices_out = []
    cursor = best_terminal
    while cursor >= 0:
        indices_out.append(cursor)
        cursor = back_edge[cursor]
    indices_out.reverse()
    return indices_out, best_total


# ---------------------------------------------------------------- windows


def safe_windows(text: str, edges: list[Edge]) -> list[tuple[int, int]]:
    if len(text) <= SEQ_LEN:
        return [(0, len(text))]
    windows = []
    start = 0
    while start < len(text):
        limit = min(len(text), start + SEQ_LEN)
        end = limit
        if limit < len(text):
            while True:
                crossing = [e for e in edges
                            if e.start < end < e.end and e.end > start]
                if not crossing:
                    break
                moved = min(e.start for e in crossing)
                if moved >= end:
                    raise RuntimeError("safe window boundary did not advance")
                end = moved
        if end <= start:
            raise ValueError(
                "provider edge exceeds the window limit; no safe split")
        windows.append((start, end))
        start = end
    return windows


def local_edges(text: str, edges: list[Edge], start: int, end: int) -> list[Edge]:
    result = []
    for edge in edges:
        if start <= edge.start and edge.end <= end:
            result.append(Edge(edge.start - start, edge.end - start,
                               edge.surface, edge.reading, edge.source,
                               edge.prior, edge.confidence, edge.locked))
        elif edge.start < end and start < edge.end:
            raise RuntimeError("internal error: a window bisected an edge")
    if not result:
        raise ValueError("no edges for a safe window")
    return result


# ---------------------------------------------------------------- resources

BUNDLE_NAME = "lexicon.txz"
# The released kashi-g2p model bundle identifies itself as
# ``##kashi-g2p v1``.  Older bundles used ``##ja_g2p v1``; both formats have
# the same section layout and are safe to deserialize here.
_BUNDLE_MAGICS = {"##ja_g2p v1", "##kashi-g2p v1"}
_BUNDLE_SECTION = "##section "


def _deserialize_bundle(raw: bytes):
    """Rebuild (table, prior_index, components, vocab) from the TSV bundle.

    Keeps dict order, per-surface entry order, list types and exact float
    values — the objects are identical to what the legacy pickles held
    (asserted by scripts/build_resource_bundle.py).
    """
    table: dict[str, tuple] = {}
    prior: dict[tuple[str, str], int] = {}
    components: dict[str, list] = {}
    vocab: list[str] = []
    section = None
    last_surface = None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("##"):
            if line in _BUNDLE_MAGICS or line == "##end":
                continue
            if line.startswith(_BUNDLE_SECTION):
                section = line[len(_BUNDLE_SECTION):]
                last_surface = None
                continue
            raise ValueError(f"bad bundle header line: {line!r}")
        if section == "pack":
            surface, reading, source, bonus, weight = line.split("\t")
            entry = (reading, source, float(bonus), float(weight))
            if surface == last_surface:
                table[surface] += (entry,)
            else:
                table[surface] = (entry,)
                last_surface = surface
        elif section == "prior":
            surface, reading, row = line.split("\t")
            prior[(surface, reading)] = int(row)
        elif section == "components":
            char, _, ids = line.partition("\t")
            components[char] = [int(v) for v in ids.split(",")] if ids else []
        elif section == "vocab":
            vocab.append(line)
    return table, prior, components, vocab


def _load_resources(resources_dir: Path | None = None):
    """Prefer the single-file bundle; fall back to the legacy pkl.gz trio."""
    res_dir = Path(resources_dir) if resources_dir else RESOURCES
    bundle = res_dir / BUNDLE_NAME
    if bundle.exists():
        with lzma.open(bundle, "rb") as fh:
            table, prior, components, vocab = _deserialize_bundle(fh.read())
        return table, prior, components, vocab
    with gzip.open(res_dir / "unified_lexicon_variants_v2.pkl.gz", "rb") as fh:
        table = pickle.load(fh)
    with gzip.open(res_dir / "prior_index.pkl.gz", "rb") as fh:
        prior = pickle.load(fh)
    with gzip.open(res_dir / "components.pkl", "rb") as fh:
        components = pickle.load(fh)
    vocab = (res_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    return table, prior, components, vocab


# ---------------------------------------------------------------- runtime


class G2POnnxRuntime:
    def __init__(self, model_dir: str | Path | None = None, *, prefer_dml: bool = True,
                 model_file: str = "model.onnx"):
        resolved_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        if not resolved_dir.exists():
            legacy_dir = PROJECT_ROOT / "models" / "ja_g2p_onnx"
            if legacy_dir.exists():
                resolved_dir = legacy_dir
        self.model_dir = resolved_dir
        resources_dir = resolved_dir / "resources" if (resolved_dir / "resources").exists() else resolved_dir
        table, prior, components, vocab = _load_resources(resources_dir)
        self.table = table
        self.prior_index = prior
        self.components = components
        self.vocab = {}
        for index, line in enumerate(vocab):
            self.vocab[line.rstrip("\r")] = index
        self.unk_id = int(self.vocab.get("[UNK]", 1))
        self.lexicon = UnifiedLexicon(self.table, use_rules=True, use_copy=False)
        providers = ort.get_available_providers()
        use = ["DmlExecutionProvider", "CPUExecutionProvider"] \
            if prefer_dml and "DmlExecutionProvider" in providers \
            else ["CPUExecutionProvider"]
        model_path = resolved_dir / model_file
        self.session = ort.InferenceSession(str(model_path), providers=use)
        self.provider_name = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self._reading_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _features(self, text: str):
        unk = self.unk_id
        input_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
        attention = np.zeros((1, SEQ_LEN), dtype=bool)
        for index, char in enumerate(text[:SEQ_LEN]):
            input_ids[0, index] = self.vocab.get(char, unk)
            attention[0, index] = True
        components = np.zeros((1, SEQ_LEN, MAX_COMPONENTS), dtype=np.int64)
        char_types = np.zeros((1, SEQ_LEN), dtype=np.int64)
        for index, char in enumerate(text[:SEQ_LEN]):
            char_types[0, index] = _char_type(char)
            for j, value in enumerate(self.components.get(char, ())[:MAX_COMPONENTS]):
                components[0, index, j] = value
        return input_ids, attention, components, char_types

    def _reading_tokens(self, reading: str) -> tuple[np.ndarray, np.ndarray]:
        """Token ids for one reading string, cached (readings repeat heavily)."""
        cached = self._reading_cache.get(reading)
        if cached is None:
            ids = np.zeros((READING_LEN,), dtype=np.int64)
            mask = np.zeros((READING_LEN,), dtype=bool)
            for j, char in enumerate(hira(reading)[:READING_LEN]):
                ids[j] = self.vocab.get(char, self.unk_id)
                mask[j] = True
            if len(self._reading_cache) < 200000:
                self._reading_cache[reading] = (ids, mask)
            return ids, mask
        return cached

    def _edge_tensors(self, edges: list[Edge]):
        count = len(edges)
        if count > MAX_EDGES:
            raise ValueError(f"{count} edges exceed the candidate cap ({MAX_EDGES})")
        starts = np.zeros((1, count), dtype=np.int64)
        ends = np.zeros((1, count), dtype=np.int64)
        reading_ids = np.zeros((1, count, READING_LEN), dtype=np.int64)
        reading_mask = np.zeros((1, count, READING_LEN), dtype=bool)
        mask = np.ones((1, count), dtype=bool)
        source_ids = np.zeros((1, count), dtype=np.int64)
        prior = np.zeros((1, count), dtype=np.float32)
        pack_ids = np.zeros((1, count), dtype=np.int64)
        locked = np.zeros((1, count), dtype=bool)
        for index, edge in enumerate(edges):
            starts[0, index] = edge.start
            ends[0, index] = edge.end
            mask[0, index] = True
            source_ids[0, index] = SOURCE_TO_ID.get(edge.source, 7)
            prior[0, index] = edge.prior
            pack_ids[0, index] = self.prior_index.get(
                (edge.surface, edge.reading), 0)
            ids_row, mask_row = self._reading_tokens(edge.reading)
            reading_ids[0, index] = ids_row
            reading_mask[0, index] = mask_row
        return dict(edge_start=starts, edge_end=ends,
                    edge_reading_ids=reading_ids,
                    edge_reading_mask=reading_mask,
                    edge_mask=mask, edge_source_ids=source_ids,
                    edge_prior=prior, edge_pack_ids=pack_ids,
                    edge_locked=locked)

    def score(self, text: str, edges: list[Edge]) -> np.ndarray:
        input_ids, attention, components, char_types = self._features(text)
        feeds = self._edge_tensors(edges)
        feeds.update(input_ids=input_ids, attention_mask=attention,
                     component_ids=components, char_type_ids=char_types)
        scores = self.session.run(["edge_scores"], feeds)[0]
        return scores[0][:len(edges)]

    def predict(self, text: str) -> dict:
        text = normalize_surface(text)
        if not text:
            return {"text": text, "reading": "", "edges": [], "score": 0.0,
                    "windows": []}
        global_edges = build_edges(text, self.lexicon)
        windows = safe_windows(text, global_edges)
        readings = []
        stitched = []
        total_score = 0.0
        window_info = []
        for start, end in windows:
            local = local_edges(text, global_edges, start, end)
            local_text = text[start:end]
            scores = self.score(local_text, local)
            indices, score = decode_viterbi(scores, local, len(local_text))
            total_score += score
            readings.append("".join(local[i].reading for i in indices))
            for i in indices:
                edge = local[i]
                stitched.append({"start": edge.start + start,
                                 "end": edge.end + start,
                                 "surface": edge.surface,
                                 "reading": edge.reading,
                                 "source": edge.source,
                                 "locked": edge.locked})
            window_info.append({"start": start, "end": end})
        return {"text": text, "reading": "".join(readings), "edges": stitched,
                "score": total_score, "windows": window_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU execution (default: DirectML if available)")
    parser.add_argument("--model", default="model.onnx",
                        help="model file in this folder (model.onnx / "
                             "model.fp16.onnx / model.int8.onnx)")
    args = parser.parse_args()

    started = time.perf_counter()
    runtime = G2POnnxRuntime(prefer_dml=not args.cpu, model_file=args.model)
    load_seconds = time.perf_counter() - started
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    results = []
    forward_seconds = 0.0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            results.append({"line": line_number, "text": line, "reading": "",
                            "score": 0.0, "tokens": []})
            continue
        t0 = time.perf_counter()
        prediction = runtime.predict(line)
        forward_seconds += time.perf_counter() - t0
        results.append({
            "line": line_number,
            "text": line,
            "reading": prediction["reading"],
            "score": round(float(prediction["score"]), 4),
            "tokens": prediction["edges"],
        })
        print(f"  {line_number}/{len(lines)} "
              f"({time.perf_counter() - started:.0f}s)", file=sys.stderr)

    payload = {
        "meta": {
            "input": str(Path(args.input).resolve()),
            "provider": runtime.provider_name,
            "lexicon": "unified pack + copy",
            "fast_mode": True,
            "lines_total": len(lines),
            "load_seconds": round(load_seconds, 1),
            "forward_seconds": round(forward_seconds, 1),
        },
        "lines": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"WROTE {output} ({payload['meta']['forward_seconds']}s forward)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
