"""Faithful port of cpp-pinyin character classifications and tone tables."""

# Hanzi ranges matching cpp-pinyin G2pglobal.h isHanzi()
HAN_RANGES = [
    (0x4E00, 0x9FA5),
]

# CJK character ranges.
CJK_RANGES = [
    (0x2E80, 0x2EFF),
    (0x2F00, 0x2FDF),
    (0x3000, 0x303F),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
]

# Special kana characters recognized by cpp-pinyin
SPECIAL_KANA = frozenset(
    "ャュョゃゅょァィゥェォぁぃぅぇぉ"
)

# ---- Tone mark -> (base char, tone digit) ----
# Covers the exact same set as ManTone.cpp tone_map
TONE_MAP: dict[int, tuple[str, str]] = {}

# Initialize TONE_MAP for the range U+00E0..U+01DC + a few extras
_tone_data = [
    # a
    (0x0101, "a", "1"), (0x00E1, "a", "2"), (0x01CE, "a", "3"), (0x00E0, "a", "4"),
    # e
    (0x0113, "e", "1"), (0x00E9, "e", "2"), (0x011B, "e", "3"), (0x00E8, "e", "4"),
    # i
    (0x012B, "i", "1"), (0x00ED, "i", "2"), (0x01D0, "i", "3"), (0x00EC, "i", "4"),
    # o
    (0x014D, "o", "1"), (0x00F3, "o", "2"), (0x01D2, "o", "3"), (0x00F2, "o", "4"),
    # u
    (0x016B, "u", "1"), (0x00FA, "u", "2"), (0x01D4, "u", "3"), (0x00F9, "u", "4"),
    # v (ü with tones)  ǖ ǘ ǚ ǜ
    (0x01D6, "v", "1"), (0x01D8, "v", "2"), (0x01DA, "v", "3"), (0x01DC, "v", "4"),
    # plain ü
    (0x00FC, "v", None),
    # syllabic n, m
    (0x0144, "n", "2"), (0x0148, "n", "3"), (0x01F9, "n", "4"),
    (0x1E3F, "m", "2"),
]
for cp, base, tone in _tone_data:
    TONE_MAP[cp] = (base, tone)

# ---- Bopomofo mappings (faithful to ManTone.cpp) ----
BOPOMOFO_INITIALS: dict[str, str] = {
    "b": "ㄅ", "p": "ㄆ", "m": "ㄇ", "f": "ㄈ",
    "d": "ㄉ", "t": "ㄊ", "n": "ㄋ", "l": "ㄌ",
    "g": "ㄍ", "k": "ㄎ", "h": "ㄏ",
    "j": "ㄐ", "q": "ㄑ", "x": "ㄒ",
    "zh": "ㄓ", "ch": "ㄔ", "sh": "ㄕ", "r": "ㄖ",
    "z": "ㄗ", "c": "ㄘ", "s": "ㄙ",
}

BOPOMOFO_FINALS: dict[str, str] = {
    "a": "ㄚ", "o": "ㄛ", "e": "ㄜ", "eh": "ㄝ",
    "ai": "ㄞ", "ei": "ㄟ", "ao": "ㄠ", "ou": "ㄡ",
    "an": "ㄢ", "en": "ㄣ", "ang": "ㄤ", "eng": "ㄥ", "er": "ㄦ",
    "i": "ㄧ", "ia": "ㄧㄚ", "io": "ㄧㄛ", "ie": "ㄧㄝ",
    "iai": "ㄧㄞ", "iao": "ㄧㄠ", "iu": "ㄧㄡ",
    "ian": "ㄧㄢ", "in": "ㄧㄣ", "iang": "ㄧㄤ", "ing": "ㄧㄥ",
    "u": "ㄨ", "ua": "ㄨㄚ", "uo": "ㄨㄛ", "uai": "ㄨㄞ",
    "ui": "ㄨㄟ", "uan": "ㄨㄢ", "un": "ㄨㄣ",
    "uang": "ㄨㄤ", "ong": "ㄨㄥ",
    "v": "ㄩ", "ve": "ㄩㄝ", "van": "ㄩㄢ", "vn": "ㄩㄣ", "iong": "ㄩㄥ",
}

BOPOMOFO_TONES: dict[int, str] = {
    1: "", 2: "\u02CA", 3: "\u02C7", 4: "\u02CB", 5: "\u02D9",
}

# Tone output style constants (matching cpp-pinyin API)
STYLE_NORMAL = 0
STYLE_TONE = 1
STYLE_TONE2 = 2
STYLE_FIRST_LETTER = 3
STYLE_BOPOMOFO = 4
STYLE_TONE3 = 8
