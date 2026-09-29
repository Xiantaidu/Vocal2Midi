"""Faithful port of cpp-pinyin ManTone / CanTone tone conversion.

Handles both Mandarin (tone-mark input) and Cantonese (TONE3 input)
pronunciations.
"""

from .constants import (
    TONE_MAP, STYLE_NORMAL, STYLE_TONE, STYLE_TONE2,
    STYLE_FIRST_LETTER, STYLE_BOPOMOFO, STYLE_TONE3,
    BOPOMOFO_INITIALS, BOPOMOFO_FINALS, BOPOMOFO_TONES,
)

# Vowels that can carry tone marks in Mandarin
_TONE_VOWELS = frozenset("aeiouv")


def _tone_mark_to_parts(pinyin: str) -> tuple[str, int]:
    """Convert tone-mark pinyin to (base, tone_number).

    ``zhōng`` -> ``(zhong, 1)``,  ``le`` -> ``(le, 5)``
    """
    result: list[str] = []
    tone = 5
    for ch in pinyin:
        cp = ord(ch)
        if cp in TONE_MAP:
            base, t = TONE_MAP[cp]
            result.append(base)
            if t is not None:
                tone = int(t)
        else:
            result.append(ch)
    return "".join(result), tone


def _tone3_to_parts(pinyin: str) -> tuple[str, int]:
    """Split TONE3 pinyin into (base, tone_number).

    ``zhong1`` -> ``(zhong, 1)``, ``zung1`` -> ``(zung, 1)``
    """
    if pinyin and pinyin[-1].isdigit():
        return pinyin[:-1], int(pinyin[-1])
    return pinyin, 5


def _pinyin_to_bopomofo(pinyin: str, tone: int) -> str:
    """Convert a pinyin syllable to Bopomofo with tone mark."""
    # Try longest initial first
    initial = ""
    for length in (2, 1):
        if len(pinyin) >= length and pinyin[:length] in BOPOMOFO_INITIALS:
            initial = BOPOMOFO_INITIALS[pinyin[:length]]
            pinyin = pinyin[length:]
            break

    # Remaining is the final
    final = BOPOMOFO_FINALS.get(pinyin, "")
    if not final:
        # Fallback: keep original
        return pinyin

    tone_mark = BOPOMOFO_TONES.get(tone, "")
    return f"{initial}{final}{tone_mark}"


def apply_tone(pinyin: str, style: int, *, v_to_u: bool = False,
               neutral_tone5: bool = True) -> str:
    """Apply tone conversion, faithfully mirroring cpp-pinyin's logic.

    Handles both input formats:
    - Tone marks (Mandarin): ``zhōng``, ``nǚ``
    - TONE3 (Cantonese): ``zung1``, ``neoi5``

    Detects the input format automatically.
    """
    is_tone3_input = bool(pinyin) and pinyin[-1].isdigit()

    if is_tone3_input:
        base, tone = _tone3_to_parts(pinyin)
    else:
        base, tone = _tone_mark_to_parts(pinyin)

    if style == STYLE_TONE3:
        s = base.replace("ü", "v").replace("U+00FC", "v")
        if v_to_u:
            s = s.replace("v", "ü")
        return f"{s}{'' if tone == 5 and not neutral_tone5 else tone}"

    if style == STYLE_NORMAL:
        s = base.replace("ü", "v").replace("U+00FC", "v")
        if v_to_u:
            s = s.replace("v", "ü")
        return s

    if style == STYLE_FIRST_LETTER:
        return base[0] if base else ""

    if style == STYLE_BOPOMOFO:
        return _pinyin_to_bopomofo(base, tone)

    if style == STYLE_TONE:
        # Tone marks: apply the tone mark to the appropriate vowel
        return _apply_tone_mark(base, tone, v_to_u)

    if style == STYLE_TONE2:
        return _apply_tone2(base, tone, v_to_u)

    return base


def _apply_tone_mark(base: str, tone: int, v_to_u: bool) -> str:
    """Apply tone mark diacritic to the pinyin syllable (TONE style)."""
    if tone == 5:
        return _replace_u_v(base, v_to_u)

    # Find the tone-bearing vowel (last of aeiouv in standard order)
    tone_vowel = None
    tone_idx = -1
    for i, ch in enumerate(base):
        if ch in _TONE_VOWELS:
            tone_vowel = ch
            tone_idx = i

    if tone_vowel is None or tone_idx < 0:
        return _replace_u_v(base, v_to_u)

    # Build the tone mark character
    marked = _vowel_with_tone(tone_vowel, tone)

    result = list(base)
    result[tone_idx] = marked
    return _replace_u_v("".join(result), v_to_u)


def _apply_tone2(base: str, tone: int, v_to_u: bool) -> str:
    """Apply TONE2 format (vowel followed by digit)."""
    s = _replace_u_v(base, v_to_u)
    if tone == 5:
        return s
    # Find tone vowel and insert number after it
    for i, ch in enumerate(s):
        if ch in _TONE_VOWELS:
            return s[:i+1] + str(tone) + s[i+1:]
    return s + str(tone)


def _vowel_with_tone(vowel: str, tone: int) -> str:
    """Return the vowel character with the given tone mark."""
    lookup = {
        ("a", 1): "ā", ("a", 2): "á", ("a", 3): "ǎ", ("a", 4): "à",
        ("e", 1): "ē", ("e", 2): "é", ("e", 3): "ě", ("e", 4): "è",
        ("i", 1): "ī", ("i", 2): "í", ("i", 3): "ǐ", ("i", 4): "ì",
        ("o", 1): "ō", ("o", 2): "ó", ("o", 3): "ǒ", ("o", 4): "ò",
        ("u", 1): "ū", ("u", 2): "ú", ("u", 3): "ǔ", ("u", 4): "ù",
        ("v", 1): "ǖ", ("v", 2): "ǘ", ("v", 3): "ǚ", ("v", 4): "ǜ",
    }
    return lookup.get((vowel, tone), vowel)


def _replace_u_v(s: str, v_to_u: bool) -> str:
    """Replace ü <-> v based on v_to_u flag."""
    if v_to_u:
        return s.replace("v", "ü")
    return s
