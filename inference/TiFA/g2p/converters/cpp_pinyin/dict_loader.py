"""Faithful port of cpp-pinyin DictUtil dictionary loading.

Handles the original colon-separated dictionary format:
  word.txt:       char:pron1,pron2,pron3
  phrases_dict.txt: phrase:pron1,pron2,pron3,pron4
  phrases_map.txt:  char:digits
  trans_word.txt:   trad:simp
  user_dict.txt:    phrase:pron1 pron2 pron3  (space-separated, TONE3)
"""


from pathlib import Path


def encode_phrase_key(chars: list[str]) -> int:
    """64-bit phrase key encoding matching cpp-pinyin's encodePhrasesKey().

    Packs 2-4 char16 values into a 64-bit integer.
    """
    key = 0
    for ch in chars:
        key = (key << 16) | ord(ch)
    return key


def load_word_dict(path: str | Path) -> dict[str, list[list[str]]]:
    """Load word.txt -> dict[char] = [[pron1], [pron2], ...].

    Original format: 汉字:拼音1,拼音2,...
    """
    result: dict[str, list[list[str]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            char, pron_str = parts
            if not char or not pron_str:
                continue
            pronunciations = [p.strip() for p in pron_str.split(",") if p.strip()]
            pron_lists = [[p] for p in pronunciations]
            result[char] = pron_lists
    return result


def load_phrase_dict(path: str | Path) -> dict[int, list[list[str]]]:
    """Load phrases_dict.txt -> dict[int_key] = [[pron1, pron2, ...], ...].

    Original format: 词组:拼音1,拼音2,拼音3,拼音4
    """
    result: dict[int, list[list[str]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            phrase_str, pron_str = parts
            if not phrase_str or len(phrase_str) < 2 or not pron_str:
                continue
            pronunciations = [p.strip() for p in pron_str.split(",") if p.strip()]
            if len(pronunciations) != len(phrase_str):
                continue
            key = encode_phrase_key(list(phrase_str))
            result.setdefault(key, []).append(pronunciations)
    return result


def load_phrase_map(path: str | Path) -> set[str]:
    """Load phrases_map.txt -> set of polyphonic character strings.

    Original format: 汉字:digits
    """
    result: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[0]:
                result.add(parts[0])
    return result


def load_trans_dict(path: str | Path) -> dict[str, str]:
    """Load trans_word.txt -> dict[trad_char] = simp_char.

    Original format: 繁体字:简体字
    """
    result: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                result[parts[0]] = parts[1]
    return result


def load_user_dict(path: str | Path) -> dict[int, list[list[str]]]:
    """Load user_dict.txt into the same format as load_phrase_dict.

    Original format: 词组:pron1 pron2 pron3  (space-separated, TONE3)
    """
    result: dict[int, list[list[str]]] = {}
    p = Path(path)
    if not p.exists():
        return result
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            phrase_str, pron_str = parts
            if not phrase_str or len(phrase_str) < 2 or not pron_str:
                continue
            pronunciations = pron_str.split()
            if len(pronunciations) != len(phrase_str):
                continue
            key = encode_phrase_key(list(phrase_str))
            result.setdefault(key, []).append(pronunciations)
    return result
