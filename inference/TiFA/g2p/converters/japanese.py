"""Japanese kana and MeCab G2P converters with shared pronunciation rules."""

from itertools import product
from pathlib import Path

from ..registry import converter
from .base import Converter, G2PConversionError, G2PGroup, G2PPath, G2PReading, G2PWord
from .dictionary import PronunciationScriptDictionaryConverter
from .text import SMALL_KANA, find_run, is_kana

# Hiragana-only romaji table, derived from cpp-kana's kanaToRomajiMap.
# Katakana is converted to hiragana before lookup.
_KANA_TO_ROMAJI: dict[str, str] = {
    # ---- sokuon ----
    "っ": "cl",
    # ---- gojuon ----
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we",
    "ん": "n",
    # ---- dakuten ----
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    # ---- handakuten ----
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    # ---- yoon / digraphs ----
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "きぇ": "kye",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo", "ぎぇ": "gye",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "じゃ": "ja",  "じゅ": "ju",  "じょ": "jo",  "じぇ": "je",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "にぇ": "nye",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo", "ひぇ": "hye",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "びぇ": "bye",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo", "ぴぇ": "pye",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "みぇ": "mye",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo", "りぇ": "rye",
    # ---- vowel-extension digraphs ----
    "いぇ": "ye",
    "うぁ": "wa", "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo",
    "ぐぁ": "gwa", "ぐぃ": "gwi", "ぐぇ": "gwe", "ぐぉ": "gwo",
    "すぁ": "swa", "すぃ": "swi", "すぇ": "swe", "すぉ": "swo",
    "ずぁ": "zwa", "ずぃ": "zwi", "ずぇ": "zwe", "ずぉ": "zwo",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
    "てぃ": "ti", "てゅ": "tyu",
    "でぃ": "di", "でゅ": "dyu",
    "とぅ": "tu",
    "どぅ": "du",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "ぶぁ": "bwa", "ぶぃ": "bwi", "ぶぇ": "bwe", "ぶぉ": "bwo",
    "ぷぁ": "pwa", "ぷぃ": "pwi", "ぷぇ": "pwe", "ぷぉ": "pwo",
    "ぬぁ": "nwa", "ぬぃ": "nwi", "ぬぇ": "nwe", "ぬぉ": "nwo",
    "むぁ": "mwa", "むぃ": "mwi", "むぇ": "mwe", "むぉ": "mwo",
    "るぁ": "rwa", "るぃ": "rwi", "るぇ": "rwe", "るぉ": "rwo",
    # ---- vu variants (hiragana U+3094) ----
    "ゔ": "vu",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    # ---- special ----
    "を": "o",
    # ---- Vocal2Midi: lexicon-coverage additions (ja_g2p pack readings) ----
    "ぁぅ": "au",
    "あぁ": "aa",
    "あぅ": "au",
    "あぇ": "ae",
    "ぃぃ": "ii",
    "いぁ": "ia",
    "いぃ": "ii",
    "いぅ": "iu",
    "いゃ": "iya",
    "いょ": "iyo",
    "えぁ": "ea",
    "えぃ": "ei",
    "えぅ": "eu",
    "えぇ": "ee",
    "おぁ": "oa",
    "おぅ": "ou",
    "かぁ": "kaa",
    "かぅ": "kau",
    "かゃ": "kya",
    "がぁ": "gaa",
    "きぃ": "kii",
    "きぅ": "kiu",
    "ぎぃ": "gii",
    "くぅ": "kuu",
    "ぐぅ": "guu",
    "けぁ": "kea",
    "けぃ": "kei",
    "けぇ": "kee",
    "けょ": "kyo",
    "げぃ": "gei",
    "げぇ": "gee",
    "こぉ": "koo",
    "ごぁ": "goa",
    "ごぉ": "goo",
    "さぁ": "saa",
    "ざぁ": "zaa",
    "ざぃ": "zai",
    "ざゃ": "jya",
    "しぁ": "shia",
    "しぃ": "shii",
    "じぁ": "jia",
    "じぃ": "jii",
    "じぉ": "jio",
    "すぅ": "suu",
    "ずぅ": "zuu",
    "せぃ": "sei",
    "せぇ": "see",
    "ぜぁ": "zea",
    "ぜぇ": "zee",
    "そぁ": "soa",
    "そぅ": "sou",
    "そぉ": "soo",
    "ぞぁ": "zoa",
    "ぞぉ": "zoo",
    "たぁ": "taa",
    "たぅ": "tau",
    "たぉ": "tao",
    "だぁ": "daa",
    "だぃ": "dai",
    "だょ": "dyo",
    "ちぁ": "chia",
    "ちぃ": "chii",
    "ちぉ": "chio",
    "ぢぃ": "dyi",
    "ぢぇ": "dye",
    "ぢゃ": "dya",
    "ぢゅ": "dyu",
    "ぢょ": "dyo",
    "つぅ": "tsuu",
    "づぁ": "dua",
    "づぉ": "duo",
    "てぅ": "teu",
    "てぇ": "tee",
    "てょ": "tyo",
    "でぁ": "dea",
    "でぅ": "deu",
    "でぇ": "dee",
    "とぁ": "toa",
    "とぇ": "toe",
    "とぉ": "too",
    "とゅ": "tyu",
    "どぉ": "doo",
    "どゅ": "dyu",
    "なぃ": "nai",
    "にぁ": "nia",
    "にぃ": "nii",
    "にぅ": "niu",
    "ぬぅ": "nuu",
    "ぬゅ": "nyu",
    "ねぃ": "nei",
    "ねぇ": "nee",
    "のぅ": "nou",
    "はぁ": "haa",
    "はぅ": "hau",
    "ばぁ": "baa",
    "ぱぁ": "paa",
    "ひぁ": "hia",
    "ひぃ": "hii",
    "びぃ": "bii",
    "ぴぁ": "pia",
    "ぴぃ": "pii",
    "ぴぅ": "piu",
    "ふぅ": "fuu",
    "ふゅ": "fyu",
    "ふょ": "fyo",
    "ぶぅ": "buu",
    "ぶゅ": "byu",
    "ぷぅ": "puu",
    "ぷゅ": "pyu",
    "べぇ": "bee",
    "ぺぇ": "pee",
    "ほぁ": "hoa",
    "ほぃ": "hoi",
    "ほぅ": "hou",
    "ほぇ": "hoe",
    "ほぉ": "hoo",
    "ほょ": "hyo",
    "ぼぁ": "boa",
    "ぼぅ": "bou",
    "ぼぉ": "boo",
    "まぁ": "maa",
    "みぃ": "mii",
    "みぅ": "miu",
    "むぅ": "muu",
    "めぁ": "mea",
    "めぇ": "mee",
    "もぃ": "moi",
    "もぅ": "mou",
    "ゆぁ": "yua",
    "ゆぃ": "yui",
    "ゆぅ": "yuu",
    "ゆぇ": "yue",
    "よぅ": "you",
    "らぁ": "raa",
    "らぃ": "rai",
    "らぅ": "rau",
    "りぁ": "ria",
    "りぃ": "rii",
    "りぅ": "riu",
    "るぅ": "ruu",
    "るゅ": "ryu",
    "れぁ": "rea",
    "れぃ": "rei",
    "れぇ": "ree",
    "ろぅ": "rou",
    "ろぉ": "roo",
    "わぁ": "waa",
    "わぃ": "wai",
    "わぉ": "wao",
    "ーぁ": "a",
    "ーぃ": "i",
    "ーぅ": "u",

}

_HIRAGANA_START = 0x3041
_KATAKANA_START = 0x30A1
_KANA_SPAN = 0x5E


def _kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana by shifting the Unicode range."""
    result: list[str] = []
    for ch in text:
        cp = ord(ch)
        if _KATAKANA_START <= cp < _KATAKANA_START + _KANA_SPAN:
            result.append(chr(cp - _KATAKANA_START + _HIRAGANA_START))
        else:
            result.append(ch)
    return "".join(result)


_CONSONANT_LEADING = frozenset(
    "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
)


def _apply_sokuon(romaji_list: list[str]) -> list[str]:
    """Resolve gemination: replace 'cl' with the leading consonant of the
    next non-empty romaji token.  Empty strings (placeholders for skipped
    kana) are passed through unchanged.  Returns a list of the same length."""
    result: list[str] = []
    i = 0
    while i < len(romaji_list):
        r = romaji_list[i]
        if r == "cl":
            # Find next non-placeholder romaji
            j = i + 1
            while j < len(romaji_list) and romaji_list[j] == "":
                j += 1
            if j < len(romaji_list):
                nxt = romaji_list[j]
                if nxt[0] in _CONSONANT_LEADING:
                    result.append(nxt[0])
                    i += 1
                    continue
        result.append(r)
        i += 1
    return result


@converter(id="japanese-kana", language="ja,jpn")
class JapaneseKanaConverter(PronunciationScriptDictionaryConverter):
    """Japanese kana-to-phoneme converter.

    Two-phase: kana -> romaji (text-to-script), then
    romaji -> phonemes (script-to-phonemes via dictionary).

    Parameters mirror cpp-kana:
        *dict_path*: romaji-to-phoneme dictionary.
        *double_written_sokuon*: enable gemination resolution
          (``cl`` + consonant -> consonant gemination).
    """

    def __init__(
        self,
        dict_path: str,
        *,
        double_written_sokuon: bool = False,
    ) -> None:
        super().__init__(dict_path=dict_path)
        self._double_written_sokuon = double_written_sokuon

    def find(self, text: str) -> tuple[int, int] | None:
        return find_run(text, is_kana)

    def script_to_paths(self, script: str) -> list[G2PPath]:
        if script == "":
            return [[]]
        if script in self._script_dict:
            return super().script_to_paths(script)
        # Single consonants from gemination pass through directly
        if len(script) == 1 and script in _CONSONANT_LEADING:
            return [[G2PGroup(script=script, phonemes=[script])]]
        raise KeyError(
            f"Script token {script!r} not found in script-to-phoneme dict."
        )

    def text_to_scripts(self, words: list[str]) -> list[list[str]]:
        # Convert katakana to hiragana for unified lookup
        hiragana_tokens = [_kata_to_hira(t) for t in words]

        # Kana -> romaji; long vowel U+30FC and handakuten U+309C are empty.
        romaji_list: list[str] = []
        for t in hiragana_tokens:
            if t in ("ー", "゜"):
                romaji_list.append("")
                continue
            r = _KANA_TO_ROMAJI.get(t)
            if r is not None:
                romaji_list.append(r)
            else:
                romaji_list.append(t)

        if self._double_written_sokuon:
            romaji_list = _apply_sokuon(romaji_list)

        return [[r] for r in romaji_list]


def _is_japanese_char(char: str) -> bool:
    cp = ord(char)
    return (
        is_kana(char)
        or char in "々〆〇"
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2FA1F
        or 0x30000 <= cp <= 0x323AF
    )


@converter(id="japanese-mecab", language="ja,jpn")
class JapaneseMecabConverter(Converter):
    """Segment full Japanese word forms and enumerate whole-word readings.

    MeCab supplies only kana pronunciations (UniDic's ``pron`` field).
    Romaji, phoneme groups, dictionary alternatives and long vowels are
    handled by :class:`JapaneseKanaConverter` without additional rules.
    The full PyPI ``unidic`` dictionary is loaded on first conversion and
    downloaded automatically if it has not been installed yet.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        nbest: int = 32,
        double_written_sokuon: bool = False,
        unidic_dir: str | None = None,
    ) -> None:
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer.")
        self._nbest = nbest
        self._double_written_sokuon = double_written_sokuon
        self._unidic_dir = unidic_dir
        self._tagger = None
        self._kana = JapaneseKanaConverter(
            dict_path=dict_path, double_written_sokuon=double_written_sokuon,
        )

    def __getstate__(self):
        # Binarization and data loaders use spawned worker processes. MeCab's
        # native tagger cannot be pickled; each worker creates its own instance.
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state

    def find(self, text: str) -> tuple[int, int] | None:
        # In particular, never take ASCII romaji/phoneme input from the
        # Japanese dictionary converter used by existing training datasets.
        return find_run(text, _is_japanese_char)

    def _get_tagger(self):
        if self._tagger is not None:
            return self._tagger
        try:
            import fugashi
            import unidic
            from unidic.download import download_version
        except ModuleNotFoundError as exc:
            if exc.name not in ("fugashi", "unidic"):
                raise
            raise ImportError(
                "japanese-mecab requires optional dependencies. Install them with: "
                'python -m pip install "fugashi>=1.3,<2" "unidic>=1.1,<2"'
            ) from exc
        if self._unidic_dir is None:
            dicdir = Path(unidic.DICDIR)
            required = (dicdir / "sys.dic", dicdir / "mecabrc")
            if not all(path.is_file() for path in required):
                # Serialize first-use downloads across spawned workers.
                import filelock  # optional dependency, MeCab dict download only

                with filelock.FileLock(str(dicdir.parent / "download.lock")):
                    if not all(path.is_file() for path in required):
                        download_version()
        else:
            dicdir = Path(self._unidic_dir)
        # Explicit paths avoid system MeCab dictionaries and support spaces
        # in Windows environment paths.
        dicdir = dicdir.resolve()
        self._tagger = fugashi.Tagger(
            f'-r "{(dicdir / "mecabrc").as_posix()}" -d "{dicdir.as_posix()}"'
        )
        return self._tagger

    def _pronunciations(self, word: str) -> list[str]:
        readings: list[str] = []
        seen: set[str] = set()
        for nodes in self._get_tagger().nbestToNodeList(word, self._nbest):
            if len(nodes) != 1 or nodes[0].surface != word:
                continue
            pron = nodes[0].feature.pron
            if not pron or pron in ("*", "-") or pron in seen:
                continue
            seen.add(pron)
            readings.append(pron)
        return readings

    def _reading(self, kana: str) -> G2PReading:
        kana_words = self._kana.convert(kana)
        alternatives = [
            [path for reading in word.readings for path in reading.paths]
            for word in kana_words
        ]
        # Keep each candidate as a complete word path. Per-kana dictionary
        # alternatives combine within this reading, not across readings.
        paths = []
        seen = set()
        for parts in product(*alternatives):
            path = [group for part in parts for group in part]
            key = tuple((group.script, tuple(group.phonemes)) for group in path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
        return G2PReading(paths=paths)

    def convert(self, text: str) -> list[G2PWord]:
        if not text:
            return []
        # Snapshot surfaces before N-best calls replace MeCab's lattice.
        surfaces: list[str] = []
        for node in self._get_tagger()(text):
            surface = node.surface
            # Keep kana digraphs intact even if MeCab splits them.
            if (
                surfaces
                and surface[0] in SMALL_KANA
                and all(is_kana(char) for char in surfaces[-1] + surface)
                and _kata_to_hira(surfaces[-1][-1] + surface[0]) in _KANA_TO_ROMAJI
            ):
                surfaces[-1] += surface
            else:
                surfaces.append(surface)

        word_readings: list[tuple[str, list[str]]] = []
        for surface in surfaces:
            pronunciations = self._pronunciations(surface)
            if not pronunciations and not all(is_kana(char) for char in surface):
                raise G2PConversionError([surface])
            if (
                word_readings
                and self._double_written_sokuon
                and any(_kata_to_hira(pron).rstrip("ー゜").endswith("っ")
                        for pron in (word_readings[-1][1] or [word_readings[-1][0]]))
            ):
                # Keep gemination and the following reading in one path choice.
                previous, previous_readings = word_readings[-1]
                word_readings[-1] = (
                    previous + surface,
                    list(dict.fromkeys(left + right for left, right in product(
                        previous_readings or [previous], pronunciations or [surface],
                    ))),
                )
            else:
                word_readings.append((surface, pronunciations))
        result: list[G2PWord] = []
        for surface, pronunciations in word_readings:
            if pronunciations:
                result.append(G2PWord(
                    text=surface, readings=[self._reading(pron) for pron in pronunciations],
                ))
            else:
                result.extend(self._kana.convert(surface))
        return result
