import re
import logging

logger = logging.getLogger(__name__)


def _get_onnx_runtime(model_dir=None):
    """kashi-g2p-onnx runtime when the model folder ships with the app, else None."""
    try:
        from inference.ja_onnx_g2p import get_ja_onnx_runtime
    except ImportError:
        return None
    return get_ja_onnx_runtime(model_dir)


def is_letter(character):
    return ('a' <= character <= 'z') or ('A' <= character <= 'Z')


def is_special_letter(character):
    special_letter = "'-’"
    return character in special_letter


def is_digit(character):
    return character.isdigit() or ('０' <= character <= '９')


def is_numeric_like(character):
    return is_digit(character) or character == '〇'


def is_kanji(character):
    code = ord(character)
    return 0x4E00 <= code <= 0x9FFF


def is_kana(character):
    code = ord(character)
    return (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)


def is_special_kana(character):
    special_kana = "ャュョゃゅょァィゥェォぁぃぅぇぉ"
    return character in special_kana


def is_japanese_symbol(character):
    return character in {"々", "〆", "ヶ", "ヵ", "ー", "〇"}


def is_japanese_char(character):
    return is_kanji(character) or is_kana(character) or is_japanese_symbol(character)

KATA_TO_ROMAJI = {
    'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
    'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
    'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
    'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
    'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
    'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
    'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
    'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
    'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
    'ワ': 'wa', 'ヲ': 'o', 'ン': 'n',
    'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
    'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
    'ダ': 'da', 'ヂ': 'ji', 'ヅ': 'zu', 'デ': 'de', 'ド': 'do',
    'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
    'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
    'キャ': 'kya', 'キュ': 'kyu', 'キョ': 'kyo',
    'シャ': 'sha', 'シュ': 'shu', 'ショ': 'sho',
    'チャ': 'cha', 'チュ': 'chu', 'チョ': 'cho',
    'ニャ': 'nya', 'ニュ': 'nyu', 'ニョ': 'nyo',
    'ヒャ': 'hya', 'ヒュ': 'hyu', 'ヒョ': 'hyo',
    'ミャ': 'mya', 'ミュ': 'myu', 'ミョ': 'myo',
    'リャ': 'rya', 'リュ': 'ryu', 'リョ': 'ryo',
    'ギャ': 'gya', 'ギュ': 'gyu', 'ギョ': 'gyo',
    'ジャ': 'ja', 'ジュ': 'ju', 'ジョ': 'jo',
    'ビャ': 'bya', 'ビュ': 'byu', 'ビョ': 'byo',
    'ピャ': 'pya', 'ピュ': 'pyu', 'ピョ': 'pyo',
    'ファ': 'fa', 'フィ': 'fi', 'フェ': 'fe', 'フォ': 'fo',
    'ヴァ': 'va', 'ヴィ': 'vi', 'ヴ': 'vu', 'ヴェ': 've', 'ヴォ': 'vo',
    'ティ': 'ti', 'ディ': 'di', 'トゥ': 'tu', 'ドゥ': 'du',
    'チェ': 'che', 'ジェ': 'je', 'シェ': 'she',
    'ウィ': 'wi', 'ウェ': 'we', 'ウォ': 'wo',
    'クァ': 'kwa', 'グァ': 'gwa',
    'ヂャ': 'dya', 'ヂュ': 'dyu', 'ヂョ': 'dyo', 'ヂェ': 'dye', 'ヂィ': 'dyi',
    'フュ': 'fyu', 'テュ': 'tyu', 'デュ': 'dyu',
    'ァ': 'a', 'ィ': 'i', 'ゥ': 'u', 'ェ': 'e', 'ォ': 'o',
    'ャ': 'ya', 'ュ': 'yu', 'ョ': 'yo',
    'ッ': 'cl',
}

VOWEL_TO_KATA = {
    'a': 'ア', 'i': 'イ', 'u': 'ウ', 'e': 'エ', 'o': 'オ',
}

_VOWEL_HIRA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}
_SMALL_KANA = set("ゃゅょャュョぁぃぅぇぉァィゥェォ")

class JaG2p:
    number_map = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        "０": "零", "１": "一", "２": "二", "３": "三", "４": "四",
        "５": "五", "６": "六", "７": "七", "８": "八", "９": "九",
        "〇": "零",
    }

    def __init__(self, engine: str = "kashi-g2p-onnx", model_dir=None):
        self.engine = str(engine or "kashi-g2p-onnx").strip().lower()
        self.model_dir = model_dir

    @staticmethod
    def _katakana_to_hiragana(text):
        result = []
        for char in text:
            code = ord(char)
            if 0x30A1 <= code <= 0x30F6:
                result.append(chr(code - 0x60))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def _hiragana_to_katakana(text):
        result = []
        for char in text:
            code = ord(char)
            if 0x3041 <= code <= 0x3096:
                result.append(chr(code + 0x60))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def _normalize_text(text):
        return re.sub(r'\s+', ' ', str(text or '')).strip()

    @staticmethod
    def split_input_string_no_regex(input_str):
        result = []
        position = 0
        while position < len(input_str):
            current_char = input_str[position]
            if is_letter(current_char) or is_special_letter(current_char):
                start = position
                while position < len(input_str) and (
                    is_letter(input_str[position]) or is_special_letter(input_str[position])
                ):
                    position += 1
                result.append(input_str[start:position])
            elif is_numeric_like(current_char):
                start = position
                while position < len(input_str) and is_numeric_like(input_str[position]):
                    position += 1
                result.append(input_str[start:position])
            elif is_japanese_char(current_char):
                start = position
                while position < len(input_str) and is_japanese_char(input_str[position]):
                    position += 1
                result.append(input_str[start:position])
            else:
                position += 1
        return result

    @staticmethod
    def _split_japanese_segment(segment):
        result = []
        position = 0
        while position < len(segment):
            current_char = segment[position]
            if is_kana(current_char):
                length = 2 if position + 1 < len(segment) and is_special_kana(segment[position + 1]) else 1
                if position + length < len(segment) and segment[position + length] == 'ー':
                    length += 1
                result.append(segment[position:position + length])
                position += length
            else:
                result.append(current_char)
                position += 1
        return result

    @classmethod
    def _fallback_entry(cls, token):
        normalized = cls._normalize_text(token)
        if not normalized:
            return []
        if all(is_letter(ch) or is_special_letter(ch) for ch in normalized):
            lowered = normalized.lower()
            return [{"orig": token, "moras": [lowered], "kana_moras": [lowered]}]
        kana_token = cls._katakana_to_hiragana(normalized) if any(is_kana(ch) for ch in normalized) else normalized
        return [{"orig": token, "moras": [normalized], "kana_moras": [kana_token]}]

    def _parse_pron_to_entry(self, original, pron):
        cleaned_pron = self._normalize_text(str(pron or '').replace("’", ""))
        if not cleaned_pron:
            return []
        kata_pron = self._hiragana_to_katakana(cleaned_pron)
        moras = self._kata2moras(kata_pron)
        kana_moras = self._kata2kana_moras(kata_pron)
        if not moras:
            return []
        return [{
            "orig": original,
            "moras": moras,
            "kana_moras": kana_moras,
        }]

    def _analyze_japanese_segment(self, segment):
        normalized_segment = self._normalize_text(segment)
        if not normalized_segment:
            return []

        # kashi-g2p-onnx (LAKE-G2P v5) first when selected: transformer-disambiguated kana
        # readings per word edge, no 775MB UniDic needed.
        if self.engine not in {"pyopenjtalk"}:
            runtime = _get_onnx_runtime(self.model_dir)
            if runtime is not None:
                try:
                    from inference.ja_onnx_g2p import normalize_edge_reading
                    predicted = runtime.predict(normalized_segment)
                    analysis = []
                    prev_surface = None
                    for edge in predicted["edges"]:
                        surface = edge["surface"]
                        reading = edge["reading"]
                        if surface == "ー" or reading == "ー":
                            # A prolonged mark as its own char-level edge repeats
                            # the previous mora's vowel (mirroring what
                            # _kata2mora_pairs does for ー inside a single
                            # reading); an orphan ー with no vowel before it is
                            # dropped since it has no pronounceable form.
                            if analysis and analysis[-1]["moras"]:
                                last_romaji = analysis[-1]["moras"][-1]
                                if last_romaji[-1:] in _VOWEL_HIRA:
                                    analysis[-1]["moras"].append(last_romaji[-1])
                                    analysis[-1]["kana_moras"].append(_VOWEL_HIRA[last_romaji[-1]])
                            continue
                        reading = normalize_edge_reading(surface, reading, prev_surface=prev_surface)
                        analysis.extend(self._parse_pron_to_entry(surface, reading))
                        prev_surface = surface
                    if analysis:
                        return analysis
                except Exception as e:
                    logger.warning(f"JaG2p ONNX backend failed ({e}); falling back to pyopenjtalk")

        try:
            import pyopenjtalk
            words_info = pyopenjtalk.run_frontend(normalized_segment)
        except Exception:
            words_info = []

        analysis = []
        for word in words_info:
            w_str = self._normalize_text(word.get('string', ''))
            if not w_str or set(w_str).issubset({',', '.', '!', '?', '、', '。', ' ', '　', '’'}):
                continue

            entry = self._parse_pron_to_entry(w_str, word.get('pron', ''))
            if entry:
                analysis.extend(entry)
            else:
                analysis.extend(self._fallback_entry(w_str))

        if analysis:
            return analysis

        smaller_tokens = self._split_japanese_segment(normalized_segment)
        if len(smaller_tokens) > 1:
            fallback_analysis = []
            for token in smaller_tokens:
                fallback_analysis.extend(self._analyze_token(token, convert_number=False))
            if fallback_analysis:
                return fallback_analysis

        if any(is_kana(ch) for ch in normalized_segment):
            direct_entry = self._parse_pron_to_entry(normalized_segment, normalized_segment)
            if direct_entry:
                return direct_entry

        return self._fallback_entry(normalized_segment)

    def _analyze_token(self, token, convert_number=True):
        normalized_token = self._normalize_text(token)
        if not normalized_token:
            return []

        if all(ch in self.number_map for ch in normalized_token):
            if convert_number:
                if self.engine not in {"pyopenjtalk"} and _get_onnx_runtime(self.model_dir) is not None:
                    # The model's number rules read digit strings natively
                    # (12年 -> じゅうにねん); keep digits unconverted.
                    return self._analyze_japanese_segment(normalized_token)
                mapped = ''.join(self.number_map.get(ch, ch) for ch in normalized_token)
                return self._analyze_japanese_segment(mapped)
            return self._fallback_entry(normalized_token)

        if all(is_letter(ch) or is_special_letter(ch) for ch in normalized_token):
            return self._fallback_entry(normalized_token)

        if any(is_japanese_char(ch) for ch in normalized_token):
            return self._analyze_japanese_segment(normalized_token)

        return self._fallback_entry(normalized_token)

    def _kata2mora_pairs(self, kata_str):
        pairs = []
        i = 0
        while i < len(kata_str):
            if i + 1 < len(kata_str) and kata_str[i:i+2] in KATA_TO_ROMAJI:
                token = kata_str[i:i+2]
                pairs.append((token, KATA_TO_ROMAJI[token]))
                i += 2
            elif kata_str[i] in KATA_TO_ROMAJI:
                token = kata_str[i]
                pairs.append((token, KATA_TO_ROMAJI[token]))
                i += 1
            elif kata_str[i] == 'ー':
                if pairs:
                    last_mora = pairs[-1][1]
                    if last_mora != 'cl' and last_mora != 'n':
                        vowel = last_mora[-1]
                        pairs.append((VOWEL_TO_KATA.get(vowel, 'ー'), vowel))
                i += 1
            else:
                char = kata_str[i]
                if char.isalpha():
                    pairs.append((char.lower(), char.lower()))
                i += 1
        return pairs

    def _kata2moras(self, kata_str):
        return [romaji for _, romaji in self._kata2mora_pairs(kata_str)]

    def _kata2kana_moras(self, kata_str):
        return [self._katakana_to_hiragana(kana) for kana, _ in self._kata2mora_pairs(kata_str)]

    def _get_analysis(self, text):
        normalized_text = self._normalize_text(text)
        analysis = []

        for token in self.split_input_string_no_regex(normalized_text):
            analysis.extend(self._analyze_token(token))

        return analysis

    def convert(self, text: str, include_tone: bool = False, convert_number: bool = True) -> str:
        """
        Convert Japanese text to romaji, preserving spacing where possible, 
        and joining by spaces for HubertFA dictionary matching.
        """
        return self.convert_list(
            self.split_input_string_no_regex(text),
            include_tone=include_tone,
            convert_number=convert_number,
        )

    def convert_list(self, input_list, include_tone: bool = False, convert_number: bool = True) -> str:
        analysis = []
        for token in input_list:
            analysis.extend(self._analyze_token(token, convert_number=convert_number))

        romaji_list = []
        for item in analysis:
            romaji_list.extend(item["moras"])
        return " ".join(romaji_list)

    def analyze_lyric_text(self, text, convert_number: bool = False) -> tuple[list[str], list[str]]:
        """Kana moras + index-aligned romaji moras from one analysis pass.

        Both lists are derived from the same mora pairs, so they stay 1:1 even
        where one token expands to several moras (ICBM -> a i shi i bi i e mu):
        the lyric matcher indexes the kana list by phonetic position, and a
        length mismatch silently desynchronizes every later token. Digraph
        halves from char-level kana edges are re-merged via the katakana table
        (し+ょ -> しょ/sho, not shi+yo), prolonged marks repeat the preceding
        vowel (ラーメン -> ra i me n), particle は/へ keep the orthographic
        reading for standalone kana edges (the mora ASR emits orthographic ha
        too, and a char-level は is more often word-internal -- はんせん --
        than a particle), and latin words resolve through the model's alnum
        table instead of leaking a raw non-phoneme token into the .lab.
        """
        pairs = []
        for token in self.split_input_string_no_regex(self._normalize_text(text)):
            entries = None
            if len(token) > 1 and all(is_letter(ch) or is_special_letter(ch) for ch in token):
                # Latin runs read through the G2P model's alnum table
                # (ICBM -> あいしーびーえむ); unknown words fall back to the
                # legacy passthrough below.
                entries = self._analyze_japanese_segment(token)
            if not entries:
                entries = self._analyze_token(token, convert_number=convert_number)
            for item in entries:
                kana = item.get("kana_moras") or []
                moras = item.get("moras") or []
                if len(kana) == len(moras):
                    pairs.extend(zip(kana, moras))
        merged = _merge_digraph_pairs(pairs)
        return [kana for kana, _romaji in merged], [romaji for _kana, romaji in merged]

    def split_string_no_regex(self, text: str) -> list[str]:
        """
        Splits the text into characters that match the number of converted romaji tokens.
        For Japanese, since mapping multi-mora words (e.g. Kanji) to individual notes 
        is ambiguous without proper grapheme-to-phoneme tokenization, we simply return
        the romaji (moras) themselves as the final lyrics.
        """
        analysis = self._get_analysis(text)
        chars = []
        
        for item in analysis:
            moras = item["moras"]
            # Just return the romaji directly as the lyric text
            chars.extend(moras)
                    
        return chars

    def split_kana_no_regex(self, text: str) -> list[str]:
        analysis = self._get_analysis(text)
        chars = []

        for item in analysis:
            chars.extend(item.get("kana_moras", []))

        return chars


def _merge_digraph_pairs(pairs):
    """Re-merge digraph halves and expand lone ー marks in a flattened
    (kana, romaji) mora pair sequence.

    Char-level kana edges arrive as separate moras (し+ょ -> shi+yo); the
    lyric matcher consumes whole moras, so adjacent halves are combined via
    the katakana table (しょ -> sho) -- concatenating the romaji halves would
    give the wrong mora ("shiyo"). A ー pair repeats the preceding vowel
    (ラ + ー -> ら + い / ra + i); an orphan ー with no vowel before it is
    dropped since it has no pronounceable form.
    """
    merged = []
    for kana, romaji in pairs:
        if merged and kana in _SMALL_KANA:
            prev_kana = merged[-1][0]
            combined = KATA_TO_ROMAJI.get(JaG2p._hiragana_to_katakana(prev_kana + kana))
            if combined is not None:
                merged[-1] = (prev_kana + kana, combined)
                continue
        if kana == "ー":
            prev_romaji = merged[-1][1] if merged else ""
            if prev_romaji[-1:] in _VOWEL_HIRA:
                merged.append((_VOWEL_HIRA[prev_romaji[-1]], prev_romaji[-1]))
            continue
        merged.append((kana, romaji))
    return merged

if __name__ == "__main__":
    g2p = JaG2p()
    text = "きょうはいい天気ですね。My way"
    logger.info("Original:", text)
    logger.info("Pinyin/Romaji string:", g2p.convert(text))
    logger.info("Split chars:", g2p.split_string_no_regex(text))
