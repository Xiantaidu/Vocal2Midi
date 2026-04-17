import pyopenjtalk
import re


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
    'ッ': 'cl',
}

VOWEL_TO_KATA = {
    'a': 'ア', 'i': 'イ', 'u': 'ウ', 'e': 'エ', 'o': 'オ',
}

class JaG2p:
    number_map = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        "０": "零", "１": "一", "２": "二", "３": "三", "４": "四",
        "５": "五", "６": "六", "７": "七", "８": "八", "９": "九",
        "〇": "零",
    }

    def __init__(self):
        pass

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

        try:
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

if __name__ == "__main__":
    g2p = JaG2p()
    text = "きょうはいい天気ですね。My way"
    print("Original:", text)
    print("Pinyin/Romaji string:", g2p.convert(text))
    print("Split chars:", g2p.split_string_no_regex(text))
