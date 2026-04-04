import pyopenjtalk
import re

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

class JaG2p:
    def __init__(self):
        pass

    def _kata2moras(self, kata_str):
        moras = []
        i = 0
        while i < len(kata_str):
            if i + 1 < len(kata_str) and kata_str[i:i+2] in KATA_TO_ROMAJI:
                moras.append(KATA_TO_ROMAJI[kata_str[i:i+2]])
                i += 2
            elif kata_str[i] in KATA_TO_ROMAJI:
                moras.append(KATA_TO_ROMAJI[kata_str[i]])
                i += 1
            elif kata_str[i] == 'ー':
                if moras:
                    last_mora = moras[-1]
                    if last_mora != 'cl' and last_mora != 'n':
                        vowel = last_mora[-1]
                        moras.append(vowel)
                i += 1
            else:
                # Fallback for anything else (e.g. English chars converted by pyopenjtalk to fullwidth or left as is)
                char = kata_str[i]
                if char.isalpha():
                    moras.append(char.lower())
                i += 1
        return moras

    def _get_analysis(self, text):
        words_info = pyopenjtalk.run_frontend(text)
        analysis = []
        
        for word in words_info:
            w_str = word.get('string', '')
            if not w_str.strip() or set(w_str).issubset({',', '.', '!', '?', '、', '。', ' ', '　', '’'}):
                continue
                
            kana_pron = word.get('pron', w_str)
            # Remove any special marks like '’'
            kana_pron = kana_pron.replace("’", "")
            
            moras = self._kata2moras(kana_pron)
            if moras:
                analysis.append({
                    "orig": w_str,
                    "moras": moras
                })
        return analysis

    def convert(self, text: str, include_tone: bool = False, convert_number: bool = True) -> str:
        """
        Convert Japanese text to romaji, preserving spacing where possible, 
        and joining by spaces for HubertFA dictionary matching.
        """
        analysis = self._get_analysis(text)
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

if __name__ == "__main__":
    g2p = JaG2p()
    text = "きょうはいい天気ですね。My way"
    print("Original:", text)
    print("Pinyin/Romaji string:", g2p.convert(text))
    print("Split chars:", g2p.split_string_no_regex(text))
