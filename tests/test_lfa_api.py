from inference.API import lfa_api


class _DummyZhG2p:
    def convert(self, text, include_tone=False, convert_number=True):
        return f"LAB:{text}"

    def split_string_no_regex(self, text):
        return list(text)


def test_create_lyric_matcher_uses_explicit_japanese_reference_chain(monkeypatch):
    calls = []

    class _FakeProcessor:
        def clean_text(self, text):
            calls.append(("clean_text", text))
            return text.strip()

        def split_text(self, text):
            calls.append(("split_text", text))
            return ["old"]

        def get_phonetic_list(self, text_list):
            calls.append(("get_phonetic_list", tuple(text_list)))
            return ["old"]

        def build_reference_lyric(self, text):
            calls.append(("build_reference_lyric", text))
            return ["か", "き"], ["ka", "ki"]

    class _FakeMatcher:
        def __init__(self, language):
            self.language = language
            self.processor = _FakeProcessor()

        def process_lyric_text(self, raw_text):
            cleaned_text = self.processor.clean_text(raw_text)
            text_list, phonetic_list = self.processor.build_reference_lyric(cleaned_text)
            return type(
                "_LyricData",
                (),
                {
                    "text_list": text_list,
                    "phonetic_list": phonetic_list,
                    "raw_text": cleaned_text,
                },
            )()

    monkeypatch.setattr(lfa_api, "LyricMatcher", _FakeMatcher)

    matcher = lfa_api.create_lyric_matcher("ja", " 歌詞 ")

    assert matcher.lyric_text_list == ["か", "き"]
    assert matcher.lyric_phonetic_list == ["ka", "ki"]
    assert calls == [
        ("clean_text", " 歌詞 "),
        ("build_reference_lyric", "歌詞"),
    ]


def test_process_asr_to_phonemes_uses_sanitized_asr_text(monkeypatch, tmp_path):
    monkeypatch.setattr(lfa_api, "get_zh_g2p", lambda: _DummyZhG2p())

    chars_dict, chunk_logs = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "北京欢迎你"}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="zh",
        matcher=None,
        lyric_output_mode="hanzi",
    )

    assert chars_dict == {"chunk_0": list("北京欢迎你")}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "LAB:北京欢迎你"
    assert "ASR Output: 北京欢迎你" in chunk_logs[0]
    assert "Filtered ASR Output:" not in chunk_logs[0]


def test_process_asr_to_phonemes_uses_direct_moras_for_japanese_romaji(tmp_path):
    chars_dict, _ = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "k a k i", "phonemes": ["k", "a", "k", "i"]}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="ja",
        matcher=None,
        lyric_output_mode="romaji",
        use_asr_phonemes=True,
    )

    assert chars_dict == {"chunk_0": ["ka", "ki"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "ka ki"


def test_process_asr_to_phonemes_converts_direct_moras_to_kana(tmp_path):
    chars_dict, _ = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "k a k i", "phonemes": ["k", "a", "k", "i"]}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="ja",
        matcher=None,
        lyric_output_mode="kana",
        use_asr_phonemes=True,
    )

    assert chars_dict == {"chunk_0": ["か", "き"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "ka ki"


def test_process_asr_to_phonemes_en_uses_words_for_lab(tmp_path):
    chars_dict, chunk_logs = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "Hello, World! I don't know."}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="en",
        matcher=None,
        lyric_output_mode="word",
    )

    assert chars_dict == {"chunk_0": ["hello", "world", "i", "don't", "know"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "hello world i don't know"
    assert "Direct ASR (No original lyrics)" in chunk_logs[0]


def test_process_asr_to_phonemes_en_matches_reference_lyrics(tmp_path):
    matcher = lfa_api.create_lyric_matcher("en", "Hello world, how are you?")

    chars_dict, _ = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "hello world how are you"}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="en",
        matcher=matcher,
        lyric_output_mode="word",
    )

    assert chars_dict == {"chunk_0": ["hello", "world", "how", "are", "you"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "hello world how are you"


class _FakeJaMatcher:
    """Matcher double whose matched phonetics differ from the ASR's reading."""

    def __init__(self, matched: bool):
        self.lyric_text_list = ["か", "ん", "た", "ん"]
        self.lyric_phonetic_list = ["ka", "n", "ta", "n"]
        self._matched = matched

    def align_lyric_with_asr(self, asr_phonetic, lyric_text, lyric_phonetic):
        if not self._matched:
            return "", "", "no matching window found"
        return (
            " ".join(self.lyric_text_list),
            " ".join(self.lyric_phonetic_list),
            "",
        )


def test_process_asr_to_phonemes_feeds_tifa_the_matched_phonetics(tmp_path):
    """chunk_N.txt must carry the matched phonetics: TiFA reads .txt (not
    .lab), so leaving the raw ASR text there meant the reference reading
    never reached the notes under the TiFA engine."""
    matcher = _FakeJaMatcher(matched=True)

    chars_dict, chunk_logs = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "wa n se n", "phonemes": ["w", "a", "N", "s", "e", "N"]}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="ja",
        matcher=matcher,
        lyric_output_mode="romaji",
        use_asr_phonemes=True,
    )

    assert chars_dict == {"chunk_0": ["ka", "n", "ta", "n"]}
    assert (tmp_path / "chunk_0.lab").read_text(encoding="utf-8") == "ka n ta n"
    assert (tmp_path / "chunk_0.txt").read_text(encoding="utf-8") == "ka n ta n"
    assert "Matched original lyrics" in chunk_logs[0]


def test_process_asr_to_phonemes_keeps_raw_text_for_tifa_without_match(tmp_path):
    """Without a match, TiFA keeps the raw ASR text (hanzi polyphone value)."""
    matcher = _FakeJaMatcher(matched=False)

    _, chunk_logs = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "wa n se n", "phonemes": ["w", "a", "N", "s", "e", "N"]}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="ja",
        matcher=matcher,
        lyric_output_mode="romaji",
        use_asr_phonemes=True,
    )

    assert (tmp_path / "chunk_0.txt").read_text(encoding="utf-8") == "wa n se n"
    assert "Matched original lyrics" not in chunk_logs[0]


def test_normalize_lyric_output_mode_en_defaults_to_word():
    assert lfa_api._normalize_lyric_output_mode("en", None) == "word"
    assert lfa_api._normalize_lyric_output_mode("en", "romaji") == "word"


def test_kata_to_romaji_covers_dakuten_digraphs():
    """The ONNX reading backend emits ぢ-row digraphs and standalone small
    kana; HFA's romaji dictionary must resolve every resulting mora."""
    from inference.LyricFA.tools.JaG2p import KATA_TO_ROMAJI

    for kata, romaji in (
        ("ヂャ", "dya"), ("ヂュ", "dyu"), ("ヂョ", "dyo"),
        ("ヂェ", "dye"), ("ヂィ", "dyi"), ("フュ", "fyu"),
        ("テュ", "tyu"), ("デュ", "dyu"), ("ァ", "a"), ("ョ", "yo"),
    ):
        assert KATA_TO_ROMAJI.get(kata) == romaji, kata


def test_normalize_lyric_output_mode_yue():
    assert lfa_api._normalize_lyric_output_mode("yue", None) == "hanzi"
    assert lfa_api._normalize_lyric_output_mode("yue", "jyutping") == "jyutping"
    assert lfa_api._normalize_lyric_output_mode("yue", "hanzi") == "hanzi"
    assert lfa_api._normalize_lyric_output_mode("yue", "invalid") == "hanzi"


def test_cantonese_lyric_matcher():
    matcher = lfa_api.create_lyric_matcher("yue", "海阔天空")
    assert matcher is not None
    assert matcher.lyric_text_list == ["海", "阔", "天", "空"]
    assert len(matcher.lyric_phonetic_list) == 4
    assert matcher.lyric_phonetic_list[0] == "hoi"


def test_process_asr_to_phonemes_cantonese(tmp_path):
    chars_dict, chunk_logs = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "海阔天空"}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="yue",
        matcher=None,
        lyric_output_mode="hanzi",
    )
    assert chars_dict == {"chunk_0": ["海", "阔", "天", "空"]}
    assert (tmp_path / "chunk_0.txt").read_text(encoding="utf-8") == "海阔天空"
    lab_text = (tmp_path / "chunk_0.lab").read_text(encoding="utf-8")
    assert "hoi" in lab_text and "tin" in lab_text


def test_cantonese_g2p_phrase_handling(tmp_path):
    chars_dict, _ = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "斗转星移 唔该"}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="yue",
        matcher=None,
        lyric_output_mode="jyutping",
    )
    assert chars_dict["chunk_0"] == ["dau", "zyun", "sing", "ji", "m", "goi"]
    lab_text = (tmp_path / "chunk_0.lab").read_text(encoding="utf-8")
    assert lab_text == "dau zyun sing ji m goi"


def test_cantonese_process_asr_to_phonemes_simplifies_traditional_text(tmp_path):
    chars_dict, _ = lfa_api.process_asr_to_phonemes(
        all_results=[{"text": "冷風偏偏吹雪"}],
        chunk_indices=[0],
        temp_dir_path=tmp_path,
        language="yue",
        matcher=None,
        lyric_output_mode="hanzi",
    )
    assert chars_dict["chunk_0"] == ["冷", "风", "偏", "偏", "吹", "雪"]
    assert (tmp_path / "chunk_0.txt").read_text(encoding="utf-8") == "冷风偏偏吹雪"


