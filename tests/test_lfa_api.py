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
