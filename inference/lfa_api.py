import pathlib
import sys

# Add vendor paths
VENDOR_DIR = pathlib.Path(__file__).parent / "vendor"
if str(VENDOR_DIR / "HubertFA") not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR / "HubertFA"))

from inference.vendor.LyricFA.tools.ZhG2p import ZhG2p
from inference.vendor.LyricFA.tools.JaG2p import JaG2p
from inference.vendor.LyricFA.tools.lyric_matcher import LyricMatcher

_zh_g2p = None
_ja_g2p = None


def _normalize_lyric_output_mode(language, lyric_output_mode):
    language = (language or "zh").lower()
    mode = (lyric_output_mode or "").lower()
    aliases = {
        "拼音": "pinyin",
        "汉字": "hanzi",
        "罗马音": "romaji",
        "假名": "kana",
    }
    mode = aliases.get(lyric_output_mode, mode)
    valid_modes = {
        "zh": {"pinyin", "hanzi"},
        "ja": {"romaji", "kana"},
    }
    defaults = {"zh": "hanzi", "ja": "romaji"}
    return mode if mode in valid_modes.get(language, set()) else defaults.get(language, "hanzi")


def _build_display_tokens(text, language, lyric_output_mode, g2p_model):
    mode = _normalize_lyric_output_mode(language, lyric_output_mode)
    if language == "ja":
        if mode == "kana" and hasattr(g2p_model, "split_kana_no_regex"):
            return g2p_model.split_kana_no_regex(text)
        return g2p_model.convert(text, include_tone=False, convert_number=True).split()

    if mode == "pinyin":
        return g2p_model.convert(text, include_tone=False, convert_number=True).split()
    return g2p_model.split_string_no_regex(text)


def _select_matched_display_tokens(language, lyric_output_mode, matched_text, matched_phonetic):
    mode = _normalize_lyric_output_mode(language, lyric_output_mode)
    phonetic_mode = (language == "zh" and mode == "pinyin") or (language == "ja" and mode == "romaji")
    source = matched_phonetic if phonetic_mode else matched_text
    return source.split() if source else []


def _join_display_tokens(language, lyric_output_mode, tokens):
    mode = _normalize_lyric_output_mode(language, lyric_output_mode)
    if (language == "zh" and mode == "pinyin") or (language == "ja" and mode == "romaji"):
        return " ".join(tokens)
    return "".join(tokens)

def get_zh_g2p():
    global _zh_g2p
    if _zh_g2p is None:
        _zh_g2p = ZhG2p("mandarin")
    return _zh_g2p

def get_ja_g2p():
    global _ja_g2p
    if _ja_g2p is None:
        _ja_g2p = JaG2p()
    return _ja_g2p

def create_lyric_matcher(language, original_lyrics):
    """
    Creates and initializes a LyricMatcher if original_lyrics is provided.
    """
    matcher = None
    if original_lyrics and original_lyrics.strip():
        matcher_lang = language if language in ["zh", "en", "ja"] else "zh"
        matcher = LyricMatcher(matcher_lang)
        processor = matcher.processor
        cleaned_lyric = processor.clean_text(original_lyrics)
        matcher.lyric_text_list = processor.split_text(cleaned_lyric)
        matcher.lyric_phonetic_list = processor.get_phonetic_list(matcher.lyric_text_list)
    return matcher

def process_asr_to_phonemes(all_results, chunk_indices, temp_dir_path, language, matcher, lyric_output_mode=None):
    """
    Processes batched ASR text, aligns with original lyrics if available, 
    and generates .lab phoneme files.
    """
    g2p_model = get_ja_g2p() if language == "ja" else get_zh_g2p()
    chars_dict = {}
    chunk_logs = []

    def _extract_text(res):
        if res is None:
            return ""
                                    
        text_attr = getattr(res, "text", None)
        if text_attr is not None:
            return str(text_attr)
                              
        if isinstance(res, dict):
            text_val = res.get("text") or res.get("transcript") or ""
            return str(text_val)
            
        return str(res)
    
    for idx, res in enumerate(all_results):
        chunk_idx = chunk_indices[idx]
        stem = f"chunk_{chunk_idx}"
        matched_lyric_text = ""
        matched_lyric_phonetic = ""
        match_reason = ""
        
        text = _extract_text(res)
        if not text.strip():
            chunk_logs.append(f"[{stem}]\nASR Output: [Empty or Failed]\nStatus: Ignored\n")
            continue

        match_status = "No original lyrics provided"
        if matcher:
            asr_text_list, asr_phonetic_list = matcher.process_asr_content(text)
            if asr_phonetic_list:
                matched_text, matched_phonetic, reason = matcher.align_lyric_with_asr(
                    asr_phonetic=asr_phonetic_list,
                    lyric_text=matcher.lyric_text_list,
                    lyric_phonetic=matcher.lyric_phonetic_list
                )
                match_reason = reason or ""
                if matched_phonetic:
                    pinyin_str = matched_phonetic
                    chars = _select_matched_display_tokens(language, lyric_output_mode, matched_text, matched_phonetic)
                    matched_lyric_text = matched_text
                    matched_lyric_phonetic = matched_phonetic
                    match_status = "Matched with original lyrics"
                else:
                    pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                    chars = _build_display_tokens(text, language, lyric_output_mode, g2p_model)
                    match_status = "Fallback to ASR (No match found)"
            else:
                pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                chars = _build_display_tokens(text, language, lyric_output_mode, g2p_model)
                match_status = "Fallback to ASR (No phonetics)"
        else:
            pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
            chars = _build_display_tokens(text, language, lyric_output_mode, g2p_model)
            match_status = "Direct ASR (No original lyrics)"

        (temp_dir_path / f"{stem}.lab").write_text(pinyin_str, encoding="utf-8")
        chars_dict[stem] = chars

        assigned_lyrics = _join_display_tokens(language, lyric_output_mode, chars)
        log_lines = [
            f"[{stem}]",
            f"ASR Output: {text}",
            f"Match Status: {match_status}",
        ]

        if matched_lyric_text:
            log_lines.append(f"Matched Lyric Segment: {matched_lyric_text}")
        if matched_lyric_phonetic:
            log_lines.append(f"Matched Lyric Phonetic: {matched_lyric_phonetic}")
        if match_reason:
            log_lines.append(f"Match Reason: {match_reason}")

        log_lines.extend([
            f"Final Assigned Lyrics: {assigned_lyrics}",
            f"FA Pinyin (.lab): {pinyin_str}",
        ])

        chunk_logs.append("\n".join(log_lines) + "\n")
            
    return chars_dict, chunk_logs
