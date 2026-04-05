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
        matcher_lang = language if language in ["zh", "en"] else "zh"
        matcher = LyricMatcher(matcher_lang)
        processor = matcher.processor
        cleaned_lyric = processor.clean_text(original_lyrics)
        matcher.lyric_text_list = processor.split_text(cleaned_lyric)
        matcher.lyric_phonetic_list = processor.get_phonetic_list(matcher.lyric_text_list)
    return matcher

def process_asr_to_phonemes(all_results, chunk_indices, temp_dir_path, language, matcher):
    """
    Processes batched ASR text, aligns with original lyrics if available, 
    and generates .lab phoneme files.
    """
    g2p_model = get_ja_g2p() if language == "ja" else get_zh_g2p()
    chars_dict = {}
    chunk_logs = []
    
    for idx, res in enumerate(all_results):
        chunk_idx = chunk_indices[idx]
        stem = f"chunk_{chunk_idx}"
        
        if res is None or not res.text.strip():
            chunk_logs.append(f"[{stem}]\nASR Output: [Empty or Failed]\nStatus: Ignored\n")
            continue

        text = res.text
        match_status = "No original lyrics provided"
        if matcher:
            asr_text_list, asr_phonetic_list = matcher.process_asr_content(text)
            if asr_phonetic_list:
                matched_text, matched_phonetic, _ = matcher.align_lyric_with_asr(
                    asr_phonetic=asr_phonetic_list,
                    lyric_text=matcher.lyric_text_list,
                    lyric_phonetic=matcher.lyric_phonetic_list
                )
                if matched_phonetic:
                    pinyin_str = matched_phonetic
                    chars = matched_text.split()
                    match_status = "Matched with original lyrics"
                else:
                    pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                    chars = g2p_model.split_string_no_regex(text)
                    match_status = "Fallback to ASR (No match found)"
            else:
                pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                chars = g2p_model.split_string_no_regex(text)
                match_status = "Fallback to ASR (No phonetics)"
        else:
            pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
            chars = g2p_model.split_string_no_regex(text)
            match_status = "Direct ASR (No original lyrics)"
            
        if getattr(g2p_model, '__class__', None).__name__ == 'JaG2p':
            chars = pinyin_str.split()
            
        (temp_dir_path / f"{stem}.lab").write_text(pinyin_str, encoding="utf-8")
        chars_dict[stem] = chars
        
        assigned_lyrics = ' '.join(chars) if getattr(g2p_model, '__class__', None).__name__ == 'JaG2p' else ''.join(chars)
        chunk_logs.append(
            f"[{stem}]\nASR Output: {text}\n"
            f"Match Status: {match_status}\n"
            f"Final Assigned Lyrics: {assigned_lyrics}\n"
            f"FA Pinyin (.lab): {pinyin_str}\n"
        )
            
    return chars_dict, chunk_logs
