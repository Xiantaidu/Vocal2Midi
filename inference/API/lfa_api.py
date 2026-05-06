from inference.LyricFA.tools.ZhG2p import ZhG2p
from inference.LyricFA.tools.JaG2p import JaG2p
from inference.LyricFA.tools.lyric_matcher import LyricMatcher

_zh_g2p = None
_ja_g2p = None


def _normalize_asr_phoneme_token(token: str) -> str:
    t = (token or "").strip()
    if not t:
        return ""
    low = t.lower()
    # HFA phoneme mode uses SP as silence separator; 'sil' is not a vocab token.
    if low in {"sil", "sp", "pau", "br"}:
        return "SP"
    if low == "ap":
        return "AP"
    if low == "ep":
        return "EP"
    if low in {"<blank>", "pad", "<unk>", "unk"}:
        return ""
    return t


def _phoneme_tokens_to_romaji_moras(tokens):
    """Convert phoneme-ASR token stream to romaji-mora tokens for HFA dictionary mode."""
    if not tokens:
        return []

    vowel_map = {"a": "a", "i": "i", "u": "u", "e": "e", "o": "o", "I": "i", "U": "u"}
    special_join = {
        ("sh", "a"): "sha", ("sh", "i"): "shi", ("sh", "u"): "shu", ("sh", "e"): "she", ("sh", "o"): "sho",
        ("ch", "a"): "cha", ("ch", "i"): "chi", ("ch", "u"): "chu", ("ch", "e"): "che", ("ch", "o"): "cho",
        ("j", "a"): "ja", ("j", "i"): "ji", ("j", "u"): "ju", ("j", "e"): "je", ("j", "o"): "jo",
        ("ts", "u"): "tsu",
        # Canonical Japanese romaji for CV combinations that should not surface as
        # raw phoneme concatenations like "nyi / ryi / hyi / myi".
        ("k", "i"): "ki", ("g", "i"): "gi",
        ("s", "i"): "shi", ("z", "i"): "ji",
        ("t", "i"): "ti", ("d", "i"): "di",
        ("n", "i"): "ni", ("h", "i"): "hi",
        ("b", "i"): "bi", ("p", "i"): "pi",
        ("m", "i"): "mi", ("r", "i"): "ri",
        ("f", "u"): "fu",
        # Palatal-series + i are usually represented by their base i-row kana.
        ("ky", "i"): "ki", ("gy", "i"): "gi",
        ("ny", "i"): "ni", ("hy", "i"): "hi",
        ("my", "i"): "mi", ("ry", "i"): "ri",
        ("by", "i"): "bi", ("py", "i"): "pi",
        # Less common but useful canonicalizations.
        ("ty", "i"): "chi", ("ty", "u"): "chu", ("ty", "o"): "cho",
        ("dy", "i"): "ji", ("dy", "u"): "ju", ("dy", "o"): "jo",
    }
    consonants = {
        "b", "by", "ch", "d", "dy", "f", "fy", "g", "gw", "gy", "h", "hy", "j", "k", "kw", "ky",
        "m", "my", "n", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts", "ty", "v", "w", "y", "z"
    }

    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "SP":
            i += 1
            continue
        if t in {"AP", "EP"}:
            out.append(t)
            i += 1
            continue
        if t == "N":
            out.append("n")
            i += 1
            continue
        if t == "cl":
            # Keep as standalone mora-like pause marker
            out.append("cl")
            i += 1
            continue
        if t in vowel_map:
            out.append(vowel_map[t])
            i += 1
            continue

        if i + 1 < len(tokens) and tokens[i + 1] in vowel_map and t in consonants:
            v_raw = tokens[i + 1]
            v = vowel_map[v_raw]
            out.append(special_join.get((t, v), f"{t}{v}"))
            i += 2
            continue

        out.append(t.lower())
        i += 1

    return out


def _align_direct_phoneme_moras_with_matcher(language, lyric_output_mode, matcher, romaji_moras):
    """Try to snap direct phoneme-ASR mora output to the provided reference lyrics."""
    if not matcher or not romaji_moras:
        return None

    # AP/EP are not part of lyric pronunciations and hurt sequence matching.
    asr_phonetic = [t for t in romaji_moras if t not in {"AP", "EP"}]
    if not asr_phonetic:
        return None

    matched_text, matched_phonetic, reason = matcher.align_lyric_with_asr(
        asr_phonetic=asr_phonetic,
        lyric_text=matcher.lyric_text_list,
        lyric_phonetic=matcher.lyric_phonetic_list,
    )
    if not matched_phonetic:
        return None

    chars = _select_matched_display_tokens(language, lyric_output_mode, matched_text, matched_phonetic)
    return {
        "matched_text": matched_text,
        "matched_phonetic": matched_phonetic,
        "reason": reason or "",
        "chars": chars,
    }


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

def process_asr_to_phonemes(
    all_results,
    chunk_indices,
    temp_dir_path,
    language,
    matcher,
    lyric_output_mode=None,
    use_asr_phonemes=False,
    write_asr_phoneme_lab=False,
):
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
        direct_phoneme_tokens = []
        if use_asr_phonemes and isinstance(res, dict):
            tokens = res.get("phonemes")
            if isinstance(tokens, list):
                direct_phoneme_tokens = []
                for t in tokens:
                    nt = _normalize_asr_phoneme_token(str(t))
                    if nt:
                        direct_phoneme_tokens.append(nt)
        if not text.strip():
            chunk_logs.append(f"[{stem}]\nASR Output: [Empty or Failed]\nStatus: Ignored\n")
            continue

        match_status = "No original lyrics provided"
        if direct_phoneme_tokens:
            romaji_moras = _phoneme_tokens_to_romaji_moras(direct_phoneme_tokens)
            matched_direct = _align_direct_phoneme_moras_with_matcher(
                language,
                lyric_output_mode,
                matcher,
                romaji_moras,
            )
            if matched_direct:
                pinyin_str = matched_direct["matched_phonetic"]
                chars = matched_direct["chars"]
                matched_lyric_text = matched_direct["matched_text"]
                matched_lyric_phonetic = matched_direct["matched_phonetic"]
                match_reason = matched_direct["reason"]
                match_status = "Direct phoneme ASR -> Matched original lyrics"
            else:
                if write_asr_phoneme_lab:
                    pinyin_str = " ".join(direct_phoneme_tokens)
                    match_status = "Direct phoneme ASR -> Raw phoneme lab"
                else:
                    pinyin_str = " ".join(romaji_moras or direct_phoneme_tokens)
                    match_status = "Direct phoneme ASR -> Romaji moras"
                chars = romaji_moras or direct_phoneme_tokens
        elif matcher:
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
