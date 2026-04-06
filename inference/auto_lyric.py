import os
import pathlib
import tempfile
import librosa
import numpy as np
import warnings
import sys
import itertools

# Legacy pipeline note:
# - This module is kept for backward compatibility (ONNX/FunASR old path).
# - Recommended main pipeline is inference/auto_lyric_hybrid.py.

# Allow running this script directly from anywhere
ROOT_DIR = pathlib.Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from funasr import AutoModel

from inference.slicer2 import Slicer
from inference.onnx_api import pad_1d_arrays, NoteInfo, quantize_notes, _save_midi, _save_text
from inference.utils import align_notes_to_words

# Add vendor paths
VENDOR_DIR = pathlib.Path(__file__).parent / "vendor"
# Only insert HubertFA to sys.path because it uses absolute imports like `from tools...`
sys.path.insert(0, str(VENDOR_DIR / "HubertFA"))

from inference.vendor.LyricFA.tools.ZhG2p import ZhG2p
from inference.vendor.LyricFA.tools.JaG2p import JaG2p
from inference.vendor.LyricFA.tools.lyric_matcher import LyricMatcher
from onnx_infer import InferenceOnnx

_zh_g2p = None
_ja_g2p = None

def free_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def get_funasr_model(model_path=None):
    print("Loading FunASR model...")
    import logging
    logging.getLogger("funasr").setLevel(logging.ERROR)
    
    if not model_path or not str(model_path).strip():
        model_path = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        
    model = AutoModel(
        model=model_path,
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
        disable_pbar=True
    )
    return model

def get_hfa_model(onnx_path: str):
    print("Loading HubertFA ONNX model...")
    model = InferenceOnnx(onnx_path=pathlib.Path(onnx_path))
    model.load_config()
    model.init_decoder()
    model.load_model()
    return model

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

def extract_vowel_boundaries(result_word, original_chars: list[str]):
    word_durs = []
    word_vuvs = []
    lyrics = []
    
    char_idx = 0
    last_end = 0.0
    
    ignore_tokens = {"SP", "AP", "EP", "br", "sil", "pau"}
    is_romaji = len(original_chars) > 0 and all(c.isascii() or c == '' for c in original_chars)
    
    for i, word in enumerate(result_word):
        if word.text in ignore_tokens:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                last_end = word.end
            continue
            
        # Normal word: vowel start
        vowel_start = word.phonemes[-1].start if len(word.phonemes) > 0 else word.start
        
        # Gap before vowel start -> treat as rest
        if vowel_start > last_end + 0.005:
            word_durs.append(vowel_start - last_end)
            word_vuvs.append(0)
            lyrics.append("")
        elif vowel_start < last_end:
            vowel_start = last_end # Clamp
            
        # Next boundary is next vowel start or word end
        next_vowel_start = word.end
        if i + 1 < len(result_word):
            next_w = result_word[i+1]
            if next_w.text not in ignore_tokens:
                next_vowel_start = next_w.phonemes[-1].start if len(next_w.phonemes) > 0 else next_w.start
                
        note_end = next_vowel_start
        if i + 1 < len(result_word) and result_word[i+1].text in ignore_tokens:
            note_end = word.end
            
        dur = note_end - vowel_start
        if dur < 0: dur = 0.0
        
        word_durs.append(dur)
        word_vuvs.append(1)
        
        if is_romaji:
            while char_idx < len(original_chars) and original_chars[char_idx].lower() != word.text.lower():
                char_idx += 1
            if char_idx < len(original_chars):
                lyrics.append(original_chars[char_idx])
                char_idx += 1
            else:
                lyrics.append(word.text)
        else:
            if char_idx < len(original_chars):
                lyrics.append(original_chars[char_idx])
                char_idx += 1
            else:
                lyrics.append(word.text)
            
        last_end = note_end

    return word_durs, word_vuvs, lyrics

def get_rms_db(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Calculate RMS energy in decibels."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    rms_db = 20 * np.log10(np.clip(rms, a_min=1e-10, a_max=None))
    return rms_db

def _sliding_window_split(
    waveform: np.ndarray,
    sr: int,
    min_len_sec: float,
    max_len_sec: float,
    target_threshold_db: float,
    frame_length: int,
    hop_length: int,
):
    """
    Internal helper for splitting a single, long audio segment.
    This function assumes the input has already been cleared of long silences.
    """
    total_samples = waveform.shape[-1]
    total_sec = total_samples / sr
    
    if total_sec <= max_len_sec:
        return [{'offset': 0.0, 'waveform': waveform}]

    rms_db = get_rms_db(waveform, frame_length=frame_length, hop_length=hop_length)
    
    chunks = []
    current_start_sec = 0.0
    
    while current_start_sec < total_sec:
        window_start_sec = current_start_sec + min_len_sec
        window_end_sec = current_start_sec + max_len_sec
        
        if window_end_sec >= total_sec:
            start_sample = int(current_start_sec * sr)
            chunk_wav = waveform[:, start_sample:] if waveform.ndim > 1 else waveform[start_sample:]
            chunks.append({'offset': current_start_sec, 'waveform': chunk_wav})
            break
            
        start_frame = librosa.time_to_frames(window_start_sec, sr=sr, hop_length=hop_length)
        end_frame = librosa.time_to_frames(window_end_sec, sr=sr, hop_length=hop_length)
        
        start_frame = max(0, min(start_frame, len(rms_db) - 1))
        end_frame = max(0, min(end_frame, len(rms_db)))
        
        if start_frame >= end_frame:
            cut_frame = end_frame
        else:
            window_rms = rms_db[start_frame:end_frame]
            safe_cut_indices = np.where(window_rms < target_threshold_db)[0]
            
            if len(safe_cut_indices) > 0:
                best_idx_in_window = safe_cut_indices[-1]
                cut_type = f"Threshold (<{target_threshold_db}dB)"
            else:
                best_idx_in_window = np.argmin(window_rms)
                cut_type = f"Local Min ({window_rms[best_idx_in_window]:.1f}dB)"
                
            cut_frame = start_frame + best_idx_in_window
            
        cut_sec = librosa.frames_to_time(cut_frame, sr=sr, hop_length=hop_length)
        
        start_sample = int(current_start_sec * sr)
        end_sample = int(cut_sec * sr)
        
        chunk_wav = waveform[:, start_sample:end_sample] if waveform.ndim > 1 else waveform[start_sample:end_sample]
        
        dur = cut_sec - current_start_sec
        print(f"    Sub-split at {cut_sec:.2f}s (duration {dur:.2f}s) - Reason: {cut_type}")
        
        chunks.append({'offset': current_start_sec, 'waveform': chunk_wav})
        current_start_sec = cut_sec

    return chunks

def grid_search_slice(
    waveform: np.ndarray,
    sr: int,
    min_len_sec: float = 4.0,
    max_len_sec: float = 20.0,
    min_interval_ms: int = 200,
    max_sil_kept_ms: int = 500,
):
    """
    Slices audio by performing a grid search for the best parameters.
    """
    print(f"Running grid search slicer for target range [{min_len_sec:.1f}s, {max_len_sec:.1f}s]...")

    # Define search space for parameters
    # More negative thresholds are more lenient (less likely to split)
    thresholds = [-45, -40, -35, -30, -25, -20] 
    # Shorter min_length is more aggressive (more likely to split)
    min_lengths_ms = [8000, 6000, 4000, 2500, 1500] 

    param_combinations = list(itertools.product(thresholds, min_lengths_ms))
    
    best_chunks = None
    best_score = float('inf')
    best_params = None

    for threshold, min_length_ms in param_combinations:
        try:
            slicer = Slicer(
                sr=sr,
                threshold=threshold,
                min_length=min_length_ms,
                min_interval=min_interval_ms,
                max_sil_kept=max_sil_kept_ms,
            )
            chunks = slicer.slice(waveform)
            
            if not chunks:
                continue

            durations = [len(c['waveform']) / sr for c in chunks]
            
            # Scoring logic: penalize chunks outside the target range
            score = 0
            num_short = 0
            num_long = 0
            
            for d in durations:
                if d < min_len_sec:
                    # Penalize more heavily for being too short
                    score += (min_len_sec - d) * 1.5 
                    num_short += 1
                elif d > max_len_sec:
                    # Penalize for being too long
                    score += (d - max_len_sec)
                    num_long += 1
            
            # Bonus for having a reasonable number of chunks (avoiding too many or too few)
            # This is a simple heuristic, might need tuning
            score += abs(len(chunks) - len(waveform) / sr / ((min_len_sec + max_len_sec) / 2)) * 0.5

            print(f"  Trying params: threshold={threshold}dB, min_length={min_length_ms}ms -> "
                  f"Score={score:.2f} ({len(chunks)} chunks, {num_short} short, {num_long} long)")

            if score < best_score:
                best_score = score
                best_chunks = chunks
                best_params = (threshold, min_length_ms)

        except Exception as e:
            print(f"  Error with params {threshold}, {min_length_ms}: {e}")
            continue

    if best_chunks:
        print(f"\nFound best slicer params: threshold={best_params[0]}dB, min_length={best_params[1]}ms")
        print(f"  - Sliced into {len(best_chunks)} chunks.")
        durations = [len(c['waveform']) / sr for c in best_chunks]
        print(f"  - Durations: min={min(durations):.2f}s, max={max(durations):.2f}s, avg={np.mean(durations):.2f}s")
    
    return best_chunks or []


def heuristic_slice(
    waveform: np.ndarray,
    sr: int,
    min_len_sec: float = 4.0,
    max_len_sec: float = 12.0,
    silence_removal_threshold_db: float = -40.0,
    min_silence_len_ms: int = 800,
    split_threshold_db: float = -30.0
):
    """
    A two-stage heuristic slicer.
    1. Removes long, obvious silences (interludes).
    2. Splits the remaining vocal segments to ensure they are within the desired length range.
    """
    # Stage 1: Remove long silences using a safe threshold
    print(f"Stage 1: Removing long silences below {silence_removal_threshold_db}dB...")
    pre_slicer = Slicer(
        sr=sr,
        threshold=silence_removal_threshold_db,
        min_length=min_silence_len_ms,
        min_interval=200,
        max_sil_kept=100
    )
    vocal_segments = pre_slicer.slice(waveform)
    print(f"  Found {len(vocal_segments)} vocal segments.")

    # Stage 2: Split long vocal segments using the sliding window method
    final_chunks = []
    print("\nStage 2: Splitting long segments to fit length constraints...")
    for segment in vocal_segments:
        seg_dur = len(segment['waveform']) / sr
        if seg_dur > max_len_sec:
            print(f"  Segment at {segment['offset']:.2f}s is too long ({seg_dur:.2f}s), applying sliding window split...")
            sub_chunks = _sliding_window_split(
                segment['waveform'], sr,
                min_len_sec=min_len_sec,
                max_len_sec=max_len_sec,
                target_threshold_db=split_threshold_db,
                frame_length=2048,
                hop_length=512,
            )
            # Adjust offsets of the sub_chunks relative to the original audio
            for sub in sub_chunks:
                sub['offset'] += segment['offset']
                final_chunks.append(sub)
        elif seg_dur >= min_len_sec:
            # Keep segment as is if it's within the valid range
            final_chunks.append(segment)
        else:
             print(f"  Discarding short segment at {segment['offset']:.2f}s (duration {seg_dur:.2f}s < {min_len_sec}s)")

    return final_chunks


def smart_slice(waveform, sr):
    """音频切片：包含基础切片、激进切片以及按RMS最小能量强制切片的逻辑"""
    slicer = Slicer(
        sr=sr,
        threshold=-30.,
        min_length=5000,
        min_interval=200,
        max_sil_kept=150,
    )
    chunks = slicer.slice(waveform)
    
    final_chunks = []
    for chunk in chunks:
        chunk_len = len(chunk['waveform'])
        chunk_dur = chunk_len / sr
        
        if chunk_dur <= 30.0:
            final_chunks.append(chunk)
            continue
            
        print(f"  [Auto Lyric] Chunk > 30s ({chunk_dur:.2f}s). Applying aggressive slicing...")
        
        slicer_agg = Slicer(
            sr=sr,
            threshold=-20.,
            min_length=4000,
            min_interval=200,
            max_sil_kept=150
        )
        sub_chunks = slicer_agg.slice(chunk['waveform'])
        
        base_offset = chunk['offset']
        for sub in sub_chunks:
            sub['offset'] += base_offset
            
            sub_len = len(sub['waveform'])
            sub_dur = sub_len / sr
            
            if sub_dur <= 30.0:
                final_chunks.append(sub)
                continue
                
            print(f"  [Auto Lyric] Sub-chunk still > 30s ({sub_dur:.2f}s). Forcing split at min energy...")
            
            sub_wav = sub['waveform']
            if len(sub_wav.shape) > 1:
                sub_wav_mono = np.mean(sub_wav, axis=0)
            else:
                sub_wav_mono = sub_wav
                
            hop_len = 512
            frame_len = 2048
            
            rms = librosa.feature.rms(y=sub_wav_mono, frame_length=frame_len, hop_length=hop_len, center=True)
            if rms.ndim > 1:
                rms = rms[0]
                
            n_frames = len(rms)
            start_frame = int(n_frames * 0.2)
            end_frame = int(n_frames * 0.8)
            
            if end_frame > start_frame:
                min_idx = np.argmin(rms[start_frame:end_frame]) + start_frame
            else:
                min_idx = n_frames // 2
                
            split_sample = min_idx * hop_len
            
            if len(sub_wav.shape) > 1:
                part1_wav = sub_wav[:, :split_sample]
                part2_wav = sub_wav[:, split_sample:]
            else:
                part1_wav = sub_wav[:split_sample]
                part2_wav = sub_wav[split_sample:]
                
            final_chunks.append({
                'offset': sub['offset'],
                'waveform': part1_wav
            })
            final_chunks.append({
                'offset': sub['offset'] + split_sample / sr,
                'waveform': part2_wav
            })

    return final_chunks

def prepare_asr_and_labels(chunks, sr, temp_dir_path, asr_model, g2p_model, matcher):
    """运行 ASR 并与原歌词匹配，生成 HubertFA 所需的 .lab 文件"""
    import soundfile as sf
    chars_dict = {}
    chunk_logs = []
    
    print("[Auto Lyric] Running ASR and preparing labels...")
    for chunk_idx, chunk in enumerate(chunks):
        chunk_wav = chunk["waveform"]
        stem = f"chunk_{chunk_idx}"
        chunk_len_s = len(chunk_wav) / sr
        
        chunk_wav_path = temp_dir_path / f"{stem}.wav"
        sf.write(chunk_wav_path, chunk_wav, sr)
        
        chunk_wav_16k = librosa.resample(chunk_wav, orig_sr=sr, target_sr=16000)
        res = asr_model.generate(input=[chunk_wav_16k], cache={}, is_final=True)
        
        if not res or len(res) == 0:
            continue
            
        text = res[0].get('text', '') if isinstance(res[0], dict) else str(res[0])
        
        if not text.strip():
            continue
            
        raw_chars = g2p_model.split_string_no_regex(text)
        if len(raw_chars) > chunk_len_s * 15:
            print(f"  [Warning] {stem}: ASR hallucination detected ({len(raw_chars)} chars in {chunk_len_s:.1f}s). Ignoring chunk.")
            chunk_logs.append(f"[{stem}]\nASR Output: {text}\nStatus: Ignored (Hallucination detected, {len(raw_chars)} chars in {chunk_len_s:.1f}s)\n")
            continue
            
        match_status = "No original lyrics provided"
        matched_result_text = text
            
        if matcher is not None:
            asr_text_list, asr_phonetic_list = matcher.process_asr_content(text)
            if asr_phonetic_list:
                matched_text, matched_phonetic, reason = matcher.align_lyric_with_asr(
                    asr_phonetic=asr_phonetic_list,
                    lyric_text=matcher.lyric_text_list,
                    lyric_phonetic=matcher.lyric_phonetic_list
                )
                if matched_phonetic:
                    pinyin_str = matched_phonetic
                    chars = matched_text.split()
                    match_status = "Matched with original lyrics"
                    matched_result_text = "".join(chars)
                else:
                    print(f"  [Warning] {stem}: No match found in original lyrics. Falling back to ASR output.")
                    pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                    chars = g2p_model.split_string_no_regex(text)
                    match_status = "Fallback to ASR (No match found)"
                    matched_result_text = "".join(chars)
            else:
                continue
        else:
            pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
            chars = g2p_model.split_string_no_regex(text)
            match_status = "Direct ASR (No original lyrics)"
            matched_result_text = "".join(chars)
        
        if getattr(g2p_model, '__class__', None).__name__ == 'JaG2p':
            chars = pinyin_str.split()
            matched_result_text = " ".join(chars)

        chunk_lab_path = temp_dir_path / f"{stem}.lab"
        chunk_lab_path.write_text(pinyin_str, encoding="utf-8")
        chars_dict[stem] = chars
        
        chunk_logs.append(f"[{stem}]\nASR Output: {text}\nMatch Status: {match_status}\nFinal Assigned Lyrics: {matched_result_text}\nFA Pinyin (.lab): {pinyin_str}\n")
    
    return chars_dict, chunk_logs

def run_hubert_fa(hfa_model, temp_dir, language="zh"):
    """运行 HubertFA 强制对齐"""
    print("[Auto Lyric] Running HubertFA forced alignment...")
    hfa_model.dataset = []
    hfa_model.predictions = []
    
    # Ensure correct dictionary path is used based on language
    dict_file = "ds-zh-pinyin-lite.txt" if language == "zh" else "japanese_dict_full.txt"
    dict_path = hfa_model.vocab_folder / dict_file
    
    hfa_model.get_dataset(wav_folder=temp_dir, language=language, g2p="dictionary", dictionary_path=dict_path)
    if len(hfa_model.dataset) > 0:
        hfa_model.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
        
    pred_dict = {p[0].stem: p for p in hfa_model.predictions}
    return pred_dict

def extract_pitches_and_align(chunks, sr, pred_dict, chars_dict, game_model, seg_threshold, seg_radius, est_threshold, batch_size=4):
    """使用 GAME 模型提取音高并对齐歌词"""
    print("[Auto Lyric] Extracting pitches with GAME...")
    all_notes = []
    batch_wavs = []
    batch_durs = []
    batch_known_durs = []
    batch_infos = []
    
    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        if stem not in pred_dict:
            continue
            
        _, wav_len, result_word = pred_dict[stem]
        word_durs, word_vuvs, lyrics = extract_vowel_boundaries(result_word, chars_dict[stem])
        
        batch_wavs.append(chunk["waveform"])
        batch_durs.append(len(chunk["waveform"]) / sr)
        batch_known_durs.append(np.array(word_durs, dtype=np.float32))
        batch_infos.append({
            "offset": chunk["offset"],
            "word_durs": word_durs,
            "word_vuvs": word_vuvs,
            "lyrics": lyrics
        })
        
    if len(batch_wavs) > 0:
        for i in range(0, len(batch_wavs), batch_size):
            b_w = batch_wavs[i:i+batch_size]
            b_d = batch_durs[i:i+batch_size]
            b_kd = batch_known_durs[i:i+batch_size]
            b_info = batch_infos[i:i+batch_size]
            
            padded_wavs = pad_1d_arrays(b_w).astype(np.float32)
            padded_kd = pad_1d_arrays(b_kd, pad_value=0.0).astype(np.float32)
            
            results = game_model.infer_batch(
                waveforms=padded_wavs,
                durations=np.array(b_d, dtype=np.float32),
                known_durations=padded_kd,
                boundary_threshold=seg_threshold,
                boundary_radius=round(seg_radius / game_model.timestep),
                score_threshold=est_threshold,
                language=0,
                ts=None
            )
            
            for chunk_res, info in zip(results, b_info):
                c_durations, c_presence, c_scores = chunk_res
                valid = c_durations > 0
                
                note_dur = c_durations[valid].tolist()
                note_midi = c_scores[valid].tolist()
                note_vuv = c_presence[valid].tolist()
                
                note_seq = [
                    librosa.midi_to_note(m, unicode=False, cents=True) if v else "rest"
                    for m, v in zip(note_midi, note_vuv)
                ]
                
                a_note_seq, a_note_dur, a_note_slur = align_notes_to_words(
                    info["word_durs"], info["word_vuvs"],
                    note_seq, note_dur,
                    apply_word_uv=True
                )
                
                lyric_idx = 0
                current_onset = info["offset"]
                
                pending_lyric = ""
                
                for n_seq, n_dur, n_slur in zip(a_note_seq, a_note_dur, a_note_slur):
                    pitch = librosa.note_to_midi(n_seq) if n_seq != "rest" else 0.0
                    
                    if n_slur == 0:
                        if lyric_idx < len(info["lyrics"]):
                            word_lyric = info["lyrics"][lyric_idx]
                            if info["word_vuvs"][lyric_idx] == 1:
                                pending_lyric = word_lyric
                            else:
                                pending_lyric = ""
                            lyric_idx += 1
                    
                if n_seq != "rest":
                    pitch = librosa.note_to_midi(n_seq, round_midi=False)
                    lyric_to_assign = ""

                    if n_slur == 0:
                        if lyric_idx < len(info["lyrics"]):
                            word_lyric = info["lyrics"][lyric_idx]
                            if info["word_vuvs"][lyric_idx] == 1:
                                pending_lyric = word_lyric
                            else:
                                pending_lyric = ""
                            lyric_idx += 1
                    
                    if pending_lyric:
                        lyric_to_assign = pending_lyric
                        pending_lyric = ""
                    else:
                        lyric_to_assign = "-"

                    is_contiguous = len(all_notes) > 0 and abs(all_notes[-1].offset - current_onset) < 0.01
                    
                    # Merge if the *current* note is a slur ('-') and matches the pitch of the *previous* note
                    can_merge = is_contiguous and abs(all_notes[-1].pitch - pitch) < 0.1 and lyric_to_assign == "-"

                    if can_merge:
                        all_notes[-1].offset += n_dur
                    else:
                        all_notes.append(NoteInfo(
                            onset=current_onset,
                            offset=current_onset + n_dur,
                            pitch=pitch,
                            lyric=lyric_to_assign
                        ))
                
                current_onset += n_dur
                    
    return all_notes

def export_artifacts(chunks, temp_dir_path, hfa_model, output_key, output_dir, output_formats):
    """导出中间产物如 TextGrid 和切片音频（如果需要）"""
    import shutil
    temp_tg_dir = temp_dir_path / "temp_tg"
    hfa_model.export(temp_tg_dir, output_format=['textgrid'])
    
    tg_subfolder = temp_tg_dir / "TextGrid"
    
    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        new_stem = f"{output_key}_{chunk_idx:03d}"
        
        if "chunks" in output_formats:
            chunk_wav_path = temp_dir_path / f"{stem}.wav"
            try:
                if chunk_wav_path.exists():
                    shutil.copy2(chunk_wav_path, output_dir / f"{new_stem}.wav")
                else:
                    print(f"[Warning] Chunk WAV file not found, skipping: {chunk_wav_path}")
            except Exception as e:
                print(f"[Error] Failed to copy chunk {chunk_wav_path}: {e}")

        if tg_subfolder.exists():
            tg_path = tg_subfolder / f"{stem}.TextGrid"
            try:
                if tg_path.exists():
                    shutil.copy2(tg_path, output_dir / f"{new_stem}.TextGrid")
            except Exception as e:
                print(f"[Error] Failed to copy TextGrid {tg_path}: {e}")


def prepare_qwen3_asr_and_labels(chunks, sr, original_lyrics, model_dir, device_pref, temp_dir_path, g2p_model, matcher):
    import os
    import sys
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    
    qwen_asr_dir = os.path.join(model_dir, "Qwen3-ASR-onnx-main")
    if qwen_asr_dir not in sys.path:
        sys.path.insert(0, qwen_asr_dir)
        
    from onnx_asr_service import OnnxAsrRuntime
    from inference.funasr_nano.dynamic_lyric_inference import find_best_match
    
    # Force CPU as requested
    providers = ["CPUExecutionProvider"]
    
    print(f"[Auto Lyric] Loading Qwen3-ASR ONNX from {model_dir} on CPU...")
    runtime = OnnxAsrRuntime(onnx_dir=Path(model_dir), providers=providers, max_new_tokens=256)
    
    def run_chunk_with_hotwords(wf_16k, hotwords):
        context = ""  # Hotwords disabled for Qwen3-ASR as requested
        result = runtime.transcribe_waveform((wf_16k, 16000), context=context)
        text = result["text"]
        return text

    if original_lyrics and original_lyrics.strip():
        lyrics_lines = [line.strip() for line in original_lyrics.split('\n') if line.strip()]
    else:
        lyrics_lines = []

    current_lyric_idx = 0
    is_aligned = False
    window_size = 16
    match_threshold = 0.3
    
    chars_dict = {}
    chunk_logs = []
    
    print("[Auto Lyric] Running Qwen3-ASR...")
    for chunk_idx, chunk in enumerate(chunks):
        wf = chunk['waveform']
        if sr != 16000:
            import librosa
            wf_16k = librosa.resample(wf, orig_sr=sr, target_sr=16000)
        else:
            wf_16k = wf
        
        stem = f"chunk_{chunk_idx}"
        chunk_wav_path = temp_dir_path / f"{stem}.wav"
        sf.write(chunk_wav_path, chunk['waveform'], sr)

        raw_text = None
        match_status = "Direct ASR (No original lyrics)"
        matched_result_text = ""
        pinyin_str = ""
        chars = []
        used_hotwords_str = "None (Disabled)"

        # Always run ASR without hotwords
        raw_text = run_chunk_with_hotwords(wf_16k, [])
        text = raw_text or ""
        
        # Still try to match with lyrics if provided, for logging and alignment
        if lyrics_lines:
            if not is_aligned:
                best_idx, match_ratio = find_best_match(raw_text, lyrics_lines)
                if match_ratio >= match_threshold:
                    current_lyric_idx = best_idx
                    is_aligned = True
                    match_status = f"Global Aligned L{best_idx} (ratio {match_ratio:.2f})"
                else:
                    match_status = f"Global Search (ratio {match_ratio:.2f})"
            else:
                window_end = min(current_lyric_idx + window_size, len(lyrics_lines))
                candidate_hotwords = lyrics_lines[current_lyric_idx:window_end]
                best_idx, match_ratio = find_best_match(raw_text, candidate_hotwords)
                if match_ratio >= match_threshold:
                    matched_global_idx = current_lyric_idx + best_idx
                    current_lyric_idx = matched_global_idx + 1
                    match_status = f"Local Matched L{matched_global_idx} (ratio {match_ratio:.2f})"
                else:
                    match_status = f"Local Hold (ratio {match_ratio:.2f})"

        if matcher is not None and text.strip():
            asr_text_list, asr_phonetic_list = matcher.process_asr_content(text)
            if asr_phonetic_list:
                matched_text, matched_phonetic, reason = matcher.align_lyric_with_asr(
                    asr_phonetic=asr_phonetic_list,
                    lyric_text=matcher.lyric_text_list,
                    lyric_phonetic=matcher.lyric_phonetic_list
                )
                if matched_phonetic:
                    pinyin_str = matched_phonetic
                    chars = matched_text.split()
                    matched_result_text = "".join(chars)
                    match_status += " | LyricFA Matched"
                else:
                    pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                    chars = g2p_model.split_string_no_regex(text)
                    matched_result_text = "".join(chars)
                    match_status += " | LyricFA Fallback"
            else:
                chars = g2p_model.split_string_no_regex(text)
                pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                matched_result_text = "".join(chars)
        elif text.strip():
            chars = g2p_model.split_string_no_regex(text)
            pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
            matched_result_text = "".join(chars)
            
        if getattr(g2p_model, '__class__', None).__name__ == 'JaG2p':
            chars = pinyin_str.split()
            matched_result_text = " ".join(chars)

        chunk_lab_path = temp_dir_path / f"{stem}.lab"
        chunk_lab_path.write_text(pinyin_str, encoding="utf-8")
        chars_dict[stem] = chars
        
        chunk_logs.append(f"[{stem}]\nHotwords Used: {used_hotwords_str}\nASR Output: {text}\nMatch Status: {match_status}\nFinal Assigned Lyrics: {matched_result_text}\nFA Pinyin (.lab): {pinyin_str}\n")
        
    return chunks, chars_dict, chunk_logs

def prepare_dynamic_asr_and_labels(chunks, sr, original_lyrics, model_dir, device_pref, temp_dir_path, g2p_model, matcher):
    import os
    import torch
    import numpy as np
    import soundfile as sf
    from inference.funasr_nano.dynamic_lyric_inference import find_best_match, run_llm_inference, get_prompt
    from inference.funasr_nano.utils import EncoderAdaptorOnnxModel, EmbeddingOnnxIOB, UnifiedKvDeltaLLMOnnxIOB, compute_feat, build_source_ids, select_device, device_from_str
    from transformers import AutoTokenizer
    
    encoder_model = os.path.join(model_dir, "encoder_adaptor.onnx")
    embedding_model = os.path.join(model_dir, "embedding.onnx")
    llm_model = os.path.join(model_dir, "llm.int8.1024.onnx")
    if not os.path.exists(llm_model):
        llm_model = os.path.join(model_dir, "llm.fp32.onnx")
    tokenizer_path = os.path.join(model_dir, "Qwen3-0.6B")
    vad_model_path = os.path.join(model_dir, "silero_vad.onnx")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None
    
    enc_dev = select_device(device_pref)
    emb_dev = select_device(device_pref)
    llm_dev = select_device(device_pref, model_path=llm_model)
    device = device_from_str("cuda" if llm_dev == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        
    encoder = EncoderAdaptorOnnxModel(encoder_model, device=enc_dev)
    embedding = EmbeddingOnnxIOB(embedding_model, device=emb_dev)
    llm = UnifiedKvDeltaLLMOnnxIOB(llm_model, device=llm_dev)

    vad_model = SileroVAD(
        model_path=vad_model_path,
        threshold=0.5,
        min_silence_duration=0.5,
        min_speech_duration=0.25,
        window_size=512,
        max_speech_duration=20,
        sample_rate=16000,
        num_threads=1,
    )
    
    def run_chunk_with_hotwords(wf_16k, hotwords):
        prompt_text = get_prompt(hotwords, language=None, itn=True)
        system_prompt = "You are a helpful assistant."
        user_prompt = f"{prompt_text}<|startofspeech|>!!<|endofspeech|>"
        
        feats = compute_feat(wf_16k, 16000, encoder.window_size, encoder.window_shift)[None, ...]
        enc_out = encoder(feats)
        enc_out = np.where(np.isfinite(enc_out), enc_out, 0.0)
        
        audio_token_len = int(enc_out.shape[1])
        source_ids_1d, fbank_beg_idx, fake_len = build_source_ids(tokenizer, system_prompt, user_prompt, audio_token_len)
        
        input_ids = torch.tensor(source_ids_1d.reshape(1, -1), device=device, dtype=torch.int64)
        text_embeds = embedding.forward_ids(input_ids).to(dtype=llm.input_torch_dtype)
        
        enc_t = torch.tensor(enc_out, device=device, dtype=llm.input_torch_dtype)
        fl = min(fake_len, enc_t.shape[1])
        text_embeds[0, fbank_beg_idx:fbank_beg_idx+fl, :] = enc_t[0, :fl, :]
        
        if llm.max_total_len > 0 and text_embeds.shape[1] >= llm.max_total_len:
            return None
            
        generated = run_llm_inference(
            llm=llm,
            embedding=embedding,
            tokenizer=tokenizer,
            inputs_embeds=text_embeds,
            device=device,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=eos_token_id,
            im_end_token_id=im_end_token_id
        )
        if generated:
            result_text = tokenizer.decode(generated, skip_special_tokens=True)
            result_text = result_text.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
            return " ".join(result_text.split())
        return ""

    if original_lyrics and original_lyrics.strip():
        lyrics_lines = [line.strip() for line in original_lyrics.split('\n') if line.strip()]
    else:
        lyrics_lines = []

    current_lyric_idx = 0
    is_aligned = False
    window_size = 8
    match_threshold = 0.3
    
    new_chunks = []
    chars_dict = {}
    chunk_logs = []
    global_chunk_idx = 0
    
    print("[Auto Lyric] Running Dynamic ASR...")
    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        chunk_wav_path = temp_dir_path / f"{stem}.wav"
        sf.write(chunk_wav_path, chunk['waveform'], sr)

        if sr != 16000:
            import librosa
            wf_16k = librosa.resample(chunk['waveform'], orig_sr=sr, target_sr=16000)
        else:
            wf_16k = chunk['waveform']
            
            raw_text = None
            match_status = "Direct ASR (No original lyrics)"
            matched_result_text = ""
            pinyin_str = ""
            chars = []
            used_hotwords_str = "None"

            if not lyrics_lines:
                raw_text = run_chunk_with_hotwords(sub_wf_16k, [])
                text = raw_text or ""
                chars = g2p_model.split_string_no_regex(text)
                pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                matched_result_text = "".join(chars)
            else:
                if not is_aligned:
                    raw_text = run_chunk_with_hotwords(sub_wf_16k, [])
                    used_hotwords_str = "[] (Global Search)"
                    if raw_text is not None:
                        best_idx, match_ratio = find_best_match(raw_text, lyrics_lines)
                        if match_ratio >= match_threshold:
                            current_lyric_idx = best_idx
                            is_aligned = True
                            match_status = f"Global Aligned L{best_idx} (ratio {match_ratio:.2f})"
                        else:
                            match_status = f"Global Search (ratio {match_ratio:.2f})"
                else:
                    window_end = min(current_lyric_idx + window_size, len(lyrics_lines))
                    current_hotwords = lyrics_lines[current_lyric_idx:window_end]
                    used_hotwords_str = str(current_hotwords)
                    raw_text = run_chunk_with_hotwords(sub_wf_16k, current_hotwords)
                    if raw_text is not None:
                        best_idx, match_ratio = find_best_match(raw_text, current_hotwords)
                        if match_ratio >= match_threshold:
                            matched_global_idx = current_lyric_idx + best_idx
                            current_lyric_idx = matched_global_idx + 1
                            match_status = f"Local Matched L{matched_global_idx} (ratio {match_ratio:.2f})"
                        else:
                            match_status = f"Local Hold (ratio {match_ratio:.2f})"

                text = raw_text or ""
                if matcher is not None and text.strip():
                    asr_text_list, asr_phonetic_list = matcher.process_asr_content(text)
                    if asr_phonetic_list:
                        matched_text, matched_phonetic, reason = matcher.align_lyric_with_asr(
                            asr_phonetic=asr_phonetic_list,
                            lyric_text=matcher.lyric_text_list,
                            lyric_phonetic=matcher.lyric_phonetic_list
                        )
                        if matched_phonetic:
                            pinyin_str = matched_phonetic
                            chars = matched_text.split()
                            matched_result_text = "".join(chars)
                            match_status += " | LyricFA Matched"
                        else:
                            pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                            chars = g2p_model.split_string_no_regex(text)
                            matched_result_text = "".join(chars)
                            match_status += " | LyricFA Fallback"
                    else:
                        chars = g2p_model.split_string_no_regex(text)
                        pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                        matched_result_text = "".join(chars)
                elif text.strip():
                    chars = g2p_model.split_string_no_regex(text)
                    pinyin_str = g2p_model.convert(text, include_tone=False, convert_number=True)
                    matched_result_text = "".join(chars)
                    
            if getattr(g2p_model, '__class__', None).__name__ == 'JaG2p':
                chars = pinyin_str.split()
                matched_result_text = " ".join(chars)

            chunk_lab_path = temp_dir_path / f"{stem}.lab"
            chunk_lab_path.write_text(pinyin_str, encoding="utf-8")
            chars_dict[stem] = chars
            
            chunk_logs.append(f"[{stem}]\nHotwords Used: {used_hotwords_str}\nASR Output: {text}\nMatch Status: {match_status}\nFinal Assigned Lyrics: {matched_result_text}\nFA Pinyin (.lab): {pinyin_str}\n")
            
    return chunks, chars_dict, chunk_logs

def auto_lyric_pipeline(
    audio_path: str,
    output_filename: str,
    game_model_path: str,
    onnx_device: str,
    hfa_onnx_path: str,
    asr_model_path: str,
    asr_method: str,
    dynamic_asr_model_dir: str,
    language: str,
    original_lyrics: str,
    output_dir: pathlib.Path,
    output_formats: list,
    slicing_method: str,
    tempo: float,
    quantization_step: int,
    pitch_format: str,
    round_pitch: bool,
    seg_threshold: float,
    seg_radius: float,
    est_threshold: float,
    ts: list,
    batch_size: int = 4
):
    """Auto Lyric 的主处理流水线"""
    output_key = pathlib.Path(output_filename).stem
    print(f"\n[Auto Lyric] Processing audio: {audio_path}")
    # Default to 44100 which is GAME's typical samplerate
    sr = 44100
    waveform, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # [Legacy] 1. 预处理和切片
    if slicing_method == "启发式切片":
        chunks = heuristic_slice(waveform, sr)
    elif slicing_method == "网格搜索切片":
        chunks = grid_search_slice(waveform, sr)
    else:
        chunks = smart_slice(waveform, sr)
    print(f"Sliced into {len(chunks)} chunks.")
    
    # 2. 初始化G2P模型
    g2p_model = get_ja_g2p() if language == "ja" else get_zh_g2p()
    
    # 3. 处理原歌词（如果提供）
    matcher = None
    if original_lyrics and original_lyrics.strip():
        print("[Auto Lyric] Processing original lyrics for matching...")
        matcher = LyricMatcher(language if language in ["zh", "en"] else "zh")
        processor = matcher.processor
        cleaned_lyric = processor.clean_text(original_lyrics)
        
        matcher.lyric_text_list = processor.split_text(cleaned_lyric)
        matcher.lyric_phonetic_list = processor.get_phonetic_list(matcher.lyric_text_list)

    all_notes = []
    chunk_logs = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        
        # 4. 运行 ASR、匹配歌词并生成标签
        if asr_method == "Dynamic Lyric (热词增强)":
            device_pref = "dml" if onnx_device == "dml" else "cpu"
            chunks, chars_dict, chunk_logs = prepare_dynamic_asr_and_labels(
                chunks, sr, original_lyrics, dynamic_asr_model_dir, device_pref, temp_dir_path, g2p_model, matcher
            )
            free_memory()
        elif asr_method == "Qwen3-ASR (热词增强)":
            device_pref = "cpu"  # Force CPU for now
            chunks, chars_dict, chunk_logs = prepare_qwen3_asr_and_labels(
                chunks, sr, original_lyrics, dynamic_asr_model_dir, device_pref, temp_dir_path, g2p_model, matcher
            )
            free_memory()
        else:
            asr_model = get_funasr_model(asr_model_path)
            chars_dict, chunk_logs = prepare_asr_and_labels(
                chunks, sr, temp_dir_path, asr_model, g2p_model, matcher
            )
            del asr_model
            free_memory()

        # 5. 运行 HubertFA 强制对齐
        hfa_model = get_hfa_model(hfa_onnx_path)
        pred_dict = run_hubert_fa(hfa_model, temp_dir, language=language)
        
        # 7. 导出额外产物 (如 TextGrid) BEFORE releasing hfa_model
        export_artifacts(chunks, temp_dir_path, hfa_model, output_key, output_dir, output_formats)
        
        del hfa_model
        free_memory()
        
        # 6. 运行 GAME 推理、提取音高并与歌词对齐
        from inference.onnx_api import load_onnx_model
        print(f"Loading GAME model from {game_model_path}...")
        game_model = load_onnx_model(pathlib.Path(game_model_path), device=onnx_device)
        
        all_notes = extract_pitches_and_align(
            chunks, sr, pred_dict, chars_dict, game_model,
            seg_threshold, seg_radius, est_threshold, batch_size
        )
        
        if hasattr(game_model, 'release'):
            game_model.release()
        del game_model
        free_memory()

    # 8. 排序、量化并保存最终文件
    all_notes.sort(key=lambda x: x.onset)
    
    log_path = output_dir / f"{output_key}_asr_match_log.txt"
    log_path.write_text("\n".join(chunk_logs), encoding="utf-8")

    if quantization_step > 0:
        quantize_notes(all_notes, tempo, quantization_step)
        
    print(f"Extracted {len(all_notes)} notes with lyrics.")

    if "mid" in output_formats:
        _save_midi(all_notes, output_dir / f"{output_key}.mid", int(tempo))
    if "txt" in output_formats:
        _save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
    if "csv" in output_formats:
        _save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)

if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("audio_path", type=click.Path(exists=True))
    @click.option("--game-model", "-gm", required=True, type=click.Path(exists=True), help="Path to GAME ONNX model directory")
    @click.option("--hfa-model", "-hm", required=True, type=click.Path(exists=True), help="Path to HubertFA ONNX model file")
    @click.option("--asr-model", "-am", type=str, default="", help="Path to local FunASR model or ModelScope ID")
    @click.option("--output-dir", "-o", type=click.Path(), default=".", help="Directory to save the outputs")
    @click.option("--lyrics", "-l", type=str, default="", help="Original reference lyrics for alignment")
    @click.option("--language", type=str, default="zh", help="Language code (default: zh)")
    @click.option("--tempo", type=float, default=120.0, help="Tempo BPM (default: 120)")
    @click.option("--quantize", type=int, default=60, help="Quantization step (default: 60 for 1/32 note. 0 = none)")
    @click.option("--asr-method", type=str, default="FunASR (默认)", help="ASR method to use: 'FunASR (默认)', 'Dynamic Lyric (热词增强)', or 'Qwen3-ASR (热词增强)'")
    @click.option("--dynamic-asr-dir", type=str, default="experiments/funasr_nano_models", help="Path to Dynamic Lyric ASR model directory")
    @click.option("--slicing-method", type=str, default="默认切片", help="Slicing method: '默认切片', '启发式切片', or '网格搜索切片'")
    @click.option("--formats", "-f", type=str, default="mid", help="Comma-separated output formats (mid,txt,csv,chunks)")
    @click.option("--pitch-format", type=click.Choice(["name", "number"]), default="name", help="Pitch format for txt/csv")
    @click.option("--round-pitch", is_flag=True, help="Round pitch values to integers")
    @click.option("--seg-threshold", type=float, default=0.2, help="Segmentation threshold")
    @click.option("--seg-radius", type=float, default=0.02, help="Segmentation radius (seconds)")
    @click.option("--est-threshold", type=float, default=0.2, help="Note presence threshold")
    @click.option("--t0", type=float, default=0.0, help="D3PM starting t0")
    @click.option("--nsteps", type=int, default=8, help="D3PM sampling steps")
    @click.option("--batch-size", "-b", type=int, default=4, help="Batch size for GAME inference")
    @click.option("--device", type=click.Choice(["cpu", "dml"]), default="dml", help="ONNX execution provider")
    def main(audio_path, game_model, hfa_model, asr_model, output_dir, lyrics, language, tempo, quantize,
             asr_method, dynamic_asr_dir, slicing_method, formats, pitch_format, round_pitch, seg_threshold, seg_radius, est_threshold,
             t0, nsteps, batch_size, device):
        """
        [Legacy] Auto Lyric Alignment Pipeline: kept for compatibility.
        Recommended main pipeline: `inference/auto_lyric_hybrid.py`.
        """
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        step = (1 - t0) / nsteps
        ts = [t0 + i * step for i in range(nsteps)]
        
        output_formats = [fmt.strip() for fmt in formats.split(",")]

        auto_lyric_pipeline(
            audio_path=audio_path,
            output_filename=pathlib.Path(audio_path).name,
            game_model_path=game_model,
            onnx_device=device,
            hfa_onnx_path=hfa_model,
            asr_model_path=asr_model,
            asr_method=asr_method,
            dynamic_asr_model_dir=dynamic_asr_dir,
            language=language,
            original_lyrics=lyrics,
            output_dir=out_dir,
            output_formats=output_formats,
            slicing_method=slicing_method,
            tempo=tempo,
            quantization_step=quantize,
            pitch_format=pitch_format,
            round_pitch=round_pitch,
            seg_threshold=seg_threshold,
            seg_radius=seg_radius,
            est_threshold=est_threshold,
            ts=ts,
            batch_size=batch_size
        )
        print("Done!")

    main()
