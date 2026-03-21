import os
import pathlib
import tempfile
import librosa
import numpy as np
import warnings
import sys

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
from inference.vendor.LyricFA.tools.lyric_matcher import LyricMatcher
from onnx_infer import InferenceOnnx

_funasr_model = None
_hfa_model = None
_zh_g2p = None

def get_funasr_model(model_path=None):
    global _funasr_model
    if _funasr_model is None:
        print("Loading FunASR model...")
        import logging
        logging.getLogger("funasr").setLevel(logging.ERROR)
        
        if not model_path or not str(model_path).strip():
            model_path = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
            
        _funasr_model = AutoModel(
            model=model_path,
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
            disable_pbar=True
        )
    return _funasr_model

def get_hfa_model(onnx_path: str):
    global _hfa_model
    if _hfa_model is None:
        print("Loading HubertFA ONNX model...")
        _hfa_model = InferenceOnnx(onnx_path=pathlib.Path(onnx_path))
        _hfa_model.load_config()
        _hfa_model.init_decoder()
        _hfa_model.load_model()
    return _hfa_model

def get_zh_g2p():
    global _zh_g2p
    if _zh_g2p is None:
        _zh_g2p = ZhG2p("mandarin")
    return _zh_g2p

def extract_vowel_boundaries(result_word, original_chars: list[str]):
    word_durs = []
    word_vuvs = []
    lyrics = []
    
    char_idx = 0
    last_end = 0.0
    
    ignore_tokens = {"SP", "AP", "EP", "br", "sil", "pau"}
    
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
        
        if char_idx < len(original_chars):
            lyrics.append(original_chars[char_idx])
            char_idx += 1
        else:
            lyrics.append(word.text)
            
        last_end = note_end

    return word_durs, word_vuvs, lyrics

def smart_slice(waveform, sr):
    """音频切片：包含基础切片、激进切片以及按RMS最小能量强制切片的逻辑"""
    slicer = Slicer(
        sr=sr,
        threshold=-30.,  
        min_length=5000,
        min_interval=200,
        max_sil_kept=200,
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

def prepare_asr_and_labels(chunks, sr, temp_dir_path, asr_model, zh_g2p, matcher):
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
            
        raw_chars = zh_g2p.split_string_no_regex(text)
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
                    pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
                    chars = zh_g2p.split_string_no_regex(text)
                    match_status = "Fallback to ASR (No match found)"
                    matched_result_text = "".join(chars)
            else:
                continue
        else:
            pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
            chars = zh_g2p.split_string_no_regex(text)
            match_status = "Direct ASR (No original lyrics)"
            matched_result_text = "".join(chars)
        
        chunk_lab_path = temp_dir_path / f"{stem}.lab"
        chunk_lab_path.write_text(pinyin_str, encoding="utf-8")
        chars_dict[stem] = chars
        
        chunk_logs.append(f"[{stem}]\nASR Output: {text}\nMatch Status: {match_status}\nFinal Assigned Lyrics: {matched_result_text}\nFA Pinyin (.lab): {pinyin_str}\n")
    
    return chars_dict, chunk_logs

def run_hubert_fa(hfa_model, temp_dir):
    """运行 HubertFA 强制对齐"""
    print("[Auto Lyric] Running HubertFA forced alignment...")
    hfa_model.dataset = []
    hfa_model.predictions = []
    hfa_model.get_dataset(wav_folder=temp_dir, language="zh", g2p="dictionary", dictionary_path=None)
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
                    apply_word_uv=False
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
                        if pending_lyric != "":
                            assigned_lyric = pending_lyric
                            pending_lyric = "" 
                            
                            all_notes.append(NoteInfo(
                                onset=current_onset,
                                offset=current_onset + n_dur,
                                pitch=pitch,
                                lyric=assigned_lyric
                            ))
                        else:
                            is_contiguous = len(all_notes) > 0 and abs(all_notes[-1].offset - current_onset) < 1e-3
                            
                            if is_contiguous:
                                if abs(all_notes[-1].pitch - pitch) < 1e-3:
                                    # Merge
                                    all_notes[-1].offset = current_onset + n_dur
                                else:
                                    # Slur
                                    all_notes.append(NoteInfo(
                                        onset=current_onset,
                                        offset=current_onset + n_dur,
                                        pitch=pitch,
                                        lyric="-"
                                    ))
                            else:
                                pass
                                
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
            if chunk_wav_path.exists():
                shutil.copy2(chunk_wav_path, output_dir / f"{new_stem}.wav")
            
        if tg_subfolder.exists():
            tg_path = tg_subfolder / f"{stem}.TextGrid"
            if tg_path.exists():
                shutil.copy2(tg_path, output_dir / f"{new_stem}.TextGrid")


def auto_lyric_pipeline(
    audio_path: str,
    output_filename: str,
    game_model,
    hfa_onnx_path: str,
    asr_model_path: str,
    language: str,
    original_lyrics: str,
    output_dir: pathlib.Path,
    output_formats: list,
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
    waveform, sr = librosa.load(audio_path, sr=game_model.samplerate, mono=True)
    
    # 1. 音频智能切片（带多级fallback）
    chunks = smart_slice(waveform, sr)
    print(f"Sliced into {len(chunks)} chunks.")
    
    # 2. 初始化各模型
    asr_model = get_funasr_model(asr_model_path)
    hfa_model = get_hfa_model(hfa_onnx_path)
    zh_g2p = get_zh_g2p()
    
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
        chars_dict, chunk_logs = prepare_asr_and_labels(
            chunks, sr, temp_dir_path, asr_model, zh_g2p, matcher
        )

        # 5. 运行 HubertFA 强制对齐
        pred_dict = run_hubert_fa(hfa_model, temp_dir)
        
        # 6. 运行 GAME 推理、提取音高并与歌词对齐
        all_notes = extract_pitches_and_align(
            chunks, sr, pred_dict, chars_dict, game_model,
            seg_threshold, seg_radius, est_threshold, batch_size
        )

        # 7. 导出额外产物 (如 TextGrid)
        export_artifacts(chunks, temp_dir_path, hfa_model, output_key, output_dir, output_formats)

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
             formats, pitch_format, round_pitch, seg_threshold, seg_radius, est_threshold,
             t0, nsteps, batch_size, device):
        """
        Auto Lyric Alignment Pipeline: Extracts notes from singing voice and automatically aligns them with lyrics using ASR.
        """
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        from inference.onnx_api import load_onnx_model
        print(f"Loading GAME model from {game_model}...")
        game_model_session = load_onnx_model(pathlib.Path(game_model), device=device)

        step = (1 - t0) / nsteps
        ts = [t0 + i * step for i in range(nsteps)]
        
        output_formats = [fmt.strip() for fmt in formats.split(",")]

        auto_lyric_pipeline(
            audio_path=audio_path,
            output_filename=pathlib.Path(audio_path).name,
            game_model=game_model_session,
            hfa_onnx_path=hfa_model,
            asr_model_path=asr_model,
            language=language,
            original_lyrics=lyrics,
            output_dir=out_dir,
            output_formats=output_formats,
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
