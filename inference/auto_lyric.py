import os
import pathlib
import tempfile
import librosa
import numpy as np
import warnings
import sys

from funasr import AutoModel

from inference.slicer2 import Slicer
from inference.onnx_api import pad_1d_arrays, NoteInfo, _save_midi, _save_text
from inference.utils import align_notes_to_words

# Add third_party paths
THIRD_PARTY_DIR = pathlib.Path(__file__).parent.parent / "third_party"
# Only insert HubertFA to sys.path because it uses absolute imports like `from tools...`
sys.path.insert(0, str(THIRD_PARTY_DIR / "HubertFA"))

from third_party.LyricFA.tools.ZhG2p import ZhG2p
from third_party.LyricFA.tools.lyric_matcher import LyricMatcher
from onnx_infer import InferenceOnnx

_funasr_model = None
_hfa_model = None
_zh_g2p = None

def get_funasr_model():
    global _funasr_model
    if _funasr_model is None:
        print("Loading FunASR model...")
        import logging
        logging.getLogger("funasr").setLevel(logging.ERROR)
        _funasr_model = AutoModel(
            model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            model_revision="v2.0.4",
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

def auto_lyric_pipeline(
    audio_path: str,
    output_filename: str,
    game_model,
    hfa_onnx_path: str,
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
    est_threshold: float
):
    print(f"\n[Auto Lyric] Processing audio: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=game_model.samplerate, mono=True)
    
    slicer = Slicer(
        sr=sr,
        threshold=-30.,  # Changed to -30.0 for more aggressive vocal slicing
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    chunks = slicer.slice(waveform)
    print(f"Sliced into {len(chunks)} chunks.")
    
    asr_model = get_funasr_model()
    hfa_model = get_hfa_model(hfa_onnx_path)
    zh_g2p = get_zh_g2p()
    
    matcher = None
    lyric_text_list = []
    lyric_phonetic_list = []
    if original_lyrics and original_lyrics.strip():
        print("[Auto Lyric] Processing original lyrics for matching...")
        matcher = LyricMatcher(language if language in ["zh", "en"] else "zh")
        processor = matcher.processor
        cleaned_lyric = processor.clean_text(original_lyrics)
        lyric_text_list = processor.split_text(cleaned_lyric)
        lyric_phonetic_list = processor.get_phonetic_list(lyric_text_list)

    all_notes = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        chars_dict = {}
        
        # 1. ASR & Lab Preparation
        print("[Auto Lyric] Running ASR and preparing labels...")
        for chunk_idx, chunk in enumerate(chunks):
            chunk_wav = chunk["waveform"]
            stem = f"chunk_{chunk_idx}"
            
            chunk_wav_path = temp_dir_path / f"{stem}.wav"
            import soundfile as sf
            sf.write(chunk_wav_path, chunk_wav, sr)
            
            # ASR expects 16000 Hz, so we must resample before inference
            chunk_wav_16k = librosa.resample(chunk_wav, orig_sr=sr, target_sr=16000)
            res = asr_model.generate(input=[chunk_wav_16k], cache={}, is_final=True)
            text = res[0].get('text', '') if isinstance(res[0], dict) else str(res[0])
            
            if not text.strip():
                continue
                
            if matcher is not None:
                asr_text_list, asr_phonetic_list = matcher.process_asr_content(text)
                if asr_phonetic_list:
                    matched_text, matched_phonetic, reason = matcher.align_lyric_with_asr(
                        asr_phonetic=asr_phonetic_list,
                        lyric_text=lyric_text_list,
                        lyric_phonetic=lyric_phonetic_list
                    )
                    if matched_phonetic:
                        pinyin_str = matched_phonetic
                        chars = matched_text.split()
                    else:
                        print(f"  [Warning] {stem}: No match found in original lyrics. Falling back to ASR output.")
                        pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
                        chars = zh_g2p.split_string_no_regex(text)
                else:
                    continue
            else:
                pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
                chars = zh_g2p.split_string_no_regex(text)
            
            chunk_lab_path = temp_dir_path / f"{stem}.lab"
            chunk_lab_path.write_text(pinyin_str, encoding="utf-8")
            chars_dict[stem] = chars

        # 2. HubertFA Forced Alignment
        print("[Auto Lyric] Running HubertFA forced alignment...")
        hfa_model.dataset = []
        hfa_model.predictions = []
        hfa_model.get_dataset(wav_folder=temp_dir, language="zh", g2p="dictionary", dictionary_path=None)
        if len(hfa_model.dataset) > 0:
            hfa_model.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
            
        pred_dict = {p[0].stem: p for p in hfa_model.predictions}
        
        # 3. GAME Inference & Alignment
        print("[Auto Lyric] Extracting pitches with GAME...")
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
            batch_size = 4
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
                    ts=[] # Don't adjust boundaries, trust HFA vowel starts
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
                            # 如果属于同等音高的延音符（n_slur == 1且音高一致，并处于连续时刻），则进行合并
                            if n_slur == 1 and len(all_notes) > 0 and \
                               abs(all_notes[-1].pitch - pitch) < 1e-5 and \
                               abs(all_notes[-1].offset - current_onset) < 1e-5 and \
                               pending_lyric == "":
                                
                                all_notes[-1].offset = current_onset + n_dur
                            else:
                                assigned_lyric = pending_lyric
                                pending_lyric = "" # Consume the lyric on the first valid note
                                
                                all_notes.append(NoteInfo(
                                    onset=current_onset,
                                    offset=current_onset + n_dur,
                                    pitch=pitch,
                                    lyric=assigned_lyric
                                ))
                            
                        current_onset += n_dur

    all_notes.sort(key=lambda x: x.onset)
    print(f"Extracted {len(all_notes)} notes with lyrics.")
    
    # Optional quantization for MIDI
    if quantization_step > 0:
        pass # The callback logic for PyTorch handled this, for ONNX it's not natively supported in _save_midi
             # but standard MIDI tempo quantization is handled implicitly in _save_midi based on tempo.
             # Strict duration quantization can be added later if needed.

    output_key = pathlib.Path(output_filename).stem
    if "mid" in output_formats:
        _save_midi(all_notes, output_dir / f"{output_key}.mid", int(tempo))
    if "txt" in output_formats:
        _save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
    if "csv" in output_formats:
        _save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)
