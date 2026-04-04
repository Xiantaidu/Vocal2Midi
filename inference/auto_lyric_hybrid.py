import os
import pathlib
import tempfile
import librosa
import numpy as np
import warnings
import sys
import torch
import traceback

# Allow running this script directly from anywhere
ROOT_DIR = pathlib.Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.slicer_api import slice_audio
from inference.onnx_api import pad_1d_arrays, NoteInfo, quantize_notes, _save_midi, _save_text
from inference.utils import align_notes_to_words

# Add vendor paths
VENDOR_DIR = pathlib.Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_DIR / "HubertFA"))
# Add GAME repo path
sys.path.insert(0, r'R:\GAME-main')

from inference.vendor.LyricFA.tools.ZhG2p import ZhG2p
from inference.vendor.LyricFA.tools.lyric_matcher import LyricMatcher

_zh_g2p = None

def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_zh_g2p():
    global _zh_g2p
    if _zh_g2p is None:
        _zh_g2p = ZhG2p("mandarin")
    return _zh_g2p

def load_qwen_model(model_path, device="cuda"):
    """
    Loads the Qwen3-ASR model using PyTorch.
    """
    print(f"Loading Qwen3-ASR PyTorch model from '{model_path}' on {device}...")
    from qwen_asr import Qwen3ASRModel

    try:
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
    except Exception as e:
        print(f"Error loading Qwen3-ASR model: {e}")
        print("Please ensure you have run 'pip install -U qwen-asr' in the 'vocal2midi_torch' environment.")
        sys.exit(1)
    return model

def load_game_model(model_dir: str, device="cuda"):
    """
    Loads the GAME model using PyTorch.
    """
    print(f"Loading GAME PyTorch model from '{model_dir}'...")
    from inference.api import load_config_for_inference, load_state_dict_for_inference
    from inference.me_infer import SegmentationEstimationInferenceModel

    model_dir = pathlib.Path(model_dir)
    config_path = model_dir / "config.yaml"
    model_path = model_dir / "model.pt"

    if not config_path.exists() or not model_path.exists():
        print(f"Error: `config.yaml` or `model.pt` not found in {model_dir}")
        sys.exit(1)

    try:
        model_config, inference_config = load_config_for_inference(config_path)
        model = SegmentationEstimationInferenceModel(
            model_config=model_config,
            inference_config=inference_config
        )
        state_dict = load_state_dict_for_inference(model_path, ema=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Error loading GAME model: {e}")
        print("Please ensure the GAME model directory and its contents are correct.")
        sys.exit(1)
    
    print("GAME PyTorch model loaded successfully.")
    return model

def load_hfa_model(model_dir, device="cuda"):
    """
    Loads the HubertFA ONNX model ensuring it runs on the specified device (CUDA).
    """
    print("Loading HubertFA ONNX model for GPU...")
    from onnx_infer import InferenceOnnx
    model = InferenceOnnx(onnx_path=pathlib.Path(model_dir) / 'model.onnx')
    model.load_config()
    model.init_decoder()
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model.model = ort.InferenceSession(str(model.model_folder / 'model.onnx'), options, providers=providers)
    print(f"HubertFA ONNX session created with providers: {model.model.get_providers()}")
    return model

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
            
        vowel_start = word.phonemes[-1].start if len(word.phonemes) > 0 else word.start
        
        if vowel_start > last_end + 0.005:
            word_durs.append(vowel_start - last_end)
            word_vuvs.append(0)
            lyrics.append("")
        elif vowel_start < last_end:
            vowel_start = last_end
            
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

def run_qwen_asr_and_fa(chunks, sr, asr_model, temp_dir_path, zh_g2p, matcher, asr_batch_size=4):
    """
    Runs ASR using the PyTorch Qwen model with batching and prepares .lab files for HubertFA.
    """
    import soundfile as sf
    print(f"[Hybrid Pipeline] Running ASR with PyTorch Qwen (Batch Size: {asr_batch_size})...")

    chars_dict = {}
    chunk_logs = []
    
    audio_paths = []
    chunk_indices = []

    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        chunk_path = temp_dir_path / f"{stem}.wav"
        
        # Save chunk to a temporary file, which is needed for both ASR and FA
        sf.write(chunk_path, chunk["waveform"], sr)

        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)

    all_results = []
    for i in range(0, len(audio_paths), asr_batch_size):
        batch_audio_paths = audio_paths[i:i+asr_batch_size]
        print(f"  Processing ASR batch {i//asr_batch_size + 1}/{(len(audio_paths) - 1)//asr_batch_size + 1}...")
        
        try:
            with torch.cuda.amp.autocast():
                # For batching, `transcribe` expects a list of file paths
                batch_results = asr_model.transcribe(audio=batch_audio_paths, language="Chinese")
            all_results.extend(batch_results)
        except Exception as e:
            print(f"Error during Qwen ASR transcription for batch starting at index {i}: {e}")
            # Add placeholders for failed items in the batch
            all_results.extend([None] * len(batch_audio_paths))

    for idx, res in enumerate(all_results):
        chunk_idx = chunk_indices[idx]
        chunk = chunks[chunk_idx]
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
                    pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
                    chars = zh_g2p.split_string_no_regex(text)
                    match_status = "Fallback to ASR (No match found)"
            else:
                pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
                chars = zh_g2p.split_string_no_regex(text)
                match_status = "Fallback to ASR (No phonetics)"
        else:
            pinyin_str = zh_g2p.convert(text, include_tone=False, convert_number=True)
            chars = zh_g2p.split_string_no_regex(text)
        
        (temp_dir_path / f"{stem}.lab").write_text(pinyin_str, encoding="utf-8")
        chars_dict[stem] = chars
        
        chunk_logs.append(
            f"[{stem}]\nASR Output: {text}\n"
            f"Match Status: {match_status}\n"
            f"Final Assigned Lyrics: {''.join(chars)}\n"
            f"FA Pinyin (.lab): {pinyin_str}\n"
        )
            
    return chars_dict, chunk_logs

def run_hubert_fa(hfa_model, temp_dir):
    """运行 HubertFA 强制对齐"""
    print("[Hybrid Pipeline] Running HubertFA forced alignment on GPU...")
    hfa_model.dataset = []
    hfa_model.predictions = []
    hfa_model.get_dataset(wav_folder=temp_dir, language="zh", g2p="dictionary", dictionary_path=None)
    if len(hfa_model.dataset) > 0:
        hfa_model.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)

    pred_dict = {p[0].stem: p for p in hfa_model.predictions}
    return pred_dict

def extract_pitches_and_align_torch(chunks, sr, pred_dict, chars_dict, game_model, device, ts, seg_threshold, seg_radius, est_threshold, batch_size=4):
    """
    Extracts pitches using the PyTorch GAME model and aligns to lyrics.
    (This version is a direct copy of the logic from the old ONNX pipeline)
    """
    print("[Hybrid Pipeline] Extracting pitches with PyTorch GAME...")
    
    all_notes = []
    batch_infos = []

    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        if stem not in pred_dict:
            continue
        
        _, _, result_word = pred_dict[stem]
        if not result_word:
            continue

        word_durs, word_vuvs, lyrics = extract_vowel_boundaries(result_word, chars_dict.get(stem, []))
        
        batch_infos.append({
            "waveform": chunk["waveform"],
            "word_durs": word_durs,
            "known_durations": np.array(word_durs, dtype=np.float32),
            "offset": chunk["offset"],
            "word_vuvs": word_vuvs,
            "lyrics": lyrics,
        })

    for i in range(0, len(batch_infos), batch_size):
        batch = batch_infos[i:i+batch_size]
        
        waveforms_np = [info["waveform"] for info in batch]
        known_durations_np = [info["known_durations"] for info in batch]

        padded_wavs = torch.from_numpy(pad_1d_arrays(waveforms_np)).to(device)
        padded_kd = torch.from_numpy(pad_1d_arrays(known_durations_np, pad_value=0.0)).to(device)
        
        boundary_threshold = torch.tensor(seg_threshold, device=device)
        boundary_radius = torch.tensor(round(seg_radius / game_model.timestep), device=device, dtype=torch.long)
        score_threshold = torch.tensor(est_threshold, device=device)
        language_tensor = torch.zeros(padded_wavs.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            try:
                durations, presence, scores = game_model(
                    waveform=padded_wavs,
                    known_durations=padded_kd,
                    boundary_threshold=boundary_threshold,
                    boundary_radius=boundary_radius,
                    score_threshold=score_threshold,
                    language=language_tensor,
                    t=ts,
                )
                
                durations = durations.cpu().numpy()
                presence = presence.cpu().numpy()
                scores = scores.cpu().numpy()

            except Exception:
                print(f"Error during GAME model inference batch:")
                traceback.print_exc()
                continue

        for k, info in enumerate(batch):
            c_durations, c_presence, c_scores = durations[k], presence[k], scores[k]
            
            valid_len = len(info['known_durations'])
            c_durations = c_durations[:valid_len]
            c_presence = c_presence[:valid_len]
            c_scores = c_scores[:valid_len]
            
            note_dur = c_durations[c_durations > 0].tolist()
            valid_presence = c_presence[c_durations > 0]
            valid_scores = c_scores[c_durations > 0]
            
            note_seq = [
                librosa.midi_to_note(m, unicode=False, cents=True) if v else "rest"
                for m, v in zip(valid_scores, valid_presence)
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

def auto_lyric_hybrid_pipeline(
    audio_path: str,
    output_filename: str,
    game_model_dir: str,
    device: str,
    hfa_model_dir: str,
    asr_model_path: str,
    ts: torch.Tensor,
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
    batch_size: int = 4,
    asr_batch_size: int = 4
):
    """Auto Lyric Hybrid (PyTorch + ONNX-GPU) Pipeline"""
    output_key = pathlib.Path(output_filename).stem
    print(f"\n[Hybrid Pipeline] Processing audio: {audio_path}")
    sr = 44100
    waveform, sr = librosa.load(audio_path, sr=sr, mono=True)

    chunks = slice_audio(waveform, sr, slicing_method)

    zh_g2p = get_zh_g2p()
    matcher = None
    if original_lyrics and original_lyrics.strip():
        matcher = LyricMatcher(language="zh")
        processor = matcher.processor
        cleaned_lyric = processor.clean_text(original_lyrics)
        matcher.lyric_text_list = processor.split_text(cleaned_lyric)
        matcher.lyric_phonetic_list = processor.get_phonetic_list(matcher.lyric_text_list)

    print("\n--- Loading Models ---")
    asr_model = load_qwen_model(asr_model_path, device=device)
    hfa_model = load_hfa_model(hfa_model_dir, device=device)
    game_model = load_game_model(game_model_dir, device=device)
    print("----------------------\n")
    
    all_notes = []
    chunk_logs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        chars_dict, chunk_logs = run_qwen_asr_and_fa(
            chunks, sr, asr_model, temp_dir_path, zh_g2p, matcher, asr_batch_size
        )
        free_memory()
        del asr_model

        pred_dict = run_hubert_fa(hfa_model, temp_dir_path)

        export_artifacts(chunks, temp_dir_path, hfa_model, output_key, output_dir, output_formats)
        
        del hfa_model
        free_memory()

        all_notes = extract_pitches_and_align_torch(
            chunks, sr, pred_dict, chars_dict, game_model, device, ts,
            seg_threshold, seg_radius, est_threshold, batch_size
        )
        del game_model
        free_memory()

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
    @click.option("--game-model", "-gm", required=True, type=click.Path(exists=True, file_okay=False), help="Path to GAME PyTorch model directory")
    @click.option("--hfa-model", "-hm", required=True, type=click.Path(exists=True, file_okay=False), help="Path to HubertFA ONNX model directory")
    @click.option("--asr-model", "-am", type=str, default="Qwen/Qwen3-ASR-1.7B", help="Path or ID for Qwen3-ASR model")
    @click.option("--output-dir", "-o", type=click.Path(), default=".", help="Directory to save the outputs")
    @click.option("--lyrics", "-l", type=str, default="", help="Original reference lyrics for alignment")
    @click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda", help="Device to run inference on")
    @click.option("--t0", type=float, default=0.0, help="D3PM starting t0")
    @click.option("--nsteps", type=int, default=8, help="D3PM sampling steps")
    def main(audio_path, game_model, hfa_model, asr_model, output_dir, lyrics, device, t0, nsteps, **kwargs):
        """
        Auto Lyric Hybrid Pipeline (PyTorch + ONNX-GPU)
        """
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        step = (1 - t0) / nsteps
        ts_list = [t0 + i * step for i in range(nsteps)]
        ts = torch.tensor(ts_list, device=device)
        
        auto_lyric_hybrid_pipeline(
            audio_path=audio_path,
            output_filename=pathlib.Path(audio_path).name,
            game_model_dir=game_model,
            device=device,
            hfa_model_dir=hfa_model,
            asr_model_path=asr_model,
            ts=ts,
            language="zh",
            original_lyrics=lyrics,
            output_dir=out_dir,
            output_formats=["mid", "txt"], # Simplified for now
            slicing_method="默认切片",
            tempo=120.0, # Simplified
            quantization_step=60, # Simplified
            pitch_format="name", # Simplified
            round_pitch=True, # Simplified
            seg_threshold=0.2,
            seg_radius=0.02,
            est_threshold=0.2,
            batch_size=4
        )
        print("Done!")

    main()
