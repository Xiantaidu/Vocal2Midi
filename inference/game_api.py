import pathlib
import sys
import torch
import numpy as np
import librosa
import traceback

# Add GAME repo path if needed
ROOT_DIR = pathlib.Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.onnx_api import pad_1d_arrays, NoteInfo
from inference.utils import align_notes_to_words

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
        raise RuntimeError(f"Error: `config.yaml` or `model.pt` not found in {model_dir}")

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
        raise RuntimeError(f"Error loading GAME model: {e}\nPlease ensure the GAME model directory and its contents are correct.")
    
    print("GAME PyTorch model loaded successfully.")
    return model

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

def extract_pitches_and_align_torch(chunks, sr, pred_dict, chars_dict, game_model, device, ts, seg_threshold, seg_radius, est_threshold, batch_size=4, debug_mode=False):
    """
    Extracts pitches using the PyTorch GAME model and aligns to lyrics.
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
            "waveform_duration": len(chunk["waveform"]) / sr,
            "word_durs": word_durs,
            "known_durations": np.array(word_durs, dtype=np.float32),
            "offset": chunk["offset"],
            "word_vuvs": word_vuvs,
            "lyrics": lyrics,
        })

    for i in range(0, len(batch_infos), batch_size):
        batch = batch_infos[i:i+batch_size]
        
        waveforms_np = [info["waveform"] for info in batch]
        waveform_durations_np = [info["waveform_duration"] for info in batch]
        known_durations_np = [info["known_durations"] for info in batch]

        padded_wavs = torch.from_numpy(pad_1d_arrays(waveforms_np)).to(device)
        padded_kd = torch.from_numpy(pad_1d_arrays(known_durations_np, pad_value=0.0)).to(device)
        waveform_durations_tensor = torch.tensor(waveform_durations_np, dtype=torch.float32, device=device)
        
        boundary_threshold = torch.tensor(seg_threshold, device=device)
        boundary_radius = torch.tensor(round(seg_radius / game_model.timestep), device=device, dtype=torch.long)
        score_threshold = torch.tensor(est_threshold, device=device)
        language_tensor = torch.zeros(padded_wavs.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            try:
                if debug_mode:
                    print(f"\n[DEBUG GAME INPUTS] Batch {i//batch_size}")
                    print(f"waveform shape: {padded_wavs.shape}")
                    print(f"known_durations shape: {padded_kd.shape}")
                    print(f"boundary_threshold: {boundary_threshold}")
                    print(f"boundary_radius: {boundary_radius}")
                    print(f"score_threshold: {score_threshold}")
                    print(f"language tensor: {language_tensor}")
                    print(f"t tensor: {ts}")

                durations, presence, scores = game_model(
                    waveform=padded_wavs,
                    known_durations=padded_kd,
                    boundary_threshold=boundary_threshold,
                    boundary_radius=boundary_radius,
                    score_threshold=score_threshold,
                    language=language_tensor,
                    t=ts,
                    waveform_durations=waveform_durations_tensor,
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
