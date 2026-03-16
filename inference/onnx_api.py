#!/usr/bin/env python3
"""
API for pure ONNX inference of GAME models.
This API is designed to be a drop-in replacement for the PyTorch-based API
for users who want to use the ONNX-exported models.
"""
import csv
import json
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import librosa
import mido
import numpy as np
import onnxruntime as ort

from inference.slicer2 import Slicer
from inference.utils import validate_phones, parse_words, align_notes_to_words


@dataclass
class NoteInfo:
    onset: float
    offset: float
    pitch: float
    lyric: str = ""


def quantize_notes(notes: list[NoteInfo], tempo: float, quantization_step: int):
    """
    Quantize notes to musical ticks grid.
    tempo: BPM
    quantization_step: ticks per step (e.g., 120 for 1/16 note, where 1 beat = 480 ticks)
    """
    if quantization_step <= 0 or not notes:
        return

    notes.sort(key=lambda n: n.onset)
    
    orig_onsets = [n.onset for n in notes]
    orig_offsets = [n.offset for n in notes]
    
    q_onsets = []
    for onset in orig_onsets:
        ticks = round(onset * tempo * 8)
        q_ticks = round(ticks / quantization_step) * quantization_step
        q_onsets.append(q_ticks)
        
    # Ensure onsets are strictly increasing
    for i in range(1, len(q_onsets)):
        if q_onsets[i] <= q_onsets[i-1]:
            q_onsets[i] = q_onsets[i-1] + quantization_step
            
    q_offsets = []
    for i in range(len(notes)):
        ticks = round(orig_offsets[i] * tempo * 8)
        q_ticks = round(ticks / quantization_step) * quantization_step
        
        if i < len(notes) - 1:
            if abs(orig_offsets[i] - orig_onsets[i+1]) < 1e-3:
                q_ticks = q_onsets[i+1]
                
        if q_ticks <= q_onsets[i]:
            q_ticks = q_onsets[i] + quantization_step
            
        if i < len(notes) - 1:
            if q_ticks > q_onsets[i+1]:
                q_ticks = q_onsets[i+1]
                
        q_offsets.append(q_ticks)
        
    for i in range(len(notes)):
        notes[i].onset = q_onsets[i] / (tempo * 8)
        notes[i].offset = q_offsets[i] / (tempo * 8)

def pad_1d_arrays(arrays: list[np.ndarray], pad_value=0.0) -> np.ndarray:
    """Pad a list of 1D numpy arrays to the maximum length and stack them."""
    if not arrays:
        return np.array([])
    max_len = max(len(arr) for arr in arrays)
    if max_len == 0:
        return np.zeros((len(arrays), 1), dtype=arrays[0].dtype)
    
    padded = []
    for arr in arrays:
        pad_width = max_len - len(arr)
        padded.append(np.pad(arr, (0, pad_width), constant_values=pad_value))
    return np.stack(padded)


class ONNXInferenceModel:
    """
    Manages ONNX inference sessions and the inference process.
    """
    
    def __init__(self, model_dir: pathlib.Path, use_dml: bool = True):
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.samplerate = self.config["samplerate"]
        self.timestep = self.config["timestep"]
        self.loop = self.config.get("loop", True)
        self.languages = self.config.get("languages", None)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable mem_pattern and cpu_mem_arena to prevent VRAM/RAM leakage caused by variable-length inputs
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        
        if use_dml:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.encoder = ort.InferenceSession(
            (model_dir / "encoder.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.segmenter = ort.InferenceSession(
            (model_dir / "segmenter.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.estimator = ort.InferenceSession(
            (model_dir / "estimator.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.dur2bd = ort.InferenceSession(
            (model_dir / "dur2bd.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.bd2dur = ort.InferenceSession(
            (model_dir / "bd2dur.onnx").as_posix(), sess_options=sess_options, providers=providers
        )

    def release(self):
        """Explicitly release ONNX sessions to free memory/VRAM immediately."""
        if hasattr(self, 'encoder'): del self.encoder
        if hasattr(self, 'segmenter'): del self.segmenter
        if hasattr(self, 'estimator'): del self.estimator
        if hasattr(self, 'dur2bd'): del self.dur2bd
        if hasattr(self, 'bd2dur'): del self.bd2dur
    
    def infer_batch(
        self,
        waveforms: np.ndarray,      # [B, L]
        durations: np.ndarray,      # [B]
        known_durations: np.ndarray, # [B, N] or None
        boundary_threshold: float,
        boundary_radius: int,
        score_threshold: float,
        language: int = 0,
        ts: list[float] = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        True batch parallel inference.
        
        Key findings:
        - ONNX encoder automatically generates the correct maskT based on the duration parameter.
        - maskT correctly handles the padding regions in all subsequent modules.
        - As long as correct durations are provided, waveforms can be padded directly without affecting results.
        - This enables true batch parallel inference.
        
        Workflow:
        1. Encoder receives padded waveforms and durations, generates maskT to identify valid frames.
        2. All subsequent modules (segmenter, bd2dur, estimator) use maskT to handle padding.
        3. maskN identifies the number of valid notes for each sample.
        4. Finally, maskN is used to filter the results, returning only valid notes.
        """
        if ts is None:
            ts = []
        
        batch_size = waveforms.shape[0]
        
        # 1. Encoder - Process entire batch directly
        # The encoder automatically generates maskT based on durations to identify valid frames for each sample
        enc_out = self.encoder.run(None, {
            "waveform": waveforms,
            "duration": durations
        })
        x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
        
        # 2. Known Boundaries - Process batch
        if known_durations is not None and known_durations.size > 0:
            known_boundaries = np.zeros_like(maskT, dtype=bool)
            for i in range(batch_size):
                valid_k_durs = known_durations[i][known_durations[i] > 0]
                if len(valid_k_durs) > 0:
                    kd_input = valid_k_durs[np.newaxis, :].astype(np.float32)
                    sample_maskT = maskT[i:i+1]
                    kb = self.dur2bd.run(None, {
                        "durations": kd_input,
                        "maskT": sample_maskT
                    })[0]
                    known_boundaries[i:i+1] = kb
        else:
            known_boundaries = np.zeros_like(maskT, dtype=bool)
        
        # 3. Segmenter Loop - Process batch
        boundaries = known_boundaries.copy()
        lang_arr = np.array([language] * batch_size, dtype=np.int64)
        
        if self.loop and len(ts) > 0:
            for t in ts:
                t_arr = np.array([t] * batch_size, dtype=np.float32)
                boundaries = self.segmenter.run(None, {
                    "x_seg": x_seg,
                    "language": lang_arr,
                    "known_boundaries": known_boundaries,
                    "prev_boundaries": boundaries,
                    "t": t_arr,
                    "maskT": maskT,
                    "threshold": np.array(boundary_threshold, dtype=np.float32),
                    "radius": np.array(boundary_radius, dtype=np.int64),
                })[0]
        else:
            t_arr = np.array([0.0] * batch_size, dtype=np.float32)
            boundaries = self.segmenter.run(None, {
                "x_seg": x_seg,
                "language": lang_arr,
                "known_boundaries": known_boundaries,
                "prev_boundaries": boundaries,
                "t": t_arr,
                "maskT": maskT,
                "threshold": np.array(boundary_threshold, dtype=np.float32),
                "radius": np.array(boundary_radius, dtype=np.int64),
            })[0]
        
        # 4. bd2dur - Process batch
        durations_out, maskN = self.bd2dur.run(None, {
            "boundaries": boundaries,
            "maskT": maskT
        })
        
        # 5. Estimator - Process batch
        presence, scores = self.estimator.run(None, {
            "x_est": x_est,
            "boundaries": boundaries,
            "maskT": maskT,
            "maskN": maskN,
            "threshold": np.array(score_threshold, dtype=np.float32)
        })
        
        # 6. Extract valid results for each sample
        results = []
        for i in range(batch_size):
            valid = maskN[i].astype(bool)
            results.append((
                durations_out[i][valid],
                presence[i][valid],
                scores[i][valid]
            ))
        
        return results


def load_onnx_model(
    model_dir: pathlib.Path,
    device: Literal["dml", "cpu"] = "dml"
) -> ONNXInferenceModel:
    """
    Loads the ONNX models from a directory and initializes the inference engine.
    """
    print("Initializing ONNX sessions...")
    use_dml = (device == "dml")
    return ONNXInferenceModel(model_dir, use_dml=use_dml)


def _save_midi(notes: list[NoteInfo], filepath: pathlib.Path, tempo: int = 120):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    
    sorted_notes = sorted(notes, key=lambda n: n.onset)
    
    last_abs_ticks = 0
    for note in sorted_notes:
        abs_onset_ticks = round(note.onset * tempo * 8)
        abs_offset_ticks = round(note.offset * tempo * 8)
        
        # Prevent overlapping notes causing negative delta_onset_ticks in mido
        if abs_onset_ticks < last_abs_ticks:
            abs_onset_ticks = last_abs_ticks
            
        if abs_offset_ticks <= abs_onset_ticks:
            abs_offset_ticks = abs_onset_ticks + 1
            
        midi_pitch = round(note.pitch)

        delta_onset_ticks = abs_onset_ticks - last_abs_ticks
        note_duration_ticks = abs_offset_ticks - abs_onset_ticks
        
        if getattr(note, "lyric", ""):
            track.append(mido.MetaMessage('lyrics', text=note.lyric, time=delta_onset_ticks))
            track.append(mido.Message("note_on", note=midi_pitch, velocity=100, time=0))
        else:
            track.append(mido.Message("note_on", note=midi_pitch, velocity=100, time=delta_onset_ticks))
            
        track.append(mido.Message("note_off", note=midi_pitch, velocity=100, time=note_duration_ticks))
        
        last_abs_ticks = abs_offset_ticks
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with mido.MidiFile(charset="utf8") as midi_file:
        midi_file.tracks.append(track)
        midi_file.save(filepath)
    print(f"Saved MIDI file: {filepath}")


def _save_text(
    notes: list[NoteInfo],
    filepath: pathlib.Path,
    file_format: Literal["txt", "csv"],
    pitch_format: Literal["number", "name"],
    round_pitch: bool,
):
    onset_list = [f"{note.onset:.3f}" for note in notes]
    offset_list = [f"{note.offset:.3f}" for note in notes]
    pitch_list = []
    for note in notes:
        pitch = note.pitch
        if round_pitch:
            pitch = round(pitch)
        pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not round_pitch) if pitch_format == "name" else f"{pitch:.3f}"
        pitch_list.append(pitch_txt)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lyric_list = [getattr(note, "lyric", "") for note in notes]
    has_lyrics = any(lyric_list)
    
    if file_format == "txt":
        with filepath.open(encoding="utf8", mode="w") as f:
            for onset, offset, pitch, lyric in zip(onset_list, offset_list, pitch_list, lyric_list):
                if has_lyrics:
                    f.write(f"{onset}\t{offset}\t{pitch}\t{lyric}\n")
                else:
                    f.write(f"{onset}\t{offset}\t{pitch}\n")
    elif file_format == "csv":
        with filepath.open(encoding="utf8", mode="w", newline="") as f:
            fieldnames = ["onset", "offset", "pitch"]
            if has_lyrics:
                fieldnames.append("lyric")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for onset, offset, pitch, lyric in zip(onset_list, offset_list, pitch_list, lyric_list):
                row = {"onset": onset, "offset": offset, "pitch": pitch}
                if has_lyrics:
                    row["lyric"] = lyric
                writer.writerow(row)
    print(f"Saved {file_format.upper()} file: {filepath}")


def enforce_max_chunk_size(chunks: list[dict], max_duration_s: float, samplerate: int) -> list[dict]:
    """Further splits chunks that are too long to control VRAM."""
    max_samples = int(max_duration_s * samplerate)
    new_chunks = []
    for chunk in chunks:
        if len(chunk['waveform']) > max_samples:
            original_offset = chunk['offset']
            for i in range(0, len(chunk['waveform']), max_samples):
                sub_chunk_wav = chunk['waveform'][i:i + max_samples]
                new_chunks.append({
                    'waveform': sub_chunk_wav,
                    'offset': original_offset + (i / samplerate)
                })
        else:
            new_chunks.append(chunk)
    return new_chunks

def extract_batch_generator(chunks: list[dict], batch_size: int, samplerate: int, max_batch_duration_s: float = 60.0):
    """Yields batches of padded audio chunks for extraction.
    Chunks are sorted by length to minimize padding overhead.
    Also splits batches if the total duration or memory usage would be too high.
    """
    # Create a local copy to sort
    sorted_chunks = sorted(chunks, key=lambda x: len(x['waveform']), reverse=True)
    
    current_batch = []
    
    for chunk in sorted_chunks:
        current_batch.append(chunk)
        
        # Check if we need to yield
        yield_batch = False
        
        # 1. Batch size limit
        if len(current_batch) >= batch_size:
            yield_batch = True
        # 2. Memory limit heuristic: current_batch_size * max_duration <= max_batch_duration_s
        elif len(current_batch) > 0:
            max_samples = len(current_batch[0]['waveform']) # First one is longest due to sort
            max_duration = max_samples / samplerate
            if max_duration * len(current_batch) > max_batch_duration_s:
                 yield_batch = True
                 
        if yield_batch:
            wavs = [c['waveform'] for c in current_batch]
            padded_wavs = pad_1d_arrays(wavs)
            durations = np.array([len(w) / samplerate for w in wavs], dtype=np.float32)
            yield padded_wavs.astype(np.float32), durations, current_batch
            current_batch = []

    # Yield remaining
    if current_batch:
        wavs = [c['waveform'] for c in current_batch]
        padded_wavs = pad_1d_arrays(wavs)
        durations = np.array([len(w) / samplerate for w in wavs], dtype=np.float32)
        yield padded_wavs.astype(np.float32), durations, current_batch


def infer_from_files(
    model: ONNXInferenceModel,
    filemap: dict[str, pathlib.Path],
    output_dir: pathlib.Path,
    output_formats: set[str],
    language_id: int,
    seg_threshold: float,
    seg_radius: float,
    ts: list[float],
    est_threshold: float,
    pitch_format: str,
    round_pitch: bool,
    tempo: float,
    quantization_step: int = 0,
    batch_size: int = 4,
    max_chunk_duration_s: float = 15.0,
):
    slicer = Slicer(
        sr=model.samplerate,
        threshold=-40.,
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    boundary_radius_frames = round(seg_radius / model.timestep)

    for key, filepath in filemap.items():
        print(f"\nProcessing: {key}")
        waveform, _ = librosa.load(filepath, sr=model.samplerate, mono=True)
        initial_chunks = slicer.slice(waveform)
        
        # Enforce max chunk size to prevent VRAM OOM
        chunks = enforce_max_chunk_size(initial_chunks, max_chunk_duration_s, model.samplerate)
        print(f"  Sliced into {len(chunks)} chunks, batch size: {batch_size}")

        all_notes = []
        
        for batch_wavs, batch_durations, batch_chunks in extract_batch_generator(chunks, batch_size, model.samplerate):
            
            # Run inference on batch
            batch_results = model.infer_batch(
                waveforms=batch_wavs,
                durations=batch_durations,
                known_durations=None,
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            
            # Process results
            for chunk_result, chunk_info in zip(batch_results, batch_chunks):
                durations, presence, scores = chunk_result
                chunk_offset = chunk_info['offset']
                
                note_onset = np.concatenate([[0], np.cumsum(durations[:-1])]) + chunk_offset
                note_offset = np.cumsum(durations) + chunk_offset
                
                for onset, offset, score, is_present in zip(note_onset, note_offset, scores, presence):
                    if offset - onset > 0 and is_present:
                        all_notes.append(NoteInfo(onset=onset, offset=offset, pitch=score))

        all_notes.sort(key=lambda x: x.onset)
        
        if quantization_step > 0:
            quantize_notes(all_notes, tempo, quantization_step)
            
        print(f"  Extracted {len(all_notes)} notes")

        output_key = pathlib.Path(key).stem
        if "mid" in output_formats:
            _save_midi(all_notes, output_dir / f"{output_key}.mid", int(tempo))
        if "txt" in output_formats:
            _save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
        if "csv" in output_formats:
            _save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)

def align_batch_generator(items: list[dict], model_samplerate: int, batch_size: int, max_batch_duration_s: float = 60.0):
    """Yields batches for alignment task. Items are sorted to minimize padding.
    Also splits batches if the total duration or memory usage would be too high.
    """
    # Sort items by file size as a proxy for audio length
    sorted_items = sorted(items, key=lambda x: len(x['wav_fn'].read_bytes()) if isinstance(x.get('wav_fn'), pathlib.Path) else 0, reverse=True)
    
    current_batch = []
    
    for item in sorted_items:
        current_batch.append(item)
        
        # We need to peek at the actual duration of the first (longest) item in the batch
        # To avoid loading it twice, we'll just use a rough estimate if we haven't loaded it,
        # but since we have to load it anyway, let's load on the fly or just use a conservative limit.
        # For simplicity, we'll just check before yielding if we hit batch limit
        if len(current_batch) >= batch_size:
            wavs = []
            durations = []
            word_durs_list = []
            
            for b_item in current_batch:
                waveform, _ = librosa.load(b_item['wav_fn'], sr=model_samplerate, mono=True)
                wavs.append(waveform)
                durations.append(len(waveform) / model_samplerate)
                word_durs_list.append(np.array(b_item['word_dur'], dtype=np.float32))
                
            padded_wavs = pad_1d_arrays(wavs)
            padded_word_durs = pad_1d_arrays(word_durs_list, pad_value=0.0)
            
            yield (
                padded_wavs.astype(np.float32), 
                np.array(durations, dtype=np.float32), 
                padded_word_durs.astype(np.float32), 
                current_batch
            )
            current_batch = []
            continue
            
        # Optional: memory limit heuristic for align
        # Since we don't load the waveform until we yield, we can estimate duration from file size
        # Very rough estimate: 1 byte ~= 1/samplerate * bytes_per_sample seconds
        if len(current_batch) > 0:
            longest_item = current_batch[0]
            if isinstance(longest_item.get('wav_fn'), pathlib.Path):
                file_size_bytes = len(longest_item['wav_fn'].read_bytes())
                # Assume 16-bit PCM mono: 2 bytes per sample
                estimated_max_duration = file_size_bytes / (model_samplerate * 2)
                if estimated_max_duration * len(current_batch) > max_batch_duration_s:
                    wavs = []
                    durations = []
                    word_durs_list = []
                    
                    for b_item in current_batch:
                        waveform, _ = librosa.load(b_item['wav_fn'], sr=model_samplerate, mono=True)
                        wavs.append(waveform)
                        durations.append(len(waveform) / model_samplerate)
                        word_durs_list.append(np.array(b_item['word_dur'], dtype=np.float32))
                        
                    padded_wavs = pad_1d_arrays(wavs)
                    padded_word_durs = pad_1d_arrays(word_durs_list, pad_value=0.0)
                    
                    yield (
                        padded_wavs.astype(np.float32), 
                        np.array(durations, dtype=np.float32), 
                        padded_word_durs.astype(np.float32), 
                        current_batch
                    )
                    current_batch = []

    # Yield remaining
    if current_batch:
        wavs = []
        durations = []
        word_durs_list = []
        
        for b_item in current_batch:
            waveform, _ = librosa.load(b_item['wav_fn'], sr=model_samplerate, mono=True)
            wavs.append(waveform)
            durations.append(len(waveform) / model_samplerate)
            word_durs_list.append(np.array(b_item['word_dur'], dtype=np.float32))
            
        padded_wavs = pad_1d_arrays(wavs)
        padded_word_durs = pad_1d_arrays(word_durs_list, pad_value=0.0)
        
        yield (
            padded_wavs.astype(np.float32), 
            np.array(durations, dtype=np.float32), 
            padded_word_durs.astype(np.float32), 
            current_batch
        )

def align_with_transcriptions(
    model: ONNXInferenceModel,
    transcription_paths: list[pathlib.Path],
    language_id: int,
    seg_threshold: float,
    seg_radius: float,
    ts: list[float],
    est_threshold: float,
    use_wb: bool,
    inplace: bool,
    save_dir: pathlib.Path,
    save_name: str,
    batch_size: int = 4,
    uv_vocab: set[str] | None = None,
    uv_word_cond: Literal["lead", "all"] = "lead",
    uv_note_cond: Literal["predict", "follow"] = "predict",
):
    boundary_radius_frames = round(seg_radius / model.timestep)
    extensions = [".wav", ".flac"]

    for index_path in transcription_paths:
        print(f"\nProcessing: {index_path.as_posix()}")
        
        with open(index_path, "r", encoding="utf8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            items = list(reader)
        
        # Pre-process items to find waveforms and calculate durations
        valid_items = []
        updated_items = [] # We'll build this and insert processed items later
        
        for item in items:
            name = item["name"]
            
            candidate_wav_fns = [index_path.parent / "wavs" / f"{name}{ext}" for ext in extensions]
            wav_fn = next((fn for fn in candidate_wav_fns if fn.is_file()), None)
            
            if wav_fn is None:
                print(f"  - WARNING: Waveform file not found for item '{name}'. Skipping.")
                # We still need to keep it in the final output, just un-updated
                updated_items.append(item)
                continue

            ph_dur = [float(d) for d in item["ph_dur"].split()]
            
            if use_wb and "ph_num" in item and "ph_seq" in item:
                ph_seq = item["ph_seq"].split()
                ph_num = [int(n) for n in item["ph_num"].split()]
                is_valid, err_msg = validate_phones(ph_seq, ph_dur, ph_num)
                if not is_valid:
                    print(f"  - WARNING: Invalid phone sequence in item '{name}': {err_msg}. Using total duration.")
                    word_dur_model = [sum(ph_dur)]
                    word_dur_align = word_dur_model
                    word_vuv_align = [1]
                else:
                    word_dur_model, _ = parse_words(
                        ph_seq, ph_dur, ph_num,
                        uv_vocab=uv_vocab,
                        uv_cond=uv_word_cond,
                        merge_consecutive_uv=uv_vocab is not None,
                    )
                    word_dur_align, word_vuv_align = parse_words(
                        ph_seq, ph_dur, ph_num,
                        uv_vocab=uv_vocab,
                        uv_cond=uv_word_cond,
                        merge_consecutive_uv=False,
                    )
            else:
                word_dur_model = [sum(ph_dur)]
                word_dur_align = word_dur_model
                word_vuv_align = [1]
                
            item['wav_fn'] = wav_fn
            item['word_dur'] = word_dur_model
            item['word_dur_align'] = word_dur_align
            item['word_vuv_align'] = word_vuv_align
            valid_items.append(item)

        # Process valid items in batches
        print(f"  Batch processing {len(valid_items)} items (batch_size={batch_size})...")
        for batch_wavs, batch_durations, batch_known_durations, batch_items in align_batch_generator(valid_items, model.samplerate, batch_size):
            
            batch_results = model.infer_batch(
                waveforms=batch_wavs,
                durations=batch_durations,
                known_durations=batch_known_durations,
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            
            for chunk_result, item in zip(batch_results, batch_items):
                durations, presence, scores = chunk_result
                valid = durations > 0
                
                note_dur = durations[valid].tolist()
                note_midi = scores[valid].tolist()
                note_vuv = presence[valid].tolist()
                
                if use_wb:
                    if uv_note_cond == "follow":
                        note_seq = [
                            librosa.midi_to_note(midi, unicode=False, cents=True)
                            for midi in note_midi
                        ]
                    else:
                        note_seq = [
                            librosa.midi_to_note(midi, unicode=False, cents=True) if vuv else "rest"
                            for midi, vuv in zip(note_midi, note_vuv)
                        ]
                    
                    note_seq, note_dur, note_slur = align_notes_to_words(
                        item["word_dur_align"], item["word_vuv_align"],
                        note_seq, note_dur,
                        apply_word_uv=(uv_note_cond == "follow"),
                    )
                else:
                    note_seq = [
                        librosa.midi_to_note(score, unicode=False, cents=True) if pres else "rest"
                        for score, pres in zip(note_midi, note_vuv)
                    ]
                    note_slur = None
                
                note_dur_str = [f"{dur:.3f}" for dur in note_dur]
                
                # Cleanup temporary fields
                item.pop('wav_fn', None)
                item.pop('word_dur', None)
                item.pop('word_dur_align', None)
                item.pop('word_vuv_align', None)
                
                item["note_seq"] = " ".join(note_seq)
                item["note_dur"] = " ".join(note_dur_str)
                if note_slur is None:
                    item.pop("note_slur", None)
                else:
                    item["note_slur"] = " ".join(str(s) for s in note_slur)
                item.pop("note_glide", None)
                updated_items.append(item)
                
        print(f"  - Processed all items.")

        if inplace:
            output_path = index_path
        elif save_dir is not None and save_name is not None:
             output_path = save_dir / save_name
        else:
            output_path = index_path.parent / (save_name or f"{index_path.stem}-midi.csv")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf8", newline="") as f:
            if "note_seq" not in fieldnames:
                fieldnames.append("note_seq")
            if "note_dur" not in fieldnames:
                fieldnames.append("note_dur")
            if use_wb and "note_slur" not in fieldnames:
                fieldnames.append("note_slur")
            elif not use_wb and "note_slur" in fieldnames:
                fieldnames.remove("note_slur")
            
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(updated_items)
        print(f"  Saved updated transcriptions to: {output_path.as_posix()}")
