import librosa
import numpy as np
import itertools
import torch
import functools
from concurrent.futures import ProcessPoolExecutor

from inference.slicer.slicer2 import Slicer


def _concat_waveforms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = -1 if a.ndim > 1 else 0
    return np.concatenate([a, b], axis=axis)


def _merge_segments(left: dict, right: dict) -> dict:
    """Merge two consecutive segments into one."""
    return {
        'offset': left['offset'],
        'waveform': _concat_waveforms(left['waveform'], right['waveform'])
    }


def _merge_tiny_chunks(chunks: list, sr: int, tiny_sec: float = 0.35) -> list:
    """Merge tiny chunks into adjacent chunks instead of discarding them."""
    if not chunks:
        return chunks

    merged = []
    pending_head = None

    for seg in chunks:
        seg_dur = len(seg['waveform']) / sr

        if seg_dur < tiny_sec:
            if merged:
                print(f"  Merging tiny segment at {seg['offset']:.2f}s ({seg_dur:.2f}s) into previous chunk")
                merged[-1] = _merge_segments(merged[-1], seg)
            else:
                if pending_head is None:
                    pending_head = seg
                else:
                    pending_head = _merge_segments(pending_head, seg)
            continue

        if pending_head is not None:
            pdur = len(pending_head['waveform']) / sr
            print(f"  Merging leading tiny segment at {pending_head['offset']:.2f}s ({pdur:.2f}s) into next chunk")
            seg = _merge_segments(pending_head, seg)
            pending_head = None

        merged.append(seg)

    if pending_head is not None:
        # No neighbor chunk available (e.g., all chunks are tiny), keep it to avoid data loss.
        print(f"  Keeping isolated tiny segment at {pending_head['offset']:.2f}s (no neighbor to merge)")
        merged.append(pending_head)

    return merged

def get_pitch_curve(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    f0_min: float = 65.0,
    f0_max: float = 1100.0,
    voiced_flag_override: np.ndarray | None = None,
    voiced_flag_override_step_sec: float | None = None,
) -> np.ndarray:
    """Calculate the pitch curve (F0) and voicing confidence."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    if voiced_flag_override is not None and voiced_flag_override_step_sec and voiced_flag_override_step_sec > 0:
        target_frames = int(np.ceil(len(y) / hop_length)) + 1
        times = np.arange(target_frames, dtype=np.float64) * (hop_length / sr)
        src_idx = np.round(times / voiced_flag_override_step_sec).astype(np.int64)
        src_idx = np.clip(src_idx, 0, len(voiced_flag_override) - 1)
        voiced_flag = np.asarray(voiced_flag_override, dtype=bool)[src_idx]
        # Return a dummy f0 to keep signature compatibility; split logic only needs voiced_flag.
        return np.zeros_like(voiced_flag, dtype=np.float32), voiced_flag
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        frame_length=hop_length * 4, # pyin requires a larger frame
        hop_length=hop_length
    )
    # Set unvoiced frames to 0
    f0[~voiced_flag] = 0
    return f0, voiced_flag

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

    thresholds = [-45, -40, -35, -30, -25, -20] 
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
            
            score = 0
            num_short = 0
            num_long = 0
            
            for d in durations:
                if d < min_len_sec:
                    score += (min_len_sec - d) * 1.5 
                    num_short += 1
                elif d > max_len_sec:
                    score += (d - max_len_sec)
                    num_long += 1
            
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
    split_threshold_db: float = -30.0,
    ultra_short_sec: float = 0.35,
):
    """
    A two-stage heuristic slicer.
    """
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
            for sub in sub_chunks:
                sub['offset'] += segment['offset']
                final_chunks.append(sub)
        elif seg_dur >= min_len_sec:
            final_chunks.append(segment)
        elif seg_dur >= ultra_short_sec:
            print(f"  Keeping short segment at {segment['offset']:.2f}s (duration {seg_dur:.2f}s < {min_len_sec}s) to avoid lyric loss")
            final_chunks.append(segment)
        else:
             print(f"  Keeping ultra-short segment at {segment['offset']:.2f}s (duration {seg_dur:.2f}s < {ultra_short_sec}s) for later merge")
             final_chunks.append(segment)

    final_chunks.sort(key=lambda x: x['offset'])
    final_chunks = _merge_tiny_chunks(final_chunks, sr=sr, tiny_sec=ultra_short_sec)
    return final_chunks

def _split_wrapper(segment, slicer_func):
    """
    A wrapper function for parallel processing.
    It takes a segment dictionary and a slicing function, and returns the processed chunks.
    """
    sub_chunks = slicer_func(segment['waveform'])
    # Add the original offset to each sub-chunk
    for sub in sub_chunks:
        sub['offset'] += segment['offset']
    return sub_chunks


def _pitch_based_split(
    waveform: np.ndarray,
    sr: int,
    min_len_sec: float,
    max_len_sec: float,
    hop_length: int,
    voiced_flag_override: np.ndarray | None = None,
    voiced_flag_override_step_sec: float | None = None,
):
    """
    Internal helper for splitting a single, long audio segment based on pitch.
    """
    total_samples = waveform.shape[-1]
    total_sec = total_samples / sr

    if total_sec <= max_len_sec:
        return [{'offset': 0.0, 'waveform': waveform}]

    f0, voiced_flag = get_pitch_curve(
        waveform,
        sr=sr,
        hop_length=hop_length,
        voiced_flag_override=voiced_flag_override,
        voiced_flag_override_step_sec=voiced_flag_override_step_sec,
    )
    
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

        start_frame = max(0, min(start_frame, len(voiced_flag) - 1))
        end_frame = max(0, min(end_frame, len(voiced_flag)))
        
        cut_frame = -1
        if start_frame < end_frame:
            unvoiced_indices = np.where(~voiced_flag[start_frame:end_frame])[0]
            if len(unvoiced_indices) > 0:
                # Find the longest continuous unvoiced segment
                max_len = 0
                best_start = -1
                for k, g in itertools.groupby(enumerate(unvoiced_indices), lambda i_x: i_x[0] - i_x[1]):
                    group = list(map(lambda i_x: i_x[1], g))
                    if len(group) > max_len:
                        max_len = len(group)
                        best_start = group[0]
                
                # Cut in the middle of the longest unvoiced segment
                if best_start != -1:
                    cut_idx_in_window = best_start + max_len // 2
                    cut_frame = start_frame + cut_idx_in_window
                    cut_type = "Pitch-based (unvoiced)"

        # Fallback to RMS-based splitting if no good unvoiced segment is found
        if cut_frame == -1:
            rms_db = get_rms_db(waveform, frame_length=hop_length*4, hop_length=hop_length)
            start_frame_rms = librosa.time_to_frames(window_start_sec, sr=sr, hop_length=hop_length)
            end_frame_rms = librosa.time_to_frames(window_end_sec, sr=sr, hop_length=hop_length)
            start_frame_rms = max(0, min(start_frame_rms, len(rms_db) - 1))
            end_frame_rms = max(0, min(end_frame_rms, len(rms_db)))
            
            if start_frame_rms < end_frame_rms:
                window_rms = rms_db[start_frame_rms:end_frame_rms]
                best_idx_in_window = np.argmin(window_rms)
                cut_frame = start_frame_rms + best_idx_in_window
                cut_type = f"Fallback to RMS (Local Min {window_rms[best_idx_in_window]:.1f}dB)"
            else: # Should not happen often
                cut_frame = end_frame
                cut_type = "Fallback (end of window)"

        cut_sec = librosa.frames_to_time(cut_frame, sr=sr, hop_length=hop_length)
        
        start_sample = int(current_start_sec * sr)
        end_sample = int(cut_sec * sr)
        
        chunk_wav = waveform[:, start_sample:end_sample] if waveform.ndim > 1 else waveform[start_sample:end_sample]
        
        dur = cut_sec - current_start_sec
        print(f"    Sub-split at {cut_sec:.2f}s (duration {dur:.2f}s) - Reason: {cut_type}")
        
        chunks.append({'offset': current_start_sec, 'waveform': chunk_wav})
        current_start_sec = cut_sec

    return chunks

def pitch_based_slice(
    waveform: np.ndarray,
    sr: int,
    min_len_sec: float = 4.0,
    max_len_sec: float = 12.0,
    silence_removal_threshold_db: float = -40.0,
    min_silence_len_ms: int = 800,
    ultra_short_sec: float = 0.35,
    voiced_flag_override: np.ndarray | None = None,
    voiced_flag_override_step_sec: float | None = None,
):
    """
    A two-stage slicer using pitch information.
    """
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

    final_chunks = []
    print("\nStage 2: Splitting long segments based on pitch...")
    
    short_segments = []
    long_segments = []
    for seg in vocal_segments:
        seg_dur = len(seg['waveform']) / sr
        if seg_dur > max_len_sec:
            long_segments.append(seg)
        elif seg_dur >= min_len_sec:
            short_segments.append(seg)
        elif seg_dur >= ultra_short_sec:
            print(f"  Keeping short segment at {seg['offset']:.2f}s (duration {seg_dur:.2f}s < {min_len_sec}s) to avoid lyric loss")
            short_segments.append(seg)
        else:
            print(f"  Keeping ultra-short segment at {seg['offset']:.2f}s (duration {seg_dur:.2f}s < {ultra_short_sec}s) for later merge")
            short_segments.append(seg)
    
    if long_segments:
        print(f"  Found {len(long_segments)} long segments to split in parallel...")
        
        # Create a partial function with fixed arguments
        slicer_func = functools.partial(
            _pitch_based_split,
            sr=sr,
            min_len_sec=min_len_sec,
            max_len_sec=max_len_sec,
            hop_length=512,
            voiced_flag_override=voiced_flag_override,
            voiced_flag_override_step_sec=voiced_flag_override_step_sec,
        )
        
        # Create a wrapper for the slicer function to handle segment offsets
        split_task = functools.partial(_split_wrapper, slicer_func=slicer_func)

        with ProcessPoolExecutor() as executor:
            # Map the task to all long segments
            processed_chunks_iter = executor.map(split_task, long_segments)
            
            # Flatten the list of lists
            processed_chunks = [chunk for sublist in processed_chunks_iter for chunk in sublist]

        final_chunks = short_segments + processed_chunks
        # Sort by offset to maintain original order
        final_chunks.sort(key=lambda x: x['offset'])
    else:
        final_chunks = short_segments

    final_chunks = _merge_tiny_chunks(final_chunks, sr=sr, tiny_sec=ultra_short_sec)
    return final_chunks

def default_slice(waveform, sr):
    """The default slicing method from Slicer."""
    slicer = Slicer(
        sr=sr,
        threshold=-30.,
        min_length=5000,
        max_sil_kept=500,
    )
    return slicer.slice(waveform)

def slice_audio(
    waveform: np.ndarray,
    sr: int,
    method: str,
    rmvpe_voiced_mask: np.ndarray | None = None,
    rmvpe_time_step_seconds: float | None = None,
):
    """
    Top-level API for slicing audio with different methods.
    """
    print(f"Slicing audio with method: '{method}'")
    if method == "智能切片":
        if rmvpe_voiced_mask is not None and rmvpe_time_step_seconds and rmvpe_time_step_seconds > 0:
            print("Using RMVPE voiced mask for smart slicing (pyin fallback disabled for this run).")
        chunks = pitch_based_slice(
            waveform,
            sr,
            voiced_flag_override=rmvpe_voiced_mask,
            voiced_flag_override_step_sec=rmvpe_time_step_seconds,
        )
    elif method == "启发式切片":
        chunks = heuristic_slice(waveform, sr)
    elif method == "网格搜索切片":
        chunks = grid_search_slice(waveform, sr)
    else: # "默认切片"
        chunks = default_slice(waveform, sr)
    
    print(f"Sliced into {len(chunks)} chunks.")
    return chunks
