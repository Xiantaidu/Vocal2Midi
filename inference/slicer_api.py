import librosa
import numpy as np
import itertools

from inference.slicer2 import Slicer

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
    split_threshold_db: float = -30.0
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
        else:
             print(f"  Discarding short segment at {segment['offset']:.2f}s (duration {seg_dur:.2f}s < {min_len_sec}s)")

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

def slice_audio(waveform: np.ndarray, sr: int, method: str):
    """
    Top-level API for slicing audio with different methods.
    """
    print(f"Slicing audio with method: '{method}'")
    if method == "启发式切片":
        chunks = heuristic_slice(waveform, sr)
    elif method == "网格搜索切片":
        chunks = grid_search_slice(waveform, sr)
    else: # "默认切片"
        chunks = default_slice(waveform, sr)
    
    print(f"Sliced into {len(chunks)} chunks.")
    return chunks
