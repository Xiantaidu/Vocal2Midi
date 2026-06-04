import pathlib
import sys
import tempfile

import librosa
from application.config import (
    DEFAULT_SLICE_MAX_SEC,
    DEFAULT_SLICE_MIN_SEC,
    validate_slice_bounds,
)

# Allow running this script directly from anywhere
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.API.slicer_api import slice_audio_with_custom_bounds as slice_audio
from inference.io.note_io import _save_midi, _save_text
from inference.quant.quantization import quantize_notes, should_apply_quantization

from inference.API.asr_api import batch_transcribe_asr, batch_transcribe_romaji_asr
from inference.API.lfa_api import create_lyric_matcher, process_asr_to_phonemes, _normalize_lyric_output_mode
from inference.API.hfa_api import load_hfa_model, run_hubert_fa, export_hfa_artifacts
from inference.API.game_api import load_game_model, extract_pitches_and_align_torch, extract_pitches_only_torch
from inference.API.rmvpe_api import RmvpeTranscriber
from inference.API.ustx_api import save_ustx
from inference.device_utils import RUNTIME_DEVICE_CHOICES, normalize_runtime_device

ROMAJI_ASR_DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "romajiASR"


def _normalize_output_formats(output_formats) -> list[str]:
    if output_formats is None:
        return []
    if isinstance(output_formats, str):
        return [output_formats.lower()]
    return [str(fmt).lower() for fmt in output_formats if fmt]


def _resolve_output_key(output_filename: str, audio_path: str) -> str:
    source_name = output_filename or pathlib.Path(audio_path).name
    output_key = pathlib.Path(source_name).stem
    if not output_key:
        raise ValueError("输出文件名不能为空")
    return output_key


def _select_romaji_asr_path(phoneme_asr_model_path: str) -> str | None:
    if phoneme_asr_model_path:
        return phoneme_asr_model_path
    if ROMAJI_ASR_DEFAULT_DIR.exists():
        return str(ROMAJI_ASR_DEFAULT_DIR)
    return None


def _select_phoneme_asr_path(phoneme_asr_model_path: str) -> str | None:
    return _select_romaji_asr_path(phoneme_asr_model_path)


def _validate_runtime_options(tempo: float, batch_size: int, asr_batch_size: int) -> None:
    if tempo <= 0:
        raise ValueError(f"tempo 必须大于 0，当前为 {tempo}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，当前为 {batch_size}")
    if asr_batch_size <= 0:
        raise ValueError(f"asr_batch_size 必须大于 0，当前为 {asr_batch_size}")


def _validate_slice_runtime_options(
    tempo: float,
    batch_size: int,
    asr_batch_size: int,
    slice_min_sec: float,
    slice_max_sec: float,
) -> None:
    _validate_runtime_options(tempo, batch_size, asr_batch_size)
    validate_slice_bounds(slice_min_sec, slice_max_sec)


def _export_chunk_wavs(chunks, sr: int, output_key: str, output_dir: pathlib.Path, cancel_checker=None) -> None:
    import soundfile as sf

    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("切片导出任务已取消")
        sf.write(output_dir / f"{output_key}_{chunk_idx:03d}.wav", chunk["waveform"], sr)


def _resolve_rmvpe_path(model_path: str) -> str:
    """Resolve the RMVPE model path."""
    if not model_path:
        raise ValueError("RMVPE 模型路径不能为空")
    return model_path


def free_memory():
    import gc
    gc.collect()

def run_qwen_asr_and_fa(
    chunks,
    sr,
    temp_dir_path,
    matcher,
    asr_model_path,
    device,
    asr_batch_size=4,
    language="zh",
    lyric_output_mode=None,
    cancel_checker=None,
):
    """
    Runs ASR using the Qwen DML+CPU runtime with batching and prepares .lab files for HubertFA.
    """
    all_results, chunk_indices = batch_transcribe_asr(
        chunks,
        sr,
        asr_model=None,
        temp_dir_path=temp_dir_path,
        asr_batch_size=asr_batch_size,
        language=language,
        cancel_checker=cancel_checker,
        asr_model_path=asr_model_path,
        device=device,
        force_subprocess=True,
        asr_timeout_sec=180,
    )
    return process_asr_to_phonemes(
        all_results,
        chunk_indices,
        temp_dir_path,
        language,
        matcher,
        lyric_output_mode=lyric_output_mode,
    )


def run_romaji_asr(
    chunks,
    sr,
    temp_dir_path,
    matcher,
    asr_model_path,
    device,
    language="ja",
    lyric_output_mode=None,
    asr_batch_size=1,
    cancel_checker=None,
):
    all_results, chunk_indices = batch_transcribe_romaji_asr(
        chunks,
        sr,
        temp_dir_path=temp_dir_path,
        model_dir=asr_model_path,
        device=device,
        asr_batch_size=asr_batch_size,
        cancel_checker=cancel_checker,
    )
    chars_dict, chunk_logs = process_asr_to_phonemes(
        all_results,
        chunk_indices,
        temp_dir_path,
        language,
        matcher,
        lyric_output_mode=lyric_output_mode,
        use_asr_phonemes=True,
    )
    return chars_dict, chunk_logs


def run_phoneme_asr_and_fa(*args, **kwargs):
    return run_romaji_asr(*args, **kwargs)

def auto_lyric_hybrid_pipeline(
    audio_path: str,
    output_filename: str,
    game_model_dir: str,
    device: str,
    hfa_model_dir: str,
    asr_model_path: str,
    ts: list[float],
    language: str,
    lyric_output_mode: str,
    original_lyrics: str,
    output_dir: pathlib.Path,
    output_formats: list,
    slicing_method: str,
    tempo: float,
    quantization_step: int,
    pitch_format: str,
    round_pitch: bool,
    quantization_mode: str,
    seg_threshold: float,
    seg_radius: float,
    est_threshold: float,
    batch_size: int = 4,
    asr_batch_size: int = 4,
    slice_min_sec: float = DEFAULT_SLICE_MIN_SEC,
    slice_max_sec: float = DEFAULT_SLICE_MAX_SEC,
    output_lyrics: bool = True,
    output_pitch_curve: bool = False,
    debug_mode: bool = False,
    rmvpe_model_path: str = "",
    phoneme_asr_model_path: str = "",
    cancel_checker=None,
):
    """Auto Lyric Hybrid ONNX pipeline."""
    device = normalize_runtime_device(device)
    _validate_slice_runtime_options(tempo, batch_size, asr_batch_size, slice_min_sec, slice_max_sec)
    output_key = _resolve_output_key(output_filename, audio_path)
    output_formats = _normalize_output_formats(output_formats)
    output_format_set = set(output_formats)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    language = (language or "zh").lower()
    lyric_output_mode = _normalize_lyric_output_mode(language, lyric_output_mode)
    print(f"\n[Hybrid Pipeline] Processing audio: {audio_path}")

    def _check_cancel():
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")

    _check_cancel()
    sr = 44100
    waveform, sr = librosa.load(audio_path, sr=sr, mono=True)
    _check_cancel()

    rmvpe_result = None
    if "ustx" in output_format_set and output_pitch_curve:
        rmvpe_model = _resolve_rmvpe_path(rmvpe_model_path)
        print(f"[Hybrid Pipeline] Running RMVPE from: {rmvpe_model}")
        rmvpe = RmvpeTranscriber(rmvpe_model, device=device)
        try:
            rmvpe_result = rmvpe.infer(waveform, sr, cancel_checker=cancel_checker)
            print(f"[Hybrid Pipeline] RMVPE done. Frames={len(rmvpe_result.midi_pitch)} step={rmvpe_result.time_step_seconds:.4f}s")
        finally:
            del rmvpe
            free_memory()

    rmvpe_voiced_mask = None
    rmvpe_step = None
    if rmvpe_result is not None and getattr(rmvpe_result, "voiced_mask", None) is not None:
        rmvpe_voiced_mask = rmvpe_result.voiced_mask
        rmvpe_step = rmvpe_result.time_step_seconds

    chunks = slice_audio(
        waveform,
        sr,
        slicing_method,
        min_len_sec=slice_min_sec,
        max_len_sec=slice_max_sec,
        rmvpe_voiced_mask=rmvpe_voiced_mask,
        rmvpe_time_step_seconds=rmvpe_step,
    )
    _check_cancel()
    if not chunks:
        raise RuntimeError("切片阶段未生成任何音频片段，已中断后续处理。")

    if output_lyrics:
        matcher = create_lyric_matcher(language, original_lyrics)
        _check_cancel()
        free_memory()
        _check_cancel()
    else:
        matcher = None
        print("\n--- No-Lyrics Mode: 跳过 ASR/HFA，仅执行 GAME 提取音高 ---\n")
    
    all_notes = []
    chunk_logs = []
    run_lyric_alignment = output_lyrics

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        pred_dict = {}
        chars_dict = {}

        if run_lyric_alignment:
            use_phoneme_asr = language == "ja" and lyric_output_mode in {"romaji", "kana"}
            phoneme_asr_path = None
            if use_phoneme_asr:
                print("\n--- Stage 1/3: Running mora ASR for Japanese lyric mode ---")
                phoneme_asr_path = _select_romaji_asr_path(phoneme_asr_model_path)
                if phoneme_asr_path is None:
                    print(
                        "[Warning] Romaji ASR model not found; "
                        "falling back to text ASR + Japanese G2P."
                    )
                    use_phoneme_asr = False

            if use_phoneme_asr:
                chars_dict, chunk_logs = run_romaji_asr(
                    chunks,
                    sr,
                    temp_dir_path,
                    matcher,
                    asr_model_path=phoneme_asr_path,
                    device=device,
                    language=language,
                    lyric_output_mode=lyric_output_mode,
                    asr_batch_size=asr_batch_size,
                    cancel_checker=cancel_checker,
                )
            else:
                if language == "ja" and lyric_output_mode in {"romaji", "kana"}:
                    print("\n--- Stage 1/3: Mora ASR unavailable; fallback to text ASR + Japanese G2P ---")
                else:
                    print("\n--- Stage 1/3: Running ASR in subprocess isolation mode ---")
                chars_dict, chunk_logs = run_qwen_asr_and_fa(
                    chunks,
                    sr,
                    temp_dir_path,
                    matcher,
                    asr_model_path=asr_model_path,
                    device=device,
                    asr_batch_size=asr_batch_size,
                    language=language,
                    lyric_output_mode=lyric_output_mode,
                    cancel_checker=cancel_checker,
                )
            _check_cancel()

            if not chars_dict:
                print(
                    "[Warning] ASR did not produce valid text for any chunk; "
                    "falling back to GAME pitch-only extraction."
                )
                run_lyric_alignment = False

            free_memory()

            if run_lyric_alignment:
                print("\n--- Stage 2/3: Loading HubertFA model ---")
                hfa_model = load_hfa_model(hfa_model_dir, device=device)
                try:
                    _check_cancel()
                    print("------------------------------------------\n")

                    pred_dict = run_hubert_fa(
                        hfa_model,
                        temp_dir_path,
                        language=language,
                        cancel_checker=cancel_checker,
                    )
                    _check_cancel()
                    if not pred_dict:
                        print(
                            "[Warning] HFA did not produce alignment for any chunk; "
                            "falling back to GAME pitch-only extraction."
                        )
                        run_lyric_alignment = False
                    else:
                        missing_hfa = sorted(set(chars_dict) - set(pred_dict))
                        if missing_hfa:
                            preview = ", ".join(missing_hfa[:8])
                            suffix = " ..." if len(missing_hfa) > 8 else ""
                            print(
                                f"[Warning] HFA missing {len(missing_hfa)} chunk(s) "
                                f"({preview}{suffix}); those chunks will use pitch-only fallback."
                            )

                        export_hfa_artifacts(
                            chunks,
                            temp_dir_path,
                            hfa_model,
                            output_key,
                            output_dir,
                            output_formats,
                            cancel_checker=cancel_checker,
                        )
                finally:
                    del hfa_model
                    free_memory()

            if run_lyric_alignment:
                print("\n--- Stage 3/3: Loading GAME model ---")
            else:
                if "chunks" in output_format_set:
                    _export_chunk_wavs(chunks, sr, output_key, output_dir, cancel_checker=cancel_checker)
                print("\n--- Fallback: Loading GAME model (pitch-only mode) ---")
        else:
            if "chunks" in output_format_set:
                _export_chunk_wavs(chunks, sr, output_key, output_dir, cancel_checker=cancel_checker)
            print("\n--- Stage 1/1: Loading GAME model (No-Lyrics Mode) ---")
        game_model = load_game_model(game_model_dir, device=device)
        try:
            _check_cancel()
            print("--------------------------------------\n")

            if run_lyric_alignment:
                aligned_result = extract_pitches_and_align_torch(
                    chunks, sr, pred_dict, chars_dict, game_model, device, ts,
                    seg_threshold, seg_radius, est_threshold, batch_size,
                    debug_mode=debug_mode,
                    cancel_checker=cancel_checker,
                    language=language,
                )
                if isinstance(aligned_result, tuple):
                    all_notes, processed_aligned_chunks = aligned_result
                else:
                    all_notes = aligned_result
                    processed_aligned_chunks = set()
                fallback_chunks = []
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk_idx not in processed_aligned_chunks:
                        fallback_chunks.append(chunk)
                if fallback_chunks:
                    print(
                        f"[Warning] Running pitch-only GAME fallback for "
                        f"{len(fallback_chunks)} chunk(s) without usable lyric alignment."
                    )
                    all_notes.extend(
                        extract_pitches_only_torch(
                            fallback_chunks, sr, game_model, device, ts,
                            seg_threshold, seg_radius, est_threshold, batch_size,
                            debug_mode=debug_mode,
                            cancel_checker=cancel_checker,
                            language=language,
                        )
                    )
            else:
                all_notes = extract_pitches_only_torch(
                    chunks, sr, game_model, device, ts,
                    seg_threshold, seg_radius, est_threshold, batch_size,
                    debug_mode=debug_mode,
                    cancel_checker=cancel_checker,
                    language=language,
                )
            _check_cancel()
        finally:
            del game_model
            free_memory()

    all_notes.sort(key=lambda x: x.onset)
    
                                                     
    export_asr_match_log = output_lyrics and (("asr_match_log" in output_format_set) or ("chunks" in output_format_set))
    if export_asr_match_log:
        log_path = output_dir / f"{output_key}_asr_match_log.txt"
        log_path.write_text("\n".join(chunk_logs), encoding="utf-8")

    if should_apply_quantization(quantization_mode, quantization_step):
        quantize_notes(all_notes, tempo, quantization_step, mode=quantization_mode)
    
    lyric_status = "with lyrics" if run_lyric_alignment else "without lyrics"
    print(f"Extracted {len(all_notes)} notes {lyric_status}.")

    if "mid" in output_format_set:
        _save_midi(all_notes, output_dir / f"{output_key}.mid", int(tempo))
    if "txt" in output_format_set:
        _save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
    if "csv" in output_format_set:
        _save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)
    if "ustx" in output_format_set:
        save_ustx(all_notes, output_dir / f"{output_key}.ustx", tempo=float(tempo), rmvpe_result=rmvpe_result)


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("audio_path", type=click.Path(exists=True))
    @click.option("--game-model", "-gm", required=True, type=click.Path(exists=True, file_okay=False), help="Path to GAME ONNX model directory")
    @click.option("--hfa-model", "-hm", required=True, type=click.Path(exists=True, file_okay=False), help="Path to HubertFA ONNX model directory")
    @click.option("--asr-model", "-am", type=str, default="experiments/Qwen3-ASR-1.7B-dml", help="Path for the local Qwen3-ASR DML model directory")
    @click.option("--output-dir", "-o", type=click.Path(), default=".", help="Directory to save the outputs")
    @click.option("--lyrics", "-l", type=str, default="", help="Original reference lyrics for alignment")
    @click.option("--device", type=click.Choice(list(RUNTIME_DEVICE_CHOICES)), default="dml", help="Runtime device (legacy 'cuda' maps to 'dml')")
    @click.option("--t0", type=float, default=0.0, help="D3PM starting t0")
    @click.option("--nsteps", type=int, default=8, help="D3PM sampling steps")
    @click.option("--debug", is_flag=True, help="Enable debug mode to print GAME inputs")
    def main(audio_path, game_model, hfa_model, asr_model, output_dir, lyrics, device, t0, nsteps, debug, **kwargs):
        """
        Auto Lyric Hybrid ONNX pipeline
        """
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        step = (1 - t0) / nsteps
        ts_list = [t0 + i * step for i in range(nsteps)]
        device = normalize_runtime_device(device)
        ts = ts_list
        
        auto_lyric_hybrid_pipeline(
            audio_path=audio_path,
            output_filename=pathlib.Path(audio_path).name,
            game_model_dir=game_model,
            device=device,
            hfa_model_dir=hfa_model,
            asr_model_path=asr_model,
            ts=ts,
            language="ja",  # Will use the UI parameter when integrated
            lyric_output_mode="romaji",
            original_lyrics=lyrics,
            output_dir=out_dir,
            output_formats=["mid", "txt"], # Simplified for now
            slicing_method="default",
            tempo=120.0, # Simplified
            quantization_step=60, # Simplified
            pitch_format="name", # Simplified
            quantization_mode="simple",
            round_pitch=True, # Simplified
            seg_threshold=0.2,
            seg_radius=0.02,
            est_threshold=0.2,
            batch_size=4,
            debug_mode=debug
        )
        print("Done!")

    main()
