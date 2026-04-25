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

from inference.API.slicer_api import slice_audio
from inference.io.note_io import _save_midi, _save_text
from inference.quant.quantization import quantize_notes, should_apply_quantization

from inference.API.asr_api import batch_transcribe_asr
from inference.API.lfa_api import create_lyric_matcher, process_asr_to_phonemes
from inference.API.hfa_api import load_hfa_model, run_hubert_fa, export_hfa_artifacts
from inference.API.game_api import load_game_model, extract_pitches_and_align_torch, extract_pitches_only_torch
from inference.API.rmvpe_api import RmvpeTranscriber
from inference.API.ustx_api import save_ustx

def _resolve_rmvpe_path(model_path: str) -> str:
    """解析 RMVPE 模型路径"""
    if not model_path:
        raise ValueError("RMVPE 模型路径不能为空")
    return model_path

def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    Runs ASR using the PyTorch Qwen model with batching and prepares .lab files for HubertFA.
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

def auto_lyric_hybrid_pipeline(
    audio_path: str,
    output_filename: str,
    game_model_dir: str,
    device: str,
    hfa_model_dir: str,
    asr_model_path: str,
    ts: torch.Tensor,
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
    output_lyrics: bool = True,
    output_pitch_curve: bool = False,
    debug_mode: bool = False,
    rmvpe_model_path: str = "",
    cancel_checker=None,
):
    """Auto Lyric Hybrid (PyTorch + ONNX-GPU) Pipeline"""
    output_key = pathlib.Path(output_filename).stem
    print(f"\n[Hybrid Pipeline] Processing audio: {audio_path}")

    def _check_cancel():
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")

    _check_cancel()
    sr = 44100
    waveform, sr = librosa.load(audio_path, sr=sr, mono=True)
    _check_cancel()

    output_format_set = set(output_formats or [])
    rmvpe_result = None
    if "ustx" in output_format_set and output_pitch_curve:
        rmvpe_model = _resolve_rmvpe_path(rmvpe_model_path)
        print(f"[Hybrid Pipeline] Running RMVPE from: {rmvpe_model}")
        rmvpe = RmvpeTranscriber(rmvpe_model, device=device)
        rmvpe_result = rmvpe.infer(waveform, sr, cancel_checker=cancel_checker)
        print(f"[Hybrid Pipeline] RMVPE done. Frames={len(rmvpe_result.midi_pitch)} step={rmvpe_result.time_step_seconds:.4f}s")

    rmvpe_voiced_mask = None
    rmvpe_step = None
    if rmvpe_result is not None and getattr(rmvpe_result, "voiced_mask", None) is not None:
        rmvpe_voiced_mask = rmvpe_result.voiced_mask
        rmvpe_step = rmvpe_result.time_step_seconds

    chunks = slice_audio(
        waveform,
        sr,
        slicing_method,
        rmvpe_voiced_mask=rmvpe_voiced_mask,
        rmvpe_time_step_seconds=rmvpe_step,
    )
    _check_cancel()

    if output_lyrics:
        matcher = create_lyric_matcher(language, original_lyrics)
        _check_cancel()

        print("\n--- Stage 1/3: Running ASR in subprocess isolation mode ---")
                                        
        free_memory()
        _check_cancel()
        print("-----------------------------------------------------------\n")
    else:
        matcher = None
        print("\n--- No-Lyrics Mode: 跳过 ASR/HFA，仅执行 GAME 提取音高 ---\n")
    
    all_notes = []
    chunk_logs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        pred_dict = {}
        chars_dict = {}

        if output_lyrics:
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
                raise RuntimeError(
                    "ASR 阶段未生成任何有效文本（可能为模型调用失败或返回为空）。"
                    "已中断后续 HFA/GAME，避免生成空结果。"
                )

            free_memory()

            print("\n--- Stage 2/3: Loading HubertFA model ---")
            hfa_model = load_hfa_model(hfa_model_dir, device=device)
            _check_cancel()
            print("------------------------------------------\n")

            pred_dict = run_hubert_fa(hfa_model, temp_dir_path, language=language, cancel_checker=cancel_checker)
            _check_cancel()

            export_hfa_artifacts(
                chunks,
                temp_dir_path,
                hfa_model,
                output_key,
                output_dir,
                output_formats,
                cancel_checker=cancel_checker,
            )

            del hfa_model
            free_memory()

            print("\n--- Stage 3/3: Loading GAME model ---")
        else:
            print("\n--- Stage 1/1: Loading GAME model (No-Lyrics Mode) ---")
        game_model = load_game_model(game_model_dir, device=device)
        _check_cancel()
        print("--------------------------------------\n")

        if output_lyrics:
            all_notes = extract_pitches_and_align_torch(
                chunks, sr, pred_dict, chars_dict, game_model, device, ts,
                seg_threshold, seg_radius, est_threshold, batch_size,
                debug_mode=debug_mode,
                cancel_checker=cancel_checker,
            )
        else:
            all_notes = extract_pitches_only_torch(
                chunks, sr, game_model, device, ts,
                seg_threshold, seg_radius, est_threshold, batch_size,
                debug_mode=debug_mode,
                cancel_checker=cancel_checker,
            )
        _check_cancel()
        del game_model
        free_memory()

    all_notes.sort(key=lambda x: x.onset)
    
                                                     
    export_asr_match_log = output_lyrics and (("asr_match_log" in output_format_set) or ("chunks" in output_format_set))
    if export_asr_match_log:
        log_path = output_dir / f"{output_key}_asr_match_log.txt"
        log_path.write_text("\n".join(chunk_logs), encoding="utf-8")

    if should_apply_quantization(quantization_mode, quantization_step):
        quantize_notes(all_notes, tempo, quantization_step, mode=quantization_mode)
    
    print(f"Extracted {len(all_notes)} notes with lyrics.")

    if "mid" in output_formats:
        _save_midi(all_notes, output_dir / f"{output_key}.mid", int(tempo))
    if "txt" in output_formats:
        _save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
    if "csv" in output_formats:
        _save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)
    if "ustx" in output_formats:
        save_ustx(all_notes, output_dir / f"{output_key}.ustx", tempo=float(tempo), rmvpe_result=rmvpe_result)


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
    @click.option("--debug", is_flag=True, help="Enable debug mode to print GAME inputs")
    def main(audio_path, game_model, hfa_model, asr_model, output_dir, lyrics, device, t0, nsteps, debug, **kwargs):
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
            language="ja",  # Will use the UI parameter when integrated
            lyric_output_mode="romaji",
            original_lyrics=lyrics,
            output_dir=out_dir,
            output_formats=["mid", "txt"], # Simplified for now
            slicing_method="默认切片",
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
