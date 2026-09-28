import pathlib
import sys
from dataclasses import dataclass

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
from inference.io.audio_io import load_audio
from inference.io.note_io import _save_midi, _save_text
from inference.quant.quantization import quantize_notes, should_apply_quantization

from inference.API.lfa_api import create_lyric_matcher, _normalize_lyric_output_mode
from inference.API.game_api import load_game_model, extract_pitches_and_align, extract_pitches_only
from inference.API.rmvpe_api import RmvpeTranscriber
from inference.API.ustx_api import save_ustx
from inference.API.vsqx_api import save_vsqx
from inference.device_utils import (
    RUNTIME_DEVICE_CHOICES,
    default_runtime_device,
    normalize_runtime_device,
)
from inference.pipeline.lyric_alignment import (
    PINYIN_ASR_DEFAULT_DIR,
    PINYIN_ASR_LANGUAGE,
    ROMAJI_ASR_DEFAULT_DIR,
    _run_lyric_alignment,
    _select_pinyin_asr_path,
    _select_romaji_asr_path,
    run_pinyin_asr,
    run_qwen_asr_and_fa,
    run_romaji_asr,
)
from inference.pipeline.memory_utils import free_memory

import logging

logger = logging.getLogger(__name__)

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


def normalize_pipeline_language(language: str | None) -> tuple[str, bool]:
    """Map user-facing language to the pipeline value and the pinyin-ASR flag.

    '中文-拼音' ("Chinese-Pinyin") / 'zh-pinyin' selects the direct pinyin ASR engine; every other
    language keeps its existing engine (ja romaji ASR or Qwen text ASR).
    """
    value = str(language or "").strip().lower()
    if value in {PINYIN_ASR_LANGUAGE, "中文-拼音", "中文拼音"}:
        return PINYIN_ASR_LANGUAGE, True
    return value or "zh", False


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


@dataclass(frozen=True)
class _PipelineContext:
    """Normalized run-wide values shared by the pipeline stages."""

    device: str
    output_key: str
    output_dir: pathlib.Path
    output_formats: list
    output_format_set: frozenset
    language: str
    fa_language: str
    lyric_output_mode: str
    use_pinyin_asr: bool


def _prepare_context(
    audio_path: str,
    output_filename: str,
    output_dir,
    output_formats,
    language: str,
    chinese_asr_engine: str,
    lyric_output_mode: str,
    tempo: float,
    batch_size: int,
    asr_batch_size: int,
    slice_min_sec: float,
    slice_max_sec: float,
    device: str,
) -> _PipelineContext:
    device = normalize_runtime_device(device)
    _validate_slice_runtime_options(tempo, batch_size, asr_batch_size, slice_min_sec, slice_max_sec)
    output_key = _resolve_output_key(output_filename, audio_path)
    output_formats = _normalize_output_formats(output_formats)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    language, use_pinyin_asr = normalize_pipeline_language(language)
    # Explicit Chinese ASR selection routes plain zh through the direct pinyin
    # ASR exactly like the legacy 'zh-pinyin' language value.
    if language == "zh" and str(chinese_asr_engine or "").strip().lower() == "pinyin":
        language = PINYIN_ASR_LANGUAGE
        use_pinyin_asr = True
    lyric_output_mode = _normalize_lyric_output_mode(language, lyric_output_mode)
    # HFA/GAME/VSQX only know the base languages; the pinyin ASR is a zh variant.
    fa_language = "zh" if language == PINYIN_ASR_LANGUAGE else language
    return _PipelineContext(
        device=device,
        output_key=output_key,
        output_dir=output_dir,
        output_formats=output_formats,
        output_format_set=frozenset(output_formats),
        language=language,
        fa_language=fa_language,
        lyric_output_mode=lyric_output_mode,
        use_pinyin_asr=use_pinyin_asr,
    )


def _extract_pitch_curve(
    waveform,
    sr,
    *,
    rmvpe_model_path,
    output_format_set,
    output_pitch_curve,
    device,
    cancel_checker,
):
    """RMVPE pitch curve for USTX/VSQX export; None when not requested.

    The model reference dies with this frame, releasing its memory before
    the next stage loads its own model.
    """
    if not (("ustx" in output_format_set or "vsqx" in output_format_set) and output_pitch_curve):
        return None
    rmvpe_model = _resolve_rmvpe_path(rmvpe_model_path)
    logger.info(f"[Hybrid Pipeline] Running RMVPE from: {rmvpe_model}")
    rmvpe = RmvpeTranscriber(rmvpe_model, device=device)
    try:
        rmvpe_result = rmvpe.infer(waveform, sr, cancel_checker=cancel_checker)
        logger.info(f"[Hybrid Pipeline] RMVPE done. Frames={len(rmvpe_result.midi_pitch)} step={rmvpe_result.time_step_seconds:.4f}s")
        return rmvpe_result
    finally:
        del rmvpe
        free_memory()


def _slice_chunks(
    waveform,
    sr,
    *,
    slicing_method,
    slice_min_sec,
    slice_max_sec,
    rmvpe_result,
    cancel_checker,
):
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
    if cancel_checker and cancel_checker():
        raise InterruptedError("任务已取消")
    if not chunks:
        raise RuntimeError("切片阶段未生成任何音频片段，已中断后续处理。")
    return chunks


def _run_game_stage(
    chunks,
    sr,
    ctx: _PipelineContext,
    outcome,
    *,
    ts,
    game_model_dir,
    batch_size,
    seg_threshold,
    seg_radius,
    est_threshold,
    cancel_checker,
):
    """Stage 3: GAME note extraction with per-chunk pitch-only fallback."""
    all_notes = []
    if outcome is not None and outcome.aligned:
        logger.info("\n--- Stage 3/3: Loading GAME model ---")
    else:
        if "chunks" in ctx.output_format_set:
            _export_chunk_wavs(chunks, sr, ctx.output_key, ctx.output_dir, cancel_checker=cancel_checker)
        if outcome is not None:
            logger.warning("\n--- Fallback: Loading GAME model (pitch-only mode) ---")
        else:
            logger.info("\n--- Stage 1/1: Loading GAME model (No-Lyrics Mode) ---")

    game_model = load_game_model(game_model_dir, device=ctx.device)
    try:
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")
        logger.info("--------------------------------------\n")

        if outcome is not None and outcome.aligned:
            aligned_result = extract_pitches_and_align(
                chunks, sr, outcome.pred_dict, outcome.chars_dict, game_model, ts,
                seg_threshold, seg_radius, est_threshold, batch_size,
                cancel_checker=cancel_checker,
                language=ctx.fa_language,
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
                logger.warning(
                    f"[Warning] Running pitch-only GAME fallback for "
                    f"{len(fallback_chunks)} chunk(s) without usable lyric alignment."
                )
                all_notes.extend(
                    extract_pitches_only(
                        fallback_chunks, sr, game_model, ts,
                        seg_threshold, seg_radius, est_threshold, batch_size,
                        cancel_checker=cancel_checker,
                        language=ctx.fa_language,
                    )
                )
        else:
            all_notes = extract_pitches_only(
                chunks, sr, game_model, ts,
                seg_threshold, seg_radius, est_threshold, batch_size,
                cancel_checker=cancel_checker,
                language=ctx.fa_language,
            )
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")
    finally:
        del game_model
        free_memory()
    return all_notes


def _export_outputs(
    all_notes,
    ctx: _PipelineContext,
    *,
    rmvpe_result,
    chunk_logs,
    output_lyrics,
    aligned,
    tempo,
    quantization_step,
    quantization_mode,
    quant_simplicity,
    pitch_format,
    round_pitch,
):
    all_notes.sort(key=lambda x: x.onset)

    export_asr_match_log = output_lyrics and (("asr_match_log" in ctx.output_format_set) or ("chunks" in ctx.output_format_set))
    if export_asr_match_log:
        log_path = ctx.output_dir / f"{ctx.output_key}_asr_match_log.txt"
        log_path.write_text("\n".join(chunk_logs), encoding="utf-8")

    if should_apply_quantization(quantization_mode, quantization_step):
        quantize_notes(all_notes, tempo, quantization_step, mode=quantization_mode, simplicity=quant_simplicity)

    lyric_status = "with lyrics" if aligned else "without lyrics"
    logger.info(f"Extracted {len(all_notes)} notes {lyric_status}.")

    if "mid" in ctx.output_format_set:
        _save_midi(all_notes, ctx.output_dir / f"{ctx.output_key}.mid", int(tempo))
    if "txt" in ctx.output_format_set:
        _save_text(all_notes, ctx.output_dir / f"{ctx.output_key}.txt", "txt", pitch_format, round_pitch)
    if "csv" in ctx.output_format_set:
        _save_text(all_notes, ctx.output_dir / f"{ctx.output_key}.csv", "csv", pitch_format, round_pitch)
    if "ustx" in ctx.output_format_set:
        save_ustx(all_notes, ctx.output_dir / f"{ctx.output_key}.ustx", tempo=float(tempo), rmvpe_result=rmvpe_result)
    if "vsqx" in ctx.output_format_set:
        save_vsqx(all_notes, ctx.output_dir / f"{ctx.output_key}.vsqx", tempo=float(tempo), language=ctx.fa_language, rmvpe_result=rmvpe_result)


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
    quant_simplicity: float = 0.0,
    slice_min_sec: float = DEFAULT_SLICE_MIN_SEC,
    slice_max_sec: float = DEFAULT_SLICE_MAX_SEC,
    output_lyrics: bool = True,
    output_pitch_curve: bool = False,
    rmvpe_model_path: str = "",
    phoneme_asr_model_path: str = "",
    pinyin_asr_model_path: str = "",
    chinese_asr_engine: str = "qwen",
    japanese_asr_engine: str = "romaji",
    asr_session=None,
    cancel_checker=None,
):
    """Auto Lyric Hybrid ONNX pipeline."""
    ctx = _prepare_context(
        audio_path, output_filename, output_dir, output_formats,
        language, chinese_asr_engine, lyric_output_mode,
        tempo, batch_size, asr_batch_size, slice_min_sec, slice_max_sec, device,
    )
    logger.info(f"\n[Hybrid Pipeline] Processing audio: {audio_path}")

    def _check_cancel():
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")

    _check_cancel()
    sr = 44100
    waveform, sr = load_audio(audio_path, sr)
    _check_cancel()

    rmvpe_result = _extract_pitch_curve(
        waveform, sr,
        rmvpe_model_path=rmvpe_model_path,
        output_format_set=ctx.output_format_set,
        output_pitch_curve=output_pitch_curve,
        device=ctx.device,
        cancel_checker=cancel_checker,
    )

    chunks = _slice_chunks(
        waveform, sr,
        slicing_method=slicing_method,
        slice_min_sec=slice_min_sec,
        slice_max_sec=slice_max_sec,
        rmvpe_result=rmvpe_result,
        cancel_checker=cancel_checker,
    )

    outcome = None
    chunk_logs = []
    if output_lyrics:
        matcher = create_lyric_matcher(ctx.language, original_lyrics)
        _check_cancel()
        free_memory()
        _check_cancel()
        outcome = _run_lyric_alignment(
            chunks, sr, ctx, matcher,
            asr_model_path=asr_model_path,
            hfa_model_dir=hfa_model_dir,
            phoneme_asr_model_path=phoneme_asr_model_path,
            pinyin_asr_model_path=pinyin_asr_model_path,
            japanese_asr_engine=japanese_asr_engine,
            asr_batch_size=asr_batch_size,
            asr_session=asr_session,
            cancel_checker=cancel_checker,
        )
        chunk_logs = outcome.chunk_logs
    else:
        logger.warning("\n--- No-Lyrics Mode: 跳过 ASR/HFA，仅执行 GAME 提取音高 ---\n")

    all_notes = _run_game_stage(
        chunks, sr, ctx, outcome,
        ts=ts,
        game_model_dir=game_model_dir,
        batch_size=batch_size,
        seg_threshold=seg_threshold,
        seg_radius=seg_radius,
        est_threshold=est_threshold,
        cancel_checker=cancel_checker,
    )

    _export_outputs(
        all_notes, ctx,
        rmvpe_result=rmvpe_result,
        chunk_logs=chunk_logs,
        output_lyrics=output_lyrics,
        aligned=outcome is not None and outcome.aligned,
        tempo=tempo,
        quantization_step=quantization_step,
        quantization_mode=quantization_mode,
        quant_simplicity=quant_simplicity,
        pitch_format=pitch_format,
        round_pitch=round_pitch,
    )


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("audio_path", type=click.Path(exists=True))
    @click.option("--game-model", "-gm", required=True, type=click.Path(exists=True, file_okay=False), help="Path to GAME ONNX model directory")
    @click.option("--hfa-model", "-hm", required=True, type=click.Path(exists=True, file_okay=False), help="Path to HubertFA ONNX model directory")
    @click.option("--asr-model", "-am", type=str, default="models/Qwen3-ASR-1.7B-dml", help="Path for the local Qwen3-ASR model directory")
    @click.option("--output-dir", "-o", type=click.Path(), default=".", help="Directory to save the outputs")
    @click.option("--lyrics", "-l", type=str, default="", help="Original reference lyrics for alignment")
    @click.option(
        "--device",
        type=click.Choice(list(RUNTIME_DEVICE_CHOICES)),
        default=default_runtime_device(),
        help="Runtime device (legacy 'cuda' maps to 'dml')",
    )
    @click.option("--t0", type=float, default=0.0, help="D3PM starting t0")
    @click.option("--nsteps", type=int, default=8, help="D3PM sampling steps")
    def main(audio_path, game_model, hfa_model, asr_model, output_dir, lyrics, device, t0, nsteps, **kwargs):
        """
        Auto Lyric Hybrid ONNX pipeline
        """
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
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
        )
        logger.info("Done!")

    main()
