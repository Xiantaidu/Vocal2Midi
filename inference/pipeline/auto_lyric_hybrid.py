import pathlib
import sys
from dataclasses import dataclass

import numpy as np

from application.config import (
    DEFAULT_SLICE_MAX_SEC,
    DEFAULT_SLICE_MIN_SEC,
    PipelineConfig,
    validate_slice_bounds,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.API.slicer_api import slice_audio_with_custom_bounds as slice_audio
from inference.io.audio_io import load_audio
from inference.io.note_io import _save_midi, _save_text
from inference.quant.quantization import quantize_notes, should_apply_quantization

from inference.API.lfa_api import create_lyric_matcher, _normalize_lyric_output_mode
from inference.API.game_api import load_game_model, extract_chunk_notes, release_game_models
from inference.API.rmvpe_api import RmvpeTranscriber
from inference.API.ustx_api import save_ustx
from inference.API.vsqx_api import save_vsqx
from inference.device_utils import normalize_runtime_device
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
from inference.pipeline.lyric_mapping import assign_chunk_lyrics
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
        raise ValueError("Output filename cannot be empty")
    return output_key


def normalize_pipeline_language(language: str | None) -> tuple[str, bool]:
    """Map language to the pipeline value and the pinyin-ASR flag."""
    value = str(language or "").strip().lower()
    if value == PINYIN_ASR_LANGUAGE:
        return PINYIN_ASR_LANGUAGE, True
    return value or "zh", False


def _validate_runtime_options(tempo: float, batch_size: int, asr_batch_size: int) -> None:
    if tempo <= 0:
        raise ValueError(f"tempo must be greater than 0, got {tempo}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than 0, got {batch_size}")
    if asr_batch_size <= 0:
        raise ValueError(f"asr_batch_size must be greater than 0, got {asr_batch_size}")


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
            raise InterruptedError("Slicing export task was cancelled")
        waveform = chunk.get("waveform")
        if waveform is None:
            logger.warning(f"[Warning] Chunk {chunk_idx} audio is not in memory; skipping its WAV export.")
            continue
        sf.write(output_dir / f"{output_key}_{chunk_idx:03d}.wav", waveform, sr)


def _resolve_rmvpe_path(model_path: str) -> str:
    if not model_path:
        raise ValueError("RMVPE model path cannot be empty")
    return model_path


@dataclass(frozen=True)
class _PipelineContext:
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
    # An explicit "pinyin" Chinese engine routes plain zh to the direct
    # pinyin ASR, exactly like the legacy 'zh-pinyin' language value.
    if language == "zh" and str(chinese_asr_engine or "").strip().lower() == "pinyin":
        language = PINYIN_ASR_LANGUAGE
        use_pinyin_asr = True
    lyric_output_mode = _normalize_lyric_output_mode(language, lyric_output_mode)
    # HFA/GAME/VSQX only know base languages; pinyin ASR is a zh variant.
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


_RMVPE_TRANSCRIBER_CACHE: dict[tuple[str, str], RmvpeTranscriber] = {}


def clear_rmvpe_cache() -> None:
    """Drop cached RMVPE transcribers so their sessions leave VRAM."""
    while _RMVPE_TRANSCRIBER_CACHE:
        _key, transcriber = _RMVPE_TRANSCRIBER_CACHE.popitem()
        transcriber.shutdown()
        del transcriber
    free_memory()


def release_aligner_models() -> None:
    """Drop cached TiFA/HubertFA bundles so their sessions leave VRAM.

    The single-song entrypoint keeps the aligners cached between songs; without
    this, the next song's GAME sessions load on top of them and both model sets
    stay resident. Best effort: freeing memory must not fail a song.
    """
    from inference.API.hfa_api import release_hfa_models
    from inference.TiFA.runtime import release_tifa_models

    for releaser in (release_tifa_models, release_hfa_models):
        try:
            releaser()
        except Exception:  # noqa: BLE001 - freeing is an optimisation
            logger.warning(
                "[Warning] Could not release the aligner sessions; continuing.", exc_info=True
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

    The transcriber is process-cached: recreating its DirectML session per
    song access-violates onnxruntime on the next song. Batch mode unloads it
    once through :func:`clear_rmvpe_cache` when the stage finishes.
    """
    if not (("ustx" in output_format_set or "vsqx" in output_format_set) and output_pitch_curve):
        return None
    rmvpe_model = _resolve_rmvpe_path(rmvpe_model_path)
    key = (rmvpe_model, str(device or "").strip().lower() or "auto")
    rmvpe = _RMVPE_TRANSCRIBER_CACHE.get(key)
    if rmvpe is None:
        logger.info(f"[Hybrid Pipeline] Running RMVPE from: {rmvpe_model}")
        rmvpe = RmvpeTranscriber(rmvpe_model, device=device)
        _RMVPE_TRANSCRIBER_CACHE[key] = rmvpe
    rmvpe_result = rmvpe.infer(waveform, sr, cancel_checker=cancel_checker)
    logger.info(f"[Hybrid Pipeline] RMVPE done. Frames={len(rmvpe_result.midi_pitch)} step={rmvpe_result.time_step_seconds:.4f}s")
    return rmvpe_result


def _slice_chunks(
    waveform,
    sr,
    *,
    slicing_method,
    slice_min_sec,
    slice_max_sec,
    rmvpe_result,
    cancel_checker,
    storage_dir=None,
):
    """Slice the waveform; with storage_dir the chunks are also written to disk.

    The slicer returns sample views of the source waveform, so keeping N songs
    sliced in memory would pin every source array. Spilling the chunk audio to
    storage_dir stores them compactly instead, and it lets the next stages run
    once GAME has been dropped to leave room for the alignment models.
    """
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
        raise InterruptedError("Task was cancelled")
    if not chunks:
        raise RuntimeError("No audio chunks generated in slicing stage, aborted.")
    if storage_dir is not None:
        _store_chunk_audio(chunks, sr, pathlib.Path(storage_dir), cancel_checker=cancel_checker)
    return chunks


def _store_chunk_audio(chunks, sr, storage_dir: pathlib.Path, cancel_checker=None) -> None:
    """Append every chunk waveform to storage_dir/chunk_000.wav.

    The chunks are contiguous back-to-back in one mono float32 file (the same
    layout read_chunk_audio expects), so the whole song costs one small file.
    """
    import soundfile as sf

    storage_dir.mkdir(parents=True, exist_ok=True)
    payload = np.concatenate(
        [np.ascontiguousarray(chunk["waveform"]).reshape(-1) for chunk in chunks]
    ) if chunks else np.zeros(0, dtype=np.float32)
    if cancel_checker and cancel_checker():
        raise InterruptedError("Task was cancelled")
    sf.write(storage_dir / "chunk_000.wav", payload, sr, subtype="FLOAT", format="WAV")


def chunk_sample_count(chunk) -> int:
    """Sample count of one chunk, from its layout or from its waveform."""
    stored = chunk.get("num_samples")
    if stored is not None:
        return int(stored)
    waveform = chunk.get("waveform")
    if waveform is None:
        raise ValueError("chunk carries neither audio nor a stored sample count")
    return int(len(waveform))


def strip_chunk_audio(chunks) -> None:
    """Keep the chunk layout (offset and sample count) and drop the audio.

    Batch mode keeps only the layout in memory and re-reads a song through
    :func:`read_chunk_audio` while a stage works on it.
    """
    for chunk in chunks or []:
        waveform = chunk.get("waveform")
        if waveform is None:
            continue
        chunk["num_samples"] = int(len(waveform))
        chunk["waveform"] = None


def read_chunk_audio(chunks, sr, storage_dir):
    """Rebuild the chunk list from storage_dir with read-only waveform views.

    Each chunk keeps the dict the slicer produced (offset, channel layout) and
    swaps in a strided view over one shared float32 buffer, so N songs on disk
    cost one array each in memory instead of a whole decoded waveform. A layout
    from :func:`strip_chunk_audio` carries the sample counts, so the audio does
    not have to stay resident between stages.
    """
    import soundfile as sf

    storage_dir = pathlib.Path(storage_dir)
    if not chunks:
        return []
    buffer = sf.read(str(storage_dir / "chunk_000.wav"), dtype="float32")[0]
    rebuilt = []
    position = 0
    for chunk in chunks:
        count = chunk_sample_count(chunk)
        view = buffer[position:position + count]
        restored = dict(chunk)
        restored["waveform"] = view
        position += count
        rebuilt.append(restored)
    return rebuilt


def release_chunk_audio(chunks) -> None:
    """Drop chunk waveforms and the RMVPE curves attached to them."""
    for chunk in chunks or []:
        chunk["waveform"] = None


def _run_game_stage(
    chunks,
    sr,
    ctx: _PipelineContext,
    *,
    ts,
    game_model_dir,
    batch_size,
    seg_threshold,
    seg_radius,
    est_threshold,
    cancel_checker,
):
    """GAME note extraction, independent of the lyrics; one note list per chunk.

    No alignment durations are fed in: the melody is transcribed from the
    audio alone and the lyrics are mapped onto it afterwards.
    """
    logger.info("\n--- Note extraction: Loading GAME model ---")
    game_model = load_game_model(game_model_dir, device=ctx.device)
    try:
        if cancel_checker and cancel_checker():
            raise InterruptedError("Task was cancelled")
        logger.info("--------------------------------------\n")
        chunk_notes = extract_chunk_notes(
            chunks, sr, game_model, ts,
            seg_threshold, seg_radius, est_threshold, batch_size,
            cancel_checker=cancel_checker,
            language=ctx.fa_language,
        )
        if cancel_checker and cancel_checker():
            raise InterruptedError("Task was cancelled")
    finally:
        # The model stays cached in load_game_model so the next song reuses
        # the same DirectML sessions. When lyrics follow, the caller drops the
        # suite before the aligner loads (see auto_lyric_hybrid_pipeline).
        free_memory()
    return chunk_notes


def _merge_lyrics(chunk_notes, chunks, ctx: _PipelineContext, outcome):
    """Map the aligned lyrics onto the GAME notes; unaligned chunks stay pitch-only."""
    if outcome is not None and outcome.aligned:
        logger.info("\n--- Lyric mapping: placing aligned lyrics on the GAME notes ---")
    elif outcome is not None:
        logger.warning("[Warning] No usable lyric alignment; exporting pitch-only notes.")
    return assign_chunk_lyrics(
        chunk_notes,
        [chunk["offset"] for chunk in chunks],
        outcome,
        ctx.fa_language,
    )


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
    chunks=None,
    sr=44100,
    cancel_checker=None,
):
    # The aligners export the chunk WAVs next to their TextGrids; without an
    # alignment the sliced audio is written here instead.
    if "chunks" in ctx.output_format_set and not aligned and chunks:
        _export_chunk_wavs(chunks, sr, ctx.output_key, ctx.output_dir, cancel_checker=cancel_checker)

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
        # float tempo: the notes were quantized (and USTX/VSQX are written) in
        # fractional BPM, so an int here would push every note off the grid.
        _save_midi(all_notes, ctx.output_dir / f"{ctx.output_key}.mid", float(tempo))
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
    alignment_engine: str = "tifa",
    tifa_model_path: str = "",
    japanese_g2p_engine: str = "kashi-g2p-onnx",
    kashi_g2p_model_path: str = "",
    asr_session=None,
    cancel_checker=None,
    chunk_storage_dir=None,
    defer_game: bool = False,
):
    """Run the whole hybrid pipeline, or only its first stage with defer_game.

    The flow is two independent branches joined at export: GAME transcribes
    the notes from the audio alone, ASR + TiFA/HubertFA produce the lyric
    timing, and the lyrics are then mapped onto the notes by time.

    With defer_game the slicing stage is the last step: the chunk list (audio
    still in memory) is returned so a batch caller can schedule the remaining
    stages itself. With chunk_storage_dir the chunk audio is also written
    there, so the list can be rebuilt later via read_chunk_audio.
    """
    ctx = _prepare_context(
        audio_path, output_filename, output_dir, output_formats,
        language, chinese_asr_engine, lyric_output_mode,
        tempo, batch_size, asr_batch_size, slice_min_sec, slice_max_sec, device,
    )
    logger.info(f"\n[Hybrid Pipeline] Processing audio: {audio_path}")

    def _check_cancel():
        if cancel_checker and cancel_checker():
            raise InterruptedError("Task was cancelled")

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
        storage_dir=chunk_storage_dir,
    )

    if defer_game:
        return chunks, rmvpe_result

    # Only this entrypoint owns the whole song in one call, so it also owns the
    # model handover: drop the previous song's aligner before GAME loads, and
    # GAME before the aligner loads again. Batch mode schedules the same
    # handover per stage and must not unload the model mid-group.
    release_aligner_models()

    # Branch 1: the melody, from the audio alone.
    chunk_notes = _run_game_stage(
        chunks, sr, ctx,
        ts=ts,
        game_model_dir=game_model_dir,
        batch_size=batch_size,
        seg_threshold=seg_threshold,
        seg_radius=seg_radius,
        est_threshold=est_threshold,
        cancel_checker=cancel_checker,
    )

    # Branch 2: the lyrics and their timing (ASR + TiFA/HubertFA).
    outcome = None
    chunk_logs = []
    if output_lyrics:
        # The melody is already transcribed, so GAME's sessions can leave VRAM
        # before the ASR worker and the aligner load theirs; otherwise both
        # model sets stay resident at once. Best effort: a release that fails
        # must not cost an otherwise good song.
        try:
            release_game_models()
        except Exception:  # noqa: BLE001 - freeing is an optimisation
            logger.warning("[Warning] Could not release the GAME sessions; continuing.", exc_info=True)
        matcher = create_lyric_matcher(
            ctx.language,
            original_lyrics,
            engine=japanese_g2p_engine,
            model_dir=kashi_g2p_model_path,
        )
        _check_cancel()
        free_memory()
        _check_cancel()
        outcome = _run_lyric_alignment(
            chunks, sr, ctx, matcher,
            asr_model_path=asr_model_path,
            hfa_model_dir=hfa_model_dir,
            tifa_model_path=tifa_model_path,
            alignment_engine=alignment_engine,
            phoneme_asr_model_path=phoneme_asr_model_path,
            pinyin_asr_model_path=pinyin_asr_model_path,
            japanese_asr_engine=japanese_asr_engine,
            japanese_g2p_engine=japanese_g2p_engine,
            kashi_g2p_model_path=kashi_g2p_model_path,
            asr_batch_size=asr_batch_size,
            asr_session=asr_session,
            cancel_checker=cancel_checker,
        )
        chunk_logs = outcome.chunk_logs
    else:
        logger.warning("\n--- No-Lyrics Mode: skipping ASR/alignment, exporting GAME notes only ---\n")

    all_notes = _merge_lyrics(chunk_notes, chunks, ctx, outcome)

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
        chunks=chunks,
        sr=sr,
        cancel_checker=cancel_checker,
    )


def pipeline_stage_slicing(audio_path: str | None = None, *, chunk_storage_dir=None, cancel_checker=None, **kwargs):
    """Batch stage 1: decode, slice, persist the chunks.

    Returns ``(chunks, rmvpe_result)``. With chunk_storage_dir the returned
    chunks are a layout only (offset + sample count): the audio stays on disk,
    so a queue of songs does not pin every decoded waveform in memory, and each
    later stage reads back the one song it is working on.
    """
    if audio_path is not None:
        kwargs["audio_path"] = audio_path
    chunks, rmvpe_result = auto_lyric_hybrid_pipeline(
        chunk_storage_dir=chunk_storage_dir,
        defer_game=True,
        cancel_checker=cancel_checker,
        **kwargs,
    )
    if chunk_storage_dir is not None:
        strip_chunk_audio(chunks)
    return chunks, rmvpe_result


def pipeline_stage_alignment(
    cfg: PipelineConfig,
    chunks,
    *,
    rmvpe_result=None,
    matcher=None,
    asr_session=None,
    cancel_checker=None,
):
    """Batch stage 2: ASR + forced alignment for one already-sliced song."""
    ctx = _prepare_context(
        cfg.audio_path, cfg.output_filename, cfg.output_dir, cfg.output_formats,
        cfg.language, cfg.chinese_asr_engine, cfg.lyric_output_mode,
        cfg.tempo, cfg.batch_size, cfg.asr_batch_size,
        cfg.slice_min_sec, cfg.slice_max_sec, cfg.device,
    )
    try:
        outcome = _run_lyric_alignment(
            chunks, 44100, ctx, matcher,
            asr_model_path=cfg.asr_model_path,
            hfa_model_dir=cfg.hfa_model_dir,
            tifa_model_path=cfg.tifa_model_path,
            alignment_engine=cfg.alignment_engine,
            phoneme_asr_model_path=cfg.phoneme_asr_model_path,
            pinyin_asr_model_path=cfg.pinyin_asr_model_path,
            japanese_asr_engine=cfg.japanese_asr_engine,
            japanese_g2p_engine=cfg.japanese_g2p_engine,
            kashi_g2p_model_path=cfg.kashi_g2p_model_path,
            asr_batch_size=cfg.asr_batch_size,
            asr_session=asr_session,
            cancel_checker=cancel_checker,
        )
    finally:
        release_chunk_audio(chunks)
    return outcome


def prepare_lyric_matcher(cfg: PipelineConfig, *, cache=None):
    """Matcher for one job; ``cache`` reuses the language processors' G2P."""
    language, use_pinyin_asr = normalize_pipeline_language(cfg.language)
    if language == "zh" and str(cfg.chinese_asr_engine or "").strip().lower() == "pinyin":
        language = PINYIN_ASR_LANGUAGE
    return create_lyric_matcher(
        language,
        cfg.original_lyrics,
        engine=cfg.japanese_g2p_engine,
        model_dir=cfg.kashi_g2p_model_path,
        g2p_cache=cache,
    )


def _context_for(cfg: PipelineConfig) -> _PipelineContext:
    return _prepare_context(
        cfg.audio_path, cfg.output_filename, cfg.output_dir, cfg.output_formats,
        cfg.language, cfg.chinese_asr_engine, cfg.lyric_output_mode,
        cfg.tempo, cfg.batch_size, cfg.asr_batch_size,
        cfg.slice_min_sec, cfg.slice_max_sec, cfg.device,
    )


def pipeline_stage_game(
    cfg: PipelineConfig,
    chunks,
    *,
    cancel_checker=None,
):
    """Batch stage: GAME note extraction; returns one note list per chunk."""
    ctx = _context_for(cfg)
    try:
        return _run_game_stage(
            chunks, 44100, ctx,
            ts=cfg.ts,
            game_model_dir=cfg.game_model_dir,
            batch_size=cfg.batch_size,
            seg_threshold=cfg.seg_threshold,
            seg_radius=cfg.seg_radius,
            est_threshold=cfg.est_threshold,
            cancel_checker=cancel_checker,
        )
    finally:
        release_chunk_audio(chunks)


def pipeline_stage_export(
    cfg: PipelineConfig,
    chunks,
    chunk_notes,
    outcome,
    *,
    rmvpe_result=None,
    storage_dir=None,
    cancel_checker=None,
):
    """Batch stage: map the lyrics onto the GAME notes and write every export.

    ``chunks`` is normally the slicer layout only; its audio lives in
    ``storage_dir`` and is read back solely for the chunk-WAV export of songs
    that ended up without an alignment.
    """
    ctx = _context_for(cfg)
    aligned = outcome is not None and outcome.aligned
    all_notes = _merge_lyrics(chunk_notes, chunks, ctx, outcome)
    export_chunks = chunks
    if (
        "chunks" in ctx.output_format_set
        and not aligned
        and storage_dir is not None
        and any(chunk.get("waveform") is None for chunk in chunks)
    ):
        export_chunks = read_chunk_audio(chunks, 44100, storage_dir)
    _export_outputs(
        all_notes, ctx,
        rmvpe_result=rmvpe_result,
        chunk_logs=list(getattr(outcome, "chunk_logs", []) or []),
        output_lyrics=cfg.output_lyrics,
        aligned=aligned,
        tempo=cfg.tempo,
        quantization_step=cfg.quantization_step,
        quantization_mode=cfg.quantization_mode,
        quant_simplicity=cfg.quant_simplicity,
        pitch_format=cfg.pitch_format,
        round_pitch=cfg.round_pitch,
        chunks=export_chunks,
        sr=44100,
        cancel_checker=cancel_checker,
    )
    return all_notes
