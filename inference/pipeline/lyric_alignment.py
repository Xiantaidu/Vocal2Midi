"""Lyric-alignment stage: phoneme/text ASR followed by HubertFA alignment.

Owns the temporary chunk wav/.lab file protocol shared with the HFA stage;
nothing outside this module reads those files.
"""
from __future__ import annotations

import logging
import pathlib
import tempfile
from dataclasses import dataclass

from inference.API.asr_api import (
    batch_transcribe_asr,
    batch_transcribe_pinyin_asr,
    batch_transcribe_romaji_asr,
)
from inference.API.hfa_api import load_hfa_model, run_hubert_fa, export_hfa_artifacts
from inference.TiFA.aligner import run_tifa_fa
from inference.TiFA.runtime import load_tifa_model
from inference.TiFA.textgrid import save_textgrids as export_textgrids
from inference.API.lfa_api import process_asr_to_phonemes
from inference.pipeline.memory_utils import free_memory

logger = logging.getLogger(__name__)


ROMAJI_ASR_DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "romajiASR"
PINYIN_ASR_DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "pinyinASR"

# Language value that routes Chinese lyric extraction through the direct
# pinyin ASR instead of Qwen text ASR.
PINYIN_ASR_LANGUAGE = "zh-pinyin"


def _select_romaji_asr_path(phoneme_asr_model_path: str) -> str | None:
    if phoneme_asr_model_path:
        return phoneme_asr_model_path
    if ROMAJI_ASR_DEFAULT_DIR.exists():
        return str(ROMAJI_ASR_DEFAULT_DIR)
    return None


def _select_pinyin_asr_path(pinyin_asr_model_path: str) -> str | None:
    if pinyin_asr_model_path:
        return pinyin_asr_model_path
    if PINYIN_ASR_DEFAULT_DIR.exists():
        return str(PINYIN_ASR_DEFAULT_DIR)
    return None


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
    asr_session=None,
):
    """
    Runs ASR using the Qwen runtime with batching and prepares .lab files for HubertFA.
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
        session=asr_session,
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


def run_pinyin_asr(
    chunks,
    sr,
    temp_dir_path,
    matcher,
    asr_model_path,
    device,
    language=PINYIN_ASR_LANGUAGE,
    lyric_output_mode=None,
    asr_batch_size=1,
    cancel_checker=None,
):
    all_results, chunk_indices = batch_transcribe_pinyin_asr(
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


@dataclass(frozen=True)
class _AlignmentOutcome:
    """Result of the lyric-alignment stage.

    ``aligned`` is the final run_lyric_alignment flag after every fallback
    decision along the way.
    """

    chars_dict: dict
    pred_dict: dict
    chunk_logs: list
    aligned: bool


def _select_phoneme_engine(
    language,
    lyric_output_mode,
    use_pinyin_asr,
    japanese_asr_engine,
    phoneme_asr_model_path,
    pinyin_asr_model_path,
):
    """Pick the direct-phoneme ASR engine; a None engine falls back to text ASR."""
    ja_wants_romaji = str(japanese_asr_engine or "").strip().lower() != "qwen"
    engine = None
    if language == "ja" and lyric_output_mode in {"romaji", "kana"} and ja_wants_romaji:
        engine = "romaji"
    elif use_pinyin_asr:
        engine = "pinyin"

    phoneme_asr_path = None
    if engine == "romaji":
        logger.info("\n--- Stage 1/3: Running mora ASR for Japanese lyric mode ---")
        phoneme_asr_path = _select_romaji_asr_path(phoneme_asr_model_path)
        if phoneme_asr_path is None:
            logger.warning(
                "[Warning] Romaji ASR model not found; "
                "falling back to text ASR + Japanese G2P."
            )
            engine = None
    elif engine == "pinyin":
        logger.info("\n--- Stage 1/3: Running pinyin ASR for Chinese-pinyin lyric mode ---")
        phoneme_asr_path = _select_pinyin_asr_path(pinyin_asr_model_path)
        if phoneme_asr_path is None:
            logger.warning(
                "[Warning] Pinyin ASR model not found; "
                "falling back to text ASR (Qwen) + Chinese G2P."
            )
            engine = None
    return engine, ja_wants_romaji, phoneme_asr_path


def _export_chunk_wavs_from_temp(
    temp_dir_path,
    chunk_count: int,
    output_key: str,
    output_dir,
    cancel_checker=None,
) -> None:
    """Copy the stage's chunk_N.wav files out under the {key}_{idx:03d} naming.

    Mirrors the WAV half of hfa_api.export_hfa_artifacts so the chunks switch
    yields the same per-chunk WAV + TextGrid pairs on the TiFA path.
    """
    import shutil

    for chunk_idx in range(chunk_count):
        if cancel_checker and cancel_checker():
            raise InterruptedError("切片导出任务已取消")
        src = temp_dir_path / f"chunk_{chunk_idx}.wav"
        if not src.is_file():
            logger.warning(f"[Warning] Chunk WAV file not found, skipping: {src}")
            continue
        shutil.copy2(src, output_dir / f"{output_key}_{chunk_idx:03d}.wav")


def _run_lyric_alignment(
    chunks,
    sr,
    ctx: _PipelineContext,
    matcher,
    *,
    asr_model_path,
    hfa_model_dir,
    tifa_model_path,
    alignment_engine,
    phoneme_asr_model_path,
    pinyin_asr_model_path,
    japanese_asr_engine,
    asr_batch_size,
    asr_session,
    cancel_checker,
) -> _AlignmentOutcome:
    """Stages 1+2: phoneme/text ASR followed by forced alignment (HFA or TiFA).

    Owns the temporary directory holding the chunk wavs and .lab files;
    nothing outside this stage reads them.
    """
    def _check_cancel():
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")

    chars_dict = {}
    pred_dict = {}
    chunk_logs = []
    aligned = True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        engine, ja_wants_romaji, phoneme_asr_path = _select_phoneme_engine(
            ctx.language, ctx.lyric_output_mode, ctx.use_pinyin_asr,
            japanese_asr_engine, phoneme_asr_model_path, pinyin_asr_model_path,
        )
        if engine == "romaji":
            chars_dict, chunk_logs = run_romaji_asr(
                chunks,
                sr,
                temp_dir_path,
                matcher,
                asr_model_path=phoneme_asr_path,
                device=ctx.device,
                language=ctx.language,
                lyric_output_mode=ctx.lyric_output_mode,
                asr_batch_size=asr_batch_size,
                cancel_checker=cancel_checker,
            )
        elif engine == "pinyin":
            chars_dict, chunk_logs = run_pinyin_asr(
                chunks,
                sr,
                temp_dir_path,
                matcher,
                asr_model_path=phoneme_asr_path,
                device=ctx.device,
                language=ctx.language,
                lyric_output_mode=ctx.lyric_output_mode,
                asr_batch_size=asr_batch_size,
                cancel_checker=cancel_checker,
            )
        else:
            if ctx.language == "ja" and ctx.lyric_output_mode in {"romaji", "kana"}:
                if ja_wants_romaji:
                    logger.warning("\n--- Stage 1/3: Mora ASR unavailable; fallback to text ASR + Japanese G2P ---")
                else:
                    logger.info("\n--- Stage 1/3: Running text ASR (Qwen) + Japanese G2P ---")
            elif ctx.use_pinyin_asr:
                logger.warning("\n--- Stage 1/3: Pinyin ASR unavailable; fallback to text ASR + Chinese G2P ---")
            else:
                logger.info("\n--- Stage 1/3: Running ASR in subprocess isolation mode ---")
            chars_dict, chunk_logs = run_qwen_asr_and_fa(
                chunks,
                sr,
                temp_dir_path,
                matcher,
                asr_model_path=asr_model_path,
                device=ctx.device,
                asr_batch_size=asr_batch_size,
                language=ctx.language,
                lyric_output_mode=ctx.lyric_output_mode,
                cancel_checker=cancel_checker,
                asr_session=asr_session,
            )
        _check_cancel()

        if not chars_dict:
            logger.warning(
                "[Warning] ASR did not produce valid text for any chunk; "
                "falling back to GAME pitch-only extraction."
            )
            aligned = False

        free_memory()

        if aligned:
            use_tifa = str(alignment_engine or "").strip().lower() == "tifa"
            if use_tifa:
                logger.info("\n--- Stage 2/3: Loading TiFA model ---")
                tifa_model = load_tifa_model(tifa_model_path, device=ctx.device)
                try:
                    _check_cancel()
                    logger.info("------------------------------------------\n")

                    pred_dict, tifa_display = run_tifa_fa(
                        tifa_model,
                        temp_dir_path,
                        language=ctx.language,
                        cancel_checker=cancel_checker,
                    )
                finally:
                    del tifa_model
                    free_memory()
            else:
                logger.info("\n--- Stage 2/3: Loading HubertFA model ---")
                hfa_model = load_hfa_model(hfa_model_dir, device=ctx.device)
                try:
                    _check_cancel()
                    logger.info("------------------------------------------\n")

                    pred_dict = run_hubert_fa(
                        hfa_model,
                        temp_dir_path,
                        language=ctx.fa_language,
                        cancel_checker=cancel_checker,
                    )
                    _check_cancel()
                    if not pred_dict:
                        logger.warning(
                            "[Warning] HFA did not produce alignment for any chunk; "
                            "falling back to GAME pitch-only extraction."
                        )
                        aligned = False
                    else:
                        missing_hfa = sorted(set(chars_dict) - set(pred_dict))
                        if missing_hfa:
                            preview = ", ".join(missing_hfa[:8])
                            suffix = " ..." if len(missing_hfa) > 8 else ""
                            logger.warning(
                                f"[Warning] HFA missing {len(missing_hfa)} chunk(s) "
                                f"({preview}{suffix}); those chunks will use pitch-only fallback."
                            )

                        export_hfa_artifacts(
                            chunks,
                            temp_dir_path,
                            hfa_model,
                            ctx.output_key,
                            ctx.output_dir,
                            ctx.output_formats,
                            cancel_checker=cancel_checker,
                        )
                finally:
                    del hfa_model
                    free_memory()
            if use_tifa:
                _check_cancel()
                if not pred_dict:
                    logger.warning(
                        "[Warning] TiFA did not produce alignment for any chunk; "
                        "falling back to GAME pitch-only extraction."
                    )
                    aligned = False
                else:
                    # TiFA words are mora-level for ja: use its own romaji/kana
                    # display tokens so every aligned word gets a lyric (the
                    # lfa ja display cannot tokenize unspaced kanji text).
                    for stem, pairs in tifa_display.items():
                        if ctx.lyric_output_mode == "kana":
                            tokens = [kana for _romaji, kana in pairs if kana]
                        else:
                            tokens = [romaji for romaji, _kana in pairs]
                        if tokens:
                            chars_dict[stem] = tokens
                    # Same chunks-switch contract as export_hfa_artifacts:
                    # the chunks switch yields per-chunk WAV + TextGrid pairs.
                    export_chunks = "chunks" in ctx.output_format_set
                    export_textgrid = ("textgrid" in ctx.output_format_set) or export_chunks
                    if export_textgrid:
                        # Pass the stem-keyed mapping so each TextGrid is named
                        # by its real chunk index and pairs with the matching
                        # {output_key}_NNN.wav (pred_dict is in rglob order, not
                        # positionally aligned to the numeric chunk indices).
                        export_textgrids(
                            pred_dict,
                            ctx.output_dir,
                            ctx.output_key,
                            cancel_checker=cancel_checker,
                        )
                    if export_chunks:
                        _export_chunk_wavs_from_temp(
                            temp_dir_path,
                            len(chunks),
                            ctx.output_key,
                            ctx.output_dir,
                            cancel_checker=cancel_checker,
                        )

    return _AlignmentOutcome(chars_dict=chars_dict, pred_dict=pred_dict, chunk_logs=chunk_logs, aligned=aligned)
