"""Stage-wise batch execution of many auto-lyric jobs.

Single-job mode walks one song through every stage. A batch does the opposite:
it runs one stage over the whole queue before starting the next one, so a model
is loaded once per stage instead of once per song:

    slicing:  A B C
    GAME:     A B C
    ASR:      A B C
    align:    A B C
    export:   A B C

GAME transcribes the notes from the audio alone; ASR + alignment only produce
lyric timing, which the export stage maps onto those notes.

Within a stage the jobs are grouped by everything that pins a model (language,
alignment/pitch engine, device, model path), so a queue mixing languages or
devices pays one load per group. Each group is released as soon as it finishes,
so only one stage's models sit in VRAM at a time.

Intermediate state is kept on disk: S1 spills the chunk audio into a per-job
temporary directory and the later stages read it back one song at a time.
"""
from __future__ import annotations

import pathlib
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from application.config import PipelineConfig

STAGE_SLICING = "slicing"
STAGE_ASR = "asr"
STAGE_ALIGNMENT = "alignment"
STAGE_GAME = "game"
STAGE_EXPORT = "export"
STAGE_ORDER = (STAGE_SLICING, STAGE_GAME, STAGE_ASR, STAGE_ALIGNMENT, STAGE_EXPORT)

STAGE_LABELS = {
    STAGE_SLICING: "Slicing",
    STAGE_GAME: "Note extraction",
    STAGE_ASR: "Speech recognition",
    STAGE_ALIGNMENT: "Forced alignment",
    STAGE_EXPORT: "Lyric mapping and export",
}


class BatchCancelled(Exception):
    """Raised after a stage stopped on the cancel checker."""


@dataclass
class BatchItem:
    """One queued song plus its per-stage products."""

    index: int
    config: PipelineConfig
    filename: str
    # Slicer layout only (offset + sample count); the audio stays on disk.
    base_chunks: Optional[list] = None
    chunks: Optional[list] = None
    rmvpe_result: object = None
    chunk_notes: Optional[list] = None
    asr: object = None
    outcome: object = None
    storage_dir: Optional[pathlib.Path] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None

    @property
    def name(self) -> str:
        return pathlib.Path(self.filename).name


def _language_group(cfg: PipelineConfig) -> tuple:
    return (
        str(cfg.language or "").strip().lower(),
        str(cfg.lyric_output_mode or "").strip().lower(),
        str(cfg.chinese_asr_engine or "").strip().lower(),
        str(cfg.japanese_asr_engine or "").strip().lower(),
    )


def _asr_group(cfg: PipelineConfig) -> tuple:
    return _language_group(cfg) + (
        str(cfg.device or "").strip().lower(),
        cfg.asr_model_path or "",
        cfg.pinyin_asr_model_path or "",
        cfg.phoneme_asr_model_path or "",
    )


def _alignment_group(cfg: PipelineConfig) -> tuple:
    return _language_group(cfg) + (
        str(cfg.alignment_engine or "").strip().lower(),
        str(cfg.device or "").strip().lower(),
        cfg.tifa_model_path or "",
        cfg.hfa_model_dir or "",
        cfg.japanese_g2p_engine or "",
        cfg.kashi_g2p_model_path or "",
    )


def _game_group(cfg: PipelineConfig) -> tuple:
    return (cfg.game_model_dir or "", str(cfg.device or "").strip().lower())


def _ordered_groups(items: list, key: Callable) -> list[list]:
    """Group items by key, keeping first-appearance order stable."""
    groups: dict[tuple, list] = {}
    for item in items:
        groups.setdefault(key(item.config), []).append(item)
    return list(groups.values())


def _check_cancel(cancel_checker) -> None:
    if cancel_checker and cancel_checker():
        raise InterruptedError("Task was cancelled")


class BatchRunner:
    """Runs a list of pipeline configs stage by stage."""

    def __init__(
        self,
        tasks: list[tuple[PipelineConfig, str]],
        *,
        log=None,
        progress=None,
        cancel_checker=None,
        asr_session=None,
        unload_models: bool = True,
    ):
        self.items = [
            BatchItem(index=index, config=config, filename=filename)
            for index, (config, filename) in enumerate(tasks, start=1)
        ]
        self._log = log or (lambda message: None)
        self._progress = progress or (lambda stage, index, total, filename: None)
        self._cancel_checker = cancel_checker
        self._asr_session = asr_session
        self._unload_models = unload_models
        self._root_dir: Optional[pathlib.Path] = None
        self._matcher_cache: dict = {}

    @property
    def total(self) -> int:
        return len(self.items)

    def run(self) -> None:
        if not self.items:
            return
        self._root_dir = pathlib.Path(tempfile.mkdtemp(prefix="v2m_batch_"))
        try:
            self._run_slicing()
            self._run_game()
            self._run_asr()
            self._run_alignment()
            self._run_export()
            self._raise_if_cancelled()
        except InterruptedError as exc:
            # Cancellation raised inside a stage is reported as one signal.
            raise BatchCancelled(str(exc) or "Batch was cancelled") from None
        finally:
            self._cleanup()
            self._close_asr_session()

    def _close_asr_session(self) -> None:
        """Stop the ASR worker once; later calls are no-ops."""
        session, self._asr_session = self._asr_session, None
        if session is None:
            return
        try:
            session.close()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass

    # -- plumbing ---------------------------------------------------------
    def _pending(self) -> list:
        return [item for item in self.items if item.failed_stage is None]

    def _raise_if_cancelled(self) -> None:
        if self._cancel_checker and self._cancel_checker():
            raise BatchCancelled("Batch was cancelled")

    def _release(self, *releasers) -> None:
        """Unload a stage's models; a no-op keeps them warm for debugging."""
        if not self._unload_models:
            return
        for releaser in releasers:
            try:
                releaser()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
        _free_memory()

    def _run_stage(
        self, stage: str, items: list, work: Callable, *, done_before: int = 0, total: int = None
    ) -> int:
        """Run ``work(item)`` for each item, isolating per-song failures.

        ``done_before``/``total`` let a stage report progress across all of its
        groups (3/3) instead of restarting at 1/1 for every group. Returns how
        many items were attempted so the caller can keep the running offset.
        """
        active = [item for item in items if item.failed_stage is None]
        stage_total = total if total is not None else len(active)
        for position, item in enumerate(active, start=1):
            _check_cancel(self._cancel_checker)
            self._progress(stage, done_before + position, stage_total, item.filename)
            try:
                work(item)
            except (InterruptedError, BatchCancelled):
                raise
            except Exception as exc:  # noqa: BLE001 - reported per song
                item.failed_stage = stage
                item.error = str(exc)
                self._log(f"[{item.name}] failed at '{STAGE_LABELS[stage]}': {exc}")
                _free_memory()
        return len(active)

    def _job_dir(self, item: BatchItem) -> pathlib.Path:
        return self._root_dir / f"job_{item.index:03d}"

    def _text_dir(self, item: BatchItem) -> pathlib.Path:
        """Where the ASR stage writes chunk wavs and .lab files for a job."""
        return self._job_dir(item) / "text"

    def _needs_lyrics(self, item: BatchItem) -> bool:
        return bool(item.config.output_lyrics)

    # -- slicing ----------------------------------------------------------
    def _run_slicing(self) -> None:
        from inference.pipeline.auto_lyric_hybrid import pipeline_stage_slicing
        from inference.pipeline.memory_utils import free_memory

        def work(item: BatchItem) -> None:
            storage = self._job_dir(item)
            kwargs = item.config.to_kwargs()
            # audio_path is passed positionally below, and cancel_checker is
            # supplied by the runner: leaving either in kwargs raises
            # "got multiple values for argument ...".
            for key in ("asr_session", "cancel_checker", "audio_path"):
                kwargs.pop(key, None)
            self._log(f"[{item.name}] {STAGE_LABELS[STAGE_SLICING]}")
            chunks, rmvpe_result = pipeline_stage_slicing(
                item.config.audio_path,
                chunk_storage_dir=storage,
                cancel_checker=self._cancel_checker,
                **kwargs,
            )
            item.base_chunks = chunks
            item.rmvpe_result = rmvpe_result
            item.storage_dir = storage
            free_memory()

        from inference.pipeline.auto_lyric_hybrid import clear_rmvpe_cache

        self._run_stage(STAGE_SLICING, self._pending(), work)
        self._release(clear_rmvpe_cache)

    # -- ASR --------------------------------------------------------------
    def _run_asr(self) -> None:
        items = [item for item in self._pending() if self._needs_lyrics(item)]
        if not items:
            if self.items:
                self._log("[Batch] No job needs speech recognition; that stage is skipped.")
            return

        from inference.API.asr_api import (
            clear_pinyin_model_cache,
            clear_romaji_model_cache,
        )

        done = 0
        for group in _ordered_groups(items, _asr_group):
            # The ASR worker stays alive across groups: restarting it between
            # groups would reload the Qwen weights, the very cost this stage
            # layout exists to avoid.
            done += self._run_stage(
                STAGE_ASR, group, self._run_asr_for, done_before=done, total=len(items)
            )
            self._release(clear_romaji_model_cache, clear_pinyin_model_cache)

        # Stage finished: the Qwen worker is the largest resident model, so
        # drop it before alignment loads its own sessions.
        if self._unload_models:
            self._close_asr_session()

    def _run_asr_for(self, item: BatchItem) -> None:
        from inference.pipeline import auto_lyric_hybrid as pipeline
        from inference.pipeline import lyric_alignment

        cfg = item.config
        item.chunks = pipeline.read_chunk_audio(item.base_chunks, 44100, item.storage_dir)
        try:
            temp_dir = self._text_dir(item)
            temp_dir.mkdir(parents=True, exist_ok=True)
            ctx = pipeline._prepare_context(
                cfg.audio_path, cfg.output_filename, cfg.output_dir, cfg.output_formats,
                cfg.language, cfg.chinese_asr_engine, cfg.lyric_output_mode,
                cfg.tempo, cfg.batch_size, cfg.asr_batch_size,
                cfg.slice_min_sec, cfg.slice_max_sec, cfg.device,
            )
            matcher = pipeline.prepare_lyric_matcher(cfg, cache=self._matcher_cache)
            self._log(f"[{item.name}] {STAGE_LABELS[STAGE_ASR]}")

            engine, ja_wants_romaji, phoneme_asr_path = lyric_alignment._select_phoneme_engine(
                ctx.language, ctx.lyric_output_mode, ctx.use_pinyin_asr,
                cfg.japanese_asr_engine, cfg.phoneme_asr_model_path, cfg.pinyin_asr_model_path,
            )
            if engine == "romaji":
                chars_dict, chunk_logs = lyric_alignment.run_romaji_asr(
                    item.chunks, 44100, temp_dir, matcher,
                    asr_model_path=phoneme_asr_path or cfg.phoneme_asr_model_path,
                    device=ctx.device,
                    language=ctx.language,
                    lyric_output_mode=ctx.lyric_output_mode,
                    asr_batch_size=cfg.asr_batch_size,
                    cancel_checker=self._cancel_checker,
                )
            elif engine == "pinyin":
                chars_dict, chunk_logs = lyric_alignment.run_pinyin_asr(
                    item.chunks, 44100, temp_dir, matcher,
                    asr_model_path=phoneme_asr_path or cfg.pinyin_asr_model_path,
                    device=ctx.device,
                    language=ctx.language,
                    lyric_output_mode=ctx.lyric_output_mode,
                    asr_batch_size=cfg.asr_batch_size,
                    cancel_checker=self._cancel_checker,
                )
            else:
                if ctx.language == "ja" and ctx.lyric_output_mode in {"romaji", "kana"} and ja_wants_romaji:
                    self._log(
                        f"[{item.name}] mora ASR model missing; falling back to text ASR + Japanese G2P."
                    )
                chars_dict, chunk_logs = lyric_alignment.run_qwen_asr_and_fa(
                    item.chunks, 44100, temp_dir, matcher,
                    asr_model_path=cfg.asr_model_path,
                    device=ctx.device,
                    asr_batch_size=cfg.asr_batch_size,
                    language=ctx.language,
                    lyric_output_mode=ctx.lyric_output_mode,
                    cancel_checker=self._cancel_checker,
                    asr_session=self._asr_session,
                    japanese_g2p_engine=cfg.japanese_g2p_engine,
                    kashi_g2p_model_path=cfg.kashi_g2p_model_path,
                )
            if not chars_dict:
                self._log(f"[{item.name}] no usable ASR text; falling back to pitch-only notes.")
            item.asr = lyric_alignment._AlignmentOutcome(
                chars_dict=chars_dict, pred_dict={}, chunk_logs=chunk_logs, aligned=False
            )
        finally:
            # A failed song must not pin its decoded audio for the rest of the
            # batch (the same reason the other stages release here).
            pipeline.release_chunk_audio(item.chunks)
            item.chunks = None

    # -- alignment --------------------------------------------------------
    def _run_alignment(self) -> None:
        items = [item for item in self._pending() if self._needs_lyrics(item)]
        if not items:
            return

        from inference.API.hfa_api import release_hfa_models
        from inference.TiFA.runtime import release_tifa_models

        done = 0
        for group in _ordered_groups(items, _alignment_group):
            done += self._run_stage(
                STAGE_ALIGNMENT, group, self._run_alignment_for,
                done_before=done, total=len(items),
            )
            self._release(release_tifa_models, release_hfa_models)

    def _run_alignment_for(self, item: BatchItem) -> None:
        from inference.pipeline import auto_lyric_hybrid as pipeline
        from inference.pipeline import lyric_alignment

        cfg = item.config
        if item.asr is None or not item.asr.chars_dict:
            return
        item.chunks = pipeline.read_chunk_audio(item.base_chunks, 44100, item.storage_dir)
        ctx = pipeline._prepare_context(
            cfg.audio_path, cfg.output_filename, cfg.output_dir, cfg.output_formats,
            cfg.language, cfg.chinese_asr_engine, cfg.lyric_output_mode,
            cfg.tempo, cfg.batch_size, cfg.asr_batch_size,
            cfg.slice_min_sec, cfg.slice_max_sec, cfg.device,
        )
        matcher = pipeline.prepare_lyric_matcher(cfg, cache=self._matcher_cache)
        self._log(f"[{item.name}] {STAGE_LABELS[STAGE_ALIGNMENT]}")

        try:
            outcome = lyric_alignment._run_lyric_alignment(
                item.chunks, 44100, ctx, matcher,
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
                asr_session=self._asr_session,
                cancel_checker=self._cancel_checker,
                # Reuse the ASR stage: it already left the chunk wavs and .lab
                # files in this directory, so alignment must not transcribe
                # the song a second time.
                text_dir=self._text_dir(item),
                asr_result=(item.asr.chars_dict, item.asr.chunk_logs),
            )
        finally:
            pipeline.release_chunk_audio(item.chunks)
            item.chunks = None
        item.outcome = lyric_alignment._AlignmentOutcome(
            chars_dict=outcome.chars_dict or item.asr.chars_dict,
            pred_dict=outcome.pred_dict,
            chunk_logs=outcome.chunk_logs or item.asr.chunk_logs,
            aligned=outcome.aligned,
        )

    # -- GAME -------------------------------------------------------------
    def _run_game(self) -> None:
        from inference.API.game_api import release_game_models
        from inference.pipeline.auto_lyric_hybrid import pipeline_stage_game

        items = self._pending()
        if not items:
            return
        done = 0
        for group in _ordered_groups(items, _game_group):
            done += self._run_stage(
                STAGE_GAME, group,
                lambda item: self._run_game_for(item, pipeline_stage_game),
                done_before=done, total=len(items),
            )
            self._release(release_game_models)

    def _run_game_for(self, item: BatchItem, pipeline_stage_game) -> None:
        from inference.pipeline import auto_lyric_hybrid as pipeline

        item.chunks = pipeline.read_chunk_audio(item.base_chunks, 44100, item.storage_dir)
        self._log(f"[{item.name}] {STAGE_LABELS[STAGE_GAME]}")
        try:
            item.chunk_notes = pipeline_stage_game(
                item.config,
                item.chunks,
                cancel_checker=self._cancel_checker,
            )
        finally:
            pipeline.release_chunk_audio(item.chunks)
            item.chunks = None

    # -- export -----------------------------------------------------------
    def _run_export(self) -> None:
        from inference.pipeline.auto_lyric_hybrid import pipeline_stage_export

        items = self._pending()
        if not items:
            return
        self._run_stage(
            STAGE_EXPORT, items,
            lambda item: self._run_export_for(item, pipeline_stage_export),
        )
        # No model is loaded here; the release only returns the stage's memory.
        self._release()

    def _run_export_for(self, item: BatchItem, pipeline_stage_export) -> None:
        self._log(f"[{item.name}] {STAGE_LABELS[STAGE_EXPORT]}")
        # Without an alignment the ASR logs still belong in the match log.
        outcome = item.outcome if item.outcome is not None else item.asr
        pipeline_stage_export(
            item.config,
            item.base_chunks or [],
            item.chunk_notes or [],
            outcome,
            rmvpe_result=item.rmvpe_result,
            storage_dir=item.storage_dir,
            cancel_checker=self._cancel_checker,
        )

    # -- teardown ---------------------------------------------------------
    def _cleanup(self) -> None:
        if self._root_dir is not None:
            shutil.rmtree(self._root_dir, ignore_errors=True)
            self._root_dir = None
        self._matcher_cache.clear()


def _free_memory() -> None:
    from inference.pipeline.memory_utils import free_memory

    free_memory()


def run_auto_lyric_batch(
    tasks: list[tuple[PipelineConfig, str]],
    *,
    log=None,
    progress=None,
    cancel_checker=None,
) -> list[BatchItem]:
    """Run a queue of jobs stage by stage and return the per-job results.

    Raises ModelNotFoundError before touching any model when a job's paths are
    unusable, and CancellationError when the queue was stopped by its cancel
    checker; per-song runtime failures are recorded on the returned items
    (``failed_stage`` / ``error``) instead of aborting the batch.
    """
    from application.exceptions import CancellationError
    from application.pipeline import _validate_model_paths, open_asr_session

    # Validate up front: a bad path would otherwise surface deep inside a stage
    # as e.g. "No such file or directory: 'config.json'", and batch mode never
    # goes through run_auto_lyric_job() where this check normally lives.
    for config, _filename in tasks:
        _validate_model_paths(config)

    session = open_asr_session()
    runner = BatchRunner(
        tasks,
        log=log,
        progress=progress,
        cancel_checker=cancel_checker,
        asr_session=session,
    )
    try:
        runner.run()  # closes the session itself, including on abort
    except (BatchCancelled, InterruptedError):
        raise CancellationError("Batch was cancelled by user.") from None
    finally:
        # Covers the case where run() never started (empty queue) or the
        # runner itself raised before its teardown ran.
        runner._close_asr_session()
    return runner.items
