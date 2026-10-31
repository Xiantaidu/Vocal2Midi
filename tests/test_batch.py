"""Stage-wise batch orchestration: grouping, ordering, failure isolation."""
import pathlib
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import application.batch as batch


def _config(**overrides):
    """A PipelineConfig stand-in carrying only what the stages read."""
    values = dict(
        audio_path="a.wav",
        output_lyrics=True,
        output_formats=[],
        output_pitch_curve=False,
        rmvpe_model_path="",
        output_filename="a.wav",
        output_dir=pathlib.Path("out"),
        tempo=120.0,
        batch_size=1,
        asr_batch_size=2,
        slice_min_sec=5.0,
        slice_max_sec=15.0,
        language="zh",
        lyric_output_mode="hanzi",
        chinese_asr_engine="qwen",
        japanese_asr_engine="romaji",
        alignment_engine="tifa",
        japanese_g2p_engine="kashi-g2p-onnx",
        device="cpu",
        asr_model_path="models/asr",
        pinyin_asr_model_path="models/pinyin",
        phoneme_asr_model_path="models/romaji",
        tifa_model_path="models/tifa",
        hfa_model_dir="models/hfa",
        kashi_g2p_model_path="models/kashi",
        game_model_dir="models/GAME",
    )
    values.update(overrides)
    cfg = SimpleNamespace(**values)
    cfg.to_kwargs = lambda: {"audio_path": cfg.audio_path}
    return cfg


def _items(*configs):
    return [
        batch.BatchItem(index=i, config=cfg, filename=f"song{i}.wav")
        for i, cfg in enumerate(configs, start=1)
    ]


def _queue(*names, **overrides):
    """Configs whose audio path matches the queued filename."""
    return [(_config(audio_path=name, **overrides), name) for name in names]


def test_ordered_groups_keeps_first_appearance_order():
    a = _config(language="zh")
    b = _config(language="ja")
    items = _items(a, b, a, b, a)
    groups = batch._ordered_groups(items, lambda cfg: batch._language_group(cfg))
    assert [len(g) for g in groups] == [3, 2]
    assert groups[0][0].filename == "song1.wav"
    assert groups[1][0].filename == "song2.wav"


def test_group_keys_separate_language_and_engine():
    zh = _config(language="zh")
    ja = _config(language="ja")
    assert batch._language_group(zh) != batch._language_group(ja)

    tifa = _config(alignment_engine="tifa")
    hfa = _config(alignment_engine="hfa")
    assert batch._alignment_group(tifa) != batch._alignment_group(hfa)

    # Pitch extraction only depends on the model and the device.
    same_lang = _config(language="ja")
    assert batch._game_group(zh) == batch._game_group(same_lang)
    assert batch._game_group(zh) != batch._game_group(_config(device="cuda"))


def test_stage_order_runs_game_before_the_lyric_stages():
    """GAME notes come from the audio alone, so they are extracted before the
    lyric stages; export (lyric mapping) runs last."""
    assert batch.STAGE_ORDER == (
        batch.STAGE_SLICING,
        batch.STAGE_GAME,
        batch.STAGE_ASR,
        batch.STAGE_ALIGNMENT,
        batch.STAGE_EXPORT,
    )


@pytest.fixture
def fake_stages(monkeypatch):
    """Replace every real stage function with a recorder."""
    calls = []
    state = {"unloaded": 0, "fail_on": None, "fail_stage": None}

    def _name_from(*args, **kwargs):
        # Slicing receives the audio path, ASR/alignment the item, and GAME
        # the config first.
        for value in args:
            if isinstance(value, batch.BatchItem):
                return value.filename, value
        audio_path = kwargs.get("audio_path")
        if audio_path is None and args:
            first = args[0]
            audio_path = getattr(first, "audio_path", first if isinstance(first, str) else None)
        if audio_path is None:
            return "<none>", None
        return pathlib.Path(str(audio_path)).name, None

    def record(stage_name):
        def run(*args, **kwargs):
            filename, item = _name_from(*args, **kwargs)
            calls.append((stage_name, filename))
            if state["fail_on"] == filename and state["fail_stage"] == stage_name:
                raise RuntimeError(f"{stage_name} blew up")
            if stage_name == batch.STAGE_SLICING:
                return ([{"offset": 0.0, "waveform": [0.0]}], None)
            if stage_name == batch.STAGE_GAME:
                return [[{"onset": 0.0}]]
            return None
        return run

    # Slicing is inlined as a closure over the pipeline helper, so stub the
    # helper; ASR/alignment are real methods; GAME calls the helper directly.
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.pipeline_stage_slicing",
        record(batch.STAGE_SLICING),
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.clear_rmvpe_cache", lambda: None
    )
    monkeypatch.setattr(
        "inference.pipeline.memory_utils.free_memory", lambda: None
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.read_chunk_audio", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.release_chunk_audio", lambda *a, **k: None
    )
    for stage_name in (batch.STAGE_ASR, batch.STAGE_ALIGNMENT, batch.STAGE_EXPORT):
        monkeypatch.setattr(batch.BatchRunner, f"_run_{stage_name}_for", record(stage_name))
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.pipeline_stage_game",
        record(batch.STAGE_GAME),
    )

    monkeypatch.setattr(
        batch.BatchRunner,
        "_release",
        lambda self, *releasers: state.__setitem__("unloaded", state["unloaded"] + 1),
    )
    monkeypatch.setattr(batch.BatchRunner, "_close_asr_session", lambda self: None)
    return SimpleNamespace(calls=calls, state=state)


def test_runner_visits_every_song_per_stage(fake_stages):
    runner = batch.BatchRunner(_queue("a.wav", "b.wav"))
    runner.run()

    # Every stage must sweep the whole queue before the next stage starts.
    order = [(stage, name) for stage, name in fake_stages.calls]
    assert order == [
        (batch.STAGE_SLICING, "a.wav"),
        (batch.STAGE_SLICING, "b.wav"),
        (batch.STAGE_GAME, "a.wav"),
        (batch.STAGE_GAME, "b.wav"),
        (batch.STAGE_ASR, "a.wav"),
        (batch.STAGE_ASR, "b.wav"),
        (batch.STAGE_ALIGNMENT, "a.wav"),
        (batch.STAGE_ALIGNMENT, "b.wav"),
        (batch.STAGE_EXPORT, "a.wav"),
        (batch.STAGE_EXPORT, "b.wav"),
    ]


def test_failed_song_is_isolated_and_later_stages_skip_it(fake_stages, capsys):
    fake_stages.state.update(fail_on="a.wav", fail_stage=batch.STAGE_ASR)
    runner = batch.BatchRunner(_queue("a.wav", "b.wav"))
    runner.run()

    failed, ok = runner.items
    assert failed.failed_stage == batch.STAGE_ASR
    assert "blew up" in failed.error
    assert ok.failed_stage is None
    # GAME ran before the failure; the lyric stages never saw the broken song.
    assert (batch.STAGE_GAME, "a.wav") in fake_stages.calls
    assert (batch.STAGE_ALIGNMENT, "a.wav") not in fake_stages.calls
    assert (batch.STAGE_EXPORT, "a.wav") not in fake_stages.calls
    assert (batch.STAGE_EXPORT, "b.wav") in fake_stages.calls


def test_stage_release_happens_after_each_stage(fake_stages):
    runner = batch.BatchRunner(_queue("a.wav", "b.wav"))
    runner.run()
    # One release per stage, not per song.
    assert fake_stages.state["unloaded"] == len(batch.STAGE_ORDER)


def test_cancellation_stops_the_queue(fake_stages):
    runner = batch.BatchRunner(_queue("a.wav", "b.wav"), cancel_checker=lambda: True)
    with pytest.raises(batch.BatchCancelled):
        runner.run()
    assert fake_stages.calls == []  # cancelled before the first song


def test_cancel_mid_queue_stops_before_the_next_song(fake_stages):
    def cancel_after_first():
        # Let the first song slice, then stop the queue.
        return len([c for c in fake_stages.calls if c[0] == batch.STAGE_SLICING]) >= 1

    runner = batch.BatchRunner(_queue("a.wav", "b.wav"), cancel_checker=cancel_after_first)
    with pytest.raises(batch.BatchCancelled):
        runner.run()

    sliced = [name for stage, name in fake_stages.calls if stage == batch.STAGE_SLICING]
    assert sliced == ["a.wav"]  # the second song never starts
    assert ("asr", "b.wav") not in fake_stages.calls


def test_temp_dir_is_removed_after_run(fake_stages):
    runner = batch.BatchRunner(_queue("a.wav"))
    runner.run()
    assert runner._root_dir is None


def test_alignment_reuses_asr_stage_instead_of_transcribing_again(monkeypatch, tmp_path):
    """Regression: alignment called _run_lyric_alignment(), which re-ran ASR.

    The user's log showed "Stage 1/3: Running ASR" under the alignment stage,
    i.e. every song was transcribed twice.
    """
    from inference.pipeline import lyric_alignment
    from inference.pipeline.lyric_alignment import _AlignmentOutcome as _Outcome

    seen = {}

    def fake_alignment(chunks, sr, ctx, matcher, **kwargs):
        seen.update(kwargs)
        return _Outcome(chars_dict={"chunk_0": ["ni"]}, pred_dict={"chunk_0": []},
                        chunk_logs=["ok"], aligned=True)

    monkeypatch.setattr(
        lyric_alignment, "_run_lyric_alignment", fake_alignment
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.read_chunk_audio", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.release_chunk_audio", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid._prepare_context",
        lambda *a, **k: SimpleNamespace(
            language="zh", lyric_output_mode="hanzi", use_pinyin_asr=False,
            device="cpu", fa_language="zh", output_key="a",
            output_dir=tmp_path, output_format_set=set(),
        ),
    )
    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.prepare_lyric_matcher",
        lambda cfg, cache=None: None,
    )

    runner = batch.BatchRunner(_queue("a.wav"))
    runner._root_dir = tmp_path
    item = runner.items[0]
    item.base_chunks = [{"offset": 0.0, "waveform": [0.0]}]
    item.storage_dir = tmp_path / "job_001"
    item.asr = _Outcome(chars_dict={"chunk_0": ["ni"]}, pred_dict={},
                        chunk_logs=["asr log"], aligned=False)

    runner._run_alignment_for(item)

    # The pre-computed ASR must be handed over, and the directory holding the
    # .lab files must be the one the ASR stage wrote.
    assert seen["asr_result"] == ({"chunk_0": ["ni"]}, ["asr log"])
    assert pathlib.Path(seen["text_dir"]) == tmp_path / "job_001" / "text"
    assert item.outcome.pred_dict == {"chunk_0": []}


def test_progress_counts_across_groups_not_per_group():
    """Regression: each language group restarted its counter at 1/1.

    Three different languages make three ASR groups, so the reported progress
    must run 1/3, 2/3, 3/3 rather than 1/1 three times.
    """
    seen = []
    tasks = [
        (_config(language=lang, audio_path=f"{lang}.wav"), f"{lang}.wav")
        for lang in ("zh", "en", "ja")
    ]
    runner = batch.BatchRunner(
        tasks,
        progress=lambda stage, done, total, name: seen.append((stage, done, total, name)),
    )
    runner._root_dir = pathlib.Path(tempfile.mkdtemp(prefix="v2m_prog_"))
    try:
        done = 0
        for group in batch._ordered_groups(runner._pending(), batch._asr_group):
            done += runner._run_stage(
                batch.STAGE_ASR, group, lambda item: None,
                done_before=done, total=len(runner.items),
            )
    finally:
        shutil.rmtree(runner._root_dir, ignore_errors=True)

    assert [(d, t) for _s, d, t, _n in seen] == [(1, 3), (2, 3), (3, 3)]


def test_slicing_stage_passes_audio_path_exactly_once(monkeypatch, tmp_path):
    """Regression: S1 passed audio_path both positionally and via to_kwargs().

    The stub mirrors the real signature on purpose: a permissive
    ``(*args, **kwargs)`` stub silently accepts the duplicate, which is how
    "got multiple values for argument 'audio_path'" reached the user.
    """
    received = {}

    def stage(audio_path=None, *, chunk_storage_dir=None, cancel_checker=None, **kwargs):
        received["audio_path"] = audio_path
        received["kwargs"] = kwargs
        return ([{"offset": 0.0, "waveform": [0.0]}], None)

    monkeypatch.setattr(
        "inference.pipeline.auto_lyric_hybrid.pipeline_stage_slicing", stage
    )
    monkeypatch.setattr("inference.pipeline.memory_utils.free_memory", lambda: None)
    monkeypatch.setattr(batch.BatchRunner, "_release", lambda self, *releasers: None)

    runner = batch.BatchRunner(_queue("a.wav"))
    runner._root_dir = tmp_path
    runner._run_slicing()

    # Surface the stage's own error first: a duplicate-argument TypeError is
    # caught by _run_stage and would otherwise only show up as a KeyError.
    item = runner.items[0]
    assert item.failed_stage is None, f"slicing failed: {item.error}"
    assert received["audio_path"] == "a.wav"
    # audio_path must travel by one route only, and cancel_checker arrives
    # through the runner rather than the config.
    assert "audio_path" not in received["kwargs"]
    assert "cancel_checker" not in received["kwargs"]


def test_empty_queue_is_a_noop():
    runner = batch.BatchRunner([])
    assert runner.run() is None
    assert runner.total == 0


def test_run_auto_lyric_batch_closes_session_once(monkeypatch):
    session = MagicMock()
    monkeypatch.setattr("application.pipeline.open_asr_session", lambda: session)
    # Session lifecycle is what this test covers; path validation has its own.
    monkeypatch.setattr("application.pipeline._validate_model_paths", lambda cfg: None)
    monkeypatch.setattr(batch.BatchRunner, "run", lambda self: None)

    items = batch.run_auto_lyric_batch(_queue("a.wav"))

    assert len(items) == 1
    session.close.assert_called_once()


def test_run_auto_lyric_batch_maps_cancel_to_cancellation_error(monkeypatch):
    from application.exceptions import CancellationError

    session = MagicMock()
    monkeypatch.setattr("application.pipeline.open_asr_session", lambda: session)
    monkeypatch.setattr("application.pipeline._validate_model_paths", lambda cfg: None)

    def boom(self):
        raise batch.BatchCancelled("stop")

    monkeypatch.setattr(batch.BatchRunner, "run", boom)

    with pytest.raises(CancellationError):
        batch.run_auto_lyric_batch(_queue("a.wav"))
    session.close.assert_called_once()


def test_run_auto_lyric_batch_rejects_missing_model_paths(monkeypatch, tmp_path):
    """A bad path must fail with ModelNotFoundError, not a stage-level
    FileNotFoundError on 'config.json'.

    Batch mode never goes through run_auto_lyric_job(), which is the only
    place that validated model paths before.
    """
    from application.exceptions import ModelNotFoundError

    def fail_hard():
        raise AssertionError("the ASR session must not open before validation")

    monkeypatch.setattr("application.pipeline.open_asr_session", fail_hard)

    # A config whose TiFA directory does not exist.
    config = _config(tifa_model_path=str(tmp_path / "no-such-tifa"))
    with pytest.raises(ModelNotFoundError) as excinfo:
        batch.run_auto_lyric_batch([(config, "a.wav")])

    message = str(excinfo.value)
    assert "TiFA model dir" in message
    assert "no-such-tifa" in message
