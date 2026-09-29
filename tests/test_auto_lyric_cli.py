"""Tests for the headless auto_lyric_cli script."""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "auto_lyric_cli.py"


def _load_cli():
    # The script imports heavy modules lazily, so loading it directly is cheap.
    spec = importlib.util.spec_from_file_location("auto_lyric_cli_test_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auto_lyric_cli = _load_cli()


@pytest.fixture
def defaults():
    return dict(auto_lyric_cli.FALLBACK_DEFAULTS, output_dir="out")


def test_argparser_defaults(defaults):
    args = auto_lyric_cli.build_argparser(defaults).parse_args(["a.wav"])
    assert args.output_dir == Path("out")
    assert args.language == "zh"
    assert args.lyric_format is None  # resolved from settings per language in main
    assert args.chinese_asr == "qwen"
    assert args.japanese_asr == "romaji"
    assert args.aligner == "hfa"
    assert args.slicing == "smart"
    assert args.quant_step == 0
    assert args.quant_mode == "smart"
    assert args.formats == ["mid"]
    assert args.no_lyrics is False
    assert args.round_pitch is True
    assert args.pitch_curve is False


def test_argparser_accepts_flags(defaults):
    args = auto_lyric_cli.build_argparser(defaults).parse_args(
        [
            "a.wav", "b.wav", "-o", "somewhere", "--language", "ja",
            "--lyric-format", "kana", "--chinese-asr", "pinyin",
            "--japanese-asr", "qwen", "--no-lyrics", "--formats", "ustx", "mid",
            "--quant-step", "480", "--quant-mode", "simple", "--no-round-pitch",
            "--pitch-curve", "--no-recursive",
        ]
    )
    assert len(args.inputs) == 2
    assert args.output_dir == Path("somewhere")
    assert args.lyric_format == "kana"
    assert args.no_lyrics is True
    assert args.formats == ["ustx", "mid"]
    assert args.quant_step == 480
    assert args.round_pitch is False
    assert args.pitch_curve is True
    assert args.no_recursive is True


def test_validate_args_rejects_invalid_language_format(defaults):
    parser = auto_lyric_cli.build_argparser(defaults)
    args = parser.parse_args(["a.wav", "--language", "ja", "--lyric-format", "hanzi"])
    with pytest.raises(ValueError, match="not valid for language"):
        auto_lyric_cli.validate_args(args)


def test_validate_args_rejects_no_lyrics_with_lyrics(defaults):
    parser = auto_lyric_cli.build_argparser(defaults)
    args = parser.parse_args(["a.wav", "--no-lyrics", "--lyrics", "hello"])
    with pytest.raises(ValueError, match="--no-lyrics"):
        auto_lyric_cli.validate_args(args)


def test_validate_args_warns_but_allows_pinyin_engine_hanzi(defaults, caplog):
    parser = auto_lyric_cli.build_argparser(defaults)
    args = parser.parse_args(["a.wav", "--chinese-asr", "pinyin", "--lyric-format", "hanzi"])
    auto_lyric_cli.validate_args(args)  # must not raise; backend coerces to pinyin


def test_validate_args_missing_lyrics_file(defaults):
    parser = auto_lyric_cli.build_argparser(defaults)
    args = parser.parse_args(["a.wav", "--lyrics-file", "missing.txt"])
    with pytest.raises(FileNotFoundError):
        auto_lyric_cli.validate_args(args)


def test_collect_inputs_expands_filters_and_dedupes(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.mp3").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.wav").write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "x.wav").write_bytes(b"")

    files = auto_lyric_cli.collect_inputs([tmp_path / "a.wav", tmp_path], out_dir)
    names = [p.name for p in files]
    assert names == ["a.wav", "b.mp3", "c.wav"]  # txt filtered, out/ excluded, deduped

    flat = auto_lyric_cli.collect_inputs([tmp_path], out_dir, recursive=False)
    assert [p.name for p in flat] == ["a.wav", "b.mp3"]


def test_collect_inputs_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        auto_lyric_cli.collect_inputs([tmp_path / "missing.wav"])
    bad = tmp_path / "song.txt"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported audio file"):
        auto_lyric_cli.collect_inputs([bad])
    with pytest.raises(ValueError, match="no audio files"):
        auto_lyric_cli.collect_inputs([tmp_path / "empty_dir"] if (tmp_path / "empty_dir").mkdir() is None else [])


def test_load_settings_defaults_reads_and_converts(monkeypatch):
    class FakeSettings:
        def __init__(self, values):
            self.values = values

        def value(self, key, default=None):
            return self.values.get(key, default)

    values = {
        "batch_size": "3",  # ini values arrive as strings
        "round_pitch": "false",
        "save_dir": "z:/custom_out",
        "lyric_output_mode_ja": "kana",
    }
    monkeypatch.setattr(
        "gui.settings_utils.create_app_settings", lambda: FakeSettings(values)
    )
    defaults = auto_lyric_cli._load_settings_defaults()
    assert defaults["batch_size"] == 3
    assert defaults["asr_batch_size"] == 2  # falls back when the key is absent
    assert defaults["round_pitch"] is False
    assert defaults["output_dir"] == "z:/custom_out"
    assert defaults["lyric_output_mode_ja"] == "kana"
    assert defaults["game_model"] == "models/GAME-1.0.3-medium-onnx"


def test_build_config_maps_fields(tmp_path, defaults):
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("你好 世界", encoding="utf-8")
    parser = auto_lyric_cli.build_argparser(defaults)
    args = parser.parse_args(
        ["a.wav", "-o", str(tmp_path / "out"), "--language", "zh", "--lyric-format", "hanzi",
         "--lyrics-file", str(lyrics_file), "--quant-step", "480", "--chinese-asr", "pinyin"]
    )
    config = auto_lyric_cli.build_config(args, Path("a.wav"), ts_list=[0.0], asr_session=MagicMock())

    assert config.audio_path == "a.wav"
    assert config.output_filename == "a.wav"
    assert config.language == "zh"
    assert config.lyric_output_mode == "hanzi"
    assert config.original_lyrics == "你好 世界"
    assert config.chinese_asr_engine == "pinyin"
    assert config.quantization_step == 480
    assert config.output_formats == ["mid"]
    assert config.output_dir == tmp_path / "out"
    assert config.asr_session is not None


def test_main_batch_runs_shares_session_and_reports_failures(monkeypatch, tmp_path, defaults):
    (tmp_path / "good.wav").write_bytes(b"")
    (tmp_path / "bad.wav").write_bytes(b"")

    calls = []

    def fake_run(cfg):
        calls.append(cfg.audio_path)
        if cfg.audio_path.endswith("bad.wav"):
            raise RuntimeError("boom")

    session = MagicMock()
    monkeypatch.setattr("application.pipeline.run_auto_lyric_job", fake_run)
    monkeypatch.setattr("application.pipeline.open_asr_session", lambda: session)

    exit_code = auto_lyric_cli.main(
        [str(tmp_path / "good.wav"), str(tmp_path / "bad.wav"), "-o", str(tmp_path / "out")]
    )

    assert exit_code == 1
    assert len(calls) == 2
    session.close.assert_called_once()  # one shared session for the whole batch


def test_main_success_returns_zero(monkeypatch, tmp_path, defaults):
    (tmp_path / "ok.wav").write_bytes(b"")
    session = MagicMock()
    monkeypatch.setattr("application.pipeline.run_auto_lyric_job", lambda cfg: None)
    monkeypatch.setattr("application.pipeline.open_asr_session", lambda: session)

    exit_code = auto_lyric_cli.main([str(tmp_path / "ok.wav"), "-o", str(tmp_path / "out")])

    assert exit_code == 0
    session.close.assert_called_once()


def test_main_cancelled_returns_130(monkeypatch, tmp_path, defaults):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")
    session = MagicMock()

    def fake_run(cfg):
        if cfg.audio_path.endswith("a.wav"):
            raise KeyboardInterrupt
        raise AssertionError("must stop after cancellation")

    monkeypatch.setattr("application.pipeline.run_auto_lyric_job", fake_run)
    monkeypatch.setattr("application.pipeline.open_asr_session", lambda: session)

    exit_code = auto_lyric_cli.main(
        [str(tmp_path / "a.wav"), str(tmp_path / "b.wav"), "-o", str(tmp_path / "out")]
    )

    assert exit_code == 130
    session.close.assert_called_once()
