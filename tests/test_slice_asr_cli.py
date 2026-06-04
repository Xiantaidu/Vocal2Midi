import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "slice_asr_cli.py"


def _load_slice_asr_cli():
    stubs = {
        "inference": types.ModuleType("inference"),
        "inference.API": types.ModuleType("inference.API"),
        "inference.API.asr_api": types.ModuleType("inference.API.asr_api"),
        "inference.API.rmvpe_api": types.ModuleType("inference.API.rmvpe_api"),
        "inference.API.slicer_api": types.ModuleType("inference.API.slicer_api"),
        "inference.device_utils": types.ModuleType("inference.device_utils"),
    }
    stubs["inference"].__path__ = []
    stubs["inference.API"].__path__ = []
    stubs["inference.API.asr_api"].batch_transcribe_asr = lambda *args, **kwargs: None
    stubs["inference.API.asr_api"].load_qwen_model = lambda *args, **kwargs: None
    stubs["inference.API.asr_api"].clear_qwen_model_cache = lambda *args, **kwargs: None
    stubs["inference.API.rmvpe_api"].RmvpeTranscriber = object
    stubs["inference.API.slicer_api"].slice_audio = lambda *args, **kwargs: None
    stubs["inference.device_utils"].RUNTIME_DEVICE_CHOICES = ("dml", "cpu", "cuda")
    stubs["inference.device_utils"].normalize_runtime_device = lambda device: device

    original_modules = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location("slice_asr_cli_test_module", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


slice_asr_cli = _load_slice_asr_cli()


def _make_mojibake(text: str) -> str:
    return text.encode("utf-8").decode("gb18030", errors="replace")


def test_argparser_accepts_no_slice():
    args = slice_asr_cli.build_argparser().parse_args(
        ["input_dir", "output_dir", "--asr-model", "model", "--no-slice"]
    )

    assert args.no_slice is True
    assert args.language == "zh"


def test_normalize_slicing_method_accepts_english_chinese_and_mojibake():
    assert slice_asr_cli.normalize_slicing_method("default") == "default"
    assert slice_asr_cli.normalize_slicing_method("智能切片") == "smart"
    assert slice_asr_cli.normalize_slicing_method(_make_mojibake("默认切片")) == "default"

    degraded = _make_mojibake("启发式切片")[:-1] + "?"
    assert slice_asr_cli.normalize_slicing_method(degraded) == "heuristic"


def test_process_one_file_no_slice_bypasses_slicer(monkeypatch, tmp_path):
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"fake")

    waveform = np.zeros(44100, dtype=np.float32)
    monkeypatch.setattr(slice_asr_cli, "load_audio", lambda path, sr=44100: (waveform, sr))

    def fake_slice_audio(*args, **kwargs):
        raise AssertionError("slice_audio should not be called when no_slice=True")

    monkeypatch.setattr(slice_asr_cli, "slice_audio", fake_slice_audio)

    def fake_rmvpe_transcriber(*args, **kwargs):
        raise AssertionError("RmvpeTranscriber should not be called when no_slice=True")

    monkeypatch.setattr(slice_asr_cli, "RmvpeTranscriber", fake_rmvpe_transcriber)
    monkeypatch.setattr(slice_asr_cli, "save_chunks", lambda *args, **kwargs: [])

    captured = {}

    def fake_batch_transcribe_asr(**kwargs):
        captured["chunks"] = kwargs["chunks"]
        return [types.SimpleNamespace(text="hello")], [0]

    monkeypatch.setattr(slice_asr_cli, "batch_transcribe_asr", fake_batch_transcribe_asr)

    chunks, labs = slice_asr_cli.process_one_file(
        audio_path=audio_path,
        output_dir=tmp_path / "out",
        asr_model_path="experiments/Qwen3-ASR-1.7B-dml",
        device="cpu",
        language="ja",
        slicing_method="default",
        asr_batch_size=1,
        no_slice=True,
    )

    assert chunks == 1
    assert labs == 1
    assert len(captured["chunks"]) == 1
    assert captured["chunks"][0]["offset"] == 0.0
    assert np.array_equal(captured["chunks"][0]["waveform"], waveform)

    output_key = slice_asr_cli.source_key(audio_path, slice_asr_cli.file_md5(audio_path))
    lab_files = list((tmp_path / "out" / "labs" / output_key).glob("*.lab"))
    assert len(lab_files) == 1
    assert lab_files[0].read_text(encoding="utf-8") == "hello"
