from __future__ import annotations

import json
from pathlib import Path

from inference import device_utils
from inference.game import onnx_runtime as game_onnx_runtime
from inference.HubertFA import onnx_infer
from inference.romaji_asr import common as romaji_common


class _FakeInputMeta:
    def __init__(self, shape=None, ort_type: str = "tensor(float)"):
        self.shape = shape if shape is not None else ["batch", "time"]
        self.type = ort_type


class _FakeSession:
    def __init__(self, providers: list[str] | None = None):
        self._providers = [
            provider[0] if isinstance(provider, tuple) else provider
            for provider in list(providers or [])
        ]

    def get_inputs(self):
        return [_FakeInputMeta()]

    def get_outputs(self):
        return []

    def get_providers(self):
        return list(self._providers)


class _FakeOrt:
    class SessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.enable_mem_pattern = True
            self.enable_cpu_mem_arena = True
            self.execution_mode = None

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    def __init__(self, available_providers: list[str]):
        self._available_providers = list(available_providers)
        self.session_calls: list[dict[str, object]] = []

    def get_available_providers(self):
        return list(self._available_providers)

    def InferenceSession(self, path, sess_options=None, providers=None):
        call = {
            "path": str(path),
            "providers": list(providers or []),
            "sess_options": sess_options,
        }
        self.session_calls.append(call)
        return _FakeSession(call["providers"])


def test_game_uses_cpu_provider_when_requested(monkeypatch, tmp_path: Path):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(game_onnx_runtime, "ort", fake_ort)
    monkeypatch.setattr(
        game_onnx_runtime,
        "resolve_onnx_providers",
        lambda device, *, label="GAME ONNX": ("cpu", ["CPUExecutionProvider"]),
    )

    model_dir = tmp_path / "game"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"samplerate": 16000, "timestep": 0.01, "loop": True, "languages": {"ja": 0}}),
        encoding="utf-8",
    )
    for filename in ["encoder.onnx", "segmenter.onnx", "estimator.onnx", "dur2bd.onnx", "bd2dur.onnx"]:
        (model_dir / filename).write_bytes(b"fake")

    model = game_onnx_runtime.GameOnnxModel(model_dir, requested_device="cpu")

    assert model.provider_name == "cpu"
    assert fake_ort.session_calls
    assert all(call["providers"] == ["CPUExecutionProvider"] for call in fake_ort.session_calls)


def test_hubertfa_helper_session_uses_cpu_provider_when_requested(monkeypatch, tmp_path: Path):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(onnx_infer, "ort", fake_ort)
    monkeypatch.setattr(
        onnx_infer,
        "resolve_onnx_providers",
        lambda device, *, label="HubertFA ONNX": ("cpu", ["CPUExecutionProvider"]),
    )

    session = onnx_infer.InferenceOnnx.create_session(tmp_path / "model.onnx", device="cpu")

    assert isinstance(session, _FakeSession)
    assert fake_ort.session_calls[0]["providers"] == ["CPUExecutionProvider"]


def test_romaji_create_session_uses_cpu_provider_when_requested(monkeypatch, tmp_path: Path):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(romaji_common, "ort", fake_ort)

    session = romaji_common.create_session(tmp_path / "model.onnx", provider="cpu")

    assert isinstance(session, _FakeSession)
    assert fake_ort.session_calls[0]["providers"] == ["CPUExecutionProvider"]


def test_resolve_onnx_providers_selects_explicit_dml_device_id(monkeypatch):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(device_utils, "ort", fake_ort)
    monkeypatch.setattr(
        device_utils,
        "_select_preferred_dml_adapter",
        lambda: device_utils._DxgiAdapterInfo(
            index=2,
            name="Discrete GPU",
            dedicated_vram_bytes=4 * (1 << 30),
            is_software=False,
        ),
    )

    provider_name, providers = device_utils.resolve_onnx_providers("dml", label="Test DML")

    assert provider_name == "dml"
    assert providers[0] == ("DmlExecutionProvider", {"device_id": "2"})
    assert providers[1] == "CPUExecutionProvider"


def test_resolve_onnx_providers_falls_back_to_cpu_without_eligible_dml_adapter(monkeypatch):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(device_utils, "ort", fake_ort)
    monkeypatch.setattr(device_utils, "_select_preferred_dml_adapter", lambda: None)

    provider_name, providers = device_utils.resolve_onnx_providers("dml", label="Test DML")

    assert provider_name == "cpu"
    assert providers == ["CPUExecutionProvider"]


def test_romaji_create_session_uses_selected_dml_adapter(monkeypatch, tmp_path: Path):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(romaji_common, "ort", fake_ort)
    monkeypatch.setattr(
        romaji_common,
        "resolve_onnx_providers",
        lambda device, *, label="Romaji ASR ONNX": (
            "dml",
            [("DmlExecutionProvider", {"device_id": "3"}), "CPUExecutionProvider"],
        ),
    )

    session = romaji_common.create_session(tmp_path / "model.onnx", provider="dml")

    assert isinstance(session, _FakeSession)
    assert fake_ort.session_calls[0]["providers"][0] == ("DmlExecutionProvider", {"device_id": "3"})
    assert session.get_providers()[0] == "DmlExecutionProvider"
