from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.API import rmvpe_api


class _FakeInputMeta:
    def __init__(self, name: str = "input"):
        self.name = name
        self.shape = [1, rmvpe_api.N_MELS, "time"]
        self.type = "tensor(float)"


class _FakeSession:
    def __init__(self, run_output: np.ndarray | None = None):
        self._run_output = run_output if run_output is not None else np.zeros((1, 513, rmvpe_api.N_CLASS), dtype=np.float32)

    def get_inputs(self):
        return [_FakeInputMeta()]

    def run(self, _output_names, _feeds):
        return [self._run_output.copy()]


class _FakeOrt:
    class SessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.enable_mem_pattern = True
            self.enable_cpu_mem_arena = True

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    def __init__(self, available_providers: list[str], run_output: np.ndarray | None = None):
        self._available_providers = list(available_providers)
        self._run_output = run_output
        self.session_calls: list[dict[str, object]] = []

    def get_available_providers(self):
        return list(self._available_providers)

    def InferenceSession(self, path, sess_options=None, providers=None):
        self.session_calls.append(
            {
                "path": path,
                "providers": list(providers or []),
                "sess_options": sess_options,
            }
        )
        return _FakeSession(self._run_output)


def _create_fake_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "rmvpe.onnx"
    model_path.write_bytes(b"fake")
    return model_path


def _patch_provider_resolution(monkeypatch, provider_name: str, providers: list[str]) -> None:
    monkeypatch.setattr(
        rmvpe_api,
        "resolve_onnx_providers",
        lambda device, *, label="RMVPE ONNX": (provider_name, list(providers)),
    )


def test_rmvpe_uses_cpu_provider_when_requested(monkeypatch, tmp_path):
    fake_ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(rmvpe_api, "ort", fake_ort)
    _patch_provider_resolution(monkeypatch, "cpu", ["CPUExecutionProvider"])

    transcriber = rmvpe_api.RmvpeTranscriber(_create_fake_model(tmp_path), device="cpu")

    assert transcriber.provider_name == "cpu"
    assert fake_ort.session_calls[0]["providers"] == ["CPUExecutionProvider"]


def test_rmvpe_falls_back_to_cpu_when_dml_is_unavailable(monkeypatch, tmp_path):
    fake_ort = _FakeOrt(["CPUExecutionProvider"])
    monkeypatch.setattr(rmvpe_api, "ort", fake_ort)
    _patch_provider_resolution(monkeypatch, "cpu", ["CPUExecutionProvider"])

    transcriber = rmvpe_api.RmvpeTranscriber(_create_fake_model(tmp_path), device="cuda")

    assert transcriber.provider_name == "cpu"
    assert fake_ort.session_calls[0]["providers"] == ["CPUExecutionProvider"]


def test_rmvpe_infer_returns_same_public_result_shape(monkeypatch, tmp_path):
    fake_ort = _FakeOrt(["CPUExecutionProvider"])
    monkeypatch.setattr(rmvpe_api, "ort", fake_ort)
    _patch_provider_resolution(monkeypatch, "cpu", ["CPUExecutionProvider"])
    transcriber = rmvpe_api.RmvpeTranscriber(_create_fake_model(tmp_path), device="cpu")

    salience = np.zeros((4, rmvpe_api.N_CLASS), dtype=np.float32)
    salience[:2, 100] = 1.0
    salience[2:, 120] = 1.0
    monkeypatch.setattr(transcriber, "_inference_salience", lambda waveform, cancel_checker=None: salience)

    result = transcriber.infer(np.zeros(16000, dtype=np.float32), sample_rate=rmvpe_api.SAMPLE_RATE)

    assert result.time_step_seconds == rmvpe_api.HOP_LENGTH / rmvpe_api.SAMPLE_RATE
    assert result.midi_pitch.shape == (4,)
    assert result.voiced_mask.shape == (4,)
    assert result.voiced_mask.tolist() == [True, True, True, True]
    assert np.all(np.isfinite(result.midi_pitch))
    assert result.midi_pitch[2] > result.midi_pitch[0]


def test_rmvpe_waveform_to_mel_trims_legacy_extra_frame(monkeypatch, tmp_path):
    fake_ort = _FakeOrt(["CPUExecutionProvider"])
    monkeypatch.setattr(rmvpe_api, "ort", fake_ort)
    _patch_provider_resolution(monkeypatch, "cpu", ["CPUExecutionProvider"])
    transcriber = rmvpe_api.RmvpeTranscriber(_create_fake_model(tmp_path), device="cpu")

    legacy_segment = np.zeros((rmvpe_api.SEG_LEN + rmvpe_api.WINDOW_LENGTH,), dtype=np.float32)
    mel = transcriber._waveform_to_mel(legacy_segment)

    assert mel.shape == (rmvpe_api.N_MELS, rmvpe_api.SEG_FRAMES)
