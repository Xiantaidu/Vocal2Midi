from __future__ import annotations

from pathlib import Path

from scripts import build_portable


def test_detect_runtime_mode_prefers_copy_without_conda_pack(monkeypatch, tmp_path):
    (tmp_path / "conda-meta").mkdir()
    monkeypatch.setattr(build_portable, "has_conda_pack", lambda: False)

    assert build_portable.detect_runtime_mode(tmp_path, "auto") == "copy"


def test_get_model_copy_plan_uses_onnx_rmvpe_only():
    plan = build_portable.get_model_copy_plan(["rmvpe"])

    assert len(plan) == 1
    assert plan[0].source == Path("e:/Vocal2Midi/experiments/RMVPE/rmvpe.onnx")
    assert plan[0].kind == "file"
