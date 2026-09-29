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
    assert plan[0].source == build_portable.PROJECT_ROOT / Path("models/RMVPE/rmvpe.onnx")
    assert plan[0].kind == "file"


def test_write_launcher_creates_both_cli_launchers(tmp_path):
    build_portable.write_launcher(tmp_path, runtime_mode_used="copy")

    slice_cli = tmp_path / "Run Slice ASR CLI.bat"
    auto_lyric_cli = tmp_path / "Run Auto Lyric CLI.bat"
    assert slice_cli.is_file()
    assert auto_lyric_cli.is_file()
    content = auto_lyric_cli.read_text(encoding="utf-8")
    assert "scripts\\auto_lyric_cli.py %*" in content  # arguments pass through
    assert "V2M_PORTABLE_ROOT" in content  # same portable env block as the other launchers
