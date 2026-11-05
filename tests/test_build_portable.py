from __future__ import annotations

from pathlib import Path

import pytest

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


def test_portable_model_verification_skips_unmanaged_qwen(monkeypatch, tmp_path):
    monkeypatch.setattr(build_portable, "PROJECT_ROOT", tmp_path)
    assert build_portable.verify_selected_models(["qwen"]) == []


def test_portable_model_verification_reports_managed_missing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(build_portable, "PROJECT_ROOT", tmp_path)
    problems = build_portable.verify_selected_models(["game"])

    assert any("Missing: game (models/GAME-1.0.3-medium-onnx/" in problem for problem in problems)


def test_portable_build_stops_before_creating_output_for_missing_managed_model(
    monkeypatch, tmp_path, capsys
):
    output_dir = tmp_path / "portable"
    monkeypatch.setattr(build_portable, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        build_portable,
        "parse_args",
        lambda: type("Args", (), {"models": ["game"], "output_dir": output_dir, "python_dir": tmp_path})(),
    )

    with pytest.raises(SystemExit, match="1"):
        build_portable.main()

    assert "Model verification failed; portable build stopped:" in capsys.readouterr().err
    assert not output_dir.exists()


def test_write_launcher_creates_both_cli_launchers(tmp_path):
    build_portable.write_launcher(tmp_path, runtime_mode_used="copy")

    slice_cli = tmp_path / "Run Slice ASR CLI.bat"
    auto_lyric_cli = tmp_path / "Run Auto Lyric CLI.bat"
    assert slice_cli.is_file()
    assert auto_lyric_cli.is_file()
    content = auto_lyric_cli.read_text(encoding="utf-8")
    assert "scripts\\auto_lyric_cli.py %*" in content  # arguments pass through
    assert "V2M_PORTABLE_ROOT" in content  # same portable env block as the other launchers
