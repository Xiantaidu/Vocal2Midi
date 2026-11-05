from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from scripts import model_assets


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry_for_file(path: Path, data: bytes) -> dict:
    return {
        "id": "fixture",
        "name": "fixture",
        "url": "https://example.invalid/fixture.zip",
        "sha256": digest(path.read_bytes()),
        "target": "models/fixture",
        "type": "zip",
        "strip_prefix": "fixture-root",
        "files": {"weights.onnx": digest(data)},
    }


def write_zip(path: Path, members: dict[str, bytes], *, symlink: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if symlink is not None:
            info = zipfile.ZipInfo("fixture-root/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "weights.onnx")


def test_verify_entries_checks_hashes_and_ignores_unmanaged_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "models/fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.onnx").write_bytes(b"valid")
    (model_dir / "manually-added.json").write_text("keep", encoding="utf-8")
    entry = {"id": "fixture", "target": "models/fixture", "type": "zip", "files": {"weights.onnx": digest(b"valid")}}

    assert model_assets.verify_entries([entry], project_root=tmp_path) == []
    (model_dir / "weights.onnx").write_bytes(b"tampered")
    problems = model_assets.verify_entries([entry], project_root=tmp_path)
    assert len(problems) == 1
    assert "SHA256 mismatch" in problems[0]
    assert "expected" in problems[0]
    assert digest(b"tampered") in problems[0]


def test_install_verifies_archive_and_extracts_rooted_and_rootless_payload(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.zip"
    model_data = b"onnx bytes"
    write_zip(payload, {
        "fixture-root/weights.onnx": model_data,
        "fixture-root/sample.py": b"print('do not install')",
        "fixture-root/README.md": b"not listed, so not installed",
    })
    entry = entry_for_file(payload, model_data)
    monkeypatch.setattr(model_assets, "_download", lambda _url, destination: destination.write_bytes(payload.read_bytes()))

    model_assets._install_one(entry, project_root=tmp_path, force=False)

    installed = tmp_path / "models/fixture"
    assert (installed / "weights.onnx").read_bytes() == model_data
    assert not (installed / "sample.py").exists()
    assert not (installed / "README.md").exists()
    assert model_assets.verify_entries([entry], project_root=tmp_path) == []


def test_install_supports_rootless_zip_and_does_not_overwrite(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.zip"
    model_data = b"rootless model"
    write_zip(payload, {"weights.onnx": model_data})
    entry = entry_for_file(payload, model_data)
    entry["strip_prefix"] = ""
    monkeypatch.setattr(model_assets, "_download", lambda _url, destination: destination.write_bytes(payload.read_bytes()))
    model_assets._install_one(entry, project_root=tmp_path, force=False)

    installed_file = tmp_path / "models/fixture/weights.onnx"
    assert installed_file.read_bytes() == model_data
    with pytest.raises(FileExistsError, match="already exists"):
        model_assets._install_one(entry, project_root=tmp_path, force=False)
    assert installed_file.read_bytes() == model_data


@pytest.mark.parametrize("member", ["../escape.onnx", "/absolute.onnx", "C:/drive.onnx", "fixture-root\\windows.onnx"])
def test_rejects_unsafe_archive_paths(tmp_path: Path, member: str) -> None:
    archive_path = tmp_path / "bad.zip"
    write_zip(archive_path, {member: b"data"})
    entry = {"id": "fixture", "strip_prefix": "fixture-root", "files": {}}
    with pytest.raises(ValueError, match="Unsafe path"):
        model_assets._extract_verified_zip(archive_path, entry, tmp_path / "out")


def test_rejects_symlink_archive_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    write_zip(archive_path, {}, symlink="yes")
    entry = {"id": "fixture", "strip_prefix": "fixture-root", "files": {}}
    with pytest.raises(ValueError, match="non-regular"):
        model_assets._extract_verified_zip(archive_path, entry, tmp_path / "out")


def test_rejects_special_archive_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo("fixture-root/fifo")
        info.create_system = 3
        info.external_attr = (stat.S_IFIFO | 0o600) << 16
        archive.writestr(info, b"")
    entry = {"id": "fixture", "strip_prefix": "fixture-root", "files": {}}
    with pytest.raises(ValueError, match="non-regular"):
        model_assets._extract_verified_zip(archive_path, entry, tmp_path / "out")


def test_rejects_executable_archive_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "executable.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo("fixture-root/tool")
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, b"data")
    entry = {"id": "fixture", "strip_prefix": "fixture-root", "files": {}}
    with pytest.raises(ValueError, match="Executable archive entry"):
        model_assets._extract_verified_zip(archive_path, entry, tmp_path / "out")


def test_hash_mismatch_does_not_create_destination(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.zip"
    write_zip(payload, {"fixture-root/weights.onnx": b"wrong"})
    entry = entry_for_file(payload, b"correct")
    # Keep the archive digest valid while retaining an incorrect extracted file digest.
    entry["sha256"] = digest(payload.read_bytes())
    monkeypatch.setattr(model_assets, "_download", lambda _url, destination: destination.write_bytes(payload.read_bytes()))

    with pytest.raises(ValueError, match="Extracted SHA256 mismatch"):
        model_assets._install_one(entry, project_root=tmp_path, force=False)
    assert not (tmp_path / "models/fixture").exists()


def test_archive_hash_mismatch_stops_before_extraction(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.zip"
    write_zip(payload, {"fixture-root/weights.onnx": b"correct"})
    entry = entry_for_file(payload, b"correct")
    entry["sha256"] = "0" * 64
    monkeypatch.setattr(model_assets, "_download", lambda _url, destination: destination.write_bytes(payload.read_bytes()))

    with pytest.raises(ValueError, match="Archive SHA256 mismatch"):
        model_assets._install_one(entry, project_root=tmp_path, force=False)
    assert not (tmp_path / "models/fixture").exists()


def test_load_manifest_rejects_non_https_and_invalid_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    value = {"version": 1, "models": [{"id": "x", "url": "http://example.invalid/x", "sha256": "0" * 64, "target": "models/x", "type": "file"}]}
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Only https"):
        model_assets.load_manifest(manifest)


def test_failed_check_reports_missing_file(tmp_path: Path) -> None:
    entry = {"id": "fixture", "target": "models/fixture", "type": "zip", "files": {"model.onnx": digest(b"x")}}
    problems = model_assets.verify_entries([entry], project_root=tmp_path)
    assert problems == ["Missing: fixture (models/fixture/model.onnx)"]
