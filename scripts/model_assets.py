"""Manifest-backed download, installation, and verification of model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "model_manifest.json"
BLOCKED_SUFFIXES = {
    ".py", ".pyc", ".pyo", ".pt", ".pth", ".pkl", ".pickle", ".ckpt",
    ".pyd", ".dill", ".joblib", ".exe", ".com", ".dll", ".so", ".dylib",
    ".node", ".wasm", ".elf", ".bat", ".cmd", ".ps1", ".sh", ".vbs",
    ".js", ".mjs", ".app", ".msi", ".sys", ".scr", ".ocx", ".drv",
    ".jar", ".class", ".o", ".a", ".c", ".cpp", ".h", ".hpp", ".rb",
    ".pl", ".php", ".lua", ".ipynb", ".dmg", ".pkg", ".deb", ".rpm",
}
EXECUTABLE_MAGICS = (b"MZ", b"\x7fELF", b"\xfe\xed\xfa", b"\xcf\xfa\xed\xfe", b"#!")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("models"), list):
        raise ValueError(f"Unsupported or invalid model manifest: {path}")
    entries = data["models"]
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("id") in ids:
            raise ValueError("Manifest model entries must be objects with unique ids")
        ids.add(entry["id"])
        parsed = urllib.parse.urlparse(entry.get("url", ""))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"Only https model URLs are allowed: {entry.get('id')}")
        _validate_digest(entry.get("sha256"), f"archive hash for {entry['id']}")
        if entry.get("type") not in {"zip", "file"}:
            raise ValueError(f"Unsupported asset type for {entry['id']}")
        _safe_relative_path(entry.get("target", ""), context=f"target for {entry['id']}")
        if entry["type"] == "zip":
            if not isinstance(entry.get("files"), dict) or not entry["files"]:
                raise ValueError(f"Zip asset has no file hashes: {entry['id']}")
            for relative, digest in entry["files"].items():
                _safe_relative_path(relative, context=f"manifest file for {entry['id']}")
                _validate_digest(digest, f"file hash for {entry['id']}:{relative}")
        elif "files" in entry:
            raise ValueError(f"Raw file asset cannot declare extracted files: {entry['id']}")
    return entries


def _validate_digest(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256 {context}")


def _safe_relative_path(value: str, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe path in {context}: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts) or any(":" in part for part in raw_parts):
        raise ValueError(f"Unsafe path in {context}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"Unsafe path in {context}: {value!r}")
    return path


def verify_entries(entries: list[dict[str, Any]], *, project_root: Path = PROJECT_ROOT) -> list[str]:
    """Return human-readable problems for selected manifest entries."""
    problems: list[str] = []
    for entry in entries:
        target = _confined_target(project_root, entry["target"])
        expected = {target.name: entry["sha256"]} if entry["type"] == "file" else entry["files"]
        base = target.parent if entry["type"] == "file" else target
        for relative, expected_hash in sorted(expected.items()):
            path = base / Path(*_safe_relative_path(relative, context=entry["id"]).parts)
            label = f"{entry['id']} ({path.relative_to(project_root)})"
            if path.is_symlink() or not path.is_file():
                problems.append(f"Missing: {label}")
                continue
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                problems.append(f"SHA256 mismatch: {label} expected {expected_hash}, got {actual_hash}")
    return problems


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Vocal2Midi-model-manager/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        final_url = urllib.parse.urlparse(response.geturl())
        if final_url.scheme != "https":
            raise ValueError("Download redirected away from https; refusing response")
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _archive_member_path(name: str, *, strip_prefix: str, entry_id: str) -> str | None:
    path = _safe_relative_path(name.rstrip("/"), context=f"archive entry in {entry_id}")
    parts = path.parts
    prefix = tuple(PurePosixPath(strip_prefix).parts) if strip_prefix else ()
    if prefix:
        if parts[:len(prefix)] != prefix:
            return None
        parts = parts[len(prefix):]
        if not parts:
            return None
    relative = PurePosixPath(*parts).as_posix()
    if Path(relative).suffix.lower() in BLOCKED_SUFFIXES:
        return None
    return relative


def _check_regular_zip_member(info: zipfile.ZipInfo, entry_id: str) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Unsafe non-regular archive entry in {entry_id}: {info.filename}")
    if file_type == stat.S_IFREG and mode & 0o111:
        raise ValueError(f"Executable archive entry is not permitted in {entry_id}: {info.filename}")
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted archive entry is not supported: {info.filename}")


def _extract_verified_zip(archive: Path, entry: dict[str, Any], staging: Path) -> None:
    expected: dict[str, str] = entry["files"]
    found: set[str] = set()
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            _check_regular_zip_member(info, entry["id"])
            raw = info.filename
            if info.is_dir():
                _safe_relative_path(raw.rstrip("/"), context=f"archive entry in {entry['id']}")
                continue
            relative = _archive_member_path(raw, strip_prefix=entry.get("strip_prefix", ""), entry_id=entry["id"])
            if relative is None:
                continue
            if relative in seen:
                raise ValueError(f"Duplicate archive entry in {entry['id']}: {relative}")
            seen.add(relative)
            if relative not in expected:
                continue
            destination = staging / Path(*_safe_relative_path(relative, context=entry["id"]).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            with destination.open("rb") as extracted:
                if extracted.read(8).startswith(EXECUTABLE_MAGICS):
                    destination.unlink()
                    raise ValueError(f"Executable content is not permitted: {entry['id']}:{relative}")
            actual = sha256_file(destination)
            if actual != expected[relative]:
                raise ValueError(f"Extracted SHA256 mismatch for {entry['id']}:{relative}: expected {expected[relative]}, got {actual}")
            found.add(relative)
    missing = sorted(set(expected) - found)
    if missing:
        raise ValueError(f"Archive {entry['id']} is missing expected files: {', '.join(missing)}")


def _install_one(entry: dict[str, Any], *, project_root: Path, force: bool) -> None:
    target = _confined_target(project_root, entry["target"])
    if target.exists() and not force:
        raise FileExistsError(f"Destination already exists (use --force to replace): {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2m-model-", dir=target.parent) as temp_name:
        temp = Path(temp_name)
        download = temp / "download"
        _download(entry["url"], download)
        actual = sha256_file(download)
        if actual != entry["sha256"]:
            raise ValueError(f"Archive SHA256 mismatch for {entry['id']}: expected {entry['sha256']}, got {actual}")
        if entry["type"] == "zip":
            prepared = temp / "prepared"
            prepared.mkdir()
            _extract_verified_zip(download, entry, prepared)
        else:
            with download.open("rb") as raw_file:
                magic = raw_file.read(8)
            if magic.startswith(EXECUTABLE_MAGICS):
                raise ValueError(f"Executable content is not permitted for {entry['id']}")
            prepared = temp / "prepared-file"
            shutil.copyfile(download, prepared)

        backup: Path | None = None
        if target.exists():
            backup = temp / "previous"
            os.replace(target, backup)
        try:
            os.replace(prepared, target)
        except BaseException:
            if backup is not None and backup.exists():
                os.replace(backup, target)
            raise


def _confined_target(project_root: Path, relative_target: str) -> Path:
    root = project_root.resolve()
    target = (root / Path(*_safe_relative_path(relative_target, context="install target").parts)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Model target resolves outside the repository: {relative_target}")
    return target


def _select(entries: list[dict[str, Any]], names: list[str] | None) -> list[dict[str, Any]]:
    if not names:
        return entries
    wanted = set(names)
    known = {entry["id"] for entry in entries}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown model id(s): {', '.join(unknown)}")
    return [entry for entry in entries if entry["id"] in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify manifest-pinned model assets.")
    parser.add_argument("--check", action="store_true", help="Verify installed models offline; do not download.")
    parser.add_argument("--models", nargs="*", help="Manifest model ids to install or check (default: all).")
    parser.add_argument("--force", action="store_true", help="Replace existing destinations after successful verification.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to the model manifest.")
    args = parser.parse_args(argv)
    try:
        entries = _select(load_manifest(args.manifest), args.models)
        if args.check:
            problems = verify_entries(entries, project_root=PROJECT_ROOT)
            if problems:
                print("Model verification failed:", file=sys.stderr)
                for problem in problems:
                    print(f"- {problem}", file=sys.stderr)
                return 1
            print(f"Verified {len(entries)} model asset(s).")
            return 0
        for entry in entries:
            _install_one(entry, project_root=PROJECT_ROOT, force=args.force)
            print(f"Installed {entry['name']} to {entry['target']}")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"Model operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
