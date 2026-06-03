from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist" / "Vocal2Midi-portable"
DEFAULT_RUNTIME_DIRNAME = "python"
RUNTIME_MARKER = ".runtime_ready"

APP_DIRS = [
    "application",
    "gui",
    "inference",
    "scripts",
]

APP_FILES = [
    "ACKNOWLEDGEMENTS.md",
    "LICENSE",
    "README.md",
    "app_fluent.py",
]

MODEL_SPECS: dict[str, tuple[str, str]] = {
    "game": ("experiments/GAME-1.0.3-medium-onnx", "dir"),
    "hfa": ("experiments/1218_hfa_model_new_dict", "dir"),
    "qwen": ("experiments/Qwen3-ASR-1.7B-dml", "dir"),
    "romaji": ("experiments/romajiASR", "dir"),
    "rmvpe": ("experiments/RMVPE/rmvpe.onnx", "file"),
}

RUNTIME_EXCLUDE_NAMES = {
    "__pycache__",
    "conda-meta",
    "info",
    "pkgs",
    "tests",
    "test",
}
RUNTIME_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

APP_EXCLUDE_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
}
APP_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class CopyPlanItem:
    source: Path
    target: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable Vocal2Midi folder with a bundled Python runtime."
    )
    parser.add_argument(
        "--python-dir",
        type=Path,
        default=Path(sys.prefix),
        help="Python runtime source directory. A portable CPython, venv, or conda env/conda-pack env all work.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Portable output directory.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=["auto", "copy", "conda-pack"],
        default="auto",
        help="How to vendor the Python runtime.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
        help="Which model assets to include.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before building.",
    )
    return parser.parse_args()


def has_conda_pack() -> bool:
    return importlib.util.find_spec("conda_pack") is not None


def looks_like_conda_env(path: Path) -> bool:
    return (path / "conda-meta").is_dir()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def should_skip(path: Path, exclude_names: set[str], exclude_suffixes: set[str]) -> bool:
    return path.name in exclude_names or path.suffix.lower() in exclude_suffixes


def copy_tree_filtered(src: Path, dst: Path, *, exclude_names: set[str], exclude_suffixes: set[str]) -> None:
    if src.is_file():
        if should_skip(src, exclude_names, exclude_suffixes):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    for root, dirnames, filenames in os.walk(src):
        root_path = Path(root)
        dirnames[:] = [d for d in dirnames if d not in exclude_names]
        rel_root = root_path.relative_to(src)
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source_file = root_path / filename
            if should_skip(source_file, exclude_names, exclude_suffixes):
                continue
            shutil.copy2(source_file, target_root / filename)


def detect_runtime_mode(runtime_src: Path, requested_mode: str) -> str:
    if requested_mode == "auto":
        if looks_like_conda_env(runtime_src) and has_conda_pack():
            return "conda-pack"
        return "copy"
    return requested_mode


def bundle_runtime(runtime_src: Path, runtime_dst: Path, runtime_mode: str) -> str:
    ensure_exists(runtime_src, "Python runtime source")
    chosen_mode = detect_runtime_mode(runtime_src, runtime_mode)
    if runtime_dst.exists():
        shutil.rmtree(runtime_dst)

    if chosen_mode == "conda-pack":
        if not looks_like_conda_env(runtime_src):
            raise ValueError("--runtime-mode conda-pack requires a conda environment source.")
        if not has_conda_pack():
            raise RuntimeError("conda-pack is not installed in the current Python environment.")
        runtime_dst.mkdir(parents=True, exist_ok=True)
        _bundle_runtime_with_conda_pack(runtime_src, runtime_dst)
    else:
        if looks_like_conda_env(runtime_src):
            print(
                "Warning: the selected runtime looks like a conda environment but is being copied directly. "
                "This usually works best on the build machine; for a more relocatable bundle, install conda-pack "
                "and rebuild with --runtime-mode conda-pack."
            )
        copy_tree_filtered(
            runtime_src,
            runtime_dst,
            exclude_names=RUNTIME_EXCLUDE_NAMES,
            exclude_suffixes=RUNTIME_EXCLUDE_SUFFIXES,
        )

    return chosen_mode


def _bundle_runtime_with_conda_pack(runtime_src: Path, runtime_dst: Path) -> None:
    import conda_pack

    with tempfile.TemporaryDirectory(prefix="vocal2midi-portable-") as tmpdir:
        archive_path = Path(tmpdir) / "runtime.zip"
        conda_pack.pack(
            prefix=str(runtime_src),
            output=str(archive_path),
            format="zip",
            force=True,
            verbose=False,
        )
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(runtime_dst)


def get_model_copy_plan(selected_models: list[str]) -> list[CopyPlanItem]:
    plan: list[CopyPlanItem] = []
    for model_name in selected_models:
        rel_path, kind = MODEL_SPECS[model_name]
        source = PROJECT_ROOT / rel_path
        ensure_exists(source, f"Model asset '{model_name}'")
        plan.append(CopyPlanItem(source=source, target=PROJECT_ROOT / rel_path, kind=kind))
    return plan


def copy_app_files(output_dir: Path) -> None:
    for rel_dir in APP_DIRS:
        source = PROJECT_ROOT / rel_dir
        copy_tree_filtered(
            source,
            output_dir / rel_dir,
            exclude_names=APP_EXCLUDE_NAMES,
            exclude_suffixes=APP_EXCLUDE_SUFFIXES,
        )
    for rel_file in APP_FILES:
        source = PROJECT_ROOT / rel_file
        ensure_exists(source, f"Application file '{rel_file}'")
        shutil.copy2(source, output_dir / rel_file)


def copy_models(output_dir: Path, selected_models: list[str]) -> None:
    for item in get_model_copy_plan(selected_models):
        rel_target = item.target.relative_to(PROJECT_ROOT)
        final_target = output_dir / rel_target
        if item.kind == "dir":
            copy_tree_filtered(
                item.source,
                final_target,
                exclude_names=APP_EXCLUDE_NAMES,
                exclude_suffixes=APP_EXCLUDE_SUFFIXES,
            )
        else:
            final_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, final_target)


def write_launcher(output_dir: Path, *, runtime_mode_used: str) -> None:
    run_gui = output_dir / "Run Vocal2Midi.bat"
    run_cli = output_dir / "Run Slice ASR CLI.bat"
    open_shell = output_dir / "Open Portable Shell.bat"

    launch_prefix = f"""@echo off
setlocal
set "ROOT=%~dp0"
set "V2M_PORTABLE_ROOT=%ROOT:~0,-1%"
set "PYTHONHOME=%ROOT%{DEFAULT_RUNTIME_DIRNAME}"
set "PYTHONPATH=%ROOT%"
set "PYTHONNOUSERSITE=1"
set "PATH=%ROOT%{DEFAULT_RUNTIME_DIRNAME};%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Scripts;%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Library\\bin;%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\DLLs;%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Lib\\site-packages\\PyQt5\\Qt5\\bin;%ROOT%inference\\qwen3asr_dml\\bin;%PATH%"
set "QT_PLUGIN_PATH=%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Lib\\site-packages\\PyQt5\\Qt5\\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Lib\\site-packages\\PyQt5\\Qt5\\plugins\\platforms"
cd /d "%ROOT%"
"""
    if runtime_mode_used == "conda-pack":
        launch_prefix += f"""
if not exist "%ROOT%{RUNTIME_MARKER}" (
  if exist "%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Scripts\\conda-unpack.exe" (
    echo Preparing bundled runtime for this machine...
    call "%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\Scripts\\conda-unpack.exe"
    if errorlevel 1 exit /b 1
  )
  type nul > "%ROOT%{RUNTIME_MARKER}"
)
"""

    run_gui.write_text(
        launch_prefix + f"""
"%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\python.exe" app_fluent.py
""",
        encoding="utf-8",
        newline="\r\n",
    )
    run_cli.write_text(
        launch_prefix + f"""
"%ROOT%{DEFAULT_RUNTIME_DIRNAME}\\python.exe" scripts\\slice_asr_cli.py %*
""",
        encoding="utf-8",
        newline="\r\n",
    )
    open_shell.write_text(
        launch_prefix + """
cmd /k
""",
        encoding="utf-8",
        newline="\r\n",
    )


def write_portable_notes(output_dir: Path, *, runtime_mode_used: str, selected_models: list[str]) -> None:
    note_path = output_dir / "PORTABLE_README.txt"
    runtime_note = (
        "The Python runtime was bundled with conda-pack. The first launch will run conda-unpack once."
        if runtime_mode_used == "conda-pack"
        else "The Python runtime was copied directly from the source directory."
    )
    note_path.write_text(
        "\n".join(
            [
                "Vocal2Midi portable package",
                "",
                "1. Keep this folder structure unchanged after extraction.",
                "2. Start the GUI with 'Run Vocal2Midi.bat'.",
                "3. Start the batch CLI with 'Run Slice ASR CLI.bat'.",
                "4. Open a shell with the bundled runtime via 'Open Portable Shell.bat'.",
                "",
                runtime_note,
                "",
                "Bundled models:",
                *(f"- {name}" for name in selected_models),
                "",
                "Portable mode writes settings to settings/vocal2midi.ini and defaults outputs to outputs/.",
            ]
        ),
        encoding="utf-8",
        newline="\r\n",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    runtime_src = args.python_dir.resolve()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "settings").mkdir(exist_ok=True)
    (output_dir / "outputs").mkdir(exist_ok=True)

    copy_app_files(output_dir)
    copy_models(output_dir, args.models)
    runtime_mode_used = bundle_runtime(
        runtime_src,
        output_dir / DEFAULT_RUNTIME_DIRNAME,
        args.runtime_mode,
    )
    write_launcher(output_dir, runtime_mode_used=runtime_mode_used)
    write_portable_notes(output_dir, runtime_mode_used=runtime_mode_used, selected_models=args.models)

    print(f"Portable package created at: {output_dir}")
    print(f"Runtime mode: {runtime_mode_used}")


if __name__ == "__main__":
    main()
