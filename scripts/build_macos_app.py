#!/usr/bin/env python3
"""Build a lightweight macOS application shell for a Vocal2Midi checkout.

The source tree and model files stay outside the bundle.  The generated app
records the checkout it was built from, sets up Homebrew's common paths, and
launches the existing Python environment without copying several gigabytes of
models into ``/Applications``.
"""

from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "Vocal2Midi"
BUNDLE_ID = "com.xiantaidu.vocal2midi.mac"


def _launcher_text() -> str:
    return r'''#!/bin/zsh
set -e
BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cat "$BUNDLE_DIR/Resources/project-path.txt" 2>/dev/null || true)"
LOG_DIR="$HOME/Library/Logs/Vocal2Midi"
LOG_FILE="$LOG_DIR/launch.log"
mkdir -p "$LOG_DIR"
if [[ -z "$PROJECT" || ! -d "$PROJECT" ]]; then
  /usr/bin/osascript -e 'display dialog "Vocal2Midi 的项目目录已移动，请重新构建应用。" with title "Vocal2Midi" buttons {"好"} default button "好"'
  exit 1
fi
if [[ ! -x "$PROJECT/.venv/bin/python" ]]; then
  /usr/bin/osascript -e 'display dialog "Vocal2Midi 的 .venv 不存在，请先安装 requirements.txt。" with title "Vocal2Midi" buttons {"好"} default button "好"'
  exit 1
fi
cd "$PROJECT"
export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:$PATH"
if [[ -x /opt/homebrew/bin/ffmpeg ]]; then
  export V2M_FFMPEG=/opt/homebrew/bin/ffmpeg
elif [[ -x /usr/local/bin/ffmpeg ]]; then
  export V2M_FFMPEG=/usr/local/bin/ffmpeg
fi
export V2M_PORTABLE_ROOT="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROJECT/.venv/bin/python" "$PROJECT/app_fluent.py" "$@" >>"$LOG_FILE" 2>&1
'''


def _write_icns(project_root: Path, resources: Path) -> str | None:
    """Create an icns file when the native macOS tools are available."""
    icon = project_root / "icon.png"
    sips = shutil.which("sips")
    iconutil = shutil.which("iconutil")
    if not icon.is_file() or not sips or not iconutil:
        return None
    iconset = resources / f"{APP_NAME}.iconset"
    iconset.mkdir()
    sizes = (
        (16, "16x16"), (32, "16x16@2x"), (32, "32x32"), (64, "32x32@2x"),
        (128, "128x128"), (256, "128x128@2x"), (256, "256x256"),
        (512, "256x256@2x"), (512, "512x512"), (1024, "512x512@2x"),
    )
    try:
        for size, name in sizes:
            subprocess.run(
                [sips, "-z", str(size), str(size), str(icon), "--out", str(iconset / f"icon_{name}.png")],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        icns = resources / f"{APP_NAME}.icns"
        subprocess.run([iconutil, "-c", "icns", str(iconset), "-o", str(icns)], check=True)
        return icns.name
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


def build_bundle(project_root: Path, output: Path, *, sign: bool = True) -> Path:
    project_root = project_root.resolve()
    output = output.expanduser().resolve()
    if not (project_root / "app_fluent.py").is_file():
        raise FileNotFoundError(f"Vocal2Midi checkout not found: {project_root}")
    if output.exists():
        shutil.rmtree(output)

    macos = output / "Contents" / "MacOS"
    resources = output / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    (resources / "project-path.txt").write_text(f"{project_root}\n", encoding="utf-8")
    launcher = macos / APP_NAME
    launcher.write_text(_launcher_text(), encoding="utf-8")
    launcher.chmod(0o755)

    icon_name = _write_icns(project_root, resources)
    info = {
        "CFBundleDevelopmentRegion": "zh_CN",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "2.0.0",
        "CFBundleVersion": "2.0.0",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "Audio File",
            "CFBundleTypeRole": "Editor",
            "LSItemContentTypes": ["public.audio", "public.mp3", "org.xiph.flac"],
        }],
    }
    if icon_name:
        info["CFBundleIconFile"] = icon_name
    with (output / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle, sort_keys=False)

    codesign = shutil.which("codesign")
    if sign and codesign:
        subprocess.run([codesign, "--force", "--deep", "--sign", "-", str(output)], check=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("dist") / f"{APP_NAME}.app")
    parser.add_argument("--install", action="store_true", help="Copy the result to /Applications.")
    parser.add_argument("--no-sign", action="store_true", help="Skip ad-hoc codesigning.")
    return parser.parse_args()


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("build_macos_app.py must run on macOS")
    args = parse_args()
    output = build_bundle(args.project_root, args.output, sign=not args.no_sign)
    if args.install:
        installed = Path("/Applications") / output.name
        if installed.exists():
            shutil.rmtree(installed)
        shutil.copytree(output, installed, symlinks=True)
        print(f"Installed {installed}")
    else:
        print(f"Built {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
