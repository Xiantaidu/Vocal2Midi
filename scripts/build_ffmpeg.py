#!/usr/bin/env python3
"""Rebuild the FFmpeg binary bundled at the repository root.

The build recipe is pinned to the upstream FFmpeg 7.1.1 release and the exact
configure line recorded in ``third_party/ffmpeg/BUILD.md``. Keeping this
script in the repository makes the LGPL corresponding-source recipe
reproducible.

Typical use::

    python scripts/build_ffmpeg.py
    python scripts/build_ffmpeg.py --tarball /path/to/ffmpeg-7.1.1.tar.xz
    python scripts/build_ffmpeg.py --output /tmp/ffmpeg

On Windows, run it from an MSYS2 MinGW x64 shell so that ``bash``, ``gcc``,
and ``make`` match the toolchain used for the distributed binary.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

FFMPEG_VERSION = "7.1.1"
FFMPEG_URL = f"https://ffmpeg.org/releases/ffmpeg-{FFMPEG_VERSION}.tar.xz"
FFMPEG_SHA256 = "733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRNAME = f"ffmpeg-{FFMPEG_VERSION}"

# Must stay in sync with third_party/ffmpeg/BUILD.md and `ffmpeg -buildconf`.
CONFIGURE_ARGS = (
    "--disable-everything",
    "--disable-autodetect",
    "--disable-doc",
    "--disable-debug",
    "--disable-network",
    "--disable-ffprobe",
    "--disable-ffplay",
    "--enable-ffmpeg",
    "--disable-avdevice",
    "--disable-swscale",
    "--disable-postproc",
    "--disable-d3d11va",
    "--disable-dxva2",
    "--disable-vulkan",
    "--disable-mediafoundation",
    "--disable-bsfs",
    "--enable-small",
    "--extra-ldflags=-static",
    "--enable-protocol=file,pipe",
    "--enable-demuxer=mov,mp3,ogg,flac,wav,aac,asf,matroska,aiff",
    "--enable-decoder=aac,aac_latm,mp3,mp3float,flac,vorbis,opus,alac,wmav1,wmav2,wmapro,pcm_u8,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_s16be",
    "--enable-parser=aac,aac_latm,mp3,flac,vorbis,opus",
    "--enable-muxer=wav,ipod,pcm_f32le",
    "--enable-encoder=aac,pcm_s16le,pcm_f32le",
    "--enable-filter=aresample,format,aformat,abuffer,abuffersink,buffer,buffersink",
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract(tarball: Path, destination: Path) -> Path:
    print(f"Extracting {tarball.name}")
    with tarfile.open(tarball, "r:xz") as archive:
        try:
            archive.extractall(destination, filter="data")
        except TypeError:  # Python < 3.12
            archive.extractall(destination)
    source_dir = destination / SOURCE_DIRNAME
    if not source_dir.is_dir():
        raise SystemExit(f"unexpected archive layout: missing {SOURCE_DIRNAME}/")
    return source_dir


def find_bash() -> str | None:
    return shutil.which("bash") or shutil.which("sh")


def configure_command() -> list[str]:
    if os.name == "nt":
        shell = find_bash()
        if not shell:
            raise SystemExit(
                "bash not found. Run this script from an MSYS2 MinGW x64 shell."
            )
        return [shell, "./configure", *CONFIGURE_ARGS]
    return ["./configure", *CONFIGURE_ARGS]


def find_make() -> str:
    for name in ("make", "gmake", "mingw32-make"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("make not found. Install make and re-run this script.")


def verify_binary(binary: Path) -> None:
    license_text = subprocess.run(
        [str(binary), "-L"], capture_output=True, text=True, check=False
    ).stdout
    if "Lesser General Public" not in license_text:
        raise SystemExit(
            f"{binary.name} does not report the LGPL license banner; refusing to install it."
        )

    buildconf = subprocess.run(
        [str(binary), "-buildconf"], capture_output=True, text=True, check=False
    ).stdout
    for forbidden in ("--enable-gpl", "--enable-nonfree", "--enable-version3"):
        if forbidden in buildconf:
            raise SystemExit(
                f"{binary.name} was configured with {forbidden}; the bundled build must stay LGPL."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the FFmpeg binary bundled with Vocal2Midi."
    )
    parser.add_argument(
        "--tarball",
        type=Path,
        help="Existing ffmpeg-7.1.1.tar.xz to use instead of downloading it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"),
        help="Where to place the built binary.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel make jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="vocal2midi-ffmpeg-") as tmpdir:
        workdir = Path(tmpdir)
        tarball = args.tarball
        if tarball is None:
            tarball = workdir / f"ffmpeg-{FFMPEG_VERSION}.tar.xz"
            download(FFMPEG_URL, tarball)
        elif not tarball.is_file():
            raise SystemExit(f"tarball not found: {tarball}")

        actual = sha256_file(tarball)
        if actual != FFMPEG_SHA256:
            raise SystemExit(
                f"SHA-256 mismatch for {tarball}:\n"
                f"  expected {FFMPEG_SHA256}\n"
                f"  actual   {actual}"
            )

        source_dir = extract(tarball, workdir)

        print("Configuring FFmpeg")
        run(configure_command(), cwd=source_dir)

        print("Building FFmpeg")
        run([find_make(), f"-j{args.jobs}"], cwd=source_dir)

        built = source_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not built.is_file():
            raise SystemExit(f"build finished but {built.name} was not produced")

        verify_binary(built)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, args.output)
        print(f"Installed {args.output}")


if __name__ == "__main__":
    main()
