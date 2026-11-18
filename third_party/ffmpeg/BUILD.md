# Building the bundled `ffmpeg.exe`

This document records the exact build recipe for the FFmpeg binary distributed
with Vocal2Midi. It exists so that the binary can be reproduced and so that the
corresponding source code can be identified precisely.

## Source

| | |
| --- | --- |
| Project | FFmpeg |
| Version | 7.1.1 |
| Tarball | <https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz> |
| SHA-256 | `733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1` |
| Patches | none |

The bundled binary was built from the unmodified upstream 7.1.1 release.

## Toolchain

The distributed binary reports the following build environment:

| | |
| --- | --- |
| Compiler | gcc 16.2.0 (Rev3, Built by MSYS2 project) |
| Target | Windows x86-64 (MinGW-w64) |
| Link mode | static (`--extra-ldflags=-static`) |
| Shell | MSYS2 MinGW x64 |

A different gcc build may produce a byte-different but functionally equivalent
binary. The license and configure options remain the same.

## Prerequisites

- MSYS2 MinGW x64 with `gcc`, `make`, `curl`, `tar`, and `xz`
- `nasm` or `yasm` (the configure line does not pass `--disable-x86asm`)

## Configure line

```sh
./configure \
  --disable-everything \
  --disable-autodetect \
  --disable-doc \
  --disable-debug \
  --disable-network \
  --disable-ffprobe \
  --disable-ffplay \
  --enable-ffmpeg \
  --disable-avdevice \
  --disable-swscale \
  --disable-postproc \
  --disable-d3d11va \
  --disable-dxva2 \
  --disable-vulkan \
  --disable-mediafoundation \
  --disable-bsfs \
  --enable-small \
  --extra-ldflags=-static \
  --enable-protocol='file,pipe' \
  --enable-demuxer='mov,mp3,ogg,flac,wav,aac,asf,matroska,aiff' \
  --enable-decoder='aac,aac_latm,mp3,mp3float,flac,vorbis,opus,alac,wmav1,wmav2,wmapro,pcm_u8,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_s16be' \
  --enable-parser='aac,aac_latm,mp3,flac,vorbis,opus' \
  --enable-muxer='wav,ipod,pcm_f32le' \
  --enable-encoder='aac,pcm_s16le,pcm_f32le' \
  --enable-filter='aresample,format,aformat,abuffer,abuffersink,buffer,buffersink'
```

This is the same line reported by `ffmpeg.exe -buildconf`. It enables only
LGPL-licensed FFmpeg components and no external libraries.

## Manual build (MSYS2)

From an **MSYS2 MinGW x64** shell:

```sh
curl -LO https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz
sha256sum ffmpeg-7.1.1.tar.xz
# must print:
# 733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1

tar -xf ffmpeg-7.1.1.tar.xz
cd ffmpeg-7.1.1
./configure \
  --disable-everything \
  --disable-autodetect \
  --disable-doc \
  --disable-debug \
  --disable-network \
  --disable-ffprobe \
  --disable-ffplay \
  --enable-ffmpeg \
  --disable-avdevice \
  --disable-swscale \
  --disable-postproc \
  --disable-d3d11va \
  --disable-dxva2 \
  --disable-vulkan \
  --disable-mediafoundation \
  --disable-bsfs \
  --enable-small \
  --extra-ldflags=-static \
  --enable-protocol='file,pipe' \
  --enable-demuxer='mov,mp3,ogg,flac,wav,aac,asf,matroska,aiff' \
  --enable-decoder='aac,aac_latm,mp3,mp3float,flac,vorbis,opus,alac,wmav1,wmav2,wmapro,pcm_u8,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_s16be' \
  --enable-parser='aac,aac_latm,mp3,flac,vorbis,opus' \
  --enable-muxer='wav,ipod,pcm_f32le' \
  --enable-encoder='aac,pcm_s16le,pcm_f32le' \
  --enable-filter='aresample,format,aformat,abuffer,abuffersink,buffer,buffersink'
make -j"$(nproc)"
```

The resulting executable is `ffmpeg.exe` in the source directory. Copy it to
the repository root.

## Automated build

`scripts/build_ffmpeg.py` performs the same steps: downloads the release
tarball, verifies the SHA-256, extracts it, runs configure and make, and copies
the resulting binary to the repository root.

```sh
python scripts/build_ffmpeg.py
```

On Windows, run it from an MSYS2 MinGW x64 shell (or make sure a MinGW-w64
`bash`, `gcc`, and `make` are on `PATH`). Use `--tarball` to reuse an already
downloaded source archive and `--jobs` to control parallelism.
