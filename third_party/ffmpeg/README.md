# FFmpeg (bundled Windows binary)

Vocal2Midi ships one prebuilt FFmpeg command-line binary at the repository
root:

| | |
| --- | --- |
| File | `ffmpeg.exe` |
| Version | FFmpeg 7.1.1 |
| License | **LGPL-2.1-or-later** |
| Build | static, minimal, no GPL or non-free components |
| Upstream | <https://ffmpeg.org/> |

The binary is invoked as a separate process by `inference/io/audio_io.py`
(`subprocess.Popen`). Vocal2Midi does not link against FFmpeg libraries, so
the application code and the FFmpeg binary are separate works.

## Why this build is LGPL, not GPL

`ffmpeg.exe -buildconf` reports the configure line documented in
[`BUILD.md`](BUILD.md). It contains:

- no `--enable-gpl` (so no GPL-only parts such as `libpostproc` or the GPL
  filters are compiled in),
- no `--enable-nonfree`,
- no `--enable-version3`,
- `--disable-autodetect`, so external libraries such as libx264/libx265 are
  not picked up.

Every enabled component (AAC/MP3/FLAC/Vorbis/Opus/ALAC/WMA/PCM decoders,
WAV/iPod muxers, the native AAC/PCM encoders, and the resample/format
filters) is part of FFmpeg's LGPL-licensed core.

`ffmpeg.exe -L` prints the LGPL-2.1-or-later notice.

## Files in this directory

| File | Purpose |
| --- | --- |
| `COPYING.LGPLv2.1` | LGPL 2.1 license text, copied verbatim from the FFmpeg 7.1.1 release tarball |
| `UPSTREAM_LICENSE.md` | FFmpeg's own `LICENSE.md` from the same release |
| `COPYING.GPLv2`, `COPYING.GPLv3`, `COPYING.LGPLv3` | Upstream license texts referenced by `UPSTREAM_LICENSE.md`; they do not apply to this build |
| `BUILD.md` | Exact source version, toolchain, configure line, and build steps |
| `SOURCE_OFFER.md` | Written offer for the corresponding source code |

## Upstream source

- Release tarball: <https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz>
- SHA-256: `733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1`

No patches or local modifications are applied; the bundled binary is built
from this unmodified upstream release.

FFmpeg is Copyright (c) 2000-2025 the FFmpeg developers.

## Verifying the bundled binary

```bat
ffmpeg.exe -L
ffmpeg.exe -buildconf
```

The first command must report the GNU Lesser General Public License, version
2.1 or later. The second must not show `--enable-gpl`, `--enable-nonfree`, or
`--enable-version3`.

To rebuild the binary from source, see [`BUILD.md`](BUILD.md) and
`scripts/build_ffmpeg.py`.
