# Vocal2Midi on macOS

<img src="icon.png" width="72" alt="Vocal2Midi" />

<p>
  <a href="https://github.com/Xiantaidu/Vocal2Midi"><img src="https://img.shields.io/badge/platform-macOS-000000.svg?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python"></a> <a href="#llamacpp-runtime"><img src="https://img.shields.io/badge/llama.cpp-b9453-orange.svg?style=flat-square" alt="llama.cpp"></a> <a href="#llamacpp-runtime"><img src="https://img.shields.io/badge/acceleration-metal%20%7C%20cpu-success.svg?style=flat-square" alt="Acceleration"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-yellow.svg?style=flat-square" alt="License"></a>
</p>

macOS setup guide for Vocal2Midi. It runs the same inference pipeline and GUI
workflow as Windows, but two components must be prepared locally: the
**llama.cpp runtime** and **ffmpeg**. Both are platform binaries and are not
shipped in the repository.

> The `ffmpeg.exe` at the repository root is the Windows build and cannot run
> on macOS.

## Highlights

- Same hybrid pipeline, model settings, and export formats as the Windows build
- Native Qt window shell launched from `run_mac.command`, or a local `.app` built with `scripts/build_macos_app.py`
- Qwen3-ASR decoder backed by a locally built `llama.cpp`, on Metal or CPU
- ONNX stages run on the CPU execution provider
- One non-invasive build script that compiles `llama.cpp` and installs it into `inference/qwen3asr_dml/bin/`

## Requirements

| Item | Requirement |
| --- | --- |
| Python | 3.10, 3.11, or 3.12 (3.12 is what the macOS build was validated on) |
| Xcode Command Line Tools | `xcode-select --install`, needed for clang when building llama.cpp |
| cmake / ninja | `pip install cmake ninja`, or `brew install cmake ninja` |
| ffmpeg | Required for audio decoding, see [ffmpeg](#ffmpeg) |
| Memory | The Qwen3-ASR f16 model is about 4 GB; 8 GB or more of available memory is recommended |

## Quick Start

### 1. Create the virtual environment

```bash
python3.12 -m venv .venv
```

On Apple Silicon with Homebrew Python, use the full interpreter path:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
```

### 2. Install dependencies

Use the macOS requirements file:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements_mac.txt
```

It is the macOS counterpart of `requirements.txt`: PySide6 is pinned to
**6.8.3** (`PySide6`, `PySide6-Essentials`, and `PySide6-Addons` must match
exactly), and it installs plain `onnxruntime` because `onnxruntime-directml`
is Windows-only. `numpy<2.0.0` is required by the current runtime.

If `pyopenjtalk` fails to build, confirm the Xcode Command Line Tools are
installed.

### 3. Build the llama.cpp runtime

See [llama.cpp Runtime](#llamacpp-runtime) below. This step is mandatory;
the decoder will not start without the libraries in
`inference/qwen3asr_dml/bin/`.

```bash
.venv/bin/python scripts/build_llama_runtime.py --backend cpu --build-dir /tmp/llama-b9453-cpu
```

### 4. Install ffmpeg

```bash
brew install ffmpeg
```

See [ffmpeg](#ffmpeg) for the alternatives when Homebrew is unavailable.

### 5. Prepare model folders

| Component | Default path |
| --- | --- |
| GAME | `models/GAME-1.0.3-medium-onnx` |
| TiFA default aligner | `models/tifa-1.0-onnx` |
| HubertFA | `models/1218_hfa_model_new_dict` |
| Qwen3-ASR | `models/Qwen3-ASR-1.7B-dml` |
| Japanese mora ASR | `models/romajiASR` |
| PinyinASR | `models/pinyinASR` |
| Japanese G2P kashi-g2p | `models/kashi-g2p-onnx` |
| RMVPE | `models/RMVPE` |

Paths can be changed in the GUI model settings page.

### 6. Launch

```bash
open run_mac.command
# or
.venv/bin/python app_fluent.py
```

`run_mac.command` prepends the Homebrew bin directories to `PATH` and exports
`V2M_FFMPEG` when it finds a Homebrew ffmpeg, which matters because apps
launched from Finder receive a deliberately small `PATH`.

## llama.cpp Runtime

The Qwen3-ASR decoder in [`inference/qwen3asr_dml/llama.py`](inference/qwen3asr_dml/llama.py)
calls the llama.cpp C API directly through ctypes, so the shared libraries must
be built locally and match the ABI the Python side expects.

### Pinned version: b9453

The `llama_model_params` structure declared in `llama.py` matches **b9453**
(commit `48b88c3`, ggml `0.13.1`) field for field:

- structure size **72 bytes**
- includes `use_mmap` / `use_direct_io` / `use_mlock`
- **no** `load_mode` / `lazy_mode`, which newer releases added

Building a newer release (b11539, for example) shifts the layout, so llama.cpp
reads a garbage `vocab_only` value. The model then loads vocabulary only,
`n_embd` becomes `0`, and transcription silently returns empty results.
Do not upgrade llama.cpp, and do not edit `llama.py` to compensate — build
b9453 instead.

### CPU or Metal?

This is the main trap on macOS. `detect_available_llama_backend()` only checks
whether `libggml.dylib` exists in `bin/`; it never probes for a real Metal
device. On a machine without Metal (Intel Mac, virtual machine, or a Ryzen
Hackintosh) it still selects Metal, and context creation fails with:

```text
RuntimeError: Context initialization failed
```

Check whether your machine actually has a Metal device:

```bash
python3 -c "import ctypes;m=ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal');m.MTLCreateSystemDefaultDevice.restype=ctypes.c_void_p;print(m.MTLCreateSystemDefaultDevice())"
```

- prints an address → Metal is available, use `--backend metal`
- prints `None` → there is no Metal device, you **must** use `--backend cpu`

### Building

The build script is non-invasive: it only clones and compiles llama.cpp and
copies the results into `inference/qwen3asr_dml/bin/`. It does not patch
llama.cpp and does not touch the project's Python sources.

```bash
# no Metal device (Intel Mac, VM, Hackintosh)
.venv/bin/python scripts/build_llama_runtime.py --backend cpu --build-dir /tmp/llama-b9453-cpu

# Apple Silicon with Metal
.venv/bin/python scripts/build_llama_runtime.py --backend metal
```

| Option | Description |
| --- | --- |
| `--tag` | llama.cpp release to build, defaults to `b9453` |
| `--backend` | `cpu` / `metal` / `vulkan` / `cuda` / `dml`, defaults to `auto` |
| `--build-dir` | cmake build directory |
| `--install-dir` | where the libraries go, defaults to `inference/qwen3asr_dml/bin` |
| `--dry-run` | print the commands without running them |
| `--skip-fetch` | reuse the existing checkout instead of cloning |
| `--no-clean` | keep libraries left over from a previous build |

Keep the default cleanup behavior. `ggml_backend_load_all()` loads every
backend module it finds next to the core libraries, so a leftover
`libggml-metal.dylib` would resurrect the Metal backend after switching to a
CPU-only build.

After installing, the script prints the detected devices. A CPU-only build
should report a single device:

```text
  [0] name=CPU desc=Apple M2 Pro
```

### Libraries produced

`inference/qwen3asr_dml/bin/` should contain these macOS names:

```text
libggml.dylib
libggml-base.dylib
libllama.dylib
libggml-cpu.dylib
libggml-metal.dylib    # only for a Metal build
```

The names are hardcoded in `llama.py` per platform (`ggml.dll` /
`ggml-base.dll` / `llama.dll` on Windows, `*.so` on Linux). All of them are
covered by `.gitignore`, so swapping builds never produces a git diff.

## ffmpeg

Audio is decoded and resampled by ffmpeg through
[`inference/io/audio_io.py`](inference/io/audio_io.py), which supports
wav / flac / mp3 / ogg / opus / m4a / aac / wma / webm / aiff. Only the ffmpeg
**executable** is needed, not the libav development headers.

Resolution order in `_find_ffmpeg()`:

1. `ffmpeg` at the repository root
2. the path in the `V2M_FFMPEG` environment variable
3. `ffmpeg` on `PATH`
4. `/opt/homebrew/bin/ffmpeg` → `/usr/local/bin/ffmpeg` → `/opt/local/bin/ffmpeg`

### Homebrew

```bash
brew install ffmpeg
```

### Building from source without Homebrew

```bash
git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg-src
cd ffmpeg-src
./configure --prefix=/usr/local --disable-shared --enable-static \
            --enable-swresample --disable-doc
make -j"$(sysctl -n hw.ncpu)"
sudo make install
```

The static binary depends only on system libraries, so installing it to
`/usr/local/bin` puts it on the resolution path above.

### Using a prebuilt binary

Drop a static ffmpeg into the repository root named `ffmpeg`, or point at it:

```bash
export V2M_FFMPEG=/path/to/ffmpeg
```

This is the most reliable option for GUI launches from Finder, where `PATH` is
too short to reach Homebrew.

## Building a macOS .app

To create a native application shell from the checkout:

```bash
.venv/bin/python scripts/build_macos_app.py --output dist/Vocal2Midi.app
# add --install to copy it to /Applications
```

The bundle keeps the source tree and models outside itself and launches the
existing virtual environment.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Context initialization failed` | No Metal device, but the build included Metal | Rebuild with `--backend cpu` and let the script clean old libraries |
| `ffmpeg not found` | ffmpeg missing, or `PATH` too short under Finder | Install ffmpeg, set `V2M_FFMPEG`, or place it at the repository root |
| Empty transcription, `n_embd = 0` | llama.cpp ABI does not match `llama.py` | Build b9453 |
| Process killed while loading the model | The f16 model is about 4 GB and memory is tight | Use a quantized gguf or free up memory |
| `cmake: command not found` | Installed into a Python environment that is not on `PATH` | `pip install cmake`; the script also probes the pip location |

A quick self-check that confirms the libraries and the model both work:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from inference.qwen3asr_dml import llama as L
L.init_llama_lib()
model, backend = L.load_model_with_backend(
    "models/Qwen3-ASR-1.7B-dml/qwen3_asr_llm.f16.gguf", backend="auto")
print("backend =", backend, "| n_embd =", L.llama_model_n_embd(model))
print("context =", L.create_context(model))
PY
```

A healthy run prints `backend=cpu` (or `metal`), `n_embd=2048`, and a
non-null context pointer.

## Related Files

- macOS requirements: [`requirements_mac.txt`](requirements_mac.txt)
- Launcher: [`run_mac.command`](run_mac.command)
- llama.cpp build script: [`scripts/build_llama_runtime.py`](scripts/build_llama_runtime.py)
- macOS app builder: [`scripts/build_macos_app.py`](scripts/build_macos_app.py)
- Main GUI entrypoint: [`app_fluent.py`](app_fluent.py)
- Architecture notes: [`docs/architecture.md`](docs/architecture.md)
- License: [`LICENSE`](LICENSE)
