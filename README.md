# Vocal2Midi

Vocal2Midi is a Windows-first desktop tool and inference pipeline for turning vocal audio into lyric-aligned MIDI, USTX, and supporting editing artifacts.

The current runtime is **ONNX-first**:

- `llama.cpp` is used for the Qwen decoder and runs on CPU.
- ONNX models default to **DirectML** and fall back to **CPU** when DirectML is unavailable.
- The main user-facing entrypoint is the Fluent GUI in [`app_fluent.py`](app_fluent.py).

## Highlights

- End-to-end vocal-to-MIDI workflow in one project
- Chinese and Japanese lyric handling
- GUI workflow for interactive use
- Batch slice + ASR CLI for folder processing
- Portable-folder packaging flow for Windows distribution
- ONNX-based inference stack for ASR, alignment, note extraction, and RMVPE

## What Vocal2Midi Does

At a high level, the hybrid pipeline looks like this:

```text
audio
  -> optional RMVPE pitch curve
  -> slicing
  -> ASR
  -> lyric matching / .lab generation
  -> HubertFA forced alignment
  -> GAME note extraction
  -> quantization
  -> export
```

There is also a no-lyrics path:

```text
audio
  -> optional RMVPE pitch curve
  -> slicing
  -> GAME pitch-only extraction
  -> export
```

## Runtime Stack

| Component | Current backend | Location |
| --- | --- | --- |
| Qwen3-ASR | ONNX Runtime + `llama.cpp` | `inference/qwen3asr_dml/` |
| Japanese mora / romaji ASR | ONNX Runtime | `inference/romaji_asr/` |
| HubertFA | ONNX Runtime | `inference/HubertFA/` |
| GAME | ONNX Runtime | `inference/game/` |
| RMVPE | ONNX Runtime | `inference/API/rmvpe_api.py` |
| Device normalization | DirectML / CPU helpers | `inference/device_utils.py` |

Some public function names still contain `_torch` for compatibility, but the active backend is ONNX-based.

## Repository Layout

```text
application/   application-layer orchestration and config objects
docs/          architecture notes and supporting docs
experiments/   local model directories
gui/           PyQt5 + qfluentwidgets desktop UI
inference/     ASR, alignment, pitch extraction, slicing, quantization, export
scripts/       batch CLI and portable build helpers
tests/         automated tests
```

## Quick Start

### 1. Install dependencies

Use your preferred Python environment, then install:

```bash
pip install -r requirements.txt
```

The main runtime dependencies are:

- `onnxruntime-directml`
- `PyQt5`
- `PyQt-Fluent-Widgets`
- `librosa`
- `soundfile`
- `mido`
- `pyopenjtalk`

An `environment.yml` file is also included as a reference environment snapshot.

### 2. Prepare model folders

By default, the GUI expects models in these locations:

| Component | Default path |
| --- | --- |
| GAME | `experiments/GAME-1.0.3-medium-onnx` |
| HubertFA | `experiments/1218_hfa_model_new_dict` |
| Qwen3-ASR | `experiments/Qwen3-ASR-1.7B-dml` |
| Japanese mora ASR | `experiments/romajiASR` |
| RMVPE | `experiments/RMVPE/rmvpe.onnx` |

You can change these paths in the GUI settings panel.

### 3. Launch the GUI

For a normal developer environment:

```bash
python app_fluent.py
```

## GUI Workflow

The GUI is the main way to use Vocal2Midi interactively. It lets you:

- choose model paths
- pick the runtime device
- set slicing mode and slice length bounds
- choose language and lyric output mode
- provide optional reference lyrics
- export MIDI, USTX, text, CSV, chunk audio, and alignment artifacts

The application-layer job entrypoint is `run_auto_lyric_job()` in [`application/pipeline.py`](application/pipeline.py), which dispatches into the hybrid inference pipeline in [`inference/pipeline/auto_lyric_hybrid.py`](inference/pipeline/auto_lyric_hybrid.py).

## Language Behavior

### Chinese

- Qwen3-ASR provides text transcription.
- The lyric matcher and G2P path prepare `.lab` content for HubertFA.
- Lyrics can be exported in Hanzi or pinyin-oriented forms depending on mode.

### Japanese

In the main hybrid lyric pipeline:

- `romaji` and `kana` lyric modes use the dedicated **mora ASR** path
- if the output mode is `romaji`, the pipeline uses mora ASR output directly
- if the output mode is `kana`, the pipeline converts matched mora output to kana for display
- if reference lyrics are provided, the reference text is processed through `pyopenjtalk`, converted to kana mora tokens, then converted again to romaji mora tokens for matching

This keeps Japanese lyric matching consistent with the mora-based ASR path instead of routing through the old phoneme-ASR forced-alignment branch.

## Runtime Device Rules

Visible device options in the current UI are:

- `dml`
- `cpu`

Notes:

- `dml` is the default ONNX device
- if DirectML is unavailable, ONNX Runtime falls back to CPU
- legacy `cuda` values are still accepted by some public interfaces, but they are normalized to `dml`
- `llama.cpp` remains CPU-based in the current design

## Slicing

The user-facing slice duration settings currently support:

- minimum slice length: `0` to `60` seconds
- maximum slice length: `0` to `60` seconds

Current defaults:

- minimum: `8.0` seconds
- maximum: `22.0` seconds

Validation rules:

- `slice_max_sec` must be greater than `0`
- `slice_min_sec` must be less than or equal to `slice_max_sec`

## Batch Slice + ASR CLI

For folder-based batch ASR processing:

```bash
python scripts/slice_asr_cli.py <input_dir> <output_dir> \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --device dml \
  --language zh
```

This CLI is focused on:

- scanning input audio files
- slicing audio or bypassing slicing
- running local Qwen3-ASR
- saving chunk audio and `.lab` outputs
- optionally saving JSON timing / ASR metadata

Supported input extensions currently include:

- `.wav`
- `.m4a`
- `.mp3`

Useful options:

```text
--no-slice              bypass slicing and send the whole file to ASR
--asr-batch-size        ASR batch size
--file-batch-size       number of audio files per batch
--rmvpe-model           enable RMVPE-assisted smart slicing
--rmvpe-batch-size      RMVPE batch size
--keep-model            keep the ASR runtime alive across the batch
--keep-rmvpe            keep the RMVPE runtime alive across the batch
--save-json             save slice timing and ASR outputs as JSON
--no-recursive          scan only the top level
--no-skip-existing      force reprocessing of existing outputs
```

Japanese whole-file example:

```bash
python scripts/slice_asr_cli.py input output \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --device dml \
  --language ja \
  --no-slice
```

## Portable Windows Distribution

Vocal2Midi supports a **portable-folder** distribution flow. This is useful when you want to ship a self-contained runtime folder instead of building an `.exe`.

Build a portable folder with:

```bash
python scripts/build_portable.py --clean
```

If the current Python runtime is a Conda environment and you want a more relocatable bundle:

```bash
python scripts/build_portable.py --runtime-mode conda-pack --clean
```

The generated portable folder is written under `dist/` and typically includes:

- a bundled `python/` runtime
- bundled model assets under `experiments/`
- `Run Vocal2Midi.bat`
- `Run Slice ASR CLI.bat`
- `Open Portable Shell.bat`

In portable mode:

- GUI settings are stored in `settings/vocal2midi.ini`
- default output files go to `outputs/`

## Windows Setup Scripts

The repository also includes Windows helper scripts:

- [`安装环境.bat`](安装环境.bat)
- [`启动GUI.bat`](启动GUI.bat)

These are useful for a smaller distribution model where the user downloads or initializes the runtime on first setup rather than receiving a fully bundled `python/` folder.

## Export Formats

Depending on the selected workflow, Vocal2Midi can export:

- `.mid`
- `.ustx`
- `.txt`
- `.csv`
- `TextGrid`
- chunk `.wav` files
- `.lab`
- ASR matching logs

## Project Notes

- The repository has already migrated away from the earlier Torch-heavy runtime design for the main inference path.
- Some historical function names remain for compatibility.
- Model assets are expected to exist locally under `experiments/` or another user-provided path.
- The codebase is still being cleaned up in places, so you may still see a few legacy names or UI strings from earlier iterations.

## License

The overall Vocal2Midi repository is distributed under the **Apache License 2.0**. See [LICENSE](LICENSE).

Third-party components, vendored code, model assets, dictionaries, and other embedded materials may also carry their own original licenses, notices, or attribution requirements. Those original notices remain applicable to the corresponding materials. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) and any embedded license files for details.

## Development and Testing

The repo includes a focused automated test suite under `tests/`.

Examples:

```bash
python -m pytest tests/test_auto_lyric_hybrid_pipeline.py
python -m pytest tests/test_asr_api.py tests/test_game_api.py tests/test_rmvpe_api.py
python -m pytest tests/test_device_selection.py tests/test_hubertfa_decoder.py
```

For architecture details, see [docs/architecture.md](docs/architecture.md).

## Related Files

- Main GUI entrypoint: [`app_fluent.py`](app_fluent.py)
- Architecture notes: [docs/architecture.md](docs/architecture.md)
- Third-party credits: [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
- License: [LICENSE](LICENSE)
