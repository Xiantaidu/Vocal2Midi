# Vocal2Midi

Vocal2Midi is a desktop and pipeline toolkit for turning vocal audio into lyric-aligned MIDI and related editing artifacts.

The current project runtime is centered on ONNX inference:

- `llama.cpp` stays on CPU.
- ONNX backends default to DirectML and fall back to CPU when DirectML is unavailable.
- The main user entrypoint is the Fluent GUI in `app_fluent.py`.

## What the project does

Vocal2Midi combines several stages into one workflow:

1. slice or load vocal audio
2. transcribe lyrics with Qwen3-ASR or the Japanese romaji ASR path
3. align lyrics with HubertFA
4. extract note timing and pitch with GAME
5. optionally extract frame-level pitch with RMVPE
6. export MIDI, USTX, text, CSV, and alignment artifacts

## Current runtime layout

The active inference stack in this repository is:

- Qwen3-ASR: `inference/qwen3asr_dml/`
- Japanese romaji ASR: `inference/romaji_asr/`
- HubertFA ONNX: `inference/HubertFA/`
- GAME ONNX: `inference/game/`
- RMVPE ONNX: `inference/API/rmvpe_api.py`
- runtime device helpers: `inference/device_utils.py`

Compatibility function names such as `extract_pitches_and_align_torch()` are still present in a few places, but their current implementation is ONNX-based.

## Repository overview

```text
application/   application-layer orchestration and config objects
docs/          project documentation
experiments/   local model directories
gui/           PyQt5 + qfluentwidgets desktop UI
inference/     ASR, alignment, pitch extraction, export, and runtime code
scripts/       command-line utilities
tests/         automated tests
```

## Default model locations

The GUI currently defaults to these local paths:

- GAME: `experiments/GAME-1.0.3-medium-onnx`
- HubertFA: `experiments/1218_hfa_model_new_dict`
- Qwen3-ASR: `experiments/Qwen3-ASR-1.7B-dml`
- Romaji ASR: `experiments/romajiASR`
- RMVPE: `experiments/RMVPE/rmvpe.onnx`

## Running the GUI

The simplest entrypoint is:

```bat
start.bat
```

That launcher currently runs:

```bat
conda activate vocal2midi_torch
python app_fluent.py
```

You can also start the GUI directly:

```bash
python app_fluent.py
```

## Batch slice + ASR CLI

For folder-based batch processing, use:

```bash
python scripts/slice_asr_cli.py <input_dir> <output_dir> \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --device dml \
  --language zh
```

Useful options:

- `--no-slice`: send the whole file to ASR as a single chunk
- `--file-batch-size`: process multiple source files per batch
- `--asr-batch-size`: control ASR batch size
- `--rmvpe-model`: enable RMVPE-assisted smart slicing when used with the matching slicing mode
- `--keep-model`: keep the ASR runtime alive across the whole batch
- `--keep-rmvpe`: keep the RMVPE runtime alive across the whole batch
- `--save-json`: save per-chunk timestamps and ASR text

Supported input extensions in the CLI are currently `.wav`, `.m4a`, and `.mp3`.

## Outputs

Depending on the selected pipeline mode, Vocal2Midi can export:

- `.mid`
- `.ustx`
- `.txt`
- `.csv`
- `TextGrid`
- chunk `.wav` files
- `.lab` alignment text
- ASR match logs

## Dependencies

This repository includes both `requirements.txt` and `environment.yml` as dependency references.

The active inference design assumes:

- ONNX Runtime for DirectML and CPU execution
- `llama.cpp` for the Qwen decoder on CPU
- PyQt5 and qfluentwidgets for the desktop UI

## Notes

- The GUI and documentation still reflect an evolving migration from older Torch-based inference paths to ONNX-based runtimes.
- Some older names remain for compatibility, even when the backend has already changed.
- Local model files are expected to exist under `experiments/` or another user-provided path.

## More documentation

- Architecture notes: [docs/architecture.md](docs/architecture.md)
- Third-party credits: [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
- License: [LICENSE](LICENSE)
