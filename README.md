# Vocal2Midi

<img src="icon.png" width="72" alt="Vocal2Midi" />

<p>
  <a href="https://github.com/Xiantaidu/Vocal2Midi"><img src="https://img.shields.io/badge/version-v2.0.0-blue.svg?style=flat-square" alt="Version"></a> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python"></a> <a href="#runtime-device-rules"><img src="https://img.shields.io/badge/platform-Windows-0078D6.svg?style=flat-square" alt="Platform"></a> <a href="#runtime-device-rules"><img src="https://img.shields.io/badge/acceleration-dml%20%7C%20cpu-success.svg?style=flat-square" alt="Acceleration"></a> <a href="#gui-workflow"><img src="https://img.shields.io/badge/UI-PySide6%20%7C%20Fluent%20Design-005FB8.svg?style=flat-square&logo=qt&logoColor=white" alt="UI"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-yellow.svg?style=flat-square" alt="License"></a>
</p>

Vocal2Midi is a Windows desktop tool and inference pipeline for turning vocal audio into lyric-aligned MIDI, USTX, VSQX, and editing artifacts.

The current runtime is **ONNX-first**:

- `llama.cpp` is used for the Qwen decoder on CPU.
- ONNX models default to **DirectML** acceleration and fall back to **CPU** when DirectML is unavailable.
- The main user-facing entrypoint is the Fluent GUI in [`app_fluent.py`](app_fluent.py).

## Highlights

- End-to-end vocal-to-MIDI workflow in one project
- Chinese, Japanese, English, and Cantonese lyric handling
- Selectable dual forced-alignment engines: default **TiFA** and **HubertFA**
- Multiple ASR engines: Qwen3-ASR, RomajiASR, and lightweight PinyinASR
- Selectable Japanese G2P engines: **kashi-g2p-onnx** with candidate beam search for TiFA, and **pyopenjtalk**
- Smart rhythmic quantization engine with configurable quantization steps
- Modern Fluent GUI workflow with multi-file batch mode and live dark/light theming
- Headless batch CLI [`auto_lyric_cli.py`](scripts/auto_lyric_cli.py) and folder slicing CLI [`slice_asr_cli.py`](scripts/slice_asr_cli.py)
- Portable-folder packaging flow for Windows distribution
- ONNX-based inference stack for ASR, alignment, note extraction, and RMVPE

## What Vocal2Midi Does

At a high level, the hybrid pipeline looks like this:

```text
audio
  -> optional RMVPE pitch curve
  -> slicing
  -> ASR: Qwen3-ASR, PinyinASR, or RomajiASR
  -> lyric matching / .lab generation
  -> forced alignment: TiFA or HubertFA
  -> GAME note extraction
  -> quantization: smart rhythmic alignment or simple
  -> export: MIDI, USTX, VSQX, TextGrid, WAV
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
| PinyinASR | ONNX Runtime | `inference/pinyin_asr/` |
| RomajiASR | ONNX Runtime | `inference/romaji_asr/` |
| kashi-g2p | ONNX Runtime | `inference/kashi_g2p_ja/` |
| TiFA | ONNX Runtime | `inference/TiFA/` |
| HubertFA | ONNX Runtime | `inference/HubertFA/` |
| GAME | ONNX Runtime | `inference/game/` |
| RMVPE | ONNX Runtime | `inference/API/rmvpe_api.py` |
| Device normalization | DirectML / CPU helpers | `inference/device_utils.py` |

## Repository Layout

```text
application/   application-layer orchestration and config objects
docs/          architecture notes and supporting docs
models/        local model directories
gui/           PySide6 + qfluentwidgets desktop UI v2.0.0
inference/     ASR, alignment, pitch extraction, slicing, quantization, export
scripts/       batch CLI and portable build helpers
tests/         automated tests
```

## Quick Start

### 1. Install dependencies

Use Python 3.10, 3.11, or 3.12, then install:

```bash
pip install -r requirements.txt
```

The main runtime dependencies are:

- `onnxruntime-directml`
- `PySide6`
- `PySide6-Fluent-Widgets`
- `librosa`
- `soundfile`
- `mido`
- `pyopenjtalk`
- `opencc-python-reimplemented`

An `environment.yml` file is also included as a reference environment snapshot.

### 2. Prepare model folders

By default, the GUI expects models in these locations:

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

You can change these paths in the GUI settings panel.

### 3. Launch the GUI

For a normal developer environment:

```bash
python app_fluent.py
```

## GUI Workflow

The GUI is the main way to use Vocal2Midi interactively. It lets you:

- choose model paths and switch between TiFA and HubertFA aligners
- choose Chinese ASR between Qwen3 and PinyinASR, and Japanese ASR between RomajiASR and Qwen3
- choose Japanese G2P between `kashi-g2p-onnx` and `pyopenjtalk`
- pick the runtime device: `dml` or `cpu`
- process single audio files or multiple files in batch mode
- set slicing mode and slice length bounds
- choose language and lyric output format: Hanzi, Pinyin, Romaji, Kana, Word, or Jyutping
- select quantization modes: smart rhythmic alignment or simple quantization
- toggle live UI language and themes
- provide optional reference lyrics
- export MIDI, USTX, VSQX, text, CSV, chunk audio, and alignment artifacts

The application-layer job entrypoint is run_auto_lyric_job in [`application/pipeline.py`](application/pipeline.py), which dispatches into the hybrid inference pipeline in [`inference/pipeline/auto_lyric_hybrid.py`](inference/pipeline/auto_lyric_hybrid.py).

## Language Behavior

### Chinese

- Transcription via `Qwen3-ASR` or lightweight `PinyinASR`.
- The lyric matcher and G2P path prepare `.lab` content for TiFA or HubertFA alignment.
- Lyrics can be exported in Hanzi or pinyin-oriented forms depending on mode.

The lightweight Chinese singing ASR integration is based on
[Xiantaidu/PinyinASR](https://github.com/Xiantaidu/PinyinASR), used for the
`models/pinyinASR` model path and the `inference/pinyin_asr/` runtime integration.

### Japanese

In the main hybrid lyric pipeline:

- `romaji` and `kana` lyric modes use the dedicated **mora ASR** path
- if the output mode is `romaji`, the pipeline uses mora ASR output directly
- if the output mode is `kana`, the pipeline converts matched mora output to kana for display
- if reference lyrics are provided, the reference text is processed through `pyopenjtalk` or `kashi-g2p-onnx`, converted to kana mora tokens, then converted again to romaji mora tokens for matching

This keeps Japanese lyric matching consistent with the mora-based ASR path instead of routing through the old phoneme-ASR forced-alignment branch.

The current Japanese mora / romaji ASR integration in this repository is based on
[Xiantaidu/RomajiASR](https://github.com/Xiantaidu/RomajiASR), the separate
Japanese singing ASR project used for the `models/romajiASR` model path
and the `inference/romaji_asr/` runtime integration.

The neural Japanese G2P integration is based on
[Xiantaidu/kashi-g2p](https://github.com/Xiantaidu/kashi-g2p), used for the
`models/kashi-g2p-onnx` model path and the `inference/kashi_g2p_ja/` runtime
integration, providing lattice beam-search candidates for TiFA alignment.

### English

- Qwen3-ASR transcribes with the `English` language prompt; CJK bleed-over,
  digits, and punctuation are stripped so only dictionary-friendly words remain.
- The `word` lyric output mode assigns one note per English word.
- After alignment, multi-syllable words are split into per-syllable time
  chunks, such as impossible -> im/poss/ible, fed to GAME as align
  units: the word lyric lands on the first note, later syllable positions
  are marked `+`, and melisma notes inside a syllable fall back to `-`.
- If reference lyrics are provided, they are matched word-by-word against the
  ASR output before alignment.
- HubertFA aligns through the DiffSinger CMU dict ds_cmudict-07b.txt shipped
  inside the HubertFA model folder; words missing from the dictionary are warned
  and skipped.
- Breath detection for AP is enabled for English, mirroring the Chinese path.

### Cantonese

- Powered by dedicated Cantonese G2P and phonetic mapping for TiFA forced alignment.
- HubertFA does not support Cantonese; Cantonese alignment requires TiFA, and HubertFA is disabled for Cantonese in the GUI.
- Lyrics can be aligned and exported in Jyutping or Hanzi formats.

## Runtime Device Rules

Visible device options in the current UI are:

- `dml`
- `cpu`

Notes:

- `dml` is the default device on Windows with DirectML GPU acceleration
- if DirectML is unavailable, ONNX Runtime automatically falls back to CPU
- legacy device values such as `cuda` are normalized to `dml`

## Slicing

The user-facing slice duration settings currently support:

- minimum slice length: `0` to `60` seconds
- maximum slice length: `0` to `60` seconds

Current defaults:

- minimum: `5.0` seconds
- maximum: `10.0` seconds

Validation rules:

- `slice_max_sec` must be greater than `0`
- `slice_min_sec` must be less than or equal to `slice_max_sec`

## Batch Slice + ASR CLI

For folder-based batch ASR processing:

```bash
python scripts/slice_asr_cli.py <input_dir> <output_dir> \
  --asr-model models/Qwen3-ASR-1.7B-dml \
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
  --asr-model models/Qwen3-ASR-1.7B-dml \
  --device dml \
  --language ja \
  --no-slice
```

## Auto Lyric CLI

Headless mode: run the full extraction pipeline from audio slicing and ASR to forced alignment and note extraction from the command line, without the GUI. Defaults mirror the GUI settings; any flag overrides them. Full option reference: [scripts/auto_lyric_cli.md](scripts/auto_lyric_cli.md).

```bash
python scripts/auto_lyric_cli.py <input_files_or_dirs...> -o <output_dir> --language zh --formats mid ustx
```

- Inputs may be audio files and directories; directory scanning is recursive by default, or flat with `--no-recursive`.
- `--no-lyrics` extracts pitch only and skips ASR and alignment entirely.
- `--lyrics` and `--lyrics-file` provide reference lyrics for alignment.
- `--language zh|ja|en|yue` with `--lyric-format pinyin|hanzi|romaji|kana|word|jyutping`.
- `--chinese-asr pinyin|qwen`, `--japanese-asr romaji|qwen`, and `--alignment-engine tifa|hfa` select the engines matching the model config page.
- A batch of files shares one ASR worker process; a failing file is reported and skipped, and the exit code is 1 if any file failed.



## Windows Setup Scripts

The repository also includes Windows helper scripts:

- [`install.bat`](install.bat)
- [`run.bat`](run.bat)

These are useful for a smaller distribution model where the user downloads or initializes the runtime on first setup rather than receiving a fully bundled `python/` folder.

## Export Formats

Depending on the selected workflow, Vocal2Midi can export:

- `.mid`
- `.ustx`
- `.vsqx`
- `.txt`
- `.csv`
- `TextGrid`
- chunk `.wav` files
- `.lab`
- ASR matching logs

## Project Notes

- The repository has already migrated away from the earlier Torch-heavy runtime design for the main inference path.
- Model assets are expected to exist locally under `models/` or another user-provided path.
- The codebase is still being cleaned up in places, so you may still see a few legacy names or UI strings from earlier iterations.

## License

The overall Vocal2Midi repository is distributed under the **Apache License 2.0**. See [LICENSE](LICENSE).

Third-party components, vendored code, model assets, dictionaries, and other embedded materials may also carry their own original licenses, notices, or attribution requirements. Those original notices remain applicable to the corresponding materials. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) and any embedded license files for details.

## Development and Testing

The repo includes a focused automated test suite under `tests/`.

Examples:

```bash
python -m pytest tests/test_auto_lyric_hybrid_pipeline.py
python -m pytest tests/test_gui_components.py tests/test_auto_lyric_cli.py
python -m pytest tests/test_asr_api.py tests/test_game_api.py tests/test_rmvpe_api.py
```

For architecture details, see [docs/architecture.md](docs/architecture.md).

## Related Files

- Main GUI entrypoint: [`app_fluent.py`](app_fluent.py)
- Architecture notes: [docs/architecture.md](docs/architecture.md)
- Third-party credits: [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
- License: [LICENSE](LICENSE)
