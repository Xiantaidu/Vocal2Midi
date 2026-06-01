# Vocal2Midi Architecture

This document describes the current structure of the Vocal2Midi repository after the move toward ONNX-based inference.

## Design goals

The codebase is organized to keep UI concerns, application orchestration, and inference runtimes separate:

```text
gui -> application -> inference
```

The intended dependency direction is one-way:

- `gui/` should not directly depend on low-level runtime details when an application-layer entrypoint exists.
- `application/` coordinates jobs and validates configuration.
- `inference/` owns the actual transcription, alignment, pitch extraction, slicing, and export logic.

## Top-level structure

```text
application/
  config.py
  pipeline.py

gui/
  fluent_main.py
  auto_lyric_view.py
  global_settings_view.py
  fluent_worker.py

inference/
  API/
  HubertFA/
  LyricFA/
  game/
  io/
  pipeline/
  qwen3asr_dml/
  quant/
  romaji_asr/
  slicer/
  device_utils.py
```

## Main runtime model

The current runtime split is:

- Qwen3-ASR encoder path: ONNX Runtime with DirectML when available
- Qwen3-ASR decoder path: `llama.cpp` on CPU
- Romaji ASR: ONNX Runtime
- HubertFA: ONNX Runtime
- GAME: ONNX Runtime
- RMVPE: ONNX Runtime

Device normalization is centralized in `inference/device_utils.py`.

The project accepts legacy `cuda` requests in some public interfaces, but those requests are normalized to `dml` in the current runtime path.

## Main user entrypoints

### Desktop GUI

- `app_fluent.py`
- `gui/fluent_main.py`

The GUI is the primary interactive entrypoint. It collects user settings, launches worker threads, and routes work into the application layer.

### Application-layer job entry

- `application/pipeline.py`

`run_auto_lyric_job()` is the main application-layer boundary used by the GUI. It validates required model paths and then dispatches into the hybrid inference pipeline.

### Hybrid pipeline

- `inference/pipeline/auto_lyric_hybrid.py`

`auto_lyric_hybrid_pipeline()` is the central end-to-end workflow for:

1. loading audio
2. optional RMVPE pitch-curve extraction
3. slicing
4. ASR or romaji ASR
5. lyric matching
6. HubertFA forced alignment
7. GAME note extraction
8. export

## Key inference APIs

### ASR

- `inference/API/asr_api.py`

Responsibilities:

- load and cache the Qwen3-ASR DML runtime
- load and cache the Japanese romaji ASR runtime
- batch transcription for chunked audio
- isolate Qwen subprocess work when needed

Implementation backends:

- `inference/qwen3asr_dml/`
- `inference/romaji_asr/`

### Lyric matching and G2P helpers

- `inference/API/lfa_api.py`
- `inference/LyricFA/`

Responsibilities:

- create lyric matchers
- normalize ASR text or phoneme output
- generate `.lab` alignment input
- support Chinese and Japanese lyric display modes

### Forced alignment

- `inference/API/hfa_api.py`
- `inference/HubertFA/`

Responsibilities:

- load the HubertFA ONNX model
- build input datasets from chunk `.wav` and `.lab` pairs
- run phoneme-level forced alignment
- export TextGrid artifacts

### Note extraction

- `inference/API/game_api.py`
- `inference/game/`

Responsibilities:

- load the GAME ONNX model set
- run note extraction with or without lyric alignment
- align GAME output to word durations when HFA output exists

Some public function names still include `_torch` for compatibility, but the active implementation is ONNX-based.

### RMVPE

- `inference/API/rmvpe_api.py`

Responsibilities:

- load the RMVPE ONNX model
- compute frame-level salience and F0
- provide interpolated MIDI pitch curves for USTX export
- optionally support smart slicing through voiced-mask information

### Slicing

- `inference/API/slicer_api.py`
- `inference/slicer/`

Responsibilities:

- default slicing
- heuristic or smart slicing paths
- optional RMVPE-assisted voiced/unvoiced guidance

## Data flow

At a high level, the hybrid pipeline looks like this:

```text
audio
  -> optional RMVPE
  -> slicing
  -> ASR / romaji ASR
  -> lyric matching and .lab generation
  -> HubertFA alignment
  -> GAME note extraction
  -> quantization
  -> export
```

There is also a no-lyrics path:

```text
audio
  -> optional RMVPE
  -> slicing
  -> GAME pitch-only extraction
  -> export
```

## Export layer

Relevant modules:

- `inference/io/note_io.py`
- `inference/API/ustx_api.py`

The export layer is responsible for writing:

- MIDI
- USTX
- TXT
- CSV
- TextGrid
- chunk artifacts and logs when requested

## Runtime and cancellation behavior

The pipeline is built to support:

- background execution from the GUI worker thread
- cancellation via `cancel_checker`
- chunk batching for ASR and GAME
- runtime reuse for batch CLI flows

Memory cleanup is currently lightweight and mostly centered on releasing cached model objects and running Python garbage collection.

## Batch CLI path

- `scripts/slice_asr_cli.py`

The batch CLI provides a folder-oriented workflow that can:

- scan audio files
- slice them or bypass slicing
- run Qwen3-ASR
- save chunks and `.lab` files
- optionally keep ASR and RMVPE runtimes alive across the batch

## Current rough edges

The repository is mid-migration, so a few historical details still remain:

- some function names still mention Torch even though the backend changed
- some UI strings are not yet normalized
- `environment.yml` still reflects older dependency history more than the current runtime design

Those issues do not change the intended architecture direction: ONNX-first inference, CPU `llama.cpp`, and stable application-layer entrypoints.
