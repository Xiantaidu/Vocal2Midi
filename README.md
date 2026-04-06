# Vocal2Midi

Vocal2Midi is an end-to-end singing voice inference toolkit that converts vocal audio into MIDI with lyric alignment.  
It is built around the GAME note extraction model and integrates ASR + forced alignment + lyric matching into one workflow.

This repository now has a **clear primary path** and a **legacy compatibility path**:

- **Primary GUI**: `app_fluent.py` (Fluent desktop UI)
- **Primary pipeline**: `inference/auto_lyric_hybrid.py` (Hybrid pipeline)
- **Legacy GUI**: `app.py` (Gradio)
- **Legacy pipeline**: `inference/auto_lyric.py`

---

## 1. Current Architecture (Important)

### Primary (Recommended)

#### GUI
- `app_fluent.py`
- Desktop UI based on PyQt5 + qfluentwidgets
- Designed as the main production interface

#### Pipeline
- `inference/auto_lyric_hybrid.py`
- Hybrid path combining:
  - Qwen3-ASR transcription
  - LyricFA matching (optional reference lyrics)
  - HubertFA forced alignment
  - GAME note extraction

This is the default direction for ongoing use and iteration.

---

### Legacy (Compatibility)

#### GUI
- `app.py`
- Gradio Web UI retained for backward compatibility

#### Pipeline
- `inference/auto_lyric.py`
- Older ONNX/FunASR-oriented auto lyric pipeline

These are kept to avoid breaking old workflows, but are no longer the primary path.

---

## 2. Features

- Vocal-to-MIDI extraction with note timing and pitch
- Auto lyric transcription and alignment
- Optional reference lyric correction via LyricFA matching
- Chinese (`zh`) and Japanese (`ja`) flow support
- Multiple output formats (`.mid`, `.txt`, `.csv`, optional chunks/TextGrid artifacts)
- Batch processing support

---

## 3. Quick Start

## 3.1 Windows launcher (recommended)

Double-click:

```bat
start.bat
```

`start.bat` now launches:

```bat
python app_fluent.py
```

So by default, you are using the **primary Fluent GUI + Hybrid pipeline**.

### 3.2 Manual launch

Primary GUI:

```bash
python app_fluent.py
```

Legacy GUI:

```bash
python app.py
```

---

## 4. Pipeline Entry Points

### 4.1 Primary Hybrid pipeline

File:

```text
inference/auto_lyric_hybrid.py
```

Purpose:
- Main auto-lyric pipeline for current development
- Preferred for better integration of ASR + matching + FA + GAME flow

### 4.2 Legacy auto lyric pipeline

File:

```text
inference/auto_lyric.py
```

Purpose:
- Backward compatibility
- Older ONNX/FunASR-style flow

CLI help:

```bash
python inference/auto_lyric.py --help
```

---

## 5. Typical Processing Flow (Primary Path)

On the primary path (`app_fluent.py` -> `auto_lyric_hybrid.py`), the pipeline is generally:

1. Load audio and slice into chunks
2. Run ASR (Qwen3-ASR)
3. (Optional) Match ASR phonetic sequence to provided reference lyrics
4. Generate `.lab` phonetic labels
5. Run HubertFA forced alignment
6. Run GAME model to extract notes/pitches
7. Align notes with lyric units
8. Export target formats

---

## 6. Installation

## 6.1 Option A: Scripted setup (Windows)

Run:

```bat
安装环境.bat
```

## 6.2 Option B: Manual setup

Example with Conda:

```bash
conda create -n vocal2midi python=3.12
conda activate vocal2midi
pip install -r requirements.txt
```

If you only need specific ONNX-only legacy workflows, you may also use `requirements_onnx.txt` where appropriate.

---

## 7. Models and Paths

You should prepare paths for at least:

- GAME model directory
- HubertFA model directory
- ASR model path (e.g., Qwen3-ASR local path)

Suggested organization:

```text
experiments/
  GAME-1.0-medium/
  1218_hfa_model_new_dict/
  ...
```

Set these in the GUI global settings before running long jobs.

---

## 8. Language Notes

- Supported language modes include `zh` and `ja` in the current GUI.
- Japanese alignment relies on Japanese G2P/token processing and LyricFA matching behavior.
- Reference lyrics can significantly improve stability when ASR contains substitutions or inserted tokens.

---

## 9. Outputs

Depending on your options, outputs can include:

- `.mid` (MIDI)
- `.txt` (note/lyric text export)
- `.csv` (tabular note/lyric export)
- chunk artifacts / TextGrid (debug or analysis workflows)

Logs are also generated for ASR/matching behavior, which helps diagnose fallback vs matched segments.

---

## 10. Troubleshooting

- If startup fails, verify Python environment and dependencies first.
- If model loading fails, verify all model paths are valid directories/files.
- If Japanese features fail due to missing components (e.g., G2P dependencies), install the required package stack in your active environment.
- For CUDA/VRAM issues, reduce batch size in GUI settings.

---

## 11. Project Status Summary

- `app_fluent.py` is the **main GUI**.
- `inference/auto_lyric_hybrid.py` is the **main pipeline**.
- `app.py` and `inference/auto_lyric.py` are **legacy compatibility paths**.
- `app_pyside2.py` has been removed.

---

## 12. Acknowledgements

This project builds on:

- [GAME](https://github.com/openvpi/GAME)
- [HubertFA](https://github.com/Soulter/HubertFA)
- [LyricFA](https://github.com/Anya1010/LyricFA)
- [FunASR](https://github.com/modelscope/FunASR)

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for attribution details.

---

## 13. License

This project follows the [MIT License](LICENSE), with the repository-level usage disclaimer preserved.
