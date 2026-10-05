# Auto Lyric CLI

`scripts/auto_lyric_cli.py` is Vocal2Midi's headless mode: it runs the full
extraction pipeline — slicing, ASR (Qwen3 / PinyinASR / RomajiASR), HubertFA
alignment, GAME note extraction, quantization — from the command line and
writes MIDI/USTX/VSQX/text outputs, with no GUI involved.

The CLI mirrors the GUI: every option defaults to the value saved by the GUI
settings pages, and any flag you pass explicitly overrides that default.

## Quick start

```bash
# Single file, MIDI output (defaults for everything else)
python scripts/auto_lyric_cli.py song.wav

# Whole folder, batch, MIDI + USTX with pitch curve
python scripts/auto_lyric_cli.py D:\\songs -o D:\\out --language zh --formats mid ustx --pitch-curve

# Pitch only, no lyrics (skips ASR/HFA entirely, fastest path)
python scripts/auto_lyric_cli.py song.wav --no-lyrics --formats mid

# Japanese song with reference lyrics for alignment
python scripts/auto_lyric_cli.py song.wav --language ja --lyric-format romaji --lyrics-file lyrics.txt
```

On Windows, the portable package ships a **Run Auto Lyric CLI.bat** launcher
that opens the CLI with the bundled runtime; arguments typed after the bat are
passed through, e.g. `Run Auto Lyric CLI.bat D:\\songs -o outputs`.

## How defaults work

Defaults are read from the same settings store the GUI uses —
`settings/vocal2midi.ini` in portable mode (when `V2M_PORTABLE_ROOT` is set),
the Windows registry store (`GAME_Extractor\\Vocal2Midi`) otherwise. Keys
read: the six model paths, `chinese_asr_engine`, `japanese_asr_engine`,
`alignment_engine`, `batch_size`, `asr_batch`, `slice_min_sec`, `slice_max_sec`, `t0`, `nsteps`,
`seg_thresh`, `seg_rad`, `est_thresh`, `pitch_format`, `round_pitch`,
`output_pitch_curve`, `save_dir` (used as the default output directory) and
`lyric_output_mode_<language>` (used as the default lyric format). Any flag
on the command line wins over the stored value; if a key was never saved, the
same fallback the GUI uses applies.

## Inputs and output

| Option | Description |
|---|---|
| `inputs` (positional, one or more) | Audio files and/or directories. Directories are recursed and filtered by extension: `.wav .m4a .flac .mp3 .ogg .opus .wma .webm .aif .aiff`. Files already inside the output directory are skipped. |
| `-o, --output-dir` | Output directory (created if missing). Default: the GUI save directory. |
| `--no-recursive` | Do not descend into subdirectories for directory inputs. |

## Lyrics and language

| Option | Description |
|---|---|
| `--language {zh,ja,en}` | Lyric language. Default `zh`. A batch runs in one language; split mixed-language folders into separate runs. |
| `--lyric-format {pinyin,hanzi,romaji,kana,word}` | Lyric output format. Default: the GUI setting for the language. Must be valid for the language (see matrix below) or the CLI exits with an error. |
| `--lyrics TEXT` | Reference lyrics inline (improves alignment accuracy). |
| `--lyrics-file PATH` | Reference lyrics from a UTF-8 text file (overrides `--lyrics`). |
| `--no-lyrics` | Pitch-only mode: ASR and HFA are skipped entirely and outputs contain no lyrics. Cannot be combined with `--lyrics`/`--lyrics-file`. |

Valid lyric formats per language:

| Language | Formats | Default |
|---|---|---|
| `zh` | `pinyin`, `hanzi` | `hanzi` |
| `ja` | `romaji`, `kana` | `romaji` |
| `en` | `word` | `word` |

## ASR engines

| Option | Description |
|---|---|
| `--chinese-asr {pinyin,qwen}` | `pinyin` runs the direct PinyinASR phoneme model (output is locked to pinyin — choosing `hanzi` emits a warning and the pipeline falls back to pinyin); `qwen` runs text ASR + G2P. |
| `--japanese-asr {romaji,qwen}` | `romaji` runs the direct RomajiASR mora model; `qwen` runs text ASR + Japanese G2P. |
| `--aligner {tifa,hfa}` | Forced-alignment engine: `tifa` runs the TiFA aligner (audio + raw ASR text, built-in G2P with polyphone disambiguation); `hfa` runs HubertFA on the phoneme sequence. Default: the GUI setting, `tifa`. |

Both default to the model config page's engine selection. The engine not in
use is not loaded and its model path is not required.

## Slicing

| Option | Description |
|---|---|
| `--slicing {smart,heuristic,default,grid}` | Slicing strategy. Default `smart`. |
| `--min-seconds` / `--max-seconds` | Chunk duration bounds in seconds (defaults from settings; 0 < min ≤ max ≤ 60). |

## Quantization

| Option | Description |
|---|---|
| `--quant-step {0,480,240,120,60,30}` | Grid in MIDI ticks: `0` off, `480` 1/4 note, `240` 1/8, `120` 1/16, `60` 1/32, `30` 1/64. Default `0` (off). |
| `--quant-mode {smart,simple}` | `smart` = rhythmic alignment engine, `simple` = plain grid snap. Default `smart`. |
| `--quant-simplicity FLOAT` | Smart-mode simplicity (higher = more forgiving). Default `0.0`, the same conservative value the GUI uses. |
| `--tempo FLOAT` | Tempo in BPM used for quantization and MIDI export. Default `120`. |

## Output formats

| Option | Description |
|---|---|
| `--formats` (one or more of) | `mid`, `txt`, `csv`, `ustx`, `vsqx`, `chunks` (chunk WAVs), `asr_match_log` (per-chunk ASR/matching report). Default `mid`. |

## Pitch options

| Option | Description |
|---|---|
| `--pitch-curve` / `--no-pitch-curve` | Embed the RMVPE pitch curve in USTX/VSQX. Requires `--rmvpe-model` when enabled. Default: the GUI setting. |
| `--pitch-format {name,number}` | Pitch notation for txt/csv outputs. |
| `--round-pitch` / `--no-round-pitch` | Round pitch values in text outputs. |

## Performance

| Option | Description |
|---|---|
| `--device {dml,cpu,cuda,metal}` | Runtime device. Default: auto-detected. |
| `--batch-size` | GAME inference batch size. |
| `--asr-batch-size` | ASR chunk batch size. |

A batch of files shares **one** ASR worker subprocess: the multi-GB Qwen model
loads once for the whole run, and only if at least one file actually reaches
the text-ASR stage. If a worker dies or the device/model changes mid-batch it
is respawned automatically.

## Advanced model parameters

| Option | Description |
|---|---|
| `--t0` / `--nsteps` | D3PM start value and sampling steps (passed to GAME/HFA decoding). |
| `--seg-threshold` | Boundary decode threshold. |
| `--seg-radius` | Boundary decode radius (seconds). |
| `--est-threshold` | Note existence threshold. |

## Model paths

| Option | Default (GUI setting) |
|---|---|
| `--game-model` | `models/GAME-1.0.3-medium-onnx` |
| `--hfa-model` | `models/1218_hfa_model_new_dict` |
| `--tifa-model` | `models/tifa-1.0-onnx` |
| `--asr-model` | `models/Qwen3-ASR-1.7B-dml` |
| `--phoneme-asr-model` | `models/romajiASR` |
| `--pinyin-asr-model` | `models/pinyinASR` |
| `--rmvpe-model` | `models/RMVPE` |

Only the paths required by the selected language, engines and output formats
are validated, so a missing RomajiASR model does not block a Chinese run.

## Batch behavior and exit codes

- Files are processed sequentially in deterministic (sorted) order; each file
  gets its own PipelineConfig built from the same arguments.
- A failing file is logged and skipped; the remaining files still run.
- Ctrl+C cancels the run and closes the ASR worker cleanly.

| Exit code | Meaning |
|---|---|
| `0` | All files processed successfully |
| `1` | At least one file failed |
| `2` | Usage/input error (bad option combination, no audio files found, ...) |
| `130` | Cancelled by the user (Ctrl+C) |

## See also

- `scripts/slice_asr_cli.py` — the older batch CLI focused on slicing + raw
  ASR text extraction (no alignment/GAME/notation outputs).
- `README.md` — GUI overview and setup.
