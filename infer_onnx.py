#!/usr/bin/env python3
"""
Command-line interface for pure ONNX inference of GAME models.
This script uses the API defined in `inference.onnx_api`.
"""

import glob
import pathlib
from typing import Any

import click
import librosa
from inference.onnx_api import (
    load_onnx_model,
    infer_from_files,
    align_with_transcriptions,
)

# A subset of options for simple CLI
_OPT_KEY_BATCH_SIZE = "batch_size"
_OPT_KEY_NUM_WORKERS = "num_workers"
_OPT_KEY_SEG_THRESHOLD = "seg_threshold"
_OPT_KEY_SEG_RADIUS = "seg_radius"
_OPT_KEY_SEG_D3PM_T0 = "t0"
_OPT_KEY_SEG_D3PM_NSTEPS = "nsteps"
_OPT_KEY_EST_THRESHOLD = "est_threshold"


# --- Validation Callbacks (copied from original infer.py) ---
def _validate_d3pm_ts(ctx, param, value) -> list[float] | None:
    if value is None:
        return None
    try:
        ts = [float(t.strip()) for t in value.split(",")]
        if not ts:
            raise ValueError("At least one T value must be provided.")
        if any(t < 0 or t >= 1 for t in ts):
            raise ValueError("All T values must be in the range (0, 1).")
        return ts
    except Exception as e:
        raise click.BadParameter(f"Invalid T values: {e}")


def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")


def _validate_output_formats(ctx, param, value) -> set[str]:
    try:
        formats = {fmt.strip().lower() for fmt in value.split(",")}
        supported_formats = {"mid", "txt", "csv"}
        if not formats.issubset(supported_formats):
            raise ValueError(f"Unsupported formats: {formats - supported_formats}.")
        return formats
    except Exception as e:
        raise click.BadParameter(f"Invalid output formats: {e}")


def _validate_uv_vocab(ctx, param, value) -> set[str] | None:
    if value is None:
        return None
    try:
        vocab = {v.strip() for v in value.split(",") if v.strip()}
        return vocab
    except Exception as e:
        raise click.BadParameter(f"Invalid UV vocab: {e}")


def _validate_path_or_glob(ctx, param, value) -> list[pathlib.Path]:
    try:
        paths = []
        for v in value:
            if glob.has_magic(v):
                paths.extend(glob.glob(v, recursive=True))
            else:
                paths.append(v)
        if not paths:
            raise FileNotFoundError(f"No files found for paths: {value}")
        paths = [pathlib.Path(p) for p in paths]
        for p in paths:
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Path does not exist or is not a file: {p}")
        return paths
    except Exception as e:
        raise click.BadParameter(f"Invalid path or glob: {e}")

# --- Helper Functions ---
def _t0_nstep_to_ts(t0: float, nsteps: int) -> list[float]:
    return [t0 + i * ((1 - t0) / nsteps) for i in range(nsteps)]

def _get_language_id(language: str, lang_map: dict[str, int]) -> int:
    if language and lang_map:
        if language not in lang_map:
            raise ValueError(f"Language '{language}' not supported. Supported: {', '.join(lang_map.keys())}")
        return lang_map[language]
    return 0

def _parse_filemap(path: pathlib.Path, exts: set[str], glb: str | None) -> dict[str, pathlib.Path]:
    if path.is_file():
        return {path.name: path}
    if path.is_dir():
        files = [f for f in path.rglob(glb if glb else "*") if f.is_file() and (glb or f.suffix.lower() in exts)]
        filemap = {f.relative_to(path).as_posix(): f for f in files}
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    raise ValueError(f"Invalid path: {path}")


@click.group()
def main():
    """ONNX-based inference for GAME."""
    pass


def shared_options(func=None, *, defaults: dict[str, Any] = None):
    defaults = defaults or {}
    options = [
        click.option("-m", "--model", required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True, path_type=pathlib.Path), help="Path to the ONNX model directory."),
        click.option("-d", "--device", type=click.Choice(["dml", "cpu"]), default="dml", show_default=True, help="Execution provider."),
        click.option("-l", "--language", type=str, help="Language code for segmentation."),
        click.option("--batch-size", type=click.IntRange(min=1), show_default=True, default=defaults.get(_OPT_KEY_BATCH_SIZE, 2), help="Batch size for inference."),
        click.option("--seg-threshold", type=click.FloatRange(0, 1, max_open=True), default=defaults.get(_OPT_KEY_SEG_THRESHOLD, 0.2), show_default=True, help="Boundary decoding threshold."),
        click.option("--seg-radius", type=click.FloatRange(min=0.01), default=defaults.get(_OPT_KEY_SEG_RADIUS, 0.02), show_default=True, help="Boundary decoding radius in seconds."),
        click.option("--t0", "--seg-d3pm-t0", type=click.FloatRange(0, 1, max_open=True), default=defaults.get(_OPT_KEY_SEG_D3PM_T0, 0.0), show_default=True, help="D3PM starting T value (t0)."),
        click.option("--nsteps", "--seg-d3pm-nsteps", type=click.IntRange(min=1), default=defaults.get(_OPT_KEY_SEG_D3PM_NSTEPS, 8), show_default=True, help="Number of D3PM sampling steps."),
        click.option("--ts", "--seg-d3pm-ts", type=str, callback=_validate_d3pm_ts, help="Custom D3PM T values, comma-separated."),
        click.option("--est-threshold", type=click.FloatRange(0, 1, max_open=True), default=defaults.get(_OPT_KEY_EST_THRESHOLD, 0.2), show_default=True, help="Note presence threshold."),
    ]
    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f
    return decorator(func) if func else decorator


@main.command(name="extract", help="Extract MIDI from audio files.")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True, readable=True, path_type=pathlib.Path))
@shared_options
@click.option("--input-formats", type=str, default="wav,flac,mp3", show_default=True, callback=_validate_exts, help="Audio file extensions to process.")
@click.option("--glob", "glb", type=str, help="Glob pattern to filter audio files.")
@click.option("--output-formats", type=str, default="mid", show_default=True, callback=_validate_output_formats, help="Output formats (mid, txt, csv).")
@click.option("--tempo", type=click.FloatRange(min=0, min_open=True), default=120, show_default=True, help="Tempo for MIDI output.")
@click.option("--pitch-format", type=click.Choice(["number", "name"]), default="name", show_default=True, help="Pitch format for text output.")
@click.option("--round-pitch", is_flag=True, help="Round pitch values in text output.")
@click.option("--output-dir", type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path), help="Directory to save results.")
def extract_cli(path: pathlib.Path, model: pathlib.Path, device: str, **kwargs):
    ts = kwargs.pop('ts') or _t0_nstep_to_ts(kwargs.pop('t0'), kwargs.pop('nsteps'))
    filemap = _parse_filemap(path, kwargs.pop('input_formats'), kwargs.pop('glb'))
    output_dir = kwargs.pop('output_dir') or (path if path.is_dir() else path.parent)
    
    inference_model = load_onnx_model(model, device)
    language_id = _get_language_id(kwargs.pop('language'), inference_model.languages)
    
    infer_from_files(
        model=inference_model,
        filemap=filemap,
        output_dir=output_dir,
        language_id=language_id,
        ts=ts,
        **kwargs
    )
    print("\nInference completed.")


@main.command(name="align", help="Align notes to DiffSinger transcriptions.")
@click.argument("paths", type=str, nargs=-1, callback=_validate_path_or_glob)
@shared_options
@click.option("--save-path", type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=pathlib.Path), help="Output file path (for single input).")
@click.option("--save-name", type=str, help="Output file name (for multiple inputs).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
@click.option("--uv-vocab", type=str, default="AP,SP,br,sil", show_default=True, callback=_validate_uv_vocab, help="Comma-separated list of unvoiced phoneme names.")
@click.option("--uv-vocab-path", type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path), help="Path to a file containing unvoiced phoneme names separated by whitespace. Overrides --uv-vocab if provided.")
@click.option("--uv-word-cond", type=click.Choice(["lead", "all"]), default="lead", show_default=True, help="Condition for determining whether a word is unvoiced. 'lead' checks only the leading phoneme; 'all' requires all phonemes to be unvoiced.")
@click.option("--uv-note-cond", type=click.Choice(["predict", "follow"]), default="predict", show_default=True, help="Condition for determining whether a note is unvoiced. 'predict' uses the raw model output; 'follow' aligns note v/uv with the word it belongs to.")
@click.option("--no-wb", is_flag=True, help="Disable word boundaries.")
def align_cli(paths: list[pathlib.Path], model: pathlib.Path, device: str, **kwargs):
    ts = kwargs.pop('ts') or _t0_nstep_to_ts(kwargs.pop('t0'), kwargs.pop('nsteps'))
    save_path = kwargs.pop('save_path')
    save_name = kwargs.pop('save_name')
    
    use_wb = not kwargs.pop('no_wb')
    uv_vocab = kwargs.pop('uv_vocab')
    uv_vocab_path = kwargs.pop('uv_vocab_path')
    uv_word_cond = kwargs.pop('uv_word_cond')
    uv_note_cond = kwargs.pop('uv_note_cond')

    if uv_vocab_path is not None:
        uv_vocab = set(uv_vocab_path.read_text(encoding="utf8").split())
    resolved_uv_vocab = uv_vocab if use_wb else None

    if use_wb and uv_note_cond == "follow" and not resolved_uv_vocab:
        print("WARNING: --uv-note-cond follow is set but the UV vocab is empty, so all words are treated as voiced and all notes will be forced voiced. This is not expected in most cases. Please check --uv-vocab and --uv-vocab-path options.")

    
    if len(paths) > 1 and save_path:
        print("WARNING: --save-path is ignored when multiple input files are provided.")
        save_path = None
    
    inplace = not save_path and not save_name
    if inplace and not kwargs.get('overwrite'):
        raise click.UsageError("In-place saving requires --overwrite flag.")

    save_dir = save_path.parent if save_path else None
    
    inference_model = load_onnx_model(model, device)
    language_id = _get_language_id(kwargs.pop('language'), inference_model.languages)

    align_with_transcriptions(
        model=inference_model,
        transcription_paths=paths,
        language_id=language_id,
        ts=ts,
        use_wb=use_wb,
        inplace=inplace,
        save_dir=save_dir,
        save_name=save_name or (save_path.name if save_path else None),
        uv_vocab=resolved_uv_vocab,
        uv_word_cond=uv_word_cond,
        uv_note_cond=uv_note_cond,
        **kwargs
    )
    print("\nAlignment completed.")

if __name__ == '__main__':
    main()
