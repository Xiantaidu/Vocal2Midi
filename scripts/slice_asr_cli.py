"""Batch slicing + local Qwen3-ASR CLI.

This script processes a folder of audio files, optionally slices each file,
runs local ASR on each chunk, and writes both chunk WAVs and `.lab` text files.

Supported input formats: `.wav`, `.m4a`, `.mp3`
Supported workflows:
- normal slicing + ASR
- `--no-slice` whole-file ASR
- JSON-driven re-slicing via `--from-json`
- persistent ASR / RMVPE runtimes via `--keep-model` / `--keep-rmvpe`
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import librosa
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

asr_api = importlib.import_module("inference.API.asr_api")
rmvpe_api = importlib.import_module("inference.API.rmvpe_api")
slicer_api = importlib.import_module("inference.API.slicer_api")
device_utils = importlib.import_module("inference.device_utils")

batch_transcribe_asr = asr_api.batch_transcribe_asr
load_qwen_model = asr_api.load_qwen_model
clear_qwen_model_cache = asr_api.clear_qwen_model_cache
RmvpeTranscriber = rmvpe_api.RmvpeTranscriber
slice_audio = slicer_api.slice_audio
slice_audio_with_custom_bounds = getattr(slicer_api, "slice_audio_with_custom_bounds", None)
RUNTIME_DEVICE_CHOICES = device_utils.RUNTIME_DEVICE_CHOICES
DEFAULT_RUNTIME_DEVICE = getattr(device_utils, "default_runtime_device", lambda: "dml")()
normalize_runtime_device = device_utils.normalize_runtime_device

DEFAULT_RMVPE_MODEL = ROOT_DIR / "experiments" / "RMVPE" / "rmvpe.onnx"
INPUT_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3"}
SOURCE_INDEX_NAME = "_source_index.json"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_SLICE_METHOD = "default"
SLICE_METHOD_CHOICES = (
    "default",
    "smart",
    "heuristic",
    "grid",
)
SLICE_METHOD_ALIASES = {
    "auto": DEFAULT_SLICE_METHOD,
    "default": "default",
    "smart": "smart",
    "heuristic": "heuristic",
    "grid": "grid",
    "默认切片": "default",
    "智能切片": "smart",
    "启发式切片": "heuristic",
    "网格搜索切片": "grid",
}
SLICE_METHOD_KEYWORDS = (
    ("smart", ("smart", "智能")),
    ("heuristic", ("heuristic", "启发式")),
    ("grid", ("grid", "网格")),
    ("default", ("default", "默认", "auto")),
)


def batch_iter(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ensure_ffmpeg_on_path() -> None:
    ffmpeg_bin = ROOT_DIR / "_ffmpeg" / "bin"
    if not ffmpeg_bin.is_dir():
        return

    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    ffmpeg_bin_str = str(ffmpeg_bin)
    if ffmpeg_bin_str not in parts:
        os.environ["PATH"] = ffmpeg_bin_str + (os.pathsep + current_path if current_path else "")


def repair_text_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    for source_encoding in ("gb18030", "gbk"):
        try:
            repaired = stripped.encode(source_encoding).decode("utf-8", errors="ignore").strip()
        except UnicodeError:
            continue
        if repaired and repaired not in candidates:
            candidates.append(repaired)
    return candidates


def normalize_slicing_method(method: str) -> str:
    if method is None:
        return DEFAULT_SLICE_METHOD

    candidates = []
    for candidate in repair_text_candidates(str(method)):
        lowered = candidate.lower()
        for value in (candidate, lowered):
            if value and value not in candidates:
                candidates.append(value)

    for candidate in candidates:
        normalized = SLICE_METHOD_ALIASES.get(candidate)
        if normalized is not None:
            return normalized

    for candidate in candidates:
        for normalized, keywords in SLICE_METHOD_KEYWORDS:
            if any(keyword in candidate for keyword in keywords):
                return normalized

    supported = ", ".join(SLICE_METHOD_CHOICES)
    raise ValueError(f"Unsupported slicing method: {method!r}. Supported values: {supported}")


def resolve_slice_bounds(
    min_seconds: Optional[float],
    max_seconds: Optional[float],
) -> Optional[tuple[float, float]]:
    if min_seconds is None and max_seconds is None:
        return None
    if min_seconds is None or max_seconds is None:
        raise ValueError("--min-seconds and --max-seconds must be provided together")

    min_seconds = float(min_seconds)
    max_seconds = float(max_seconds)
    if min_seconds < 0:
        raise ValueError("--min-seconds must be greater than or equal to 0")
    if max_seconds <= 0:
        raise ValueError("--max-seconds must be greater than 0")
    if min_seconds > max_seconds:
        raise ValueError("--min-seconds must be less than or equal to --max-seconds")
    return min_seconds, max_seconds


def collect_audio_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in INPUT_AUDIO_EXTENSIONS]
    return sorted(files)


def load_audio(path: Path, sr: int = DEFAULT_SAMPLE_RATE):
    ensure_ffmpeg_on_path()
    try:
        return librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        if path.suffix.lower() == ".m4a":
            ffmpeg_found = shutil.which("ffmpeg") is not None
            ffmpeg_status = "ffmpeg was found on PATH." if ffmpeg_found else "ffmpeg was not found on PATH."
            raise RuntimeError(
                f"Failed to read M4A file: {path}\n"
                f"{ffmpeg_status}\n"
                "Install FFmpeg and add it to PATH, or place ffmpeg.exe under _ffmpeg/bin/."
            ) from exc
        raise


def extract_text(result) -> str:
    if result is None:
        return ""

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return str(text_attr)

    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcript") or "")

    return str(result)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(audio_path: Path, source_md5: Optional[str] = None) -> str:
    md5 = source_md5 or file_md5(audio_path)
    return f"{safe_stem(audio_path)}_{md5[:8]}"


def free_runtime_memory() -> None:
    gc.collect()


def should_use_rmvpe_for_slicing(slicing_method: str, rmvpe_model_path: Optional[str]) -> bool:
    return normalize_slicing_method(slicing_method) == "smart" and bool(rmvpe_model_path)


def has_existing_outputs(
    audio_path: Path,
    output_dir: Path,
    source_md5: str,
    recursive_output: bool = True,
) -> bool:
    stem = source_key(audio_path, source_md5)
    lab_dir = output_dir / "labs"
    slice_dir = output_dir / "slices"
    if recursive_output:
        lab_dir = lab_dir / stem
        slice_dir = slice_dir / stem

    json_path = output_dir / "jsons" / f"{stem}.json"
    if json_path.is_file():
        return True
    if lab_dir.is_dir() and any(lab_dir.glob(f"{stem}_chunk*.lab")):
        return True
    if slice_dir.is_dir() and any(slice_dir.glob(f"{stem}_chunk*.wav")):
        return True
    return False


def source_index_path(output_dir: Path) -> Path:
    return output_dir / "jsons" / SOURCE_INDEX_NAME


def load_source_index(output_dir: Path) -> dict:
    path = source_index_path(output_dir)
    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def save_source_index(output_dir: Path, index: dict) -> None:
    path = source_index_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def index_has_completed_output(index: dict, source_md5: str, output_dir: Path) -> Optional[str]:
    record = index.get(source_md5)
    if not isinstance(record, dict):
        return None

    key = record.get("output_key")
    if not key:
        return None

    json_path = output_dir / "jsons" / f"{key}.json"
    lab_dir = output_dir / "labs" / key
    slice_dir = output_dir / "slices" / key
    if json_path.is_file():
        return str(key)
    if lab_dir.is_dir() and any(lab_dir.glob("*.lab")):
        return str(key)
    if slice_dir.is_dir() and any(slice_dir.glob("*.wav")):
        return str(key)
    return None


def update_source_index(
    index: dict,
    audio_path: Path,
    output_key: str,
    source_md5: str,
    chunks: int,
    labs: int,
) -> None:
    index[source_md5] = {
        "output_key": output_key,
        "source_name": audio_path.name,
        "source_path": str(audio_path.resolve()),
        "chunks": chunks,
        "labs": labs,
    }


def save_timestamps_json(
    json_dir: Path,
    source_stem: str,
    chunks,
    results,
    chunk_indices,
    sr: int,
    source_audio: Optional[Path] = None,
    source_md5: Optional[str] = None,
) -> Path:
    """Save slice offsets, durations, and ASR text to a JSON file."""
    json_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for result_index, result in enumerate(results):
        chunk_index = chunk_indices[result_index]
        chunk = chunks[chunk_index]
        offset = float(chunk.get("offset", 0.0))
        waveform = chunk["waveform"]
        duration = float(len(waveform) / sr)
        records.append(
            {
                "index": chunk_index,
                "offset": round(offset, 6),
                "duration": round(duration, 6),
                "text": extract_text(result).strip(),
            }
        )

    records.sort(key=lambda record: record["offset"])
    json_path = json_dir / f"{source_stem}.json"
    payload = {
        "source": {
            "path": str(source_audio.resolve()) if source_audio is not None else None,
            "md5": source_md5,
        },
        "chunks": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Timestamps saved: {json_path}")
    return json_path


def slice_audio_from_json(
    json_path: Path,
    source_audio: Path,
    output_dir: Path,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> int:
    """Re-slice an audio file using chunk offsets from a saved JSON file."""
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")
    if not source_audio.is_file():
        raise FileNotFoundError(f"Source audio file does not exist: {source_audio}")

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    records = loaded.get("chunks", []) if isinstance(loaded, dict) else loaded
    if not records:
        print(f"[SKIP] Empty JSON chunk list: {json_path}")
        return 0

    stem = safe_stem(source_audio)
    waveform, actual_sr = load_audio(source_audio, sr=sr)
    if waveform.size == 0:
        print(f"[SKIP] Empty audio: {source_audio}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        index = int(record["index"])
        offset = float(record["offset"])
        duration = float(record["duration"])
        text = str(record.get("text", ""))

        start_sample = max(0, int(round(offset * actual_sr)))
        end_sample = min(waveform.shape[-1], int(round((offset + duration) * actual_sr)))
        if start_sample >= end_sample:
            print(f"  [SKIP] chunk {index}: invalid range [{offset:.4f}s - {offset + duration:.4f}s]")
            continue

        chunk_wav = waveform[..., start_sample:end_sample]
        wav_name = f"{stem}_chunk{index:04d}_off{offset:08.2f}s_dur{duration:07.2f}s.wav"
        wav_path = output_dir / wav_name
        sf.write(wav_path, chunk_wav, actual_sr)
        written += 1

        if text:
            lab_name = f"{stem}_chunk{index:04d}_off{offset:08.2f}s.lab"
            (output_dir / lab_name).write_text(text, encoding="utf-8")

    print(f"  Sliced {written}/{len(records)} chunks from JSON timestamps -> {output_dir}")
    return written


def save_chunks(chunk_dir: Path, source_stem: str, chunks, sr: int):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, chunk in enumerate(chunks):
        offset = float(chunk.get("offset", 0.0))
        waveform = chunk["waveform"]
        duration = len(waveform) / sr
        name = f"{source_stem}_chunk{index:04d}_off{offset:08.2f}s_dur{duration:07.2f}s.wav"
        path = chunk_dir / name
        sf.write(path, waveform, sr)
        saved_paths.append(path)
    return saved_paths


def run_slicer(
    waveform,
    sr: int,
    slicing_method: str,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
    rmvpe_voiced_mask=None,
    rmvpe_time_step_seconds: Optional[float] = None,
):
    normalized_method = normalize_slicing_method(slicing_method)
    if min_seconds is not None or max_seconds is not None:
        if slice_audio_with_custom_bounds is None:
            raise RuntimeError("Custom slice bounds are not supported by this build of slicer_api.")
        return slice_audio_with_custom_bounds(
            waveform,
            sr,
            normalized_method,
            min_len_sec=min_seconds,
            max_len_sec=max_seconds,
            rmvpe_voiced_mask=rmvpe_voiced_mask,
            rmvpe_time_step_seconds=rmvpe_time_step_seconds,
        )

    return slice_audio(
        waveform,
        sr,
        normalized_method,
        rmvpe_voiced_mask=rmvpe_voiced_mask,
        rmvpe_time_step_seconds=rmvpe_time_step_seconds,
    )


def process_one_file(
    audio_path: Path,
    output_dir: Path,
    asr_model_path: str,
    device: str,
    language: str,
    slicing_method: str,
    asr_batch_size: int,
    recursive_output: bool = True,
    save_json: bool = False,
    no_slice: bool = False,
    asr_model=None,
    rmvpe_model_path: Optional[str] = None,
    rmvpe_batch_size: int = 8,
    rmvpe_model: Optional[RmvpeTranscriber] = None,
    source_md5: Optional[str] = None,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
):
    """Process one audio file: slice -> ASR -> chunk WAVs / lab files / JSON."""
    output_stem = source_key(audio_path, source_md5)
    wav_out_dir = output_dir / "slices"
    lab_out_dir = output_dir / "labs"
    json_out_dir = output_dir / "jsons"
    if recursive_output:
        wav_out_dir = wav_out_dir / output_stem
        lab_out_dir = lab_out_dir / output_stem

    waveform, sr = load_audio(audio_path, sr=DEFAULT_SAMPLE_RATE)
    if waveform.size == 0:
        print(f"[SKIP] Empty audio: {audio_path}")
        return 0, 0

    normalized_method = normalize_slicing_method(slicing_method)
    print(f"\n[FILE] {audio_path.name}")

    if no_slice:
        print("  [NO-SLICE] Skipping slicer; using the full audio as a single chunk.")
        chunks = [{"offset": 0.0, "waveform": waveform}]
    else:
        rmvpe_voiced_mask = None
        rmvpe_step = None
        if should_use_rmvpe_for_slicing(normalized_method, rmvpe_model_path):
            own_rmvpe_model = rmvpe_model is None
            if own_rmvpe_model:
                print(f"  [RMVPE] Loading model for smart slicing: {rmvpe_model_path}")
                rmvpe_model = RmvpeTranscriber(rmvpe_model_path, device=device, batch_size=rmvpe_batch_size)
            try:
                print("  [RMVPE] Running voiced/unvoiced detection for smart slicing...")
                rmvpe_result = rmvpe_model.infer(waveform, sr)
                rmvpe_voiced_mask = rmvpe_result.voiced_mask
                rmvpe_step = rmvpe_result.time_step_seconds
                print(f"  [RMVPE] Done. Frames={len(rmvpe_result.midi_pitch)} step={rmvpe_step:.4f}s")
            finally:
                if own_rmvpe_model:
                    del rmvpe_model
                    free_runtime_memory()

        chunks = run_slicer(
            waveform=waveform,
            sr=sr,
            slicing_method=normalized_method,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            rmvpe_voiced_mask=rmvpe_voiced_mask,
            rmvpe_time_step_seconds=rmvpe_step,
        )

    if not chunks:
        print("  No chunks generated, skipping.")
        return 0, 0

    with tempfile.TemporaryDirectory(prefix=f"vocal2midi_{output_stem}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        save_chunks(wav_out_dir, output_stem, chunks, sr)

        if asr_model is not None:
            results, chunk_indices = batch_transcribe_asr(
                chunks=chunks,
                sr=sr,
                asr_model=asr_model,
                temp_dir_path=tmp_path,
                asr_batch_size=asr_batch_size,
                language=language,
                cancel_checker=None,
                device=device,
                force_subprocess=False,
                asr_timeout_sec=180,
            )
        else:
            results, chunk_indices = batch_transcribe_asr(
                chunks=chunks,
                sr=sr,
                asr_model=None,
                temp_dir_path=tmp_path,
                asr_batch_size=asr_batch_size,
                language=language,
                cancel_checker=None,
                asr_model_path=asr_model_path,
                device=device,
                force_subprocess=True,
                asr_timeout_sec=180,
            )

        lab_out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for result_index, result in enumerate(results):
            chunk_index = chunk_indices[result_index]
            chunk = chunks[chunk_index]
            offset = float(chunk.get("offset", 0.0))
            lab_text = extract_text(result).strip()
            lab_name = f"{output_stem}_chunk{chunk_index:04d}_off{offset:08.2f}s.lab"
            (lab_out_dir / lab_name).write_text(lab_text, encoding="utf-8")
            written += 1

        if save_json:
            save_timestamps_json(
                json_out_dir,
                output_stem,
                chunks,
                results,
                chunk_indices,
                sr,
                source_audio=audio_path,
                source_md5=source_md5,
            )

    print(f"  chunks: {len(chunks)}, labs: {written}")
    return len(chunks), written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch slice audio files and run the local Qwen3-ASR ONNX runtime."
    )
    parser.add_argument("input_dir", type=Path, help="Input folder containing audio files")
    parser.add_argument("output_dir", type=Path, help="Output folder for slices and lab files")
    parser.add_argument("--asr-model", required=True, help="Local Qwen3-ASR DML model directory")
    parser.add_argument(
        "--device",
        default=DEFAULT_RUNTIME_DEVICE,
        choices=list(RUNTIME_DEVICE_CHOICES),
        help="Runtime device. Legacy 'cuda' is accepted and mapped to 'dml'.",
    )
    parser.add_argument("--language", default="zh", choices=["zh", "ja"], help="ASR language")
    parser.add_argument(
        "--slicing-method",
        default=DEFAULT_SLICE_METHOD,
        help=(
            "Slicing strategy. Supported values: default, smart, heuristic, grid. "
            "Chinese labels and repaired legacy mojibake values are also accepted."
        ),
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=None,
        help="Optional minimum target chunk duration in seconds. Must be paired with --max-seconds.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional maximum target chunk duration in seconds. Must be paired with --min-seconds.",
    )
    parser.add_argument(
        "--no-slice",
        action="store_true",
        help="Bypass slicing and send the whole file to ASR as a single chunk.",
    )
    parser.add_argument("--asr-batch-size", type=int, default=4, help="ASR batch size")
    parser.add_argument(
        "--rmvpe-model",
        default=str(DEFAULT_RMVPE_MODEL),
        help="RMVPE model path used by smart slicing. Pass an empty string to disable it.",
    )
    parser.add_argument("--rmvpe-batch-size", type=int, default=8, help="RMVPE batch size")
    parser.add_argument("--file-batch-size", type=int, default=1, help="Number of audio files to process per batch")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top level of the input directory")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess files even if outputs already exist")
    parser.add_argument("--save-json", action="store_true", help="Save slice timing and ASR outputs as JSON")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Slice by an existing JSON timing file together with --source-audio and --output-dir.",
    )
    parser.add_argument(
        "--source-audio",
        type=Path,
        default=None,
        help="Source audio file used with --from-json.",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Reuse the ASR runtime in the current process across the whole batch.",
    )
    parser.add_argument(
        "--keep-rmvpe",
        action="store_true",
        help="Reuse the RMVPE runtime in the current process during smart slicing.",
    )
    return parser


def validate_args(args) -> None:
    args.device = normalize_runtime_device(args.device)
    args.slicing_method = normalize_slicing_method(args.slicing_method)
    resolve_slice_bounds(args.min_seconds, args.max_seconds)

    if args.file_batch_size <= 0:
        raise ValueError("--file-batch-size must be greater than 0")
    if args.asr_batch_size <= 0:
        raise ValueError("--asr-batch-size must be greater than 0")
    if args.rmvpe_batch_size <= 0:
        raise ValueError("--rmvpe-batch-size must be greater than 0")


def main():
    args = build_argparser().parse_args()
    ensure_ffmpeg_on_path()
    validate_args(args)

    if args.from_json is not None:
        if args.source_audio is None:
            raise ValueError("--from-json requires --source-audio")

        written = slice_audio_from_json(
            json_path=args.from_json.resolve(),
            source_audio=args.source_audio.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(f"\nDone. Sliced {written} chunks from JSON.")
        return 0

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    rmvpe_model_path = str(args.rmvpe_model).strip()
    slice_bounds = resolve_slice_bounds(args.min_seconds, args.max_seconds)
    min_seconds = slice_bounds[0] if slice_bounds is not None else None
    max_seconds = slice_bounds[1] if slice_bounds is not None else None

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    audio_files = collect_audio_files(input_dir, recursive=not args.no_recursive)
    if not audio_files:
        exts = ", ".join(sorted(INPUT_AUDIO_EXTENSIONS))
        print(f"No audio files found ({exts}) in: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    source_index = load_source_index(output_dir)
    total_chunks = 0
    total_labs = 0
    skipped_existing = 0
    skipped_failed = 0

    use_rmvpe_for_slicing = should_use_rmvpe_for_slicing(args.slicing_method, rmvpe_model_path)
    if args.keep_rmvpe and args.no_slice:
        print("[Keep-RMVPE] Ignored: --no-slice bypasses slicing entirely.")
    elif args.keep_rmvpe and not use_rmvpe_for_slicing:
        print("[Keep-RMVPE] Ignored: RMVPE is only used with --slicing-method smart and a non-empty --rmvpe-model.")
    if args.no_slice and slice_bounds is not None:
        print("[Slice Bounds] Ignored: --no-slice sends the full file directly to ASR.")

    asr_model = None
    rmvpe_model = None
    if args.keep_model:
        print(f"[Keep-Model] Loading ASR runtime once for all files: {args.asr_model}")
        asr_model = load_qwen_model(args.asr_model, args.device, use_cache=False)
        print("[Keep-Model] Runtime loaded. It will be reused for the entire batch.")
    if args.keep_rmvpe and use_rmvpe_for_slicing:
        print(f"[Keep-RMVPE] Loading RMVPE runtime once for all files: {rmvpe_model_path}")
        rmvpe_model = RmvpeTranscriber(rmvpe_model_path, device=args.device, batch_size=args.rmvpe_batch_size)
        print("[Keep-RMVPE] Runtime loaded. It will be reused for the entire batch.")

    try:
        total_batches = math.ceil(len(audio_files) / args.file_batch_size)
        print(f"Found {len(audio_files)} audio files. Processing in file batches of {args.file_batch_size}...")
        for batch_no, file_batch in enumerate(batch_iter(audio_files, args.file_batch_size), start=1):
            print(f"\n=== File batch {batch_no} / {total_batches} ===")
            for audio_path in file_batch:
                try:
                    source_md5 = file_md5(audio_path)
                except Exception as exc:
                    skipped_failed += 1
                    print(f"\n[SKIP failed] {audio_path}")
                    print(f"  MD5 {type(exc).__name__}: {exc}")
                    continue

                output_key = source_key(audio_path, source_md5)
                if not args.no_skip_existing:
                    indexed_key = index_has_completed_output(source_index, source_md5, output_dir)
                    if indexed_key is not None:
                        skipped_existing += 1
                        print(f"\n[SKIP existing] {audio_path.name} -> {indexed_key} (md5 index)")
                        continue
                    if has_existing_outputs(audio_path, output_dir, source_md5):
                        skipped_existing += 1
                        print(f"\n[SKIP existing] {audio_path.name} -> {output_key}")
                        continue

                try:
                    chunks, labs = process_one_file(
                        audio_path=audio_path,
                        output_dir=output_dir,
                        asr_model_path=args.asr_model,
                        device=args.device,
                        language=args.language,
                        slicing_method=args.slicing_method,
                        asr_batch_size=args.asr_batch_size,
                        save_json=args.save_json,
                        no_slice=args.no_slice,
                        asr_model=asr_model,
                        rmvpe_model_path=rmvpe_model_path,
                        rmvpe_batch_size=args.rmvpe_batch_size,
                        rmvpe_model=rmvpe_model,
                        source_md5=source_md5,
                        min_seconds=min_seconds,
                        max_seconds=max_seconds,
                    )
                except Exception as exc:
                    skipped_failed += 1
                    print(f"\n[SKIP failed] {audio_path}")
                    print(f"  {type(exc).__name__}: {exc}")
                    continue

                update_source_index(source_index, audio_path, output_key, source_md5, chunks, labs)
                save_source_index(output_dir, source_index)
                total_chunks += chunks
                total_labs += labs
    finally:
        if rmvpe_model is not None:
            print("\n[Keep-RMVPE] Releasing cached RMVPE runtime...")
            del rmvpe_model
            free_runtime_memory()
        if asr_model is not None:
            print("\n[Keep-Model] Releasing cached ASR runtime...")
            clear_qwen_model_cache()
            del asr_model
            free_runtime_memory()

    print(
        f"\nDone. Total chunks: {total_chunks}, total labs: {total_labs}, "
        f"skipped existing: {skipped_existing}, skipped failed: {skipped_failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
