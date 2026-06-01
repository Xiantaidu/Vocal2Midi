"""Batch slice + Qwen3-ASR CLI.

杈撳叆涓€涓寘鍚?.wav / .m4a 鐨勭洰褰曪紝榛樿鍏堝垏鐗囷紝鍐嶅鍒囩墖鍋?Qwen3-ASR锛?鏈€鍚庢妸鍒囩墖 wav 鍜屽搴旂殑 .lab 绾枃鏈緭鍑哄埌鎸囧畾鐩綍銆?
鍔犱笂 --no-slice 鏃讹紝浼氱洿鎺ユ妸鏁存闊抽褰撴垚涓€涓?chunk 閫佸叆 ASR銆?"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from inference.API.asr_api import batch_transcribe_asr, load_qwen_model, clear_qwen_model_cache
from inference.API.rmvpe_api import RmvpeTranscriber
from inference.API.slicer_api import slice_audio
from inference.device_utils import RUNTIME_DEVICE_CHOICES, normalize_runtime_device


DEFAULT_RMVPE_MODEL = ROOT_DIR / "experiments" / "RMVPE" / "rmvpe.onnx"
INPUT_AUDIO_EXTENSIONS = (".wav", ".m4a", ".mp3")
SOURCE_INDEX_NAME = "_source_index.json"


def batch_iter(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_ffmpeg_on_path() -> None:
    ffmpeg_bin = ROOT_DIR / "_ffmpeg" / "bin"
    if ffmpeg_bin.is_dir():
        current_path = os.environ.get("PATH", "")
        parts = current_path.split(os.pathsep) if current_path else []
        ffmpeg_bin_str = str(ffmpeg_bin)
        if ffmpeg_bin_str not in parts:
            os.environ["PATH"] = ffmpeg_bin_str + (os.pathsep + current_path if current_path else "")


def collect_audio_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    patterns = []
    for ext in INPUT_AUDIO_EXTENSIONS:
        patterns.append(f"**/*{ext}" if recursive else f"*{ext}")
        patterns.append(f"**/*{ext.upper()}" if recursive else f"*{ext.upper()}")

    files = {}
    for pattern in patterns:
        for p in input_dir.glob(pattern):
            if p.is_file():
                files[str(p.resolve())] = p
    return sorted(files.values())


def load_audio(path: Path, sr: int = 44100):
    ensure_ffmpeg_on_path()
    try:
        return librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        if path.suffix.lower() == ".m4a":
            ffmpeg_status = "ffmpeg was not found on PATH." if shutil.which("ffmpeg") is None else "ffmpeg was found on PATH."
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
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(audio_path: Path, source_md5: Optional[str] = None) -> str:
    md5 = source_md5 or file_md5(audio_path)
    return f"{safe_stem(audio_path)}_{md5[:8]}"


def free_runtime_memory():
    import gc

    gc.collect()


def should_use_rmvpe_for_slicing(slicing_method: str, rmvpe_model_path: Optional[str]) -> bool:
    return slicing_method == "鏅鸿兘鍒囩墖" and bool(rmvpe_model_path)


def has_existing_outputs(audio_path: Path, output_dir: Path, source_md5: str, recursive_output: bool = True) -> bool:
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
    rec = index.get(source_md5)
    if not isinstance(rec, dict):
        return None
    key = rec.get("output_key")
    if not key:
        return None
    json_path = output_dir / "jsons" / f"{key}.json"
    lab_dir = output_dir / "labs" / key
    slice_dir = output_dir / "slices" / key
    if json_path.is_file() or (lab_dir.is_dir() and any(lab_dir.glob("*.lab"))) or (slice_dir.is_dir() and any(slice_dir.glob("*.wav"))):
        return str(key)
    return None


def update_source_index(index: dict, audio_path: Path, output_key: str, source_md5: str, chunks: int, labs: int) -> None:
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
):
    """灏嗗垏鐗囩殑绮剧‘鏃堕棿鎴冲拰 ASR 杞啓缁撴灉淇濆瓨涓?JSON 鏂囦欢銆?
    姣忎釜 chunk 鐨勮褰曞寘鍚細
        - index: 鍒囩墖搴忓彿
        - offset: 鍒囩墖鍦ㄥ師濮嬮煶棰戜腑鐨勮捣濮嬫椂闂?(绉?
        - duration: 鍒囩墖鏃堕暱 (绉?
        - text: ASR 杞啓鏂囨湰锛坙ab 鍐呭锛?    """
    json_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, res in enumerate(results):
        chunk_idx = chunk_indices[idx]
        chunk = chunks[chunk_idx]
        offset = float(chunk.get("offset", 0.0))
        wav = chunk["waveform"]
        dur = float(len(wav) / sr)
        text = extract_text(res).strip()
        records.append({
            "index": chunk_idx,
            "offset": round(offset, 6),
            "duration": round(dur, 6),
            "text": text,
        })
    # 鎸?offset 鎺掑簭淇濊瘉鏃堕棿椤哄簭
    records.sort(key=lambda r: r["offset"])
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
    sr: int = 44100,
) -> int:
    """鏍规嵁 JSON 涓褰曠殑绮剧‘鏃堕棿鎴冲闊抽杩涜鍒囧垎銆?
    JSON 鏍煎紡搴斾负 save_timestamps_json 杈撳嚭鐨勬牸寮忥細
        [{"index": 0, "offset": 0.0, "duration": 5.0, "text": "..."}, ...]

    杈撳嚭鍛藉悕瑙勫垯涓庡師鍒囩墖涓€鑷达細
        {stem}_chunk{index:04d}_off{offset:08.2f}s_dur{duration:07.2f}s.wav
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON 鏂囦欢涓嶅瓨鍦? {json_path}")
    if not source_audio.is_file():
        raise FileNotFoundError(f"闊抽鏂囦欢涓嶅瓨鍦? {source_audio}")

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    records = loaded.get("chunks", []) if isinstance(loaded, dict) else loaded
    if not records:
        print(f"[SKIP] JSON 鏂囦欢涓虹┖: {json_path}")
        return 0

    stem = safe_stem(source_audio)
    waveform, actual_sr = load_audio(source_audio, sr=sr)
    if waveform.size == 0:
        print(f"[SKIP] Empty audio: {source_audio}")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for rec in records:
        idx = rec["index"]
        offset = rec["offset"]
        dur = rec["duration"]
        text = rec.get("text", "")

        start_sample = int(offset * actual_sr)
        end_sample = int((offset + dur) * actual_sr)
        # 瑁佸壀瓒呭嚭闊抽闀垮害鐨勯儴鍒?        end_sample = min(end_sample, waveform.shape[-1])
        if start_sample >= end_sample:
            print(f"  [SKIP] chunk {idx}: invalid range [{offset:.4f}s - {offset + dur:.4f}s]")
            continue

        chunk_wav = waveform[..., start_sample:end_sample]
        name = f"{stem}_chunk{idx:04d}_off{offset:08.2f}s_dur{dur:07.2f}s.wav"
        out_path = output_dir / name
        sf.write(out_path, chunk_wav, actual_sr)
        written += 1

        # 鍚屾椂杈撳嚭瀵瑰簲鐨?lab 鏂囦欢
        if text:
            lab_name = f"{stem}_chunk{idx:04d}_off{offset:08.2f}s.lab"
            (output_dir / lab_name).write_text(text, encoding="utf-8")

    print(f"  Sliced {written}/{len(records)} chunks from JSON timestamps 鈫?{output_dir}")
    return written


def save_chunks(chunk_dir: Path, source_stem: str, chunks, sr: int):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, chunk in enumerate(chunks):
        offset = float(chunk.get("offset", 0.0))
        wav = chunk["waveform"]
        dur = len(wav) / sr
        name = f"{source_stem}_chunk{idx:04d}_off{offset:08.2f}s_dur{dur:07.2f}s.wav"
        out_path = chunk_dir / name
        sf.write(out_path, wav, sr)
        saved.append(out_path)
    return saved


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
):
    """澶勭悊鍗曚釜闊抽鏂囦欢锛氬垏鐗?鈫?ASR 鈫?杈撳嚭 .wav + .lab銆?
    褰?asr_model 涓嶄负 None 鏃讹紝鐩存帴鍦ㄤ富杩涚▼涓娇鐢ㄦ寔涔呭寲 ASR 杩愯鏃惰繘琛岃浆鍐欙紙璺宠繃瀛愯繘绋嬶級锛?    閬垮厤姣忎釜鏂囦欢閲嶅鍒濆鍖?Qwen3 DML runtime銆?    """
    output_stem = source_key(audio_path, source_md5)
    wav_out_dir = output_dir / "slices"
    lab_out_dir = output_dir / "labs"
    json_out_dir = output_dir / "jsons"
    if recursive_output:
        wav_out_dir = wav_out_dir / output_stem
        lab_out_dir = lab_out_dir / output_stem

    sr = 44100
    waveform, sr = load_audio(audio_path, sr=sr)
    if waveform.size == 0:
        print(f"[SKIP] Empty audio: {audio_path}")
        return 0, 0

    print(f"\n[FILE] {audio_path.name}")

    rmvpe_voiced_mask = None
    rmvpe_step = None
    if no_slice:
        print("  [NO-SLICE] Skipping slicer; using the full audio as a single chunk.")
        chunks = [{"offset": 0.0, "waveform": waveform}]
    elif should_use_rmvpe_for_slicing(slicing_method, rmvpe_model_path):
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

        chunks = slice_audio(
            waveform,
            sr,
            slicing_method,
            rmvpe_voiced_mask=rmvpe_voiced_mask,
            rmvpe_time_step_seconds=rmvpe_step,
        )
    else:
        chunks = slice_audio(
            waveform,
            sr,
            slicing_method,
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
            # 鎸佷箙鍖栨ā鍨嬫ā寮忥細涓昏繘绋嬪唴鐩存帴鎺ㄧ悊锛岄伩鍏嶅弽澶嶅姞杞?鍗歌浇妯″瀷
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
            # Subprocess mode: create a fresh ASR runtime for this file.
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
        for idx, res in enumerate(results):
            chunk_idx = chunk_indices[idx]
            chunk = chunks[chunk_idx]
            offset = float(chunk.get("offset", 0.0))
            lab_text = extract_text(res).strip()
            lab_name = f"{output_stem}_chunk{chunk_idx:04d}_off{offset:08.2f}s.lab"
            (lab_out_dir / lab_name).write_text(lab_text, encoding="utf-8")
            written += 1

        # 淇濆瓨绮剧‘鏃堕棿鎴?JSON
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
        default="dml",
        choices=list(RUNTIME_DEVICE_CHOICES),
        help="Runtime device. Legacy 'cuda' is accepted and mapped to 'dml'.",
    )
    parser.add_argument("--language", default="zh", choices=["zh", "ja"], help="ASR language")
    parser.add_argument(
        "--slicing-method",
        default="榛樿鍒囩墖",
        choices=["榛樿鍒囩墖", "鏅鸿兘鍒囩墖", "鍚彂寮忓垏鐗?", "缃戞牸鎼滅储鍒囩墖"],
        help="Slicing strategy. Ignored when --no-slice is enabled.",
    )
    parser.add_argument(
        "--no-slice",
        action="store_true",
        help="Bypass slicing and send the whole file to ASR as a single chunk",
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
        help="Slice by an existing JSON timing file together with --source-audio and --output-dir",
    )
    parser.add_argument(
        "--source-audio",
        type=Path,
        default=None,
        help="Source audio file used with --from-json",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Reuse the ASR runtime in the current process across the whole batch",
    )
    parser.add_argument(
        "--keep-rmvpe",
        action="store_true",
        help="Reuse the RMVPE runtime in the current process during smart slicing",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    ensure_ffmpeg_on_path()
    args.device = normalize_runtime_device(args.device)

    # ---- 浠?JSON 鍒囧垎鐨勭嫭绔嬫ā寮?----
    if args.from_json is not None:
        if args.source_audio is None:
            raise ValueError("--from-json 妯″紡闇€瑕佸悓鏃舵寚瀹?--source-audio")
        json_path = args.from_json.resolve()
        source_audio = args.source_audio.resolve()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        written = slice_audio_from_json(
            json_path=json_path,
            source_audio=source_audio,
            output_dir=output_dir,
        )
        print(f"\nDone. Sliced {written} chunks from JSON.")
        return 0

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    rmvpe_model_path = str(args.rmvpe_model).strip()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"杈撳叆鐩綍涓嶅瓨鍦? {input_dir}")

    audio_files = collect_audio_files(input_dir, recursive=not args.no_recursive)
    if not audio_files:
        exts = ", ".join(INPUT_AUDIO_EXTENSIONS)
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
        print("[Keep-RMVPE] Ignored: RMVPE is only used with --slicing-method 鏅鸿兘鍒囩墖 and a non-empty --rmvpe-model.")

    # 鎸佷箙鍖栨ā鍨嬶細涓昏繘绋嬩腑涓€娆℃€у姞杞斤紝鎵€鏈夋枃浠跺叡浜?    asr_model = None
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
        print(f"Found {len(audio_files)} audio files. Processing in file batches of {args.file_batch_size}...")
        for batch_no, file_batch in enumerate(batch_iter(audio_files, args.file_batch_size), start=1):
            print(f"\n=== File batch {batch_no} / {(len(audio_files) + args.file_batch_size - 1) // args.file_batch_size} ===")
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

    print(
        f"\nDone. Total chunks: {total_chunks}, total labs: {total_labs}, "
        f"skipped existing: {skipped_existing}, skipped failed: {skipped_failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



