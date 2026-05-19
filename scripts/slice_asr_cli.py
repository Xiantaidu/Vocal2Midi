"""Batch slice + Qwen3-ASR CLI.

输入一个包含 .wav / .m4a 的目录，先切片，再对切片做 Qwen3-ASR，
最后把切片 wav 和对应的 .lab 纯文本输出到指定目录。
"""

from __future__ import annotations

import argparse
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


DEFAULT_RMVPE_MODEL = ROOT_DIR / "experiments" / "RMVPE" / "rmvpe.pt"
INPUT_AUDIO_EXTENSIONS = (".wav", ".m4a")


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
            ffmpeg_status = "当前 PATH 中未找到 ffmpeg。" if shutil.which("ffmpeg") is None else "当前 PATH 中已找到 ffmpeg。"
            raise RuntimeError(
                f"读取 M4A 失败: {path}\n"
                f"{ffmpeg_status}\n"
                "请安装 FFmpeg 并加入 PATH，或把 ffmpeg.exe 放到项目的 _ffmpeg/bin/ 目录。"
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


def free_torch_memory():
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def should_use_rmvpe_for_slicing(slicing_method: str, rmvpe_model_path: Optional[str]) -> bool:
    return slicing_method == "智能切片" and bool(rmvpe_model_path)


def has_existing_outputs(audio_path: Path, output_dir: Path, recursive_output: bool = True) -> bool:
    stem = safe_stem(audio_path)
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


def save_timestamps_json(json_dir: Path, source_stem: str, chunks, results, chunk_indices, sr: int):
    """将切片的精确时间戳和 ASR 转写结果保存为 JSON 文件。

    每个 chunk 的记录包含：
        - index: 切片序号
        - offset: 切片在原始音频中的起始时间 (秒)
        - duration: 切片时长 (秒)
        - text: ASR 转写文本（lab 内容）
    """
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
    # 按 offset 排序保证时间顺序
    records.sort(key=lambda r: r["offset"])
    json_path = json_dir / f"{source_stem}.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Timestamps saved: {json_path}")
    return json_path


def slice_audio_from_json(
    json_path: Path,
    source_audio: Path,
    output_dir: Path,
    sr: int = 44100,
) -> int:
    """根据 JSON 中记录的精确时间戳对音频进行切分。

    JSON 格式应为 save_timestamps_json 输出的格式：
        [{"index": 0, "offset": 0.0, "duration": 5.0, "text": "..."}, ...]

    输出命名规则与原切片一致：
        {stem}_chunk{index:04d}_off{offset:08.2f}s_dur{duration:07.2f}s.wav
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
    if not source_audio.is_file():
        raise FileNotFoundError(f"音频文件不存在: {source_audio}")

    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not records:
        print(f"[SKIP] JSON 文件为空: {json_path}")
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
        # 裁剪超出音频长度的部分
        end_sample = min(end_sample, waveform.shape[-1])
        if start_sample >= end_sample:
            print(f"  [SKIP] chunk {idx}: invalid range [{offset:.4f}s - {offset + dur:.4f}s]")
            continue

        chunk_wav = waveform[..., start_sample:end_sample]
        name = f"{stem}_chunk{idx:04d}_off{offset:08.2f}s_dur{dur:07.2f}s.wav"
        out_path = output_dir / name
        sf.write(out_path, chunk_wav, actual_sr)
        written += 1

        # 同时输出对应的 lab 文件
        if text:
            lab_name = f"{stem}_chunk{idx:04d}_off{offset:08.2f}s.lab"
            (output_dir / lab_name).write_text(text, encoding="utf-8")

    print(f"  Sliced {written}/{len(records)} chunks from JSON timestamps → {output_dir}")
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
    asr_model=None,
    rmvpe_model_path: Optional[str] = None,
    rmvpe_batch_size: int = 8,
    rmvpe_model: Optional[RmvpeTranscriber] = None,
):
    """处理单个音频文件：切片 → ASR → 输出 .wav + .lab。

    当 asr_model 不为 None 时，直接在主进程中使用持久化模型进行转写（跳过子进程），
    避免每个文件重复加载模型到 VRAM。
    """
    wav_out_dir = output_dir / "slices"
    lab_out_dir = output_dir / "labs"
    json_out_dir = output_dir / "jsons"
    if recursive_output:
        wav_out_dir = wav_out_dir / safe_stem(audio_path)
        lab_out_dir = lab_out_dir / safe_stem(audio_path)

    sr = 44100
    waveform, sr = load_audio(audio_path, sr=sr)
    if waveform.size == 0:
        print(f"[SKIP] Empty audio: {audio_path}")
        return 0, 0

    print(f"\n[FILE] {audio_path.name}")

    rmvpe_voiced_mask = None
    rmvpe_step = None
    if should_use_rmvpe_for_slicing(slicing_method, rmvpe_model_path):
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
                free_torch_memory()

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

    with tempfile.TemporaryDirectory(prefix=f"vocal2midi_{safe_stem(audio_path)}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        save_chunks(wav_out_dir, safe_stem(audio_path), chunks, sr)

        if asr_model is not None:
            # 持久化模型模式：主进程内直接推理，避免反复加载/卸载模型
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
            # 子进程模式：每个文件创建一个新的子进程加载模型
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
            lab_name = f"{safe_stem(audio_path)}_chunk{chunk_idx:04d}_off{offset:08.2f}s.lab"
            (lab_out_dir / lab_name).write_text(lab_text, encoding="utf-8")
            written += 1

        # 保存精确时间戳 JSON
        if save_json:
            save_timestamps_json(json_out_dir, safe_stem(audio_path), chunks, results, chunk_indices, sr)

    print(f"  chunks: {len(chunks)}, labs: {written}")
    return len(chunks), written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch slice WAV/M4A files and run Qwen3-ASR to export .wav + .lab outputs."
    )
    parser.add_argument("input_dir", type=Path, help="包含 .wav / .m4a 的输入文件夹")
    parser.add_argument("output_dir", type=Path, help="切片与 .lab 输出目录")
    parser.add_argument("--asr-model", required=True, help="Qwen3-ASR 模型路径或 HuggingFace ID")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="ASR 设备")
    parser.add_argument("--language", default="zh", choices=["zh", "ja"], help="ASR 语言")
    parser.add_argument(
        "--slicing-method",
        default="默认切片",
        choices=["默认切片", "智能切片", "启发式切片", "网格搜索切片"],
        help="切片策略；默认切片 = default_slice()（Slicer(threshold=-30, min_length=5000, max_sil_kept=500)）",
    )
    parser.add_argument("--asr-batch-size", type=int, default=4, help="ASR 批大小")
    parser.add_argument(
        "--rmvpe-model",
        default=str(DEFAULT_RMVPE_MODEL),
        help="RMVPE 模型路径；智能切片时用于有声/无声检测。传空字符串可关闭 RMVPE 辅助切片",
    )
    parser.add_argument("--rmvpe-batch-size", type=int, default=8, help="RMVPE 推理批大小")
    parser.add_argument("--file-batch-size", type=int, default=1, help="文件批大小（按文件分批处理）")
    parser.add_argument("--no-recursive", action="store_true", help="只扫描输入目录第一层音频文件")
    parser.add_argument("--no-skip-existing", action="store_true", help="不跳过已有输出，强制重新处理")
    parser.add_argument("--save-json", action="store_true", help="保存每个文件的切片时间戳和 ASR 结果为 JSON")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="从 JSON 时间戳文件对音频进行切分（需配合 --source-audio 和 --output-dir）",
    )
    parser.add_argument(
        "--source-audio",
        type=Path,
        default=None,
        help="配合 --from-json 使用的源音频文件路径",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="ASR 模型持久驻留 VRAM，避免每个文件重复加载（显著节省批量处理时间）",
    )
    parser.add_argument(
        "--keep-rmvpe",
        action="store_true",
        help="智能切片时让 RMVPE 模型持久驻留 VRAM，避免每个文件重复加载",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    ensure_ffmpeg_on_path()

    # ---- 从 JSON 切分的独立模式 ----
    if args.from_json is not None:
        if args.source_audio is None:
            raise ValueError("--from-json 模式需要同时指定 --source-audio")
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
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    audio_files = collect_audio_files(input_dir, recursive=not args.no_recursive)
    if not audio_files:
        exts = ", ".join(INPUT_AUDIO_EXTENSIONS)
        print(f"No audio files found ({exts}) in: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    total_labs = 0
    skipped_existing = 0
    skipped_failed = 0

    use_rmvpe_for_slicing = should_use_rmvpe_for_slicing(args.slicing_method, rmvpe_model_path)
    if args.keep_rmvpe and not use_rmvpe_for_slicing:
        print("[Keep-RMVPE] Ignored: RMVPE is only used with --slicing-method 智能切片 and a non-empty --rmvpe-model.")

    # 持久化模型：主进程中一次性加载，所有文件共享
    asr_model = None
    rmvpe_model = None
    if args.keep_model:
        print(f"[Keep-Model] Loading ASR model into VRAM once for all files: {args.asr_model}")
        asr_model = load_qwen_model(args.asr_model, args.device, use_cache=False)
        print("[Keep-Model] Model loaded. Will remain in VRAM for the entire batch.")
    if args.keep_rmvpe and use_rmvpe_for_slicing:
        print(f"[Keep-RMVPE] Loading RMVPE model into VRAM once for all files: {rmvpe_model_path}")
        rmvpe_model = RmvpeTranscriber(rmvpe_model_path, device=args.device, batch_size=args.rmvpe_batch_size)
        print("[Keep-RMVPE] Model loaded. Will remain in VRAM for the entire batch.")

    try:
        print(f"Found {len(audio_files)} audio files. Processing in file batches of {args.file_batch_size}...")
        for batch_no, file_batch in enumerate(batch_iter(audio_files, args.file_batch_size), start=1):
            print(f"\n=== File batch {batch_no} / {(len(audio_files) + args.file_batch_size - 1) // args.file_batch_size} ===")
            for audio_path in file_batch:
                if not args.no_skip_existing and has_existing_outputs(audio_path, output_dir):
                    skipped_existing += 1
                    print(f"\n[SKIP existing] {audio_path.name}")
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
                        asr_model=asr_model,
                        rmvpe_model_path=rmvpe_model_path,
                        rmvpe_batch_size=args.rmvpe_batch_size,
                        rmvpe_model=rmvpe_model,
                    )
                except Exception as exc:
                    skipped_failed += 1
                    print(f"\n[SKIP failed] {audio_path}")
                    print(f"  {type(exc).__name__}: {exc}")
                    continue
                total_chunks += chunks
                total_labs += labs
    finally:
        if rmvpe_model is not None:
            print("\n[Keep-RMVPE] Releasing RMVPE model from VRAM...")
            del rmvpe_model
            free_torch_memory()
        if asr_model is not None:
            print("\n[Keep-Model] Releasing ASR model from VRAM...")
            clear_qwen_model_cache()
            del asr_model

    print(
        f"\nDone. Total chunks: {total_chunks}, total labs: {total_labs}, "
        f"skipped existing: {skipped_existing}, skipped failed: {skipped_failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
