"""Batch slice + Qwen3-ASR CLI.

输入一个包含 .wav 的目录，先切片，再对切片做 Qwen3-ASR，
最后把切片 wav 和对应的 .lab 纯文本输出到指定目录。
"""

from __future__ import annotations

import argparse
import json
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
from inference.API.slicer_api import slice_audio


def batch_iter(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def collect_wavs(input_dir: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


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
    waveform, actual_sr = librosa.load(str(source_audio), sr=sr, mono=True)
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
    waveform, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    if waveform.size == 0:
        print(f"[SKIP] Empty audio: {audio_path}")
        return 0, 0

    print(f"\n[FILE] {audio_path.name}")
    chunks = slice_audio(waveform, sr, slicing_method)
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
        description="Batch slice WAV files and run Qwen3-ASR to export .wav + .lab outputs."
    )
    parser.add_argument("input_dir", type=Path, help="包含 .wav 的输入文件夹")
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
    parser.add_argument("--file-batch-size", type=int, default=1, help="文件批大小（按文件分批处理）")
    parser.add_argument("--no-recursive", action="store_true", help="只扫描输入目录第一层 wav 文件")
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
    return parser


def main():
    args = build_argparser().parse_args()

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

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    wav_files = collect_wavs(input_dir, recursive=not args.no_recursive)
    if not wav_files:
        print(f"No wav files found in: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    total_labs = 0

    # 持久化模型：主进程中一次性加载，所有文件共享
    asr_model = None
    if args.keep_model:
        print(f"[Keep-Model] Loading ASR model into VRAM once for all files: {args.asr_model}")
        asr_model = load_qwen_model(args.asr_model, args.device, use_cache=False)
        print("[Keep-Model] Model loaded. Will remain in VRAM for the entire batch.")

    try:
        print(f"Found {len(wav_files)} wav files. Processing in file batches of {args.file_batch_size}...")
        for batch_no, file_batch in enumerate(batch_iter(wav_files, args.file_batch_size), start=1):
            print(f"\n=== File batch {batch_no} / {(len(wav_files) + args.file_batch_size - 1) // args.file_batch_size} ===")
            for audio_path in file_batch:
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
                )
                total_chunks += chunks
                total_labs += labs
    finally:
        if asr_model is not None:
            print("\n[Keep-Model] Releasing ASR model from VRAM...")
            clear_qwen_model_cache()
            del asr_model

    print(f"\nDone. Total chunks: {total_chunks}, total labs: {total_labs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())