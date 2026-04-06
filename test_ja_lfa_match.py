import argparse
import os
import pathlib
import sys

import librosa
import soundfile as sf
import torch


ROOT_DIR = pathlib.Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "inference" / "vendor"
if str(VENDOR_DIR / "HubertFA") not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR / "HubertFA"))

from inference.slicer_api import slice_audio
from inference.asr_api import load_qwen_model
from inference.lfa_api import create_lyric_matcher, process_asr_to_phonemes


def _load_lyrics_text(lyrics_file: str) -> str:
    """Best-effort lyrics loader with friendly fallback.
    Returns empty string when the path is missing/invalid, so test can still run.
    """
    if not lyrics_file:
        return ""

    # Common placeholder patterns from docs/messages
    if "<" in lyrics_file and ">" in lyrics_file:
        print(f"[Warning] --lyrics-file looks like a placeholder, ignored: {lyrics_file}")
        return ""

    p = pathlib.Path(lyrics_file)
    if not p.exists():
        print(f"[Warning] Lyrics file not found, ignored: {lyrics_file}")
        return ""

    for enc in ("utf-8", "utf-8-sig", "gbk", "shift_jis"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    # Last resort
    return p.read_text(encoding="utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Japanese LFA matching smoke test")
    parser.add_argument("--audio", required=True, help="Path to input wav")
    parser.add_argument("--lyrics-file", default="", help="Path to original lyrics txt (optional)")
    parser.add_argument("--asr-model", default=r"C:\Users\Xiantaidu\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B")
    parser.add_argument("--output-dir", default="temp_ja_lfa_test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_lyrics = _load_lyrics_text(args.lyrics_file)

    print(f"Loading audio: {audio_path}")
    sr = 44100
    waveform, sr = librosa.load(audio_path, sr=sr, mono=True)

    print("Slicing audio...")
    chunks = slice_audio(waveform, sr, "默认切片")
    print(f"Got {len(chunks)} chunks.")

    asr_model = load_qwen_model(args.asr_model, device=args.device)
    matcher = create_lyric_matcher("ja", original_lyrics)

    audio_paths = []
    chunk_indices = []
    for chunk_idx, chunk in enumerate(chunks):
        stem = f"chunk_{chunk_idx}"
        chunk_path = output_dir / f"{stem}.wav"
        sf.write(chunk_path, chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)

    all_results = []
    asr_lang = "Japanese"
    for i in range(0, len(audio_paths), args.batch_size):
        batch_audio_paths = audio_paths[i:i + args.batch_size]
        print(f"  Processing ASR batch {i // args.batch_size + 1}/{(len(audio_paths) - 1) // args.batch_size + 1}...")
        with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
            batch_results = asr_model.transcribe(audio=batch_audio_paths, language=asr_lang)
        all_results.extend(batch_results)

    chars_dict, chunk_logs = process_asr_to_phonemes(
        all_results=all_results,
        chunk_indices=chunk_indices,
        temp_dir_path=output_dir,
        language="ja",
        matcher=matcher,
    )

    log_path = output_dir / "ja_lfa_match_log.txt"
    log_path.write_text("\n".join(chunk_logs), encoding="utf-8")

    print("\n=== LFA Match Summary ===")
    matched = sum("Matched with original lyrics" in x for x in chunk_logs)
    fallback = sum("Fallback to ASR" in x for x in chunk_logs)
    direct = sum("Direct ASR" in x for x in chunk_logs)
    print(f"Matched chunks: {matched}")
    print(f"Fallback chunks: {fallback}")
    print(f"Direct chunks: {direct}")
    print(f"Total valid chunks: {len(chars_dict)}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
