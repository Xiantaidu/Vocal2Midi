# coding=utf-8
import wave
from pathlib import Path
from typing import List

import numpy as np
import scipy.signal

SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]


def normalize_language_name(language: str) -> str:
    if language is None:
        raise ValueError("language is None")
    value = str(language).strip()
    if not value:
        raise ValueError("language is empty")
    return value[:1].upper() + value[1:].lower()


def validate_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def _load_wav_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        src_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    if src_rate != sample_rate:
        audio = scipy.signal.resample_poly(audio, sample_rate, src_rate)

    return audio.astype(np.float32, copy=False)


def load_audio(audio_path, sample_rate=16000, start_second=None, duration=None):
    audio_path = Path(audio_path)
    audio = None
    try:
        from pydub import AudioSegment

        load_kwargs = {"frame_rate": sample_rate, "channels": 1}
        if start_second is not None:
            load_kwargs["start_second"] = start_second
        if duration:
            load_kwargs["duration"] = duration
        audio_segment = AudioSegment.from_file(str(audio_path), **load_kwargs)
        bit_depth = audio_segment.sample_width * 8
        max_val = float(1 << (bit_depth - 1))
        audio = np.array(
            audio_segment.set_channels(1).set_frame_rate(sample_rate).get_array_of_samples()
        ) / max_val
        audio = audio.astype(np.float32)
    except Exception:
        if audio_path.suffix.lower() != ".wav":
            raise
        audio = _load_wav_audio(audio_path, sample_rate)

    start_sample = int((start_second or 0.0) * sample_rate)
    end_sample = None
    if duration:
        end_sample = start_sample + int(duration * sample_rate)
    return audio[start_sample:end_sample].astype(np.float32, copy=False)
