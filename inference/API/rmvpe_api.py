from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import get_window

from inference.device_utils import resolve_onnx_providers

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - handled at runtime
    ort = None


SAMPLE_RATE = 16000
HOP_LENGTH = 160
WINDOW_LENGTH = 2048
N_MELS = 128
N_CLASS = 360
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
RMVPE_CONST = 1997.3794084376191
SEG_LEN = 160 * 512
SEG_FRAMES = SEG_LEN // HOP_LENGTH
CENTS_MAPPING = np.linspace(0, 7180, N_CLASS, dtype=np.float32) + RMVPE_CONST


@dataclass
class RmvpeResult:
    time_step_seconds: float
    midi_pitch: np.ndarray
    voiced_mask: np.ndarray | None = None


class RmvpeTranscriber:
    def __init__(self, model_path: str | Path, device: str = "dml", batch_size: int = 8, threshold: float = 0.03):
        self.model_path = Path(model_path)
        self.requested_device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.threshold = float(threshold)

        if not self.model_path.exists():
            raise FileNotFoundError(f"RMVPE model not found: {self.model_path}")
        if ort is None:
            raise RuntimeError("onnxruntime is required for RMVPE ONNX inference.")

        self.provider_name, providers = self._resolve_providers(self.requested_device)
        self.session = self._create_session(providers)
        self.input_name = self.session.get_inputs()[0].name
        self.fixed_batch_size = self._get_fixed_batch_size()

        self.window = get_window("hann", WINDOW_LENGTH, fftbins=True).astype(np.float32)
        self.mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=WINDOW_LENGTH,
            n_mels=N_MELS,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            htk=True,
        ).astype(np.float32)

    @staticmethod
    def _resolve_providers(device: str) -> tuple[str, list[str]]:
        return resolve_onnx_providers(device, label="RMVPE ONNX")

    def _create_session(self, providers: list[str]):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        return ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)

    def _get_fixed_batch_size(self) -> int | None:
        shape = self.session.get_inputs()[0].shape
        if not shape:
            return None
        dim0 = shape[0]
        return int(dim0) if isinstance(dim0, int) else None

    def infer(self, waveform: np.ndarray, sample_rate: int, cancel_checker=None) -> RmvpeResult:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        if waveform.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return RmvpeResult(
                time_step_seconds=HOP_LENGTH / SAMPLE_RATE,
                midi_pitch=empty,
                voiced_mask=empty.astype(bool),
            )

        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        waveform = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)

        salience = self._inference_salience(waveform, cancel_checker=cancel_checker)
        f0_hz = self._salience_to_f0(salience, self.threshold)
        voiced_mask = f0_hz > 0
        midi_pitch = self._f0_to_interpolated_midi(f0_hz)
        return RmvpeResult(
            time_step_seconds=HOP_LENGTH / SAMPLE_RATE,
            midi_pitch=midi_pitch,
            voiced_mask=voiced_mask,
        )

    def _inference_salience(self, audio: np.ndarray, cancel_checker=None) -> np.ndarray:
        if cancel_checker and cancel_checker():
            raise InterruptedError("RMVPE task cancelled")

        padded_audio = self._pad_audio(audio)
        segments = self._en_frame(padded_audio)
        if len(segments) == 0:
            return np.zeros((0, N_CLASS), dtype=np.float32)

        outputs = self._forward_in_mini_batch(segments, cancel_checker=cancel_checker)
        merged = self._de_frame(outputs)
        target_frames = len(audio) // HOP_LENGTH + 1
        return merged[:target_frames]

    def _pad_audio(self, audio: np.ndarray) -> np.ndarray:
        audio_len = len(audio)
        seg_nums = int(np.ceil(audio_len / SEG_LEN)) + 1
        pad_len = seg_nums * SEG_LEN - audio_len + SEG_LEN // 2
        return np.concatenate(
            [
                np.zeros(SEG_LEN // 4, dtype=np.float32),
                audio.astype(np.float32, copy=False),
                np.zeros(pad_len - SEG_LEN // 4, dtype=np.float32),
            ]
        )

    def _en_frame(self, audio: np.ndarray) -> np.ndarray:
        audio_len = len(audio)
        padded = np.concatenate(
            [
                np.zeros(1024, dtype=np.float32),
                audio.astype(np.float32, copy=False),
                np.zeros(1024, dtype=np.float32),
            ]
        )
        segments = []
        start = 0
        while start + SEG_LEN <= audio_len:
            segments.append(padded[start : start + SEG_LEN + WINDOW_LENGTH])
            start += SEG_LEN // 2
        if not segments:
            return np.zeros((0, SEG_LEN + WINDOW_LENGTH), dtype=np.float32)
        return np.stack(segments, axis=0).astype(np.float32, copy=False)

    def _forward_in_mini_batch(self, segments: np.ndarray, cancel_checker=None) -> np.ndarray:
        out_segments: list[np.ndarray] = []
        for batch_start in range(0, int(segments.shape[0]), self.batch_size):
            if cancel_checker and cancel_checker():
                raise InterruptedError("RMVPE task cancelled")
            batch = segments[batch_start : batch_start + self.batch_size]
            if self.fixed_batch_size == 1:
                for segment in batch:
                    out_segments.append(self._run_single_segment(segment))
                continue

            mel_batch = np.stack([self._waveform_to_mel(segment) for segment in batch], axis=0)
            expected = self.fixed_batch_size or mel_batch.shape[0]
            if mel_batch.shape[0] < expected:
                pad_shape = (expected - mel_batch.shape[0],) + mel_batch.shape[1:]
                mel_batch = np.concatenate([mel_batch, np.zeros(pad_shape, dtype=np.float32)], axis=0)
            output = self.session.run(None, {self.input_name: mel_batch})[0]
            out_segments.extend(np.asarray(output[: batch.shape[0]], dtype=np.float32))

        if not out_segments:
            return np.zeros((0, 0, N_CLASS), dtype=np.float32)
        return np.stack(out_segments, axis=0)

    def _run_single_segment(self, segment: np.ndarray) -> np.ndarray:
        mel = self._waveform_to_mel(segment)[np.newaxis, :, :]
        output = self.session.run(None, {self.input_name: mel})[0]
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3:
            return output[0]
        if output.ndim == 2:
            return output
        raise RuntimeError(f"Unexpected RMVPE ONNX output shape: {output.shape}")

    def _waveform_to_mel(self, waveform: np.ndarray) -> np.ndarray:
        stft = librosa.stft(
            waveform.astype(np.float32, copy=False),
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            center=False,
        )
        magnitudes = np.abs(stft).astype(np.float32, copy=False)
        mel_output = self.mel_basis @ magnitudes
        mel_output = np.log(np.clip(mel_output, a_min=1e-5, a_max=None)).astype(np.float32, copy=False)
        # The exported ONNX graph expects the fixed RMVPE segment path to be 512 frames.
        # librosa.stft(center=False) yields 513 frames for the legacy 83968-sample segment,
        # so we trim the trailing frame to match the exported UNet skip dimensions.
        if mel_output.shape[1] == SEG_FRAMES + 1:
            mel_output = mel_output[:, :SEG_FRAMES]
        return np.ascontiguousarray(mel_output)

    def _de_frame(self, segments: np.ndarray) -> np.ndarray:
        if len(segments) == 0:
            return np.zeros((0, N_CLASS), dtype=np.float32)
        output = [seg[SEG_FRAMES // 4 : int(SEG_FRAMES * 0.75)] for seg in segments]
        return np.concatenate(output, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _to_local_average_cents(salience: np.ndarray, center: int | None = None, thred: float = 0.0) -> float:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        s = salience[start:end]
        if np.max(s) <= thred:
            return 0.0
        product_sum = np.sum(s * CENTS_MAPPING[start:end])
        weight_sum = np.sum(s)
        return float(product_sum / max(weight_sum, 1e-8))

    def _salience_to_f0(self, salience: np.ndarray, threshold: float) -> np.ndarray:
        f0 = np.zeros((salience.shape[0],), dtype=np.float32)
        for i in range(salience.shape[0]):
            row = salience[i]
            if float(np.max(row)) <= threshold:
                continue
            cents = self._to_local_average_cents(row, thred=threshold)
            f0[i] = 10.0 * (2.0 ** (cents / 1200.0))
        return f0

    @staticmethod
    def _f0_to_interpolated_midi(f0: np.ndarray) -> np.ndarray:
        midi = np.full_like(f0, np.nan, dtype=np.float32)
        voiced = f0 > 0
        midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

        voiced_indices = np.where(~np.isnan(midi))[0]
        if len(voiced_indices) == 0:
            return midi

        first = voiced_indices[0]
        midi[:first] = midi[first]

        prev = first
        idx = first + 1
        n = len(midi)
        while idx < n:
            if not np.isnan(midi[idx]):
                prev = idx
                idx += 1
                continue

            gap_start = idx
            while idx < n and np.isnan(midi[idx]):
                idx += 1

            if idx < n:
                left = midi[prev]
                right = midi[idx]
                gap_len = idx - prev
                for i in range(1, gap_len):
                    ratio = i / gap_len
                    midi[prev + i] = left + (right - left) * ratio
                prev = idx
            else:
                midi[gap_start:] = midi[prev]

        return midi

    def shutdown(self) -> None:
        self.session = None

    release = shutdown
