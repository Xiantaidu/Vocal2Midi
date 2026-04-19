from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel
from scipy.signal import get_window


# ===== RMVPE constants (aligned with RMVPE-main/src/constants.py) =====
SAMPLE_RATE = 16000
HOP_LENGTH = 160
WINDOW_LENGTH = 2048
N_MELS = 128
N_CLASS = 360
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
RMVPE_CONST = 1997.3794084376191


@dataclass
class RmvpeResult:
    time_step_seconds: float
    midi_pitch: np.ndarray
    voiced_mask: np.ndarray | None = None


# ===== RMVPE model (aligned with RMVPE-main) =====
class _BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class _STFT(nn.Module):
    """Adapted from RMVPE-main/src/spec.py"""

    def __init__(self, filter_length: int, hop_length: int, win_length: int | None = None, window: str = "hann"):
        super().__init__()
        if win_length is None:
            win_length = filter_length
        self.filter_length = filter_length
        self.hop_length = hop_length

        fourier_basis = np.fft.fft(np.eye(filter_length))
        cutoff = int(filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            fft_window = get_window(window, win_length, fftbins=True)
            if win_length < filter_length:
                # Equivalent to librosa.util.pad_center for 1d vector
                pad_left = (filter_length - win_length) // 2
                pad_right = filter_length - win_length - pad_left
                fft_window = np.pad(fft_window, (pad_left, pad_right))
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_batches, num_samples = input_data.size(0), input_data.size(1)
        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data, self.forward_basis, stride=self.hop_length, padding=0)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)
        return magnitude, phase


class _MelSpectrogram(nn.Module):
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.stft = _STFT(WINDOW_LENGTH, HOP_LENGTH, WINDOW_LENGTH)
        mel_basis = librosa_mel(
            sr=SAMPLE_RATE,
            n_fft=WINDOW_LENGTH,
            n_mels=n_mels,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        magnitudes, _ = self.stft(y)
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
        return mel_output


class _ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class _ResEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([_ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(_ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class _ResDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([_ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(_ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # 对齐上采样与跳连张量的时频尺寸，避免奇偶长度导致的 1-bin 偏差
        if x.shape[2] != concat_tensor.shape[2] or x.shape[3] != concat_tensor.shape[3]:
            h = min(x.shape[2], concat_tensor.shape[2])
            w = min(x.shape[3], concat_tensor.shape[3])
            x = x[:, :, :h, :w]
            concat_tensor = concat_tensor[:, :, :h, :w]
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, in_size: int, n_encoders: int, kernel_size, n_blocks: int, out_channels: int = 16):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for _ in range(n_encoders):
            self.layers.append(_ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=0.01))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            c, x = self.layers[i](x)
            concat_tensors.append(c)
        return x, concat_tensors


class _Intermediate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList([_ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum=0.01)])
        for _ in range(n_inters - 1):
            self.layers.append(_ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum=0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class _Decoder(nn.Module):
    def __init__(self, in_channels: int, n_decoders: int, stride, n_blocks: int):
        super().__init__()
        self.n_decoders = n_decoders
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(_ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum=0.01))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class _TimbreFilter(nn.Module):
    def __init__(self, latent_rep_channels):
        super().__init__()
        self.layers = nn.ModuleList([_ConvBlockRes(ch[0], ch[0]) for ch in latent_rep_channels])

    def forward(self, x_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(x_tensors[i]) for i, layer in enumerate(self.layers)]


class _DeepUnet(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        kernel_size=(2, 2),
        n_blocks: int = 4,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        use_timbre_filter: bool = False,
    ):
        super().__init__()
        self.encoder = _Encoder(1, n_mels, en_de_layers, kernel_size, n_blocks, 16)
        self.intermediate = _Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.use_timbre_filter = use_timbre_filter
        if use_timbre_filter:
            self.tf = _TimbreFilter(self.encoder.latent_channels)
        self.decoder = _Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        if self.use_timbre_filter:
            concat_tensors = self.tf(concat_tensors)
        x = self.decoder(x, concat_tensors)
        return x


class _E2E(nn.Module):
    """Aligned with RMVPE-main/src/model.py::E2E"""

    def __init__(self, n_mels: int = N_MELS, use_timbre_filter: bool = False):
        super().__init__()
        self.n_mels = n_mels
        self.mel = _MelSpectrogram(n_mels=n_mels)
        self.unet = _DeepUnet(
            n_mels=n_mels,
            kernel_size=(2, 2),
            n_blocks=4,
            en_de_layers=5,
            inter_layers=4,
            use_timbre_filter=use_timbre_filter,
        )
        self.cnn = nn.Conv2d(16, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            _BiGRU(3 * n_mels, 256, 1),
            nn.Linear(512, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        hidden_vec = 0
        if len(self.fc) == 4:
            for i in range(len(self.fc)):
                x = self.fc[i](x)
                if i == 0:
                    hidden_vec = x
        return hidden_vec, x


class RmvpeTranscriber:
    def __init__(self, model_path: str | Path, device: str = "cuda", batch_size: int = 8, threshold: float = 0.03):
        self.model_path = Path(model_path)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RMVPE model not found: {self.model_path}")

        self.model = self._load_checkpoint(self.model_path).to(self.device)
        self.model.eval()

        self.seg_len = 160 * 512
        self.seg_frames = self.seg_len // HOP_LENGTH

    def _load_checkpoint(self, ckpt_path: Path) -> _E2E:
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        # Infer variant from checkpoint structure:
        # 1) fc GRU input dim => 3 * n_mels
        # 2) whether unet.tf.* exists => DeepUnet(with TF) or DeepUnet0(no TF)
        n_mels = N_MELS
        if isinstance(state_dict, dict) and "fc.0.gru.weight_ih_l0" in state_dict:
            n_mels = int(state_dict["fc.0.gru.weight_ih_l0"].shape[1] // 3)
        use_timbre_filter = any(str(k).startswith("unet.tf.") for k in state_dict.keys())

        model = _E2E(n_mels=n_mels, use_timbre_filter=use_timbre_filter)
        incompat = model.load_state_dict(state_dict, strict=False)

        ignored_missing = {"mel.mel_basis", "mel.stft.forward_basis"}
        real_missing = [k for k in incompat.missing_keys if k not in ignored_missing]
        if real_missing or incompat.unexpected_keys:
            raise RuntimeError(
                "RMVPE checkpoint mismatch. "
                f"missing={real_missing[:20]} unexpected={incompat.unexpected_keys[:20]}"
            )
        return model

    def infer(self, waveform: np.ndarray, sample_rate: int, cancel_checker=None) -> RmvpeResult:
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        waveform = np.clip(waveform.astype(np.float32), -1.0, 1.0)

        audio = torch.from_numpy(waveform).to(self.device)
        salience = self._inference_salience(audio, cancel_checker=cancel_checker)
        f0_hz = self._salience_to_f0(salience, self.threshold)
        voiced_mask = f0_hz > 0
        midi_pitch = self._f0_to_interpolated_midi(f0_hz)
        return RmvpeResult(
            time_step_seconds=HOP_LENGTH / SAMPLE_RATE,
            midi_pitch=midi_pitch,
            voiced_mask=voiced_mask,
        )

    def _inference_salience(self, audio: torch.Tensor, cancel_checker=None) -> np.ndarray:
        with torch.no_grad():
            if cancel_checker and cancel_checker():
                raise InterruptedError("RMVPE 任务已取消")
            padded_audio = self._pad_audio(audio)
            segments = self._en_frame(padded_audio)
            outputs = self._forward_in_mini_batch(segments, cancel_checker=cancel_checker)
            merged = self._de_frame(outputs)[: (len(audio) // HOP_LENGTH + 1)]
            return merged.detach().float().cpu().numpy()

    def _pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_len = len(audio)
        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = seg_nums * self.seg_len - audio_len + self.seg_len // 2
        return torch.cat(
            [
                torch.zeros(self.seg_len // 4, device=self.device),
                audio,
                torch.zeros(pad_len - self.seg_len // 4, device=self.device),
            ]
        )

    def _en_frame(self, audio: torch.Tensor) -> torch.Tensor:
        audio_len = len(audio)
        audio = torch.cat(
            [
                torch.zeros(1024, device=self.device),
                audio,
                torch.zeros(1024, device=self.device),
            ]
        )
        segments = []
        start = 0
        while start + self.seg_len <= audio_len:
            segments.append(audio[start : start + self.seg_len + 2048])
            start += self.seg_len // 2
        return torch.stack(segments, dim=0)

    def _forward_in_mini_batch(self, segments: torch.Tensor, cancel_checker=None) -> torch.Tensor:
        out_segments = []
        segments_num = segments.shape[0]
        batch_start = 0
        while True:
            if cancel_checker and cancel_checker():
                raise InterruptedError("RMVPE 任务已取消")
            if batch_start + self.batch_size >= segments_num:
                batch_tmp = segments[batch_start:].shape[0]
                segment_in = torch.cat(
                    [
                        segments[batch_start:],
                        torch.zeros_like(segments)[: self.batch_size - batch_tmp].to(self.device),
                    ],
                    dim=0,
                )
                _, out_tmp = self.model(segment_in)
                out_segments.append(out_tmp[:batch_tmp])
                break
            segment_in = segments[batch_start : batch_start + self.batch_size]
            _, out_tmp = self.model(segment_in)
            out_segments.append(out_tmp)
            batch_start += self.batch_size
        return torch.cat(out_segments, dim=0)

    def _de_frame(self, segments: torch.Tensor) -> torch.Tensor:
        output = [seg[self.seg_frames // 4 : int(self.seg_frames * 0.75)] for seg in segments]
        return torch.cat(output, dim=0)

    @staticmethod
    def _to_local_average_cents(salience: np.ndarray, center: int | None = None, thred: float = 0.0) -> float:
        cents_mapping = np.linspace(0, 7180, N_CLASS) + RMVPE_CONST
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        s = salience[start:end]
        if np.max(s) <= thred:
            return 0.0
        product_sum = np.sum(s * cents_mapping[start:end])
        weight_sum = np.sum(s)
        return float(product_sum / max(weight_sum, 1e-8))

    def _salience_to_f0(self, salience: np.ndarray, threshold: float) -> np.ndarray:
        # salience: [T, 360]
        f0 = np.zeros((salience.shape[0],), dtype=np.float32)
        for i in range(salience.shape[0]):
            row = salience[i]
            if float(np.max(row)) <= threshold:
                f0[i] = 0.0
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







