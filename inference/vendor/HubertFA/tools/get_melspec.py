import torch
import torch.nn.functional as F
import torchaudio


class MelSpectrogram(torch.nn.Module):
    def __init__(
            self,
            n_mel_channels: int,
            sampling_rate: int,
            win_length: int,
            hop_length: int,
            n_fft: int = None,
            mel_fmin: int = 0,
            mel_fmax: int = None,
            clamp: float = 1e-5,
            device: str = None,
    ):
        super().__init__()
        self.device: str = device
        self.n_fft: int = win_length if n_fft is None else n_fft
        self.win_length: int = win_length
        self.hop_length: int = hop_length
        self.sampling_rate: int = sampling_rate
        self.n_mel_channels: int = n_mel_channels
        self.clamp: float = clamp
        self.register_buffer("window", torch.hann_window(win_length, device=self.device))

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mel_channels,
            sample_rate=self.sampling_rate,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_stft=self.n_fft // 2 + 1,
            mel_scale="htk",
        ).to(device)

    def forward(self, audio):
        pad_left = self.n_fft // 2
        pad_right = (self.n_fft + 1) // 2
        audio = F.pad(audio, (pad_left, pad_right))

        spectrogram = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 + spectrogram[..., 1] ** 2)
        mel_output = self.mel_scale(spectrogram)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class MelSpecExtractor(torch.nn.Module):
    def __init__(
            self,
            n_mels: int,
            sample_rate: int,
            window_size: int,
            hop_size: int,
            n_fft: int,
            f_min: int,
            f_max: int,
            clamp: int,
            device: str | torch.cuda.device = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.melspec_transform = MelSpectrogram(
            n_mel_channels=n_mels,
            sampling_rate=sample_rate,
            win_length=window_size,
            hop_length=hop_size,
            n_fft=n_fft,
            mel_fmin=f_min,
            mel_fmax=f_max,
            clamp=clamp,
            device=device
        )

    def __call__(self, waveform, key_shift=0):
        return self.melspec_transform(waveform.unsqueeze(0))
