import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio

power_computer = None


def load_wav(path, target_sr, hop_size, device='cpu'):
    waveform, sr = torchaudio.load(str(path))
    if target_sr != sr and target_sr is not None:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    return waveform.to(device), len(waveform) / target_sr, len(waveform) // hop_size + 1


class PowerCurveComputer(nn.Module):
    def __init__(self, window_size=1024, hop_size=512):
        super().__init__()
        self.window_size = window_size + 1 if window_size % 2 == 0 else window_size
        self.hop_size = hop_size

    def forward(self, waveform, n_frames):
        wav_size = waveform.size(0)

        frame_centers = torch.arange(0, n_frames, device=waveform.device) * self.hop_size + self.hop_size // 2
        window_starts = (frame_centers - self.window_size // 2).clamp(min=0)

        indices = torch.arange(self.window_size, device=waveform.device).unsqueeze(0)  # [1, window_size]
        window_indices = window_starts.unsqueeze(1) + indices  # [n_frames, window_size]

        window_indices = torch.clamp(window_indices, 0, wav_size - 1)
        window_audio = waveform[window_indices.long()]  # [n_frames, window_size]

        valid_mask = (window_indices >= 0) & (window_indices < wav_size)
        squared_sum = (window_audio ** 2).sum(dim=1)
        valid_count = valid_mask.sum(dim=1).float()

        rms = torch.sqrt(squared_sum / valid_count.clamp(min=1))
        return 20 * torch.log10(rms.clamp(min=1e-10) / 1.0)  # power_db [T]


def get_curves(waveform, n_frames, window_size=1024, hop_size=512):
    global power_computer
    if power_computer is None:
        power_computer = PowerCurveComputer(window_size, hop_size)
    power_curve = power_computer(waveform, n_frames)
    return torch.stack([power_curve]).unsqueeze(0)  # [B, C, T]


def plot_multiple_curves(waveform, sr, curves, curve_names, hop_size):
    time_axis_waveform = torch.arange(waveform.size(0)) / sr
    time_axis_curves = (torch.arange(curves[0].size(0)) * hop_size + hop_size // 2) / sr

    fig, axes = plt.subplots(len(curves) + 1, 1, figsize=(12, 6 + 2 * len(curves)))

    axes[0].plot(time_axis_waveform.numpy(), waveform.squeeze().numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)

    for i, (curve, name) in enumerate(zip(curves, curve_names), 1):
        axes[i].plot(time_axis_curves.numpy(), curve.numpy())
        axes[i].set_title(f'{name} Curve')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(name)
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wav_file_path = r"C:\Users\99662\Desktop\hfa测试wav\wav\courenao_000.wav"

    audio, wav_length, total_frames = load_wav(wav_file_path, 44100, 512, device='cpu')
    power_computer = PowerCurveComputer(window_size=1024, hop_size=512)
    power_ = power_computer(audio, total_frames)
    plot_multiple_curves(audio.cpu(), 44100, [power_.cpu()], ['Power (dB)'], 512)
