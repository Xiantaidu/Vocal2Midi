# File: src/dataset.py
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

try:
    from src.augment import AudioAugmenter
except ModuleNotFoundError:
    from augment import AudioAugmenter


class PhonemeDataset(Dataset):
    def __init__(self, manifest_path: str, vocab_path: str, config: dict, is_train: bool = True):
        self.items = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.config = config
        self.is_train = is_train
        self.sr = config.get("sample_rate", 16000)
        self.max_len = int(config.get("max_seconds", 12.0) * self.sr)
        self.augmenter = AudioAugmenter(config.get("augment", {})) if is_train else None

    def __len__(self):
        return len(self.items)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        # to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        return waveform.numpy()[0]

    def __getitem__(self, idx):
        item = self.items[idx]
        wav = self._load_audio(item["audio"])

        # waveform augmentation
        if self.augmenter:
            wav = self.augmenter.apply_waveform(wav, self.sr)

        # truncate
        if len(wav) > self.max_len:
            wav = wav[: self.max_len]

        # encode phones
        phone_ids = [self.vocab.get(p, self.vocab.get("<unk>", 1)) for p in item["phones"]]

        return {
            "input_values": torch.tensor(wav, dtype=torch.float32),
            "labels": torch.tensor(phone_ids, dtype=torch.long),
            "input_length": len(wav),
            "audio_path": item["audio"],
        }


def collate_fn(batch):
    input_values = [b["input_values"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_lengths = [b["input_length"] for b in batch]

    # Pad audio sequences
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)

    # Pad label sequences
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Build attention_mask based on actual lengths (not value comparison)
    # This correctly handles samples that may contain legitimate zero-valued audio frames
    batch_size, max_len = input_values.shape
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, length in enumerate(input_lengths):
        attention_mask[i, :length] = 1

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }