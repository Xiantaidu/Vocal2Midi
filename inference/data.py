import csv
import pathlib
from typing import Any, Literal

import librosa
import torch.utils.data

from inference.slicer2 import Slicer
from inference.utils import validate_phones, parse_words
from training.data import collate_nd


class SlicedAudioFileIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self, filemap: dict[str, pathlib.Path],
            samplerate: int,
            slicer: Slicer,
            language: int = 0,
    ):
        assert samplerate == slicer.sr, "Samplerate mismatches between dataset and slicer!"
        self.filemap = filemap
        self.samplerate = samplerate
        self.slicer = slicer
        self.language = language

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        items = list(self.filemap.items())
        if worker is None:
            start, step = 0, 1
        else:
            start = worker.id
            step = worker.num_workers
        for key, filepath in items[start::step]:
            waveform, _ = librosa.load(filepath, sr=self.samplerate, mono=True)
            chunks = self.slicer.slice(waveform)
            num_parts = len(chunks)
            for chunk in chunks:
                chunk_wav = torch.from_numpy(chunk["waveform"]).float()
                yield {
                    "key": key,
                    "offset": chunk["offset"],
                    "num_parts": num_parts,
                    "waveform": chunk_wav,
                    "samplerate": self.samplerate,
                    "duration": chunk_wav.shape[0] / self.samplerate,
                    "language": self.language,
                }

    @classmethod
    def collate(cls, samples: list[dict[str, Any]]) -> dict[str, Any]:
        sr = {s["samplerate"] for s in samples}
        if len(sr) > 1:
            raise ValueError(f"Multiple samplerate values found in batch: {sr}")
        samplerate = sr.pop()
        batch = {
            "size": len(samples),
            "samplerate": samplerate,
            "key": [s["key"] for s in samples],
            "offset": [s["offset"] for s in samples],
            "length": [s["duration"] for s in samples],
            "num_parts": [s["num_parts"] for s in samples],
            "waveform": collate_nd(
                [s["waveform"] for s in samples], pad_value=0.,
            ),
            "known_durations": torch.FloatTensor([[s["duration"]] for s in samples]),
            "language": torch.LongTensor([s["language"] for s in samples]),
        }
        return batch


class DiffSingerTranscriptionsDataset(torch.utils.data.Dataset):
    def __init__(
            self, filelist: list[pathlib.Path],
            samplerate: int,
            extensions: list[str] = None,
            language: int = 0,
            use_wb: bool = True,
            uv_vocab: set[str] | None = None,
            uv_word_cond: Literal["lead", "all"] = "all",
    ):
        self.samplerate = samplerate
        if extensions is None:
            extensions = [".wav", ".flac"]
        self.extensions = extensions
        self.language = language
        self.use_wb = use_wb
        if uv_word_cond not in ("lead", "all"):
            raise ValueError(f"Invalid uv_word_cond: '{uv_word_cond}'. Must be 'lead' or 'all'.")
        self.uv_vocab = uv_vocab
        self.uv_word_cond = uv_word_cond
        self.filelist = filelist
        self.itemlist = []
        for index in filelist:
            with open(index, "r", encoding="utf8") as f:
                items = list(csv.DictReader(f))
                if len(items) == 0:
                    raise ValueError(f"No items found in index \'{index.as_posix()}\'.")
                for item in items:
                    self.itemlist.append(self.parse_item(index, item))

    def parse_item(self, index: pathlib.Path, item: dict[str, Any]) -> dict[str, Any]:
        name = item["name"]
        candidate_wav_fns = [
            index.parent / "wavs" / f"{name}{ext}"
            for ext in self.extensions
        ]
        wav_fn = None
        for fn in candidate_wav_fns:
            if fn.is_file():
                wav_fn = fn
                break
        if wav_fn is None:
            raise FileNotFoundError(
                f"Waveform file not found for item \'{name}\' in index \'{index.as_posix()}\'. "
                f"Tried candidates: {[fn.as_posix() for fn in candidate_wav_fns]}"
            )
        ph_dur = [float(d) for d in item["ph_dur"].split()]
        if self.use_wb:
            ph_seq = item["ph_seq"].split()
            ph_num = [int(n) for n in item["ph_num"].split()]
            is_valid, err_msg = validate_phones(ph_seq, ph_dur, ph_num)
            if not is_valid:
                raise ValueError(
                    f"Invalid phone sequence in item \'{name}\' in index \'{index.as_posix()}\': {err_msg}"
                )
            word_dur, _ = parse_words(
                ph_seq, ph_dur, ph_num,
                uv_vocab=self.uv_vocab,
                uv_cond=self.uv_word_cond,
                merge_consecutive_uv=self.uv_vocab is not None,
            )
        else:
            word_dur = [sum(ph_dur)]
        return {
            "index": index.as_posix(),
            "name": name,
            "wav_fn": wav_fn,
            "word_dur": word_dur,
        }

    def __len__(self):
        return len(self.itemlist)

    def __getitem__(self, idx):
        item = self.itemlist[idx]
        waveform, _ = librosa.load(item["wav_fn"], sr=self.samplerate, mono=True)
        known_durations = torch.FloatTensor(item["word_dur"])
        return {
            "index": item["index"],
            "name": item["name"],
            "waveform": torch.from_numpy(waveform).float(),
            "samplerate": self.samplerate,
            "known_durations": known_durations,
            "language": self.language,
        }

    @classmethod
    def collate(cls, samples: list[dict[str, Any]]) -> dict[str, Any]:
        sr = {s["samplerate"] for s in samples}
        if len(sr) > 1:
            raise ValueError(f"Multiple samplerate values found in batch: {sr}")
        samplerate = sr.pop()
        batch = {
            "size": len(samples),
            "samplerate": samplerate,
            "index": [s["index"] for s in samples],
            "name": [s["name"] for s in samples],
            "waveform": collate_nd(
                [s["waveform"] for s in samples], pad_value=0.,
            ),
            "known_durations": collate_nd(
                [s["known_durations"] for s in samples], pad_value=0.,
            ),
            "language": torch.LongTensor([s["language"] for s in samples]),
        }
        return batch
