import abc
import csv
import json
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal

import librosa
import lightning.pytorch.callbacks
import matplotlib.pyplot as plt
import mido
import torch
import torch.nn.functional as F

from inference.me_infer_module import InferenceModule
from inference.utils import validate_phones, parse_words, align_notes_to_words
from lib import logging
from lib.plot import note_to_figure

__all__ = [
    "SaveCombinedFileCallback",
    "SaveCombinedMidiFileCallback",
    "SaveCombinedTextFileCallback",
    "UpdateDiffSingerTranscriptionsCallback",
    "VisualizeNoteComparisonCallback",
    "ExportMetricSummaryCallback",
]


@dataclass
class _NoteInfo:
    onset: float
    offset: float
    pitch: float


class SaveCombinedFileCallback(lightning.pytorch.callbacks.Callback, abc.ABC):
    def __init__(self, output_dir: str | pathlib.Path):
        super().__init__()
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        self.output_dir = output_dir
        self.counters: dict[str, int] = {}
        self.notes: dict[str, list[_NoteInfo]] = {}

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        batch_size = batch["size"]
        for i in range(batch_size):
            key: str = batch["key"][i]
            num_parts = batch["num_parts"][i]
            if key not in self.counters:
                self.counters[key] = 0
                self.notes[key] = []
            offset: float = batch["offset"][i]
            length: float = batch["length"][i]
            durations = outputs["durations"][i]
            scores = outputs["scores"][i]
            presence = outputs["presence"][i]
            note_onset = F.pad(
                durations, (1, 0), mode="constant", value=0
            ).cumsum(dim=0).clamp(max=length).add(offset)
            note_offset = durations.cumsum(dim=0).clamp(max=length).add(offset)
            for onset, offset, score, valid in zip(
                    note_onset.tolist(),
                    note_offset.tolist(),
                    scores.tolist(),
                    presence.tolist(),
            ):
                if offset - onset <= 0:
                    continue
                if not valid:
                    continue
                self.notes[key].append(_NoteInfo(
                    onset=onset,
                    offset=offset,
                    pitch=score,
                ))
            self.counters[key] += 1
            if self.counters[key] >= num_parts:
                self.save_file(key, logger_fn=trainer.progress_bar_callback.print)

    def on_predict_epoch_end(self, trainer: lightning.pytorch.Trainer, *args, **kwargs) -> None:
        for key in self.counters.keys():
            self.save_file(key, logger_fn=trainer.progress_bar_callback.print)

    def save_file(self, key: str, logger_fn: Callable) -> None:
        sorted_notes = sorted(self.notes[key], key=lambda x: (x.onset, x.offset, x.pitch))
        last_time = 0
        i = 0
        while i < len(sorted_notes):
            note = sorted_notes[i]
            note.onset = max(note.onset, last_time)
            note.offset = max(note.offset, note.onset)
            if note.offset <= note.onset:
                sorted_notes.pop(i)
            else:
                last_time = note.offset
                i += 1
        self.flush(key, sorted_notes, logger_fn)
        del self.counters[key]
        del self.notes[key]

    @abc.abstractmethod
    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        pass


class SaveCombinedMidiFileCallback(SaveCombinedFileCallback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            tempo: int = 120,
    ):
        super().__init__(output_dir)
        self.tempo = tempo

    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.tempo), time=0))
        last_time = 0
        for note in notes:
            onset_ticks = round(note.onset * self.tempo * 8)
            offset_ticks = round(note.offset * self.tempo * 8)
            midi_pitch = round(note.pitch)
            if offset_ticks <= onset_ticks:
                continue
            track.append(mido.Message(
                "note_on", note=midi_pitch, time=onset_ticks - last_time
            ))
            track.append(mido.Message(
                "note_off", note=midi_pitch, time=offset_ticks - onset_ticks
            ))
            last_time = offset_ticks

        filepath = (self.output_dir / key).with_suffix(".mid")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with mido.MidiFile(charset="utf8") as midi_file:
            midi_file.tracks.append(track)
            midi_file.save(filepath)

        logging.info(f"Saved MIDI file: {filepath.as_posix()}", callback=logger_fn)


class SaveCombinedTextFileCallback(SaveCombinedFileCallback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            file_format: Literal["txt", "csv"] = "csv",
            pitch_format: Literal["number", "name"] = "name",
            round_pitch: bool = False,
    ):
        super().__init__(output_dir)
        self.file_format = file_format
        self.pitch_format = pitch_format
        self.round_pitch = round_pitch

    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        onset_list = [
            f"{note.onset:.3f}" for note in notes
        ]
        offset_list = [
            f"{note.offset:.3f}" for note in notes
        ]
        pitch_list = []
        for note in notes:
            pitch = note.pitch
            if self.round_pitch:
                pitch = round(pitch)
                pitch_txt = str(pitch)
            else:
                pitch_txt = f"{pitch:.3f}"
            if self.pitch_format == "name":
                pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not self.round_pitch)
            pitch_list.append(pitch_txt)

        if self.file_format == "txt":
            filepath = (self.output_dir / key).with_suffix(".txt")
            with filepath.open(encoding="utf8", mode="w") as f:
                for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                    f.write(f"{onset}\t{offset}\t{pitch}\n")
            logging.info(f"Saved text file: {filepath.as_posix()}", callback=logger_fn)
        elif self.file_format == "csv":
            filepath = (self.output_dir / key).with_suffix(".csv")
            with filepath.open(encoding="utf8", mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["onset", "offset", "pitch"])
                writer.writeheader()
                for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                    writer.writerow({
                        "onset": onset,
                        "offset": offset,
                        "pitch": pitch,
                    })
            logging.info(f"Saved CSV file: {filepath.as_posix()}", callback=logger_fn)


class UpdateDiffSingerTranscriptionsCallback(lightning.pytorch.callbacks.Callback):
    def __init__(
            self, filelist: list[pathlib.Path],
            overwrite: bool = False,
            save_dir: str | pathlib.Path = None,
            save_filename: str = "transcriptions-midi.csv",
            use_wb: bool = True,
            uv_vocab: set[str] | None = None,
            uv_word_cond: Literal["lead", "all"] = "all",
            uv_note_cond: Literal["predict", "follow"] = "predict",
    ):
        super().__init__()
        self.overwrite = overwrite
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir
        self.save_filename = save_filename
        self.use_wb = use_wb
        if uv_word_cond not in ("lead", "all"):
            raise ValueError(f"Invalid uv_word_cond: '{uv_word_cond}'. Must be 'lead' or 'all'.")
        if uv_note_cond not in ("predict", "follow"):
            raise ValueError(f"Invalid uv_note_cond: '{uv_note_cond}'. Must be 'predict' or 'follow'.")
        self.uv_vocab = uv_vocab
        self.uv_word_cond = uv_word_cond
        self.uv_note_cond = uv_note_cond
        self.index_map: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        self.lengths: dict[str, int] = {}
        self.counters: dict[str, int] = {}
        for index in filelist:
            with open(index, "r", encoding="utf8") as f:
                items = list(csv.DictReader(f))
                o_dict = OrderedDict()
                for item in items:
                    o_dict[item["name"]] = item
                key = index.as_posix()
                self.index_map[key] = o_dict
                self.lengths[key] = len(items)
                self.counters[key] = 0

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        batch_size = batch["size"]
        for i in range(batch_size):
            index: str = batch["index"][i]
            name: str = batch["name"][i]
            durations = outputs["durations"][i]
            scores = outputs["scores"][i]
            presence = outputs["presence"][i]
            valid = durations > 0

            note_dur = durations[valid].tolist()
            note_midi = scores[valid].tolist()
            note_vuv = presence[valid].tolist()

            item = self.index_map[index][name]
            if self.use_wb:
                if self.uv_note_cond == "follow":
                    # When "follow", defer v/uv to align_notes_to_words; use raw pitch for all notes.
                    note_seq = [
                        librosa.midi_to_note(midi, unicode=False, cents=True)
                        for midi in note_midi
                    ]
                else:  # "predict"
                    # When "predict", apply presence-based "rest" now so alignment preserves them.
                    note_seq = [
                        librosa.midi_to_note(midi, unicode=False, cents=True) if vuv else "rest"
                        for midi, vuv in zip(note_midi, note_vuv)
                    ]
                ph_seq = item["ph_seq"].split()
                ph_dur = [float(d) for d in item["ph_dur"].split()]
                ph_num = [int(n) for n in item["ph_num"].split()]
                is_valid, err_msg = validate_phones(ph_seq, ph_dur, ph_num)
                if not is_valid:
                    raise ValueError(
                        f"Invalid phone sequence in item \'{name}\' in index \'{index}\': {err_msg}"
                    )
                word_dur, word_vuv = parse_words(
                    ph_seq, ph_dur, ph_num,
                    uv_vocab=self.uv_vocab,
                    uv_cond=self.uv_word_cond,
                    merge_consecutive_uv=False,
                )
                note_seq, note_dur, note_slur = align_notes_to_words(
                    word_dur, word_vuv,
                    note_seq, note_dur,
                    apply_word_uv=(self.uv_note_cond == "follow"),
                )
            else:
                # No alignment: apply presence-based "rest" directly from model output.
                note_seq = [
                    librosa.midi_to_note(score, unicode=False, cents=True) if pres else "rest"
                    for score, pres in zip(note_midi, note_vuv)
                ]
                note_slur = None

            item["note_seq"] = " ".join(note_seq)
            item["note_dur"] = " ".join(f"{dur:.3f}" for dur in note_dur)
            if note_slur is None:
                item.pop("note_slur", None)
            else:
                item["note_slur"] = " ".join(str(s) for s in note_slur)
            item.pop("note_glide", None)
            self.counters[index] += 1
            if self.counters[index] >= self.lengths[index]:
                self.flush(index, logger_fn=trainer.progress_bar_callback.print)

    def on_predict_epoch_end(self, trainer: lightning.pytorch.Trainer, *args, **kwargs) -> None:
        for index in self.counters.keys():
            self.flush(index, logger_fn=trainer.progress_bar_callback.print)

    def flush(self, key: str, logger_fn: Callable):
        items = list(self.index_map[key].values())
        index = pathlib.Path(key)
        if self.overwrite:
            save_path = index
        elif self.save_dir is not None:
            save_path = self.save_dir / self.save_filename
        else:
            save_path = index.parent / self.save_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open(encoding="utf8", mode="w", newline="") as f:
            fieldnames = list(items[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)
        del self.index_map[key]
        del self.lengths[key]
        del self.counters[key]
        logging.info(f"Saved transcriptions: {save_path.as_posix()}", callback=logger_fn)


class VisualizeNoteComparisonCallback(lightning.pytorch.callbacks.Callback):
    def __init__(self, save_dir: str | pathlib.Path, num_digits: int = None):
        super().__init__()
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir
        self.num_digits = num_digits

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        for i in range(len(batch["indices"])):
            idx = batch["indices"][i].item()
            title = batch["names"][i]
            N = batch["N"][i].item()
            scores_gt = batch["scores"][i, :N]
            presence_gt = batch["presence"][i, :N]
            durations_gt = batch["durations"][i, :N]
            N_pred = outputs["N"][i].item()
            scores_pred = outputs["scores"][i, :N_pred]
            presence_pred = outputs["presence"][i, :N_pred]
            durations_pred = outputs["durations_frame"][i, :N_pred]

            fig = note_to_figure(
                note_midi_gt=scores_gt.cpu().numpy(),
                note_rest_gt=(~presence_gt).cpu().numpy(),
                note_dur_gt=durations_gt.cpu().numpy(),
                note_midi_pred=scores_pred.cpu().numpy(),
                note_rest_pred=(~presence_pred).cpu().numpy(),
                note_dur_pred=durations_pred.cpu().numpy(),
                title=title
            )

            name = str(idx)
            if self.num_digits is not None:
                name = name.zfill(self.num_digits)

            save_path = self.save_dir / f"{name}.jpg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)


class ExportMetricSummaryCallback(lightning.pytorch.callbacks.Callback):
    def __init__(self, save_path: str | pathlib.Path):
        super().__init__()
        if isinstance(save_path, str):
            save_path = pathlib.Path(save_path)
        self.save_path = save_path

    def on_test_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: InferenceModule,
            *args, **kwargs
    ) -> None:
        summary: dict[str, float] = {}
        for name, metric in pl_module.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                for k, v in value.items():
                    summary[k] = v.item()
            else:
                summary[name] = value.item()
        summary_str = json.dumps(summary, indent=4)
        trainer.progress_bar_callback.print(summary_str)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open(encoding="utf8", mode="w") as f:
            f.write(summary_str)
        logging.info(
            f"Saved metric summary: {self.save_path.as_posix()}",
            callback=trainer.progress_bar_callback.print
        )
