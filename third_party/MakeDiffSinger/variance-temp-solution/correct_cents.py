import json
import math
import warnings
from collections import OrderedDict

import librosa
import numpy as np
import tqdm
import pathlib
from csv import DictReader, DictWriter

import click

from get_pitch import get_pitch_parselmouth

warns = []


def get_aligned_pitch(wav_path: pathlib.Path, total_secs: float, timestep: float):
    waveform, _ = librosa.load(wav_path, sr=44100, mono=True)
    _, f0, _ = get_pitch_parselmouth(waveform, 512, 44100)
    pitch = librosa.hz_to_midi(f0)
    if pitch.shape[0] < total_secs / timestep:
        pad = math.ceil(total_secs / timestep) - pitch.shape[0]
        pitch = np.pad(pitch, [0, pad], mode='constant', constant_values=[0, pitch[-1]])
    return pitch


def correct_cents_item(
        name: str, item: OrderedDict, ref_pitch: np.ndarray,
        timestep: float, error_ratio: float
):
    note_seq = item['note_seq'].split()
    note_dur = [float(d) for d in item['note_dur'].split()]
    assert len(note_seq) == len(note_dur)

    start = 0.
    note_seq_correct = []
    for i, (note, dur) in enumerate(zip(note_seq, note_dur)):
        end = start + dur
        if note == 'rest':
            start = end
            note_seq_correct.append('rest')
            continue

        midi = librosa.note_to_midi(note, round_midi=False)
        start_idx = math.floor(start / timestep)
        end_idx = math.ceil(end / timestep)
        note_pitch = ref_pitch[start_idx: end_idx]
        note_pitch_close = note_pitch[(note_pitch >= midi - 0.5) & (note_pitch < midi + 0.5)]
        if len(note_pitch_close) < len(note_pitch) * error_ratio or len(note_pitch) == 0:
            warns.append({
                'position': name,
                'note_index': i,
                'note_value': note
            })
            if len(note_pitch) == 0 or len(note_pitch_close) == 0:
                start = end
                note_seq_correct.append(note)
                continue
        midi_correct = np.mean(note_pitch_close)
        note_seq_correct.append(librosa.midi_to_note(midi_correct, cents=True, unicode=False))

        start = end

    item['note_seq'] = ' '.join(note_seq_correct)


def save_warnings(save_dir: pathlib.Path):
    if len(warns) > 0:
        save_path = save_dir.resolve() / 'warnings.csv'
        with open(save_path, 'w', encoding='utf8', newline='') as f:
            writer = DictWriter(f, fieldnames=['position', 'note_index', 'note_value'])
            writer.writeheader()
            writer.writerows(warns)
        warnings.warn(
            message=f'possible labeling errors saved in {save_path}',
            category=UserWarning
        )
        warnings.filterwarnings(action='default')


@click.group(help='Apply cents correction to note sequences')
def correct_cents():
    pass


@correct_cents.command(help='Apply cents correction to note sequences in transcriptions.csv')
@click.argument('transcriptions', metavar='TRANSCRIPTIONS')
@click.argument('waveforms', metavar='WAVS')
@click.option('--error_ratio', metavar='RATIO', type=float, default=0.4,
              help='If the percentage of pitch points within a deviation of 50 cents compared to the note label '
                   'is lower than this value, a warning will be raised.')
def csv(
        transcriptions,
        waveforms,
        error_ratio
):
    transcriptions = pathlib.Path(transcriptions).resolve()
    waveforms = pathlib.Path(waveforms).resolve()
    with open(transcriptions, 'r', encoding='utf8') as f:
        reader = DictReader(f)
        items: list[OrderedDict] = []
        for item in reader:
            items.append(OrderedDict(item))

    timestep = 512 / 44100
    for item in tqdm.tqdm(items):
        item: OrderedDict
        ref_pitch = get_aligned_pitch(
            wav_path=waveforms / (item['name'] + '.wav'),
            total_secs=sum(float(d) for d in item['note_dur'].split()),
            timestep=timestep
        )
        correct_cents_item(
            name=item['name'], item=item, ref_pitch=ref_pitch,
            timestep=timestep, error_ratio=error_ratio
        )

    with open(transcriptions, 'w', encoding='utf8', newline='') as f:
        writer = DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num', 'note_seq', 'note_dur'])
        writer.writeheader()
        writer.writerows(items)
    save_warnings(transcriptions.parent)


@correct_cents.command(help='Apply cents correction to note sequences in DS files')
@click.argument('ds_dir', metavar='DS_DIR')
@click.option('--error_ratio', metavar='RATIO', type=float, default=0.4,
              help='If the percentage of pitch points within a deviation of 50 cents compared to the note label '
                   'is lower than this value, a warning will be raised.')
def ds(
        ds_dir,
        error_ratio
):
    ds_dir = pathlib.Path(ds_dir).resolve()
    assert ds_dir.exists(), 'The directory of DS files does not exist.'

    timestep = 512 / 44100
    for ds_file in tqdm.tqdm(ds_dir.glob('*.ds')):
        if not ds_file.is_file():
            continue

        assert ds_file.with_suffix('.wav').exists(), \
            f'Missing corresponding .wav file of {ds_file.name}.'
        with open(ds_file, 'r', encoding='utf8') as f:
            params = json.load(f)
        if not isinstance(params, list):
            params = [params]
        params = [OrderedDict(p) for p in params]

        ref_pitch = get_aligned_pitch(
            wav_path=ds_file.with_suffix('.wav'),
            total_secs=params[-1]['offset'] + sum(float(d) for d in params[-1]['note_dur'].split()),
            timestep=timestep
        )
        for i, param in enumerate(params):
            start_idx = math.floor(param['offset'] / timestep)
            end_idx = math.ceil((param['offset'] + sum(float(d) for d in param['note_dur'].split())) / timestep)
            correct_cents_item(
                name=f'{ds_file.stem}#{i}', item=param, ref_pitch=ref_pitch[start_idx: end_idx],
                timestep=timestep, error_ratio=error_ratio
            )

        with open(ds_file, 'w', encoding='utf8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    save_warnings(ds_dir)


if __name__ == '__main__':
    correct_cents()
