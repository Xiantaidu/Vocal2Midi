import csv
import math
import pathlib

import click
import librosa
import numpy as np
import tqdm
from typing import List

from get_pitch import get_pitch


@click.command(help='Estimate note pitch from transcriptions and corresponding waveforms')
@click.argument('transcriptions', metavar='TRANSCRIPTIONS')
@click.argument('waveforms', metavar='WAVS')
@click.option('--pe', metavar='ALGORITHM', default='parselmouth',
              help='Pitch extractor (parselmouth, rmvpe)')
@click.option('--rest_uv_ratio', metavar='RATIO', type=float, default=0.85,
              help='The minimum percentage of unvoiced length for a note to be regarded as rest')
def estimate_midi(
        transcriptions: str,
        waveforms: str,
        pe: str = 'parselmouth',
        rest_uv_ratio: float = 0.85
):
    transcriptions = pathlib.Path(transcriptions).resolve()
    waveforms = pathlib.Path(waveforms).resolve()
    with open(transcriptions, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        items: List[dict] = []
        for item in reader:
            items.append(item)

    timestep = 512 / 44100
    for item in tqdm.tqdm(items):
        item: dict
        ph_dur = [float(d) for d in item['ph_dur'].split()]
        ph_num = [int(n) for n in item['ph_num'].split()]
        assert sum(ph_num) == len(ph_dur), f'ph_num does not sum to number of phones in \'{item["name"]}\'.'

        word_dur = []
        i = 0
        for num in ph_num:
            word_dur.append(sum(ph_dur[i: i + num]))
            i += num

        total_secs = sum(ph_dur)
        waveform, _ = librosa.load(waveforms / (item['name'] + '.wav'), sr=44100, mono=True)
        _, f0, uv = get_pitch(pe, waveform, 512, 44100)
        pitch = librosa.hz_to_midi(f0)
        if pitch.shape[0] < total_secs / timestep:
            pad = math.ceil(total_secs / timestep) - pitch.shape[0]
            pitch = np.pad(pitch, [0, pad], mode='constant', constant_values=[0, pitch[-1]])
            uv = np.pad(uv, [0, pad], mode='constant')

        note_seq = []
        note_dur = []
        start = 0.
        for dur in word_dur:
            end = start + dur
            start_idx = math.floor(start / timestep)
            end_idx = math.ceil(end / timestep)
            word_pitch = pitch[start_idx: end_idx]
            word_uv = uv[start_idx: end_idx]
            word_valid_pitch = np.extract(~word_uv & (word_pitch >= 0), word_pitch)
            if len(word_valid_pitch) < (1 - rest_uv_ratio) * (end_idx - start_idx):
                note_seq.append('rest')
            else:
                counts = np.bincount(np.round(word_valid_pitch).astype(np.int64))
                midi = counts.argmax()
                midi = np.mean(word_valid_pitch[(word_valid_pitch >= midi - 0.5) & (word_valid_pitch < midi + 0.5)])
                note_seq.append(librosa.midi_to_note(midi, cents=True, unicode=False))
            note_dur.append(dur)

            start = end

        item['note_seq'] = ' '.join(note_seq)
        item['note_dur'] = ' '.join([str(round(d, 6)) for d in note_dur])

    with open(transcriptions, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num', 'note_seq', 'note_dur'])
        writer.writeheader()
        writer.writerows(items)


if __name__ == '__main__':
    estimate_midi()
