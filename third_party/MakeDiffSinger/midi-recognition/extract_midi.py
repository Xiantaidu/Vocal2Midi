import csv
import json
import pathlib

import click
import librosa
from typing import List, Tuple


@click.command(help='Extract MIDI sequences from OpenSVIP json files and add them into transcriptions.csv')
@click.argument('json_dir', metavar='JSONS')
@click.argument('csv_file', metavar='TRANSCRIPTIONS')
@click.option('--key', type=int, default=0, show_default=True,
              metavar='SEMITONES', help='Key transition')
def extract_midi(json_dir, csv_file, key):
    json_dir = pathlib.Path(json_dir).resolve()
    assert json_dir.exists(), 'The json directory does not exist.'
    tags_file = json_dir / 'tags.json'
    assert tags_file.exists(), 'The tags.json does not exist.'
    csv_file = pathlib.Path(csv_file).resolve()
    assert csv_file.resolve(), 'The path to transcriptions.csv does not exist.'
    tol = 0.001

    with open(tags_file, 'r', encoding='utf8') as f:
        tags: dict = json.load(f)

    # Read MIDI sequences
    note_seq_map: dict = {}  # key: merged filename, value: note sequence
    for json_file in json_dir.iterdir():
        if json_file.stem not in tags or not json_file.is_file() or json_file.suffix != '.json':
            continue
        with open(json_file, 'r', encoding='utf8') as f:
            json_obj: dict = json.load(f)
        assert len(json_obj['SongTempoList']) == 1, \
            f'[ERROR] {json_file.name}: there must be one and only one single tempo in the project.'

        tempo = json_obj['SongTempoList'][0]['BPM']
        midi_seq: list = json_obj['TrackList'][0]['NoteList']
        note_seq: List[Tuple[str, float]] = []  # (note, duration)
        prev_pos: int = 0  # in ticks
        for i, midi in enumerate(midi_seq):
            if prev_pos < midi['StartPos']:
                note_seq.append(
                    ('rest', (midi['StartPos'] - prev_pos) / 8 / tempo)
                )
            note_seq.append(
                (librosa.midi_to_note(midi['KeyNumber'] + key, unicode=False), midi['Length'] / 8 / tempo)
            )
            prev_pos = midi['StartPos'] + midi['Length']
        remain_secs = prev_pos / 8 / tempo - sum(t['duration'] for t in tags[json_file.stem])
        if remain_secs > tol:
            note_seq.append(
                ('rest', remain_secs)
            )
        note_seq_map[json_file.stem] = note_seq

    # Load transcriptions
    transcriptions: dict = {}  # key: split filename, value: attr dict
    with open(csv_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for attrs in reader:
            transcriptions[attrs['name']] = attrs

    # Split note sequence and add into transcriptions
    for merged_name, note_seq in note_seq_map.items():
        note_seq: Tuple[str, float]
        idx = 0
        offset = 0.
        cur_note_secs = 0.
        cur_clip_secs = 0.
        for split_tag in tags[merged_name]:
            split_note_seq = []
            while idx < len(note_seq):
                cur_note_dur = note_seq[idx][1] - offset
                if cur_note_secs + cur_note_dur <= cur_clip_secs + split_tag['duration']:
                    split_note_seq.append(
                        (note_seq[idx][0], cur_note_dur)
                    )
                    idx += 1
                    cur_note_secs += cur_note_dur
                    offset = 0.
                else:
                    offset = cur_clip_secs + split_tag['duration'] - cur_note_secs
                    cur_note_secs += offset
                    cur_clip_secs += split_tag['duration']
                    split_note_seq.append(
                        (note_seq[idx][0], offset)
                    )
                    break
            if idx == len(note_seq) and cur_clip_secs + split_tag['duration'] - cur_note_secs >= tol:
                split_note_seq.append(
                    ('rest', cur_clip_secs + split_tag['duration'] - cur_note_secs)
                )
            if split_tag['filename'] not in transcriptions:
                continue
            dst_dict = transcriptions[split_tag['filename']]
            dst_dict['note_seq'] = ' '.join(n[0] for n in split_note_seq)
            dst_dict['note_dur'] = ' '.join(str(n[1]) for n in split_note_seq)

    with open(csv_file, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num', 'note_seq', 'note_dur'])
        writer.writeheader()
        writer.writerows(v for _, v in transcriptions.items())


if __name__ == '__main__':
    extract_midi()
