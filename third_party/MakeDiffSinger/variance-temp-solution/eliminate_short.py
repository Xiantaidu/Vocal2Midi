import json
import pathlib
from collections import OrderedDict

import click


@click.command(help='Eliminate short slur notes in DS files')
@click.argument('ds_dir', metavar='DS_DIR')
@click.argument('threshold', type=float, metavar='THRESHOLD')
def eliminate_short(
        ds_dir,
        threshold: float
):
    ds_dir = pathlib.Path(ds_dir).resolve()
    assert ds_dir.exists(), 'The directory of DS files does not exist.'

    for ds in ds_dir.iterdir():
        if not ds.is_file() or ds.suffix != '.ds':
            continue

        with open(ds, 'r', encoding='utf8') as f:
            params = json.load(f)
        if not isinstance(params, list):
            params = [params]
        params = [OrderedDict(p) for p in params]

        for param in params:
            note_list = [
                (note, float(dur), bool(int(slur)))
                for note, dur, slur
                in zip(param['note_seq'].split(), param['note_dur'].split(), param['note_slur'].split())
            ]
            word_note_div = []
            cache = []
            for note in note_list:
                if len(cache) == 0 or note[2]:
                    cache.append(note)
                else:
                    word_note_div.append(cache)
                    cache = [note]
            if len(cache) > 0:
                word_note_div.append(cache)

            word_note_div_new = []
            for i in range(len(word_note_div)):
                word_note_seq = word_note_div[i]
                if len(word_note_seq) == 1 or all(n[1] < threshold for n in word_note_seq):
                    word_note_div_new.append(word_note_seq)
                    continue

                word_note_seq_new = []
                j = 0
                prev_merge = 0.
                while word_note_seq[j][1] < threshold:
                    # Enumerate leading short notes
                    prev_merge += word_note_seq[j][1]
                    j += 1
                # Iter note sequence
                while j < len(word_note_seq):
                    k = j + 1
                    while k < len(word_note_seq) and word_note_seq[k][1] < threshold:
                        k += 1
                    post_merge = sum(n[1] for n in word_note_seq[j + 1: k])
                    if k < len(word_note_seq):
                        post_merge /= 2
                    word_note_seq_new.append(
                        (word_note_seq[j][0], prev_merge + word_note_seq[j][1] + post_merge, False)
                    )
                    prev_merge = post_merge
                    j = k

                word_note_div_new.append(word_note_seq_new)

            note_seq_new = []
            note_dur_new = []
            note_slur_new = []
            for word_note_seq in word_note_div_new:
                note_seq_new += [n[0] for n in word_note_seq]
                note_dur_new += [n[1] for n in word_note_seq]
                note_slur_new += [pos > 0 for pos in range(len(word_note_seq))]
            param['note_seq'] = ' '.join(note_seq_new)
            param['note_dur'] = ' '.join(str(round(d, 6)) for d in note_dur_new)
            param['note_slur'] = ' '.join(str(int(s)) for s in note_slur_new)

        with open(ds, 'w', encoding='utf8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    eliminate_short()
