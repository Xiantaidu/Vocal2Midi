import csv
import pathlib

import click


@click.command(help='Add ph_num attribute into transcriptions.csv')
@click.argument('transcription', metavar='TRANSCRIPTIONS')
@click.option('--dictionary', metavar='DICTIONARY')
@click.option('--vowels', metavar='FILE')
@click.option('--consonants', metavar='FILE')
def add_ph_num(
        transcription: str,
        dictionary: str = None,
        vowels: str = None,
        consonants: str = None
):
    assert dictionary is not None or (vowels is not None and consonants is not None), \
        'Either dictionary file or vowels and consonants file should be specified.'
    if dictionary is not None:
        dictionary = pathlib.Path(dictionary).resolve()
        vowels = {'SP', 'AP'}
        consonants = set()
        with open(dictionary, 'r', encoding='utf8') as f:
            rules = f.readlines()
        for r in rules:
            syllable, phonemes = r.split('\t')
            phonemes = phonemes.split()
            assert len(phonemes) <= 2, 'We only support two-phase dictionaries for automatically adding ph_num.'
            if len(phonemes) == 1:
                vowels.add(phonemes[0])
            else:
                consonants.add(phonemes[0])
                vowels.add(phonemes[1])
    else:
        vowels_path = pathlib.Path(vowels).resolve()
        consonants_path = pathlib.Path(consonants).resolve()
        vowels = {'SP', 'AP'}
        consonants = set()
        with open(vowels_path, 'r', encoding='utf8') as f:
            vowels.update(f.read().split())
        with open(consonants_path, 'r', encoding='utf8') as f:
            consonants.update(f.read().split())
        overlapped = vowels.intersection(consonants)
        assert len(vowels.intersection(consonants)) == 0, \
            'Vowel set and consonant set overlapped. The following phonemes ' \
            'appear both as vowels and as consonants:\n' \
            f'{sorted(overlapped)}'

    transcription = pathlib.Path(transcription).resolve()
    items: list[dict] = []
    with open(transcription, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            items.append(item)

    for item in items:
        item: dict
        ph_seq = item['ph_seq'].split()
        for ph in ph_seq:
            assert ph in vowels or ph in consonants, \
                f'Invalid phoneme symbol \'{ph}\' in \'{item["name"]}\'.'
        ph_num = []
        i = 0
        while i < len(ph_seq):
            j = i + 1
            while j < len(ph_seq) and ph_seq[j] in consonants:
                j += 1
            ph_num.append(str(j - i))
            i = j
        item['ph_num'] = ' '.join(ph_num)

    with open(transcription, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)


if __name__ == '__main__':
    add_ph_num()
