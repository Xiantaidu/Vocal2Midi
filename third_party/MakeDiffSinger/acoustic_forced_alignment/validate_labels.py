import pathlib

import click
import matplotlib.pyplot as plt
import tqdm

import distribution


# noinspection PyShadowingBuiltins
@click.command(help='Validate transcription labels')
@click.option('--dir', required=True, help='Path to the segments directory')
@click.option('--dictionary', required=True, help='Path to the dictionary file')
def validate_labels(dir, dictionary):
    # Load dictionary
    dict_path = pathlib.Path(dictionary)
    with open(dict_path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    dictionary = {}
    phoneme_set = set()
    for r in rules:
        phonemes = r[1].split()
        dictionary[r[0]] = phonemes
        phoneme_set.update(phonemes)

    # Run checks
    check_failed = False
    covered = set()
    phoneme_map = {}
    for ph in sorted(phoneme_set):
        phoneme_map[ph] = 0

    segments_dir = pathlib.Path(dir)
    filelist = list(segments_dir.glob('*.wav'))

    for file in tqdm.tqdm(filelist):
        filename = file.stem
        annotation = file.with_suffix('.lab')
        if not annotation.exists():
            print(f'No annotation found for \'{filename}\'!')
            check_failed = True
            continue
        with open(annotation, 'r', encoding='utf8') as f:
            syllables = f.read().strip().split()
        if not syllables:
            print(f'Annotation file \'{annotation}\' is empty!')
            check_failed = True
        else:
            oov = []
            for s in syllables:
                if s not in dictionary:
                    oov.append(s)
                else:
                    for ph in dictionary[s]:
                        phoneme_map[ph] += 1
                    covered.update(dictionary[s])
            if oov:
                print(f'Syllable(s) {oov} not allowed in annotation file \'{annotation}\'')
                check_failed = True

    # Phoneme coverage
    uncovered = phoneme_set - covered
    if uncovered:
        print(f'The following phonemes are not covered!')
        print(sorted(uncovered))
        print('Please add more recordings to cover these phonemes.')
        check_failed = True

    if not check_failed:
        print('All annotations are well prepared.')

    phoneme_list = sorted(phoneme_set)
    phoneme_counts = [phoneme_map[ph] for ph in phoneme_list]
    distribution.draw_distribution(
        title='Phoneme Distribution Summary',
        x_label='Phoneme',
        y_label='Number of occurrences',
        items=phoneme_list,
        values=phoneme_counts
    )
    phoneme_summary = segments_dir / 'phoneme_distribution.jpg'
    plt.savefig(fname=phoneme_summary,
                bbox_inches='tight',
                pad_inches=0.25)
    print(f'Phoneme distribution summary saved to {phoneme_summary}')


if __name__ == '__main__':
    validate_labels()
