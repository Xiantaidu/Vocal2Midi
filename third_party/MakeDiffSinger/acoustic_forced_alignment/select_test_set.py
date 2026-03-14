import csv
import random
from collections import defaultdict
from pathlib import Path

import click
import yaml


# noinspection PyShadowingBuiltins
@click.command(help='Randomly select test samples')
@click.argument(
    'config',
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, writable=True, path_type=Path),
    metavar="CONFIG"
)
@click.option(
    '--rel_path',
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
    default=None,
    help='Path that is relative to the paths mentioned in the config file.'
)
@click.option(
    '--min', '_min',
    show_default=True,
    type=click.IntRange(min=1),
    default=10,
    help='Minimum number of test samples.'
)
@click.option(
    '--max', '_max',
    show_default=True,
    type=click.IntRange(min=1),
    default=20,
    help='Maximum number of test samples (note that each speaker will have at least one test sample).'
)
@click.option(
    '--per_speaker',
    show_default=True,
    type=click.IntRange(min=1),
    default=4,
    help='Expected number of test samples per speaker.'
)
def select_test_set(config, rel_path, _min, _max, per_speaker):
    assert _min <= _max, 'min must be smaller or equal to max'
    with open(config, 'r', encoding='utf8') as f:
        hparams = yaml.safe_load(f)

    spk_map = None
    spk_ids = hparams['spk_ids']
    speakers = hparams['speakers']
    raw_data_dirs = list(map(Path, hparams['raw_data_dir']))
    assert isinstance(speakers, list), 'Speakers must be a list'
    assert len(speakers) == len(raw_data_dirs), \
        'Number of raw data dirs must equal number of speaker names!'
    if not spk_ids:
        spk_ids = list(range(len(raw_data_dirs)))
    else:
        assert len(spk_ids) == len(raw_data_dirs), \
            'Length of explicitly given spk_ids must equal the number of raw datasets.'
    assert max(spk_ids) < hparams['num_spk'], \
        f'Index in spk_id sequence {spk_ids} is out of range. All values should be smaller than num_spk.'

    spk_map = {}
    path_spk_map = defaultdict(list)
    for ds_id, (spk_name, raw_path, spk_id) in enumerate(zip(speakers, raw_data_dirs, spk_ids)):
        if spk_name in spk_map and spk_map[spk_name] != spk_id:
            raise ValueError(f'Invalid speaker ID assignment. Name \'{spk_name}\' is assigned '
                                f'with different speaker IDs: {spk_map[spk_name]} and {spk_id}.')
        spk_map[spk_name] = spk_id
        path_spk_map[spk_id].append((ds_id, rel_path / raw_path if rel_path else raw_path))

    training_cases = []
    for spk_raw_dirs in path_spk_map.values():
        training_case = []
        # training cases from the same speaker are grouped together
        for ds_id, raw_data_dir in spk_raw_dirs:
            with open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (raw_data_dir / 'wavs' / f'{row["name"]}.wav').exists():
                        training_case.append(f'{ds_id}:{row["name"]}')
        training_cases.append(training_case)

    test_prefixes = []
    total = min(_max, max(_min, per_speaker * len(training_cases)))
    quotient, remainder = total // len(training_cases), total % len(training_cases)
    if quotient == 0:
        test_counts = [1] * len(training_cases)
    else:
        test_counts = [quotient + 1] * remainder + [quotient] * (len(training_cases) - remainder)
    for i, count in enumerate(test_counts):
        test_prefixes += sorted(random.sample(training_cases[i], count))
    if not hparams['test_prefixes'] or click.confirm('Overwrite existing test prefixes?', abort=False):
        hparams['test_prefixes'] = test_prefixes
        hparams['num_valid_plots'] = len(test_prefixes)
        with open(config, 'w', encoding='utf8') as f:
            yaml.dump(hparams, f, sort_keys=False)
        print('Test prefixes saved.')
    else:
        print('Test prefixes not saved, aborted.')

if __name__ == '__main__':
    select_test_set()
