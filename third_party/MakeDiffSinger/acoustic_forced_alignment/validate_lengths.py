import librosa
import tqdm
import os
import pathlib

import click


def length(src: str):
    if os.path.isfile(src) and src.endswith('.wav'):
        return librosa.get_duration(filename=src) / 3600.
    elif os.path.isdir(src):
        total = 0
        for ch in [os.path.join(src, c) for c in os.listdir(src)]:
            total += length(ch)
        return total
    return 0


# noinspection PyShadowingBuiltins
@click.command(help='Validate segment lengths')
@click.option('--dir', required=True, help='Path to the segments directory')
def validate_lengths(dir):
    dir = pathlib.Path(dir)
    assert dir.exists() and dir.is_dir(), 'The chosen path does not exist or is not a directory.'

    reported = False
    filelist = list(dir.glob('*.wav'))
    total_length = 0.
    for file in tqdm.tqdm(filelist):
        wave_seconds = librosa.get_duration(filename=str(file))
        if wave_seconds < 2.:
            reported = True
            print(f'Too short! \'{file}\' has a length of {round(wave_seconds, 1)} seconds!')
        if wave_seconds > 20.:
            reported = True
            print(f'Too long! \'{file}\' has a length of {round(wave_seconds, 1)} seconds!')
        total_length += wave_seconds / 3600.

    print(f'Found {len(filelist)} segments with total length of {round(total_length, 2)} hours.')

    if not reported:
        print('All segments have proper length.')


if __name__ == '__main__':
    validate_lengths()
