import tqdm
import json
import pathlib
from collections import OrderedDict

import click
import librosa
import numpy as np
import soundfile


@click.command(help='Merge clips into segments of similar length')
@click.argument('input_wavs', metavar='INPUT_WAVS')
@click.argument('output_wavs', metavar='OUTPUT_WAVS')
@click.option('--length', type=int, required=False, default=240, metavar='SECONDS')
@click.option('--sr', type=int, required=False, default=16000)
def merge_wavs(
        input_wavs, output_wavs, length, sr
):
    input_wavs = pathlib.Path(input_wavs).resolve()
    assert input_wavs.exists(), 'The input directory does not exist.'
    output_wavs = pathlib.Path(output_wavs).resolve()
    assert not output_wavs.exists() or all(False for _ in output_wavs.iterdir()), \
        'The output directory is not empty.'

    output_wavs.mkdir(parents=True, exist_ok=True)
    tags = OrderedDict()
    count = 0
    cache: list[tuple[str, np.ndarray]] = []
    cache_len = 0.

    def save_cache():
        nonlocal tags, count, cache, cache_len
        waveform_merged = np.concatenate(tuple(c[1] for c in cache))
        filename = (output_wavs / str(count).zfill(8)).with_suffix('.wav')
        soundfile.write(
            str(filename),
            waveform_merged, sr, format='WAV'
        )
        tags[str(filename.stem)] = [
            {
                'filename': c[0],
                'duration': c[1].shape[0] / sr
            }
            for c in cache
        ]
        cache.clear()
        cache_len = 0.
        count += 1

    for wav in tqdm.tqdm(input_wavs.iterdir()):
        if not wav.is_file() or wav.suffix != '.wav':
            continue
        y, _ = librosa.load(wav, sr=sr, mono=True)
        cur_len = y.shape[0] / sr
        if len(cache) > 0 and cache_len + cur_len >= length:
            save_cache()
        cache.append((wav.stem, y))
        cache_len += cur_len
    if len(cache) > 0:
        save_cache()

    tags_path = output_wavs / 'tags.json'
    with open(tags_path, 'w', encoding='utf8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
        print(f'Timestamps saved to {tags_path}')


if __name__ == '__main__':
    merge_wavs()
