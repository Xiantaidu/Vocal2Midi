import pathlib
import re
from typing import Dict, List

import click
import librosa
import natsort
import numpy
import soundfile
import textgrid
import tqdm


def remove_suffix(string, suffix_pattern):
    match = re.search(f'{suffix_pattern}$', string)
    if not match:
        return string
    return string[:-len(match.group())]


@click.command(help='Combine segmented 2-tier TextGrids and wavs into 3-tier TextGrids and long wavs')
@click.option(
    '--wavs', required=True,
    help='Directory containing the segmented wav files'
)
@click.option(
    '--tg', required=False,
    help='Directory containing the segmented TextGrid files (defaults to wav directory)'
)
@click.option(
    '--out', required=True,
    help='Path to output directory for combined files'
)
@click.option(
    '--suffix', required=False, default=r'_\d+',
    help='Filename suffix pattern for file combination'
)
@click.option(
    '--wav_subtype', required=False, default='PCM_16',
    help='Wav subtype (defaults to PCM_16)'
)
@click.option(
    '--overwrite', is_flag=True,
    help='Overwrite existing files'
)
def combine_tg(wavs, tg, out, suffix, wav_subtype, overwrite):
    wav_path_in = pathlib.Path(wavs)
    tg_path_in = wav_path_in if tg is None else pathlib.Path(tg)
    del tg
    combined_path_out = pathlib.Path(out)
    combined_path_out.mkdir(parents=True, exist_ok=True)
    filelist: Dict[str, List[pathlib.Path]] = {}
    for tg_file in tg_path_in.glob('*.TextGrid'):
        stem = remove_suffix(tg_file.stem, suffix)
        if stem not in filelist:
            filelist[stem] = [tg_file]
        else:
            filelist[stem].append(tg_file)
    for name, files in tqdm.tqdm(sorted(filelist.items(), key=lambda kv: kv[0])):
        wav_segments = []
        tg = textgrid.TextGrid()
        sentences_tier = textgrid.IntervalTier(name='sentences')
        words_tier = textgrid.IntervalTier(name='words')
        phones_tier = textgrid.IntervalTier(name='phones')
        sentence_start = 0.
        sr = None
        for tg_file in natsort.natsorted(files):
            wav_file = (wav_path_in / tg_file.name).with_suffix('.wav')
            waveform, sr_ = librosa.load(wav_file, sr=None)
            if sr is None:
                sr = sr_
            else:
                assert sr_ == sr, f'Cannot combine \'{tg_file.stem}\': incompatible samplerate ({sr_} != {sr})'
            sentence_end = waveform.shape[0] / sr + sentence_start
            wav_segments.append(waveform)
            sentences_tier.add(minTime=sentence_start, maxTime=sentence_end, mark=wav_file.stem)
            sentence_tg = textgrid.TextGrid()
            sentence_tg.read(tg_file)
            start = sentence_start
            for j, word in enumerate(sentence_tg[0]):
                if j == len(sentence_tg[0]) - 1:
                    end = sentence_end
                else:
                    end = start + word.duration()
                words_tier.add(minTime=start, maxTime=end, mark=word.mark)
                start = end
            start = sentence_start
            for j, phone in enumerate(sentence_tg[1]):
                if j == len(sentence_tg[1]) - 1:
                    end = sentence_end
                else:
                    end = start + phone.duration()
                phones_tier.add(minTime=start, maxTime=end, mark=phone.mark)
                start = end
            sentence_start = sentence_end
        tg.append(sentences_tier)
        tg.append(words_tier)
        tg.append(phones_tier)

        tg_file_out = combined_path_out / f'{name}.TextGrid'
        wav_file_out = tg_file_out.with_suffix('.wav')
        if wav_file_out.exists() and not overwrite:
            raise FileExistsError(str(wav_file_out))
        if tg_file_out.exists() and not overwrite:
            raise FileExistsError(str(tg_file_out))

        tg.write(tg_file_out)
        full_wav = numpy.concatenate(wav_segments)
        soundfile.write(wav_file_out, full_wav, samplerate=sr, subtype=wav_subtype)


if __name__ == '__main__':
    combine_tg()
