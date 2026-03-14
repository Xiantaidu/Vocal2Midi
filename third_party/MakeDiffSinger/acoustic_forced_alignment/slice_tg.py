import pathlib

import click
import librosa
import soundfile
import textgrid
import tqdm


@click.command(help='Slice 3-tier TextGrids and long recordings into segmented 2-tier TextGrids and wavs')
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
    '--preserve_sentence_names', is_flag=True,
    help='Whether to use sentence marks as filenames (will be re-numbered by default)'
)
@click.option(
    '--digits', required=False, type=int, default=3,
    help='Number of suffix digits (defaults to 3, will be padded with zeros on the left)'
)
@click.option(
    '--wav_subtype', required=False, default='PCM_16',
    help='Wav subtype (defaults to PCM_16)'
)
@click.option(
    '--overwrite', is_flag=True,
    help='Overwrite existing files'
)
def slice_tg(wavs, tg, out, preserve_sentence_names, digits, wav_subtype, overwrite):
    wav_path_in = pathlib.Path(wavs)
    tg_path_in = wav_path_in if tg is None else pathlib.Path(tg)
    del tg
    sliced_path_out = pathlib.Path(out)
    sliced_path_out.mkdir(parents=True, exist_ok=True)
    for tg_file in tqdm.tqdm(tg_path_in.glob('*.TextGrid')):
        tg = textgrid.TextGrid()
        tg.read(tg_file)
        wav, sr = librosa.load((wav_path_in / tg_file.name).with_suffix('.wav'), sr=None)
        sentences_tier = tg[0]
        words_tier = tg[1]
        phones_tier = tg[2]
        idx = 0
        for sentence in sentences_tier:
            if sentence.mark == '':
                continue
            sentence_tg = textgrid.TextGrid()
            sentence_words_tier = textgrid.IntervalTier(name='words')
            sentence_phones_tier = textgrid.IntervalTier(name='phones')
            for word in words_tier:
                min_time = max(sentence.minTime, word.minTime)
                max_time = min(sentence.maxTime, word.maxTime)
                if min_time >= max_time:
                    continue
                sentence_words_tier.add(
                    minTime=min_time - sentence.minTime, maxTime=max_time - sentence.minTime, mark=word.mark
                )
            for phone in phones_tier:
                min_time = max(sentence.minTime, phone.minTime)
                max_time = min(sentence.maxTime, phone.maxTime)
                if min_time >= max_time:
                    continue
                sentence_phones_tier.add(
                    minTime=min_time - sentence.minTime, maxTime=max_time - sentence.minTime, mark=phone.mark
                )
            sentence_tg.append(sentence_words_tier)
            sentence_tg.append(sentence_phones_tier)

            if preserve_sentence_names:
                tg_file_out = sliced_path_out / f'{sentence.mark}.TextGrid'
                wav_file_out = tg_file_out.with_suffix('.wav')
            else:
                tg_file_out = sliced_path_out / f'{tg_file.stem}_{str(idx).zfill(digits)}.TextGrid'
                wav_file_out = tg_file_out.with_suffix('.wav')
            if tg_file_out.exists() and not overwrite:
                raise FileExistsError(str(tg_file_out))
            if wav_file_out.exists() and not overwrite:
                raise FileExistsError(str(wav_file_out))

            sentence_tg.write(tg_file_out)
            sentence_wav = wav[int(sentence.minTime * sr): min(wav.shape[0], int(sentence.maxTime * sr) + 1)]
            soundfile.write(
                wav_file_out,
                sentence_wav, samplerate=sr, subtype=wav_subtype
            )
            idx += 1


if __name__ == '__main__':
    slice_tg()
