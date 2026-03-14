import pathlib

import click
import textgrid
import tqdm


@click.command(help='Align words tiers in TextGrids to phones tiers')
@click.option('--tg', required=True, help='Path to TextGrids (2-tier or 3-tier format)')
@click.option('--dictionary', required=True, help='Path to the dictionary file')
@click.option(
    '--out', required=False,
    help='Path to save the aligned TextGrids (defaults to the input directory)'
)
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
def align_tg_words(tg, dictionary, out, overwrite):
    tg_path_in = pathlib.Path(tg)
    dict_path = pathlib.Path(dictionary)
    tg_path_out = pathlib.Path(out) if out is not None else tg_path_in
    tg_path_out.mkdir(parents=True, exist_ok=True)

    with open(dict_path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    dictionary = {
        'SP': ['SP'],
        'AP': ['AP']
    }
    phoneme_set = {'SP', 'AP'}
    for r in rules:
        phonemes = r[1].split()
        dictionary[r[0]] = phonemes
        phoneme_set.update(phonemes)

    for tgfile in tqdm.tqdm(tg_path_in.glob('*.TextGrid')):
        tg = textgrid.TextGrid()
        tg.read(tgfile)
        old_words_tier: textgrid.IntervalTier = tg[-2]
        if old_words_tier.name != 'words':
            raise ValueError(
                f'Invalid tier name or order in \'{tgfile}\'. '
                f'The words tier should be the 1st tier of a 2-tier TextGrid, '
                f'or the 2nd tier of a 3-tier TextGrid.'
            )
        phones_tier: textgrid.IntervalTier = tg[-1]
        new_words_tier = textgrid.IntervalTier(name='words')
        word_seq = [i.mark for i in old_words_tier]
        word_div = []
        ph_seq = [i.mark for i in phones_tier]
        ph_dur = [i.duration() for i in phones_tier]
        idx = 0
        for i, word in enumerate(word_seq):
            if word not in dictionary:
                raise ValueError(f'Error invalid word in \'{tgfile}\' at {i}: {word}')
            word_ph_seq = dictionary[word]
            ph_num = len(word_ph_seq)
            word_div.append(ph_num)
            if word_ph_seq != ph_seq[idx: idx + ph_num]:
                print(
                    f'Warning: word and phones mismatch in \'{tgfile}\' '
                    f'at word {i}, phone {idx}: {word} => {ph_seq[idx: idx + len(word_ph_seq)]}'
                )
            idx += ph_num
        for i, phone in enumerate(ph_seq):
            if phone not in phoneme_set:
                raise ValueError(f'Error: invalid phone in \'{tgfile}\' at {i}: {phone}')
        if sum(word_div) != len(ph_dur):
            raise ValueError(
                f'Error: word_div does not sum to number of phones in \'{tgfile}\'. '
                f'Check the warnings above for more detailed mismatching positions.'
            )
        start = 0.
        idx = 0
        for j in range(len(word_seq)):
            end = start + sum(ph_dur[idx: idx + word_div[j]])
            new_words_tier.add(minTime=start, maxTime=end, mark=word_seq[j])
            start = end
            idx += word_div[j]
        tg.tiers[-2] = new_words_tier
        tg_file_out = tg_path_out / tgfile.name
        if tg_file_out.exists() and not overwrite:
            raise FileExistsError(str(tg_file_out))
        tg.write(tg_file_out)


if __name__ == '__main__':
    align_tg_words()
