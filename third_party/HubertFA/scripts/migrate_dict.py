import csv
import pathlib
from typing import List, TextIO

import click
import textgrid
import yaml


class PhonemeDictionary:
    def __init__(self, f: TextIO):
        rules: dict = {}
        phonemes = set()
        for line in f:
            word, phs = line.strip().split("\t")
            word = word.strip()
            phs = phs.split()
            rules[word] = phs
            for ph in phs:
                phonemes.add(ph)
        self.rules = rules
        self.phonemes = phonemes


def load_dictionary(dict_path: pathlib.Path) -> PhonemeDictionary:
    with open(dict_path, "r", encoding="utf8") as f:
        return PhonemeDictionary(f)


def compare_dictionaries(d1: PhonemeDictionary, d2: PhonemeDictionary) -> List[str]:
    word_diff = []
    for word in d1.rules:
        if word not in d2.rules:
            continue
        phs1 = d1.rules[word]
        phs2 = d2.rules[word]
        if phs1 != phs2:
            word_diff.append(word)
    return word_diff


def load_and_validate_dictionaries(source_path: pathlib.Path, target_path: pathlib.Path):
    src_dict = load_dictionary(source_path)
    tgt_dict = load_dictionary(target_path)
    word_diff = compare_dictionaries(src_dict, tgt_dict)
    for word in word_diff:
        if len(src_dict.rules[word]) != len(tgt_dict.rules[word]):
            raise ValueError(
                f"Cannot migrate transcription automatically due to different number of phonemes for word '{word}'. "
                f"Source: {src_dict.rules[word]}, Target: {tgt_dict.rules[word]}"
            )
    return src_dict, tgt_dict, word_diff


def shared_dict_options(func):
    options = [
        click.option(
            "--source-dict", required=True, type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            help="Path to the source dictionary file."
        ),
        click.option(
            "--target-dict", required=True, type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            help="Path to the target dictionary file."
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def shared_save_file_option(func):
    option = click.option(
        "--save-path", required=False, type=click.Path(
            dir_okay=False, file_okay=True, writable=True, path_type=pathlib.Path
        ),
        help="Path to save the migrated file. If not specified, use the original file path."
    )
    func = option(func)
    return func


def shared_save_dir_option(func):
    option = click.option(
        "--save-path", required=True, type=click.Path(
            dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path
        ),
        help="Directory to save the migrated files."
    )
    func = option(func)
    return func


def shared_overwrite_file_option(func):
    option = click.option(
        "--overwrite", is_flag=True, help="Overwrite the existing file if save path exists."
    )
    func = option(func)
    return func


def replace_for_sequence(seq: list, src_dict: PhonemeDictionary, tgt_dict: PhonemeDictionary, word_diff: List[str]):
    for word in word_diff:
        n = len(src_dict.rules[word])
        for i in range(len(seq) - n + 1):
            if seq[i:i + n] == src_dict.rules[word]:
                seq[i:i + n] = tgt_dict.rules[word]



@click.group(help="Migrate dictionary for dataset labels.")
def cli():
    pass


@cli.command(
    name="tg",
    help="Migrate dictionary for TextGrids\n"
         "(Note: This function only considers the \"phones\" tier in the TextGrids.)"
)
@click.argument(
    "textgrids", type=click.Path(
        exists=True, dir_okay=True, file_okay=False, readable=True, path_type=pathlib.Path
    )
)
@shared_dict_options
@shared_save_dir_option
@shared_overwrite_file_option
def migrate_for_textgrids(
        textgrids: pathlib.Path,
        source_dict: pathlib.Path,
        target_dict: pathlib.Path,
        save_path: pathlib.Path,
        overwrite: bool = False,
):
    src_dict, tgt_dict, word_diff = load_and_validate_dictionaries(source_dict, target_dict)

    for tg_file in textgrids.glob("*.TextGrid"):
        save_file = save_path / tg_file.name
        if not overwrite and save_file.exists():
            raise FileExistsError(f"File {save_file} already exists. Use --overwrite to overwrite it.")

        tg = textgrid.TextGrid()
        tg.read(tg_file, encoding="utf8")
        phones_tier: textgrid.IntervalTier = tg.getFirst("phones")
        if phones_tier is None:
            raise ValueError(f"TextGrid {tg_file} has no 'phones' tier.")
        ph_seq = [interval.mark for interval in phones_tier]
        new_ph_seq = ph_seq.copy()
        replace_for_sequence(new_ph_seq, src_dict, tgt_dict, word_diff)
        for interval, new_ph in zip(phones_tier, new_ph_seq):
            interval.mark = new_ph

        with open(save_file, "w", encoding="utf8") as f:
            tg.write(f)


@cli.command(
    name="csv",
    help="Migrate dictionary for transcriptions.csv"
)
@click.argument(
    "transcription", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    )
)
@shared_dict_options
@shared_save_file_option
@shared_overwrite_file_option
def migrate_for_transcriptions_csv(
        transcription: pathlib.Path,
        source_dict: pathlib.Path,
        target_dict: pathlib.Path,
        save_path: pathlib.Path = None,
        overwrite: bool = False,
):
    if save_path is None:
        save_path = transcription

    if not overwrite and save_path.exists():
        raise FileExistsError(f"File {save_path} already exists. Use --overwrite to overwrite it.")

    with open(transcription, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        items = list(reader)

    src_dict, tgt_dict, word_diff = load_and_validate_dictionaries(source_dict, target_dict)

    for item in items:
        ph_seq = item["ph_seq"].split()
        new_ph_seq = ph_seq.copy()
        replace_for_sequence(new_ph_seq, src_dict, tgt_dict, word_diff)
        item["ph_seq"] = " ".join(new_ph_seq)

    with open(save_path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


@cli.command(
    name="yaml",
    help="Migrate OpenUTAU dsdict.yaml dictionary\n"
         "(Note: this function only changes \"entries\" in the YAML. \"symbols\" should be manually edited if needed.)"
)
@click.argument(
    "dsdict", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    )
)
@click.option(
    "--lang", type=str, required=True,
    help="Language code to filter entries to be migrated. If --main option is given, "
         "all entries without language prefix will be migrated."
)
@click.option(
    "--main", is_flag=True,
    help="If set, the given language code will be treated as the main language of the given dsdict.yaml."
)
@shared_dict_options
@shared_save_file_option
@shared_overwrite_file_option
def migrate_for_dsdict_yaml(
        dsdict: pathlib.Path,
        lang: str,
        main: bool,
        source_dict: pathlib.Path,
        target_dict: pathlib.Path,
        save_path: pathlib.Path = None,
        overwrite: bool = False,
):
    if save_path is None:
        save_path = dsdict
    if not overwrite and save_path.exists():
        raise FileExistsError(f"File {save_path} already exists. Use --overwrite to overwrite it.")

    with open(dsdict, "r", encoding="utf8") as f:
        obj = yaml.safe_load(f)
    entries = obj.get("entries")
    if not isinstance(entries, list):
        raise ValueError("The YAML file contains no \"entries\" key or it is not a list.")

    src_dict, tgt_dict, word_diff = load_and_validate_dictionaries(source_dict, target_dict)
    for entry in entries:
        word = entry["grapheme"]
        if main and word in word_diff:
            pass
        elif word.startswith(f"{lang}/"):
            real_word = word[len(lang) + 1:]
            if real_word in word_diff:
                word = real_word
            else:
                continue
        else:
            continue
        entry["phonemes"] = [
            f"{lang}/{ph}"
            for ph in tgt_dict.rules[word]
        ]

    with open(save_path, "w", encoding="utf8") as f:
        yaml.safe_dump(obj, f, indent=2, allow_unicode=True, sort_keys=False)


if __name__ == '__main__':
    cli()
