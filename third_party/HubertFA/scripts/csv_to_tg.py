import csv
import os
import pathlib
import shutil

import click
import textgrid


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None


class Trie:
    def __init__(self, dict_path):
        self.depth = 0
        self.root = TrieNode()
        self.build_trie(dict_path)

    def insert(self, phonemes, raw_word):
        node = self.root
        for char in phonemes:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.value = raw_word

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end_of_word:
            return node.value
        return None

    def build_trie(self, dict_path):
        for word, raw in self.read_dictionary(dict_path):
            self.insert(word, raw)

    def read_dictionary(self, filename):
        dictionary = []
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:
                    raw = line[0]
                    phonemes = line[1].split()
                    self.depth = max(self.depth, len(phonemes))
                    dictionary.append((phonemes, raw))
        return dictionary


@click.command()
@click.option(
    "--ds_csv", type=str, help="path to the input csv file"
)
@click.option(
    "--spk_name", type=str, help="name for the output folder"
)
@click.option(
    "--folder", default="data/evaluate", type=str, help="path to the output folder"
)
@click.option(
    "--dictionary",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
@click.option(
    "--ignore",
    type=str,
    default="",  # AP,SP,<AP>,<SP>,,pau,cl
    help="Ignored phone marks, split by commas",
    show_default=True,
)
def main(ds_csv, spk_name, folder, dictionary, ignore):
    if ignore == "":
        ignore_phonemes = ['SP', 'GS']
    else:
        ignore_phonemes = ignore.split(",")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist")

    output_folder = pathlib.Path(folder) / spk_name
    tg_folder = output_folder / "TextGrid"
    wav_folder = output_folder / "wavs"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(tg_folder, exist_ok=True)
    os.makedirs(wav_folder, exist_ok=True)

    trie = Trie(dictionary)

    for p in ['AP', 'SP', 'EP', 'GS']:
        trie.insert([p], p)

    with open(ds_csv, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        reader.__next__()
        for row in reader:
            out = True
            name = row[0]
            cursor = 0
            out_tg = textgrid.TextGrid()
            tier_words = textgrid.IntervalTier(name="words")
            tier_phones = textgrid.IntervalTier(name="phones")
            phones = row[1].split(" ")
            ph_dur = row[2].split(" ")
            if len(phones) != len(ph_dur):
                print(f"{name}: number of phones and duration are not equal, skipping.")
                continue

            phone_temp = []
            phone_duration = []
            words = []
            lab = []
            for phone, dur in zip(phones, ph_dur):
                if len(phone_temp) > trie.depth:
                    print(f"{file}: {phone_temp}")
                phone_temp.append(phone)
                phone_duration.append(float(dur))
                node = trie.search(phone_temp)
                if node is not None:
                    if node not in ignore_phonemes:
                        words.append(node)
                        lab.append(node)
                    else:
                        cursor_added = round(cursor + sum(phone_duration), 6)
                        tier_words.intervals.append(textgrid.Interval(cursor, cursor_added, "SP"))
                        tier_phones.intervals.append(textgrid.Interval(cursor, cursor_added, "SP"))
                        cursor = cursor_added
                        phone_temp.clear()
                        phone_duration.clear()
                        continue
                    tier_words.intervals.append(textgrid.Interval(cursor, round(cursor + sum(phone_duration), 6), node))
                    for i, p in enumerate(phone_temp):
                        cursor_added = round(cursor + phone_duration[i], 6)
                        tier_phones.intervals.append(textgrid.Interval(cursor, cursor_added, p))
                        cursor = cursor_added
                    phone_temp.clear()
                    phone_duration.clear()
                elif len(phone_temp) > trie.depth:
                    print(f"error:\r\n{name}: {phone_temp}")
                    print("exit with error, please check the phonemes in the textgrid file\r\n")
                    out = False
            if out:
                out_tg.tiers.insert(0, tier_words)
                out_tg.tiers.insert(1, tier_phones)
                out_tg_path = os.path.join(tg_folder, f"{name}.TextGrid")
                out_tg.write(out_tg_path)
                out_wav_path = os.path.join(wav_folder, f"{name}.wav")
                shutil.copy(os.path.join(pathlib.Path(ds_csv).parent, f"wavs/{name}.wav"), out_wav_path)

                out_lab_path = os.path.join(wav_folder, f"{name}.lab")
                with open(out_lab_path, 'w', encoding='utf-8') as f:
                    f.write(" ".join(lab))


if __name__ == '__main__':
    main()
