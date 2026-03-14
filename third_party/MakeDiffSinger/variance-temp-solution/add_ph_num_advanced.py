import csv
import pathlib
from typing import Tuple, List

import click
import textgrid


class RuleTerm:
    def __init__(self, key: str, is_wildcard: bool = False):
        self.key = key
        self.is_wildcard = is_wildcard

    def __repr__(self):
        if self.is_wildcard:
            return f"*{self.key}"
        return self.key


class QueryTerm:
    def __init__(self, specified_key: str = None, wildcard_key: str = None):
        self.specified_key = specified_key
        self.wildcard_key = wildcard_key

    def __repr__(self):
        return str((self.specified_key, self.wildcard_key))


class TrieNode:
    def __init__(self):
        self.children = {}
        self.wildcards = {}
        self.value = None

    def __setitem__(self, key: Tuple[RuleTerm], value):
        if not key:
            self.value = value
        else:
            term, *key = key
            if term.is_wildcard:
                if term.key not in self.wildcards:
                    self.wildcards[term.key] = TrieNode()
                self.wildcards[term.key][(*key,)] = value
            else:
                if term.key not in self.children:
                    self.children[term.key] = TrieNode()
                self.children[term.key][(*key,)] = value

    def __getitem__(self, key: Tuple[RuleTerm]):
        if not key:
            return self.value
        term, *key = key
        if term.is_wildcard:
            if term.key not in self.wildcards:
                return None
            return self.wildcards[term.key][(*key,)]
        if term.key not in self.children:
            return None
        return self.children[term.key][(*key,)]

    def find_paths(self, query: List[QueryTerm]) -> List[RuleTerm]:
        if not query:
            return []
        term, *query = query
        paths = []
        if term.specified_key in self.children:
            if self.children[term.specified_key].value is not None:
                paths.append([RuleTerm(term.specified_key, False)])
            for path in self.children[term.specified_key].find_paths(query):
                paths.append([RuleTerm(term.specified_key, False), *path])
        if term.wildcard_key in self.wildcards:
            if self.wildcards[term.wildcard_key].value is not None:
                paths.append([RuleTerm(term.wildcard_key, True)])
            for path in self.wildcards[term.wildcard_key].find_paths(query):
                paths.append([RuleTerm(term.wildcard_key, True), *path])
        return paths

    def find_best_path(self, query: Tuple[QueryTerm]) -> List[RuleTerm]:
        paths = self.find_paths(list(query))
        return max(
            paths,
            default=None,
            key=lambda p: (
                len(p),
                sum(not t.is_wildcard for t in p),
                min(enumerate(p), key=lambda e: (not e[1].is_wildcard, e[0]))[0]
            )
        )


CONSONANT = 0
VOWEL = 1
LIQUID = 2


@click.command(help='Add ph_num attribute into transcriptions.csv (advanced mode)')
@click.argument(
    'transcription',
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    metavar='TRANSCRIPTIONS'
)
@click.option(
    '--tg', required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    help='Path to TextGrids'
)
@click.option(
    '--vowels',
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    metavar='FILE',
    help='Path to the file containing vowels'
)
@click.option(
    '--consonants',
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    metavar='FILE',
    help='Path to the file containing consonants'
)
@click.option(
    '--liquids',
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    metavar='FILE',
    help='Path to the file containing liquids'
)
def add_ph_num_advanced(
        transcription: pathlib.Path,
        tg: pathlib.Path,
        vowels: pathlib.Path = None,
        consonants: pathlib.Path = None,
        liquids: pathlib.Path = None
):
    with open(transcription, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        items = list(reader)
    phoneme_type_map = {
        'AP': VOWEL,
        'SP': VOWEL,
    }
    if vowels is not None:
        with open(vowels, 'r', encoding='utf8') as f:
            for v in f.read().split():
                phoneme_type_map[v] = VOWEL
    if consonants is not None:
        with open(consonants, 'r', encoding='utf8') as f:
            for c in f.read().split():
                phoneme_type_map[c] = CONSONANT
    if liquids is not None:
        with open(liquids, 'r', encoding='utf8') as f:
            for l in f.read().split():
                phoneme_type_map[l] = LIQUID

    trie = TrieNode()
    trie[(
        RuleTerm(VOWEL, True),
    )] = [0]
    trie[(
        RuleTerm(CONSONANT, True),
        RuleTerm(LIQUID, True),
        RuleTerm(VOWEL, True),
    )] = [1]
    trie[(
        RuleTerm(LIQUID, True),
        RuleTerm(LIQUID, True),
        RuleTerm(VOWEL, True),
    )] = [1]

    for item in items:
        name = item['name']
        tg_path = tg / f"{name}.TextGrid"
        tg_obj = textgrid.TextGrid()
        tg_obj.read(tg_path, encoding='utf8')
        words_tier = tg_obj[0]
        phones_tier = tg_obj[1]

        if item['ph_seq'].split() != [i.mark for i in phones_tier]:
            raise ValueError(f"Error: ph_seq mismatch in item: {name}")
        for phone_idx, phone_interval in enumerate(phones_tier):
            if phone_interval.mark not in phoneme_type_map:
                raise ValueError(
                    f"Error: invalid phone in item: {name}, index: {phone_idx}, phone: {phone_interval.mark}"
                )

        is_onset = []
        for word_idx, word_interval in enumerate(words_tier):
            start_ph_idx = min(
                enumerate(tg_obj[1]),
                key=lambda e: abs(e[1].minTime - word_interval.minTime)
            )[0]
            end_ph_idx = min(
                enumerate(tg_obj[1]),
                key=lambda e: abs(e[1].maxTime - word_interval.maxTime)
            )[0]
            if phones_tier[start_ph_idx].minTime != word_interval.minTime:
                raise ValueError(
                    f"Error: word minTime not aligned to phone minTime in item: "
                    f"{name}, index: {word_idx}, word: {word_interval.mark}"
                )
            if phones_tier[end_ph_idx].maxTime != word_interval.maxTime:
                raise ValueError(
                    f"Error: word maxTime not aligned to phone maxTime in item: "
                    f"{name}, index: {word_idx}, word: {word_interval.mark}"
                )
            word_phones = [i.mark for i in phones_tier[start_ph_idx:end_ph_idx + 1]]
            i = 0
            while i < len(word_phones):
                query = [
                    QueryTerm(specified_key=ph, wildcard_key=phoneme_type_map[ph])
                    for ph in word_phones[i:]
                ]
                best_path = trie.find_best_path(query)
                if not best_path:
                    is_onset.append(False)
                    i += 1
                    continue
                onsets = trie[best_path]
                is_onset.extend(
                    j in onsets
                    for j in range(len(best_path))
                )
                i += len(best_path)
        acc = 0
        ph_num = []
        for flag in is_onset:
            if flag:
                if acc > 0:
                    ph_num.append(acc)
                acc = 1
            else:
                acc += 1
        if acc > 0:
            ph_num.append(acc)
        item['ph_num'] = ' '.join(str(n) for n in ph_num)

    with open(transcription, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)


if __name__ == '__main__':
    add_ph_num_advanced()
