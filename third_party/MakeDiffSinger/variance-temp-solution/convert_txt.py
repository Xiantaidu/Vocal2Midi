import csv
import pathlib

import click


@click.command(help='Migrate transcriptions.txt in old datasets to transcriptions.csv')
@click.argument('input_txt', metavar='INPUT')
def convert_txt(
        input_txt: str
):
    input_txt = pathlib.Path(input_txt).resolve()
    assert input_txt.exists(), 'The input file does not exist.'
    with open(input_txt, 'r', encoding='utf8') as f:
        utterances = f.readlines()
    utterances = [u.split('|') for u in utterances]
    utterances = [
        {
            'name': u[0],
            'ph_seq': u[2],
            'ph_dur': u[5]
        }
        for u in utterances
    ]

    with open(input_txt.with_suffix('.csv'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur'])
        writer.writeheader()
        writer.writerows(utterances)


if __name__ == '__main__':
    convert_txt()
