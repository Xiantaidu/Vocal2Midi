import click

from tools.lyric_matcher import LyricMatchingPipeline


@click.command(help='Match original lyrics with ASR results and generate Minlabel JSON files')
@click.option('--lyric_folder', required=True, help='Folder containing lyric files (*.txt).')
@click.option('--lab_folder', required=True, help='Folder containing ASR result files (*.lab).')
@click.option('--json_folder', required=True, help='Output folder for JSON files.')
@click.option('--language', required=True, type=click.Choice(['zh', 'en']),
              help='Language: zh(Chinese), en(English).')
@click.option('--diff_threshold', default=5, type=int, help='Difference threshold for printing (default: 5).')
def match_lyric(
        lyric_folder: str,
        lab_folder: str,
        json_folder: str,
        language: str,
        diff_threshold: int
) -> None:
    if not all([lyric_folder, lab_folder, json_folder]):
        raise ValueError('Missing required folder path parameters.')

    pipeline = LyricMatchingPipeline(
        lyric_folder=lyric_folder,
        lab_folder=lab_folder,
        json_folder=json_folder,
        language=language,
        diff_threshold=diff_threshold
    )
    pipeline.execute()


if __name__ == '__main__':
    match_lyric()
