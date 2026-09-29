"""TextGrid export for TiFA predictions.

Mirrors HubertFA's Exporter.save_textgrids (same textgrid package, same
words/phones tier layout and {output_key}_{idx:03d} file naming) so the
chunks/textgrid debug outputs stay identical across aligners.
"""
from __future__ import annotations

import logging
import pathlib

import textgrid as tg_lib

logger = logging.getLogger(__name__)


def save_textgrids(
    predictions: list,
    output_dir: pathlib.Path,
    output_key: str,
    chunks: list | None = None,
    cancel_checker=None,
) -> None:
    """Write one TextGrid per prediction as {output_key}_{idx:03d}.TextGrid.

    ``predictions`` is a list of (wav_path, wav_length, WordList) tuples in
    chunk order, matching the export_hfa_artifacts conventions.
    """
    for chunk_idx, (wav_path, wav_length, words) in enumerate(predictions):
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")
        tg = tg_lib.TextGrid(minTime=0, maxTime=wav_length)
        word_tier = tg_lib.IntervalTier(name="words", minTime=0.0, maxTime=wav_length)
        phone_tier = tg_lib.IntervalTier(name="phones", minTime=0.0, maxTime=wav_length)

        for word in words:
            word_tier.add(minTime=word.start, maxTime=word.end, mark=word.text)
            for phoneme in word.phonemes:
                phone_tier.add(minTime=max(0, phoneme.start), maxTime=phoneme.end, mark=phoneme.text)

        tg.append(word_tier)
        tg.append(phone_tier)

        new_stem = f"{output_key}_{chunk_idx:03d}"
        target = output_dir / f"{new_stem}.TextGrid"
        target.parent.mkdir(parents=True, exist_ok=True)
        tg.write(str(target))
    logger.info(f"[TiFA] Saved {len(predictions)} TextGrid file(s) to {output_dir}")
