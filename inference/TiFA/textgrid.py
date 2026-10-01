"""TextGrid export for TiFA predictions.

Mirrors HubertFA's Exporter.save_textgrids (same textgrid package, same
words/phones tier layout and {output_key}_{idx:03d} file naming) so the
chunks/textgrid debug outputs stay identical across aligners.
"""
from __future__ import annotations

import logging
import pathlib
import re

import textgrid as tg_lib

logger = logging.getLogger(__name__)


def _chunk_index(stem) -> int:
    """Numeric chunk index from a ``chunk_N`` stem (or an int passed directly)."""
    if isinstance(stem, int):
        return stem
    m = re.search(r"(\d+)\s*$", str(stem))
    return int(m.group(1)) if m else 0


def save_textgrids(
    predictions,
    output_dir: pathlib.Path,
    output_key: str,
    chunks: list | None = None,
    cancel_checker=None,
) -> None:
    """Write one TextGrid per prediction as {output_key}_{idx:03d}.TextGrid.

    ``predictions`` is the ``{stem: (wav_path, wav_length, WordList)}`` mapping
    produced by run_tifa_fa. The output index is the chunk's own numeric suffix
    (``chunk_10`` -> 010), NOT the position in the mapping: pred_dict is built
    in rglob order and iteration position would misname every chunk past
    chunk_9, so a TextGrid would land next to the wrong {output_key}_NNN.wav.
    A plain list is still accepted (index = position) for legacy callers.
    """
    items = predictions.items() if hasattr(predictions, "items") else enumerate(predictions)
    count = 0
    for stem, (wav_path, wav_length, words) in items:
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

        new_stem = f"{output_key}_{_chunk_index(stem):03d}"
        target = output_dir / f"{new_stem}.TextGrid"
        target.parent.mkdir(parents=True, exist_ok=True)
        tg.write(str(target))
        count += 1
    logger.info(f"[TiFA] Saved {count} TextGrid file(s) to {output_dir}")
