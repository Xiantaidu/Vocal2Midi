import json
import pathlib
import warnings
from pathlib import Path
from typing import Dict
from typing import Union

import click
import textgrid as tg
import tqdm
from textgrid import PointTier, Point

from tools.metrics import (
    CustomPointTier,
    BoundaryEditRatio,
    BoundaryEditRatioWeighted,
    IntersectionOverUnion,
    Metric,
    VlabelerEditRatio,
)


def interval_tier_to_point_tier(tier: tg.IntervalTier) -> tg.PointTier:
    point_tier = tg.PointTier(name=tier.name)
    point_tier.add(0.0, "")
    for interval in tier:
        if point_tier[-1].mark == "" and point_tier[-1].time == interval.minTime:
            point_tier[-1].mark = interval.mark
        else:
            point_tier.add(interval.minTime, interval.mark)
        point_tier.add(interval.maxTime, "")

    return point_tier


def textgrid_from_file(textgrid_path: Union[str, Path]) -> tg.TextGrid:
    """Read a TextGrid file and return a TextGrid object."""
    _textgrid = tg.TextGrid()
    _textgrid.read(textgrid_path, encoding="utf-8")
    for idx, tier in enumerate(_textgrid):
        if isinstance(tier, tg.IntervalTier):
            _textgrid.tiers[idx] = interval_tier_to_point_tier(tier)

    return _textgrid


def remove_ignored_phonemes(point_tier: PointTier, ignored_phonemes_list=None):
    if ignored_phonemes_list is None:
        ignored_phonemes_list = []
    res_tier = CustomPointTier(name=point_tier.name)
    for point in point_tier:
        if point.mark not in ignored_phonemes_list:
            res_tier.addPoint(point)
    return res_tier


def quantize_tier(tier: PointTier, frame_length: float) -> CustomPointTier:
    """Quantize tier times to frame boundaries"""
    new_tier = CustomPointTier(name=tier.name)
    points = sorted(tier.points, key=lambda p: p.time)  # 确保时间点有序
    for point in points:
        # Quantize to nearest frame boundary
        quantized_time = round(point.time / frame_length)
        new_tier.addPoint(Point(quantized_time, point.mark))
    return new_tier


@click.command(
    help="Calculate metrics between the FA predictions and the targets (ground truth)."
)
@click.argument(
    "pred",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    metavar="PRED_DIR",
)
@click.argument(
    "target",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    metavar="TARGET_DIR",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Compare files in subdirectories recursively",
)
@click.option(
    "--strict", "-s", is_flag=True, help="Raise errors on mismatching phone sequences"
)
@click.option(
    "--ignore",
    type=str,
    default="SP",  # AP,SP,<AP>,<SP>,,pau,cl
    help="Ignored phone marks, split by commas",
    show_default=True,
)
@click.option(
    "--frame_length",
    "-fl",
    type=float,
    default=512 / 44100,
    help="Frame length in seconds for quantization (default: 512/44100)",
    show_default=True,
)
def main(pred: str, target: str, recursive: bool, strict: bool, ignore: str, frame_length: float):
    pred_dir = pathlib.Path(pred)
    target_dir = pathlib.Path(target)
    if recursive:
        iterable = list(pred_dir.rglob("*.TextGrid"))
    else:
        iterable = list(pred_dir.glob("*.TextGrid"))
    ignored = [ph.strip() for ph in ignore.split(",") if ph.strip()]
    metrics: Dict[str, Metric] = {
        "BoundaryEditRatio": BoundaryEditRatio(),
        "BoundaryEditRatioWeighted": BoundaryEditRatioWeighted(),
        "VlabelerEditRatio_1-2frames": VlabelerEditRatio(move_min_frames=1, move_max_frames=2),
        "VlabelerEditRatio_3-5frames": VlabelerEditRatio(move_min_frames=3, move_max_frames=5),
        "VlabelerEditRatio_6-9frames": VlabelerEditRatio(move_min_frames=6, move_max_frames=9),
        "VlabelerEditRatio_10+frames": VlabelerEditRatio(move_min_frames=10, move_max_frames=10000),
        "IntersectionOverUnion": IntersectionOverUnion(),
    }

    cnt = 0
    for pred_file in tqdm.tqdm(iterable):
        target_file = target_dir / pred_file.relative_to(pred_dir)
        if not target_file.exists():
            warnings.warn(
                f'The prediction file "{pred_file}" has no matching target file, '
                f'which should be "{target_file}".',
                category=UserWarning,
            )
            continue

        pred_tier = textgrid_from_file(pred_file)[-1]
        target_tier = textgrid_from_file(target_file)[-1]

        # Remove ignored phonemes
        pred_tier = remove_ignored_phonemes(pred_tier, ignored)
        target_tier = remove_ignored_phonemes(target_tier, ignored)

        # Quantize to frame boundaries
        pred_tier = quantize_tier(pred_tier, frame_length)
        target_tier = quantize_tier(target_tier, frame_length)

        for metric in metrics.values():
            try:
                metric.update(pred_tier, target_tier)
            except AssertionError as e:
                if not strict:
                    warnings.warn(
                        f"Failed to evaluate metric {metric.__class__.__name__} for file {pred_file}: {e}",
                        category=UserWarning,
                    )
                    continue
                else:
                    raise e

        cnt += 1

    if cnt == 0:
        raise RuntimeError(
            "Unable to compare any files in the given directories. "
            "Matching files should have same names and same relative paths, "
            "containing the same phone sequences except for spaces."
        )
    result = {key: metric.compute() for key, metric in metrics.items()}
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
