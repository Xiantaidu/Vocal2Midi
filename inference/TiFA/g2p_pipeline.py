"""Build the default TiFA G2P pipeline for Vocal2Midi (zh/ja/en).

Mirrors the upstream configs/g2p.yaml converter chain without the pydantic
config machinery of g2p.api: preprocessors (filter-punctuation,
strip-whitespace, lowercase) then chinese-pinyin, japanese-mecab (optional),
lstm (en), dictionary (zh), dictionary (ja), passthrough.
"""
from __future__ import annotations

import logging
from pathlib import Path

import inference.TiFA.g2p.converters  # noqa: F401 - populates the registry
import inference.TiFA.g2p.preprocessors  # noqa: F401
from inference.TiFA.g2p.pipeline import G2PPipeline
from inference.TiFA.g2p.registry import get_converter, get_preprocessor

logger = logging.getLogger(__name__)


def build_g2p_pipeline(model_dir: str | Path) -> G2PPipeline:
    """Construct the converter chain rooted at the TiFA model directory."""
    root = Path(model_dir)
    dictionaries = root / "dictionaries"

    converters = [
        get_converter("chinese-pinyin")(dict_path=str(dictionaries / "ds-zh-pinyin-lite.txt")),
    ]
    try:
        mecab = get_converter("japanese-mecab")(
            dict_path=str(dictionaries / "japanese_dict_full.txt"),
            nbest=32,
            double_written_sokuon=False,
        )
        # fugashi/unidic are imported lazily on first use, so a construction
        # probe is required to detect their absence.
        mecab.convert("あ")
        converters.append(mecab)
    except Exception as e:
        # fugashi/unidic are optional; kana and romaji input still converts
        # through the Japanese dictionary below.
        logger.warning(f"Japanese MeCab converter unavailable, kanji input falls back to the dictionary: {e}")

    lstm = get_converter("lstm")(
        dict_path=str(dictionaries / "ds_cmudict-07b.txt"),
        model_path=str(root / "assets" / "LstmG2p-Eng"),
        beam_size=16,
    )
    lstm.language = ("en",)
    converters.append(lstm)

    zh_dictionary = get_converter("dictionary")(dict_path=str(dictionaries / "ds-zh-pinyin-lite.txt"))
    zh_dictionary.language = ("zh",)
    ja_dictionary = get_converter("dictionary")(dict_path=str(dictionaries / "japanese_dict_full.txt"))
    ja_dictionary.language = ("ja",)
    converters.extend([zh_dictionary, ja_dictionary, get_converter("passthrough")()])

    return G2PPipeline(
        preprocessors=[
            get_preprocessor("filter-punctuation")(),
            get_preprocessor("strip-whitespace")(),
            get_preprocessor("lowercase")(),
        ],
        converters=converters,
    )
