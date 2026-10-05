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
from inference.ja_onnx_g2p import DEFAULT_MODEL_DIR as JA_G2P_MODEL_DIR
from inference.TiFA.g2p.pipeline import G2PPipeline
from inference.TiFA.g2p.registry import get_converter, get_preprocessor

logger = logging.getLogger(__name__)


def build_g2p_pipeline(
    model_dir: str | Path,
    japanese_g2p_engine: str = "kashi-g2p-onnx",
    kashi_g2p_model_dir: str | Path | None = None,
) -> G2PPipeline:
    """Construct the converter chain rooted at the TiFA model directory."""
    root = Path(model_dir)
    dictionaries = root / "dictionaries"

    converters = [
        get_converter("chinese-pinyin")(dict_path=str(dictionaries / "ds-zh-pinyin-lite.txt")),
    ]
    jyutping_dict_path = dictionaries / "jyutping_dict.txt"
    if jyutping_dict_path.is_file():
        converters.append(get_converter("yue-jyutping")(dict_path=str(jyutping_dict_path)))

    ja_engine = str(japanese_g2p_engine or "kashi-g2p-onnx").strip().lower()
    kashi_path = Path(kashi_g2p_model_dir) if kashi_g2p_model_dir else JA_G2P_MODEL_DIR

    if ja_engine in {"kashi-g2p-onnx", "ja_g2p", "ja_g2p_onnx"} and (kashi_path / "model.onnx").is_file():
        # Kanji-bearing runs go through the kashi-g2p-onnx model first: its
        # transformer disambiguates polyphones in sentence context and its
        # readings convert through the same DiffSinger Japanese dictionary.
        import inference.TiFA.japanese_onnx  # noqa: F401 - populates the registry

        converters.append(get_converter("japanese-onnx")(
            dict_path=str(dictionaries / "japanese_dict_full.txt"),
            model_dir=str(kashi_path),
        ))
    elif ja_engine == "pyopenjtalk":
        try:
            import inference.TiFA.japanese_pyopenjtalk  # noqa: F401 - populates the registry

            converters.append(get_converter("japanese-pyopenjtalk")(
                dict_path=str(dictionaries / "japanese_dict_full.txt"),
            ))
        except Exception as e:
            logger.warning(f"Japanese pyopenjtalk converter unavailable: {e}")
    lexicon_path = root / "dictionaries" / "ja_lexicon.txz"
    if lexicon_path.is_file():
        # Kanji-bearing spans claim their readings from the ja_g2p lexicon
        # (all candidates emitted; the scoring DP picks in audio context).
        import inference.TiFA.japanese_lexicon  # noqa: F401 - populates the registry

        converters.append(get_converter("japanese-lexicon")(
            lexicon_path=str(lexicon_path),
            dict_path=str(dictionaries / "japanese_dict_full.txt"),
        ))
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
        # fugashi/unidic are optional; the kana converter covers kana input
        # and the Japanese dictionary below covers romaji.
        logger.warning(f"Japanese MeCab converter unavailable, kanji input falls back to the dictionary: {e}")
        kana = get_converter("japanese-kana")(
            dict_path=str(dictionaries / "japanese_dict_full.txt"),
            double_written_sokuon=False,
        )
        converters.append(kana)

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
    converters.extend([zh_dictionary, ja_dictionary])
    if jyutping_dict_path.is_file():
        yue_dictionary = get_converter("dictionary")(dict_path=str(jyutping_dict_path))
        yue_dictionary.language = ("yue",)
        converters.append(yue_dictionary)
    converters.append(get_converter("passthrough")())

    return G2PPipeline(
        preprocessors=[
            get_preprocessor("filter-punctuation")(),
            get_preprocessor("strip-whitespace")(),
            get_preprocessor("lowercase")(),
        ],
        converters=converters,
    )
