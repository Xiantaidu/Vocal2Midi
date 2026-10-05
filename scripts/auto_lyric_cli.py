"""Headless auto-lyric CLI: run the full extraction pipeline without the GUI.

Feeds one or more audio files (or directories) through the auto lyric hybrid
pipeline (ASR / HFA / GAME) and writes MIDI/USTX/VSQX/... outputs. Defaults
mirror the GUI settings (settings/vocal2midi.ini in portable mode, registry
otherwise); explicit flags override them.

Full option reference: scripts/auto_lyric_cli.md
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

INPUT_AUDIO_EXTENSIONS = {".wav", ".m4a", ".flac", ".mp3", ".ogg", ".opus", ".wma", ".webm", ".aif", ".aiff"}
LANGUAGE_CHOICES = ("zh", "ja", "en", "yue")
# backend value -> valid lyric output modes per language (mirrors the GUI tables)
LYRIC_FORMATS_BY_LANGUAGE = {
    "zh": ("pinyin", "hanzi"),
    "ja": ("romaji", "kana"),
    "en": ("word",),
    "yue": ("jyutping", "hanzi"),
}
DEFAULT_LYRIC_FORMAT = {"zh": "hanzi", "ja": "romaji", "en": "word", "yue": "hanzi"}
LYRIC_FORMAT_CHOICES = ("pinyin", "hanzi", "romaji", "kana", "word", "jyutping")
EXPORT_FORMAT_CHOICES = ("mid", "txt", "csv", "ustx", "vsqx", "chunks", "asr_match_log")
QUANT_STEP_CHOICES = (0, 480, 240, 120, 60, 30)
QUANT_MODE_CHOICES = ("smart", "simple")

# Same fallbacks the GUI pages use; only used when the settings store does
# not carry the key yet.
FALLBACK_DEFAULTS = {
    "game_model": "models/GAME-1.0.3-medium-onnx",
    "hfa_model": "models/1218_hfa_model_new_dict",
    "tifa_model": "models/tifa-1.0-onnx",
    "asr_model": "models/Qwen3-ASR-1.7B-dml",
    "phoneme_asr_model": "models/romajiASR",
    "pinyin_asr_model": "models/pinyinASR",
    "rmvpe_model": "models/RMVPE",
    "kashi_g2p_model": "models/kashi-g2p-onnx",
    "chinese_asr_engine": "qwen",
    "japanese_asr_engine": "romaji",
    "alignment_engine": "tifa",
    "japanese_g2p_engine": "kashi-g2p-onnx",
    "batch_size": 1,
    "asr_batch_size": 2,
    "slice_min_sec": 5.0,
    "slice_max_sec": 10.0,
    "t0": 0.0,
    "nsteps": 8,
    "seg_threshold": 0.2,
    "seg_radius": 0.02,
    "est_threshold": 0.2,
    "pitch_format": "name",
    "round_pitch": True,
    "output_pitch_curve": False,
    "output_dir": None,
    "lyric_output_mode_zh": "hanzi",
    "lyric_output_mode_ja": "romaji",
    "lyric_output_mode_en": "word",
    "lyric_output_mode_yue": "hanzi",
}


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return fallback


def _load_settings_defaults() -> dict:
    """GUI settings mirrored as CLI defaults; missing keys use GUI fallbacks."""
    defaults = dict(FALLBACK_DEFAULTS)
    try:
        settings_utils = importlib.import_module("gui.settings_utils")
        settings = settings_utils.create_app_settings()
        for key in (
            "game_model",
            "hfa_model",
            "tifa_model",
            "asr_model",
            "phoneme_asr_model",
            "pinyin_asr_model",
            "rmvpe_model",
            "chinese_asr_engine",
            "japanese_asr_engine",
            "alignment_engine",
            "pitch_format",
        ):
            defaults[key] = str(settings.value(key, defaults[key]) or defaults[key])
        for key in ("batch_size", "nsteps"):
            defaults[key] = _as_int(settings.value(key, defaults[key]), defaults[key])
        for key in ("asr_batch_size",):
            defaults[key] = _as_int(settings.value("asr_batch", defaults[key]), defaults[key])
        for key in ("slice_min_sec", "slice_max_sec", "t0", "seg_threshold", "seg_radius", "est_threshold"):
            defaults[key] = _as_float(settings.value(key, defaults[key]), defaults[key])
        defaults["round_pitch"] = _as_bool(settings.value("round_pitch", defaults["round_pitch"]), defaults["round_pitch"])
        defaults["output_pitch_curve"] = _as_bool(
            settings.value("output_pitch_curve", defaults["output_pitch_curve"]), defaults["output_pitch_curve"]
        )
        for language in LANGUAGE_CHOICES:
            key = f"lyric_output_mode_{language}"
            fallback_val = defaults.get(key, DEFAULT_LYRIC_FORMAT.get(language, "hanzi"))
            defaults[key] = str(settings.value(key, fallback_val) or fallback_val)
        saved_dir = str(settings.value("save_dir", "") or "").strip()
        if saved_dir:
            defaults["output_dir"] = saved_dir
        else:
            defaults["output_dir"] = str(settings_utils.default_output_dir(ROOT_DIR))
    except Exception as e:
        logger.debug(f"Settings unavailable, using built-in defaults: {e}")
        if not defaults["output_dir"]:
            settings_utils = importlib.import_module("gui.settings_utils")
            defaults["output_dir"] = str(settings_utils.default_output_dir(ROOT_DIR))
    return defaults


def build_argparser(defaults: dict) -> argparse.ArgumentParser:
    device_utils = importlib.import_module("inference.device_utils")
    parser = argparse.ArgumentParser(
        prog="auto_lyric_cli",
        description="Headless Vocal2Midi: run the auto lyric extraction pipeline without the GUI.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio files or directories to process")
    parser.add_argument("-o", "--output-dir", type=Path, default=defaults["output_dir"],
                        help="Output directory (default: the GUI save directory)")
    parser.add_argument("--language", choices=LANGUAGE_CHOICES, default="zh", help="Lyric language (default: zh)")
    parser.add_argument("--lyric-format", choices=LYRIC_FORMAT_CHOICES, default=None,
                        help="Lyric output format (default: the GUI setting for the language)")
    parser.add_argument("--chinese-asr", choices=("pinyin", "qwen"), default=defaults["chinese_asr_engine"],
                        help="Chinese ASR engine (default: the GUI setting)")
    parser.add_argument("--japanese-asr", choices=("romaji", "qwen"), default=defaults["japanese_asr_engine"],
                        help="Japanese ASR engine (default: the GUI setting)")
    parser.add_argument("--japanese-g2p", choices=("kashi-g2p-onnx", "pyopenjtalk"),
                        default=defaults.get("japanese_g2p_engine", "kashi-g2p-onnx"),
                        help="Japanese G2P engine (default: the GUI setting, kashi-g2p-onnx)")
    parser.add_argument("--aligner", choices=("tifa", "hfa"), default=defaults["alignment_engine"],
                        help="Forced-alignment engine (default: the GUI setting, tifa)")
    parser.add_argument("--lyrics", default="", help="Reference lyrics text for alignment")
    parser.add_argument("--lyrics-file", type=Path, default=None,
                        help="Read reference lyrics from a UTF-8 text file (overrides --lyrics)")
    parser.add_argument("--no-lyrics", action="store_true",
                        help="Skip ASR/HFA and extract pitch only (no lyrics in outputs)")
    parser.add_argument("--slicing", choices=("smart", "heuristic", "default", "grid"), default="smart",
                        help="Audio slicing method (default: smart)")
    parser.add_argument("--min-seconds", type=float, default=defaults["slice_min_sec"], help="Slice min duration (s)")
    parser.add_argument("--max-seconds", type=float, default=defaults["slice_max_sec"], help="Slice max duration (s)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Do not recurse into subdirectories for directory inputs")
    parser.add_argument("--quant-step", type=int, choices=QUANT_STEP_CHOICES, default=0,
                        help="Quantization grid in ticks (0 = off, 480 = 1/4 note)")
    parser.add_argument("--quant-mode", choices=QUANT_MODE_CHOICES, default="smart", help="Quantization algorithm")
    parser.add_argument("--quant-simplicity", type=float, default=0.0,
                        help="Smart quantization simplicity (0 = conservative, the GUI value)")
    parser.add_argument("--formats", nargs="+", choices=EXPORT_FORMAT_CHOICES, default=["mid"],
                        help="Output formats (default: mid)")
    parser.add_argument("--device", choices=list(device_utils.RUNTIME_DEVICE_CHOICES),
                        default=device_utils.default_runtime_device(), help="Runtime device")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"], help="GAME batch size")
    parser.add_argument("--asr-batch-size", type=int, default=defaults["asr_batch_size"], help="ASR batch size")
    parser.add_argument("--t0", type=float, default=defaults["t0"], help="D3PM starting t0")
    parser.add_argument("--nsteps", type=int, default=defaults["nsteps"], help="D3PM sampling steps")
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo BPM for quantization and MIDI export")
    parser.add_argument("--pitch-curve", action=argparse.BooleanOptionalAction, default=defaults["output_pitch_curve"],
                        help="Embed the RMVPE pitch curve in USTX/VSQX (default: the GUI setting)")
    parser.add_argument("--round-pitch", action=argparse.BooleanOptionalAction, default=defaults["round_pitch"],
                        help="Round pitch values in text outputs (default: the GUI setting)")
    parser.add_argument("--pitch-format", choices=("name", "number"), default=defaults["pitch_format"],
                        help="Pitch format for txt/csv outputs")
    parser.add_argument("--seg-threshold", type=float, default=defaults["seg_threshold"], help="Decode threshold")
    parser.add_argument("--seg-radius", type=float, default=defaults["seg_radius"], help="Decode radius (s)")
    parser.add_argument("--est-threshold", type=float, default=defaults["est_threshold"], help="Note existence threshold")
    parser.add_argument("--game-model", default=defaults["game_model"], help="GAME ONNX model directory")
    parser.add_argument("--hfa-model", default=defaults["hfa_model"], help="HubertFA ONNX model directory")
    parser.add_argument("--tifa-model", default=defaults["tifa_model"], help="TiFA ONNX model directory")
    parser.add_argument("--kashi-g2p-model", default=defaults["kashi_g2p_model"], help="kashi-g2p ONNX model directory")
    parser.add_argument("--asr-model", default=defaults["asr_model"], help="Qwen3-ASR model directory")
    parser.add_argument("--phoneme-asr-model", default=defaults["phoneme_asr_model"], help="Romaji ASR model directory")
    parser.add_argument("--pinyin-asr-model", default=defaults["pinyin_asr_model"], help="Pinyin ASR model directory")
    parser.add_argument("--rmvpe-model", default=defaults["rmvpe_model"], help="RMVPE model path")
    return parser


def validate_args(args) -> None:
    if args.language not in LYRIC_FORMATS_BY_LANGUAGE:
        raise ValueError(f"unsupported language: {args.language}")
    if args.lyric_format is not None and args.lyric_format not in LYRIC_FORMATS_BY_LANGUAGE[args.language]:
        valid = ", ".join(LYRIC_FORMATS_BY_LANGUAGE[args.language])
        raise ValueError(f"lyric format '{args.lyric_format}' is not valid for language '{args.language}' (valid: {valid})")
    if args.no_lyrics and (args.lyrics or args.lyrics_file):
        raise ValueError("--no-lyrics cannot be combined with --lyrics/--lyrics-file")
    if args.lyrics_file is not None and not Path(args.lyrics_file).is_file():
        raise FileNotFoundError(f"lyrics file not found: {args.lyrics_file}")
    if not 0 < args.min_seconds <= args.max_seconds:
        raise ValueError(f"invalid slice duration bounds: min={args.min_seconds}, max={args.max_seconds}")
    if args.nsteps <= 0:
        raise ValueError(f"nsteps must be positive, got {args.nsteps}")
    # PinyinASR can only emit pinyin; the backend coerces hanzi to pinyin.
    if args.language == "zh" and args.chinese_asr == "pinyin" and args.lyric_format == "hanzi":
        logger.warning("PinyinASR only outputs pinyin; the lyric format will fall back to pinyin.")
    if args.language == "yue" and args.aligner == "hfa":
        raise ValueError("Cantonese ('yue') is not supported by HubertFA (HFA); use --aligner tifa instead.")


def collect_inputs(inputs, output_dir: Path | None = None, recursive: bool = True) -> list[Path]:
    """Expand files/directories into a deterministic, deduplicated file list."""
    files: list[Path] = []
    seen: set[Path] = set()
    output_resolved = output_dir.resolve() if output_dir is not None else None
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
            for candidate in sorted(candidates):
                if candidate.is_file() and candidate.suffix.lower() in INPUT_AUDIO_EXTENSIONS:
                    files.append(candidate)
        elif path.is_file():
            if path.suffix.lower() not in INPUT_AUDIO_EXTENSIONS:
                raise ValueError(f"unsupported audio file: {path}")
            files.append(path)
        else:
            raise FileNotFoundError(f"input not found: {path}")

    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if output_resolved is not None:
            try:
                resolved.relative_to(output_resolved)
                continue  # never re-process files written into the output dir
            except ValueError:
                pass
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("no audio files found in the given inputs")
    return unique


def build_config(args, audio_path: Path, ts_list: list, asr_session=None):
    application_config = importlib.import_module("application.config")
    output_lyrics = not args.no_lyrics
    original_lyrics = ""
    if output_lyrics:
        if args.lyrics_file is not None:
            original_lyrics = Path(args.lyrics_file).read_text(encoding="utf-8").strip()
        elif args.lyrics:
            original_lyrics = args.lyrics.strip()
    return application_config.PipelineConfig(
        audio_path=str(audio_path),
        output_filename=audio_path.name,
        output_dir=Path(args.output_dir),
        game_model_dir=args.game_model,
        hfa_model_dir=args.hfa_model,
        asr_model_path=args.asr_model,
        device=args.device,
        language=args.language,
        ts=ts_list,
        lyric_output_mode=args.lyric_format,
        original_lyrics=original_lyrics,
        output_formats=list(args.formats),
        output_lyrics=output_lyrics,
        output_pitch_curve=args.pitch_curve,
        slicing_method=args.slicing,
        slice_min_sec=args.min_seconds,
        slice_max_sec=args.max_seconds,
        tempo=args.tempo,
        quantization_step=args.quant_step,
        quantization_mode=args.quant_mode,
        quant_simplicity=args.quant_simplicity,
        pitch_format=args.pitch_format,
        round_pitch=args.round_pitch,
        seg_threshold=args.seg_threshold,
        seg_radius=args.seg_radius,
        est_threshold=args.est_threshold,
        batch_size=args.batch_size,
        asr_batch_size=args.asr_batch_size,
        rmvpe_model_path=args.rmvpe_model,
        phoneme_asr_model_path=args.phoneme_asr_model,
        pinyin_asr_model_path=args.pinyin_asr_model,
        chinese_asr_engine=args.chinese_asr,
        japanese_asr_engine=args.japanese_asr,
        alignment_engine=args.aligner,
        tifa_model_path=args.tifa_model,
        japanese_g2p_engine=args.japanese_g2p,
        kashi_g2p_model_path=args.kashi_g2p_model,
        asr_session=asr_session,
    )


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    defaults = _load_settings_defaults()
    parser = build_argparser(defaults)
    args = parser.parse_args(argv)
    if args.lyric_format is None:
        args.lyric_format = defaults.get(
            f"lyric_output_mode_{args.language}", DEFAULT_LYRIC_FORMAT[args.language]
        )

    try:
        validate_args(args)
        input_files = collect_inputs(args.inputs, Path(args.output_dir), recursive=not args.no_recursive)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        return 2

    application_pipeline = importlib.import_module("application.pipeline")
    gui_fluent_utils = importlib.import_module("gui.fluent_utils")
    ts_list = gui_fluent_utils.t0_nstep_to_ts(args.t0, args.nsteps)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(input_files)} file(s) -> {args.output_dir}")
    failed = 0
    cancelled = False
    session = application_pipeline.open_asr_session()
    try:
        for index, audio_path in enumerate(input_files, start=1):
            logger.info(f"========== [{index}/{len(input_files)}] {audio_path.name} ==========")
            try:
                config = build_config(args, audio_path, ts_list, asr_session=session)
                application_pipeline.run_auto_lyric_job(config)
            except KeyboardInterrupt:
                logger.warning("Cancelled by user.")
                cancelled = True
                break
            except Exception as e:
                failed += 1
                details = getattr(e, "details", "")
                message = f"{e}" + (f" ({details})" if details and str(details) != str(e) else "")
                logger.error(f"Failed: {audio_path.name}: {message}")
    finally:
        session.close()

    if cancelled:
        return 130
    succeeded = len(input_files) - failed
    logger.info(f"Done: {succeeded} succeeded, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
