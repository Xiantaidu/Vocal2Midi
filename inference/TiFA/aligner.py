"""Host-side TiFA alignment: audio + text -> phoneme/word boundaries.

Implements the ONNX.md host pipeline per chunk: G2P -> candidate grid ->
spectrogram -> (optional) pronunciation scoring DP -> select -> model
similarities -> Viterbi -> WordList. The returned prediction dict matches
the HubertFA contract consumed by game_api: {stem: (wav_path, wav_length,
WordList)}.
"""
from __future__ import annotations

import logging
import math
import pathlib

import librosa
import numpy as np
import soundfile as sf

from inference.API.hfa_api import _repair_pred_dict_short_words
from inference.HubertFA.tools.align_word import Phoneme, Word, WordList
from inference.TiFA.decoding import decode_alignment_flat
from inference.TiFA.g2p.encoding import G2PEncodingError, encode_paths
from inference.TiFA.g2p_pipeline import build_g2p_pipeline
from inference.TiFA.runtime import TifaModel
from inference.TiFA.scoring_dp import select_sample

logger = logging.getLogger(__name__)

# Same symbol sets as the upstream configs/g2p.yaml vocabulary block.
GLOBAL_SYMBOLS = ("AP", "SP", "EP", "GS", "sil", "br", "pau")
STOP_SYMBOLS = ("SP", "sil", "pau")
DEFAULT_SKIP_PENALTY = 0.5
_EPSILON = 0.001


def _strip_language_prefix(symbol: str, language: str) -> str:
    prefix = f"{language}/"
    return symbol[len(prefix):] if symbol.startswith(prefix) else symbol


def _load_chunk_waveform(path: pathlib.Path, target_sr: int) -> np.ndarray:
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _padded_length(sample_count: int, model: TifaModel) -> int:
    """Right-pad so L is a hop multiple and satisfies L > ceil((win-hop)/2)."""
    minimum = math.ceil((model.win_size - model.hop_size) / 2) + 1
    count = max(sample_count, minimum)
    remainder = count % model.hop_size
    return count + ((model.hop_size - remainder) % model.hop_size)


def _build_word_list(
    spans: np.ndarray,
    tokens: np.ndarray,
    word_ids: np.ndarray,
    choices: np.ndarray,
    lexicon: list[list[dict]],
    texts: list[str],
    vocabulary,
    language: str,
    timestep: float,
    duration: float,
) -> WordList:
    """Group aligned tokens into words matching the HubertFA WordList contract."""
    word_list = WordList()
    phoneme_cursor = 0.0
    n = len(tokens)
    i = 0
    while i < n:
        j = i
        while j < n and word_ids[j] == word_ids[i]:
            j += 1
        word_index = word_ids[i] - 1
        text = texts[word_index] if 0 <= word_index < len(texts) else f"word{word_index}"
        choice = int(choices[word_index])
        candidate = lexicon[word_index][choice - 1] if choice > 0 else None
        candidate_phonemes = list(candidate["phonemes"]) if candidate else []

        phonemes = []
        for k in range(i, j):
            onset = float(spans[k, 0]) * timestep
            offset = float(spans[k, 1]) * timestep
            symbol = candidate_phonemes[k - i] if k - i < len(candidate_phonemes) else None
            if symbol is None:
                symbols = vocabulary.decode(int(tokens[k]))
                symbol = symbols[0] if symbols else str(int(tokens[k]))
            label = _strip_language_prefix(symbol, language)
            onset = max(onset, phoneme_cursor)
            offset = max(offset, onset + _EPSILON)
            phonemes.append(Phoneme(onset, offset, label))
            phoneme_cursor = offset

        start = phonemes[0].start
        end = max(phonemes[-1].end, start + _EPSILON)
        word = Word(start, end, text)
        for phoneme in phonemes:
            word.append_phoneme(phoneme)
        word_list.append(word)
        i = j

    word_list.fill_small_gaps(duration)
    word_list.add_SP(duration)
    return word_list


def run_tifa_fa(
    model: TifaModel,
    temp_dir,
    language: str = "zh",
    cancel_checker=None,
    g2p_pipeline=None,
) -> dict:
    """Align every chunk_N.wav/.txt pair in temp_dir; returns the HFA pred_dict.

    Chunks whose text cannot be converted or encoded are skipped with a
    warning; the pipeline treats missing predictions as pitch-only fallbacks.
    """
    language = (language or "zh").strip().lower()
    g2p = g2p_pipeline or build_g2p_pipeline(model.model_dir)
    pred_dict: dict = {}
    timestep = model.timestep

    for wav_path in sorted(pathlib.Path(temp_dir).rglob("*.wav")):
        if cancel_checker and cancel_checker():
            raise InterruptedError("任务已取消")
        stem = wav_path.stem
        text_path = wav_path.with_suffix(".txt")
        if not text_path.is_file():
            text_path = wav_path.with_suffix(".lab")
        if not text_path.is_file():
            logger.warning(f"[TiFA] {stem}: no paired text file; chunk falls back to pitch-only")
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning(f"[TiFA] {stem}: empty text; chunk falls back to pitch-only")
            continue

        try:
            g2p_words = g2p.convert(text, languages=[language])
        except Exception as e:
            logger.warning(f"[TiFA] {stem}: G2P failed ({e}); chunk falls back to pitch-only")
            continue
        try:
            data, lexicon, texts = encode_paths(
                g2p_words,
                model.vocabulary,
                "discard",
                languages=[language],
                global_symbols=GLOBAL_SYMBOLS,
                stop_symbols=STOP_SYMBOLS,
            )
        except G2PEncodingError as e:
            logger.warning(f"[TiFA] {stem}: {e}; chunk falls back to pitch-only")
            continue
        if not data["paths"].any():
            logger.warning(f"[TiFA] {stem}: no valid token sequence; chunk falls back to pitch-only")
            continue

        paths = data["paths"][None].astype(np.int64)  # [1,P,C]
        words = data["words"][None].astype(np.int64)  # [1,P]
        groups = data["groups"][None].astype(np.int64)  # [1,P,C]
        candidates = data["candidates"][None].astype(bool)  # [1,W,C]

        waveform = _load_chunk_waveform(wav_path, model.samplerate)
        duration = len(waveform) / model.samplerate
        padded = _padded_length(len(waveform), model)
        waveform = np.pad(waveform, (0, padded - len(waveform)))

        try:
            spec_out = model.run("spectrogram", {
                "waveform": waveform[None].astype(np.float32),
                "duration": np.array([duration], dtype=np.float32),
            })
            spectrogram, mask_t = spec_out["spectrogram"], spec_out["maskT"]
            if not mask_t.any():
                logger.warning(f"[TiFA] {stem}: no valid audio frames; chunk falls back to pitch-only")
                continue

            prepared = model.run("prepare", {
                "paths": paths, "words": words, "candidates": candidates,
                "grouped": np.array(False),
            })
            template_tokens, segments, mapping = (
                prepared["tokens"].astype(np.int64),
                prepared["segments"].astype(np.int64),
                prepared["mapping"].astype(np.int64),
            )
            choices = candidates.any(axis=-1).astype(np.int64)  # [1,W]

            if (segments > 0).any():
                scoring_mask_n = template_tokens != 0
                model_out = model.run("model", {
                    "spectrogram": spectrogram,
                    "tokens": template_tokens,
                    "maskT": mask_t,
                    "maskN": scoring_mask_n,
                })
                logits = model_out["logits"].astype(np.float32)
                scored = model.run("score", {
                    "logits": logits, "paths": paths, "words": words,
                    "segments": segments, "mapping": mapping,
                })
                chosen, _ = select_sample(
                    candidates[0],
                    scored["descriptors"][0], scored["lengths"][0],
                    scored["costs"][0], scored["tails"][0], scored["capacity"][0],
                )
                choices[0] = chosen

            selected = model.run("select", {
                "paths": paths, "words": words, "groups": groups, "choices": choices,
            })
            best_tokens = selected["best_tokens"].astype(np.int64)
            best_words = selected["best_words"].astype(np.int64)
            best_groups = selected["best_groups"].astype(np.int64)
            mask_n = selected["maskN"]

            token_count = int(mask_n.sum())
            frame_count = int(mask_t.sum())
            if token_count == 0 or frame_count == 0:
                logger.warning(f"[TiFA] {stem}: empty alignment; chunk falls back to pitch-only")
                continue

            align_out = model.run("model", {
                "spectrogram": spectrogram,
                "tokens": best_tokens,
                "maskT": mask_t,
                "maskN": mask_n,
            })
            similarities = align_out["similarities"]

            spans = decode_alignment_flat(
                similarities[:, :frame_count, :token_count],
                np.array([frame_count], dtype=np.int64),
                np.array([token_count], dtype=np.int64),
                best_groups[0][None, :token_count],
                skip_penalty=DEFAULT_SKIP_PENALTY,
            )
        except Exception as e:
            logger.warning(f"[TiFA] {stem}: alignment failed ({e}); chunk falls back to pitch-only")
            continue

        valid_spans = spans[0, :token_count]
        valid_tokens = best_tokens[0, :token_count]
        valid_words = best_words[0, :token_count]
        word_list = _build_word_list(
            valid_spans, valid_tokens, valid_words, choices[0], lexicon,
            texts, model.vocabulary, language, timestep, duration,
        )
        pred_dict[stem] = (wav_path, duration, word_list)
        logger.info(f"[TiFA] {stem}: aligned {token_count} tokens into {len(word_list)} word(s)")

    _repair_pred_dict_short_words(pred_dict)
    return pred_dict
