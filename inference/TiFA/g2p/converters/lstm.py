"""LSTM G2P converter using ONNX encoder-decoder models.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..registry import converter
from .paradigm import LexiconConverter
from .text import word_spans


@dataclass
class _Beam:
    tokens: list[int]
    score: float
    hidden: np.ndarray
    cell: np.ndarray
    finished: bool = False

    @property
    def normalized_score(self) -> float:
        # Unlike upstream LstmG2p, divide the cumulative log probability
        # directly: multiplying it by the old length counts it again.
        # Exclude BOS from the generated sequence length; include EOS.
        return self.score / max(1, len(self.tokens) - 1)


@converter(id="lstm", language=None)
class LSTMConverter(LexiconConverter):
    """Dictionary-backed G2P with LSTM ONNX model for OOV words.

    Parameters:
        *dict_path*: pronunciation dictionary (tab-separated).
        *model_path*: directory containing ``encoder.onnx``,
          ``decoder.onnx``, ``char.json``, ``phonemes.json``.
        *beam_size*: number of hypotheses to retain.
          Set to 1 for greedy decoding. Returns distinct pronunciations
          ranked by mean log probability, including EOS in the length.
    """

    def __init__(
        self, *, dict_path: str = None, model_path: str, beam_size: int = 16
    ) -> None:
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
            raise ValueError("beam_size must be a positive integer")
        super().__init__(dict_path=dict_path)
        self._beam_size = beam_size

        model_dir = Path(model_path)
        with open(model_dir / "char.json", "r", encoding="utf-8") as f:
            self._char_vocab: dict[str, int] = json.load(f)
        with open(model_dir / "phonemes.json", "r", encoding="utf-8") as f:
            self._phoneme_vocab: dict[str, int] = json.load(f)

        self._idx_to_phoneme: dict[int, str] = {
            v: k for k, v in self._phoneme_vocab.items()
        }
        self._unk_idx = self._phoneme_vocab["<unk>"]
        self._pad_idx = self._phoneme_vocab["<pad>"]
        self._bos_idx = self._phoneme_vocab["<bos>"]
        self._eos_idx = self._phoneme_vocab["<eos>"]
        self._char_unk_idx = self._char_vocab.get("<unk>", 0)

        self._model_path = str(model_dir)
        self._encoder_session = None
        self._decoder_session = None
        self._max_len = 48

    def __getstate__(self):
        # Native ONNX sessions cannot be pickled for spawned workers.
        state = self.__dict__.copy()
        state["_encoder_session"] = None
        state["_decoder_session"] = None
        return state

    # ------------------------------------------------------------------
    # LexiconConverter contract
    # ------------------------------------------------------------------

    def find(self, text: str) -> tuple[int, int] | None:
        for begin, end in word_spans(text):
            word = text[begin:end].lower()
            if word in self._dict or all(c in self._char_vocab for c in word):
                return begin, end
        return None

    def infer_oov(self, token: str) -> list[list[str]]:
        return self._predict(token)

    # ------------------------------------------------------------------
    # ONNX inference
    # ------------------------------------------------------------------

    def _ensure_sessions(self) -> None:
        if self._encoder_session is not None:
            return
        import onnxruntime as ort

        self._encoder_session = ort.InferenceSession(
            f"{self._model_path}/encoder.onnx"
        )
        self._decoder_session = ort.InferenceSession(
            f"{self._model_path}/decoder.onnx"
        )

    def _predict(self, word: str) -> list[list[str]]:
        self._ensure_sessions()

        word = word.lower().strip()
        indices = [
            self._char_vocab.get(c, self._char_unk_idx) for c in word
        ]
        src = np.array(
            [[self._bos_idx] + indices + [self._eos_idx]], dtype=np.int64
        )

        # Encoder
        encoder_outputs, hidden, cell = self._encoder_session.run(
            None, {"input_ids": src}
        )

        # Keep each hypothesis's decoder state separate; finished beams persist.
        beams = [_Beam([self._bos_idx], 0.0, hidden, cell)]

        for _step in range(self._max_len):
            if all(beam.finished for beam in beams):
                break
            candidates: list[_Beam] = []
            for beam in beams:
                if beam.finished:
                    candidates.append(beam)
                    continue
                logits, hidden, cell, _ = self._decoder_session.run(
                    None,
                    {
                        "decoder_input": np.array([[beam.tokens[-1]]], dtype=np.int64),
                        "hidden": beam.hidden,
                        "cell": beam.cell,
                        "encoder_outputs": encoder_outputs,
                    },
                )
                log_probs = logits[0, 0, :].astype(np.float64)
                log_probs -= np.max(log_probs)
                log_probs -= np.log(np.exp(log_probs).sum())
                # All extensions of one parent have the same length, so only
                # its best beam_size tokens can survive global pruning.
                top_ids = np.argsort(-log_probs, kind="stable")[:self._beam_size]
                for pred_id in top_ids:
                    pred_id = int(pred_id)
                    candidates.append(_Beam(
                        tokens=beam.tokens + [pred_id],
                        score=beam.score + float(log_probs[pred_id]),
                        hidden=hidden,
                        cell=cell,
                        finished=pred_id == self._eos_idx,
                    ))
            candidates.sort(key=lambda beam: beam.normalized_score, reverse=True)
            beams = candidates[:self._beam_size]

        # Special-token removal can collapse distinct beams to one reading.
        # Preserve the highest-ranked occurrence of each pronunciation.
        pronunciations = dict.fromkeys(
            tuple(self._decode(beam.tokens[1:])) for beam in beams
        )
        return [list(phonemes) for phonemes in pronunciations]

    def _decode(self, pred_ids: list[int]) -> list[str]:
        result: list[str] = []
        for idx in pred_ids:
            if idx in (
                self._unk_idx,
                self._pad_idx,
                self._bos_idx,
                self._eos_idx,
            ):
                continue
            ph = self._idx_to_phoneme.get(idx)
            if ph is not None:
                result.append(ph)
        return result
