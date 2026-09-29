"""ONNX session management and model configuration for the TiFA aligner."""
from __future__ import annotations

import json
import pathlib

import onnxruntime as ort

from inference.TiFA.lib.vocabulary import Vocabulary
from inference.device_utils import resolve_onnx_providers

GRAPH_NAMES = ("spectrogram", "prepare", "score", "select", "model")


class TifaModel:
    """The five TiFA ONNX graphs plus timing config and phone vocabulary."""

    def __init__(self, model_dir: str | pathlib.Path, device: str | None = None):
        self.model_dir = pathlib.Path(model_dir)
        self.config: dict = json.loads(
            (self.model_dir / "config.json").read_text(encoding="utf-8")
        )
        self.vocabulary = Vocabulary.from_file(self.model_dir / "vocabulary.json")

        provider_name, providers = resolve_onnx_providers(device, label="TiFA ONNX")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.sessions: dict[str, ort.InferenceSession] = {
            name: ort.InferenceSession(
                str(self.model_dir / f"{name}.onnx"), options, providers=providers
            )
            for name in GRAPH_NAMES
        }
        self.provider_name = provider_name

    @property
    def samplerate(self) -> int:
        return int(self.config["samplerate"])

    @property
    def timestep(self) -> float:
        return float(self.config["timestep"])

    @property
    def hop_size(self) -> int:
        return int(self.config["hop_size"])

    @property
    def fft_size(self) -> int:
        return int(self.config["fft_size"])

    @property
    def win_size(self) -> int:
        return int(self.config["win_size"])

    def run(self, name: str, feed: dict) -> dict:
        session = self.sessions[name]
        names = [output.name for output in session.get_outputs()]
        return dict(zip(names, session.run(None, feed)))


def load_tifa_model(model_dir: str | pathlib.Path, device: str | None = None) -> TifaModel:
    """Load the TiFA ONNX bundle; naming parity with load_hfa_model."""
    return TifaModel(model_dir, device=device)
