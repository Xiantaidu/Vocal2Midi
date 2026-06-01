import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference.device_utils import resolve_onnx_providers


class GameOnnxModel:
    def __init__(self, model_dir: Path, *, requested_device: str = "dml"):
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"GAME ONNX config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.samplerate = int(self.config["samplerate"])
        self.timestep = float(self.config["timestep"])
        self.loop = bool(self.config.get("loop", True))
        self.languages = dict(self.config.get("languages") or {})
        self.requested_device = str(requested_device)
        self.provider_name, providers = self._resolve_providers(self.requested_device)
        self._sessions = self._load_sessions(providers)

    @staticmethod
    def _resolve_providers(device: str) -> tuple[str, list[str]]:
        return resolve_onnx_providers(device, label="GAME ONNX")

    def _load_sessions(self, providers: list[str]) -> dict[str, ort.InferenceSession]:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        filenames = {
            "encoder": "encoder.onnx",
            "segmenter": "segmenter.onnx",
            "estimator": "estimator.onnx",
            "dur2bd": "dur2bd.onnx",
            "bd2dur": "bd2dur.onnx",
        }
        sessions = {}
        for key, filename in filenames.items():
            path = self.model_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Required GAME ONNX file not found: {path}")
            sessions[key] = ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)
        return sessions

    def shutdown(self) -> None:
        self._sessions.clear()

    release = shutdown

    def infer_batch(
        self,
        *,
        waveforms: np.ndarray,
        durations: np.ndarray,
        known_durations: np.ndarray | None,
        boundary_threshold: float,
        boundary_radius: int,
        score_threshold: float,
        language: int = 0,
        ts: list[float] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ts = list(ts or [])
        batch_size = int(waveforms.shape[0])

        enc_out = self._sessions["encoder"].run(
            None,
            {
                "waveform": waveforms.astype(np.float32, copy=False),
                "duration": durations.astype(np.float32, copy=False),
            },
        )
        x_seg, x_est, mask_t = enc_out[0], enc_out[1], enc_out[2]

        if known_durations is not None and known_durations.size > 0:
            known_boundaries = np.zeros_like(mask_t, dtype=bool)
            for idx in range(batch_size):
                valid = known_durations[idx][known_durations[idx] > 0]
                if len(valid) == 0:
                    continue
                kb = self._sessions["dur2bd"].run(
                    None,
                    {
                        "durations": valid[np.newaxis, :].astype(np.float32, copy=False),
                        "maskT": mask_t[idx : idx + 1],
                    },
                )[0]
                known_boundaries[idx : idx + 1] = kb
        else:
            known_boundaries = np.zeros_like(mask_t, dtype=bool)

        boundaries = known_boundaries.copy()
        lang_arr = np.full((batch_size,), int(language), dtype=np.int64)

        if self.loop and ts:
            for t in ts:
                boundaries = self._sessions["segmenter"].run(
                    None,
                    {
                        "x_seg": x_seg,
                        "language": lang_arr,
                        "known_boundaries": known_boundaries,
                        "prev_boundaries": boundaries,
                        "t": np.full((batch_size,), float(t), dtype=np.float32),
                        "maskT": mask_t,
                        "threshold": np.array(boundary_threshold, dtype=np.float32),
                        "radius": np.array(boundary_radius, dtype=np.int64),
                    },
                )[0]
        else:
            boundaries = self._sessions["segmenter"].run(
                None,
                {
                    "x_seg": x_seg,
                    "language": lang_arr,
                    "known_boundaries": known_boundaries,
                    "prev_boundaries": boundaries,
                    "t": np.zeros((batch_size,), dtype=np.float32),
                    "maskT": mask_t,
                    "threshold": np.array(boundary_threshold, dtype=np.float32),
                    "radius": np.array(boundary_radius, dtype=np.int64),
                },
            )[0]

        durations_out, mask_n = self._sessions["bd2dur"].run(
            None,
            {
                "boundaries": boundaries,
                "maskT": mask_t,
            },
        )
        presence, scores = self._sessions["estimator"].run(
            None,
            {
                "x_est": x_est,
                "boundaries": boundaries,
                "maskT": mask_t,
                "maskN": mask_n,
                "threshold": np.array(score_threshold, dtype=np.float32),
            },
        )

        results = []
        for idx in range(batch_size):
            valid = mask_n[idx].astype(bool)
            results.append(
                (
                    durations_out[idx][valid],
                    presence[idx][valid],
                    scores[idx][valid],
                )
            )
        return results
