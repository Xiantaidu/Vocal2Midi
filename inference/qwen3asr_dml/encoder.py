# coding=utf-8
import os
import time

import numpy as np
import onnxruntime as ort
import scipy.signal

from inference.device_utils import resolve_onnx_providers


class FastWhisperMel:
    """NumPy/SciPy mel extractor compatible with the Qwen3-ASR frontend."""

    def __init__(
        self,
        filter_path: str | None = None,
        n_mels: int = 128,
        sr: int = 16000,
        n_fft: int = 400,
        f_min: int = 0,
        f_max: int = 8000,
        norm: str | None = "slaney",
        mel_scale: str = "slaney",
    ):
        self.n_fft = n_fft
        self.hop_length = 160
        self.n_mels = n_mels

        if filter_path and os.path.exists(filter_path):
            self.filters = np.load(filter_path)
        else:
            self.filters = self._generate_filters(sr, n_fft, n_mels, f_min, f_max, norm, mel_scale)

        self.window = scipy.signal.get_window("hann", self.n_fft, fftbins=True)

    def _generate_filters(self, sr, n_fft, n_mels, f_min, f_max, norm, mel_scale):
        def hz_to_mel(freq, scale):
            if scale == "htk":
                return 2595.0 * np.log10(1.0 + (freq / 700.0))
            f_min_sl, f_sp_sl = 0.0, 200.0 / 3
            mels = (freq - f_min_sl) / f_sp_sl
            min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
            min_log_mel = (min_log_hz - f_min_sl) / f_sp_sl
            if isinstance(freq, np.ndarray):
                mask = freq >= min_log_hz
                mels[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
            elif freq >= min_log_hz:
                mels = min_log_mel + np.log(freq / min_log_hz) / logstep
            return mels

        def mel_to_hz(mels, scale):
            if scale == "htk":
                return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
            f_min_sl, f_sp_sl = 0.0, 200.0 / 3
            freqs = f_min_sl + f_sp_sl * mels
            min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
            min_log_mel = (min_log_hz - f_min_sl) / f_sp_sl
            if isinstance(mels, np.ndarray):
                mask = mels >= min_log_mel
                freqs[mask] = min_log_hz * np.exp(logstep * (mels[mask] - min_log_mel))
            elif mels >= min_log_mel:
                freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
            return freqs

        n_freqs = n_fft // 2 + 1
        all_freqs = np.linspace(0, sr // 2, n_freqs)
        m_pts = np.linspace(hz_to_mel(f_min, mel_scale), hz_to_mel(f_max, mel_scale), n_mels + 2)
        f_pts = mel_to_hz(m_pts, mel_scale)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = np.maximum(0, np.minimum(down_slopes, up_slopes))

        if norm == "slaney":
            enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
            fb *= enorm[np.newaxis, :]

        return fb.astype(np.float32)

    def __call__(self, audio: np.ndarray, dtype=np.float32) -> np.ndarray:
        pad_len = int(self.n_fft // 2)
        y = np.pad(audio, pad_len, mode="reflect")

        num_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        shape = (self.n_fft, num_frames)
        strides = (y.itemsize, self.hop_length * y.itemsize)
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

        stft_res = np.fft.rfft(frames * self.window[:, np.newaxis], axis=0)
        magnitudes = np.abs(stft_res) ** 2
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        n_frames_out = audio.shape[-1] // self.hop_length
        log_spec = log_spec[:, :n_frames_out]
        return log_spec.astype(dtype, copy=False)


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return int(output_lengths)


class QwenAudioEncoder:
    """Split Qwen3-ASR encoder with real ONNX Runtime batching."""

    def __init__(
        self,
        frontend_path: str,
        backend_path: str,
        use_dml: bool = True,
        warmup_sec: float = 5.0,
        verbose: bool = True,
    ):
        self.verbose = verbose

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        requested_device = "dml" if use_dml else "cpu"
        _, providers = resolve_onnx_providers(requested_device, label="Qwen3 Encoder ONNX")

        if self.verbose:
            print(f"--- [Encoder] Loading split ONNX models (DML: {use_dml}) ---")
            print(f"    Frontend: {os.path.basename(frontend_path)}")
            print(f"    Backend:  {os.path.basename(backend_path)}")

        self.sess_fe = ort.InferenceSession(frontend_path, sess_options=sess_opts, providers=providers)
        self.sess_be = ort.InferenceSession(backend_path, sess_options=sess_opts, providers=providers)
        self.frontend_providers = list(self.sess_fe.get_providers())
        self.backend_providers = list(self.sess_be.get_providers())
        self.provider_name = (
            "dml"
            if self.frontend_providers and self.frontend_providers[0] == "DmlExecutionProvider"
            else "cpu"
        )
        self.mel_extractor = FastWhisperMel()

        try:
            fe_input_type = self.sess_fe.get_inputs()[0].type
            self.input_dtype = np.float16 if "float16" in fe_input_type else np.float32
        except Exception:
            self.input_dtype = np.float32

        if warmup_sec > 0:
            if self.verbose:
                print(f"--- [Encoder] Warming up with {warmup_sec:.1f}s of random audio ---")
            dummy_wav = np.random.randn(int(16000 * warmup_sec)).astype(np.float32)
            _ = self.encode(dummy_wav)
            if self.verbose:
                print("--- [Encoder] Warmup complete ---")

    def _prepare_mel(self, audio: np.ndarray) -> tuple[list[np.ndarray], int]:
        mel = self.mel_extractor(audio, dtype=self.input_dtype)
        total_frames = mel.shape[1]
        pad_len = (100 - (total_frames % 100)) % 100
        if pad_len > 0:
            mel = np.pad(mel, ((0, 0), (0, pad_len)), mode="constant")
        num_chunks = mel.shape[1] // 100
        chunks = [mel[:, idx * 100 : (idx + 1) * 100] for idx in range(num_chunks)]
        return chunks, get_feat_extract_output_lengths(total_frames)

    def _run_frontend_batch(
        self,
        mel_chunks_per_audio: list[list[np.ndarray]],
        output_lengths: list[int],
    ) -> list[np.ndarray]:
        hidden_segments = [[] for _ in mel_chunks_per_audio]
        max_chunks = max((len(chunks) for chunks in mel_chunks_per_audio), default=0)

        for chunk_idx in range(max_chunks):
            active_indices = [idx for idx, chunks in enumerate(mel_chunks_per_audio) if chunk_idx < len(chunks)]
            if not active_indices:
                continue
            batch = np.stack(
                [mel_chunks_per_audio[idx][chunk_idx] for idx in active_indices],
                axis=0,
            ).astype(self.input_dtype, copy=False)
            outputs = self.sess_fe.run(None, {"chunk_mel": batch})[0]
            for batch_row, audio_idx in enumerate(active_indices):
                hidden_segments[audio_idx].append(outputs[batch_row])

        hidden_states_list = []
        for output_length, segments in zip(output_lengths, hidden_segments):
            if not segments:
                hidden_states_list.append(np.zeros((0, 0), dtype=self.input_dtype))
                continue
            hidden_states = np.concatenate(segments, axis=0)
            hidden_states_list.append(hidden_states[:output_length, :])
        return hidden_states_list

    def _run_backend_batch(self, hidden_states_list: list[np.ndarray]) -> list[np.ndarray]:
        if not hidden_states_list:
            return []

        max_len = max(hidden_states.shape[0] for hidden_states in hidden_states_list)
        if max_len == 0:
            return [hidden_states for hidden_states in hidden_states_list]

        batch = len(hidden_states_list)
        hidden_dim = hidden_states_list[0].shape[1]
        hidden_dtype = hidden_states_list[0].dtype
        hidden_batch = np.zeros((batch, max_len, hidden_dim), dtype=hidden_dtype)
        mask = np.zeros((batch, 1, max_len, max_len), dtype=hidden_dtype)
        mask_fill_value = np.asarray(-1e4, dtype=hidden_dtype).item()

        for batch_idx, hidden_states in enumerate(hidden_states_list):
            valid_len = hidden_states.shape[0]
            hidden_batch[batch_idx, :valid_len, :] = hidden_states
            if valid_len < max_len:
                mask[batch_idx, :, :, valid_len:] = mask_fill_value

        audio_embd_batch = self.sess_be.run(
            None,
            {
                "hidden_states": hidden_batch,
                "attention_mask": mask,
            },
        )[0]
        return [
            audio_embd_batch[batch_idx, : hidden_states.shape[0], :]
            for batch_idx, hidden_states in enumerate(hidden_states_list)
        ]

    def encode_batch(self, audios: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        if not audios:
            return [], 0.0

        t0 = time.time()
        prepared = [self._prepare_mel(audio) for audio in audios]
        mel_chunks_per_audio = [item[0] for item in prepared]
        output_lengths = [item[1] for item in prepared]
        hidden_states_list = self._run_frontend_batch(mel_chunks_per_audio, output_lengths)
        audio_embds = self._run_backend_batch(hidden_states_list)
        return audio_embds, time.time() - t0

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        audio_embds, elapsed = self.encode_batch([audio])
        audio_embd = audio_embds[0] if audio_embds else np.zeros((0, 0), dtype=self.input_dtype)
        return audio_embd, elapsed
