#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
# Common utility functions for inference and batch inference

import re
from typing import List, Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer


def pick_providers(device: str):
    providers = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    if device == "dml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"] if "DmlExecutionProvider" in providers else ["CPUExecutionProvider"]
    # auto
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in providers:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)


def compute_feat(samples: np.ndarray, sample_rate: int, window_size: int, window_shift: int):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    if online_fbank.num_frames_ready == 0:
        return np.zeros((0, 80 * window_size), dtype=np.float32)

    features = np.stack([online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)])

    T = (features.shape[0] - window_size) // window_shift + 1
    if T <= 0:
        return np.zeros((0, features.shape[1] * window_size), dtype=np.float32)

    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    return np.ascontiguousarray(features, dtype=np.float32)


def sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id=None,
    im_end_token_id=None,
    step: int = 0,
) -> int:
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32)

    logits = np.where(np.isfinite(logits), logits, float("-inf"))

    if temperature == 0.0:
        if step == 0:
            logits = logits.copy()
            if eos_token_id is not None:
                logits[eos_token_id] = float("-inf")
            if im_end_token_id is not None:
                logits[im_end_token_id] = float("-inf")
        return int(np.argmax(logits))

    logits = logits / float(temperature)

    if step == 0:
        logits = logits.copy()
        if eos_token_id is not None:
            logits[eos_token_id] = float("-inf")
        if im_end_token_id is not None:
            logits[im_end_token_id] = float("-inf")

    if top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        max_logit = np.max(sorted_logits)
        if np.isfinite(max_logit):
            exp_logits = np.exp(sorted_logits - max_logit)
            cumulative_probs = np.cumsum(exp_logits)
            if cumulative_probs[-1] > 0:
                cumulative_probs = cumulative_probs / cumulative_probs[-1]
                sorted_indices_to_remove = sorted_indices[cumulative_probs > top_p]
                if len(sorted_indices_to_remove) > 0:
                    sorted_indices_to_remove = sorted_indices_to_remove[1:]
                    logits = logits.copy()
                    logits[sorted_indices_to_remove] = float("-inf")

    max_logit = np.max(logits)
    if not np.isfinite(max_logit):
        probs = np.ones_like(logits) / len(logits)
    else:
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits)
        if not np.isfinite(sum_exp) or sum_exp <= 0:
            probs = np.ones_like(logits) / len(logits)
        else:
            probs = exp_logits / sum_exp
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                probs = np.ones_like(logits) / len(logits)

    token_id = np.random.choice(len(probs), p=probs)
    return int(token_id)


def build_source_ids(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str, audio_token_len: int):
    pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

    source_input = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    splits = pattern.split(source_input)

    source_ids = []
    fbank_beg = -1
    fake_token_len = 0

    for sub_str in splits:
        if not sub_str:
            continue
        if not sub_str.startswith("<|startofspeech|>"):
            source_ids += tokenizer.encode(sub_str)
        else:
            fake_token_len = int(audio_token_len)
            fbank_beg = len(source_ids)
            source_ids += [0] * fake_token_len

    if fbank_beg < 0:
        fbank_beg = len(source_ids)
        fake_token_len = int(audio_token_len)
        source_ids += [0] * fake_token_len

    return np.array(source_ids, dtype=np.int64), int(fbank_beg), int(fake_token_len)


def select_device(device_pref: str, model_path: str = None) -> str:
    providers = ort.get_available_providers()
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        return "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    if device_pref == "dml":
        return "dml" if "DmlExecutionProvider" in providers else "cpu"
    # auto
    if model_path and "int8" in model_path.lower():
        return "cpu"
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    if "DmlExecutionProvider" in providers:
        return "dml"
    return "cpu"


def setup_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None
    return tokenizer, eos_token_id, im_end_token_id


def load_and_resample_audio(filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    samples, sr = load_audio(filename)
    if sr != target_sr:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return samples, sr


def np_dtype_from_ort(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    if "int64" in s:
        return np.int64
    raise RuntimeError(f"Unsupported ORT type: {ort_type}")


def torch_dtype_from_np(np_dtype: np.dtype):
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.int64:
        return torch.int64
    raise RuntimeError(f"Unsupported numpy dtype: {np_dtype}")


def device_from_str(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


def bind_torch_input(io_binding, name: str, t: torch.Tensor):
    t = t.contiguous()
    dev = t.device
    device_type = "cuda" if dev.type == "cuda" else "cpu"
    device_id = int(dev.index or 0) if dev.type == "cuda" else 0
    np_dtype = (
        np.float16 if t.dtype == torch.float16 else
        np.float32 if t.dtype == torch.float32 else
        np.int64 if t.dtype == torch.int64 else None
    )
    if np_dtype is None:
        raise RuntimeError(f"Unsupported torch dtype for bind_input: {t.dtype}")
    io_binding.bind_input(
        name=name,
        device_type=device_type,
        device_id=device_id,
        element_type=np_dtype,
        shape=tuple(t.shape),
        buffer_ptr=t.data_ptr(),
    )


def bind_torch_output(io_binding, name: str, t: torch.Tensor):
    t = t.contiguous()
    dev = t.device
    device_type = "cuda" if dev.type == "cuda" else "cpu"
    device_id = int(dev.index or 0) if dev.type == "cuda" else 0
    np_dtype = (
        np.float16 if t.dtype == torch.float16 else
        np.float32 if t.dtype == torch.float32 else
        np.int64 if t.dtype == torch.int64 else None
    )
    if np_dtype is None:
        raise RuntimeError(f"Unsupported torch dtype for bind_output: {t.dtype}")
    io_binding.bind_output(
        name=name,
        device_type=device_type,
        device_id=device_id,
        element_type=np_dtype,
        shape=tuple(t.shape),
        buffer_ptr=t.data_ptr(),
    )


def bind_torch_tensor(io: ort.IOBinding, name: str, t: torch.Tensor, is_input: bool):
    if not t.is_cuda:
        raise RuntimeError(f"Tensor for '{name}' must be CUDA tensor")
    if not t.is_contiguous():
        t = t.contiguous()

    if t.dtype == torch.float16:
        elem = np.float16
    elif t.dtype == torch.float32:
        elem = np.float32
    elif t.dtype == torch.int64:
        elem = np.int64
    else:
        raise RuntimeError(f"Unsupported torch dtype for binding: {t.dtype}")

    if is_input:
        io.bind_input(
            name=name,
            device_type="cuda",
            device_id=0,
            element_type=elem,
            shape=list(t.shape),
            buffer_ptr=int(t.data_ptr()),
        )
    else:
        io.bind_output(
            name=name,
            device_type="cuda",
            device_id=0,
            element_type=elem,
            shape=list(t.shape),
            buffer_ptr=int(t.data_ptr()),
        )
    return t


def pick_last_logits_np(logits: np.ndarray, prompt_len: int):
    if logits.ndim != 3:
        raise RuntimeError(f"Bad logits ndim={logits.ndim}, shape={logits.shape}")
    if logits.shape[1] == 1:
        return logits[0, 0, :]
    return logits[0, prompt_len - 1, :]


def pick_last_logits_torch(logits_t: torch.Tensor):
    if logits_t.dim() != 3 or logits_t.shape[1] != 1:
        raise RuntimeError(f"Bad logits torch shape: {tuple(logits_t.shape)}")
    return logits_t[0, 0, :]


class EncoderAdaptorOnnxModel:
    def __init__(self, filename: str, device: str = "auto"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.window_size = int(meta.get("lfr_window_size", 7))
        self.window_shift = int(meta.get("lfr_window_shift", 6))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out_name], {self.in_name: x})[0]


class EmbeddingOnnxIOB:
    def __init__(self, filename: str, device: str = "auto"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = {o.name: o for o in self.sess.get_outputs()}
        self.in_np_dtype = np_dtype_from_ort(ins[self.in_name].type)
        self.out_np_dtype = np_dtype_from_ort(outs[self.out_name].type)
        self.out_torch_dtype = torch_dtype_from_np(self.out_np_dtype)
        out_shape = self.sess.get_outputs()[0].shape
        self.embed_dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else None
        if self.embed_dim is None:
            raise RuntimeError("Embedding output dim is dynamic; please export with static embed dim.")

    def forward_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        out = torch.empty((B, S, self.embed_dim), device=input_ids.device, dtype=self.out_torch_dtype).contiguous()
        io = self.sess.io_binding()
        bind_torch_input(io, self.in_name, input_ids)
        bind_torch_output(io, self.out_name, out)
        self.sess.run_with_iobinding(io)
        return out


class UnifiedKvDeltaLLMOnnxIOB:
    def __init__(self, filename: str, device: str = "auto"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        meta = self.sess.get_modelmeta().custom_metadata_map

        self.quant_type = str(meta.get("quantization_type", ""))
        self.num_layers = int(meta.get("num_layers", 0) or 0)
        self.max_total_len = int(meta.get("max_total_len", 0) or 0)
        self.num_kv_heads = int(meta.get("num_kv_heads", 0) or 0)
        self.head_dim = int(meta.get("head_dim", 0) or 0)
        self.vocab_size = int(meta.get("vocab_size", 0) or 0)
        self.logits_mode = str(meta.get("logits_mode", ""))

        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = {o.name: o for o in self.sess.get_outputs()}
        self.input_np_dtype = np_dtype_from_ort(ins["inputs_embeds"].type)
        self.cache_np_dtype = np_dtype_from_ort(ins["cache_key_0"].type)
        self.logits_np_dtype = np_dtype_from_ort(outs["logits"].type)

        self.input_torch_dtype = torch_dtype_from_np(self.input_np_dtype)
        self.cache_torch_dtype = torch_dtype_from_np(self.cache_np_dtype)
        self.logits_torch_dtype = torch_dtype_from_np(self.logits_np_dtype)

        if self.num_layers <= 0:
            self.num_layers = len([k for k in ins.keys() if k.startswith("cache_key_")])

        if self.vocab_size <= 0:
            lshape = outs["logits"].shape
            if isinstance(lshape[-1], int):
                self.vocab_size = int(lshape[-1])

        if self.max_total_len <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise RuntimeError("Missing max_total_len/num_kv_heads/head_dim in metadata.")
        if self.vocab_size <= 0:
            raise RuntimeError("Missing vocab_size in metadata.")

        if self.logits_mode and self.logits_mode != "last":
            print(f"[WARN] logits_mode meta is '{self.logits_mode}', but this runner assumes last-token logits.")

    def alloc_caches(self, batch: int, device: torch.device):
        caches_k, caches_v = [], []
        for _ in range(self.num_layers):
            ck = torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim),
                             device=device, dtype=self.cache_torch_dtype).contiguous()
            cv = torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim),
                             device=device, dtype=self.cache_torch_dtype).contiguous()
            caches_k.append(ck)
            caches_v.append(cv)
        return caches_k, caches_v

    def alloc_logits(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.empty((batch, 1, self.vocab_size), device=device, dtype=self.logits_torch_dtype).contiguous()

    def alloc_kv_deltas(self, batch: int, seq: int, device: torch.device):
        key_deltas, val_deltas = [], []
        for _ in range(self.num_layers):
            kd = torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous()
            vd = torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous()
            key_deltas.append(kd)
            val_deltas.append(vd)
        return key_deltas, val_deltas

    def alloc_outputs(self, batch: int, seq: int, device: torch.device):
        logits = torch.empty((batch, 1, self.vocab_size), device=device, dtype=self.logits_torch_dtype).contiguous()
        key_deltas, val_deltas = [], []
        for _ in range(self.num_layers):
            kd = torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous()
            vd = torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous()
            key_deltas.append(kd)
            val_deltas.append(vd)
        return logits, key_deltas, val_deltas

    def run_iobinding(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        caches_k: List[torch.Tensor],
        caches_v: List[torch.Tensor],
        logits_out: torch.Tensor,
        key_deltas_out: List[torch.Tensor],
        val_deltas_out: List[torch.Tensor],
    ):
        B = int(inputs_embeds.shape[0])
        S = int(inputs_embeds.shape[1])

        if tuple(logits_out.shape) != (B, 1, self.vocab_size):
            raise RuntimeError(f"logits_out shape mismatch: expect ({B},1,{self.vocab_size}), got {tuple(logits_out.shape)}")
        for i in range(self.num_layers):
            if tuple(key_deltas_out[i].shape) != (B, S, self.num_kv_heads, self.head_dim):
                raise RuntimeError(f"key_delta_out[{i}] shape mismatch: got {tuple(key_deltas_out[i].shape)}, expect ({B},{S},{self.num_kv_heads},{self.head_dim})")
            if tuple(val_deltas_out[i].shape) != (B, S, self.num_kv_heads, self.head_dim):
                raise RuntimeError(f"value_delta_out[{i}] shape mismatch: got {tuple(val_deltas_out[i].shape)}, expect ({B},{S},{self.num_kv_heads},{self.head_dim})")

        io = self.sess.io_binding()
        bind_torch_input(io, "inputs_embeds", inputs_embeds)
        bind_torch_input(io, "attention_mask", attention_mask)
        bind_torch_input(io, "cache_position", cache_position)
        for i in range(self.num_layers):
            bind_torch_input(io, f"cache_key_{i}", caches_k[i])
            bind_torch_input(io, f"cache_value_{i}", caches_v[i])

        bind_torch_output(io, "logits", logits_out)
        for i in range(self.num_layers):
            bind_torch_output(io, f"key_delta_{i}", key_deltas_out[i])
            bind_torch_output(io, f"value_delta_{i}", val_deltas_out[i])

        self.sess.run_with_iobinding(io)
        return logits_out, key_deltas_out, val_deltas_out
