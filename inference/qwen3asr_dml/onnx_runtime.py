"""Qwen3-ASR 分裂 ONNX 推理后端。

下载脚本使用的模型由一个音频编码器和两个自回归解码图组成：

* ``encoder.int4.onnx`` 把 16 kHz log-mel 转成音频 token；
* ``decoder_init.int4.onnx`` 处理带音频占位符的初始 prompt；
* ``decoder_step.int4.onnx`` 使用外部 embedding 和 KV cache 逐 token 解码。

在 macOS 上 CoreMLExecutionProvider 可以覆盖编码器的一部分算子，但当前
int4 ``decoder_step`` 图在 M4 上会在 CoreML 编译阶段失败。因此这里对
``device=metal`` 采用“编码器尝试 CoreML、两个解码器固定 CPU”的安全策略；
失败时自动回退 CPU，不影响模型可用性。GGUF 模型仍由同目录的 llama.cpp
后端负责 Metal 全量卸载。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from inference.device_utils import normalize_runtime_device

from .encoder import FastWhisperMel
from .schema import TranscribeResult


ENDOFTEXT_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ASR_TEXT_TOKEN_ID = 151704
EOS_TOKEN_IDS = {ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID}

LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "en": "English",
    "english": "English",
}


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _build_prompt_ids(
    tokenizer: Tokenizer,
    audio_token_count: int,
    language: str | None,
    context: str | None,
) -> list[int]:
    """构造与 Qwen3-ASR ONNX 导出图一致的 multimodal prompt。"""
    ids = [IM_START_TOKEN_ID, *_encode(tokenizer, "system\n")]
    if context:
        ids.extend(_encode(tokenizer, context))
    ids.extend(
        [
            IM_END_TOKEN_ID,
            *_encode(tokenizer, "\n"),
            IM_START_TOKEN_ID,
            *_encode(tokenizer, "user\n"),
            AUDIO_START_TOKEN_ID,
            *([AUDIO_PAD_TOKEN_ID] * audio_token_count),
            AUDIO_END_TOKEN_ID,
            IM_END_TOKEN_ID,
            *_encode(tokenizer, "\n"),
            IM_START_TOKEN_ID,
            *_encode(tokenizer, "assistant\n"),
        ]
    )
    if language:
        ids.extend(_encode(tokenizer, f"language {language}"))
        ids.append(ASR_TEXT_TOKEN_ID)
    return ids


def _resolve_variant(model_dir: Path, name: str) -> Path:
    for suffix in ("int4", ""):
        filename = f"{name}.{suffix}.onnx" if suffix else f"{name}.onnx"
        path = model_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到 Qwen3-ASR ONNX 文件: {name}")


def _load_embedding(model_dir: Path) -> np.ndarray:
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    shape = (
        config.get("embed_tokens_shape")
        or [config["decoder"]["vocab_size"], config["decoder"]["hidden_size"]]
    )
    dtype = np.float16 if config.get("embed_tokens_dtype") == "float16" else np.float32
    return np.fromfile(model_dir / "embed_tokens.bin", dtype=dtype).reshape(shape).astype(np.float32)


class QwenOnnxASREngine:
    """基于 ONNX Runtime 的 Qwen3-ASR 推理器。"""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        max_decode_tokens: int = 512,
        verbose: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.max_decode_tokens = int(max_decode_tokens)
        self.verbose = bool(verbose)
        self.device = normalize_runtime_device(device)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.embedding_table = _load_embedding(self.model_dir)
        self.mel_extractor = FastWhisperMel()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")

        encoder_providers = ["CPUExecutionProvider"]
        if (
            self.device == "metal"
            and sys.platform == "darwin"
            and "CoreMLExecutionProvider" in ort.get_available_providers()
        ):
            encoder_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        encoder_path = _resolve_variant(self.model_dir, "encoder")
        init_path = _resolve_variant(self.model_dir, "decoder_init")
        step_path = _resolve_variant(self.model_dir, "decoder_step")
        if self.verbose:
            print(f"[Qwen ONNX] encoder={encoder_path.name}, init={init_path.name}, step={step_path.name}")

        # Decoder step 在 M4 CoreML 编译阶段会失败，故明确使用 CPU；这不是静默
        # 忽略请求，而是为了避免把一个可加载的模型变成启动即崩溃的模型。
        self.encoder_session = ort.InferenceSession(str(encoder_path), opts, providers=encoder_providers)
        self.decoder_init_session = ort.InferenceSession(
            str(init_path), opts, providers=["CPUExecutionProvider"]
        )
        self.decoder_step_session = ort.InferenceSession(
            str(step_path), opts, providers=["CPUExecutionProvider"]
        )
        self.encoder_runtime = {
            "encoder_provider": self.encoder_session.get_providers()[0],
            "frontend_providers": list(self.encoder_session.get_providers()),
            "backend_providers": [],
        }
        self.decoder_backend = "onnx-cpu"
        self._mel_dtype = np.float32

    def _decode(self, audio: np.ndarray, language: str | None, context: str | None) -> tuple[str, dict]:
        t_start = time.perf_counter()
        mel = self.mel_extractor(audio, dtype=self._mel_dtype)[None, :, :]
        audio_features = self.encoder_session.run(["audio_features"], {"mel": mel})[0]
        if audio_features.shape[1] == 0:
            return "", {"total_time": time.perf_counter() - t_start}

        prompt_ids = _build_prompt_ids(self.tokenizer, audio_features.shape[1], language, context)
        position_ids = np.arange(len(prompt_ids), dtype=np.int64)[None, :]
        init_inputs = {
            "input_ids": np.asarray(prompt_ids, dtype=np.int64)[None, :],
            "position_ids": position_ids,
            "audio_features": audio_features,
            "audio_offset": np.asarray([prompt_ids.index(AUDIO_PAD_TOKEN_ID)], dtype=np.int64),
        }
        logits, present_keys, present_values = self.decoder_init_session.run(
            ["logits", "present_keys", "present_values"], init_inputs
        )
        next_token = int(np.argmax(logits[0, -1, :]))
        tokens = [next_token]
        position = len(prompt_ids)
        for _ in range(max(self.max_decode_tokens - 1, 0)):
            if next_token in EOS_TOKEN_IDS:
                break
            step_inputs = {
                "input_embeds": self.embedding_table[next_token][None, None, :],
                "position_ids": np.asarray([[position]], dtype=np.int64),
                "past_keys": present_keys,
                "past_values": present_values,
            }
            logits, present_keys, present_values = self.decoder_step_session.run(
                ["logits", "present_keys", "present_values"], step_inputs
            )
            next_token = int(np.argmax(logits[0, -1, :]))
            tokens.append(next_token)
            position += 1

        while tokens and tokens[-1] in EOS_TOKEN_IDS:
            tokens.pop()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text, {
            "total_time": time.perf_counter() - t_start,
            "audio_duration": len(audio) / 16000.0,
            "encoder_tokens": int(audio_features.shape[1]),
            "decode_tokens": len(tokens),
        }

    def transcribe(
        self,
        audio,
        language: str | None = None,
        context: str | None = None,
        batch_size: int | None = None,
    ):
        from .utils import load_audio

        normalized_language = LANGUAGE_ALIASES.get(str(language or "").lower(), language)
        if isinstance(audio, (str, Path)):
            waveform = load_audio(str(audio))
            text, performance = self._decode(waveform, normalized_language, context)
            return TranscribeResult(text=text, performance=performance)
        if isinstance(audio, Iterable):
            return self.transcribe_batch(list(audio), language=language, context=context, batch_size=batch_size)
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    def transcribe_batch(
        self,
        audio_files: list[str | Path],
        language: str | None = None,
        context: str | None = None,
        batch_size: int | None = None,
    ) -> list[TranscribeResult]:
        from .utils import load_audio

        del batch_size  # 当前解码图的 KV cache 为单流，按文件复用 session 更稳妥。
        normalized_language = LANGUAGE_ALIASES.get(str(language or "").lower(), language)
        results = []
        for audio_file in audio_files:
            waveform = load_audio(str(audio_file))
            text, performance = self._decode(waveform, normalized_language, context)
            results.append(TranscribeResult(text=text, performance=performance))
        return results

    def shutdown(self) -> None:
        self.encoder_session = None
        self.decoder_init_session = None
        self.decoder_step_session = None
        self.embedding_table = None
