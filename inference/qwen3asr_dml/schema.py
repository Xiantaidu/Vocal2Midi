# coding=utf-8
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


class MsgType(Enum):
    CMD_ENCODE = auto()
    CMD_STOP = auto()
    MSG_EMBD = auto()
    MSG_READY = auto()
    MSG_DONE = auto()
    MSG_ERROR = auto()


@dataclass
class StreamingMessage:
    msg_type: MsgType
    data: Any = None
    is_last: bool = False
    encode_time: float = 0.0


@dataclass
class DecodeResult:
    text: str = ""
    stable_tokens: List[int] = field(default_factory=list)
    t_prefill: float = 0.0
    t_generate: float = 0.0
    n_prefill: int = 0
    n_generate: int = 0
    is_aborted: bool = False


@dataclass
class ASREngineConfig:
    model_dir: str
    encoder_frontend_fn: str = "qwen3_asr_encoder_frontend.fp16.onnx"
    encoder_backend_fn: str = "qwen3_asr_encoder_backend.fp16.onnx"
    llm_fn: str = "qwen3_asr_llm.f16.gguf"
    use_dml: bool = True
    n_ctx: int = 2048
    chunk_size: float = 40.0
    memory_num: int = 1
    max_decode_tokens: int = 512
    llama_backend: str = "auto"
    verbose: bool = True


@dataclass
class TranscribeResult:
    text: str
    performance: Optional[dict] = None
