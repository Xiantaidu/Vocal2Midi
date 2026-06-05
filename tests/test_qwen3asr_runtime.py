from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference import device_utils
from inference.qwen3asr_dml import llama as llama_module
from inference.qwen3asr_dml.asr import QwenASREngine
from inference.qwen3asr_dml.llama import (
    LLAMA_BACKEND_AUTO,
    LLAMA_BACKEND_CPU,
    LLAMA_SPLIT_MODE_NONE,
    LLAMA_BACKEND_VULKAN,
    detect_available_llama_backend,
)
from inference.qwen3asr_dml.runtime import Qwen3ASRDmlModel
from inference.qwen3asr_dml.runtime import (
    resolve_encoder_filenames,
    resolve_llm_filename,
    resolve_llama_backend,
)
from inference.qwen3asr_dml.schema import MsgType, StreamingMessage


def test_runtime_transcribe_iterable_uses_engine_batch():
    calls = []

    class _DummyEngine:
        def transcribe(self, **kwargs):
            calls.append(("single", kwargs))
            return "single"

        def transcribe_batch(self, **kwargs):
            calls.append(("batch", kwargs))
            return ["a", "b"]

    model = Qwen3ASRDmlModel.__new__(Qwen3ASRDmlModel)
    model._engine = _DummyEngine()

    result = model.transcribe(["a.wav", "b.wav"], language="ja", context="ctx")

    assert result == ["a", "b"]
    assert calls == [
        (
            "batch",
            {
                "audio_files": ["a.wav", "b.wav"],
                "language": "Japanese",
                "context": "ctx",
            },
        )
    ]


def test_runtime_transcribe_iterable_respects_batch_size():
    calls = []

    class _DummyEngine:
        def transcribe_batch(self, **kwargs):
            calls.append(kwargs)
            return [f"out:{path}" for path in kwargs["audio_files"]]

    model = Qwen3ASRDmlModel.__new__(Qwen3ASRDmlModel)
    model._engine = _DummyEngine()

    result = model.transcribe(["a.wav", "b.wav", "c.wav"], language="zh", batch_size=2)

    assert result == ["out:a.wav", "out:b.wav", "out:c.wav"]
    assert calls == [
        {
            "audio_files": ["a.wav", "b.wav"],
            "language": "Chinese",
            "context": None,
        },
        {
            "audio_files": ["c.wav"],
            "language": "Chinese",
            "context": None,
        },
    ]


def test_runtime_exposes_encoder_and_decoder_runtime_summary():
    model = Qwen3ASRDmlModel.__new__(Qwen3ASRDmlModel)
    model._engine = SimpleNamespace(
        encoder_runtime={
            "encoder_provider": "dml",
            "frontend_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "backend_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        },
        decoder_backend="vulkan",
    )

    assert model.encoder_provider_name == "dml"
    assert model.encoder_frontend_providers == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert model.encoder_backend_providers == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert model.decoder_backend == "vulkan"


def test_engine_asr_batch_sends_real_batched_encode_requests():
    sent_messages = []
    queued_messages = [
        StreamingMessage(
            MsgType.MSG_EMBD,
            data=[
                np.ones((13, 4), dtype=np.float32),
                np.ones((13, 4), dtype=np.float32) * 2,
            ],
            encode_time=0.1,
        ),
        StreamingMessage(
            MsgType.MSG_EMBD,
            data=[
                np.ones((13, 4), dtype=np.float32) * 3,
            ],
            encode_time=0.2,
        ),
    ]

    class _Queue:
        def __init__(self, recv=None):
            self._recv = recv or []

        def put(self, item):
            sent_messages.append(item)

        def get(self):
            return self._recv.pop(0)

    engine = QwenASREngine.__new__(QwenASREngine)
    engine.verbose = False
    engine.to_worker_q = _Queue()
    engine.from_enc_q = _Queue(queued_messages)
    engine._build_prompt_embd = lambda audio_embd, context, language: audio_embd
    engine._safe_decode = lambda full_embd, rollback_num, temperature: SimpleNamespace(
        text=f"len{full_embd.shape[0]}",
        n_prefill=full_embd.shape[0],
        t_prefill=0.0,
        n_generate=1,
        t_generate=0.0,
    )

    short_audio = np.zeros(16000, dtype=np.float32)
    long_audio = np.zeros(32000, dtype=np.float32)
    results = engine.asr_batch(
        [short_audio, long_audio],
        context="",
        language=None,
        chunk_size_sec=1.0,
        memory_chunks=1,
    )

    assert [msg.msg_type for msg in sent_messages] == [MsgType.CMD_ENCODE, MsgType.CMD_ENCODE]
    assert [len(msg.data) for msg in sent_messages] == [2, 1]
    assert [result.text for result in results] == ["len13", "len13len13"]


def test_resolve_encoder_filenames_prefers_fp16(tmp_path: Path):
    (tmp_path / "qwen3_asr_llm.q4_k.gguf").write_bytes(b"llm")
    (tmp_path / "qwen3_asr_encoder_frontend.int4.onnx").write_bytes(b"int4fe")
    (tmp_path / "qwen3_asr_encoder_backend.int4.onnx").write_bytes(b"int4be")
    (tmp_path / "qwen3_asr_encoder_frontend.fp16.onnx").write_bytes(b"fp16fe")
    (tmp_path / "qwen3_asr_encoder_backend.fp16.onnx").write_bytes(b"fp16be")

    assert resolve_encoder_filenames(tmp_path) == (
        "qwen3_asr_encoder_frontend.fp16.onnx",
        "qwen3_asr_encoder_backend.fp16.onnx",
    )


def test_resolve_llm_filename_prefers_fp16(tmp_path: Path):
    (tmp_path / "qwen3_asr_llm.q4_k.gguf").write_bytes(b"q4")
    (tmp_path / "qwen3_asr_llm.f16.gguf").write_bytes(b"f16")

    assert resolve_llm_filename(tmp_path) == "qwen3_asr_llm.f16.gguf"


def test_detect_available_llama_backend_prefers_vulkan_when_present(tmp_path: Path):
    assert detect_available_llama_backend(tmp_path) == LLAMA_BACKEND_CPU
    (tmp_path / ("ggml-vulkan.dll")).write_bytes(b"vk")
    assert detect_available_llama_backend(tmp_path) == LLAMA_BACKEND_VULKAN


def test_resolve_llama_backend_forces_cpu_when_runtime_device_is_cpu():
    assert resolve_llama_backend("cpu", LLAMA_BACKEND_AUTO) == LLAMA_BACKEND_CPU


def test_resolve_llama_backend_keeps_auto_when_runtime_device_is_dml():
    assert resolve_llama_backend("dml", LLAMA_BACKEND_AUTO) == LLAMA_BACKEND_AUTO


def test_resolve_backend_and_adapter_falls_back_to_cpu_without_eligible_gpu(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(llama_module, "detect_available_llama_backend", lambda lib_dir=None: LLAMA_BACKEND_VULKAN)
    monkeypatch.setattr(
        llama_module,
        "select_preferred_gpu_adapter",
        lambda min_dedicated_vram_bytes=device_utils.MIN_GPU_DEDICATED_VRAM_BYTES: None,
    )

    backend, adapter = llama_module._resolve_backend_and_adapter(LLAMA_BACKEND_AUTO, tmp_path)

    assert backend == LLAMA_BACKEND_CPU
    assert adapter is None


def test_configure_model_params_for_backend_pins_selected_vulkan_gpu():
    model_params = SimpleNamespace(n_gpu_layers=None, split_mode=None, main_gpu=None)
    adapter = device_utils.DxgiAdapterInfo(
        index=2,
        name="Discrete GPU",
        dedicated_vram_bytes=4 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.n_gpu_layers == -1
    assert model_params.split_mode == LLAMA_SPLIT_MODE_NONE
    assert model_params.main_gpu == 2
