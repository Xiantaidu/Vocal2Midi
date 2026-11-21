import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference import device_utils
from inference.qwen3asr_dml import llama as llama_module
from inference.qwen3asr_dml.asr import QwenASREngine
from inference.qwen3asr_dml.llama import (
    LLAMA_BACKEND_AUTO,
    LLAMA_BACKEND_CPU,
    LLAMA_BACKEND_METAL,
    LLAMA_SPLIT_MODE_NONE,
    LLAMA_BACKEND_VULKAN,
    detect_available_llama_backend,
    _metal_backend_filenames,
    _vulkan_backend_filename,
)
from inference.qwen3asr_dml import runtime as runtime_module
from inference.qwen3asr_dml.runtime import Qwen3ASRDmlModel
from inference.qwen3asr_dml.runtime import (
    resolve_model_dir,
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


def test_resolve_model_dir_accepts_split_onnx_package(tmp_path: Path):
    for filename in [
        "embed_tokens.bin",
        "tokenizer.json",
        "encoder.int4.onnx",
        "decoder_init.int4.onnx",
        "decoder_step.int4.onnx",
    ]:
        (tmp_path / filename).write_bytes(b"model")

    assert resolve_model_dir(tmp_path) == tmp_path


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
    (tmp_path / _vulkan_backend_filename()).write_bytes(b"vk")
    assert detect_available_llama_backend(tmp_path) == LLAMA_BACKEND_VULKAN


def test_detect_available_llama_backend_detects_metal_on_macos(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(llama_module.sys, "platform", "darwin")
    (tmp_path / _metal_backend_filenames()[0]).write_bytes(b"metal")

    assert detect_available_llama_backend(tmp_path) == LLAMA_BACKEND_METAL


def test_resolve_llama_backend_forces_cpu_when_runtime_device_is_cpu(monkeypatch):
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")

    assert resolve_llama_backend("cpu", LLAMA_BACKEND_AUTO) == LLAMA_BACKEND_CPU


def test_resolve_llama_backend_keeps_auto_for_cpu_when_macos(monkeypatch):
    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")

    assert resolve_llama_backend("cpu", LLAMA_BACKEND_AUTO) == LLAMA_BACKEND_AUTO


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


def _install_fake_ggml_devices(monkeypatch, devices):
    """Install a fake ggml device table: [(device_name, description), ...]."""
    pointers = [0x1000 + index for index in range(len(devices))]
    entries = dict(zip(pointers, devices))

    monkeypatch.setattr(llama_module, "ggml_backend_dev_count", lambda: len(devices))
    monkeypatch.setattr(llama_module, "ggml_backend_dev_get", lambda index: pointers[index])
    monkeypatch.setattr(
        llama_module,
        "ggml_backend_dev_name",
        lambda device: entries[device][0].encode("utf-8"),
    )
    monkeypatch.setattr(
        llama_module,
        "ggml_backend_dev_description",
        lambda device: entries[device][1].encode("utf-8"),
    )
    monkeypatch.setattr(llama_module, "ggml_backend_dev_backend_reg", None)
    monkeypatch.setattr(llama_module, "ggml_backend_reg_name", None)
    return pointers


def test_configure_model_params_for_backend_falls_back_without_device_mapping(monkeypatch):
    for name in (
        "ggml_backend_dev_count",
        "ggml_backend_dev_get",
        "ggml_backend_dev_name",
        "ggml_backend_dev_description",
        "ggml_backend_dev_backend_reg",
        "ggml_backend_reg_name",
    ):
        monkeypatch.setattr(llama_module, name, None)

    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
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
    # main_gpu indexes llama.cpp's device list, not the DXGI adapter order.
    assert model_params.main_gpu == 0
    assert model_params.devices is None


def test_configure_model_params_pins_matching_ggml_vulkan_device(monkeypatch):
    pointers = _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Vulkan0", "AMD Radeon(TM) 610M"),
            ("Vulkan1", "NVIDIA GeForce RTX 5060 Laptop GPU"),
            ("CPU", "AMD Ryzen 9 8945HX"),
        ],
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=1,
        name="NVIDIA GeForce RTX 5060 Laptop GPU",
        dedicated_vram_bytes=8 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.main_gpu == 0
    # The DXGI adapter (index 1) maps to the matching Vulkan device, not to
    # the same numeric index in llama.cpp's list.
    assert model_params.devices is not None
    assert model_params.devices[0] == pointers[1]
    assert model_params.devices[1] is None


def test_configure_model_params_uses_ggml_backend_registry_to_identify_vulkan(monkeypatch):
    pointers = _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Custom0", "NVIDIA GeForce RTX 5060 Laptop GPU"),
            ("CPU", "AMD Ryzen 9 8945HX"),
        ],
    )
    registries = {pointers[0]: 0x9000, pointers[1]: 0x9001}
    monkeypatch.setattr(
        llama_module, "ggml_backend_dev_backend_reg", lambda device: registries[device]
    )
    monkeypatch.setattr(
        llama_module,
        "ggml_backend_reg_name",
        lambda registry: b"Vulkan" if registry == 0x9000 else b"CPU",
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=1,
        name="NVIDIA GeForce RTX 5060 Laptop GPU",
        dedicated_vram_bytes=8 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    # Device names are backend-defined; the registry name is authoritative.
    assert model_params.devices[0] == pointers[0]


def test_configure_model_params_matches_device_name_ignoring_formatting(monkeypatch):
    pointers = _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Vulkan0", "  NVIDIA   GeForce RTX 5060 Laptop GPU "),
            ("Vulkan1", "AMD Radeon(TM) 610M"),
        ],
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=3,
        name="NVIDIA GeForce RTX 5060 Laptop GPU",
        dedicated_vram_bytes=8 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.devices[0] == pointers[0]


def test_configure_model_params_pins_only_vulkan_device_when_name_differs(monkeypatch):
    pointers = _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Vulkan0", "Renamed GPU"),
            ("CPU", "Some CPU"),
        ],
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=1,
        name="Discrete GPU",
        dedicated_vram_bytes=4 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.devices[0] == pointers[0]


def test_configure_model_params_accepts_vendor_suffix_in_ggml_name(monkeypatch):
    pointers = _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Vulkan0", "NVIDIA GeForce RTX 5060 Laptop GPU (NVIDIA)"),
            ("Vulkan1", "AMD Radeon(TM) 610M"),
        ],
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=1,
        name="NVIDIA GeForce RTX 5060 Laptop GPU",
        dedicated_vram_bytes=8 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.devices[0] == pointers[0]


def test_configure_model_params_does_not_match_similar_model_names(monkeypatch):
    _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Vulkan0", "NVIDIA GeForce RTX 5060 Ti"),
            ("Vulkan1", "AMD Radeon(TM) 610M"),
        ],
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=1,
        name="NVIDIA GeForce RTX 5060",
        dedicated_vram_bytes=8 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.devices is None


def test_configure_model_params_falls_back_when_multiple_devices_do_not_match(monkeypatch):
    _install_fake_ggml_devices(
        monkeypatch,
        [
            ("Vulkan0", "GPU A"),
            ("Vulkan1", "GPU B"),
        ],
    )
    model_params = SimpleNamespace(
        n_gpu_layers=None, split_mode=None, main_gpu=None, devices=None
    )
    adapter = device_utils.DxgiAdapterInfo(
        index=1,
        name="GPU C",
        dedicated_vram_bytes=4 * (1 << 30),
        is_software=False,
    )

    llama_module._configure_model_params_for_backend(
        model_params,
        LLAMA_BACKEND_VULKAN,
        adapter=adapter,
    )

    assert model_params.main_gpu == 0
    assert model_params.devices is None


def test_python_log_callback_maps_ggml_log_levels(caplog):
    caplog.set_level(logging.DEBUG, logger=llama_module.logger.name)

    llama_module.python_log_callback(1, b"debug message", None)
    llama_module.python_log_callback(2, b"info message", None)
    llama_module.python_log_callback(3, b"warn message", None)
    llama_module.python_log_callback(4, b"error message", None)
    llama_module.python_log_callback(5, b"continuation", None)

    records = [(record.levelno, record.getMessage()) for record in caplog.records]

    assert (logging.DEBUG, "[llama.cpp] debug message") in records
    assert (logging.INFO, "[llama.cpp] info message") in records
    assert (logging.WARNING, "[llama.cpp] warn message") in records
    assert (logging.ERROR, "[llama.cpp] error message") in records
    assert (logging.DEBUG, "[llama.cpp] continuation") in records


def test_python_log_callback_keeps_errors_visible_at_warning_level(caplog):
    caplog.set_level(logging.WARNING, logger=llama_module.logger.name)

    llama_module.python_log_callback(4, b"invalid value for main_gpu", None)

    assert any(
        record.levelno == logging.ERROR
        and "invalid value for main_gpu" in record.getMessage()
        for record in caplog.records
    )
