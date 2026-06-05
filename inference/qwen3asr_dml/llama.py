import sys
import os
import ctypes
import codecs
import struct
import time
import numpy as np
from . import gguf
from .gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from typing import List, Union
from pathlib import Path
from os.path import relpath
from typing import Union
from . import logger
from inference.device_utils import (
    MIN_GPU_DEDICATED_VRAM_BYTES,
    DxgiAdapterInfo,
    describe_gpu_adapter,
    format_gib,
    select_preferred_gpu_adapter,
)

# =========================================================================
# Configuration
# =========================================================================
# When QUIET_LOGS is True, suppress direct logging. Logs are currently routed through logger.
QUIET_LOGS = False
_log_callback_ref = None

# =========================================================================
# Type Definitions
# =========================================================================

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]

class llama_logit_bias(ctypes.Structure):
    _fields_ = [
        ("token", llama_token),
        ("bias", ctypes.c_float),
    ]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# =========================================================================
# Llama Library Bindings
# =========================================================================

# Global library references
llama = None
ggml = None
ggml_base = None

# Global function pointers
llama_log_set = None
llama_backend_init = None
llama_backend_free = None
llama_model_default_params = None
llama_model_load_from_file = None
llama_model_free = None
llama_model_get_vocab = None
llama_context_default_params = None
llama_init_from_model = None
llama_free = None
llama_batch_init = None
llama_batch_free = None
llama_decode = None
llama_get_logits = None
llama_get_logits_ith = None
llama_get_embeddings = None
llama_tokenize = None
llama_vocab_n_tokens = None
llama_vocab_eos = None
llama_token_to_piece = None
llama_get_memory = None
llama_memory_clear = None
llama_model_n_embd = None

# Sampler
llama_sampler_chain_default_params = None
llama_sampler_chain_init = None
llama_sampler_chain_add = None
llama_sampler_init_greedy = None
llama_sampler_init_dist = None
llama_sampler_init_temp = None
llama_sampler_init_top_k = None
llama_sampler_init_top_p = None
llama_sampler_sample = None
llama_sampler_free = None

LLAMA_BACKEND_CPU = "cpu"
LLAMA_BACKEND_VULKAN = "vulkan"
LLAMA_BACKEND_AUTO = "auto"
LLAMA_SPLIT_MODE_NONE = 0


def _vulkan_backend_filename() -> str:
    if sys.platform == "win32":
        return "ggml-vulkan.dll"
    if sys.platform == "darwin":
        return "libggml-vulkan.dylib"
    return "libggml-vulkan.so"


def detect_available_llama_backend(lib_dir: str | Path | None = None) -> str:
    base_dir = Path(lib_dir) if lib_dir is not None else Path(__file__).parent / "bin"
    if (base_dir / _vulkan_backend_filename()).is_file():
        return LLAMA_BACKEND_VULKAN
    return LLAMA_BACKEND_CPU


def _resolve_backend_and_adapter(
    backend: str,
    lib_dir: str | Path | None = None,
) -> tuple[str, DxgiAdapterInfo | None]:
    requested_backend = (backend or LLAMA_BACKEND_AUTO).strip().lower()
    if requested_backend not in {LLAMA_BACKEND_AUTO, LLAMA_BACKEND_CPU, LLAMA_BACKEND_VULKAN}:
        logger.warning(f"Unknown llama backend '{backend}', falling back to auto.")
        requested_backend = LLAMA_BACKEND_AUTO

    detected_backend = detect_available_llama_backend(lib_dir)
    selected_backend = detected_backend if requested_backend == LLAMA_BACKEND_AUTO else requested_backend
    if selected_backend == LLAMA_BACKEND_VULKAN and detected_backend != LLAMA_BACKEND_VULKAN:
        logger.warning("Requested Vulkan llama backend, but ggml-vulkan backend DLL is missing. Falling back to CPU.")
        return LLAMA_BACKEND_CPU, None
    if selected_backend != LLAMA_BACKEND_VULKAN:
        return selected_backend, None

    adapter = select_preferred_gpu_adapter(MIN_GPU_DEDICATED_VRAM_BYTES)
    if adapter is None:
        logger.warning(
            "No GPU adapter with at least "
            f"{format_gib(MIN_GPU_DEDICATED_VRAM_BYTES)} dedicated VRAM was found for Vulkan offload. "
            "Falling back to CPU."
        )
        return LLAMA_BACKEND_CPU, None

    logger.info(f"Selected Vulkan {describe_gpu_adapter(adapter)}.")
    return LLAMA_BACKEND_VULKAN, adapter


def _configure_model_params_for_backend(
    model_params,
    active_backend: str,
    *,
    n_gpu_layers: int | None = None,
    adapter: DxgiAdapterInfo | None = None,
) -> None:
    if active_backend == LLAMA_BACKEND_VULKAN:
        model_params.n_gpu_layers = -1 if n_gpu_layers is None else int(n_gpu_layers)
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE
        if adapter is not None:
            model_params.main_gpu = int(adapter.index)
            logger.info(
                "llama.cpp decoder is running with Vulkan offload on "
                f"{describe_gpu_adapter(adapter)} (n_gpu_layers={model_params.n_gpu_layers})."
            )
        else:
            logger.info(
                "llama.cpp decoder is running with Vulkan offload "
                f"(n_gpu_layers={model_params.n_gpu_layers})."
            )
        return

    model_params.n_gpu_layers = 0
    logger.info("llama.cpp decoder is running in CPU-only mode.")

def init_llama_lib():
    """Initialize the llama.cpp library with cross-platform loading support."""
    global llama, ggml, ggml_base
    global llama_log_set, llama_backend_init, llama_backend_free
    global llama_model_default_params, llama_model_load_from_file, llama_model_free, llama_model_get_vocab
    global llama_context_default_params, llama_init_from_model, llama_free
    global llama_batch_init, llama_batch_free, llama_batch_get_one
    global llama_decode, llama_get_logits, llama_get_logits_ith, llama_get_embeddings, llama_tokenize
    global llama_get_memory, llama_memory_clear, llama_model_n_embd
    global llama_vocab_n_tokens, llama_vocab_eos, llama_token_to_piece
    global llama_sampler_chain_default_params, llama_sampler_chain_init, llama_sampler_chain_add
    global llama_sampler_init_greedy, llama_sampler_init_dist, llama_sampler_init_temp
    global llama_sampler_init_top_k, llama_sampler_init_top_p, llama_sampler_sample, llama_sampler_free
    global llama_sampler_init_logit_bias
    global _log_callback_ref

    if llama is not None:
        return

    # Resolve the directory that contains the shared libraries (the module's bin folder).
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")

    # Platform-specific shared library names.
    if sys.platform == "win32":
        GGML_DLL = "ggml.dll"
        GGML_BASE_DLL = "ggml-base.dll"
        LLAMA_DLL = "llama.dll"
    elif sys.platform == "darwin":
        GGML_DLL = "libggml.dylib"
        GGML_BASE_DLL = "libggml-base.dylib"
        LLAMA_DLL = "libllama.dylib"
    else:
        GGML_DLL = "libggml.so"
        GGML_BASE_DLL = "libggml-base.so"
        LLAMA_DLL = "libllama.so"

    ggml = ctypes.CDLL(os.path.join(lib_dir, GGML_DLL))
    ggml_base = ctypes.CDLL(os.path.join(lib_dir, GGML_BASE_DLL))
    llama = ctypes.CDLL(os.path.join(lib_dir, LLAMA_DLL))

    # Register the log callback.
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    llama_log_set = llama.llama_log_set
    llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
    llama_log_set.restype = None
    
    # Enable log routing by default.
    configure_logging(quiet=QUIET_LOGS)

    # Load backend implementations.
    ggml_backend_load_all = ggml.ggml_backend_load_all
    ggml_backend_load_all.argtypes = []
    ggml_backend_load_all.restype = None
    ggml_backend_load_all()

    llama_backend_init = llama.llama_backend_init
    llama_backend_init.argtypes = []
    llama_backend_init.restype = None
    llama_backend_init()

    # Bind the remaining exported functions.
    llama_backend_free = llama.llama_backend_free
    llama_backend_free.argtypes = []
    llama_backend_free.restype = None

    llama_model_default_params = llama.llama_model_default_params
    llama_model_default_params.argtypes = []
    llama_model_default_params.restype = llama_model_params

    llama_model_load_from_file = llama.llama_model_load_from_file
    llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama_model_load_from_file.restype = ctypes.c_void_p

    llama_model_free = llama.llama_model_free
    llama_model_free.argtypes = [ctypes.c_void_p]
    llama_model_free.restype = None

    llama_model_get_vocab = llama.llama_model_get_vocab
    llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    llama_model_get_vocab.restype = ctypes.c_void_p

    llama_model_n_embd = llama.llama_model_n_embd
    llama_model_n_embd.argtypes = [ctypes.c_void_p]
    llama_model_n_embd.restype = ctypes.c_int32

    # Context
    llama_context_default_params = llama.llama_context_default_params
    llama_context_default_params.argtypes = []
    llama_context_default_params.restype = llama_context_params

    llama_init_from_model = llama.llama_init_from_model
    llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama_init_from_model.restype = ctypes.c_void_p

    llama_free = llama.llama_free
    llama_free.argtypes = [ctypes.c_void_p]
    llama_free.restype = None

    # Batch
    llama_batch_init = llama.llama_batch_init
    llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama_batch_init.restype = llama_batch

    llama_batch_free = llama.llama_batch_free
    llama_batch_free.argtypes = [llama_batch]
    llama_batch_free.restype = None
    
    llama_batch_get_one = llama.llama_batch_get_one
    llama_batch_get_one.argtypes = [ctypes.POINTER(llama_token), ctypes.c_int32]
    llama_batch_get_one.restype = llama_batch

    # Decode
    llama_decode = llama.llama_decode
    llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama_decode.restype = ctypes.c_int32

    # Logits
    llama_get_logits = llama.llama_get_logits
    llama_get_logits.argtypes = [ctypes.c_void_p]
    llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_logits_ith = llama.llama_get_logits_ith
    llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_embeddings = llama.llama_get_embeddings
    llama_get_embeddings.argtypes = [ctypes.c_void_p]
    llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

    # Tokenize
    llama_tokenize = llama.llama_tokenize
    llama_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(llama_token), ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    llama_tokenize.restype = ctypes.c_int32

    # Vocab
    llama_vocab_n_tokens = llama.llama_vocab_n_tokens
    llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    llama_vocab_n_tokens.restype = ctypes.c_int32

    llama_vocab_eos = llama.llama_vocab_eos
    llama_vocab_eos.argtypes = [ctypes.c_void_p]
    llama_vocab_eos.restype = llama_token

    llama_token_to_piece = llama.llama_token_to_piece
    llama_token_to_piece.argtypes = [ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
    llama_token_to_piece.restype = ctypes.c_int

    # Memory (KV Cache)
    llama_get_memory = llama.llama_get_memory
    llama_get_memory.argtypes = [ctypes.c_void_p]
    llama_get_memory.restype = ctypes.c_void_p

    llama_memory_clear = llama.llama_memory_clear
    llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    llama_memory_clear.restype = None

    # Sampler
    llama_sampler_chain_default_params = llama.llama_sampler_chain_default_params
    llama_sampler_chain_default_params.argtypes = []
    llama_sampler_chain_default_params.restype = llama_sampler_chain_params

    llama_sampler_chain_init = llama.llama_sampler_chain_init
    llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
    llama_sampler_chain_init.restype = ctypes.c_void_p

    llama_sampler_chain_add = llama.llama_sampler_chain_add
    llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    llama_sampler_chain_add.restype = None

    llama_sampler_init_greedy = llama.llama_sampler_init_greedy
    llama_sampler_init_greedy.argtypes = []
    llama_sampler_init_greedy.restype = ctypes.c_void_p

    llama_sampler_init_dist = llama.llama_sampler_init_dist
    llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    llama_sampler_init_dist.restype = ctypes.c_void_p

    llama_sampler_init_temp = llama.llama_sampler_init_temp
    llama_sampler_init_temp.argtypes = [ctypes.c_float]
    llama_sampler_init_temp.restype = ctypes.c_void_p

    llama_sampler_init_top_k = llama.llama_sampler_init_top_k
    llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    llama_sampler_init_top_k.restype = ctypes.c_void_p

    llama_sampler_init_top_p = llama.llama_sampler_init_top_p
    llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    llama_sampler_init_top_p.restype = ctypes.c_void_p

    llama_sampler_sample = llama.llama_sampler_sample
    llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    llama_sampler_sample.restype = llama_token

    llama_sampler_free = llama.llama_sampler_free
    llama_sampler_free.argtypes = [ctypes.c_void_p]
    llama_sampler_free.restype = None

    llama_sampler_init_logit_bias = llama.llama_sampler_init_logit_bias
    llama_sampler_init_logit_bias.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(llama_logit_bias)]
    llama_sampler_init_logit_bias.restype = ctypes.c_void_p

def load_model(model_path: str):
    """
    Load a GGUF model while handling initialization and path encoding details.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        model: Pointer to the loaded llama_model.
    """
    lib_dir = Path(__file__).parent / 'bin'
    model_path = Path(model_path)
    model_rel = Path(relpath(model_path, lib_dir))

    # Temporarily switch to the DLL directory and prepend it to PATH.
    original_cwd = Path.cwd()
    os.chdir(lib_dir)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(os.getcwd())
    os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']
    logger.info(f"Changed directory to: {Path.cwd()}")

    # Initialize the backend and load the model.
    init_llama_lib()
    model_params = llama_model_default_params()
    _configure_model_params_for_backend(model_params, LLAMA_BACKEND_CPU)
    model = llama_model_load_from_file(
        model_rel.as_posix().encode('utf-8'),
        model_params
    )

    if model:
        os.chdir(original_cwd)
        logger.info(f"Restored directory to: {Path.cwd()}")
        return model
    else:
        logger.error(f'当前路径：{Path.cwd()}')
        logger.error(f'模型绝对路径：{model_path.as_posix()}')
        logger.error(f'模型可访问性：{model_path.exists()}')
        logger.error(f"模型加载失败: {model_path}")
        return None

def load_model_with_backend(
    model_path: str,
    backend: str = LLAMA_BACKEND_AUTO,
    n_gpu_layers: int | None = None,
    quiet: bool = False,
):
    """Load a GGUF model with optional Vulkan offload and CPU fallback."""
    lib_dir = Path(__file__).parent / "bin"
    model_path = Path(model_path)
    model_rel = Path(relpath(model_path, lib_dir))

    original_cwd = Path.cwd()
    os.chdir(lib_dir)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.getcwd())
    os.environ["PATH"] = os.getcwd() + os.pathsep + os.environ["PATH"]
    logger.info(f"Changed directory to: {Path.cwd()}")

    init_llama_lib()
    configure_logging(quiet=quiet)
    selected_backend, selected_adapter = _resolve_backend_and_adapter(backend, lib_dir)

    def _try_load(active_backend: str, adapter: DxgiAdapterInfo | None):
        model_params = llama_model_default_params()
        _configure_model_params_for_backend(
            model_params,
            active_backend,
            n_gpu_layers=n_gpu_layers,
            adapter=adapter,
        )
        return llama_model_load_from_file(
            model_rel.as_posix().encode("utf-8"),
            model_params,
        )

    model = _try_load(selected_backend, selected_adapter)
    active_backend = selected_backend
    if not model and selected_backend == LLAMA_BACKEND_VULKAN:
        logger.warning("Failed to load llama model with Vulkan offload; retrying in CPU-only mode.")
        model = _try_load(LLAMA_BACKEND_CPU, None)
        active_backend = LLAMA_BACKEND_CPU

    if model:
        os.chdir(original_cwd)
        logger.info(f"Restored directory to: {Path.cwd()}")
        return model, active_backend

    logger.error(f"Current working directory: {Path.cwd()}")
    logger.error(f"Model absolute path: {model_path.as_posix()}")
    logger.error(f"Model accessible: {model_path.exists()}")
    logger.error(f"Model load failed: {model_path}")
    return None, active_backend


def create_context(model, n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq_max=1, 
                   embeddings=False, pooling_type=0, flash_attn=True, 
                   offload_kqv=True, no_perf=True, n_threads=None):
    """Create an ASR-oriented llama context."""
    params = llama_context_default_params()
    params.n_ctx = n_ctx
    params.n_batch = n_batch
    params.n_ubatch = n_ubatch
    params.n_seq_max = n_seq_max
    params.embeddings = embeddings
    params.pooling_type = pooling_type
    params.flash_attn_type = 1 if flash_attn else 0  # 1 = ON, 0 = OFF (auto typically uses what's available)
    params.offload_kqv = offload_kqv
    params.no_perf = no_perf
    
    if n_threads:
        params.n_threads = n_threads
        params.n_threads_batch = n_threads
    else:
        params.n_threads = os.cpu_count() // 2
        params.n_threads_batch = os.cpu_count()

    return llama_init_from_model(model, params)

class LlamaModel:
    """Object-oriented wrapper for a llama model."""
    def __init__(self, path, backend: str = LLAMA_BACKEND_AUTO, quiet: bool = False):
        self.ptr, self.backend = load_model_with_backend(path, backend=backend, quiet=quiet)
            
        self.vocab = llama_model_get_vocab(self.ptr)
        self.n_embd = llama_model_n_embd(self.ptr)
        self.eos_token = llama_vocab_eos(self.vocab)

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = True) -> List[int]:
        """(Native) Convert text into a list of token IDs."""
        return text_to_tokens(self.vocab, text, add_special, parse_special)

    def detokenize(self, tokens: List[int]) -> str:
        """(Native) Convert a list of token IDs back to text."""
        if tokens is None or len(tokens) == 0: return ""
        all_bytes = b"".join([self.token_to_bytes(tid) for tid in tokens])
        return all_bytes.decode('utf-8', errors='replace')

    def token_to_bytes(self, token_id: int) -> bytes:
        """(Native) Convert a single token to raw bytes."""
        return token_to_bytes(self.vocab, token_id)
        
    def token_to_piece(self, token_id: int) -> str:
        """(Native) Convert a single token to its string piece."""
        return self.token_to_bytes(token_id).decode('utf-8', errors='replace')

    def token_bos(self) -> int:
        return llama_vocab_bos(self.vocab)

    def token_eos(self) -> int:
        return llama_vocab_eos(self.vocab)
        
    def token_to_id(self, text: str) -> int:
        """(Native) Convert a single token string to an ID (exact matches only)."""
        # Reuse tokenize() to look up the ID.
        res = self.tokenize(text, add_special=False, parse_special=True)
        return res[0] if res else -1

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            llama_model_free(self.ptr)
            self.ptr = None

class LlamaContext:
    """Object-oriented wrapper for a llama context."""
    def __init__(self, model, n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq_max=1, 
                 embeddings=False, pooling_type=0, flash_attn=True, 
                 offload_kqv=True, no_perf=True, n_threads=None, n_threads_batch=None):
        self.model = model # Keep a model reference so it is not released early.
        params = llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_ubatch = n_ubatch
        params.n_seq_max = n_seq_max
        params.embeddings = embeddings
        params.pooling_type = pooling_type
        params.flash_attn_type = 1 if flash_attn else 0
        params.offload_kqv = offload_kqv
        params.no_perf = no_perf
        
        # Thread configuration.
        cpu_count = os.cpu_count() or 4
        if n_threads:
            params.n_threads = n_threads
        else:
            params.n_threads = cpu_count // 2

        if n_threads_batch:
            params.n_threads_batch = n_threads_batch
        else:
            params.n_threads_batch = n_threads if n_threads else cpu_count

        self.ptr = llama_init_from_model(model.ptr, params)
        if not self.ptr:
            raise RuntimeError("上下文初始化失败")

    def decode(self, batch):
        struct = batch.struct if hasattr(batch, 'struct') else batch
        return llama_decode(self.ptr, struct)

    def decode_token(self, token_id):
        """
        Atomic helper that sets up a single-token batch and executes decoding.
        """
        return self.decode(get_one_batch(token_id))

    def get_logits(self):
        """Return logits for the last token in the batch with logits enabled."""
        return llama_get_logits(self.ptr)

    def get_logits_ith(self, i: int):
        """Return logits for the i-th token when its logits flag is enabled."""
        return llama_get_logits_ith(self.ptr, i)

    def get_embeddings(self):
        return llama_get_embeddings(self.ptr)

    def clear_kv_cache(self):
        mem = llama_get_memory(self.ptr)
        llama_memory_clear(mem, True)

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            llama_free(self.ptr)
            self.ptr = None

class LlamaBatch:
    """Object-oriented batch wrapper with direct attribute access."""
    def __init__(self, n_tokens, embd_dim=0, n_seq_max=1):
        self.struct = llama_batch_init(n_tokens, embd_dim, n_seq_max)
        self.n_tokens_max = n_tokens

    @property
    def n_tokens(self): return self.struct.n_tokens
    @n_tokens.setter
    def n_tokens(self, val): self.struct.n_tokens = val

    @property
    def token(self): return self.struct.token
    @property
    def embd(self): return self.struct.embd
    @property
    def pos(self): return self.struct.pos
    @property
    def n_seq_id(self): return self.struct.n_seq_id
    @property
    def seq_id(self): return self.struct.seq_id
    @property
    def logits(self): return self.struct.logits

    def set_embd(self, data: np.ndarray, pos: Union[np.ndarray, int] = 0, seq_id: int = 0):
        """
        Higher-level API that injects embedding data and initializes positions.

        Args:
            data: Embedding tensor with shape [n_tokens, dim].
            pos: Position information.
                 - If an int, treat it as the starting offset and generate
                   [offset, offset+1, ...] automatically.
                 - If an np.ndarray, copy it directly into the position buffer
                   to support complex encodings such as Qwen3.
            seq_id: Sequence ID.
        """
        n_tokens = data.shape[0]
        if n_tokens > self.n_tokens_max:
            raise ValueError(f"Batch 空间不足: {n_tokens} > {self.n_tokens_max}")
        
        # 1. Copy embedding memory.
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        ctypes.memmove(self.embd, data.ctypes.data, data.nbytes)
        
        # 2. Handle position information.
        if isinstance(pos, int):
            # Generate linear positions automatically.
            pos_offset = pos
            for i in range(n_tokens):
                self.pos[i] = pos_offset + i
        elif isinstance(pos, np.ndarray):
            # Complex external positions (for example Qwen3 multi-plane positions).
            # Do not require pos length to equal n_tokens because strided layouts
            # are allowed, but the data must still fit within batch capacity.
            if not pos.flags['C_CONTIGUOUS']:
                pos = np.ascontiguousarray(pos)
            
            # Copy directly with memmove.
            # self.pos is a ctypes pointer, so it can be written directly.
            ctypes.memmove(self.pos, pos.ctypes.data, pos.nbytes)
        else:
            raise TypeError(f"Unsupported pos type: {type(pos)}")

        # 3. Fill in the remaining metadata.
        self.n_tokens = n_tokens
        for i in range(n_tokens):
            self.n_seq_id[i] = 1
            self.seq_id[i][0] = seq_id
            self.logits[i] = 1 if i == n_tokens - 1 else 0
        
        return self

    def __del__(self):
        if hasattr(self, 'struct'):
            llama_batch_free(self.struct)

def get_one_batch(token_id: int):
    """
    Highly optimized low-level helper that builds an allocation-free batch for
    single-token generation.
    Equivalent to C++ llama_batch_get_one(&token, 1).
    It avoids llama_batch_init allocations and lets the backend infer pos.
    """
    token_arr = (llama_token * 1)(token_id)
    return llama_batch_get_one(token_arr, 1)

class LlamaSampler:
    """Object-oriented wrapper for a sampler."""
    def __init__(self, temperature=0.8, top_k=50, top_p=1.0, seed=None, logit_bias=None, n_vocab=0):
        import time
        if seed is None:
            seed = int(time.time())
            
        sparams = llama_sampler_chain_default_params()
        self.ptr = llama_sampler_chain_init(sparams)
        
        # Logit bias, including range- and mask-style constraints.
        if logit_bias and n_vocab > 0 and isinstance(logit_bias, dict):
            n_bias = len(logit_bias)
            BiasArray = llama_logit_bias * n_bias
            bias_data = BiasArray()
            
            for i, (token, bias) in enumerate(logit_bias.items()):
                bias_data[i].token = token
                bias_data[i].bias = bias
            
            llama_sampler_chain_add(self.ptr, llama_sampler_init_logit_bias(n_vocab, n_bias, bias_data))

        if temperature > 0:
            llama_sampler_chain_add(self.ptr, llama_sampler_init_top_k(top_k))
            llama_sampler_chain_add(self.ptr, llama_sampler_init_top_p(top_p, 1))
            llama_sampler_chain_add(self.ptr, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(self.ptr, llama_sampler_init_dist(seed))
        else:
            llama_sampler_chain_add(self.ptr, llama_sampler_init_greedy())

        self._neg_inf = -1e9

    def sample(self, ctx, idx=-1, limit_start=None, limit_end=None):
        """
        Sample one token.

        Args:
            limit_start (int, optional): Inclusive start index of the allowed sampling range.
            limit_end (int, optional): Exclusive end index of the allowed sampling range.
        """
        ctx_ptr = ctx
        if hasattr(ctx, 'ptr'):
            ctx_ptr = ctx.ptr
            
        # Apply dynamic range limits by editing the logits buffer in place.
        if (limit_start is not None or limit_end is not None) and hasattr(ctx, 'get_logits') and hasattr(ctx, 'model'):
            # Need n_vocab via LlamaContext -> LlamaModel -> vocab -> n_tokens.
            if hasattr(ctx.model, 'vocab'):
                n_vocab = llama_vocab_n_tokens(ctx.model.vocab)
                
                # Get a NumPy view over the logits buffer.
                logits_ptr = ctx.get_logits()
                logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
                
                # Modify logits in place.
                # This affects the current sampling step only; the next decode call
                # overwrites the buffer, so the mutation is safe.
                
                s = max(0, limit_start) if limit_start is not None else 0
                e = min(n_vocab, limit_end) if limit_end is not None else n_vocab
                
                if s > 0:
                    logits[0:s] = self._neg_inf
                if e < n_vocab:
                    logits[e:] = self._neg_inf
        
        return llama_sampler_sample(self.ptr, ctx_ptr, idx)

    def free(self):
        """Release sampler resources."""
        if hasattr(self, 'ptr') and self.ptr:
            llama_sampler_free(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

    def __del__(self):
        self.free()


class ASRStreamDecoder:
    """Streaming decoder specialized for ASR with reporter-aware byte decoding."""
    def __init__(self, vocab, reporter=None):
        self.vocab = vocab
        self.reporter = reporter
        self.byte_decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
        self.generated_text = ""
        self.tokens_generated = 0
        self.tokens = []

    def push(self, token_id: int):
        """Push one token and return any newly decoded text fragment."""
        raw_bytes = token_to_bytes(self.vocab, token_id)
        text_piece = self.byte_decoder.decode(raw_bytes, final=False)
        self.tokens.append(text_piece)
        self.tokens_generated += 1
        
        self.generated_text += text_piece
        
        if self.reporter:
            self.reporter.stream(text_piece)
            
        return text_piece

    def flush(self):
        """Flush any buffered bytes and return the decoded remainder."""
        remaining = self.byte_decoder.decode(b"", final=True)
        self.tokens.append(remaining)
        self.generated_text += remaining
        return remaining


def python_log_callback(level, message, user_data):
    """
    llama.cpp log callback.
    level: 
        2 = ERROR
        3 = WARN
        4 = INFO
        5 = DEBUG
    """
    if not message: return
    try:
        msg_str = message.decode('utf-8', errors='replace').strip()
        if not msg_str or msg_str in ['.', '\n']: return
        
        if level == 1:
            logger.info(f"[llama.cpp] {msg_str}")
        elif level == 2:
            logger.warning(f"[llama.cpp] {msg_str}")
        elif level == 3:
            logger.error(f"[llama.cpp] {msg_str}")
        elif level >= 4:
            logger.debug(f"[llama.cpp] {msg_str}")
        else:
            logger.info(f"[llama.cpp] {msg_str}")
    except Exception as e:
        # Prevent callback failures from crashing the process.
        print(f"日志回调出错: {e}")

def configure_logging(quiet=False):
    """Configure the llama.cpp log callback."""
    global _log_callback_ref
    if not llama_log_set: return
    
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    if not quiet:
        _log_callback_ref = LOG_CALLBACK(python_log_callback)
        llama_log_set(_log_callback_ref, None)
    else:
        # For quiet mode, register a no-op callback or raise the logger level elsewhere.
        _log_callback_ref = LOG_CALLBACK(lambda l, m, u: None)
        llama_log_set(_log_callback_ref, None)



# =========================================================================
# Embedding Table
# =========================================================================



class LlamaEmbeddingTable:
    """Dynamically dequantized embedding table that supports table[ids] access."""
    def __init__(self, raw_data, qtype):
        self.raw_data = raw_data
        self.qtype = qtype
        
    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, tokens):
        from gguf.quants import dequantize
        
        # Return directly for native float tensors.
        if self.raw_data.dtype in (np.float32, np.float16):
            return self.raw_data[tokens].astype(np.float32)
            
        # Use the official library for fast dequantization.
        return dequantize(self.raw_data[tokens], self.qtype.value)




def _skip_gguf_value(mm, offs, v_type):
    # UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
    fixed = [1, 1, 2, 2, 4, 4, 4, 1, -1, -2, 8, 8, 8]
    val_len = fixed[v_type]
    if val_len > 0:
        return offs + val_len
    elif val_len == -1: # string
        slen = struct.unpack_from("<Q", mm, offs)[0]
        return offs + 8 + slen
    elif val_len == -2: # array
        itype, alen = struct.unpack_from("<IQ", mm, offs)
        offs += 12
        if itype == 8: # string array
            for _ in range(alen):
                slen = struct.unpack_from("<Q", mm, offs)[0]
                offs += 8 + slen
        else:
            item_len = fixed[itype]
            if item_len > 0:
                offs += item_len * alen
            else:
                raise ValueError("Nested arrays or unknown type not supported in fast skip")
        return offs

def get_token_embeddings_gguf(model_path, target_tensor="token_embd.weight", quiet: bool = False):
    """
    Extremely fast GGUF embedding extraction through direct binary addressing.
    This avoids loading the full model and avoids parsing a tokenizer with
    150k entries, reducing runtime to under 50 ms.
    """
    t_start = time.time()
    mm = np.memmap(model_path, mode='r')
    
    # Read header metadata.
    tensor_count, kv_count = struct.unpack_from("<QQ", mm, 8)
    offs = 24
    alignment = 32
    
    # Fast-skip or scan all KV fields.
    for _ in range(kv_count):
        key_len = struct.unpack_from("<Q", mm, offs)[0]
        offs += 8
        if key_len == 17 and mm[offs:offs+17].tobytes() == b'general.alignment':
            offs += 17
            v_type = struct.unpack_from("<I", mm, offs)[0]
            offs += 4
            if v_type == 4: # UINT32
                alignment = struct.unpack_from("<I", mm, offs)[0]
                offs += 4
                continue
        else:
            offs += key_len
            
        v_type = struct.unpack_from("<I", mm, offs)[0]
        offs += 4
        offs = _skip_gguf_value(mm, offs, v_type)
        
    # Scan tensor infos to find the target tensor.
    target_rel_offset = None
    target_type = None
    target_shape = None # GGUF stores shape in reverse order: [n_embd, vocab_size]
    
    target_bytes = target_tensor.encode('utf-8')
    for _ in range(tensor_count):
        name_len = struct.unpack_from("<Q", mm, offs)[0]
        offs += 8
        is_target = False
        if name_len == len(target_bytes) and mm[offs:offs+name_len].tobytes() == target_bytes:
            is_target = True
        offs += name_len
        
        n_dims = struct.unpack_from("<I", mm, offs)[0]
        offs += 4
        
        shape = struct.unpack_from(f"<{n_dims}Q", mm, offs) # Returns a tuple.
        offs += 8 * n_dims
        
        t_type = struct.unpack_from("<I", mm, offs)[0]
        offs += 4
        
        rel_offset = struct.unpack_from("<Q", mm, offs)[0]
        offs += 8
        
        if is_target:
            target_shape = shape
            target_type = t_type
            target_rel_offset = rel_offset
            
    # Compute the data-section offset and load the tensor.
    padding = offs % alignment
    if padding != 0:
        offs += (alignment - padding)
    data_offset = offs
    
    if target_shape is None:
        logger.error(f"无法在 {model_path} 中找到 {target_tensor}")
        return None
        
    abs_offset = data_offset + target_rel_offset
    n_embd = target_shape[0]     # Feature dimension.
    vocab_size = target_shape[1] # Vocabulary size.
    
    qtype = GGMLQuantizationType(target_type)
    if qtype in GGML_QUANT_SIZES:
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        bytes_per_row = (n_embd // block_size) * type_size
    else:
        # F32 or F16.
        if qtype == GGMLQuantizationType.F32:
            bytes_per_row = n_embd * 4
        elif qtype == GGMLQuantizationType.F16:
            bytes_per_row = n_embd * 2
        else:
            raise ValueError(f"未知的数据格式支持: {qtype.name}")

    total_bytes = vocab_size * bytes_per_row
    raw_data = mm[abs_offset : abs_offset + total_bytes]
    
    if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        if qtype == GGMLQuantizationType.F32:
            raw_data = raw_data.view(np.float32).reshape(vocab_size, n_embd)
        else:
            raw_data = raw_data.view(np.float16).reshape(vocab_size, n_embd)
    else:
        raw_data = raw_data.reshape(vocab_size, bytes_per_row)
        
    total_time = time.time() - t_start
    if not quiet:
        logger.info(f"--- [QwenASR] 已极速载入 Embedding 视图 ({total_time*1000:.1f}ms) ---")
        logger.info(f"    - 量化格式: {qtype.name} ({n_embd} dims, {vocab_size} tokens)")
    
    return LlamaEmbeddingTable(raw_data, qtype)



# =========================================================================
# Utilities
# =========================================================================


def text_to_tokens(vocab, text, add_special=False, parse_special=True):
    text_bytes = text.encode("utf-8")
    n_tokens_max = len(text_bytes) + 32
    tokens = (llama_token * n_tokens_max)()
    n = llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, add_special, parse_special)
    return [tokens[i] for i in range(n)] if n >= 0 else []

def token_to_bytes(vocab, token_id):
    buf = ctypes.create_string_buffer(256)
    n = llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
    return buf.raw[:n] if n > 0 else b""
