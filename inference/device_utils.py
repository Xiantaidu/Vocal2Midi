from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache

import onnxruntime as ort


RUNTIME_DEVICE_CHOICES = ("dml", "cpu", "cuda")
VISIBLE_RUNTIME_DEVICE_CHOICES = ("dml", "cpu")

ProviderSpec = str | tuple[str, dict[str, str]]
MIN_GPU_DEDICATED_VRAM_BYTES = 1 << 30

_DEVICE_ALIASES = {
    "": "dml",
    "cuda": "dml",
    "directml": "dml",
    "dml": "dml",
    "gpu": "dml",
    "cpu": "cpu",
}
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
_DXGI_ERROR_NOT_FOUND = 0x887A0002


@dataclass(frozen=True)
class DxgiAdapterInfo:
    index: int
    name: str
    dedicated_vram_bytes: int
    is_software: bool


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint32),
    ]


_IID_IDXGIFactory1 = _GUID(
    0x770AAE78,
    0xF26F,
    0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)


def normalize_runtime_device(device: str | None, default: str = "dml") -> str:
    value = str(device or default).strip().lower()
    if not value:
        value = default
    return _DEVICE_ALIASES.get(value, value)


def _hresult_code(value: int) -> int:
    return ctypes.c_uint32(value).value


def _call_com_method(obj: ctypes.c_void_p, method_index: int, restype, argtypes, *args):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    func_ptr = int(vtable[method_index])
    func = ctypes.WINFUNCTYPE(restype, *argtypes)(func_ptr)
    return func(*args)


def _release_com(obj: ctypes.c_void_p | None) -> None:
    if obj is None or not getattr(obj, "value", None):
        return
    _call_com_method(obj, 2, ctypes.c_ulong, [ctypes.c_void_p], obj)


@lru_cache(maxsize=1)
def _enumerate_dxgi_adapters() -> tuple[DxgiAdapterInfo, ...]:
    if not hasattr(ctypes, "WinDLL"):
        return ()
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return ()

    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    adapters: list[DxgiAdapterInfo] = []
    try:
        hr = create_factory(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
        if hr < 0 or not factory.value:
            return ()

        adapter_index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = _call_com_method(
                factory,
                12,
                ctypes.c_long,
                [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
                factory,
                adapter_index,
                ctypes.byref(adapter),
            )
            if _hresult_code(hr) == _DXGI_ERROR_NOT_FOUND:
                break
            if hr < 0 or not adapter.value:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                hr_desc = _call_com_method(
                    adapter,
                    10,
                    ctypes.c_long,
                    [ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1)],
                    adapter,
                    ctypes.byref(desc),
                )
                if hr_desc >= 0:
                    adapters.append(
                        DxgiAdapterInfo(
                            index=adapter_index,
                            name=str(desc.Description).rstrip("\x00").strip() or f"Adapter {adapter_index}",
                            dedicated_vram_bytes=int(desc.DedicatedVideoMemory),
                            is_software=bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE),
                        )
                    )
            finally:
                _release_com(adapter)
            adapter_index += 1
    except Exception:
        return ()
    finally:
        _release_com(factory)

    return tuple(adapters)


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / float(1 << 30):.1f} GiB"


def describe_gpu_adapter(adapter: DxgiAdapterInfo) -> str:
    return f"adapter {adapter.index}: {adapter.name} ({format_gib(adapter.dedicated_vram_bytes)} dedicated VRAM)"


@lru_cache(maxsize=8)
def select_preferred_gpu_adapter(
    min_dedicated_vram_bytes: int = MIN_GPU_DEDICATED_VRAM_BYTES,
) -> DxgiAdapterInfo | None:
    eligible = [
        adapter
        for adapter in _enumerate_dxgi_adapters()
        if not adapter.is_software and adapter.dedicated_vram_bytes >= int(min_dedicated_vram_bytes)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda adapter: (adapter.dedicated_vram_bytes, -adapter.index))


def _select_preferred_dml_adapter() -> DxgiAdapterInfo | None:
    return select_preferred_gpu_adapter(MIN_GPU_DEDICATED_VRAM_BYTES)


def resolve_onnx_providers(device: str | None, *, label: str = "ONNX") -> tuple[str, list[ProviderSpec]]:
    normalized = normalize_runtime_device(device)
    available = set(ort.get_available_providers())
    if normalized == "cpu":
        return "cpu", ["CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        adapter = _select_preferred_dml_adapter()
        if adapter is not None:
            print(
                f"[{label}] Using DirectML {describe_gpu_adapter(adapter)}."
            )
            return "dml", [("DmlExecutionProvider", {"device_id": str(adapter.index)}), "CPUExecutionProvider"]
        print(
            f"[{label}] No DirectML adapter with at least "
            f"{format_gib(MIN_GPU_DEDICATED_VRAM_BYTES)} dedicated VRAM was found; "
            "falling back to CPUExecutionProvider."
        )
    else:
        print(f"[{label}] DmlExecutionProvider unavailable; falling back to CPUExecutionProvider.")
    return "cpu", ["CPUExecutionProvider"]


def use_dml(device: str | None) -> bool:
    return normalize_runtime_device(device) != "cpu"
