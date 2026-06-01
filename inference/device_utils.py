from __future__ import annotations

import onnxruntime as ort


RUNTIME_DEVICE_CHOICES = ("dml", "cpu", "cuda")
VISIBLE_RUNTIME_DEVICE_CHOICES = ("dml", "cpu")

_DEVICE_ALIASES = {
    "": "dml",
    "cuda": "dml",
    "directml": "dml",
    "dml": "dml",
    "gpu": "dml",
    "cpu": "cpu",
}


def normalize_runtime_device(device: str | None, default: str = "dml") -> str:
    value = str(device or default).strip().lower()
    if not value:
        value = default
    return _DEVICE_ALIASES.get(value, value)


def resolve_onnx_providers(device: str | None, *, label: str = "ONNX") -> tuple[str, list[str]]:
    normalized = normalize_runtime_device(device)
    available = set(ort.get_available_providers())
    if normalized == "cpu":
        return "cpu", ["CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return "dml", ["DmlExecutionProvider", "CPUExecutionProvider"]
    print(f"[{label}] DmlExecutionProvider unavailable; falling back to CPUExecutionProvider.")
    return "cpu", ["CPUExecutionProvider"]


def use_dml(device: str | None) -> bool:
    return normalize_runtime_device(device) != "cpu"
