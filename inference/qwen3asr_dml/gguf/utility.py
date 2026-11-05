from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import os
import json
import numpy as np


def fill_templated_filename(filename: str, output_type: str | None) -> str:
    # Given a file name fill in any type templates e.g. 'some-model-name.{ftype}.gguf'
    ftype_lowercase: str = output_type.lower() if output_type is not None else ""
    ftype_uppercase: str = output_type.upper() if output_type is not None else ""
    return filename.format(ftype_lowercase,
                           outtype=ftype_lowercase, ftype=ftype_lowercase,
                           OUTTYPE=ftype_uppercase, FTYPE=ftype_uppercase)


def model_weight_count_rounded_notation(model_params_count: int, min_digits: int = 2) -> str:
    if model_params_count > 1e12 :
        # Trillions Of Parameters
        scaled_model_params = model_params_count * 1e-12
        scale_suffix = "T"
    elif model_params_count > 1e9 :
        # Billions Of Parameters
        scaled_model_params = model_params_count * 1e-9
        scale_suffix = "B"
    elif model_params_count > 1e6 :
        # Millions Of Parameters
        scaled_model_params = model_params_count * 1e-6
        scale_suffix = "M"
    else:
        # Thousands Of Parameters
        scaled_model_params = model_params_count * 1e-3
        scale_suffix = "K"

    fix = max(min_digits - len(str(round(scaled_model_params)).lstrip('0')), 0)

    return f"{scaled_model_params:.{fix}f}{scale_suffix}"


def size_label(total_params: int, shared_params: int, expert_params: int, expert_count: int) -> str:

    if expert_count > 0:
        pretty_size = model_weight_count_rounded_notation(abs(shared_params) + abs(expert_params), min_digits=2)
        size_class = f"{expert_count}x{pretty_size}"
    else:
        size_class = model_weight_count_rounded_notation(abs(total_params), min_digits=2)

    return size_class


def naming_convention(model_name: str | None, base_name: str | None, finetune_string: str | None, version_string: str | None, size_label: str | None, output_type: str | None, model_type: Literal['vocab', 'LoRA'] | None = None) -> str:
    # Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#gguf-naming-convention

    if base_name is not None:
        name = base_name.strip().replace(' ', '-').replace('/', '-')
    elif model_name is not None:
        name = model_name.strip().replace(' ', '-').replace('/', '-')
    else:
        name = "ggml-model"

    parameters = f"-{size_label}" if size_label is not None else ""

    finetune = f"-{finetune_string.strip().replace(' ', '-')}" if finetune_string is not None else ""

    version = f"-{version_string.strip().replace(' ', '-')}" if version_string is not None else ""

    encoding = f"-{output_type.strip().replace(' ', '-').upper()}" if output_type is not None else ""

    kind = f"-{model_type.strip().replace(' ', '-')}" if model_type is not None else ""

    return f"{name}{parameters}{finetune}{version}{encoding}{kind}"


@dataclass
class LocalTensorRange:
    filename: Path
    offset: int
    size: int


@dataclass
class LocalTensor:
    dtype: str
    shape: tuple[int, ...]
    data_range: LocalTensorRange

    def mmap_bytes(self) -> np.ndarray:
        return np.memmap(self.data_range.filename, mode='c', offset=self.data_range.offset, shape=self.data_range.size)


class SafetensorsLocal:
    """
        Read a safetensors file from the local filesystem.

        Custom parsing gives a bit more control over the memory usage.
        The official safetensors library doesn't expose file ranges.
    """

    tensors: dict[str, LocalTensor]

    def __init__(self, filename: Path):
        with open(filename, "rb") as f:
            metadata_length = int.from_bytes(f.read(8), byteorder='little')
            file_size = os.stat(filename).st_size
            if file_size < 8 + metadata_length:
                raise ValueError(f"Could not read complete metadata. Need {8 + metadata_length} bytes, got {file_size}")

            metadata_str = f.read(metadata_length).decode('utf-8')
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse safetensors metadata as JSON: {e}")

            data_start_offset = f.tell()

            tensors: dict[str, LocalTensor] = {}
            for name, meta in metadata.items():
                if name == "__metadata__":
                    # ignore metadata, it's not a tensor
                    continue

                tensors[name] = LocalTensor(
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    data_range=LocalTensorRange(
                        filename,
                        data_start_offset + meta["data_offsets"][0],
                        meta["data_offsets"][1] - meta["data_offsets"][0],
                    ),
                )

            # order by name (same as default safetensors behavior)
            # ref: https://github.com/huggingface/safetensors/blob/0816a1ae1d6b731cefd67f061d80d1cadd0dd7bb/bindings/python/src/lib.rs#L606
            self.tensors = dict(sorted(tensors.items(), key=lambda t: t[0]))

    def __enter__(self, *args, **kwargs):
        del args, kwargs  # unused
        return self.tensors

    def __exit__(self, *args, **kwargs):
        del args, kwargs  # unused
