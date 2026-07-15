from __future__ import annotations

from typing import Any

import torch


def move_tensors_to_device(value: Any, device: torch.device) -> Any:
    """Move a nested, small result payload directly between CUDA devices."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_device(item, device) for item in value)
    return value


def electric_field_output_bytes(
    global_shape: tuple[int, int, int],
    *,
    frequency_count: int = 1,
    complex_output: bool = False,
) -> int:
    nx, ny, nz = (int(value) for value in global_shape)
    elements = (
        (nx - 1) * ny * nz
        + nx * (ny - 1) * nz
        + nx * ny * (nz - 1)
    )
    element_size = 8 if complex_output else 4
    return elements * max(int(frequency_count), 1) * element_size


def effective_cuda_free_bytes(device: torch.device) -> int:
    with torch.cuda.device(device):
        driver_free, _total = torch.cuda.mem_get_info(device)
        reusable_cache = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return int(driver_free + max(int(reusable_cache), 0))


__all__ = [
    "effective_cuda_free_bytes",
    "electric_field_output_bytes",
    "move_tensors_to_device",
]
