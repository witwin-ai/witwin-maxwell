from __future__ import annotations

import torch

from .output import effective_cuda_free_bytes, electric_field_output_bytes


def _frequency_count(dft_frequency) -> int:
    if isinstance(dft_frequency, (tuple, list)):
        return max(len(dft_frequency), 1)
    return 1


def local_dft_working_set_bytes(
    local_shape: tuple[int, int, int],
    *,
    dft_frequency,
    full_field_dft: bool,
) -> int:
    """Budget real/imag accumulators plus the postprocessed complex local fields."""

    if dft_frequency is None or not full_field_dft:
        return 0
    complex_field_bytes = electric_field_output_bytes(
        local_shape,
        frequency_count=_frequency_count(dft_frequency),
        complex_output=True,
    )
    return 2 * complex_field_bytes


def _require_capacity(device: torch.device, required: int, *, operation: str) -> dict[str, int]:
    available = effective_cuda_free_bytes(device)
    if required * 1.05 > available:
        raise MemoryError(
            f"{operation} requires {required} bytes on {device}, but only {available} "
            "effective bytes are free. Disable full-field DFT/gathering or choose a device "
            "with more free memory."
        )
    return {"required_bytes": int(required), "available_bytes": int(available)}


def require_local_dft_capacity(
    device: torch.device,
    local_shape: tuple[int, int, int],
    *,
    dft_frequency,
    full_field_dft: bool,
) -> dict[str, int]:
    pending = local_dft_working_set_bytes(
        local_shape,
        dft_frequency=dft_frequency,
        full_field_dft=full_field_dft,
    )
    result = _require_capacity(device, pending, operation="Local multi-GPU DFT working set")
    result["pending_local_dft_bytes"] = int(pending)
    return result


def require_gather_capacity(
    device: torch.device,
    global_shape: tuple[int, int, int],
    *,
    dft_frequency,
    full_field_dft: bool,
    pending_local_dft_bytes: int = 0,
) -> dict[str, int]:
    """Fail before stepping when the result device's combined working set cannot fit."""

    complex_output = bool(dft_frequency is not None and full_field_dft)
    output_bytes = electric_field_output_bytes(
        global_shape,
        frequency_count=_frequency_count(dft_frequency),
        complex_output=complex_output,
    )
    required = output_bytes + max(int(pending_local_dft_bytes), 0)
    result = _require_capacity(device, required, operation="Explicit multi-GPU field gathering")
    result["output_bytes"] = int(output_bytes)
    result["pending_local_dft_bytes"] = int(pending_local_dft_bytes)
    return result


__all__ = [
    "local_dft_working_set_bytes",
    "require_gather_capacity",
    "require_local_dft_capacity",
]
