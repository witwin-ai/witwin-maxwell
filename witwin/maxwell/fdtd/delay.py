from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass
class PreparedBidirectionalDelay:
    """Fixed-shape two-way per-port delay state for CUDA-graph replay."""

    delay_seconds: tuple[float, ...]
    dt: float
    ring_length: int
    integer_steps: torch.Tensor
    fractional_coefficients: torch.Tensor
    has_fractional: torch.Tensor
    zero_delay: torch.Tensor
    forward_ring: torch.Tensor
    reverse_ring: torch.Tensor
    forward_previous_input: torch.Tensor
    forward_previous_output: torch.Tensor
    reverse_previous_input: torch.Tensor
    reverse_previous_output: torch.Tensor
    forward_integer_sample: torch.Tensor
    forward_fractional_sample: torch.Tensor
    forward_temp: torch.Tensor
    reverse_integer_sample: torch.Tensor
    reverse_fractional_sample: torch.Tensor
    reverse_temp: torch.Tensor
    port_offsets: torch.Tensor
    read_positions: torch.Tensor
    read_indices: torch.Tensor
    write_indices: torch.Tensor
    cursor: torch.Tensor

    @property
    def port_count(self) -> int:
        return len(self.delay_seconds)


def prepare_bidirectional_delay(
    delay_seconds: Sequence[float],
    *,
    dt: float,
    max_delay_steps: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> PreparedBidirectionalDelay:
    """Allocate bounded buffers for lossless one-way delays in both directions."""

    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar.")
    if not isinstance(max_delay_steps, int) or isinstance(max_delay_steps, bool) or max_delay_steps < 1:
        raise ValueError("max_delay_steps must be a positive integer.")
    if not dtype.is_floating_point:
        raise TypeError("delay buffers require a real floating-point dtype.")
    resolved_device = torch.device(device)
    values = tuple(float(value) for value in delay_seconds)
    if not values:
        raise ValueError("delay_seconds must contain at least one port delay.")
    delay = torch.tensor(values, device=resolved_device, dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(delay))) or not bool(torch.all(delay >= 0.0)):
        raise ValueError("delay_seconds must contain finite non-negative values.")
    if bool(torch.any((delay > 0.0) & (delay < float(dt)))):
        raise ValueError("Every nonzero explicit delay must be at least one FDTD step.")
    samples = delay / float(dt)
    integer_steps = torch.floor(samples).to(dtype=torch.int64)
    fractional = samples - integer_steps.to(dtype=samples.dtype)
    snap = 256.0 * torch.finfo(samples.dtype).eps
    carry = fractional >= 1.0 - snap
    integer_steps.add_(carry.to(dtype=torch.int64))
    fractional = torch.where(
        carry | (fractional <= snap), torch.zeros_like(fractional), fractional
    )
    required_steps = torch.ceil(samples).to(dtype=torch.int64)
    buffer_steps = int(torch.max(required_steps).item())
    if buffer_steps > max_delay_steps:
        raise ValueError(
            f"Delay realization requires {buffer_steps} steps, exceeding max_delay_steps={max_delay_steps}."
        )
    coefficient = torch.where(
        fractional > 0.0,
        (1.0 - fractional) / (1.0 + fractional),
        torch.zeros_like(fractional),
    ).to(device=resolved_device, dtype=dtype)
    has_fractional = (fractional > 0.0).to(device=resolved_device)
    zero_delay = (delay == 0.0).to(device=resolved_device)
    integer_steps = integer_steps.to(device=resolved_device)
    ring_length = max(1, buffer_steps)
    port_count = len(values)
    ring_shape = (port_count, ring_length)
    def vector() -> torch.Tensor:
        return torch.zeros((port_count,), device=resolved_device, dtype=dtype)

    return PreparedBidirectionalDelay(
        delay_seconds=values,
        dt=float(dt),
        ring_length=ring_length,
        integer_steps=integer_steps,
        fractional_coefficients=coefficient,
        has_fractional=has_fractional,
        zero_delay=zero_delay,
        forward_ring=torch.zeros(ring_shape, device=resolved_device, dtype=dtype),
        reverse_ring=torch.zeros(ring_shape, device=resolved_device, dtype=dtype),
        forward_previous_input=vector(),
        forward_previous_output=vector(),
        reverse_previous_input=vector(),
        reverse_previous_output=vector(),
        forward_integer_sample=vector(),
        forward_fractional_sample=vector(),
        forward_temp=vector(),
        reverse_integer_sample=vector(),
        reverse_fractional_sample=vector(),
        reverse_temp=vector(),
        port_offsets=torch.arange(
            0,
            port_count * ring_length,
            ring_length,
            device=resolved_device,
            dtype=torch.int64,
        ),
        read_positions=torch.empty((port_count,), device=resolved_device, dtype=torch.int64),
        read_indices=torch.empty((port_count,), device=resolved_device, dtype=torch.int64),
        write_indices=torch.empty((port_count,), device=resolved_device, dtype=torch.int64),
        cursor=torch.zeros((), device=resolved_device, dtype=torch.int64),
    )


def _prepare_indices(runtime: PreparedBidirectionalDelay) -> None:
    torch.sub(runtime.cursor, runtime.integer_steps, out=runtime.read_positions)
    torch.remainder(runtime.read_positions, runtime.ring_length, out=runtime.read_positions)
    torch.add(runtime.port_offsets, runtime.read_positions, out=runtime.read_indices)
    torch.add(runtime.port_offsets, runtime.cursor, out=runtime.write_indices)


def _read_direction(
    runtime: PreparedBidirectionalDelay,
    output_value: torch.Tensor,
    *,
    ring: torch.Tensor,
    previous_input: torch.Tensor,
    previous_output: torch.Tensor,
    integer_sample: torch.Tensor,
    fractional_sample: torch.Tensor,
    temp: torch.Tensor,
) -> None:
    torch.index_select(
        ring.view(-1),
        0,
        runtime.read_indices,
        out=integer_sample,
    )
    torch.mul(
        runtime.fractional_coefficients,
        integer_sample,
        out=fractional_sample,
    )
    fractional_sample.add_(previous_input)
    torch.mul(
        runtime.fractional_coefficients,
        previous_output,
        out=temp,
    )
    fractional_sample.sub_(temp)
    torch.where(runtime.has_fractional, fractional_sample, integer_sample, out=output_value)
    output_value.masked_fill_(runtime.zero_delay, 0.0)
    previous_input.copy_(integer_sample)
    previous_output.copy_(fractional_sample)


def read_bidirectional_delay(
    runtime: PreparedBidirectionalDelay,
    forward_output: torch.Tensor,
    reverse_output: torch.Tensor,
) -> None:
    """Read delayed waves before the current samples are known.

    Zero-delay entries are set to zero for the caller to replace with its
    implicit same-step solution.  A matching :func:`write_bidirectional_delay`
    call must follow exactly once.
    """

    expected = (runtime.port_count,)
    if forward_output.shape != expected or reverse_output.shape != expected:
        raise ValueError(f"bidirectional delay values must have shape {expected}.")
    if any(
        value.device != runtime.forward_ring.device or value.dtype != runtime.forward_ring.dtype
        for value in (forward_output, reverse_output)
    ):
        raise ValueError("bidirectional delay values must share the prepared device and dtype.")
    _prepare_indices(runtime)
    _read_direction(
        runtime,
        forward_output,
        ring=runtime.forward_ring,
        previous_input=runtime.forward_previous_input,
        previous_output=runtime.forward_previous_output,
        integer_sample=runtime.forward_integer_sample,
        fractional_sample=runtime.forward_fractional_sample,
        temp=runtime.forward_temp,
    )
    _read_direction(
        runtime,
        reverse_output,
        ring=runtime.reverse_ring,
        previous_input=runtime.reverse_previous_input,
        previous_output=runtime.reverse_previous_output,
        integer_sample=runtime.reverse_integer_sample,
        fractional_sample=runtime.reverse_fractional_sample,
        temp=runtime.reverse_temp,
    )


def write_bidirectional_delay(
    runtime: PreparedBidirectionalDelay,
    forward_input: torch.Tensor,
    reverse_input: torch.Tensor,
) -> None:
    """Write current waves and advance the shared bounded ring cursor."""

    expected = (runtime.port_count,)
    if forward_input.shape != expected or reverse_input.shape != expected:
        raise ValueError(f"bidirectional delay values must have shape {expected}.")
    if any(
        value.device != runtime.forward_ring.device or value.dtype != runtime.forward_ring.dtype
        for value in (forward_input, reverse_input)
    ):
        raise ValueError("bidirectional delay values must share the prepared device and dtype.")
    runtime.forward_ring.view(-1).index_copy_(0, runtime.write_indices, forward_input)
    runtime.reverse_ring.view(-1).index_copy_(0, runtime.write_indices, reverse_input)
    runtime.cursor.add_(1)
    runtime.cursor.remainder_(runtime.ring_length)


def advance_bidirectional_delay(
    runtime: PreparedBidirectionalDelay,
    forward_input: torch.Tensor,
    reverse_input: torch.Tensor,
    forward_output: torch.Tensor,
    reverse_output: torch.Tensor,
) -> None:
    """Advance both directions once without allocation or host synchronization."""

    expected = (runtime.port_count,)
    tensors = (forward_input, reverse_input, forward_output, reverse_output)
    if any(value.shape != expected for value in tensors):
        raise ValueError(f"bidirectional delay values must have shape {expected}.")
    if any(
        value.device != runtime.forward_ring.device or value.dtype != runtime.forward_ring.dtype
        for value in tensors
    ):
        raise ValueError("bidirectional delay values must share the prepared device and dtype.")

    read_bidirectional_delay(runtime, forward_output, reverse_output)
    torch.where(runtime.zero_delay, forward_input, forward_output, out=forward_output)
    torch.where(runtime.zero_delay, reverse_input, reverse_output, out=reverse_output)
    write_bidirectional_delay(runtime, forward_input, reverse_input)


__all__ = [
    "PreparedBidirectionalDelay",
    "advance_bidirectional_delay",
    "prepare_bidirectional_delay",
    "read_bidirectional_delay",
    "write_bidirectional_delay",
]
