from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import torch

from ...compiler.ports import CompiledPortGeometry
from ...network import PortData


class _PortPlanWithLocalGeometry(Protocol):
    """Per-port ownership plan carrying compiled geometry and shard-local indices."""

    geometry: CompiledPortGeometry
    local_voltage_indices: tuple[tuple[int, int, int], ...]


def local_port_geometry(
    plan: _PortPlanWithLocalGeometry,
    *,
    device: torch.device,
) -> CompiledPortGeometry:
    """Rebuild a port's compiled geometry with shard-local voltage indices on ``device``."""

    return replace(
        plan.geometry,
        voltage_indices=torch.as_tensor(
            plan.local_voltage_indices,
            device=device,
            dtype=torch.int64,
        ),
        voltage_weights=plan.geometry.voltage_weights.to(device=device),
    )


def move_nested(value, device: torch.device):
    """Move every tensor in a nested dict/tuple/list container to ``device``."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device) for item in value]
    return value


def move_port_data(data: PortData, device: torch.device, *, metadata) -> PortData:
    """Move one finalized port record to ``device`` with replacement metadata."""

    return replace(
        data,
        frequencies=data.frequencies.to(device=device),
        voltage=data.voltage.to(device=device),
        current=data.current.to(device=device),
        z0=move_nested(data.z0, device),
        available_power=move_nested(data.available_power, device),
        beta=move_nested(data.beta, device),
        characteristic_impedance=move_nested(data.characteristic_impedance, device),
        tracking_confidence=move_nested(data.tracking_confidence, device),
        metadata=move_nested(metadata, device),
    )
