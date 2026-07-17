from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


@dataclass(frozen=True)
class ArrayRunData:
    """In-memory full-wave column data retained for array postprocessing."""

    manifest: Any
    column_results: tuple[tuple[Any, ...], ...]
    incident: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.incident, torch.Tensor) or not self.incident.is_complex():
            raise TypeError("incident must be a complex torch.Tensor with shape [F, N].")
        if self.incident.ndim != 2 or 0 in self.incident.shape:
            raise ValueError("incident must have non-empty shape [F, N].")
        if len(self.column_results) != self.incident.shape[1]:
            raise ValueError("column_results must contain one entry per incident-wave column.")
        if any(not column for column in self.column_results):
            raise ValueError("Every array basis column must retain at least one FDTD Result.")


def compact_array_column_result(result, *, prepared_scene):
    """Drop solver state and retain only declared closed-surface payloads."""

    from .monitors import ClosedSurfaceMonitor
    from .result import Result

    constants = SimpleNamespace(
        c=result.solver.c,
        eps0=result.solver.eps0,
        mu0=result.solver.mu0,
    )
    retained_monitor_names = set()
    for monitor in result.scene.monitors:
        if isinstance(monitor, ClosedSurfaceMonitor):
            retained_monitor_names.add(monitor.name)
            retained_monitor_names.update(face.name for face in monitor.faces)
    retained_monitors = {
        name: payload
        for name, payload in result.monitors.items()
        if name in retained_monitor_names
    }
    return Result(
        method=result.method,
        scene=result.scene,
        prepared_scene=prepared_scene,
        frequency=result.frequency,
        frequencies=result.frequencies,
        solver=constants,
        monitors=retained_monitors,
        ports={},
        metadata=result._metadata,
        solver_stats=result.solver_stats,
    )


__all__ = ["ArrayRunData", "compact_array_column_result"]
