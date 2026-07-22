"""Public data types for deterministic dynamic dielectric breakdown.

These types are the user-facing container for the field-duration/latching
breakdown feedback model. The physical model is deterministic and uncalibrated:
it flips a cell from ``intact`` to ``conducting`` once the energy-consistent
cell-center field magnitude stays at or above ``critical_field`` for a
contiguous ``minimum_duration``, then ramps the cell conductivity toward
``post_breakdown_conductivity``. It is a comparison / conductive-path model, not
a validated arc or device-failure predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


# Per-cell breakdown state codes shared between the runtime state machine and the
# result accessors. ``recovering``/``failed`` are reserved for later phases; v1
# only produces ``intact`` and ``conducting``.
BREAKDOWN_STATE_INTACT = 0
BREAKDOWN_STATE_RECOVERING = 1
BREAKDOWN_STATE_CONDUCTING = 2
BREAKDOWN_STATE_FAILED = 3

BREAKDOWN_STATE_NAMES = {
    BREAKDOWN_STATE_INTACT: "intact",
    BREAKDOWN_STATE_RECOVERING: "recovering",
    BREAKDOWN_STATE_CONDUCTING: "conducting",
    BREAKDOWN_STATE_FAILED: "failed",
}


@dataclass(frozen=True)
class BreakdownEvent:
    """A single deterministic breakdown trigger.

    ``cell_index`` is the global linear node id on the material grid
    ``(Nx, Ny, Nz)`` (row-major), giving a stable single/multi-GPU ordering key
    together with ``step``. ``position`` is the cell-center coordinate in metres.
    ``field_before`` is the energy-consistent cell-center ``|E|`` at the trigger
    step. ``deposited_energy_at_trigger`` is the local electric field energy
    ``0.5 * eps * |E|^2 * cell_volume`` stored in the cell at the trigger step.
    """

    step: int
    time: float
    cell_index: int
    position: tuple[float, float, float]
    material_id: int
    field_before: float
    state_before: int
    state_after: int
    deposited_energy_at_trigger: float

    @property
    def state_before_name(self) -> str:
        return BREAKDOWN_STATE_NAMES.get(int(self.state_before), "unknown")

    @property
    def state_after_name(self) -> str:
        return BREAKDOWN_STATE_NAMES.get(int(self.state_after), "unknown")


@dataclass(frozen=True)
class BreakdownResultData:
    """Deterministic-breakdown outputs assembled from a completed FDTD run.

    ``events`` is ordered deterministically by ``(step, cell_index)``.
    ``final_state`` is the per-node int8 state mask on the material grid.
    ``dissipated_energy`` is the per-node cumulative breakdown-dissipated energy
    ``integral(sigma_breakdown * |E|^2 dV dt)`` in joules, scattered from the Yee
    edges that carry the conduction term. ``capability_level`` is fixed to the
    uncalibrated deterministic tier.
    """

    events: tuple[BreakdownEvent, ...]
    final_state: torch.Tensor
    dissipated_energy: torch.Tensor
    total_dissipated_energy: float
    grid_shape: tuple[int, int, int]
    capability_level: str = "deterministic-breakdown, uncalibrated"

    @property
    def triggered_count(self) -> int:
        return len(self.events)

    def events_for_material(self, material_id: int) -> tuple[BreakdownEvent, ...]:
        return tuple(
            event for event in self.events if int(event.material_id) == int(material_id)
        )


__all__ = [
    "BreakdownEvent",
    "BreakdownResultData",
    "BREAKDOWN_STATE_INTACT",
    "BREAKDOWN_STATE_RECOVERING",
    "BREAKDOWN_STATE_CONDUCTING",
    "BREAKDOWN_STATE_FAILED",
    "BREAKDOWN_STATE_NAMES",
]
