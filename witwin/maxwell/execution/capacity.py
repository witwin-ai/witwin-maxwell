from __future__ import annotations

import torch

from ..fdtd.distributed.capacity import local_dft_working_set_bytes
from ..fdtd.distributed.output import electric_field_output_bytes
from ..scene import Scene, prepare_scene

# Conservative multiplier over the three staggered E-field element counts that
# approximates one FDTD shard's persistent device state: the six Yee field
# components plus the per-component update coefficients, material inverses and
# CPML/auxiliary buffers all scale with the same cell count. This is a placement
# preflight heuristic (fail before OOM), not an exact allocator accounting.
_FIELD_STATE_MULTIPLIER = 8


def _uniform_node_counts(scene: Scene) -> tuple[int, int, int] | None:
    grid = scene.grid
    if grid.is_custom or grid.is_auto:
        return None
    bounds = scene.domain.bounds
    spacing = grid.spacing
    counts = []
    for (low, high), delta in zip(bounds, spacing):
        extent = float(high) - float(low)
        if delta is None or float(delta) <= 0.0 or extent <= 0.0:
            return None
        counts.append(int(round(extent / float(delta))) + 1)
    return tuple(counts)


def _node_counts(scene: Scene) -> tuple[int, int, int] | None:
    counts = _uniform_node_counts(scene)
    if counts is not None:
        return counts
    try:
        prepared = prepare_scene(scene)
    except Exception:
        return None
    return (int(prepared.Nx), int(prepared.Ny), int(prepared.Nz))


def estimate_scene_footprint_bytes(
    scene: Scene,
    *,
    frequencies=None,
    full_field_dft: bool = False,
) -> int | None:
    """Estimate one Simulation's resident device footprint for placement.

    Returns ``None`` when the grid shape cannot be resolved cheaply, in which
    case the scheduler falls back to order-based placement for that task.
    """

    counts = _node_counts(scene)
    if counts is None:
        return None
    field_bytes = electric_field_output_bytes(counts, frequency_count=1, complex_output=False)
    resident = _FIELD_STATE_MULTIPLIER * field_bytes
    dft_bytes = local_dft_working_set_bytes(
        counts,
        dft_frequency=tuple(frequencies) if frequencies else None,
        full_field_dft=full_field_dft,
    )
    return int(resident + dft_bytes)


def estimate_simulation_footprint_bytes(simulation) -> int | None:
    """Estimate the resident footprint of a prepared ``Simulation`` task."""

    scene = getattr(simulation, "scene", None)
    if not isinstance(scene, Scene):
        return None
    config = getattr(simulation, "config", None)
    full_field_dft = bool(getattr(config, "full_field_dft", False))
    return estimate_scene_footprint_bytes(
        scene,
        frequencies=getattr(simulation, "frequencies", None),
        full_field_dft=full_field_dft,
    )


__all__ = [
    "estimate_scene_footprint_bytes",
    "estimate_simulation_footprint_bytes",
]
