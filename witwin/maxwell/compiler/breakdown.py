"""Compile per-node deterministic dielectric-breakdown parameters.

The breakdown descriptor is a discrete per-cell property, not a physically
blended material coefficient, so it is stamped independently of the heavy
subpixel material-averaging path. A node is breakdown-capable where a
breakdown-carrying structure's occupancy reaches the staircase threshold
(``>= 0.5``); overlapping breakdown structures resolve last-writer-wins in the
same priority order the material compiler uses.

The layout stays self-contained: it reuses only the shared geometry-occupancy
sampler and the scene coordinate grids, and produces plain per-node tensors on
the ``(Nx, Ny, Nz)`` material grid that the FDTD runtime maps onto Yee edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .materials import _bulk_structures, _geometry_occupancy


@dataclass(frozen=True)
class CompiledBreakdownLayout:
    """Per-node breakdown parameters on the ``(Nx, Ny, Nz)`` material grid."""

    has_breakdown: bool
    node_mask: torch.Tensor
    critical_field: torch.Tensor
    minimum_duration: torch.Tensor
    post_conductivity: torch.Tensor
    ramp_time_explicit: torch.Tensor
    ramp_steps: torch.Tensor
    material_id: torch.Tensor
    grid_shape: tuple[int, int, int]

    @property
    def capable_count(self) -> int:
        return int(self.node_mask.sum().item())


def _structure_breakdown(structure):
    material = getattr(structure, "material", None)
    if material is None:
        return None
    return getattr(material, "breakdown", None)


def scene_has_breakdown(scene) -> bool:
    """Whether any enabled bulk structure carries a breakdown descriptor."""
    for structure in _bulk_structures(scene):
        if _structure_breakdown(structure) is not None:
            return True
    return False


def compile_breakdown_layout(scene) -> CompiledBreakdownLayout:
    device = scene.device
    shape = (scene.Nx, scene.Ny, scene.Nz)
    node_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    critical_field = torch.zeros(shape, dtype=torch.float32, device=device)
    minimum_duration = torch.zeros(shape, dtype=torch.float32, device=device)
    post_conductivity = torch.zeros(shape, dtype=torch.float32, device=device)
    ramp_time_explicit = torch.zeros(shape, dtype=torch.float32, device=device)
    ramp_steps = torch.zeros(shape, dtype=torch.int64, device=device)
    material_id = torch.full(shape, -1, dtype=torch.int64, device=device)

    structures = list(scene.structures)
    has_breakdown = False
    for structure in _bulk_structures(scene):
        breakdown = _structure_breakdown(structure)
        if breakdown is None:
            continue
        has_breakdown = True
        occupancy = _geometry_occupancy(scene, structure.geometry)
        covered = occupancy >= 0.5
        # Stable material id: the structure's index in the scene structure list.
        try:
            sid = structures.index(structure)
        except ValueError:
            sid = -1
        node_mask = node_mask | covered
        critical_field = torch.where(
            covered,
            torch.full_like(critical_field, float(breakdown.critical_field)),
            critical_field,
        )
        minimum_duration = torch.where(
            covered,
            torch.full_like(minimum_duration, float(breakdown.minimum_duration)),
            minimum_duration,
        )
        post_conductivity = torch.where(
            covered,
            torch.full_like(post_conductivity, float(breakdown.post_breakdown_conductivity)),
            post_conductivity,
        )
        ramp_time_explicit = torch.where(
            covered,
            torch.full_like(
                ramp_time_explicit,
                float(breakdown.ramp_time) if breakdown.ramp_time is not None else 0.0,
            ),
            ramp_time_explicit,
        )
        ramp_steps = torch.where(
            covered,
            torch.full_like(ramp_steps, int(breakdown.default_ramp_steps)),
            ramp_steps,
        )
        material_id = torch.where(
            covered, torch.full_like(material_id, int(sid)), material_id
        )

    return CompiledBreakdownLayout(
        has_breakdown=has_breakdown,
        node_mask=node_mask.contiguous(),
        critical_field=critical_field.contiguous(),
        minimum_duration=minimum_duration.contiguous(),
        post_conductivity=post_conductivity.contiguous(),
        ramp_time_explicit=ramp_time_explicit.contiguous(),
        ramp_steps=ramp_steps.contiguous(),
        material_id=material_id.contiguous(),
        grid_shape=shape,
    )


__all__ = [
    "CompiledBreakdownLayout",
    "compile_breakdown_layout",
    "scene_has_breakdown",
]
