from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .materials import (
    _blend_material,
    _box_parameter_field,
    _bulk_structures,
    _geometry_occupancy,
    _static_periodic_shift_options,
    _structure_material,
)

# Cells whose tissue fill fraction is below this threshold carry too little mass
# to define a stable per-tissue SAR (dividing a vanishing absorbed power by a
# vanishing mass). They are marked invalid and excluded, never zero-filled.
OCCUPANCY_EPSILON = 1.0e-6

# Sentinel tissue id for cells that carry no SAR (mass-bearing) material.
BACKGROUND_TISSUE_ID = -1


@dataclass(frozen=True)
class CompiledMassDensity:
    """Node-grid tissue mass model sharing the EM compiler occupancy provenance.

    ``rho_cell`` is the occupancy-weighted *effective* mass density in kg/m^3, so
    a cell's tissue mass is ``rho_cell * cell_volume`` and the true tissue density
    of an occupied cell is ``rho_cell / occupancy``. Point SAR consumes it as
    ``SAR = q_cell / rho_cell`` where ``q_cell`` is the colocated absorbed-power
    density (the occupancy cancels, recovering the true tissue SAR). All large
    tensors stay on the scene device; ``rho_cell`` and ``occupancy`` keep their
    autograd graph, while the discrete ``tissue_id`` labels are stop-grad.
    """

    rho_cell: torch.Tensor
    occupancy: torch.Tensor
    tissue_id: torch.Tensor
    cell_volume: torch.Tensor
    tissue_names: Mapping[int, str]
    rho_cell_unit: str = "kg/m^3"
    cell_volume_unit: str = "m^3"
    collocation: str = "material node grid (cell centers); effective density is occupancy-weighted"

    @property
    def device(self) -> torch.device:
        return self.rho_cell.device

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.rho_cell.shape)


def _cell_volume_field(scene) -> torch.Tensor:
    """Dual-cell (node control) volume per material cell center on the node grid."""
    device = scene.device
    dx = torch.as_tensor(scene.dx_dual64, device=device, dtype=torch.float32)
    dy = torch.as_tensor(scene.dy_dual64, device=device, dtype=torch.float32)
    dz = torch.as_tensor(scene.dz_dual64, device=device, dtype=torch.float32)
    return dx[:, None, None] * dy[None, :, None] * dz[None, None, :]


def _resolve_density_value(scene, structure, mass_density):
    if torch.is_tensor(mass_density):
        return _box_parameter_field(
            scene,
            structure.geometry,
            mass_density,
            name="Material.mass_density grid",
        )
    return float(mass_density)


def compile_mass_density(scene) -> CompiledMassDensity:
    """Rasterize tissue mass density onto material cell centers (the node grid).

    Materials without ``mass_density`` are excluded from the mass model (they
    remain valid EM media). Overlapping tissues resolve by the same priority
    order the EM material compiler uses; the effective density and tissue fill
    fraction blend through the identical soft-occupancy path
    (``_geometry_occupancy``), so the SAR mass model and the EM loss model never
    disagree about where tissue is.
    """
    if not all(hasattr(scene, name) for name in ("x_half", "y_half", "z_half")):
        from ..scene import prepare_scene

        scene = prepare_scene(scene)

    shape = (scene.Nx, scene.Ny, scene.Nz)
    device = scene.device
    rho_cell = torch.zeros(shape, device=device, dtype=torch.float32)
    occupancy = torch.zeros(shape, device=device, dtype=torch.float32)
    tissue_id = torch.full(shape, BACKGROUND_TISSUE_ID, device=device, dtype=torch.int64)

    material_ids: dict[int, int] = {}
    tissue_names: dict[int, str] = {}
    for structure in _bulk_structures(scene):
        material = _structure_material(structure)
        if material is None or getattr(material, "mass_density", None) is None:
            continue
        key = id(material)
        if key not in material_ids:
            new_id = len(material_ids)
            material_ids[key] = new_id
            tissue_names[new_id] = str(getattr(material, "name", None) or f"tissue_{new_id}")
        current_id = material_ids[key]

        occ = _geometry_occupancy(
            scene,
            structure.geometry,
            periodic_shift_options=_static_periodic_shift_options(scene, structure.geometry),
        )
        density_value = _resolve_density_value(scene, structure, material.mass_density)
        # Priority overwrite: a later (higher-priority) tissue displaces earlier
        # tissue where it is present, exactly like the EM eps/sigma blend.
        rho_cell = _blend_material(rho_cell, occ, value=density_value)
        occupancy = _blend_material(occupancy, occ, value=1.0)
        present = occ > OCCUPANCY_EPSILON
        tissue_id = torch.where(present, torch.full_like(tissue_id, current_id), tissue_id)

    return CompiledMassDensity(
        rho_cell=rho_cell,
        occupancy=occupancy.clamp(0.0, 1.0),
        tissue_id=tissue_id,
        cell_volume=_cell_volume_field(scene),
        tissue_names=MappingProxyType(dict(tissue_names)),
    )


__all__ = [
    "CompiledMassDensity",
    "compile_mass_density",
    "OCCUPANCY_EPSILON",
    "BACKGROUND_TISSUE_ID",
]
