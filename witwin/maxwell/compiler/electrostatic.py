from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from witwin.core.material import VACUUM_PERMITTIVITY

from ..electrostatic.api import ChargeDensity, ElectrostaticBoundarySpec, ElectrostaticTerminal
from ..media import DiagonalTensor3, Tensor3x3
from ..scene import prepare_scene

_AXES = ("x", "y", "z")


@dataclass
class TerminalConstraint:
    """A compiled equipotential conductor: its cell mask and its constraint."""

    name: str
    mask: torch.Tensor  # bool (nx, ny, nz)
    potential: float | None
    charge: float | None
    grounded: bool

    @property
    def is_floating(self) -> bool:
        return self.potential is None and self.charge is not None


@dataclass
class CompiledElectrostatics:
    """Solver-ready electrostatic block on the cell-centred finite-volume grid.

    All geometry tensors are float64 on the scene device. ``epsilon_r`` is the
    dimensionless relative permittivity at cell centres; ``free_charge`` is the
    total free charge per cell in Coulombs (density integrated over the cell
    volume). ``eps0`` folds the vacuum permittivity into the face conductances.
    """

    device: str
    dtype: torch.dtype
    shape: tuple[int, int, int]
    # Cell-centre coordinates (nx,), (ny,), (nz,)
    xc: torch.Tensor
    yc: torch.Tensor
    zc: torch.Tensor
    # Cell widths / primal spacings (nx,), (ny,), (nz,)
    hx: torch.Tensor
    hy: torch.Tensor
    hz: torch.Tensor
    # Cell-centre spacings between neighbours (nx-1,), (ny-1,), (nz-1,)
    dxc: torch.Tensor
    dyc: torch.Tensor
    dzc: torch.Tensor
    cell_volume: torch.Tensor  # (nx, ny, nz)
    epsilon_r: torch.Tensor  # (nx, ny, nz)
    free_charge: torch.Tensor  # (nx, ny, nz) Coulombs
    terminals: list[TerminalConstraint]
    boundary: ElectrostaticBoundarySpec
    eps0: float = VACUUM_PERMITTIVITY


def _require_electrostatic_boundary(scene) -> None:
    """Electrostatics owns its own BCs; reject grid-extending Scene boundaries."""
    boundary = scene.boundary
    if boundary.kind != "none":
        raise NotImplementedError(
            "The electrostatic solver uses ElectrostaticBoundarySpec for its boundary "
            "conditions and requires Scene.boundary = BoundarySpec.none() so the grid is "
            f"not extended by PML/periodic padding (got Scene boundary kind {boundary.kind!r})."
        )


def _static_epsilon_scalar(material, terminal_name_hint: str | None = None):
    """Extract a real, positive DC relative permittivity from a Material.

    Rejects dispersive materials (no explicit DC value), full/diagonal tensor
    permittivity (Phase 4 tensor-eps, out of scope), and PEC markers (which must
    be modelled as ElectrostaticTerminal conductors, not dielectrics).
    """
    if bool(getattr(material, "is_pec", False)):
        raise NotImplementedError(
            "A PEC-material structure has no dielectric permittivity for the electrostatic "
            "solver; represent conductors with ElectrostaticTerminal (equipotential) instead."
        )
    if bool(getattr(material, "is_dispersive", False)):
        raise NotImplementedError(
            "Dispersive materials do not define a zero-frequency permittivity; the "
            "electrostatic solver refuses to guess a DC limit. Supply a non-dispersive "
            "Material with an explicit real static permittivity."
        )
    sample = material.evaluate_static()
    value = sample.eps_r
    if isinstance(value, (DiagonalTensor3, Tensor3x3)):
        raise NotImplementedError(
            "Anisotropic (tensor) permittivity is not supported by the scalar electrostatic "
            "operator (Phase 4). Use an isotropic Material."
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise NotImplementedError(
                "Per-cell tensor permittivity is not supported by the electrostatic compiler."
            )
        scalar = value
    else:
        scalar = value
    eps = torch.as_tensor(scalar)
    if torch.is_complex(eps):
        if float(eps.imag.abs().max()) > 0.0:
            raise NotImplementedError(
                "Complex permittivity is not a valid DC static permittivity for electrostatics."
            )
        eps = eps.real
    return eps


def _sorted_bulk_structures(scene):
    indexed = [
        (index, structure)
        for index, structure in enumerate(scene.structures)
        if getattr(structure, "enabled", True)
    ]
    indexed.sort(key=lambda item: (int(getattr(item[1], "priority", 0)), item[0]))
    return [structure for _, structure in indexed]


def _cell_geometry(prepared, dtype, device):
    def _t(values):
        return torch.as_tensor(np.asarray(values, dtype=np.float64), device=device, dtype=dtype)

    xc = _t(prepared.x_half64)
    yc = _t(prepared.y_half64)
    zc = _t(prepared.z_half64)
    hx = _t(prepared.dx_primal64)
    hy = _t(prepared.dy_primal64)
    hz = _t(prepared.dz_primal64)
    dxc = xc[1:] - xc[:-1]
    dyc = yc[1:] - yc[:-1]
    dzc = zc[1:] - zc[:-1]
    cell_volume = hx[:, None, None] * hy[None, :, None] * hz[None, None, :]
    return xc, yc, zc, hx, hy, hz, dxc, dyc, dzc, cell_volume


def _cell_center_meshgrid(xc, yc, zc):
    return torch.meshgrid(xc, yc, zc, indexing="ij")


def _geometry_occupancy(geometry, xx, yy, zz, beta):
    """Soft occupancy of a geometry at query points (differentiable in geometry)."""
    return geometry.to_mask(
        xx.to(torch.float32),
        yy.to(torch.float32),
        zz.to(torch.float32),
        offset=0.0,
        beta=beta,
    ).to(xx.dtype)


def _rasterize_epsilon(scene, xx, yy, zz, dtype, device, beta):
    eps = torch.ones(xx.shape, dtype=dtype, device=device)
    for structure in _sorted_bulk_structures(scene):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        eps_value = _static_epsilon_scalar(material).to(device=device, dtype=dtype)
        if float(eps_value) <= 0.0:
            raise ValueError(
                "Electrostatic static permittivity must be > 0; "
                f"got eps_r={float(eps_value)}."
            )
        occupancy = _geometry_occupancy(structure.geometry, xx, yy, zz, beta)
        eps = (1.0 - occupancy) * eps + occupancy * eps_value
    return eps


def _rasterize_free_charge(scene, xx, yy, zz, cell_volume, dtype, device, beta):
    charge = torch.zeros(xx.shape, dtype=dtype, device=device)
    charge_sources = list(getattr(scene, "charge_densities", ()))
    for source in charge_sources:
        if not isinstance(source, ChargeDensity):
            raise TypeError("Scene.charge_densities must contain ChargeDensity entries.")
        occupancy = _geometry_occupancy(source.geometry, xx, yy, zz, beta)
        density = torch.as_tensor(source.density, dtype=dtype, device=device)
        charge = charge + occupancy * density
    # density (C/m^3) integrated over the cell volume gives Coulombs per cell.
    return charge * cell_volume


def _terminal_mask(terminal, xx, yy, zz):
    signed_distance = terminal.geometry.signed_distance(
        xx.to(torch.float32), yy.to(torch.float32), zz.to(torch.float32)
    )
    return signed_distance <= 0.0


def compile_electrostatics(
    scene,
    boundary: ElectrostaticBoundarySpec | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> CompiledElectrostatics:
    prepared = prepare_scene(scene)
    _require_electrostatic_boundary(prepared)
    if boundary is None:
        boundary = ElectrostaticBoundarySpec.grounded_box()
    if not isinstance(boundary, ElectrostaticBoundarySpec):
        raise TypeError("boundary must be an ElectrostaticBoundarySpec.")

    device = prepared.device
    (xc, yc, zc, hx, hy, hz, dxc, dyc, dzc, cell_volume) = _cell_geometry(prepared, dtype, device)
    shape = (int(xc.numel()), int(yc.numel()), int(zc.numel()))
    if min(shape) < 1:
        raise ValueError("Electrostatic grid must have at least one cell per axis.")

    xx, yy, zz = _cell_center_meshgrid(xc, yc, zc)
    min_spacing = float(min(float(hx.min()), float(hy.min()), float(hz.min())))
    beta = 0.05 * min_spacing

    epsilon_r = _rasterize_epsilon(scene, xx, yy, zz, dtype, device, beta)
    free_charge = _rasterize_free_charge(scene, xx, yy, zz, cell_volume, dtype, device, beta)

    terminals: list[TerminalConstraint] = []
    claimed = torch.zeros(shape, dtype=torch.bool, device=device)
    declared = list(getattr(scene, "electrostatic_terminals", ()))
    for terminal in declared:
        if not isinstance(terminal, ElectrostaticTerminal):
            raise TypeError("Scene.electrostatic_terminals must contain ElectrostaticTerminal entries.")
        mask = _terminal_mask(terminal, xx, yy, zz)
        if not bool(mask.any()):
            raise ValueError(
                f"Electrostatic terminal {terminal.name!r} resolves to zero grid cells; the "
                "conductor is thinner than one cell and is swallowed by the mesh. Refine the "
                "grid or enlarge the conductor."
            )
        overlap = mask & claimed
        if bool(overlap.any()):
            raise ValueError(
                f"Electrostatic terminal {terminal.name!r} overlaps an earlier terminal on "
                f"{int(overlap.sum())} cells; equipotential conductors must be disjoint."
            )
        claimed = claimed | mask
        terminals.append(
            TerminalConstraint(
                name=terminal.name,
                mask=mask,
                potential=terminal.potential,
                charge=terminal.charge,
                grounded=terminal.grounded,
            )
        )

    return CompiledElectrostatics(
        device=device,
        dtype=dtype,
        shape=shape,
        xc=xc,
        yc=yc,
        zc=zc,
        hx=hx,
        hy=hy,
        hz=hz,
        dxc=dxc,
        dyc=dyc,
        dzc=dzc,
        cell_volume=cell_volume,
        epsilon_r=epsilon_r,
        free_charge=free_charge,
        terminals=terminals,
        boundary=boundary,
    )
