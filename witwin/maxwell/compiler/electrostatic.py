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
class TensorEpsilon:
    """Per-cell symmetric relative-permittivity tensor field (dimensionless).

    Each of the six independent components is a ``(nx, ny, nz)`` tensor on the
    scene device. The stored tensor is symmetric (``xy``/``xz``/``yz`` are the
    single off-diagonal entries, shared by the mirrored pair) and, cell by cell,
    positive-definite. The diagonal components drive the conservative two-point
    face fluxes; the off-diagonal components drive the symmetric cross-derivative
    coupling.
    """

    xx: torch.Tensor
    yy: torch.Tensor
    zz: torch.Tensor
    xy: torch.Tensor
    xz: torch.Tensor
    yz: torch.Tensor

    @property
    def requires_grad(self) -> bool:
        return any(
            t.requires_grad for t in (self.xx, self.yy, self.zz, self.xy, self.xz, self.yz)
        )


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
    # Full symmetric-positive-definite tensor permittivity, or ``None`` for the
    # isotropic scalar path. When present, ``epsilon_r`` carries the diagonal
    # average purely for display/provenance and the operator reads the tensor.
    epsilon_tensor: TensorEpsilon | None = None
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


def _reject_material_regions(scene) -> None:
    """Fail closed on RF ``MaterialRegion`` design regions.

    The electrostatic compiler rasterizes permittivity only from ``Structure``
    bulk media; it does not consume ``Scene.material_regions`` (the RF density
    design-region object). Silently ignoring them would solve with eps_r=1 inside
    a declared region, which is wrong physics, so reject them explicitly instead.
    """
    regions = list(getattr(scene, "material_regions", ()))
    if regions:
        raise NotImplementedError(
            "The electrostatic solver does not support Scene.material_regions "
            f"(found {len(regions)}); MaterialRegion design regions are an RF-path "
            "feature and are not rasterized into the electrostatic permittivity. "
            "Represent static dielectrics with Structure(material=Material(...)) bulk "
            "media instead."
        )


def _material_static_matrix(material) -> np.ndarray:
    """Extract a real, symmetric-positive-definite DC permittivity 3x3 matrix.

    Returns the relative-permittivity tensor as a ``(3, 3)`` float64 array. An
    isotropic scalar material maps to ``eps * I``; a ``DiagonalTensor3`` maps to
    the axis-aligned diagonal; a ``Tensor3x3`` maps to its symmetric rows. The
    diagonal-only cases still round-trip exactly through the isotropic/diagonal
    face-flux operator (the off-diagonal block is exactly zero).

    Rejects dispersive materials (no explicit DC value) and PEC markers (which
    must be modelled as ElectrostaticTerminal conductors, not dielectrics). A
    per-cell tensor sample and a complex permittivity remain unsupported. A
    non-symmetric or non-positive-definite tensor is a physically invalid static
    permittivity and fails with a ValueError.
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
    value = material.evaluate_static().eps_r
    if isinstance(value, DiagonalTensor3):
        matrix = np.diag(np.asarray(value.as_tuple(), dtype=np.float64))
    elif isinstance(value, Tensor3x3):
        matrix = np.asarray(value.rows, dtype=np.float64)
    else:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise NotImplementedError(
                    "Per-cell tensor permittivity is not supported by the electrostatic compiler."
                )
            scalar_t = value.reshape(())
        else:
            scalar_t = torch.as_tensor(value)
        if torch.is_complex(scalar_t):
            if float(scalar_t.imag.abs()) > 0.0:
                raise NotImplementedError(
                    "Complex permittivity is not a valid DC static permittivity for electrostatics."
                )
            scalar_t = scalar_t.real
        matrix = float(scalar_t) * np.eye(3, dtype=np.float64)

    scale = max(float(np.abs(matrix).max()), 1.0)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-9 * scale):
        raise ValueError(
            "Electrostatic static permittivity must be a symmetric tensor; a lossless "
            "anisotropic permittivity is symmetric positive-definite."
        )
    matrix = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(matrix)
    if float(eigvals.min()) <= 0.0:
        raise ValueError(
            "Electrostatic static permittivity must be positive-definite (> 0); got "
            f"eigenvalues {tuple(float(v) for v in eigvals)}."
        )
    return matrix


def _is_isotropic_matrix(matrix: np.ndarray) -> bool:
    """True when the 3x3 tensor is a scalar multiple of the identity."""
    diag = np.diag(matrix)
    off = matrix - np.diag(diag)
    return bool(np.all(off == 0.0)) and bool(diag[0] == diag[1] == diag[2])


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
    """Rasterize the permittivity, returning ``(epsilon_r, epsilon_tensor)``.

    When every bulk material is isotropic the scalar ``epsilon_r`` field is
    returned and ``epsilon_tensor`` is ``None`` (the exact isotropic path). As
    soon as any material is anisotropic (a per-axis diagonal or a full tensor),
    the whole scene is lowered into a six-component :class:`TensorEpsilon`
    field, with isotropic media promoted to ``eps * I``. ``epsilon_r`` then
    carries the diagonal average as a scalar provenance/display field.
    """
    entries = []
    for structure in _sorted_bulk_structures(scene):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        entries.append((structure, _material_static_matrix(material)))

    anisotropic = any(not _is_isotropic_matrix(matrix) for _, matrix in entries)

    if not anisotropic:
        eps = torch.ones(xx.shape, dtype=dtype, device=device)
        for structure, matrix in entries:
            eps_value = float(matrix[0, 0])
            occupancy = _geometry_occupancy(structure.geometry, xx, yy, zz, beta)
            eps = (1.0 - occupancy) * eps + occupancy * eps_value
        return eps, None

    # Tensor path: background vacuum is the identity tensor; every structure
    # blends its (possibly anisotropic) tensor in by soft occupancy.
    ones = torch.ones(xx.shape, dtype=dtype, device=device)
    zeros = torch.zeros(xx.shape, dtype=dtype, device=device)
    comps = {"xx": ones.clone(), "yy": ones.clone(), "zz": ones.clone(),
             "xy": zeros.clone(), "xz": zeros.clone(), "yz": zeros.clone()}
    index = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
             "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    for structure, matrix in entries:
        occupancy = _geometry_occupancy(structure.geometry, xx, yy, zz, beta)
        for key, (i, j) in index.items():
            value = float(matrix[i, j])
            comps[key] = (1.0 - occupancy) * comps[key] + occupancy * value
    tensor = TensorEpsilon(**comps)
    epsilon_r = (comps["xx"] + comps["yy"] + comps["zz"]) / 3.0
    return epsilon_r, tensor


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
    _reject_material_regions(scene)
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

    epsilon_r, epsilon_tensor = _rasterize_epsilon(scene, xx, yy, zz, dtype, device, beta)
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
        epsilon_tensor=epsilon_tensor,
    )
