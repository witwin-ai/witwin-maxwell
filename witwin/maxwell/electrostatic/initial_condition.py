from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Documented default for the discrete-Gauss consistency gate. The residual is a
# dimensionless relative quantity (see ``_gauss_residual``); ``1e-3`` accepts the
# interpolation/interface discretization error of a well-resolved charge-free
# pre-bias while still rejecting a grossly inconsistent mapping. Interface-heavy
# scenes (the electrostatic harmonic-face permittivity differs from the FDTD
# edge-averaged permittivity at material boundaries) may need a looser tolerance,
# passed explicitly; never silently accepted.
DEFAULT_GAUSS_TOLERANCE = 1.0e-3


def _cells_to_nodes(values: torch.Tensor, axis: int, cell_coords: torch.Tensor, node_coords: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a cell-centred tensor onto primary-node samples.

    ``values`` is sampled at ``n`` cell centres along ``axis``; the result is
    sampled at the ``n + 1`` primary nodes bracketing them. Interior nodes use the
    two nearest cell centres with distance-weighted linear interpolation. The two
    boundary nodes lie outside the cell-centre span and are linearly *extrapolated*
    from the two nearest cell centres. Both are exact for a field that varies
    linearly (hence exact for a uniform gradient, and for a linear potential the
    boundary node recovers the electrode potential exactly); a constant field also
    stays constant, so the discrete curl-free property is preserved. A single cell
    on the axis falls back to a constant. All arithmetic stays on the input device.
    """
    values = values.movedim(axis, 0)
    n = values.shape[0]
    if node_coords.numel() != n + 1:
        raise ValueError(
            f"cell->node interpolation expects {n + 1} nodes for {n} cells, got {node_coords.numel()}."
        )
    out = values.new_empty((n + 1,) + tuple(values.shape[1:]))
    if n == 1:
        out[0] = values[0]
        out[1] = values[0]
        return out.movedim(0, axis)
    cc = cell_coords.to(values.dtype)
    nn = node_coords.to(values.dtype)
    for m in range(1, n):
        span = cc[m] - cc[m - 1]
        w = (nn[m] - cc[m - 1]) / span
        out[m] = (1.0 - w) * values[m - 1] + w * values[m]
    # Linear extrapolation to the two boundary nodes.
    w0 = (nn[0] - cc[0]) / (cc[1] - cc[0])
    out[0] = values[0] + w0 * (values[1] - values[0])
    wn = (nn[n] - cc[n - 1]) / (cc[n - 1] - cc[n - 2])
    out[n] = values[n - 1] + wn * (values[n - 1] - values[n - 2])
    return out.movedim(0, axis)


def _potential_to_nodes(phi_cell, prepared, dtype):
    """Interpolate the cell-centred potential onto the FDTD primary node grid.

    Returns ``phi_node`` of shape ``(Nx, Ny, Nz)`` (one sample per Yee corner).
    Taking Yee edge differences of this nodal scalar reproduces exactly the Yee
    E-component sample locations, and the discrete curl of that gradient is
    identically zero (the ``d.d = 0`` identity), so ``H = 0`` is preserved and the
    injected field is a discrete FDTD steady state in every lossless cell.
    """
    device = phi_cell.device
    xc = torch.as_tensor(np.asarray(prepared.x_half64), dtype=dtype, device=device)
    yc = torch.as_tensor(np.asarray(prepared.y_half64), dtype=dtype, device=device)
    zc = torch.as_tensor(np.asarray(prepared.z_half64), dtype=dtype, device=device)
    xn = torch.as_tensor(np.asarray(prepared.x_nodes64), dtype=dtype, device=device)
    yn = torch.as_tensor(np.asarray(prepared.y_nodes64), dtype=dtype, device=device)
    zn = torch.as_tensor(np.asarray(prepared.z_nodes64), dtype=dtype, device=device)
    phi = phi_cell.to(dtype)
    phi = _cells_to_nodes(phi, 0, xc, xn)
    phi = _cells_to_nodes(phi, 1, yc, yn)
    phi = _cells_to_nodes(phi, 2, zc, zn)
    return phi


def _node_edge_fields(phi_node, prepared, dtype):
    """Yee edge fields ``E = -grad(phi_node)`` on the staggered grid.

    ``Ex`` sits on x-edges ``(x_half, y_node, z_node)`` with shape ``(Nx-1, Ny, Nz)``,
    and analogously for ``Ey`` / ``Ez`` -- exactly the native FDTD component shapes.
    """
    device = phi_node.device
    dx = torch.as_tensor(np.asarray(prepared.dx_primal64), dtype=dtype, device=device)
    dy = torch.as_tensor(np.asarray(prepared.dy_primal64), dtype=dtype, device=device)
    dz = torch.as_tensor(np.asarray(prepared.dz_primal64), dtype=dtype, device=device)
    Ex = -(phi_node[1:, :, :] - phi_node[:-1, :, :]) / dx[:, None, None]
    Ey = -(phi_node[:, 1:, :] - phi_node[:, :-1, :]) / dy[None, :, None]
    Ez = -(phi_node[:, :, 1:] - phi_node[:, :, :-1]) / dz[None, None, :]
    return Ex, Ey, Ez


def _gauss_residual(Ex, Ey, Ez, solver, ic, dtype):
    """Dimensionless discrete-Gauss consistency residual of the mapped field.

    Compares the Yee divergence of ``D = eps_fdtd * E_init`` (built from the
    FDTD-compiled edge permittivity) against the electrostatic free-charge density
    at the shared interior primary nodes ``div D - rho_free``. Interior nodes that
    touch a conductor (electrostatic terminal) cell are excluded: the discrete
    divergence there equals the *induced surface charge*, which is a genuine charge
    the free-charge source term does not carry, not a mapping inconsistency. The
    residual is normalised by the characteristic charge-density scale
    ``|D|_peak / h_min`` so a uniform / conductor-bounded charge-free field reads
    ~roundoff while a grossly inconsistent mapping reads O(1). Returns
    ``(residual, abs_residual)``.
    """
    device = Ex.device
    eps_Ex = solver.eps_Ex.to(dtype)
    eps_Ey = solver.eps_Ey.to(dtype)
    eps_Ez = solver.eps_Ez.to(dtype)
    Dx = eps_Ex * Ex.to(dtype)
    Dy = eps_Ey * Ey.to(dtype)
    Dz = eps_Ez * Ez.to(dtype)

    xc = torch.as_tensor(np.asarray(solver.scene.x_half64), dtype=dtype, device=device)
    yc = torch.as_tensor(np.asarray(solver.scene.y_half64), dtype=dtype, device=device)
    zc = torch.as_tensor(np.asarray(solver.scene.z_half64), dtype=dtype, device=device)
    dxc = xc[1:] - xc[:-1]
    dyc = yc[1:] - yc[:-1]
    dzc = zc[1:] - zc[:-1]

    # div D at interior primary nodes (i in 1..Nx-2, ...): each partial derivative
    # lands on the same node, then restrict the transverse axes to interior nodes.
    div_x = ((Dx[1:, :, :] - Dx[:-1, :, :]) / dxc[:, None, None])[:, 1:-1, 1:-1]
    div_y = ((Dy[:, 1:, :] - Dy[:, :-1, :]) / dyc[None, :, None])[1:-1, :, 1:-1]
    div_z = ((Dz[:, :, 1:] - Dz[:, :, :-1]) / dzc[None, None, :])[1:-1, 1:-1, :]
    div_d = div_x + div_y + div_z  # (Nx-2, Ny-2, Nz-2), C/m^3

    # Free-charge density at cell centres -> interior primary nodes.
    rho_cell = (ic.free_charge / ic.cell_volume).to(dtype)
    xn = torch.as_tensor(np.asarray(solver.scene.x_nodes64), dtype=dtype, device=device)
    yn = torch.as_tensor(np.asarray(solver.scene.y_nodes64), dtype=dtype, device=device)
    zn = torch.as_tensor(np.asarray(solver.scene.z_nodes64), dtype=dtype, device=device)
    rho_node = _cells_to_nodes(rho_cell, 0, xc, xn)
    rho_node = _cells_to_nodes(rho_node, 1, yc, yn)
    rho_node = _cells_to_nodes(rho_node, 2, zc, zn)
    rho_node = rho_node[1:-1, 1:-1, 1:-1]

    # Keep only interior nodes whose eight surrounding cells are all conductor-free:
    # an interior primary node (i,j,k) is the shared corner of cells with indices
    # {i-1,i} x {j-1,j} x {k-1,k}, so a node is "clean" iff none of those cells is a
    # conductor. Induced surface charge on a terminal is not a free-charge Gauss
    # violation and would otherwise dominate the residual.
    clean = torch.ones_like(div_d, dtype=torch.bool)
    if ic.conductor_mask is not None:
        cond = ic.conductor_mask.to(device=device, dtype=torch.bool)
        # A primary node (i,j,k) is clean iff none of its eight corner cells
        # {i-1,i} x {j-1,j} x {k-1,k} is a conductor cell.
        cond_f = cond.to(dtype)
        # Count conductor cells among the 8 corner cells of each interior node.
        # cond has Nx-1 cells; pairing adjacent cells gives Nx-2 interior nodes,
        # already aligned with div_d (interior node i corresponds to index i-1).
        acc = (
            cond_f[:-1, :-1, :-1] + cond_f[1:, :-1, :-1]
            + cond_f[:-1, 1:, :-1] + cond_f[1:, 1:, :-1]
            + cond_f[:-1, :-1, 1:] + cond_f[1:, :-1, 1:]
            + cond_f[:-1, 1:, 1:] + cond_f[1:, 1:, 1:]
        )  # shape (Nx-2, Ny-2, Nz-2), aligned with div_d
        clean = acc == 0

    diff = (div_d - rho_node)[clean]
    abs_res = float(diff.abs().max()) if diff.numel() else 0.0
    d_peak = max(float(Dx.abs().max()), float(Dy.abs().max()), float(Dz.abs().max()))
    h_min = float(min(float(dxc.min()), float(dyc.min()), float(dzc.min())))
    char_div = d_peak / h_min if h_min > 0.0 else d_peak
    tiny = 1.0e-300
    residual = abs_res / (char_div + tiny)
    return residual, abs_res


@dataclass
class ElectrostaticInitialCondition:
    """A DC electrostatic solution mapped onto the FDTD Yee grid as an initial field.

    Built with :meth:`from_result` from a ``Result(method="electrostatic")``. At
    FDTD ``prepare()`` the condition seeds the staggered ``E`` buffers with
    ``E = -grad(phi)`` and leaves ``H`` and the CPML memory at zero, so a lossless
    interior cell starts in a discrete FDTD steady state (the discrete curl of a
    discrete gradient is exactly zero, hence ``H`` stays zero and ``E`` stays
    constant with no source). ``H = 0`` neglects any DC magnetic pre-bias, which
    electrostatics does not model.

    The electrostatic grid must be node-identical to the FDTD scene grid; a
    mismatch fails closed. After mapping, the discrete-Gauss residual of
    ``eps_fdtd * E`` against the electrostatic free charge is recorded in
    :attr:`gauss_residual` and rejected when it exceeds :attr:`tolerance`.
    """

    phi: torch.Tensor            # cell-centred potential (nx, ny, nz)
    xc: torch.Tensor             # cell-centre coordinates (nx,)
    yc: torch.Tensor             # (ny,)
    zc: torch.Tensor             # (nz,)
    epsilon_r: torch.Tensor      # DC relative permittivity at cell centres (nx, ny, nz)
    free_charge: torch.Tensor    # free charge per cell in Coulombs (nx, ny, nz)
    cell_volume: torch.Tensor    # (nx, ny, nz)
    energy: float                # electrostatic field energy [J]
    tolerance: float
    provenance: dict[str, Any]
    conductor_mask: torch.Tensor | None = None  # bool (nx, ny, nz): pinned conductor cells
    gauss_residual: float | None = None
    gauss_residual_abs: float | None = None

    @classmethod
    def from_result(cls, dc_result, *, tolerance: float = DEFAULT_GAUSS_TOLERANCE) -> "ElectrostaticInitialCondition":
        """Wrap a ``Result(method="electrostatic")`` as an FDTD pre-bias condition."""
        if getattr(dc_result, "method", None) != "electrostatic":
            raise ValueError(
                "ElectrostaticInitialCondition.from_result requires a "
                "Result(method='electrostatic') produced by Simulation.electrostatic(...); "
                f"got method {getattr(dc_result, 'method', None)!r}."
            )
        data = dc_result.electrostatic  # raises AttributeError if not electrostatic
        tol = float(tolerance)
        if tol < 0.0:
            raise ValueError("ElectrostaticInitialCondition tolerance must be >= 0.")
        provenance = {
            "source": "electrostatic",
            "capability_level": "electrostatic-prebias",
            "grid_shape": tuple(int(s) for s in data.potential.shape),
            "dc_residual": float(data.residual),
            "dc_residual_abs": float(data.residual_abs),
            "dc_iterations": int(data.iterations),
            "dc_gauss_error": float(data.gauss_error),
            "energy": float(data.energy),
            "terminal_charges": {
                name: float(data.terminal_charge(name)) for name in data.terminal_names
            },
            "magnetic_prebias": "zero (electrostatic solution carries no H)",
            "tolerance": tol,
        }
        return cls(
            phi=data.potential.detach(),
            xc=data.xc.detach(),
            yc=data.yc.detach(),
            zc=data.zc.detach(),
            epsilon_r=data.epsilon_r.detach(),
            free_charge=data.free_charge.detach(),
            cell_volume=data.cell_volume.detach(),
            energy=float(data.energy),
            tolerance=tol,
            provenance=provenance,
            conductor_mask=(
                data.conductor_mask.detach() if data.conductor_mask is not None else None
            ),
        )

    def _validate_grid(self, prepared) -> None:
        """Fail closed unless the DC cell-centre grid matches the FDTD grid nodes."""
        expected = {
            "x": np.asarray(prepared.x_half64, dtype=np.float64),
            "y": np.asarray(prepared.y_half64, dtype=np.float64),
            "z": np.asarray(prepared.z_half64, dtype=np.float64),
        }
        actual = {
            "x": self.xc.detach().cpu().numpy().astype(np.float64),
            "y": self.yc.detach().cpu().numpy().astype(np.float64),
            "z": self.zc.detach().cpu().numpy().astype(np.float64),
        }
        for axis in ("x", "y", "z"):
            exp, act = expected[axis], actual[axis]
            if exp.shape != act.shape or not np.allclose(exp, act, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "ElectrostaticInitialCondition grid does not match the FDTD scene grid on "
                    f"axis {axis!r} (electrostatic cell centres {act.shape} vs FDTD {exp.shape}). "
                    "The DC solve and the FDTD run must use an identical Domain, GridSpec, and a "
                    "grid-non-extending boundary (electrostatic BoundarySpec.none() pairs with an "
                    "FDTD PEC/PMC/periodic boundary; a PML boundary extends the grid and is rejected)."
                )

    def map_to_yee(self, prepared, *, dtype=torch.float64):
        """Return ``(Ex, Ey, Ez)`` on the Yee edges for ``prepared`` (validated grid)."""
        self._validate_grid(prepared)
        phi_node = _potential_to_nodes(self.phi.to(prepared.device), prepared, dtype)
        return _node_edge_fields(phi_node, prepared, dtype)

    def apply_to_solver(self, solver) -> None:
        """Seed the solver's ``E`` buffers with the mapped DC field (H, CPML left zero).

        Runs the grid-identity check and the discrete-Gauss consistency gate, then
        writes the staggered field into ``solver.Ex/Ey/Ez`` in place. Must be called
        after ``solver.init_field()`` (the compiled edge permittivity ``eps_E*`` and
        the zeroed field buffers must already exist).
        """
        prepared = solver.scene
        Ex, Ey, Ez = self.map_to_yee(prepared, dtype=torch.float64)
        residual, residual_abs = _gauss_residual(Ex, Ey, Ez, solver, self, torch.float64)
        self.gauss_residual = residual
        self.gauss_residual_abs = residual_abs
        self.provenance["gauss_residual"] = residual
        self.provenance["gauss_residual_abs"] = residual_abs
        if residual > self.tolerance:
            raise ValueError(
                "Electrostatic pre-bias failed the discrete-Gauss consistency gate: relative "
                f"residual {residual:.3e} exceeds tolerance {self.tolerance:.3e} (absolute "
                f"{residual_abs:.3e} C/m^3). The electrostatic (harmonic-face) and FDTD "
                "(edge-averaged) permittivity discretizations disagree too strongly for this "
                "mapping; refine the mesh at material interfaces or raise the tolerance "
                "explicitly (do not accept a large residual silently)."
            )
        solver.Ex[...] = Ex.to(solver.Ex.dtype)
        solver.Ey[...] = Ey.to(solver.Ey.dtype)
        solver.Ez[...] = Ez.to(solver.Ez.dtype)
