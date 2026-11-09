"""Conformal two-sided thin-sheet runtime (PEC and finite copper, any tilt).

The native Yee block updates every field first. The sheet patch then replaces
the H rows of cut/sub-faces and the E rows of every edge whose Ampere
circulation reads them, using the sparse circulation matrix of
:mod:`..compiler.thin_sheet`. Second-side sub-face fluxes, pierced-edge
segments and on-sheet traces live in private float64 buffers. Finite copper
carries independent two-sided chord fields,
advanced by a local Newmark solve with the foil's passive diffusion two-port.
A fixed even Chebyshev polynomial and 2x2 block Jacobi keep the solve sparse
and device resident.

Execution scope (validated at preparation): native CUDA or CPU reference
backend, real fields, single device, forward only, non-periodic boundaries, isotropic linear non-dispersive
non-magnetic media in the patch, patch outside CPML, no electric sources,
non-feed ports, lumped elements, wires, TFSF or other conductor owners on the patch.
"""
from __future__ import annotations

import math
import os
from copy import copy

import numpy as np
import torch

from ..compiler.thin_sheet import _compile_thin_sheet, compile_thin_sheet, thin_sheet_structures

_E = ("Ex", "Ey", "Ez")
_H = ("Hx", "Hy", "Hz")
CFL_SAFETY = 0.985


def _gather_torch(solver, plan, extra_buffer, values):
    for name, positions, flat in plan["native"]:
        values.index_copy_(0, positions, getattr(solver, name).reshape(-1).index_select(0, flat).double())
    if len(plan["extra_positions"]):
        values.index_copy_(0, plan["extra_positions"], extra_buffer.index_select(0, plan["extra_slots"]))
    return values


def _scatter_torch(solver, plan, extra_buffer, values):
    for name, positions, flat in plan["native"]:
        field = getattr(solver, name)
        field.view(-1).index_copy_(0, flat, values.index_select(0, positions).to(field.dtype))
    if len(plan["extra_positions"]):
        extra_buffer.index_copy_(0, plan["extra_slots"], values.index_select(0, plan["extra_positions"]))


def _gather_fused(solver, plan, extra_buffer, values):
    native = plan["fused"]
    native["gather"](*(getattr(solver, name) for name in native["names"]),
                     extra_buffer, native["components"], native["offsets"], values)
    return values


def _scatter_fused(solver, plan, extra_buffer, values):
    native = plan["fused"]
    native["scatter"](*(getattr(solver, name) for name in native["names"]),
                      extra_buffer, native["components"], native["offsets"], values)


def _copper_rhs_torch(self, solver, old, new):
    cu, dt = self.copper, float(solver.dt)
    count = cu["count"]
    history = (cu["history"] @ cu["state"]).T
    current, previous = self.chord_e.reshape(-1), self.chord_e_prev.reshape(-1)
    mass, conductance = cu["mass"].reshape(-1), cu["conductance"].reshape(-1)
    rhs = (mass / dt - conductance / 2) * current + self.chord_f.reshape(-1)
    if self.feeds:
        local_current = torch.einsum("tij,tj->ti", cu["local"][:count], self.chord_e)
        # Feed matrices already include the side-average factors.
        chord_area = self.feed_area
        rhs = rhs - (chord_area[:, None] * local_current / 2
                     + cu["area"][:count, None] * history[:count]).reshape(-1)
        rhs = rhs - torch.mv(cu["feed_conductance"], current) / 2
        for feed in self.feeds:
            for column in feed["columns"]:
                if column["kind"] == "chord":
                    rhs[column["dofs"]] += column["conductance"] * column["u"] * feed["binding"].drive
    else:
        rhs = rhs - (cu["area"][:count, None] * (self.chord_e @ cu["effective"].T / 2 + history[:count])).reshape(-1)
    rhs = rhs - dt / 4 * (torch.mv(cu["Acx"], new + 2 * old + self.x_prev)
                          + torch.mv(cu["Acc"], 2 * current + previous))
    return history, rhs


def _chord_forcing_torch(self, solver, old, new, updated):
    cu, dt = self.copper, float(solver.dt)
    current, previous = self.chord_e.reshape(-1), self.chord_e_prev.reshape(-1)
    self.chord_f.sub_((dt / 4 * (torch.mv(cu["Acx"], new + 2 * old + self.x_prev)
                                   + torch.mv(cu["Acc"], updated + 2 * current + previous))).reshape(-1, 2))


def _copper_state_torch(self, solver, old, new, updated, pair_old, pair_new):
    cu, dt = self.copper, float(solver.dt)
    trace_old = torch.cat((self.chord_e, pair_old)).T
    trace_new = torch.cat((updated.reshape(-1, 2), pair_new)).T
    state = cu["state"]
    next_state = cu["decay"][:, None] * state + cu["drive"] @ (trace_new + trace_old)
    for feed in self.feeds:
        next_state.index_fill_(1, feed["ids"], 0)
    midpoint_e, midpoint_state = (trace_new + trace_old) / 2, (next_state + state) / 2
    loss = (cu["rates"][:, None] * midpoint_state.square()).sum(0)
    loss = loss + (midpoint_e * (cu["D"] @ midpoint_e)).sum(0)
    self.loss.add_(dt * torch.sum(loss * cu["area"]))
    for feed in self.feeds:
        binding = feed["binding"]
        for index, column in enumerate(feed["columns"]):
            binding.column_voltage[index] = torch.dot(midpoint_e.T[column["ids"]].reshape(-1), column["u"])
        g = binding.column_conductance
        binding.voltage.copy_(torch.dot(g, binding.column_voltage) / g.sum())
        binding.current.copy_(torch.sum(g * (binding.drive - binding.column_voltage)))
    state.copy_(next_state)
    self.chord_e_prev.copy_(self.chord_e)
    self.chord_e.copy_(updated.reshape(-1, 2))
    self.x_prev.copy_(old)
    new[cu["promoted_position"].reshape(-1)] = self.chord_e[cu["promoted_chords"]].reshape(-1)


def _copper_rhs_fused(self, solver, old, new):
    cu = self.copper
    plan = cu["fused"]
    plan["rhs_kernel"](cu["state"], cu["history"], cu["effective"], cu.get("local"),
        cu["mass"], cu["conductance"], cu["area"], self.feed_area,
        self.chord_e, self.chord_e_prev, self.chord_f, old, new, self.x_prev,
        *plan["acx"], plan["acx_lanes"], *plan["acc"], plan["acc_lanes"], *plan["feed"],
        plan["history"], plan["rhs"], plan["coupled"], float(solver.dt), bool(self.feeds))
    rhs = plan["rhs"]
    if self.feeds:
        for feed in self.feeds:
            for column in feed["columns"]:
                if column["kind"] == "chord":
                    rhs[column["dofs"]] += column["conductance"] * column["u"] * feed["binding"].drive
        rhs.sub_(plan["coupled"])
    return plan["history"].T, rhs


def _chord_forcing_fused(self, solver, old, new, updated):
    plan = self.copper["fused"]
    plan["forcing_kernel"](*plan["acx"], plan["acx_lanes"], *plan["acc"], plan["acc_lanes"], old, new, self.x_prev,
        self.chord_e, self.chord_e_prev, updated, self.chord_f, float(solver.dt))


def _copper_state_fused(self, solver, old, new, updated, pair_old, pair_new):
    cu = self.copper
    plan = cu["fused"]
    plan["state_kernel"](cu["state"], cu["decay"], cu["drive"], cu["rates"], cu["D"], cu["area"],
        plan["feed_mask"], self.chord_e, self.chord_e_prev, updated, pair_old, pair_new,
        old, new, self.x_prev, plan["promoted"], plan["midpoint"],
        self.loss, plan["partials"], plan["counter"], float(solver.dt))
    midpoint_e = plan["midpoint"].T
    for feed in self.feeds:
        binding = feed["binding"]
        for index, column in enumerate(feed["columns"]):
            binding.column_voltage[index] = torch.dot(midpoint_e.T[column["ids"]].reshape(-1), column["u"])
        g = binding.column_conductance
        binding.voltage.copy_(torch.dot(g, binding.column_voltage) / g.sum())
        binding.current.copy_(torch.sum(g * (binding.drive - binding.column_voltage)))


def _chord_solve_torch(self, rhs):
    """Even Chebyshev residual s_k(mu)^2: 0 < P <= S^-1, with block Jacobi."""
    cu = self.copper
    if not cu["count"]:
        return rhs.clone()
    a, b, k = cu["chebyshev"]
    theta, delta = (b+a)/2, (b-a)/2
    sigma = theta / delta
    y, residual = torch.zeros_like(rhs), rhs.clone()
    for _ in range(2):
        rho = 1 / sigma
        direction = torch.mv(cu["R"], torch.mv(cu["R"], residual)) / theta
        for step in range(k):
            y = y + direction
            residual = residual - torch.mv(cu["S"], direction)
            if step + 1 < k:
                rho_next = 1 / (2 * sigma - rho)
                direction = rho_next * rho * direction + (2 * rho_next / delta) * torch.mv(cu["R"], torch.mv(cu["R"], residual))
                rho = rho_next
    return y


def _chord_solve_fused(self, rhs):
    cu = self.copper
    if not cu["count"]:
        return rhs.clone()
    plan = cu["fused"]["chord_solve"]
    # Each result owns its storage, including when callers retain successive solves.
    y = torch.empty_like(rhs)
    residual, direction, temporary = (plan[name] for name in ("residual", "direction", "temporary"))
    plan["initialize_kernel"](rhs, y, residual)
    for _ in range(2):
        plan["matvec_kernel"](*plan["R"], plan["R_lanes"], residual, temporary)
        plan["direction_kernel"](*plan["R"], plan["R_lanes"], temporary, direction, 0., plan["inverse_theta"], True)
        for step in range(plan["degree"]):
            plan["residual_kernel"](*plan["S"], plan["S_lanes"], direction, y, residual)
            if step + 1 < plan["degree"]:
                plan["matvec_kernel"](*plan["R"], plan["R_lanes"], residual, temporary)
                alpha, beta = plan["coefficients"][step]
                plan["direction_kernel"](*plan["R"], plan["R_lanes"], temporary, direction, alpha, beta, False)
    return y


def _sheet_csr_lanes(matrix):
    """Smallest power of two >= ceil(mean row nnz / 2), clamped to [1, 8]."""
    target = max(1, math.ceil(matrix.values().numel() / (2 * max(1, matrix.shape[0]))))
    return min(8, 1 << (target - 1).bit_length())


def _sparse(rows, cols, vals, shape):
    import scipy.sparse as sp
    return sp.csr_matrix((vals, (rows, cols)), shape=shape)


def _require_scope(solver):
    device = torch.device(solver.device)
    if not (bool(getattr(solver, "cpu_reference", False)) or device.type == "cuda"):
        raise NotImplementedError(
            "Conformal thin sheets run on the native CUDA backend or the explicit CPU reference backend.")
    for axis in "xyz":
        if solver.scene.boundary.axis_kind(axis) in ("periodic", "bloch"):
            raise NotImplementedError("Conformal thin sheets do not yet support periodic or Bloch boundaries.")
    for name in ("complex_fields_enabled", "full_aniso_enabled", "nonlinear_enabled", "modulation_enabled",
                 "gyromagnetic_enabled", "breakdown_enabled", "dispersive_enabled", "magnetic_dispersive_enabled",
                 "tfsf_enabled"):
        if bool(getattr(solver, name, False)):
            raise NotImplementedError(f"Conformal thin sheets do not support {name.removesuffix('_enabled')} execution.")
    if getattr(solver.scene, "coordinates", None) is not None:
        raise NotImplementedError("Conformal thin sheets require Cartesian coordinates.")
    if getattr(solver.scene, "_fdtd_shard_layout", None) is not None:
        raise NotImplementedError("Conformal thin sheets require single-device execution.")
    if getattr(solver, "_wire_runtime", None) is not None:
        raise NotImplementedError("Conformal thin sheets do not yet support wires in the same scene.")


def _parents(complex_, kind):
    if kind == "e":
        return complex_.e_component, complex_.e_index
    return complex_.h_component, complex_.h_index


def _gather_native(solver, names, components, index, dtype=torch.float64):
    values = torch.empty(len(components), dtype=dtype)
    for comp in range(3):
        mask = components == comp
        if mask.any():
            ijk = torch.as_tensor(index[mask], dtype=torch.long)
            field = getattr(solver, names[comp])
            ijk = ijk.to(field.device)
            values[torch.as_tensor(np.flatnonzero(mask))] = field[ijk[:, 0], ijk[:, 1], ijk[:, 2]].to("cpu", dtype)
    return values


def _side_materials(solver, c):
    """Per-unknown permittivity/conductivity with the sheet's two sides kept apart.

    Native edge materials near the sheet are averaged across it (subpixel or
    staircase sampling), which would give both sides the same mixed value on a
    dielectric interface. Each side's medium is read from same-side unknowns
    between one and two cells from the plane and must be homogeneous there;
    every definite-side unknown within one cell then receives its own side's
    medium. A value within one cell that is not a mixture of the two side media
    reveals a third material in the cut cells and is rejected.
    """
    eps = _gather_native(solver, tuple("eps_" + n for n in _E), c.e_component, c.e_index).numpy()
    sigma = _gather_native(solver, tuple("sigma_e_" + n for n in _E), c.e_component, c.e_index).numpy()
    h = max(float(np.max(getattr(solver.scene, f"d{a}_primal64"))) for a in "xyz")
    distance = np.abs(c.e_phi)
    near = distance < h
    media = {}
    for side in (-1, 1):
        ring = (c.e_side == side) & ~c.e_fixed & ~near & (distance <= 2 * h)
        if not ring.any():
            ring = (c.e_side == side) & ~c.e_fixed
        if not ring.any():
            continue
        e_side, s_side = eps[ring], sigma[ring]
        if (np.ptp(e_side) > 1e-6 * e_side.max()) or (np.ptp(s_side) > 1e-6 * max(s_side.max(), 1e-30)):
            raise NotImplementedError(
                "Conformal thin sheets require a homogeneous isotropic medium on each side within two cells of "
                "the sheet; a material boundary there is a mixed cut cell the runtime cannot represent.")
        media[side] = (float(e_side[0]), float(s_side[0]))
    # Unknowns whose Yee dual box lies wholly on one side cannot carry a value
    # averaged across the sheet: a deviation from their side medium is a
    # material boundary inside the cut region.
    scene = solver.scene
    primal = [np.asarray(getattr(scene, f"d{a}_primal64")) for a in "xyz"]
    extents = np.zeros((len(c.e_component), 3))
    for comp in range(3):
        select = c.e_component == comp
        index = c.e_index[select]
        for axis in range(3):
            if axis == comp:
                extents[select, axis] = primal[axis][np.minimum(index[:, axis], len(primal[axis]) - 1)]
            else:
                lower = np.where(index[:, axis] > 0, primal[axis][np.maximum(index[:, axis] - 1, 0)], 0.0)
                upper = np.where(index[:, axis] < len(primal[axis]), primal[axis][np.minimum(index[:, axis], len(primal[axis]) - 1)], 0.0)
                extents[select, axis] = 0.5 * (lower + upper)
    reach = 0.5 * np.sum(np.abs(extents) * np.abs(c.e_normal), -1)
    one_sided = near & (c.e_side != 0) & ~c.e_fixed & ~c.e_split & (distance >= reach * (1 - 1e-9))
    for side, (eps_side, sigma_side) in media.items():
        select = one_sided & (c.e_side == side)
        if (np.any(np.abs(eps[select] - eps_side) > 1e-6 * eps_side)
                or np.any(np.abs(sigma[select] - sigma_side) > 1e-6 * max(abs(sigma_side), 1e-30))):
            raise NotImplementedError(
                "A third material enters the thin-sheet cut cells; mixed sub-cell media are not represented.")
    if len(media) == 2:
        bounds_e = sorted((media[-1][0], media[1][0]))
        bounds_s = sorted((media[-1][1], media[1][1]))
        close = near & (c.e_side != 0) & ~c.e_fixed
        tolerance = 1e-6
        if (np.any(eps[close] < bounds_e[0] * (1 - tolerance)) or np.any(eps[close] > bounds_e[1] * (1 + tolerance))
                or np.any(sigma[close] < bounds_s[0] - tolerance * max(bounds_s[1], 1e-30))
                or np.any(sigma[close] > bounds_s[1] + tolerance * max(bounds_s[1], 1e-30))):
            raise NotImplementedError(
                "A third material enters the thin-sheet cut cells; mixed sub-cell media are not represented.")
    for side, (eps_side, sigma_side) in media.items():
        mask = (c.e_side == side) & near
        eps[mask], sigma[mask] = eps_side, sigma_side
    return eps, sigma


def _absorber_overlap(solver, c, e_dofs, h_dofs):
    from .runtime.materials import _absorbing_region_node_mask
    mask = _absorbing_region_node_mask(solver)
    if mask is None:
        return False
    mask = mask.detach().cpu().numpy().astype(bool)
    grown = mask.copy()
    for axis in range(3):
        for shift in (-1, 1):
            grown |= np.roll(mask, shift, axis=axis)
    def touches(components, index, offsets):
        for comp_offsets in offsets:
            for comp in range(3):
                select = components == comp
                if not select.any():
                    continue
                for offset in comp_offsets[comp]:
                    point = np.clip(index[select] + np.array(offset), 0, np.array(mask.shape) - 1)
                    if grown[point[:, 0], point[:, 1], point[:, 2]].any():
                        return True
        return False
    edge_offsets = [{comp: [(0, 0, 0), tuple(np.eye(3, dtype=int)[comp])] for comp in range(3)}]
    face_offsets = []
    for comp in range(3):
        b, cc = [axis for axis in range(3) if axis != comp]
        corners = [np.zeros(3, int), np.eye(3, dtype=int)[b], np.eye(3, dtype=int)[cc], np.eye(3, dtype=int)[b] + np.eye(3, dtype=int)[cc]]
        face_offsets.append({axis: ([tuple(v) for v in corners] if axis == comp else []) for axis in range(3)})
    return (touches(c.e_component[e_dofs], c.e_index[e_dofs], edge_offsets)
            or touches(c.h_component[h_dofs], c.h_index[h_dofs], face_offsets))


def prepare_thin_sheet_time_step(solver):
    """Compile sheet complexes and tighten dt to the composite leapfrog bound."""
    solver._thin_sheet_complexes = ()
    structures = thin_sheet_structures(solver.scene)
    if not structures:
        return
    _require_scope(solver)
    for structure in structures:
        material = structure.material
        if any(torch.is_tensor(value) and value.requires_grad
               for value in (getattr(material, "thickness", None), getattr(material, "conductivity", None))):
            raise NotImplementedError("Conformal thin sheets are forward-only; trainable copper parameters are not "
                                      "supported.")
    from scipy.sparse.linalg import eigsh
    import scipy.sparse as sp
    complexes = []
    worst = 0.0
    for structure in structures:
        complex_, geometry = _compile_thin_sheet(solver.scene, structure)
        if complex_.is_pec and complex_.feeds:
            # Reuse the chord compiler for PEC windows; all arm chords stay held.
            from ..copper_foil import CopperFoil
            from witwin.core import Structure
            chord_structure = Structure(geometry=structure.geometry, material=CopperFoil(
                thickness=35e-6, frequency_range=(solver.source_frequency / 2, 2 * solver.source_frequency)))
            if geometry is None:
                # On-sheet PEC holds change the mass partition inputs.
                chord_scene = copy(solver.scene)
                chord_scene.structures = tuple(chord_structure if s is structure else s for s in solver.scene.structures)
                geometry = compile_thin_sheet(chord_scene, chord_structure)
            else:
                geometry.name = str(getattr(chord_structure, "name", None) or "thin_sheet")
            geometry.is_pec, geometry.material = True, complex_.material
            geometry.e_fixed = complex_.e_fixed.copy()
            geometry.chord_fixed[:] = True
            for feed in geometry.feeds:
                geometry.chord_fixed[feed.chord_index[feed.chord_index >= 0]] = False
                pairs = feed.trace_dofs[feed.trace_dofs[:, 0] >= 0]
                geometry.e_fixed[pairs.reshape(-1)] = False
            complex_ = geometry
        eps, _ = _side_materials(solver, complex_)
        dynamic = ~complex_.e_fixed
        if not complex_.is_pec or complex_.feeds:
            dynamic[complex_.trace_dofs.reshape(-1)] = False
        prefix = "k_" if complex_.is_pec and not complex_.feeds else "k_x_"
        K = _sparse(*(getattr(complex_, prefix + suffix) for suffix in ("rows", "cols", "vals")),
                    (len(complex_.h_area), len(complex_.e_volume)))
        K = K[:, np.flatnonzero(dynamic)]
        lh = complex_.h_dual / (solver.mu0 * complex_.h_area)
        me = eps[dynamic] * complex_.e_volume[dynamic]
        scaled = sp.diags(np.sqrt(lh)) @ K @ sp.diags(1 / np.sqrt(me))
        operator = (scaled.T @ scaled).tocsr()
        rng = np.random.default_rng(0)
        value = float(eigsh(operator, k=1, which="LA", tol=1e-6, v0=rng.standard_normal(operator.shape[0]),
                            return_eigenvectors=False)[0])
        complex_.statistics["spectral_radius"] = value
        worst = max(worst, value)
        complexes.append(complex_)
    feed_names = {f.port_name for c in complexes for f in c.feeds}
    if any(port.name not in feed_names for port in solver.scene.ports):
        raise NotImplementedError("Conformal thin sheets support only sheet gap feeds as ports in the same scene.")
    limit = 2.0 / math.sqrt(worst)
    requested = getattr(solver, "_requested_dt", None)
    if float(solver.dt) > CFL_SAFETY * limit:
        if requested is not None:
            raise ValueError(f"TimeConfig.dt exceeds the conformal thin-sheet leapfrog limit {CFL_SAFETY * limit:.12g} s.")
        solver.dt = CFL_SAFETY * limit
        if hasattr(solver, "_time_step_metadata"):
            solver._time_step_metadata["limiting_factor"] = "thin_sheet_cfl"
    solver._thin_sheet_cfl_limit = limit
    solver._thin_sheet_complexes = tuple(complexes)


def _chebyshev_degree(kappa, tolerance):
    """Smallest k with (kappa - 1) eps_k^2 <= tolerance, eps_k = 1 / T_k((kappa + 1) / (kappa - 1))."""
    if kappa <= 1 + 1e-12:
        return 1
    sigma = (kappa + 1) / (kappa - 1)
    for k in range(1, 400):
        eps = 1.0 / math.cosh(k * math.acosh(sigma))
        if (kappa - 1) * eps * eps <= tolerance:
            return k
    raise RuntimeError("The copper preconditioner spectrum is too wide for the Chebyshev solve.")


def _feed_quadrature(c, feed):
    """Partition the physical gap at chord midplanes, clipping the rim cells."""
    facet = c.surface.facets[feed.facet]
    transverse = np.cross(facet.normal, feed.direction)
    basis = np.stack((feed.direction, transverse))
    boundary = (facet.origin + np.concatenate((facet.rim, facet.folds)) @ facet.basis
                - feed.midpoint) @ basis.T
    tolerance = 1e-9 * np.min(feed.chord_length)
    crossings = []
    for start, end in boundary:
        if abs(start[0]) <= tolerance:
            crossings.append(start[1])
        if start[0] * end[0] < 0:
            crossings.append(start[1] - start[0] * (end[1] - start[1]) / (end[0] - start[0]))
    lower = max((y for y in crossings if y < -tolerance), default=0.0)
    upper = min((y for y in crossings if y > tolerance), default=0.0)
    driven = ~feed.open_chords
    centers = (np.array([path.mean(0) for path in feed.chord_path])[driven] - feed.midpoint) @ basis.T

    def partition(values, lo, hi):
        order = np.argsort(values)
        groups = np.cumsum(np.r_[0, np.diff(values[order]) > tolerance])
        inverse = np.empty_like(groups)
        inverse[order] = groups
        sites = np.bincount(inverse, weights=values) / np.bincount(inverse)
        boundaries = np.r_[lo, (sites[:-1] + sites[1:]) / 2, hi]
        return np.diff(boundaries)[inverse], inverse

    length, longitudinal = partition(centers[:, 0], -feed.length / 2, feed.length / 2)
    width, across = partition(centers[:, 1], lower, upper)
    # Coincident traces share one partition cell rather than duplicate its area.
    _, inverse, multiplicity = np.unique(np.stack((longitudinal, across), axis=1), axis=0,
                                         return_inverse=True, return_counts=True)
    lengths, widths = np.zeros(len(driven)), np.zeros(len(driven))
    lengths[driven], widths[driven] = length, width / multiplicity[inverse]
    columns = np.full(len(driven), -1, dtype=np.int64)
    columns[driven] = across
    return lengths, widths, columns


class _SheetRuntime:
    """Device-resident patch operators and private unknowns of one sheet.

    Host preparation (SciPy) builds the sparse operators once; every per-step
    operation is a fixed sequence of device kernels without host
    synchronization, so the patch runs inside the captured native CUDA graph.
    """

    def __init__(self, solver, complex_):
        import scipy.sparse as sp
        self.complex = complex_
        c = complex_
        device = torch.device(solver.device)
        self.device = device
        kernel = os.environ.get("WITWIN_MAXWELL_THIN_SHEET_KERNEL", "fused")
        if kernel not in ("fused", "torch"):
            raise ValueError("WITWIN_MAXWELL_THIN_SHEET_KERNEL must be fused or torch.")
        self._fused_kernels = device.type == "cuda" and kernel == "fused"
        dt = float(solver.dt)
        nE, nH = len(c.e_volume), len(c.h_area)
        K = (_sparse(c.k_rows, c.k_cols, c.k_vals, (nH, nE)) if c.is_pec and not c.feeds else
             _sparse(c.k_x_rows, c.k_x_cols, c.k_x_vals, (nH, nE))).tocsr()
        special_h = np.flatnonzero(c.h_special)
        patch_e = (set(np.unique(K[special_h].tocsr().indices).tolist()) | set(np.flatnonzero(c.e_split).tolist())
                   | set(np.flatnonzero(c.e_modified).tolist()))
        patch_e = np.array(sorted(patch_e), dtype=np.int64)
        self.patch_h = special_h
        self.patch_e = patch_e
        # Columns read by H rows and rows read by E rows.
        Kh = K[special_h].tocsr()
        self.h_cols = np.unique(Kh.indices)
        Ke = K.tocsc()[:, patch_e].tocsc()
        self.e_rows = np.unique(Ke.indices)
        self._check_box(solver, patch_e)
        # Coefficients: side-resolved media, native magnetic coefficients.
        eps, sigma = _side_materials(solver, c)
        cb_native = _gather_native(solver, tuple("c" + n.lower() + "_curl" for n in _E), c.e_component, c.e_index).numpy()
        da = _gather_native(solver, tuple("c" + n.lower() + "_decay" for n in _H), c.h_component, c.h_index).numpy()
        db = _gather_native(solver, tuple("c" + n.lower() + "_curl" for n in _H), c.h_component, c.h_index).numpy()
        pe = patch_e
        dynamic = ~c.e_fixed[pe]
        if not c.is_pec or c.feeds:
            promoted = c.chord_trace[c.chord_face[:, 0] < 0] - c.statistics["active_split_faces"]
            dynamic &= ~np.isin(pe, c.trace_dofs[promoted])
        if _absorber_overlap(solver, c, pe, special_h):
            raise NotImplementedError("A thin-sheet patch intersects the CPML absorber; keep the sheet at least "
                                      "two cells inside the physical domain.")
        if np.any(cb_native[pe][dynamic] == 0):
            raise NotImplementedError("A thin-sheet patch edge overlaps a PEC-suppressed electric update.")
        ca = (eps - 0.5 * sigma * dt) / (eps + 0.5 * sigma * dt)
        cb = dt / (eps + 0.5 * sigma * dt)
        ph = special_h
        if not (np.allclose(da[ph], 1.0, rtol=0, atol=1e-6) and np.allclose(db[ph], dt / solver.mu0, rtol=2e-6, atol=0)):
            raise NotImplementedError("Thin-sheet patch faces require non-magnetic, magnetically lossless media.")
        # H update: H = Da H - Db/A * (Kh @ E[h_cols])
        Kh_local = Kh[:, self.h_cols].tocsr()
        Kh_local = sp.diags(db[ph] / c.h_area[ph]) @ Kh_local
        self.h_matrix = self._csr(Kh_local)
        self.h_decay = self._tensor(da[ph])
        # E update rows: sum_h K[h,e] dual_h H_h  (for e in patch)
        KeT = (Ke[self.e_rows].T.tocsr()) @ sp.diags(c.h_dual[self.e_rows])
        scale = np.where(dynamic, cb[pe] / c.e_volume[pe], 0.0)
        self.e_matrix = self._csr(sp.diags(scale) @ KeT)
        self.e_keep = self._tensor(np.where(dynamic, ca[pe], 0.0))
        # Reuse validated host coefficients when diagnostics are first requested.
        self._energy_setup = (eps, K)
        # Gather/scatter plans.
        self.plan_patch_h = self._subset_plan(solver, "h", self.patch_h)
        self.plan_h_cols = self._subset_plan(solver, "e", self.h_cols)
        self.plan_patch_e = self._subset_plan(solver, "e", self.patch_e)
        self.plan_e_rows = self._subset_plan(solver, "h", self.e_rows)
        self.plan_all_e = self._subset_plan(solver, "e", np.arange(nE))
        self.plan_all_h = self._subset_plan(solver, "h", np.arange(nH))
        self.e_extra = torch.zeros(max(c.extra_e, 1), dtype=torch.float64, device=device)
        self.h_extra = torch.zeros(max(c.extra_h, 1), dtype=torch.float64, device=device)
        # Preallocated step buffers (stable addresses for graph capture).
        self.h_old = torch.zeros(len(self.patch_h), dtype=torch.float64, device=device)
        self.h_new = torch.zeros_like(self.h_old)
        self.e_cols = torch.zeros(len(self.h_cols), dtype=torch.float64, device=device)
        self.e_old = torch.zeros(len(self.patch_e), dtype=torch.float64, device=device)
        self.e_new = torch.zeros_like(self.e_old)
        self.h_rows = torch.zeros(len(self.e_rows), dtype=torch.float64, device=device)
        self.loss = torch.zeros((), dtype=torch.float64, device=device)
        self.copper = None
        self.feeds = []
        if not c.is_pec or c.feeds:
            self._init_copper(solver, eps, sigma, K, cb, pe)

    def ensure_energy(self, solver):
        """Allocate diagnostic storage only when energy sampling is requested."""
        if hasattr(self, "energy"):
            return
        eps, K = self._energy_setup
        self._init_energy(solver, eps, K)
        del self._energy_setup

    def _init_energy(self, solver, eps, K):
        """Prepare sparse device operators for the diagnostic/guard functional."""
        import scipy.sparse as sp
        from ..compiler.thin_sheet import _dual_extent

        c = self.complex
        primal = [np.asarray(getattr(solver.scene, f"d{a}_primal64")) for a in "xyz"]
        dual = [_dual_extent(p, np.arange(len(p) + 1)) for p in primal]
        solver._thin_sheet_energy_widths = {
            (a, kind): self._tensor(values[i])
            for i, a in enumerate("xyz") for kind, values in (("primal", primal), ("dual", dual))
        }

        def native_volume(kind):
            components, indices = (c.e_component, c.e_index) if kind == "e" else (c.h_component, c.h_index)
            volume = np.empty(len(indices))
            for comp in range(3):
                mask = components == comp
                ijk = indices[mask]
                extents = [primal[a] if (a == comp) == (kind == "e") else dual[a] for a in range(3)]
                volume[mask] = np.prod([extents[a][ijk[:, a]] for a in range(3)], axis=0)
            return volume

        native_eps = _gather_native(solver, tuple("eps_" + name for name in _E), c.e_component, c.e_index).numpy()
        e_mass = eps * c.e_volume
        if self.copper is not None:
            promoted = c.chord_trace[c.chord_face[:, 0] < 0] - c.statistics["active_split_faces"]
            e_mass[c.trace_dofs[promoted].reshape(-1)] = 0
        self.energy = dict(
            e_mass=self._tensor(e_mass - native_eps * native_volume("e") * (c.e_extra < 0)),
            h_mass=self._tensor(solver.mu0 * (c.h_area*c.h_dual - native_volume("h")*(c.h_extra < 0))),
            dual=self._tensor(c.h_dual), lh=self._tensor(c.h_dual/(solver.mu0*c.h_area)),
            K=self._csr(K), K_patch=self._csr(K[:, self.patch_e]),
        )
        if self.copper is not None:
            Kc = _sparse(c.k_c_rows, c.k_c_cols, c.k_c_vals,
                         (len(c.h_area) + len(c.sheet_face_area), self.chord_e.numel()))
            Kc = Kc @ sp.diags(np.repeat(~c.chord_fixed, 2).astype(float))
            root_lh = np.sqrt(c.sheet_face_dual/(solver.mu0*c.sheet_face_area_eff))
            B = (Kc[len(c.h_area):].T @ sp.diags(root_lh)).tocsr()
            self.energy.update(Kc=self._csr(Kc[:len(c.h_area)]), KcT=self._csr(Kc[:len(c.h_area)].T),
                               Ks=self._csr(Kc[len(c.h_area):]), root_lh=self._tensor(root_lh),
                               flux=_prepare_sheet_flux(self, B))

    def _tensor(self, values, dtype=torch.float64):
        return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype).to(self.device)

    def _csr(self, matrix):
        matrix = matrix.tocsr()
        matrix.sort_indices()
        index = torch.int32 if matrix.nnz < 2 ** 31 - 1 else torch.int64
        return torch.sparse_csr_tensor(self._tensor(matrix.indptr, index), self._tensor(matrix.indices, index),
                                       self._tensor(matrix.data), size=matrix.shape)

    @property
    def cumulative_loss(self):
        """Copper heat (J) dissipated so far; reading it synchronizes the device."""
        return float(self.loss)

    def _check_box(self, solver, patch_e):
        c = self.complex
        lo, hi = c.statistics["box_lower"], c.statistics["box_upper"]
        shape = (solver.Nx, solver.Ny, solver.Nz)
        index = c.e_index[patch_e]
        comps = c.e_component[patch_e]
        for axis in range(3):
            transverse = comps != axis
            low_bad = (index[:, axis] <= lo[axis]) & (lo[axis] > 0) & transverse
            high_bad = (index[:, axis] >= hi[axis]) & (hi[axis] < shape[axis] - 1) & transverse
            if low_bad.any() or high_bad.any():
                bad = index[low_bad | high_bad][:4]
                raise RuntimeError(f"Thin-sheet patch reached its local compilation box {lo}-{hi} on axis {axis}: {bad.tolist()}.")

    def _subset_plan(self, solver, kind, subset):
        """Gather/scatter plan for a DOF subset: native flat indices and extra slots."""
        c = self.complex
        comps = c.e_component if kind == "e" else c.h_component
        index = c.e_index if kind == "e" else c.h_index
        extra = c.e_extra if kind == "e" else c.h_extra
        names = _E if kind == "e" else _H
        subset = np.asarray(subset, dtype=np.int64)
        offsets = extra[subset].copy() if self._fused_kernels else None
        native = []
        for comp in range(3):
            mask = (comps[subset] == comp) & (extra[subset] < 0)
            positions = np.flatnonzero(mask)
            if not len(positions):
                continue
            shape = getattr(solver, names[comp]).shape
            ijk = index[subset[positions]]
            flat = (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]
            native.append((names[comp], self._tensor(positions, torch.int64), self._tensor(flat, torch.int64)))
            if self._fused_kernels:
                offsets[positions] = flat
        extra_positions = np.flatnonzero(extra[subset] >= 0)
        plan = dict(size=len(subset), native=native, extra_positions=self._tensor(extra_positions, torch.int64),
                    extra_slots=self._tensor(extra[subset[extra_positions]], torch.int64))
        if self._fused_kernels:
            from .cuda.backend import get_compiled_extension
            extension = get_compiled_extension()
            components = np.where(extra[subset] >= 0, 3, comps[subset])
            plan["fused"] = dict(names=names, components=self._tensor(components, torch.int32),
                                 offsets=self._tensor(offsets, torch.int64),
                                 gather=extension.thin_sheet_gather, scatter=extension.thin_sheet_scatter)
        return plan

    @staticmethod
    def gather(solver, plan, extra_buffer, out=None):
        values = torch.empty(plan["size"], dtype=torch.float64, device=extra_buffer.device) if out is None else out
        return (_gather_fused if "fused" in plan else _gather_torch)(solver, plan, extra_buffer, values)

    @staticmethod
    def scatter(solver, plan, extra_buffer, values):
        (_scatter_fused if "fused" in plan else _scatter_torch)(solver, plan, extra_buffer, values)

    def _init_copper(self, solver, eps, sigma, K, cb, pe):
        """Foil two-port arithmetic on independent chords and exact on-sheet pairs."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        from ..compiler.copper_foil import compile_copper_response
        c, dt = self.complex, float(solver.dt)
        if c.is_pec:
            rates, B, D = (torch.zeros(shape, dtype=torch.float64) for shape in ((0,), (0, 2), (2, 2)))
        else:
            c.material._frequencies((solver.source_frequency, *getattr(solver, "_requested_port_frequencies", ())))
            response = compile_copper_response(c.material)
            rates, B, D = (x.double() for x in (response.rates, response.state_space.B, response.state_space.D))
        denominator = 1 + dt * rates / 2
        decay = (1 - dt * rates / 2) / denominator
        drive = dt / 2 * B / denominator[:, None]
        effective = (D + B.T @ drive).numpy()
        count = len(c.chord_length)
        cut = c.statistics["active_split_faces"]
        pair_ids = np.array([i for i in range(len(c.trace_dofs)) if cut + i not in c.chord_trace], dtype=np.int64)
        if c.is_pec:
            gap_traces = np.concatenate([f.trace_index for f in c.feeds])
            pair_ids = pair_ids[np.isin(pair_ids + cut, gap_traces)]
        pair_dofs = c.trace_dofs[pair_ids]
        area = np.r_[c.chord_length * c.chord_width,
                     (c.trace_length[cut:] * c.trace_width[cut:])[pair_ids]]
        side_eps = np.array([np.median(eps[(c.e_side == side) & ~c.e_fixed]) for side in (-1, 1)])
        side_sigma = np.array([np.median(sigma[(c.e_side == side) & ~c.e_fixed]) for side in (-1, 1)])
        mass = c.chord_volume * side_eps
        conductance = c.chord_volume * side_sigma
        cu = self.copper = dict(
            rates=self._tensor(rates), decay=self._tensor(decay), drive=self._tensor(drive), D=self._tensor(D),
            history=self._tensor(B.T * ((1 + decay) / 2)[None, :]),
            effective=self._tensor(effective), area=self._tensor(area),
            state=torch.zeros((len(rates), len(area)), dtype=torch.float64, device=self.device),
            pair_position=self._tensor(np.searchsorted(pe, pair_dofs), torch.int64),
            mass=self._tensor(mass), conductance=self._tensor(conductance), count=count)
        local = np.broadcast_to(effective, (len(area), 2, 2)).copy()
        copper_area = area.copy()
        feed_conductance = sp.csr_matrix((2*count, 2*count))
        column_blocks = []
        for feed in c.feeds:
            binding = next(r.sheet_feed for r in solver._port_runtimes if r.port.name == feed.port_name)
            ids = np.where(feed.chord_index >= 0, feed.chord_index,
                           count + np.searchsorted(pair_ids, feed.trace_index - cut))
            copper_area[ids] = 0
            local[ids] = 0
            length, width, labels = _feed_quadrature(c, feed)
            columns = []
            for label in np.unique(labels[labels >= 0]):
                members = np.flatnonzero(labels == label)
                traces = ids[members]
                u = np.repeat(length[members] * feed.chord_projection[members] / 2, 2)
                share = np.sum(length[members] * width[members]) / feed.length
                g = float(share / (float(binding.resistance) * feed.width))
                block = g * np.outer(u, u)
                dofs = (2*traces[:, None] + np.arange(2)).reshape(-1)
                column = dict(ids=self._tensor(traces, torch.int64), u=self._tensor(u),
                              conductance=g, kind="chord" if traces[0] < count else "pair")
                # Along the Cartesian feed axis a planar column has one trace kind.
                if column["kind"] == "chord":
                    rows, cols = np.meshgrid(dofs, dofs, indexing="ij")
                    feed_conductance += _sparse(rows.reshape(-1), cols.reshape(-1), block.reshape(-1),
                                                (2*count, 2*count))
                    column_blocks.append(dofs)
                    column["dofs"] = self._tensor(dofs, torch.int64)
                else:
                    pair = traces - count
                    pair_dof = pair_dofs[pair].reshape(-1)
                    capacity = eps[pair_dof] * c.e_volume[pair_dof] / dt
                    loss = sigma[pair_dof] * c.e_volume[pair_dof] / 2
                    column.update(pair=self._tensor(pair, torch.int64),
                                  inverse=self._tensor(np.linalg.inv(np.diag(capacity + loss) + block/2)),
                                  rhs=self._tensor(np.diag(capacity - loss) - block/2))
                columns.append(column)
            binding.column_voltage = torch.zeros(len(columns), dtype=torch.float64, device=self.device)
            binding.column_conductance = self._tensor([column["conductance"] for column in columns])
            frequencies = next(r.frequencies for r in solver._port_runtimes if r.port.name == feed.port_name)
            binding.column_sum = torch.zeros((len(frequencies), len(columns)), dtype=torch.complex128, device=self.device)
            binding.column_term = torch.empty_like(binding.column_sum)
            self.feeds.append(dict(binding=binding, geometry=feed, ids=self._tensor(ids, torch.int64),
                                   length=self._tensor(length), width=self._tensor(width), columns=columns))
        self.feed_area = self._tensor(area[:count])
        if c.feeds:
            cu["area"] = self._tensor(copper_area)
            cu["local"] = self._tensor(local)
            cu["feed_conductance"] = self._csr(feed_conductance)
        # The aligned path uses exactly the foil's per-area 2x2 CN matrices.
        capacity = eps[pair_dofs] * c.e_volume[pair_dofs] / area[count:, None]
        loss = sigma[pair_dofs] * c.e_volume[pair_dofs] / area[count:, None]
        lhs = np.broadcast_to(effective / 2, (len(pair_ids), 2, 2)).copy()
        if c.feeds:
            lhs = local[count:].copy() / 2
        rhs = -lhs.copy()
        for side in range(2):
            lhs[:, side, side] += capacity[:, side] / dt + loss[:, side] / 2
            rhs[:, side, side] += capacity[:, side] / dt - loss[:, side] / 2
        cu.update(inverse=self._tensor(np.linalg.inv(lhs)), rhs_matrix=self._tensor(rhs),
                  pair_forcing=self._csr((K.T @ sp.diags(c.h_dual))[pair_dofs.reshape(-1)][:, self.e_rows]),
                  pair_area=self._tensor(area[count:]))
        promoted = np.flatnonzero(c.chord_face[:, 0] < 0)
        promoted_dofs = c.trace_dofs[c.chord_trace[promoted] - cut]
        cu.update(promoted_chords=self._tensor(promoted, torch.int64),
                  promoted_position=self._tensor(np.searchsorted(pe, promoted_dofs), torch.int64))
        self.chord_e = torch.zeros((count, 2), dtype=torch.float64, device=self.device)
        self.chord_e_prev = torch.zeros_like(self.chord_e)
        self.chord_f = torch.zeros_like(self.chord_e)
        self.x_prev = torch.zeros_like(self.e_old)
        nc = 2 * count
        Kc = _sparse(c.k_c_rows, c.k_c_cols, c.k_c_vals,
                     (len(c.h_area) + len(c.sheet_face_area), nc))
        active = np.repeat(~c.chord_fixed, 2).astype(float)
        Kc = Kc @ sp.diags(active)
        Lh = np.r_[c.h_dual / (solver.mu0 * c.h_area),
                   c.sheet_face_dual / (solver.mu0 * c.sheet_face_area_eff)]
        Acc = (Kc.T @ sp.diags(Lh) @ Kc).tocsr()
        Acx = (Kc[:len(c.h_area)].T @ sp.diags(Lh[:len(c.h_area)]) @ K[:, pe]).tocsr()
        Q = (sp.diags(np.where(~c.e_fixed[pe], cb[pe] / c.e_volume[pe], 0)) @ Acx.T * dt / 4).tocsr()
        S = (sp.diags((mass / dt + conductance / 2).reshape(-1))
             + (sp.block_diag(list(area[:count, None, None] * local[:count] / 2), format="csr")
                if c.feeds and count else sp.kron(sp.diags(area[:count]), sp.csr_matrix(effective / 2), format="csr"))
             + feed_conductance/2 + dt / 4 * (Acc - Acx @ Q)).tocsr()
        cu.update(Acc=self._csr(Acc), Acx=self._csr(Acx), Q=self._csr(Q), S=self._csr(S),
                  h_chord=self._csr(sp.diags(dt / (solver.mu0 * c.h_area[self.patch_h])) @ Kc[self.patch_h]),
                  held=self._tensor(active.reshape(-1, 2)))
        if count:
            if c.feeds:
                grouped = {int(dof)//2 for block in column_blocks for dof in block}
                groups = column_blocks + [np.array([2*i, 2*i+1]) for i in range(count) if i not in grouped]
                rows, cols, values = [], [], []
                for dofs in groups:
                    block = S[dofs][:, dofs].toarray()
                    eigenvalues, vectors = np.linalg.eigh(block)
                    root = (vectors * (1/np.sqrt(eigenvalues))[None, :]) @ vectors.T
                    rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
                    rows.extend(rr.reshape(-1)); cols.extend(cc.reshape(-1)); values.extend(root.reshape(-1))
                R = _sparse(rows, cols, values, S.shape)
            else:
                blocks = np.stack([S[2*i:2*i+2, 2*i:2*i+2].toarray() for i in range(count)])
                values, vectors = np.linalg.eigh(blocks)
                roots = (vectors * (1 / np.sqrt(values))[:, None, :]) @ vectors.transpose(0, 2, 1)
                R = sp.block_diag(list(roots), format="csr")
            scaled = (R @ S @ R).tocsr()
            rng = np.random.default_rng(0)
            low = float(eigsh(scaled, k=1, which="SA", v0=rng.standard_normal(nc), tol=1e-10,
                              return_eigenvectors=False)[0])
            high = float(eigsh(scaled, k=1, which="LA", v0=rng.standard_normal(nc), tol=1e-10,
                               return_eigenvectors=False)[0])
            a, b = .98 * low, 1.02 * high
            degree = _chebyshev_degree(b / a, 1e-12)
            cu.update(R=self._csr(R), chebyshev=(a, b, degree), spectrum=(low, high),
                      error_bound=1 / math.cosh(degree * math.acosh((b+a)/(b-a))) ** 2)
        if self._fused_kernels:
            self._init_copper_fused()

    def _init_copper_fused(self):
        from .cuda.backend import get_compiled_extension
        extension = get_compiled_extension()
        cu = self.copper
        traces, count = len(cu["area"]), cu["count"]

        def csr(matrix):
            return matrix.crow_indices().to(torch.int32), matrix.col_indices().to(torch.int32), matrix.values()

        promoted = torch.full_like(self.chord_e, -1, dtype=torch.int64)
        promoted[cu["promoted_chords"]] = cu["promoted_position"]
        cu["fused"] = dict(
            acx=csr(cu["Acx"]), acc=csr(cu["Acc"]),
            acx_lanes=_sheet_csr_lanes(cu["Acx"]), acc_lanes=_sheet_csr_lanes(cu["Acc"]),
            feed=csr(cu["feed_conductance"]) if self.feeds else (None, None, None),
            rhs_kernel=extension.thin_sheet_copper_rhs,
            forcing_kernel=extension.thin_sheet_chord_forcing,
            state_kernel=extension.thin_sheet_copper_state,
            history=torch.empty((2, traces), dtype=torch.float64, device=self.device),
            rhs=torch.empty(2 * count, dtype=torch.float64, device=self.device),
            coupled=torch.empty(2 * count, dtype=torch.float64, device=self.device),
            midpoint=torch.empty((traces, 2), dtype=torch.float64, device=self.device),
            feed_mask=cu["area"] == 0, promoted=promoted,
            partials=torch.empty((max(traces, self.e_old.numel()) + 127) // 128,
                                 dtype=torch.float64, device=self.device),
            counter=torch.zeros((), dtype=torch.int32, device=self.device))
        if count:
            a, b, degree = cu["chebyshev"]
            theta, delta = (b+a)/2, (b-a)/2
            sigma = theta / delta
            rho = 1 / sigma
            coefficients = []
            for _ in range(degree - 1):
                rho_next = 1 / (2 * sigma - rho)
                coefficients.append((rho_next * rho, 2 * rho_next / delta))
                rho = rho_next
            cu["fused"]["chord_solve"] = dict(
                R=csr(cu["R"]), S=csr(cu["S"]), degree=degree,
                R_lanes=_sheet_csr_lanes(cu["R"]), S_lanes=_sheet_csr_lanes(cu["S"]),
                inverse_theta=1 / theta, coefficients=tuple(coefficients),
                initialize_kernel=extension.thin_sheet_chord_initialize,
                matvec_kernel=extension.thin_sheet_chord_matvec,
                direction_kernel=extension.thin_sheet_chord_direction,
                residual_kernel=extension.thin_sheet_chord_residual,
                residual=torch.empty(2 * count, dtype=torch.float64, device=self.device),
                direction=torch.empty(2 * count, dtype=torch.float64, device=self.device),
                temporary=torch.empty(2 * count, dtype=torch.float64, device=self.device))

    def begin(self, solver):
        self.gather(solver, self.plan_patch_h, self.h_extra, self.h_old)

    def magnetic(self, solver):
        e_cols = self.gather(solver, self.plan_h_cols, self.e_extra, self.e_cols)
        torch.mv(self.h_matrix, e_cols, out=self.h_new)
        self.h_new.neg_().addcmul_(self.h_decay, self.h_old)
        if self.copper is not None:
            predictor = (2 * self.chord_e + self.chord_e_prev).reshape(-1) / 4
            self.h_new.sub_(torch.mv(self.copper["h_chord"], predictor))
        self.scatter(solver, self.plan_patch_h, self.h_extra, self.h_new)
        self.gather(solver, self.plan_patch_e, self.e_extra, self.e_old)

    def electric(self, solver):
        h_rows = self.gather(solver, self.plan_e_rows, self.h_extra, self.h_rows)
        old, new = self.e_old, self.e_new
        torch.mv(self.e_matrix, h_rows, out=new)
        new.addcmul_(self.e_keep, old)
        if self.copper is not None:
            cu = self.copper
            count = cu["count"]
            history, rhs = (_copper_rhs_fused if "fused" in cu else _copper_rhs_torch)(
                self, solver, old, new)
            current, previous = self.chord_e.reshape(-1), self.chord_e_prev.reshape(-1)
            extrapolated = 2 * current - previous
            updated = extrapolated + self.chord_solve(rhs - torch.mv(cu["S"], extrapolated))
            updated = updated * cu["held"].reshape(-1)
            new.sub_(torch.mv(cu["Q"], updated))
            (_chord_forcing_fused if "fused" in cu else _chord_forcing_torch)(
                self, solver, old, new, updated)
            self.h_new.sub_(torch.mv(cu["h_chord"], updated / 4))
            self.scatter(solver, self.plan_patch_h, self.h_extra, self.h_new)
            position = cu["pair_position"]
            pair_old = old[position]
            forcing = torch.mv(cu["pair_forcing"], h_rows).reshape(-1, 2) / cu["pair_area"][:, None]
            pair_rhs = torch.einsum("tij,tj->ti", cu["rhs_matrix"], pair_old) + forcing - history[count:]
            pair_new = torch.einsum("tij,tj->ti", cu["inverse"], pair_rhs)
            for feed in self.feeds:
                for column in feed["columns"]:
                    if column["kind"] == "pair":
                        pair = column["pair"]
                        raw_forcing = (forcing[pair] * cu["pair_area"][pair, None]).reshape(-1)
                        source = column["conductance"] * column["u"] * feed["binding"].drive
                        coupled_rhs = torch.mv(column["rhs"], pair_old[pair].reshape(-1)) + raw_forcing + source
                        pair_new[pair] = torch.mv(column["inverse"], coupled_rhs).reshape(-1, 2)
            new[position.reshape(-1)] = pair_new.reshape(-1)
            (_copper_state_fused if "fused" in cu else _copper_state_torch)(
                self, solver, old, new, updated, pair_old, pair_new)
        self.scatter(solver, self.plan_patch_e, self.e_extra, new)

    def chord_solve(self, rhs):
        """Even Chebyshev residual s_k(mu)^2: 0 < P <= S^-1, with block Jacobi."""
        return (_chord_solve_fused if "fused" in self.copper else _chord_solve_torch)(self, rhs)

    def state_tensors(self):
        """Mutable private state included in graph-capture snapshots."""
        tensors = [self.e_extra, self.h_extra, self.loss]
        if self.copper is not None:
            tensors.extend((self.copper["state"], self.chord_e, self.chord_e_prev,
                            self.chord_f, self.x_prev))
        for feed in self.feeds:
            binding = feed["binding"]
            tensors.extend((binding.drive, binding.voltage, binding.current, binding.column_voltage, binding.column_sum))
        return tensors

    def stored_copper_energy(self):
        if self.copper is None:
            return 0.0
        return 0.5 * float((self.copper["state"].square() * self.copper["area"]).sum())


def initialize_thin_sheets(solver):
    solver._thin_sheets = ()
    complexes = getattr(solver, "_thin_sheet_complexes", ())
    if not complexes:
        return
    runtimes = tuple(_SheetRuntime(solver, complex_) for complex_ in complexes)
    owned = {}
    for runtime in runtimes:
        c = runtime.complex
        for dof in runtime.patch_e:
            key = (int(c.e_component[dof]), *map(int, c.e_index[dof]))
            if owned.setdefault(key, id(runtime)) != id(runtime):
                raise ValueError("Separate thin sheets share a Yee patch; refine the grid or separate the sheets.")
    _reject_overlaps(solver, runtimes)
    solver._thin_sheets = runtimes


def _reject_overlaps(solver, runtimes):
    patch = {name: set() for name in (*_E, *_H)}
    for runtime in runtimes:
        c = runtime.complex
        for dof in runtime.patch_e:
            patch[_E[int(c.e_component[dof])]].add(tuple(map(int, c.e_index[dof])))
    for runtime in runtimes:
        c = runtime.complex
        for dof in runtime.patch_h:
            patch[_H[int(c.h_component[dof])]].add(tuple(map(int, c.h_index[dof])))
    for term in (*getattr(solver, "_source_terms", ()), *getattr(solver, "_electric_source_terms", ()),
                 *getattr(solver, "_magnetic_source_terms", ())):
        name = term["field_name"].removesuffix("_imag")
        if not patch.get(name):
            continue
        offsets = np.array(term["offsets"])
        for key in ("patch", "cw_cos_patch", "cw_sin_patch", "coeff_patch"):
            values = term.get(key)
            if values is None:
                continue
            nonzero = torch.nonzero(values != 0).cpu().numpy() + offsets
            if any(tuple(map(int, point)) in patch[name] for point in nonzero):
                raise NotImplementedError("A source overlaps a conformal thin-sheet patch; move the source away from the sheet.")
    for entry in getattr(solver, "_copper_foils", ()):
        name = entry["electric_name"].removesuffix("_imag")
        for point in entry["indices"].cpu().numpy():
            if tuple(map(int, point)) in patch[name]:
                raise NotImplementedError("A conformal thin sheet overlaps an axis-aligned CopperFoil interface.")
    cut = getattr(solver, "_cut_surface", None)
    if cut is not None:
        for name, group in cut["groups"].items():
            for point in group["indices"].cpu().numpy():
                if tuple(map(int, point)) in patch[name]:
                    raise NotImplementedError("A conformal thin sheet overlaps a PEC cut-surface constraint.")
    if getattr(solver, "_surface_impedance", None):
        raise NotImplementedError("Conformal thin sheets do not yet support surface-impedance boundaries in the same scene.")
    _warn_side_observers(solver, runtimes)
    if getattr(solver, "_lumped_element_runtimes", ()):
        raise NotImplementedError("Conformal thin sheets do not yet support lumped elements in the same scene.")


def _warn_side_observers(solver, runtimes):
    """Observers on split unknowns read the native slot, i.e. the -1 side only."""
    split = set()
    for runtime in runtimes:
        c = runtime.complex
        for dof in np.flatnonzero(c.e_split):
            split.add((_E[int(c.e_component[dof])], *map(int, c.e_index[dof])))
        for dof in np.flatnonzero(c.h_extra >= 0):
            split.add((_H[int(c.h_component[dof])], *map(int, c.h_index[dof])))
    if not split:
        return
    from .observers import _field_name, resolve_point_observer
    touched = []
    for observer in (*getattr(solver, "observers", ()), *getattr(solver, "time_observers", ())):
        if observer.get("kind") not in ("point", "field_time") or observer.get("region_kind", "point") != "point":
            continue
        components = observer.get("components") or (observer.get("component"),)
        for component in components:
            if component is None:
                continue
            index = resolve_point_observer(solver, {"component": component, "position": observer["position"]})
            if (_field_name(component), *index) in split:
                touched.append(str(observer.get("name")))
    if touched:
        import warnings
        warnings.warn(f"Monitors {sorted(set(touched))} sample split thin-sheet unknowns and read only the sheet's "
                      "-1 side (the side opposite the SurfaceMesh normal).", RuntimeWarning, stacklevel=3)


def begin_thin_sheet_step(solver):
    for runtime in getattr(solver, "_thin_sheets", ()):
        runtime.begin(solver)


def advance_thin_sheet_magnetic(solver):
    for runtime in getattr(solver, "_thin_sheets", ()):
        runtime.magnetic(solver)


def advance_thin_sheet_electric(solver):
    for runtime in getattr(solver, "_thin_sheets", ()):
        if not runtime.feeds:
            runtime.electric(solver)


def _sheet_flux_factor(B):
    """Factor the full-row least-squares map on its column range, on the host.

    The sheet incidence has at most two faces per chord row. Unanchored
    connected components can have one weighted constant null vector. Pin one
    column in each such component, then remove that null component after the
    solve. No forcing rows are discarded, including inconsistent rows.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components, breadth_first_order
    from scipy.sparse.linalg import splu

    B = B.tocsr(copy=True)
    B.eliminate_zeros()
    B.sort_indices()
    if np.any(np.diff(B.indptr) > 2):
        raise ValueError("Sheet flux incidence requires at most two faces per chord row.")
    normal = (B.T @ B).tocsr()
    count = B.shape[1]
    _, labels = connected_components(normal, directed=False)
    single = np.flatnonzero(np.diff(B.indptr) == 1)
    anchored = np.unique(labels[B.indices[B.indptr[single]]])
    free_components = np.setdiff1d(np.unique(labels), anchored)
    null_rows, null_cols, null_values, pins = [], [], [], []
    for component in free_components:
        columns = np.flatnonzero(labels == component)
        local = B[:, columns].tocsr()
        rows = np.flatnonzero(np.diff(local.indptr) == 2)
        a, b = local.indices[local.indptr[rows]], local.indices[local.indptr[rows]+1]
        va, vb = local.data[local.indptr[rows]], local.data[local.indptr[rows]+1]
        # Parallel chord constraints have the same ratio; one tree is enough
        # to construct a candidate, and every row verifies it below.
        graph = sp.coo_matrix((np.r_[-va/vb, -vb/va], (np.r_[a, b], np.r_[b, a])),
                              shape=(len(columns), len(columns))).tocsr()
        graph.sum_duplicates()
        multiplicity = sp.coo_matrix((np.ones(2*len(a)), (np.r_[a,b], np.r_[b,a])),
                                     shape=graph.shape).tocsr()
        graph.data /= multiplicity.data
        order, parent = breadth_first_order(graph, 0, directed=False)
        vector = np.ones(len(columns))
        for child in order[1:]:
            vector[child] = vector[parent[child]] * graph[parent[child], child]
        error = np.linalg.norm(local @ vector)
        scale = np.linalg.norm(local.data) * np.linalg.norm(vector)
        if error > 64*np.finfo(np.float64).eps*scale:
            continue
        vector /= np.linalg.norm(vector)
        null_rows.extend(columns)
        null_cols.extend([len(pins)]*len(columns))
        null_values.extend(vector)
        pins.append(columns[0])
    null = sp.coo_matrix((null_values, (null_rows, null_cols)), shape=(count, len(pins))).tocsr()
    columns = np.delete(np.arange(count), pins)
    if len(columns):
        # A symmetric minimum-degree ordering limits fill. With pivoting off,
        # sparse LU of this SPD matrix is LDL^T; absorb sqrt(D) into L.
        lu = splu(normal[columns][:, columns].tocsc(), permc_spec="MMD_AT_PLUS_A",
                  diag_pivot_thresh=0., options={"SymmetricMode": True})
        diagonal = lu.U.diagonal()
        if not np.array_equal(lu.perm_r, lu.perm_c) or np.any(diagonal <= 0):
            raise RuntimeError("Sheet flux range factorization lost positive definiteness.")
        columns = columns[np.argsort(lu.perm_c)]
        lower = (lu.L @ sp.diags(np.sqrt(diagonal))).tocsr()
    else:
        lower = sp.csr_matrix((0, 0), dtype=np.float64)
    lower.sort_indices()
    return dict(L=lower, LT=lower.T.tocsr(), BT=B[:, columns].T.tocsr(),
                columns=columns, null=null, nullT=null.T.tocsr(),
                faces=count, rank=len(columns), normal_nnz=normal.nnz)


def _prepare_sheet_flux(runtime, B):
    """Store the fp64 range factors and fixed row widths on the runtime device."""
    factor = _sheet_flux_factor(B)
    plan = {key: runtime._csr(factor[key]) for key in ("L", "LT", "BT")}
    plan["columns"] = runtime._tensor(factor["columns"], torch.int64)
    plan["faces"] = factor["faces"]
    plan["rank"] = factor["rank"]
    plan["normal_nnz"] = factor["normal_nnz"]
    if factor["null"].shape[1]:
        plan["null"] = runtime._csr(factor["null"])
        plan["nullT"] = runtime._csr(factor["nullT"])
    if runtime.device.type == "cuda":
        plan["row_widths"] = {
            key: int(np.diff(factor[key].indptr).max(initial=0))
            for key in ("BT", "null", "nullT")
        }
    return plan


def _sheet_flux_row_sum(matrix, width, value):
    """Fixed row reductions without shared floating-point accumulators.

    Width is host metadata prepared with the CSR. Padding exists only in
    per-sample scratch; CSR values and indices remain the persistent owners.
    """
    if width == 0:
        return value.new_zeros(matrix.shape[0])
    pointers = matrix.crow_indices()
    lanes = torch.arange(width, device=value.device, dtype=pointers.dtype)
    offsets = pointers[:-1, None] + lanes[None, :]
    valid = offsets < pointers[1:, None]
    offsets = offsets.clamp(max=matrix._nnz()-1).reshape(-1)
    columns = matrix.col_indices().index_select(0, offsets)
    weights = matrix.values().index_select(0, offsets)
    products = (weights*value.index_select(0, columns)).reshape(matrix.shape[0], width)
    return torch.where(valid, products, 0.).sum(dim=1)


def _sheet_flux(plan, forcing):
    """Full-row minimum-norm least squares with two sparse triangular solves.

    This is B^T (B B^T)^+ f evaluated on the column range: for full column
    rank it is (B^T B)^-1 B^T f. The pinned solve plus orthogonal nullspace
    projection gives the same minimum-norm answer for closed sheet loops.
    The sparse CSR triangular solves run on the forcing's stream (cuSPARSE on
    CUDA, MKL on CPU); sampling never reads the host.
    """
    rhs = (_sheet_flux_row_sum(plan["BT"], plan["row_widths"]["BT"], forcing)
           if forcing.is_cuda else torch.mv(plan["BT"], forcing))
    if plan["rank"]:
        rhs = torch.triangular_solve(rhs[:, None], plan["L"], upper=False).solution
        rhs = torch.triangular_solve(rhs, plan["LT"], upper=True).solution[:, 0]
    z = forcing.new_zeros(plan["faces"])
    z.index_copy_(0, plan["columns"], rhs)
    if "null" in plan:
        if z.is_cuda:
            coefficients = _sheet_flux_row_sum(plan["nullT"], plan["row_widths"]["nullT"], z)
            z = z - _sheet_flux_row_sum(plan["null"], plan["row_widths"]["null"], coefficients)
        else:
            z = z - torch.mv(plan["null"], torch.mv(plan["nullT"], z))
    return z


@torch.no_grad()
def thin_sheet_energy_device(solver, *, modified=False):
    """Device float64 equivalent of :func:`thin_sheet_energy`, without host reads.

    Sample after completed step n: native/split E and copper modes are at n,
    native/split H at n-1/2, and chord_e/chord_e_prev at n/n-1. Reconstruct
    the sheet flux from the same chord_f, E_new-E_old pairing as the reference.
    With modified=True, shift H to integer time and use M-dt^2 A_xx/4,
    including the chord average (c^n+c^(n-1))/2. This is the qualified Newmark
    functional, rather than a plain E/H norm of the additional sheet buffers.
    Geometry, masses and sparse operators are prepared once on the host.
    """
    from .runtime.passivity import _yee_energy

    for runtime in solver._thin_sheets:
        if not hasattr(runtime, "energy"):
            runtime.ensure_energy(solver)
    widths = solver._thin_sheet_energy_widths
    total = _yee_energy(solver, magnetic_mass=solver.mu0, widths=widths)
    dt = float(solver.dt)
    circulation = []
    if modified:
        fields = [getattr(solver, name).double() for name in _E]
        for a, b, d in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            shape_b, shape_d, shape_a = [1]*3, [1]*3, [1]*3
            shape_b[b] = widths["xyz"[b], "primal"].numel()
            shape_d[d] = widths["xyz"[d], "primal"].numel()
            shape_a[a] = widths["xyz"[a], "dual"].numel()
            curl = (torch.diff(fields[d], dim=b)*widths["xyz"[d], "primal"].view(shape_d)
                    - torch.diff(fields[b], dim=d)*widths["xyz"[b], "primal"].view(shape_b))
            circulation.append(curl)
            total = total - dt/2 * (widths["xyz"[a], "dual"].view(shape_a)
                                    * getattr(solver, _H[a]).double()*curl).sum()
    for runtime in solver._thin_sheets:
        p, cu = runtime.energy, runtime.copper
        e = runtime.gather(solver, runtime.plan_all_e, runtime.e_extra)
        h = runtime.gather(solver, runtime.plan_all_h, runtime.h_extra)
        total = total + .5*(p["e_mass"]*e.square()).sum() + .5*(p["h_mass"]*h.square()).sum()
        if modified:
            # Replace the native cross term by the patch's Newmark invariant.
            for name, positions, flat in runtime.plan_all_h["native"]:
                curl = circulation[_H.index(name)].reshape(-1).index_select(0, flat)
                total = total + dt/2*(p["dual"][positions]*h[positions]*curl).sum()
        gx = torch.mv(p["K"], e)
        shift = dt/2*gx
        if cu is not None:
            current = runtime.chord_e.reshape(-1)
            pair = current + runtime.chord_e_prev.reshape(-1)
            total = total + .5*(cu["mass"].reshape(-1)*current.square()).sum()
            total = total + .5*(cu["state"].square()*cu["area"]).sum()
            shift = shift + dt/4*torch.mv(p["Kc"], pair)
            dx = runtime.e_new - runtime.e_old
            forcing = (runtime.chord_f.reshape(-1) - torch.mv(p["KcT"], p["dual"]*h)
                       + dt/4*torch.mv(p["KcT"], p["lh"]*torch.mv(p["K_patch"], dx)))
            z = _sheet_flux(p["flux"], forcing)
            if modified:
                z = z - dt/4*p["root_lh"]*torch.mv(p["Ks"], pair)
            total = total + .5*z.square().sum()
        if modified:
            total = total - (p["dual"]*h*shift).sum() + .5*(p["lh"]*shift.square()).sum()
            total = total - dt**2/8*(p["lh"]*gx.square()).sum()
    return total.detach()


@torch.no_grad()
def thin_sheet_energy(solver, *, modified=False):
    """Physical or modified field energy (J), including chords and copper storage.

    ``modified=True`` uses the staggered Newmark invariant: electric mass
    M - dt^2 A_xx/4 and the magnetic field transformed to the integer time:
    Hhat = H - dt K_x E/(2 mu A) - dt K_c(c + c_prev)/(4 mu A).
    Sheet-face flux is recovered from F in its generated curl range; it is
    never stored or advanced as an additional runtime field.
    """
    scene = solver.scene
    primal = [np.asarray(getattr(scene, f"d{a}_primal64")) for a in "xyz"]
    from ..compiler.thin_sheet import _dual_extent
    dual = [_dual_extent(p, np.arange(len(p) + 1)) for p in primal]
    total = 0.0
    dt = float(solver.dt)
    native_circulation = []
    if modified:
        fields = [getattr(solver, name).double().cpu().numpy() for name in _E]
        for a, b, d in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            shape_b, shape_d = [1]*3, [1]*3
            shape_b[b], shape_d[d] = len(primal[b]), len(primal[d])
            native_circulation.append(np.diff(fields[d], axis=b)*primal[d].reshape(shape_d)
                                      - np.diff(fields[b], axis=d)*primal[b].reshape(shape_b))
    for comp, name in enumerate(_E):
        extents = [dual[a] if a != comp else primal[a] for a in range(3)]
        field = getattr(solver, name)
        volume = torch.as_tensor(np.einsum("i,j,k->ijk", *extents), device=field.device)
        eps = getattr(solver, "eps_" + name).double()
        total += 0.5 * float((eps * volume * field.double().square()).sum())
    for comp, name in enumerate(_H):
        extents = [primal[a] if a != comp else dual[a] for a in range(3)]
        field = getattr(solver, name)
        volume = torch.as_tensor(np.einsum("i,j,k->ijk", *extents), device=field.device)
        total += 0.5 * solver.mu0 * float((volume * field.double().square()).sum())
        if modified:
            extent = [1]*3
            extent[comp] = len(dual[comp])
            total -= dt/2 * np.sum(dual[comp].reshape(extent)*field.double().cpu().numpy()*native_circulation[comp])
    for runtime in getattr(solver, "_thin_sheets", ()):
        c = runtime.complex
        eps, _ = _side_materials(solver, c)
        native_eps = _gather_native(solver, tuple("eps_" + name for name in _E), c.e_component, c.e_index).numpy()
        e = runtime.gather(solver, runtime.plan_all_e, runtime.e_extra).cpu().numpy()
        h = runtime.gather(solver, runtime.plan_all_h, runtime.h_extra).cpu().numpy()
        native_e = c.e_extra < 0
        native_h = c.h_extra < 0
        # Replace native-volume contributions of every complex unknown by its own mass.
        e_native_volume = np.empty(len(e))
        for comp in range(3):
            m = c.e_component == comp
            idx = c.e_index[m]
            ext = [dual[a][idx[:, a]] if a != comp else primal[a][idx[:, a]] for a in range(3)]
            e_native_volume[m] = ext[0] * ext[1] * ext[2]
        h_native_volume = np.empty(len(h))
        for comp in range(3):
            m = c.h_component == comp
            idx = c.h_index[m]
            ext = [primal[a][idx[:, a]] if a != comp else dual[a][idx[:, a]] for a in range(3)]
            h_native_volume[m] = ext[0] * ext[1] * ext[2]
        total -= 0.5 * float(np.sum((native_eps * e_native_volume * e ** 2)[native_e]))
        total += 0.5 * float(np.sum(eps * c.e_volume * e ** 2))
        total -= 0.5 * solver.mu0 * float(np.sum((h_native_volume * h ** 2)[native_h]))
        total += 0.5 * solver.mu0 * float(np.sum(c.h_area * c.h_dual * h ** 2))
        total += runtime.stored_copper_energy()
        if modified:
            # Remove the native Yee cross term before inserting the patch invariant.
            for comp in range(3):
                mask = native_h & (c.h_component == comp)
                ijk = c.h_index[mask]
                total += dt/2 * np.sum(c.h_dual[mask]*h[mask]*native_circulation[comp][tuple(ijk.T)])
        Kx = (_sparse(c.k_rows, c.k_cols, c.k_vals, (len(h), len(e))) if c.is_pec and not c.feeds else
              _sparse(c.k_x_rows, c.k_x_cols, c.k_x_vals, (len(h), len(e))))
        gx = Kx @ e
        lh = c.h_dual/(solver.mu0*c.h_area)
        shift = dt/2 * gx
        if runtime.copper is not None:
            import scipy.sparse as sp
            from scipy.sparse.linalg import lsqr
            cu = runtime.copper
            nc = runtime.chord_e.numel()
            Kc = _sparse(c.k_c_rows, c.k_c_cols, c.k_c_vals, (len(h)+len(c.sheet_face_area), nc))
            Kc = Kc @ sp.diags(np.repeat(~c.chord_fixed, 2).astype(float))
            current = runtime.chord_e.cpu().numpy().reshape(-1)
            previous = runtime.chord_e_prev.cpu().numpy().reshape(-1)
            total += .5*np.sum(cu["mass"].cpu().numpy().reshape(-1)*current**2)
            promoted = c.chord_trace[c.chord_face[:, 0] < 0] - c.statistics["active_split_faces"]
            dofs = c.trace_dofs[promoted].reshape(-1)
            total -= .5*np.sum(eps[dofs]*c.e_volume[dofs]*e[dofs]**2)
            shift += dt/4 * (Kc[:len(h)] @ (current+previous))
            sheet_lh = c.sheet_face_dual/(solver.mu0*c.sheet_face_area_eff)
            # z = sqrt(mu A dual) H_sheet. Its generated curl range gives the
            # unique minimum-energy flux, including loops with dependent rows.
            B = Kc[len(h):].T @ sp.diags(np.sqrt(sheet_lh))
            dx = runtime.e_new.cpu().numpy() - runtime.e_old.cpu().numpy()
            acx_dx = Kc[:len(h)].T @ (lh*(Kx[:, runtime.patch_e] @ dx))
            # F = Kc_sub^T dual H - dt Acx (E_new-E_old)/4 + B z.
            forcing = runtime.chord_f.cpu().numpy().reshape(-1) - Kc[:len(h)].T @ (c.h_dual*h) + dt/4*acx_dx
            z = lsqr(B, forcing, atol=1e-13, btol=1e-13, iter_lim=4*len(sheet_lh))[0] if len(sheet_lh) else np.empty(0)
            if modified:
                z -= dt/4*np.sqrt(sheet_lh)*(Kc[len(h):] @ (current+previous))
            total += .5*np.dot(z, z)
        if modified:
            total += -np.sum(c.h_dual*h*shift) + .5*np.sum(lh*shift**2) - dt**2/8*np.sum(lh*gx**2)
    return float(total)
