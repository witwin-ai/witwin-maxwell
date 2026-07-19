from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

from ..compiler.electrostatic import CompiledElectrostatics, TerminalConstraint, compile_electrostatics
from .api import ElectrostaticBoundarySpec, ElectrostaticSolverConfig

_AXES = ("x", "y", "z")


def _slab_full(slab: torch.Tensor, axis: int, side: str, axis_len: int) -> torch.Tensor:
    """Embed a boundary slab (size 1 along ``axis``) into a full-size tensor.

    Out-of-place so gradients flow through the boundary permittivity.
    """
    other = axis_len - 1
    if other == 0:
        return slab
    zeros_shape = list(slab.shape)
    zeros_shape[axis] = other
    zeros = torch.zeros(zeros_shape, dtype=slab.dtype, device=slab.device)
    if side == "low":
        return torch.cat([slab, zeros], dim=axis)
    return torch.cat([zeros, slab], dim=axis)


class ElectrostaticOperator:
    """Matrix-free cell-centred finite-volume Laplacian ``-div(eps grad phi)``.

    Face conductances use the harmonic-mean interface permittivity; boundary
    Dirichlet faces add a half-cell ghost conductance. The operator is symmetric
    positive (semi-)definite. All tensors are on the scene device in the solver
    dtype.
    """

    def __init__(self, compiled: CompiledElectrostatics):
        self.compiled = compiled
        self.device = compiled.device
        self.dtype = compiled.dtype
        self.shape = compiled.shape
        self.eps0 = compiled.eps0
        eps = compiled.epsilon_r
        hx, hy, hz = compiled.hx, compiled.hy, compiled.hz
        dxc, dyc, dzc = compiled.dxc, compiled.dyc, compiled.dzc

        area_x = (hy[None, :, None] * hz[None, None, :])
        area_y = (hx[:, None, None] * hz[None, None, :])
        area_z = (hx[:, None, None] * hy[None, :, None])

        self.gx = self.eps0 * _harmonic(eps[:-1, :, :], eps[1:, :, :]) * area_x / dxc[:, None, None]
        self.gy = self.eps0 * _harmonic(eps[:, :-1, :], eps[:, 1:, :]) * area_y / dyc[None, :, None]
        self.gz = self.eps0 * _harmonic(eps[:, :, :-1], eps[:, :, 1:]) * area_z / dzc[None, None, :]

        # Boundary Dirichlet ghost conductances: eps at the boundary cell over a
        # half-cell distance to the domain face. Neumann/symmetry faces contribute
        # nothing (zero normal displacement).
        self._boundary: list[dict[str, Any]] = []
        axis_area = {0: area_x, 1: area_y, 2: area_z}
        half_cell = {0: hx, 1: hy, 2: hz}
        for axis, name in enumerate(_AXES):
            for side in ("low", "high"):
                kind, value = compiled.boundary.face(name, side)
                if kind != "dirichlet":
                    continue
                index = 0 if side == "low" else self.shape[axis] - 1
                eps_slab = eps.select(axis, index).unsqueeze(axis)
                area_slab = _broadcast_face(axis_area[axis], axis, index, self.shape)
                dist = half_cell[axis][index] * 0.5
                gb = self.eps0 * eps_slab * area_slab / dist
                self._boundary.append(
                    {"axis": axis, "side": side, "gb": gb, "value": float(value)}
                )

        self.diag = self._build_diag()

    def _build_diag(self) -> torch.Tensor:
        diag = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        diag = diag + _axis_diag(self.gx, 0, self.shape)
        diag = diag + _axis_diag(self.gy, 1, self.shape)
        diag = diag + _axis_diag(self.gz, 2, self.shape)
        for entry in self._boundary:
            diag = diag + _slab_full(entry["gb"], entry["axis"], entry["side"], self.shape[entry["axis"]])
        return diag

    def apply_full(self, phi: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.shape, dtype=phi.dtype, device=phi.device)
        fx = self.gx * (phi[:-1, :, :] - phi[1:, :, :])
        out = out + _axis_divergence(fx, 0, self.shape)
        fy = self.gy * (phi[:, :-1, :] - phi[:, 1:, :])
        out = out + _axis_divergence(fy, 1, self.shape)
        fz = self.gz * (phi[:, :, :-1] - phi[:, :, 1:])
        out = out + _axis_divergence(fz, 2, self.shape)
        for entry in self._boundary:
            axis, side = entry["axis"], entry["side"]
            index = 0 if side == "low" else self.shape[axis] - 1
            phi_slab = phi.select(axis, index).unsqueeze(axis)
            out = out + _slab_full(entry["gb"] * phi_slab, axis, side, self.shape[axis])
        return out

    def rhs_full(self) -> torch.Tensor:
        b = self.compiled.free_charge.clone()
        for entry in self._boundary:
            b = b + _slab_full(entry["gb"] * entry["value"], entry["axis"], entry["side"], self.shape[entry["axis"]])
        return b

    def rhs_boundary(self) -> torch.Tensor:
        """Right-hand side from Dirichlet boundary values only (no free charge).

        Used by the capacitance extraction, which measures the pure conductor
        response and ignores any volumetric free-charge sources.
        """
        b = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        for entry in self._boundary:
            b = b + _slab_full(entry["gb"] * entry["value"], entry["axis"], entry["side"], self.shape[entry["axis"]])
        return b

    def field_energy(self, phi: torch.Tensor) -> torch.Tensor:
        energy = 0.5 * (self.gx * (phi[:-1, :, :] - phi[1:, :, :]) ** 2).sum()
        energy = energy + 0.5 * (self.gy * (phi[:, :-1, :] - phi[:, 1:, :]) ** 2).sum()
        energy = energy + 0.5 * (self.gz * (phi[:, :, :-1] - phi[:, :, 1:]) ** 2).sum()
        for entry in self._boundary:
            axis, side = entry["axis"], entry["side"]
            index = 0 if side == "low" else self.shape[axis] - 1
            phi_slab = phi.select(axis, index).unsqueeze(axis)
            energy = energy + 0.5 * (entry["gb"] * (phi_slab - entry["value"]) ** 2).sum()
        return energy

    def boundary_electrode_charge(self, phi: torch.Tensor) -> float:
        """Total free charge on all Dirichlet-boundary electrodes (Coulombs)."""
        total = phi.new_zeros(())
        for entry in self._boundary:
            axis, side = entry["axis"], entry["side"]
            index = 0 if side == "low" else self.shape[axis] - 1
            phi_slab = phi.select(axis, index).unsqueeze(axis)
            total = total + (entry["gb"] * (entry["value"] - phi_slab)).sum()
        return total

    def boundary_work(self, phi: torch.Tensor) -> torch.Tensor:
        work = phi.new_zeros(())
        for entry in self._boundary:
            axis, side = entry["axis"], entry["side"]
            index = 0 if side == "low" else self.shape[axis] - 1
            phi_slab = phi.select(axis, index).unsqueeze(axis)
            work = work + (entry["value"] * entry["gb"] * (entry["value"] - phi_slab)).sum()
        return work


def _harmonic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2.0 * a * b / (a + b)


def _broadcast_face(area: torch.Tensor, axis: int, index: int, shape) -> torch.Tensor:
    face = area.select(axis, 0) if area.shape[axis] == 1 else area.select(axis, index)
    target = list(shape)
    target[axis] = 1
    return face.reshape([s for i, s in enumerate(target)])


def _axis_divergence(flux: torch.Tensor, axis: int, shape) -> torch.Tensor:
    zeros_shape = list(shape)
    zeros_shape[axis] = 1
    zeros = torch.zeros(zeros_shape, dtype=flux.dtype, device=flux.device)
    high = torch.cat([flux, zeros], dim=axis)
    low = torch.cat([zeros, flux], dim=axis)
    return high - low


def _axis_diag(conductance: torch.Tensor, axis: int, shape) -> torch.Tensor:
    zeros_shape = list(shape)
    zeros_shape[axis] = 1
    zeros = torch.zeros(zeros_shape, dtype=conductance.dtype, device=conductance.device)
    high = torch.cat([conductance, zeros], dim=axis)
    low = torch.cat([zeros, conductance], dim=axis)
    return high + low


@dataclass
class PCGReport:
    iterations: int
    residual: float  # relative ||r|| / ||b||
    residual_abs: float
    converged: bool


def _pcg(apply_A, b, inv_diag, *, tol, max_iterations):
    """Jacobi-preconditioned conjugate gradient on the free-cell system."""
    b_norm = torch.linalg.vector_norm(b)
    x = torch.zeros_like(b)
    if float(b_norm) == 0.0:
        return x, PCGReport(iterations=0, residual=0.0, residual_abs=0.0, converged=True)
    r = b - apply_A(x)
    z = inv_diag * r
    p = z.clone()
    rz = (r * z).sum()
    report = PCGReport(iterations=0, residual=float("inf"), residual_abs=float("inf"), converged=False)
    for iteration in range(1, int(max_iterations) + 1):
        Ap = apply_A(p)
        pAp = (p * Ap).sum()
        if float(pAp) <= 0.0:
            raise RuntimeError(
                "Electrostatic PCG encountered a non-positive curvature p^T A p="
                f"{float(pAp):.3e}; the operator is not positive definite (check for an "
                "under-constrained/gauge-singular problem)."
            )
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = torch.linalg.vector_norm(r)
        rel = float(r_norm / b_norm)
        report.iterations = iteration
        report.residual = rel
        report.residual_abs = float(r_norm)
        if rel <= tol:
            report.converged = True
            break
        z = inv_diag * r
        rz_new = (r * z).sum()
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, report


@dataclass
class ElectrostaticResultData:
    """Typed electrostatic solver output on the cell-centred grid.

    Coordinates and units are fixed: ``potential`` [V] at cell centres,
    ``E`` [V/m] and ``D`` [C/m^2] colocated at cell centres (E is the central
    finite difference of the potential, one-sided at domain edges), ``energy``
    [J], per-terminal ``charge`` [C]. ``residual`` is the relative PCG residual;
    ``gauss_error`` is the maximum per-cell discrete-Gauss residual over the free
    (non-conductor) cells.
    """

    potential: torch.Tensor
    Ex: torch.Tensor
    Ey: torch.Tensor
    Ez: torch.Tensor
    Dx: torch.Tensor
    Dy: torch.Tensor
    Dz: torch.Tensor
    epsilon_r: torch.Tensor
    free_charge: torch.Tensor
    cell_volume: torch.Tensor
    xc: torch.Tensor
    yc: torch.Tensor
    zc: torch.Tensor
    energy: torch.Tensor
    residual: float
    residual_abs: float
    iterations: int
    gauss_error: float
    _charges: dict[str, torch.Tensor] = field(default_factory=dict)
    boundary_charge: torch.Tensor | None = None

    @property
    def E(self) -> torch.Tensor:
        """Electric field stacked on the last axis: shape (nx, ny, nz, 3)."""
        return torch.stack((self.Ex, self.Ey, self.Ez), dim=-1)

    @property
    def D(self) -> torch.Tensor:
        return torch.stack((self.Dx, self.Dy, self.Dz), dim=-1)

    @property
    def terminal_names(self) -> tuple[str, ...]:
        return tuple(self._charges)

    def terminal_charge(self, name: str) -> torch.Tensor:
        if name not in self._charges:
            raise KeyError(
                f"No electrostatic terminal named {name!r}; available: {tuple(self._charges)}."
            )
        return self._charges[name]


def _cell_e_field(phi, coord, axis):
    if phi.shape[axis] < 2:
        return torch.zeros_like(phi)
    grad = torch.gradient(phi, spacing=(coord,), dim=axis)[0]
    return -grad


def _pinned_value(shape, dtype, device, entries) -> torch.Tensor:
    """Build a full-grid tensor holding each pinned cell's fixed potential.

    ``entries`` is an iterable of ``(mask, value)`` pairs; masks must be disjoint
    (the compiler already rejects overlapping terminals). Cells outside every
    mask stay at zero, which is also the value the reduced solve assigns to free
    cells.
    """
    value = torch.zeros(shape, dtype=dtype, device=device)
    for mask, magnitude in entries:
        value = torch.where(mask, torch.full(shape, float(magnitude), dtype=dtype, device=device), value)
    return value


def _pcg_free(operator, free_mask, b_free, config):
    """Jacobi-PCG solve of the reduced free-cell SPD system ``A_free x = b_free``.

    ``A_free(x) = free * apply_full(free * x)`` and ``b_free`` is already projected
    onto the free cells (zero on pinned cells). Returns ``(x, PCGReport)`` with
    ``x`` supported on the free cells (zero on pinned cells). Runs under
    ``no_grad``; differentiability is provided by the implicit-diff wrapper, not
    by taping the iteration.
    """
    free = free_mask.to(b_free.dtype)

    def apply_A(x):
        return free * operator.apply_full(free * x)

    inv_diag = torch.where(free_mask, 1.0 / operator.diag, torch.zeros_like(operator.diag))
    with torch.no_grad():
        x, report = _pcg(
            apply_A,
            b_free,
            inv_diag,
            tol=config.tolerance,
            max_iterations=config.max_iterations,
        )
    if not report.converged:
        raise RuntimeError(
            "Electrostatic PCG failed to converge: "
            f"{report.iterations} iterations, relative residual {report.residual:.3e} "
            f"(tolerance {config.tolerance:.3e}, absolute residual {report.residual_abs:.3e}). "
            "Increase max_iterations, loosen tolerance, or check the problem conditioning."
        )
    return x, report


def _reduced_solve(operator, free_mask, fixed_value, b_full, config):
    """Solve the Dirichlet-reduced SPD system on the free cells.

    ``fixed_value`` pins the potential on the complement of ``free_mask`` (it is
    zero on free cells). Returns ``(phi_full, PCGReport)`` with ``phi_full`` equal
    to ``fixed_value`` on pinned cells and the solved potential on free cells.
    """
    free = free_mask.to(fixed_value.dtype)
    b_free = free * (b_full - operator.apply_full(fixed_value))
    x, report = _pcg_free(operator, free_mask, b_free, config)
    return x + fixed_value, report


def _build_operator_with(compiled, eps, free_charge):
    """Rebuild an :class:`ElectrostaticOperator` with substituted field tensors.

    Used by the implicit-differentiation wrapper: the operator conductances and
    boundary right-hand side are analytic functions of ``eps`` (and ``free_charge``
    for the volumetric source), so re-deriving them through ``dataclasses.replace``
    lets autograd form the residual VJP without hand-coding ``dA/d(eps)``.
    """
    comp = dataclasses.replace(compiled, epsilon_r=eps, free_charge=free_charge)
    return ElectrostaticOperator(comp)


class _ElectrostaticSolve(torch.autograd.Function):
    """Implicit-differentiation wrapper around the reduced electrostatic solve.

    Forward solves the Dirichlet-reduced SPD system ``A(theta) phi = b(theta)``
    with Jacobi-PCG under ``no_grad`` (the iteration is never taped). Backward uses
    the implicit function theorem on the full-grid residual

        G_free(phi, theta)  = apply_full(phi; eps) - b(theta)    (free cells)
        G_pinned(phi, theta) = phi - fixed_value                 (pinned cells)

    with ``G = 0`` at the solution. Because ``A`` is symmetric positive definite
    the adjoint multiplier on the free cells solves the *same* reduced operator,
    reusing the forward preconditioner; the pinned-cell multiplier follows in
    closed form. The parameter gradients are then the residual vector-Jacobian
    product ``-(dG/d theta)^T mu``, evaluated by a single autograd pass over the
    re-derived operator so ``d(eps)`` / ``d(free_charge)`` / ``d(fixed_value)``
    never need explicit stencil derivatives.
    """

    @staticmethod
    def forward(ctx, eps, free_charge, fixed_value, payload):
        compiled, free_mask, config, use_free_charge, stats = payload
        operator = _build_operator_with(compiled, eps.detach(), free_charge.detach())
        b_full = operator.rhs_full() if use_free_charge else operator.rhs_boundary()
        free = free_mask.to(eps.dtype)
        b_free = free * (b_full - operator.apply_full(fixed_value.detach()))
        x, report = _pcg_free(operator, free_mask, b_free, config)
        phi = x + fixed_value.detach()
        ctx.save_for_backward(eps, free_charge, fixed_value, phi)
        ctx.payload = payload
        if stats is not None:
            stats["report"] = report
        return phi

    @staticmethod
    def backward(ctx, grad_phi):
        eps, free_charge, fixed_value, phi = ctx.saved_tensors
        compiled, free_mask, config, use_free_charge, _stats = ctx.payload
        free = free_mask.to(eps.dtype)

        # Adjoint multiplier. A is SPD, so A_free^T = A_free and the free-cell
        # block reuses the forward reduced operator/preconditioner.
        operator = _build_operator_with(compiled, eps.detach(), free_charge.detach())
        rhs = free * grad_phi
        mu_free, _ = _pcg_free(operator, free_mask, rhs, config)
        # Pinned-cell multiplier: mu_p = grad_phi_p - (A_fp^T mu_f)_p, and since A is
        # symmetric (A_fp^T mu_f)_p = apply_full(embed(mu_f))_p.
        resid_p = grad_phi - operator.apply_full(mu_free)
        mu_full = mu_free + (1.0 - free) * resid_p

        # Residual VJP: grad_theta = -(dG/d theta)^T mu, formed by autograd over the
        # re-derived operator with the solution held constant.
        with torch.enable_grad():
            eps_ = eps.detach().requires_grad_(True)
            fc_ = free_charge.detach().requires_grad_(True)
            fx_ = fixed_value.detach().requires_grad_(True)
            operator_ = _build_operator_with(compiled, eps_, fc_)
            b_ = operator_.rhs_full() if use_free_charge else operator_.rhs_boundary()
            phi_c = phi.detach()
            g_free = free * (operator_.apply_full(phi_c) - b_)
            g_pinned = (1.0 - free) * (phi_c - fx_)
            residual = g_free + g_pinned
            g_eps, g_fc, g_fx = torch.autograd.grad(
                residual,
                (eps_, fc_, fx_),
                grad_outputs=-mu_full,
                allow_unused=True,
            )
        return g_eps, g_fc, g_fx, None


def differentiable_solve(compiled, fixed_value, free_mask, config, *, use_free_charge=True):
    """Solve ``A(theta) phi = b(theta)`` with implicit-diff backward support.

    Returns ``(phi_full, PCGReport)``. ``phi`` is differentiable with respect to
    ``compiled.epsilon_r`` and ``compiled.free_charge`` (and ``fixed_value``); the
    forward PCG still runs under ``no_grad`` so there is no autograd-tape overhead.
    """
    stats: dict[str, Any] = {}
    payload = (compiled, free_mask, config, use_free_charge, stats)
    phi = _ElectrostaticSolve.apply(
        compiled.epsilon_r, compiled.free_charge, fixed_value, payload
    )
    return phi, stats["report"]


def _terminal_charges(reaction, terminals) -> dict[str, torch.Tensor]:
    return {t.name: reaction[t.mask].sum() for t in terminals}


def _solve_floating_superposition(
    operator, compiled, fixed_terms, floating_terms, free_mask, config, *, has_level_anchor
):
    """Resolve floating-conductor potentials by linear superposition.

    Each floating conductor is an equipotential of unknown level ``alpha_j`` whose
    induced free charge is prescribed. Because the field is linear in the pinned
    potentials, ``phi = phi_base + sum_j alpha_j * phi_unit_j`` where

    - ``phi_base`` fixes the potential terminals at their voltages, pins every
      floating conductor at 0, and keeps the true sources / boundary values on;
    - ``phi_unit_j`` pins floating conductor ``j`` at 1 V, everything else at 0,
      with sources and boundary values off (homogeneous).

    The prescribed-charge constraints give a small dense ``k x k`` system
    ``M alpha = q_target - q_base`` where ``M_ij`` is the charge induced on
    floating conductor ``i`` by a unit potential on conductor ``j`` (a capacitance
    submatrix). ``k`` equals the number of floating conductors, so this reduction
    solve is tiny; it runs on CPU float64 with a rank-revealing least-squares
    driver so a gauge-singular (fully floating, insulated) system is handled
    cleanly rather than silently mis-solved.
    """
    shape = compiled.shape
    dtype = compiled.dtype
    device = compiled.device

    fixed_entries = [(t.mask, t.potential) for t in fixed_terms]
    fixed_value_base = _pinned_value(shape, dtype, device, fixed_entries)
    phi_base, report_base = _reduced_solve(
        operator, free_mask, fixed_value_base, operator.rhs_full(), config
    )
    reaction_base = operator.apply_full(phi_base) - operator.rhs_full()

    zero_rhs = torch.zeros(shape, dtype=dtype, device=device)
    phi_units: list[torch.Tensor] = []
    reaction_units: list[torch.Tensor] = []
    reports = [report_base]
    for term in floating_terms:
        unit_value = _pinned_value(shape, dtype, device, [(term.mask, 1.0)])
        phi_j, report_j = _reduced_solve(operator, free_mask, unit_value, zero_rhs, config)
        phi_units.append(phi_j)
        reaction_units.append(operator.apply_full(phi_j))
        reports.append(report_j)

    k = len(floating_terms)
    q_base = torch.stack([reaction_base[t.mask].sum() for t in floating_terms])
    q_target = torch.tensor(
        [float(t.charge) for t in floating_terms], dtype=dtype, device=device
    )
    matrix = torch.stack(
        [
            torch.stack([reaction_units[j][floating_terms[i].mask].sum() for j in range(k)])
            for i in range(k)
        ]
    )

    rhs_vec = q_target - q_base

    if not has_level_anchor:
        # No grounded electrode or Dirichlet boundary: the floating-conductor
        # capacitance submatrix is singular (uniform-potential null vector). The
        # discrete divergence theorem forces the total charge to balance, so the
        # prescribed conductor charges plus any free charge must sum to zero;
        # otherwise the constraints are physically incompatible.
        total_free = float(compiled.free_charge.sum())
        total_q = float(q_target.sum()) + total_free
        scale = float(q_target.abs().max()) + abs(total_free) + 1.0e-300
        if abs(total_q) > 1.0e-6 * scale:
            raise ValueError(
                "Floating-conductor charge constraints are incompatible with an insulated "
                "(pure-Neumann) enclosure: the prescribed conductor charges plus any free "
                f"charge must sum to zero (got net {total_q:.3e} C). Add a grounded electrode "
                "or a Dirichlet boundary as a charge return path."
            )

    # Tiny k x k control solve. CPU least squares (gelsd) is rank-revealing; a
    # singular gauge direction is nulled by rcond and gauge-fixed afterwards.
    matrix_cpu = matrix.detach().double().cpu()
    rhs_cpu = rhs_vec.detach().double().cpu()
    lstsq = torch.linalg.lstsq(matrix_cpu, rhs_cpu.unsqueeze(-1), driver="gelsd", rcond=1.0e-10)
    alpha = lstsq.solution.squeeze(-1).to(device=device, dtype=dtype)

    phi = phi_base
    for j in range(k):
        phi = phi + alpha[j] * phi_units[j]
    return phi, reports


def solve_electrostatics(
    compiled: CompiledElectrostatics,
    config: ElectrostaticSolverConfig | None = None,
) -> ElectrostaticResultData:
    if config is None:
        config = ElectrostaticSolverConfig()
    dtype = config.dtype
    if compiled.dtype != dtype:
        raise ValueError(
            "CompiledElectrostatics dtype must match the solver dtype; compile with the "
            "same dtype the solver uses."
        )
    operator = ElectrostaticOperator(compiled)
    device = compiled.device
    shape = compiled.shape

    fixed_terms = [t for t in compiled.terminals if not t.is_floating]
    floating_terms = [t for t in compiled.terminals if t.is_floating]

    all_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    for terminal in compiled.terminals:
        all_mask = all_mask | terminal.mask
    free_mask = ~all_mask
    if not bool(free_mask.any()):
        raise ValueError("Electrostatic problem has no free cells; every cell is pinned.")

    # A well-posed reduced solve needs at least one pinned cell (any conductor) or
    # a Dirichlet boundary face; otherwise the free-cell operator is singular.
    has_solve_anchor = bool(all_mask.any()) or compiled.boundary.has_dirichlet
    # The absolute potential level is fixed only by a Dirichlet boundary or a
    # fixed-potential/grounded terminal; floating conductors do not anchor it.
    has_level_anchor = bool(fixed_terms) or compiled.boundary.has_dirichlet
    if not has_solve_anchor:
        raise NotImplementedError(
            "Pure-Neumann electrostatic problems are gauge-singular (potential defined only "
            "up to a constant) and require charge-compatibility plus a gauge fix; supply at "
            "least one Dirichlet boundary face or a fixed-potential/grounded terminal."
        )

    requires_grad = compiled.epsilon_r.requires_grad or compiled.free_charge.requires_grad
    b_full = operator.rhs_full()
    if floating_terms:
        if requires_grad:
            raise NotImplementedError(
                "Gradients through the floating-conductor superposition solve are not "
                "implemented; the prescribed-charge k x k reduction is not yet differentiated. "
                "Detach the permittivity / free charge, or use only fixed-potential "
                "(grounded / potential=) terminals for a differentiable electrostatic solve."
            )
        phi, reports = _solve_floating_superposition(
            operator, compiled, fixed_terms, floating_terms, free_mask, config,
            has_level_anchor=has_level_anchor,
        )
        report = reports[0]
        if not has_level_anchor:
            # Only floating conductors and an insulated boundary: the potential
            # level is a pure gauge freedom. Fix it with mean(phi) = 0.
            phi = phi - phi.mean()
    else:
        # Without floating conductors, any terminal set consists of fixed-potential
        # electrodes, so a solve anchor is also a level anchor; the pure-Neumann,
        # no-conductor case was already rejected by the has_solve_anchor check.
        # The reduced solve routes through the implicit-diff wrapper so downstream
        # energy / charge / field quantities are differentiable in eps and charge.
        fixed_value = _pinned_value(shape, dtype, device, [(t.mask, t.potential) for t in fixed_terms])
        phi, report = differentiable_solve(
            compiled, fixed_value, free_mask, config, use_free_charge=True
        )

    free = free_mask.to(dtype)
    reaction = operator.apply_full(phi) - b_full
    gauss_error = float((reaction * free).detach().abs().max()) if bool(free_mask.any()) else 0.0

    charges = _terminal_charges(reaction, compiled.terminals)

    energy = operator.field_energy(phi)

    Ex = _cell_e_field(phi, compiled.xc, 0)
    Ey = _cell_e_field(phi, compiled.yc, 1)
    Ez = _cell_e_field(phi, compiled.zc, 2)
    eps_abs = compiled.eps0 * compiled.epsilon_r
    Dx = eps_abs * Ex
    Dy = eps_abs * Ey
    Dz = eps_abs * Ez

    boundary_charge = operator.boundary_electrode_charge(phi) if operator._boundary else None

    return ElectrostaticResultData(
        potential=phi,
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
        Dx=Dx,
        Dy=Dy,
        Dz=Dz,
        epsilon_r=compiled.epsilon_r,
        free_charge=compiled.free_charge,
        cell_volume=compiled.cell_volume,
        xc=compiled.xc,
        yc=compiled.yc,
        zc=compiled.zc,
        energy=energy,
        residual=report.residual,
        residual_abs=report.residual_abs,
        iterations=report.iterations,
        gauss_error=gauss_error,
        _charges=charges,
        boundary_charge=boundary_charge,
    )


class ElectrostaticSimulation:
    """Runner returned by ``Simulation.electrostatic(...)``.

    Keeps the public entry ``Scene -> Simulation -> Result``: ``run()`` returns a
    standard ``Result(method="electrostatic")`` whose ``result.electrostatic``
    accessor is the typed :class:`ElectrostaticResultData`.
    """

    def __init__(self, scene, boundary=None, solver=None):
        self.scene = scene
        if boundary is None:
            boundary = ElectrostaticBoundarySpec.grounded_box()
        if not isinstance(boundary, ElectrostaticBoundarySpec):
            raise TypeError("boundary must be an ElectrostaticBoundarySpec.")
        self.boundary = boundary
        if solver is None:
            solver = ElectrostaticSolverConfig()
        if not isinstance(solver, ElectrostaticSolverConfig):
            raise TypeError("solver must be an ElectrostaticSolverConfig.")
        self.solver = solver

    def prepare(self) -> CompiledElectrostatics:
        return compile_electrostatics(self.scene, self.boundary, dtype=self.solver.dtype)

    def run(self):
        from ..result import Result

        compiled = self.prepare()
        data = solve_electrostatics(compiled, self.solver)
        return Result(
            method="electrostatic",
            scene=self.scene,
            frequency=0.0,
            solver_stats={
                "iterations": data.iterations,
                "residual": data.residual,
                "residual_abs": data.residual_abs,
                "gauss_error": data.gauss_error,
            },
            raw_output=data,
        )
