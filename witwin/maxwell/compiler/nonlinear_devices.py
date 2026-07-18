"""Nonlinear circuit device compiler and GPU Newton core (Phase 0, standalone).

This module turns the public nonlinear device declarations
(:mod:`witwin.maxwell.circuit_devices`) into fixed-shape, per-model-signature
batches (:class:`CompiledNonlinearDevice`) that expose analytic conduction
current ``i(v)`` / conductance ``di/dv`` and stored charge ``q(v)`` /
capacitance ``dq/dv``, and it provides a small dense-MNA residual/Jacobian
assembler (:class:`NonlinearMNASystem`) plus the Newton-Raphson core
(:func:`newton_solve`).

Phase-0 scope is deliberately *standalone*: node-voltage unknowns on analytic
scalar / small systems, no FDTD coupling and no LinearMNASystem integration yet.
The Newton core carries the two contracts that every later slice builds on:

* a **dual convergence gate** -- an iterate is accepted only when *both* the
  scaled KCL residual and the Newton update satisfy their own tolerances, so a
  residual-only (or update-only) false convergence is impossible; and
* **stable ``expm1`` + ``pnjlim`` junction limiting**, so the exponential device
  law never overflows and a rootless / non-convergent system fails closed
  deterministically within the iteration cap instead of returning garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from ..circuit_devices import (
    Diode,
    NonlinearSolveConfig,
    PiecewiseLinearIV,
    PolynomialIV,
    VoltageDependentCapacitor,
)

# CODATA 2018 fixed physical constants.
_BOLTZMANN = 1.380649e-23
_ELEMENTARY_CHARGE = 1.602176634e-19


class NonlinearDeviceError(RuntimeError):
    """A device produced a non-finite or otherwise invalid value inside an iterate."""


class NonlinearConvergenceError(RuntimeError):
    """Newton failed to satisfy the dual convergence gate within ``max_iterations``.

    Carries the localized diagnostics (residual norm, per-node residual, the worst
    node index, and the residual-norm iteration trajectory) so a failure points at
    the offending node/device instead of silently returning an unconverged state.
    """

    def __init__(self, message, *, stats):
        super().__init__(message)
        self.stats = stats


@dataclass(frozen=True)
class NonlinearSolveStats:
    """Per-solve Newton diagnostics (all host scalars, produced once at the end)."""

    converged: bool
    iterations: int
    residual_norm: float
    update_norm: float
    residual_scale: float
    condition_estimate: float
    line_search_reductions: int
    worst_node: int
    residual_trajectory: tuple[float, ...]


def _thermal_voltage(temperature: torch.Tensor) -> torch.Tensor:
    return (_BOLTZMANN / _ELEMENTARY_CHARGE) * temperature


def _dual_gate_converged(
    residual_norm: torch.Tensor,
    update_norm: torch.Tensor,
    residual_scale: torch.Tensor,
    solution_norm: torch.Tensor,
    config: NonlinearSolveConfig,
) -> torch.Tensor:
    """Both gates must pass; a single-gate pass is never convergence.

    * residual gate: ``||F|| <= atol + rtol * max(scale)`` (scale = characteristic
      branch/source current magnitude);
    * update gate: ``||dx|| <= uatol + urtol * ||x||``.
    """

    residual_ok = residual_norm <= config.absolute_tolerance + config.relative_tolerance * residual_scale
    update_ok = update_norm <= config.update_absolute_tolerance + config.update_relative_tolerance * solution_norm
    return residual_ok & update_ok


def _pnjlim(vnew: torch.Tensor, vold: torch.Tensor, vte: torch.Tensor, vcrit: torch.Tensor) -> torch.Tensor:
    """SPICE-style pn-junction voltage limiting on the diode terminal voltage.

    Bounds the per-iteration junction-voltage change so ``exp(v / vte)`` cannot
    overflow, without clamping the accepted solution to a wrong value: limiting
    is inactive near the root, where Newton recovers quadratic convergence.
    """

    active = (vnew > vcrit) & ((vnew - vold).abs() > 2.0 * vte)
    positive_old = vold > 0.0
    arg = 1.0 + (vnew - vold) / vte
    limited_from_positive = torch.where(arg > 0.0, vold + vte * torch.log(arg.clamp_min(1e-30)), vcrit)
    safe_ratio = (vnew / vte).clamp_min(1e-30)
    limited_from_nonpositive = torch.where(vnew > 0.0, vte * torch.log(safe_ratio), vnew)
    limited = torch.where(positive_old, limited_from_positive, limited_from_nonpositive)
    return torch.where(active, limited, vnew)


def _horner(coefficients: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Evaluate ``sum_k coefficients[..., k] * v**k`` (constant term first)."""

    result = torch.zeros_like(v)
    for index in range(coefficients.shape[-1] - 1, -1, -1):
        result = result * v + coefficients[..., index]
    return result


def _horner_derivative(coefficients: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    degree = coefficients.shape[-1]
    if degree < 2:
        return torch.zeros_like(v)
    powers = torch.arange(1, degree, device=coefficients.device, dtype=torch.long)
    derivative_coeffs = coefficients[..., 1:] * powers.to(coefficients.dtype)
    return _horner(derivative_coeffs, v)


@dataclass(frozen=True)
class CompiledNonlinearDevice:
    """Fixed-shape batch of same-signature nonlinear devices.

    ``positive`` / ``negative`` are node-unknown indices (``-1`` denotes ground).
    ``parameters`` holds the model tensors stacked along a leading device axis.
    """

    kind: str
    names: tuple[str, ...]
    positive: torch.Tensor
    negative: torch.Tensor
    parameters: dict[str, torch.Tensor]

    @property
    def batch_size(self) -> int:
        return self.positive.shape[0]

    def terminal_voltage(self, x: torch.Tensor) -> torch.Tensor:
        zero = x.new_zeros(())
        pos = torch.where(self.positive >= 0, x[self.positive.clamp_min(0)], zero)
        neg = torch.where(self.negative >= 0, x[self.negative.clamp_min(0)], zero)
        return pos - neg

    # -- model laws (accept real or complex v; derivatives are analytic) -------
    def conduction(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(i(v), di/dv)``."""

        if self.kind == "diode":
            saturation = self.parameters["saturation_current"]
            vte = self.parameters["thermal_voltage"]
            arg = v / vte
            current = saturation * torch.expm1(arg)
            conductance = (saturation / vte) * torch.exp(arg)
            return current, conductance
        if self.kind == "polynomial_iv":
            coefficients = self.parameters["coefficients"]
            return _horner(coefficients, v), _horner_derivative(coefficients, v)
        if self.kind == "piecewise_linear_iv":
            return self._piecewise_conduction(v)
        # Charge-only devices carry no conduction current.
        zero = torch.zeros_like(v)
        return zero, zero

    def _piecewise_conduction(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        voltages = self.parameters["voltages"]
        currents = self.parameters["currents"]
        knots = voltages.shape[-1]
        query = v.real if v.is_complex() else v
        segment = torch.searchsorted(voltages, query.unsqueeze(-1), right=True).squeeze(-1) - 1
        segment = segment.clamp(0, knots - 2)
        left_v = torch.gather(voltages, -1, segment.unsqueeze(-1)).squeeze(-1)
        right_v = torch.gather(voltages, -1, (segment + 1).unsqueeze(-1)).squeeze(-1)
        left_i = torch.gather(currents, -1, segment.unsqueeze(-1)).squeeze(-1)
        right_i = torch.gather(currents, -1, (segment + 1).unsqueeze(-1)).squeeze(-1)
        slope = (right_i - left_i) / (right_v - left_v)
        current = left_i + slope * (v - left_v)
        return current, slope

    def charge(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q(v), dq/dv)``."""

        if self.kind == "voltage_dependent_capacitor":
            coefficients = self.parameters["q_coefficients"]
            return _horner(coefficients, v), _horner_derivative(coefficients, v)
        if self.kind == "diode":
            capacitance = self.parameters["junction_capacitance"]
            return capacitance * v, capacitance + torch.zeros_like(v)
        zero = torch.zeros_like(v)
        return zero, zero

    def limit(self, v_new: torch.Tensor, v_old: torch.Tensor) -> torch.Tensor:
        if self.kind != "diode":
            return v_new
        vte = self.parameters["thermal_voltage"]
        vcrit = self.parameters["critical_voltage"]
        return _pnjlim(v_new, v_old, vte, vcrit)


def _group_key(device) -> tuple:
    if isinstance(device, Diode):
        return ("diode",)
    if isinstance(device, PiecewiseLinearIV):
        return ("piecewise_linear_iv", int(device.voltages.numel()))
    if isinstance(device, PolynomialIV):
        return ("polynomial_iv", int(device.coefficients.numel()))
    if isinstance(device, VoltageDependentCapacitor):
        return ("voltage_dependent_capacitor", int(device.q_coefficients.numel()))
    raise TypeError(f"{type(device).__name__} is not a compiled nonlinear device type.")


def _stack(values, *, device, dtype) -> torch.Tensor:
    return torch.stack([value.to(device=device, dtype=dtype) for value in values], dim=0)


def compile_nonlinear_devices(
    devices,
    node_index,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[CompiledNonlinearDevice, ...]:
    """Group nonlinear devices by model signature into fixed-shape batches.

    ``node_index`` maps a node name to its unknown index; the ground node maps to
    ``-1`` (either explicitly or by being absent).
    """

    device = torch.device(device)

    def index_of(node) -> int:
        if node.is_ground:
            return -1
        return int(node_index[node.name])

    groups: dict[tuple, list] = {}
    for item in devices:
        groups.setdefault(_group_key(item), []).append(item)

    compiled: list[CompiledNonlinearDevice] = []
    for key, members in groups.items():
        kind = key[0]
        names = tuple(member.name for member in members)
        positive = torch.tensor([index_of(member.positive) for member in members], device=device, dtype=torch.long)
        negative = torch.tensor([index_of(member.negative) for member in members], device=device, dtype=torch.long)
        parameters: dict[str, torch.Tensor] = {}
        if kind == "diode":
            saturation = _stack([m.saturation_current for m in members], device=device, dtype=dtype)
            ideality = _stack([m.ideality for m in members], device=device, dtype=dtype)
            temperature = _stack([m.temperature for m in members], device=device, dtype=dtype)
            junction = _stack([m.junction_capacitance for m in members], device=device, dtype=dtype)
            vte = ideality * _thermal_voltage(temperature)
            critical = vte * torch.log(vte / (math.sqrt(2.0) * saturation))
            parameters = {
                "saturation_current": saturation,
                "ideality": ideality,
                "thermal_voltage": vte,
                "critical_voltage": critical,
                "junction_capacitance": junction,
            }
        elif kind == "polynomial_iv":
            parameters = {"coefficients": _stack([m.coefficients for m in members], device=device, dtype=dtype)}
        elif kind == "piecewise_linear_iv":
            parameters = {
                "voltages": _stack([m.voltages for m in members], device=device, dtype=dtype),
                "currents": _stack([m.currents for m in members], device=device, dtype=dtype),
            }
        elif kind == "voltage_dependent_capacitor":
            parameters = {"q_coefficients": _stack([m.q_coefficients for m in members], device=device, dtype=dtype)}
        compiled.append(
            CompiledNonlinearDevice(
                kind=kind,
                names=names,
                positive=positive,
                negative=negative,
                parameters=parameters,
            )
        )
    return tuple(compiled)


def _scatter_currents(target: torch.Tensor, positive, negative, values, *, sign: float) -> None:
    """Accumulate ``sign * values`` into ``target[positive]`` and ``-`` into negative."""

    pos_mask = positive >= 0
    neg_mask = negative >= 0
    if bool(pos_mask.any()):
        target.index_add_(0, positive[pos_mask], (sign * values)[pos_mask])
    if bool(neg_mask.any()):
        target.index_add_(0, negative[neg_mask], (-sign * values)[neg_mask])


def _stamp_conductance(matrix: torch.Tensor, positive, negative, conductance) -> None:
    pos_mask = positive >= 0
    neg_mask = negative >= 0
    both = pos_mask & neg_mask
    if bool(pos_mask.any()):
        p = positive[pos_mask]
        matrix.index_put_((p, p), conductance[pos_mask], accumulate=True)
    if bool(neg_mask.any()):
        n = negative[neg_mask]
        matrix.index_put_((n, n), conductance[neg_mask], accumulate=True)
    if bool(both.any()):
        p = positive[both]
        n = negative[both]
        matrix.index_put_((p, n), -conductance[both], accumulate=True)
        matrix.index_put_((n, p), -conductance[both], accumulate=True)


@dataclass
class NonlinearMNASystem:
    """Dense node-voltage MNA residual/Jacobian assembler for the standalone path.

    ``conductance`` and ``injection`` are the constant linear block (resistors,
    Norton-equivalent sources); the nonlinear device contribution is assembled
    per iterate. Node ``0`` (ground) is eliminated, so unknowns are the
    ``num_unknowns`` non-ground node voltages.
    """

    num_unknowns: int
    conductance: torch.Tensor
    injection: torch.Tensor
    devices: tuple[CompiledNonlinearDevice, ...]

    @property
    def dtype(self) -> torch.dtype:
        return self.conductance.dtype

    @property
    def device(self) -> torch.device:
        return self.conductance.device

    def true_residual(self, x: torch.Tensor, *, gmin: float = 0.0) -> torch.Tensor:
        """KCL residual ``F(x)`` using the exact device laws (no limiting)."""

        residual = self.conductance @ x - self.injection
        if gmin:
            residual = residual + gmin * x
        for compiled in self.devices:
            v = compiled.terminal_voltage(x)
            current, _ = compiled.conduction(v)
            if not bool(torch.all(torch.isfinite(current))):
                bad = compiled.names[int(torch.nonzero(~torch.isfinite(current))[0, 0])]
                raise NonlinearDeviceError(
                    f"Nonlinear device {bad!r} produced a non-finite current at v={v.tolist()}."
                )
            _scatter_currents(residual, compiled.positive, compiled.negative, current, sign=1.0)
        return residual

    def residual_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.injection.abs().max() if self.injection.numel() else self.injection.new_zeros(())
        for compiled in self.devices:
            v = compiled.terminal_voltage(x)
            current, _ = compiled.conduction(v)
            if current.numel():
                scale = torch.maximum(scale, current.abs().max())
        return scale


def _assemble_linear_step(
    system: NonlinearMNASystem,
    x: torch.Tensor,
    limiting_state: list[torch.Tensor],
    *,
    gmin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = system.conductance.clone()
    rhs = system.injection.clone()
    if gmin:
        matrix = matrix + gmin * torch.eye(system.num_unknowns, dtype=system.dtype, device=system.device)
    for slot, compiled in enumerate(system.devices):
        v_raw = compiled.terminal_voltage(x)
        v_lim = compiled.limit(v_raw, limiting_state[slot])
        limiting_state[slot] = v_lim
        current, conductance = compiled.conduction(v_lim)
        if not (bool(torch.all(torch.isfinite(current))) and bool(torch.all(torch.isfinite(conductance)))):
            raise NonlinearDeviceError(
                f"Nonlinear device group {compiled.kind!r} produced a non-finite linearization."
            )
        _stamp_conductance(matrix, compiled.positive, compiled.negative, conductance)
        companion = current - conductance * v_lim
        _scatter_currents(rhs, compiled.positive, compiled.negative, companion, sign=-1.0)
    return matrix, rhs


def _solve_linear(matrix: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, float]:
    if not bool(torch.all(torch.isfinite(matrix))) or not bool(torch.all(torch.isfinite(rhs))):
        raise NonlinearDeviceError("Nonlinear MNA assembled a non-finite linear system.")
    try:
        solution = torch.linalg.solve(matrix, rhs)
    except RuntimeError as error:  # singular / below-pivot Jacobian
        raise NonlinearDeviceError(f"Nonlinear Jacobian is singular: {error}") from error
    condition = float(torch.linalg.cond(matrix).item())
    return solution, condition


def newton_solve(
    system: NonlinearMNASystem,
    x0: torch.Tensor,
    config: NonlinearSolveConfig,
    *,
    gmin: float = 0.0,
) -> tuple[torch.Tensor, NonlinearSolveStats]:
    """GPU Newton with dual convergence gate, ``pnjlim`` limiting, and backtracking.

    Returns ``(solution, stats)`` on convergence. On non-convergence within
    ``max_iterations`` it raises :class:`NonlinearConvergenceError` (default
    ``failure='raise'``) or returns the last iterate with ``stats.converged`` set
    to ``False`` (``failure='record_and_stop'``).
    """

    x = x0.clone()
    limiting_state = [compiled.terminal_voltage(x) for compiled in system.devices]
    trajectory: list[float] = []
    condition = float("inf")
    line_search_reductions = 0
    residual_norm = torch.tensor(float("inf"), dtype=system.dtype, device=system.device)
    update_norm = torch.tensor(float("inf"), dtype=system.dtype, device=system.device)

    for iteration in range(1, config.max_iterations + 1):
        matrix, rhs = _assemble_linear_step(system, x, limiting_state, gmin=gmin)
        candidate, condition = _solve_linear(matrix, rhs)
        if not bool(torch.all(torch.isfinite(candidate))):
            raise NonlinearDeviceError("Nonlinear Newton produced a non-finite iterate.")

        if config.line_search == "backtracking":
            candidate, reductions = _backtrack(
                system, x, candidate, gmin=gmin, max_steps=config.max_line_search_steps
            )
            line_search_reductions += reductions

        update = candidate - x
        update_norm = torch.linalg.vector_norm(update)
        x = candidate

        residual = system.true_residual(x, gmin=gmin)
        residual_norm = torch.linalg.vector_norm(residual)
        trajectory.append(float(residual_norm.item()))
        scale = system.residual_scale(x)
        solution_norm = torch.linalg.vector_norm(x)

        if bool(_dual_gate_converged(residual_norm, update_norm, scale, solution_norm, config)):
            worst = int(torch.argmax(residual.abs()).item()) if residual.numel() else -1
            stats = NonlinearSolveStats(
                converged=True,
                iterations=iteration,
                residual_norm=float(residual_norm.item()),
                update_norm=float(update_norm.item()),
                residual_scale=float(scale.item()),
                condition_estimate=condition,
                line_search_reductions=line_search_reductions,
                worst_node=worst,
                residual_trajectory=tuple(trajectory),
            )
            return x, stats

    residual = system.true_residual(x, gmin=gmin)
    worst = int(torch.argmax(residual.abs()).item()) if residual.numel() else -1
    scale = system.residual_scale(x)
    stats = NonlinearSolveStats(
        converged=False,
        iterations=config.max_iterations,
        residual_norm=float(residual_norm.item()),
        update_norm=float(update_norm.item()),
        residual_scale=float(scale.item()),
        condition_estimate=condition,
        line_search_reductions=line_search_reductions,
        worst_node=worst,
        residual_trajectory=tuple(trajectory),
    )
    if config.failure == "record_and_stop":
        return x, stats
    raise NonlinearConvergenceError(
        f"Nonlinear Newton failed to converge in {config.max_iterations} iterations: "
        f"residual_norm={stats.residual_norm:.3e} (scale={stats.residual_scale:.3e}), "
        f"update_norm={stats.update_norm:.3e}, worst node index={worst}, "
        f"residual trajectory={['%.3e' % value for value in trajectory]}.",
        stats=stats,
    )


def _backtrack(
    system: NonlinearMNASystem,
    x: torch.Tensor,
    candidate: torch.Tensor,
    *,
    gmin: float,
    max_steps: int,
) -> tuple[torch.Tensor, int]:
    """Halve the Newton step until the true residual does not increase."""

    base = torch.linalg.vector_norm(system.true_residual(x, gmin=gmin))
    step = candidate - x
    trial = candidate
    scale = 1.0
    for reductions in range(max_steps + 1):
        try:
            trial_residual = torch.linalg.vector_norm(system.true_residual(trial, gmin=gmin))
        except NonlinearDeviceError:
            # An over-long step drove a device output non-finite; shrink instead
            # of failing -- the shortened step is re-checked on the next pass.
            trial_residual = torch.tensor(float("inf"), dtype=system.dtype, device=system.device)
        if bool(torch.isfinite(trial_residual)) and bool(trial_residual <= base):
            return trial, reductions
        scale *= 0.5
        trial = x + scale * step
    return trial, max_steps


__all__ = [
    "CompiledNonlinearDevice",
    "NonlinearConvergenceError",
    "NonlinearDeviceError",
    "NonlinearMNASystem",
    "NonlinearSolveStats",
    "compile_nonlinear_devices",
    "newton_solve",
]
