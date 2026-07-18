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
            resistive = tuple(m.name for m in members if bool(m.series_resistance != 0.0))
            if resistive:
                raise NotImplementedError(
                    f"Diodes {resistive} declare a nonzero series_resistance, but the "
                    "compiled conduction law is the ideal Shockley junction i(v); the "
                    "series-resistance branch (an internal node with the ohmic drop) is "
                    "not assembled, so honouring the parameter requires the extended "
                    "device topology. This fails closed until that branch is implemented "
                    "rather than silently solving the resistance-free junction."
                )
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

    When ``dt`` is set (via :meth:`init_transient`), the assembler also folds each
    device's stored-charge companion into the residual and Jacobian using the same
    bilinear-transform convention as the linear reactive companion
    (``compiler/mna.py`` / ``lumped.py``): trapezoidal uses ``2/dt`` with the
    prior capacitive-current history, backward Euler uses ``1/dt`` with none. The
    charge history advances only on an accepted step (:meth:`advance_charge_state`),
    so a rejected/limited Newton iterate never corrupts the integrator memory.
    """

    num_unknowns: int
    conductance: torch.Tensor
    injection: torch.Tensor
    devices: tuple[CompiledNonlinearDevice, ...]
    dt: float | None = None
    integration: str = "trapezoidal"
    charge_prev: list[torch.Tensor] = field(default_factory=list)
    capacitive_current_prev: list[torch.Tensor] = field(default_factory=list)

    @property
    def dtype(self) -> torch.dtype:
        return self.conductance.dtype

    @property
    def device(self) -> torch.device:
        return self.conductance.device

    def init_transient(self, x0: torch.Tensor, dt: float, integration: str) -> None:
        """Arm the charge integrator from the operating point ``x0``.

        ``charge_prev`` is seeded with ``q(v0)`` (not zero) so the first companion
        step measures the true charge increment from the DC operating point, and
        the capacitive-current history starts at zero (DC steady state carries no
        displacement current).
        """

        if integration not in ("trapezoidal", "backward_euler"):
            raise ValueError("integration must be 'trapezoidal' or 'backward_euler'.")
        if not (dt > 0.0):
            raise ValueError("dt must be positive.")
        self.dt = float(dt)
        self.integration = integration
        self.charge_prev = []
        self.capacitive_current_prev = []
        for compiled in self.devices:
            v = compiled.terminal_voltage(x0)
            q, _ = compiled.charge(v)
            self.charge_prev.append(q)
            self.capacitive_current_prev.append(torch.zeros_like(q))

    def _charge_factor(self) -> float:
        return (2.0 if self.integration == "trapezoidal" else 1.0) / self.dt

    def _charge_companion(
        self, slot: int, compiled: CompiledNonlinearDevice, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trapezoidal/BE stored-charge companion ``(i_cap(v), dcap/dv)``."""

        q, capacitance = compiled.charge(v)
        factor = self._charge_factor()
        current = factor * (q - self.charge_prev[slot])
        if self.integration == "trapezoidal":
            current = current - self.capacitive_current_prev[slot]
        return current, factor * capacitance

    def advance_charge_state(self, x: torch.Tensor) -> None:
        """Commit the charge integrator history after an accepted step."""

        if self.dt is None:
            return
        for slot, compiled in enumerate(self.devices):
            v = compiled.terminal_voltage(x)
            current, _ = self._charge_companion(slot, compiled, v)
            q, _ = compiled.charge(v)
            self.charge_prev[slot] = q
            self.capacitive_current_prev[slot] = current

    def _device_current(
        self, slot: int, compiled: CompiledNonlinearDevice, v: torch.Tensor
    ) -> torch.Tensor:
        current, _ = compiled.conduction(v)
        if self.dt is not None:
            cap_current, _ = self._charge_companion(slot, compiled, v)
            current = current + cap_current
        return current

    def true_residual(self, x: torch.Tensor, *, gmin: float = 0.0) -> torch.Tensor:
        """KCL residual ``F(x)`` using the exact device laws (no limiting)."""

        residual = self.conductance @ x - self.injection
        if gmin:
            residual = residual + gmin * x
        for slot, compiled in enumerate(self.devices):
            v = compiled.terminal_voltage(x)
            current = self._device_current(slot, compiled, v)
            if not bool(torch.all(torch.isfinite(current))):
                bad = compiled.names[int(torch.nonzero(~torch.isfinite(current))[0, 0])]
                raise NonlinearDeviceError(
                    f"Nonlinear device {bad!r} produced a non-finite current at v={v.tolist()}."
                )
            _scatter_currents(residual, compiled.positive, compiled.negative, current, sign=1.0)
        return residual

    def residual_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.injection.abs().max() if self.injection.numel() else self.injection.new_zeros(())
        for slot, compiled in enumerate(self.devices):
            v = compiled.terminal_voltage(x)
            current = self._device_current(slot, compiled, v)
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
        if system.dt is not None:
            cap_current, cap_conductance = system._charge_companion(slot, compiled, v_lim)
            current = current + cap_current
            conductance = conductance + cap_conductance
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

    if x0.dtype != system.dtype:
        raise ValueError(
            f"newton_solve x0 dtype {x0.dtype} does not match the system dtype {system.dtype}."
        )
    if x0.device != system.device:
        raise ValueError(
            f"newton_solve x0 device {x0.device} does not match the system device {system.device}."
        )
    if x0.shape != (system.num_unknowns,):
        raise ValueError(
            f"newton_solve x0 must have shape {(system.num_unknowns,)}, got {tuple(x0.shape)}."
        )
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
    # No damped step reduced the residual within the cap. Return the starting
    # iterate so the accepted point is never worse than where the step began (the
    # last computed trial was halved again and never residual-checked); Newton
    # then stalls and the iteration cap fails closed deterministically.
    return x, max_steps


def _gmin_ladder(config: NonlinearSolveConfig) -> tuple[float, ...]:
    """Descending gmin continuation ladder ending exactly at zero.

    A shunt conductance ``gmin`` on every node makes the first (cold-start) solves
    trivially well-conditioned; the ladder decreases geometrically and the final
    solve is at ``gmin=0`` so the accepted operating point contains no residual
    artificial conductance (fail-closed boundary 2).
    """

    if config.gmin_steps <= 0 or config.gmin_start <= 0.0:
        return (0.0,)
    ladder = [config.gmin_start * (0.1 ** step) for step in range(config.gmin_steps)]
    ladder.append(0.0)
    return tuple(ladder)


def solve_dc_operating_point(
    system: NonlinearMNASystem,
    injection: torch.Tensor,
    config: NonlinearSolveConfig,
    *,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, NonlinearSolveStats]:
    """DC operating point with gmin continuation (charge devices held open).

    The stored charge is a DC open circuit, so the operating-point solve runs the
    pure-conduction system (``dt`` temporarily cleared). The gmin ladder warm-starts
    each solve from the previous converged point and the last rung is ``gmin=0`` so
    no artificial conductance survives into the returned state.
    """

    saved_dt = system.dt
    system.dt = None
    try:
        if injection.shape != (system.num_unknowns,):
            raise ValueError(
                f"injection must have shape {(system.num_unknowns,)}, got {tuple(injection.shape)}."
            )
        saved_injection = system.injection
        system.injection = injection.to(device=system.device, dtype=system.dtype)
        try:
            x = x0.clone() if x0 is not None else torch.zeros(
                system.num_unknowns, device=system.device, dtype=system.dtype
            )
            stats = None
            for gmin in _gmin_ladder(config):
                x, stats = newton_solve(system, x, config, gmin=gmin)
            return x, stats
        finally:
            system.injection = saved_injection
    finally:
        system.dt = saved_dt


def _pwl_segments(system: NonlinearMNASystem, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Per-PWL-device segment index at the node voltages ``x`` (empty otherwise)."""

    segments = []
    for compiled in system.devices:
        if compiled.kind != "piecewise_linear_iv":
            continue
        voltages = compiled.parameters["voltages"]
        v = compiled.terminal_voltage(x)
        knots = voltages.shape[-1]
        segment = torch.searchsorted(voltages, v.unsqueeze(-1), right=True).squeeze(-1) - 1
        segments.append(segment.clamp(0, knots - 2))
    return tuple(segments)


@dataclass(frozen=True)
class NonlinearTransientResult:
    """Standalone nonlinear transient trajectory and per-step Newton diagnostics.

    ``converged`` is ``True`` only when the DC start and every committed step
    converged. Under ``NonlinearSolveConfig(failure='record_and_stop')`` a
    non-convergent step truncates the run: the trajectory is populated up to and
    including the last converged step, ``converged`` is ``False``, and
    ``stopped_step`` records the index of the step that failed to converge (or
    ``0`` when the DC operating point itself did not converge). ``stopped_step``
    is ``None`` on a fully converged run.
    """

    times: torch.Tensor
    node_voltages: torch.Tensor
    kcl_residual_norm: torch.Tensor
    iterations: tuple[int, ...]
    conditions: tuple[float, ...]
    breakpoint_steps: tuple[int, ...]
    converged: bool
    stopped_step: int | None = None


def run_nonlinear_transient(
    system: NonlinearMNASystem,
    times: torch.Tensor,
    source_injection,
    config: NonlinearSolveConfig,
    *,
    integration: str = "trapezoidal",
    x0: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> NonlinearTransientResult:
    """Newton-in-the-loop transient on the standalone nonlinear MNA system.

    ``times`` is the strictly increasing sample grid; the remaining points are
    solved with a fixed step ``dt = times[k+1]-times[k]`` (a non-uniform grid is
    honoured per step). ``source_injection(t)`` returns the linear-block Norton
    injection vector at time ``t`` (solved at the step end ``t_{k+1}`` for the
    trapezoidal/BE companion). When a piecewise-linear device crosses a knot inside
    a step, that single step is re-solved with a local backward-Euler companion so
    the trapezoidal reactive companion does not ring on the conduction-slope kink
    (the standalone analogue of ``LinearMNASystem._local_backward_euler_steps``).

    ``times[0]`` sets the initial state: by default it is the DC operating point
    (``solve_dc_operating_point``, gmin/source continuation, ``x0`` warm start);
    pass ``initial_state`` to prescribe the ``t=0`` node voltages directly (an
    initial-condition run such as a charged-capacitor discharge), which seeds the
    charge integrator from that state instead of the DC steady state.
    """

    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be a 1-D grid with at least two samples.")
    if integration not in ("trapezoidal", "backward_euler"):
        raise ValueError("integration must be 'trapezoidal' or 'backward_euler'.")

    dc_converged = True
    if initial_state is not None:
        if initial_state.shape != (system.num_unknowns,):
            raise ValueError(
                f"initial_state must have shape {(system.num_unknowns,)}, got {tuple(initial_state.shape)}."
            )
        x_dc = initial_state.to(device=system.device, dtype=system.dtype).clone()
    else:
        x_dc, dc_stats = solve_dc_operating_point(
            system, source_injection(float(times[0])), config, x0=x0
        )
        # An unconverged DC start (only reachable under record_and_stop; the raise
        # path never returns) must not silently seed the whole transient.
        dc_converged = bool(dc_stats.converged)
    num_steps = times.numel()
    voltages = x_dc.new_zeros((num_steps, system.num_unknowns))
    voltages[0] = x_dc
    residual_norm = x_dc.new_zeros((num_steps,))
    iterations = [0]
    conditions = [float("inf")]
    breakpoint_steps: list[int] = []

    if not dc_converged:
        # DC operating point did not converge: the t=0 sample is invalid, so the
        # trajectory is truncated at step 0 before any transient step is taken.
        return NonlinearTransientResult(
            times=times,
            node_voltages=voltages,
            kcl_residual_norm=residual_norm,
            iterations=tuple(iterations),
            conditions=tuple(conditions),
            breakpoint_steps=tuple(breakpoint_steps),
            converged=False,
            stopped_step=0,
        )

    system.init_transient(x_dc, float(times[1] - times[0]), integration)
    x = x_dc
    stopped_step: int | None = None
    for step in range(1, num_steps):
        dt = float(times[step] - times[step - 1])
        system.dt = dt
        system.injection = source_injection(float(times[step])).to(device=system.device, dtype=system.dtype)
        # First step: run backward Euler even when the caller asked for trapezoidal.
        # Trapezoidal needs the prior capacitive-current history i_cap^0, which is
        # unknown for a prescribed non-DC initial state (a charged capacitor at
        # t=0 carries a nonzero displacement current). A BE first step needs no
        # history, is unconditionally stable, and hands the exact i_cap^1 to the
        # trapezoidal steps that follow (the standard TR startup), so the run does
        # not carry the first-step artifact of assuming i_cap^0 = 0.
        step_integration = "backward_euler" if step == 1 else integration
        system.integration = step_integration
        pre_segments = _pwl_segments(system, x)
        candidate, stats = newton_solve(system, x, config)
        post_segments = _pwl_segments(system, candidate)
        crossed = any(bool(torch.any(a != b)) for a, b in zip(pre_segments, post_segments))
        breakpoint_step = crossed and step_integration == "trapezoidal"
        if breakpoint_step:
            # Local backward-Euler breakpoint step: recompute this step from the
            # committed prior state with the BE companion so the trapezoidal
            # reactive companion does not ring on the conduction-slope kink.
            system.integration = "backward_euler"
            candidate, stats = newton_solve(system, x, config)
            breakpoint_steps.append(step)
        # A non-convergent step (only reachable under record_and_stop; the raise
        # path never returns) must not be committed as a valid sample nor advance
        # the charge history. Truncate the trajectory at the prior converged step.
        if not stats.converged:
            stopped_step = step
            break
        x = candidate
        # Measure the converged KCL residual and advance the charge integrator
        # under the SAME companion the step was solved with (backward Euler for a
        # breakpoint step), *before* advancing rewrites charge_prev to q(x) (which
        # would zero the companion and report a false residual). Restore the
        # requested companion only after the breakpoint step is committed.
        residual_norm[step] = torch.linalg.vector_norm(system.true_residual(x))
        system.advance_charge_state(x)
        if breakpoint_step:
            system.integration = integration
        voltages[step] = x
        iterations.append(stats.iterations)
        conditions.append(stats.condition_estimate)

    return NonlinearTransientResult(
        times=times,
        node_voltages=voltages,
        kcl_residual_norm=residual_norm,
        iterations=tuple(iterations),
        conditions=tuple(conditions),
        breakpoint_steps=tuple(breakpoint_steps),
        converged=stopped_step is None,
        stopped_step=stopped_step,
    )


@dataclass(frozen=True)
class MultistartReport:
    """Distinct DC operating points found from a seed sweep (multistability audit)."""

    roots: tuple[torch.Tensor, ...]
    seeds: tuple[torch.Tensor, ...]
    seed_to_root: tuple[int, ...]

    @property
    def num_operating_points(self) -> int:
        return len(self.roots)


def multistart_dc(
    system: NonlinearMNASystem,
    injection: torch.Tensor,
    seeds,
    config: NonlinearSolveConfig,
    *,
    dedup_tolerance: float = 1.0e-6,
) -> MultistartReport:
    """Run the DC solve from several seeds and report the distinct operating points.

    Devices with an incremental-negative-resistance region (a PWL curve with a
    negative-slope segment, or a load line crossing an N-shaped curve) admit
    multiple operating points; a single Newton run only reports the branch its
    seed falls into. This sweeps seeds and deduplicates the converged roots so a
    caller can see whether the operating point is unique or multistable.
    """

    seeds = tuple(seed.to(device=system.device, dtype=system.dtype) for seed in seeds)
    roots: list[torch.Tensor] = []
    seed_to_root: list[int] = []
    for seed in seeds:
        try:
            root, stats = solve_dc_operating_point(system, injection, config, x0=seed)
        except NonlinearConvergenceError:
            # Benign: this seed's Newton run did not settle on a root. A
            # NonlinearDeviceError (a non-finite device evaluation) is a
            # device-model bug, not a missed operating point, so it is left to
            # propagate rather than being masked as seed_to_root=-1.
            seed_to_root.append(-1)
            continue
        if not stats.converged:
            seed_to_root.append(-1)
            continue
        matched = -1
        for index, existing in enumerate(roots):
            if float(torch.linalg.vector_norm(root - existing)) <= dedup_tolerance:
                matched = index
                break
        if matched < 0:
            matched = len(roots)
            roots.append(root)
        seed_to_root.append(matched)
    return MultistartReport(roots=tuple(roots), seeds=seeds, seed_to_root=tuple(seed_to_root))


__all__ = [
    "CompiledNonlinearDevice",
    "MultistartReport",
    "NonlinearConvergenceError",
    "NonlinearDeviceError",
    "NonlinearMNASystem",
    "NonlinearSolveStats",
    "NonlinearTransientResult",
    "compile_nonlinear_devices",
    "multistart_dc",
    "newton_solve",
    "run_nonlinear_transient",
    "solve_dc_operating_point",
]
