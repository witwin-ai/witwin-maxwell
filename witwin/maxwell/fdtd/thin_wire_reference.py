"""Torch reference model for the energy-paired thin-wire discretization.

This module is intentionally independent of the production stepping runtime. It
defines the numerical contract used to validate the native implementation, and it
is deliberately not a CPU fallback: nothing in the solver may import it.

The binding to the production path lives in
``tests/fdtd/thin_wire/test_thin_wire_forward.py``, which steps the compiled CUDA
network against :class:`AxisAlignedWireReference` at
``ACCEPTANCE_BUDGET.reference_rtol`` and checks the production wire CFL bound
against :meth:`AxisAlignedWireReference.maximum_stable_dt`. Keep that binding
alive: without a consumer this reference silently stops constraining anything.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6


@dataclass(frozen=True)
class AcceptanceBudget:
    """Pre-registered acceptance limits for the thin-wire implementation."""

    reference_rtol: float = 1.0e-5
    analytic_relative_error: float = 2.0e-2
    energy_charge_relative_error: float = 1.0e-2
    gradient_relative_error: float = 2.0e-2
    convergence_levels: int = 3
    no_wire_runtime_regression: float = 1.0e-2


ACCEPTANCE_BUDGET = AcceptanceBudget()


@dataclass(frozen=True)
class WirePerUnitParameters:
    """Kernel-matched transmission-line coefficients per unit length."""

    coupling_distance: torch.Tensor
    inductance: torch.Tensor
    capacitance: torch.Tensor
    wave_speed: torch.Tensor
    impedance: torch.Tensor
    coupling_method: str


@dataclass(frozen=True)
class AxisAlignedWireCoefficients:
    """Length-integrated coefficients used by the discrete wire recurrence."""

    segment_inductance: torch.Tensor
    node_capacitance: torch.Tensor


@dataclass(frozen=True)
class WireReferenceState:
    """Integer-step field/charge and preceding half-step wire current."""

    electric: torch.Tensor
    charge: torch.Tensor
    current_half: torch.Tensor


def _floating_tensor(value, *, like: torch.Tensor | None = None) -> torch.Tensor:
    is_tensor = isinstance(value, torch.Tensor)
    tensor = value if is_tensor else torch.as_tensor(value)
    if tensor.is_complex():
        raise TypeError("physical wire parameters must be real")
    if tensor.dtype == torch.bool:
        raise TypeError("physical wire parameters must not be boolean")
    if like is not None:
        return tensor.to(dtype=like.dtype, device=like.device)
    if is_tensor and tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=torch.float64)


def _require_all(condition: torch.Tensor, message: str) -> None:
    if not bool(torch.all(condition)):
        raise ValueError(message)


def _require_real_floating(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")
    _require_all(torch.isfinite(tensor), f"{name} must be finite")


def coupling_distance(
    transverse_spacing,
    *,
    method: Literal["bspline", "legacy_edge"] = "bspline",
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the geometric-mean radius of a transverse coupling kernel.

    ``bspline`` uses the BS1 x BS1 transverse support selected for the
    charge-conserving axis-aligned coupling. ``legacy_edge`` is a uniform-grid
    historical comparator and is not used by the production contract.
    """

    spacing = _floating_tensor(transverse_spacing, like=like)
    if spacing.ndim == 0 or spacing.shape[-1] != 2:
        raise ValueError("transverse_spacing must have a trailing dimension of 2")
    _require_all(torch.isfinite(spacing), "transverse_spacing must be finite")
    _require_all(spacing > 0, "transverse_spacing must be positive")
    if method == "legacy_edge":
        return 0.230 * torch.sqrt(spacing[..., 0] * spacing[..., 1])
    if method != "bspline":
        raise ValueError(f"unsupported coupling-distance method {method!r}")

    # Tensor-product midpoint quadrature of
    # exp(int BS1(u) BS1(v) log(sqrt((du*u)^2 + (dv*v)^2)) du dv).
    # An even order avoids evaluating the integrable logarithmic singularity at
    # the wire axis. This is preparation/reference work, not a stepping hot path.
    order = 768
    coordinate = (
        (torch.arange(order, dtype=spacing.dtype, device=spacing.device) + 0.5)
        * (2.0 / order)
        - 1.0
    )
    weight = torch.clamp(1.0 - coordinate.abs(), min=0.0)
    u = coordinate[:, None]
    v = coordinate[None, :]
    quadrature_weight = weight[:, None] * weight[None, :]
    du = spacing[..., 0, None, None]
    dv = spacing[..., 1, None, None]
    radius = torch.sqrt((du * u).square() + (dv * v).square())
    log_mean = (quadrature_weight * torch.log(radius)).sum(dim=(-2, -1))
    log_mean = log_mean / quadrature_weight.sum()
    return torch.exp(log_mean)


def wire_per_unit_parameters(
    radius,
    transverse_spacing,
    *,
    permittivity=EPSILON_0,
    permeability=MU_0,
    method: Literal["bspline", "legacy_edge"] = "bspline",
) -> WirePerUnitParameters:
    """Compute kernel-matched per-unit-length wire coefficients.

    The coupling distance belongs to the interpolation/deposition kernel. The
    returned inductance and capacitance are the only wire-network coefficients;
    no complementary exterior term is added to the auxiliary state.
    """

    radius_t = _floating_tensor(radius)
    spacing_t = _floating_tensor(transverse_spacing, like=radius_t)
    epsilon_t = _floating_tensor(permittivity, like=radius_t)
    mu_t = _floating_tensor(permeability, like=radius_t)
    distance = coupling_distance(spacing_t, method=method, like=radius_t)

    _require_all(torch.isfinite(radius_t), "radius must be finite")
    _require_all(torch.isfinite(epsilon_t), "permittivity must be finite")
    _require_all(torch.isfinite(mu_t), "permeability must be finite")
    _require_all(radius_t > 0, "radius must be positive")
    _require_all(epsilon_t > 0, "permittivity must be positive")
    _require_all(mu_t > 0, "permeability must be positive")
    _require_all(radius_t < distance, "radius must be below the coupling distance")

    inductance = mu_t * torch.log(distance / radius_t) / (2.0 * math.pi)
    capacitance = mu_t * epsilon_t / inductance
    wave_speed = torch.rsqrt(inductance * capacitance)
    impedance = torch.sqrt(inductance / capacitance)
    return WirePerUnitParameters(
        coupling_distance=distance,
        inductance=inductance,
        capacitance=capacitance,
        wave_speed=wave_speed,
        impedance=impedance,
        coupling_method=method,
    )


def assemble_axis_aligned_coefficients(
    parameters: WirePerUnitParameters,
    segment_lengths,
    node_dual_lengths,
) -> AxisAlignedWireCoefficients:
    """Integrate per-length coefficients onto segments and dual nodes.

    Segment current is piecewise constant, so ``L_segment = L' * length``.
    Node charge represents line charge over its dual length, so
    ``C_node = C' * dual_length``. The field sampler must likewise contain the
    oriented line-integral weights for the same physical segment.
    """

    if parameters.inductance.ndim != 0 or parameters.capacitance.ndim != 0:
        raise ValueError(
            "Phase 0 axis-aligned assembly requires scalar per-unit parameters"
        )
    segment_length_t = _floating_tensor(segment_lengths, like=parameters.inductance)
    node_length_t = _floating_tensor(node_dual_lengths, like=parameters.capacitance)
    _require_all(torch.isfinite(segment_length_t), "segment_lengths must be finite")
    _require_all(torch.isfinite(node_length_t), "node_dual_lengths must be finite")
    _require_all(segment_length_t > 0, "segment_lengths must be positive")
    _require_all(node_length_t > 0, "node_dual_lengths must be positive")
    return AxisAlignedWireCoefficients(
        segment_inductance=parameters.inductance * segment_length_t,
        node_capacitance=parameters.capacitance * node_length_t,
    )


class AxisAlignedWireReference:
    """Dense float64-sized reference for the staggered wire recurrence.

    ``incidence`` uses +1 at the segment tail and -1 at its head. ``sampling``
    maps Yee electric degrees of freedom to oriented segment electromotive force.
    Its exact transpose is used for field deposition.
    """

    def __init__(
        self,
        *,
        incidence: torch.Tensor,
        sampling: torch.Tensor,
        segment_inductance: torch.Tensor,
        node_capacitance: torch.Tensor,
        field_mass: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> None:
        tensors = (
            incidence,
            sampling,
            segment_inductance,
            node_capacitance,
            field_mass,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("reference coefficients must be torch tensors")
        for name, tensor in zip(
            (
                "incidence",
                "sampling",
                "segment_inductance",
                "node_capacitance",
                "field_mass",
            ),
            tensors,
        ):
            _require_real_floating(name, tensor)
        dtype = segment_inductance.dtype
        device = segment_inductance.device
        if any(tensor.dtype != dtype or tensor.device != device for tensor in tensors):
            raise ValueError("reference coefficients must share one dtype and device")
        if incidence.ndim != 2 or sampling.ndim != 2:
            raise ValueError("incidence and sampling must be matrices")
        node_count, segment_count = incidence.shape
        if sampling.shape[0] != segment_count:
            raise ValueError("sampling rows must match the segment count")
        if segment_inductance.shape != (segment_count,):
            raise ValueError("segment_inductance must have one value per segment")
        if node_capacitance.shape != (node_count,):
            raise ValueError("node_capacitance must have one value per node")
        if field_mass.shape != (sampling.shape[1],):
            raise ValueError("field_mass must have one value per sampled E degree")
        _require_all(segment_inductance > 0, "segment inductance must be positive")
        _require_all(node_capacitance > 0, "node capacitance must be positive")
        _require_all(field_mass > 0, "field mass must be positive")

        self.incidence = incidence
        self.sampling = sampling
        self.segment_inductance = segment_inductance
        self.node_capacitance = node_capacitance
        self.field_mass = field_mass
        self.dt = torch.as_tensor(dt, dtype=dtype, device=device)
        if self.dt.ndim != 0:
            raise ValueError("dt must be a scalar")
        _require_all(torch.isfinite(self.dt), "dt must be finite")
        _require_all(self.dt > 0, "dt must be positive")

    def _validate_state_tensor(
        self, name: str, tensor: torch.Tensor, shape: tuple[int, ...]
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        _require_real_floating(name, tensor)
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != self.segment_inductance.dtype or tensor.device != self.segment_inductance.device:
            raise ValueError("reference state must share the coefficient dtype and device")

    def _validate_state(self, state: WireReferenceState) -> None:
        self._validate_state_tensor(
            "electric", state.electric, (self.sampling.shape[1],)
        )
        self._validate_state_tensor("charge", state.charge, (self.incidence.shape[0],))
        self._validate_state_tensor(
            "current_half", state.current_half, (self.incidence.shape[1],)
        )

    def drive(self, electric: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        self._validate_state_tensor("electric", electric, (self.sampling.shape[1],))
        self._validate_state_tensor("charge", charge, (self.incidence.shape[0],))
        voltage = charge / self.node_capacitance
        return self.sampling @ electric + self.incidence.mT @ voltage

    def initialize_current_half(
        self,
        electric: torch.Tensor,
        charge: torch.Tensor,
        current_at_integer: torch.Tensor,
    ) -> torch.Tensor:
        """Back-shift an integer-time current to the preceding half step."""

        self._validate_state_tensor(
            "current_at_integer", current_at_integer, (self.incidence.shape[1],)
        )
        return current_at_integer - 0.5 * self.dt * self.drive(
            electric, charge
        ) / self.segment_inductance

    def step(self, state: WireReferenceState) -> WireReferenceState:
        """Advance one lossless staggered step using reciprocal G/G-transpose."""

        self._validate_state(state)
        current_next = state.current_half + self.dt * self.drive(
            state.electric, state.charge
        ) / self.segment_inductance
        electric_next = state.electric - self.dt * (
            self.sampling.mT @ current_next
        ) / self.field_mass
        charge_next = state.charge - self.dt * (self.incidence @ current_next)
        return WireReferenceState(electric_next, charge_next, current_next)

    def continuity_residual(
        self, before: WireReferenceState, after: WireReferenceState
    ) -> torch.Tensor:
        self._validate_state(before)
        self._validate_state(after)
        return (after.charge - before.charge) / self.dt + self.incidence @ after.current_half

    def staggered_energy(
        self, state: WireReferenceState, current_next: torch.Tensor
    ) -> torch.Tensor:
        """Exact leapfrog invariant at the integer coordinate time."""

        self._validate_state(state)
        self._validate_state_tensor(
            "current_next", current_next, (self.incidence.shape[1],)
        )
        field = 0.5 * torch.sum(self.field_mass * state.electric.square())
        charge = 0.5 * torch.sum(state.charge.square() / self.node_capacitance)
        current = 0.5 * torch.sum(
            self.segment_inductance * current_next * state.current_half
        )
        return field + charge + current

    def maximum_stable_dt(self) -> torch.Tensor:
        """Return the strict lossless leapfrog limit ``2 / omega_max``."""

        field_term = self.sampling @ (
            self.sampling.mT / self.field_mass[:, None]
        )
        charge_term = self.incidence.mT @ (
            self.incidence / self.node_capacitance[:, None]
        )
        inv_sqrt_l = torch.rsqrt(self.segment_inductance)
        operator = inv_sqrt_l[:, None] * (field_term + charge_term) * inv_sqrt_l[None, :]
        omega_squared = torch.linalg.eigvalsh(operator).amax()
        if bool(omega_squared <= 0):
            return torch.full_like(omega_squared, torch.inf)
        return 2.0 * torch.rsqrt(omega_squared)


__all__ = [
    "ACCEPTANCE_BUDGET",
    "AcceptanceBudget",
    "AxisAlignedWireCoefficients",
    "AxisAlignedWireReference",
    "EPSILON_0",
    "MU_0",
    "WirePerUnitParameters",
    "WireReferenceState",
    "assemble_axis_aligned_coefficients",
    "coupling_distance",
    "wire_per_unit_parameters",
]
