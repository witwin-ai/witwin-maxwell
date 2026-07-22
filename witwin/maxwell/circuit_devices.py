"""Public nonlinear circuit device declarations.

These are *device-terminal* nonlinear elements (diode, behavioral I-V, voltage
dependent charge) that enter a :class:`~witwin.maxwell.circuits.Circuit` through
``Circuit.add`` exactly like the linear lumped elements. They are distinct from
the nonlinear electromagnetic *material* constitutive path; a device describes a
terminal current ``i(v)`` and/or a stored charge ``q(v)`` and nothing about the
surrounding field.

Every device is a frozen dataclass that satisfies the shared
``_CircuitDeviceContract`` (``.terminals`` / ``.parameters`` /
``.initial_condition`` / ``.kind``), so scene, checkpoint, and adjoint traversal
treat it the same way they treat the linear devices.

Transistor devices (``BJT`` / ``MOSFET``) are reserved public surfaces that fail
closed until their own independent go/no-go phase gate passes; a parser or a
factory recognising a model card is not the same thing as supported physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .circuits import CircuitNode, _CircuitDeviceContract, _device_terminals, _scalar_tensor


def _positive_scalar(value, *, name: str, allow_zero: bool) -> torch.Tensor:
    tensor = _scalar_tensor(value, name=name)
    if allow_zero:
        if bool(tensor < 0.0):
            raise ValueError(f"{name} must be non-negative.")
    elif bool(tensor <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return tensor


def _vector_tensor(value, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.get_default_dtype())
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor.")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if tensor.is_complex():
        raise ValueError(f"{name} must be real.")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} must be finite.")
    return tensor


@dataclass(frozen=True)
class NonlinearSolveConfig:
    """Newton solve configuration for the nonlinear device path.

    Both convergence gates are independent and *both* must be satisfied to accept
    an iterate; a residual-only or update-only pass never counts as converged.
    """

    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-10
    update_relative_tolerance: float = 1.0e-9
    update_absolute_tolerance: float = 1.0e-12
    max_iterations: int = 20
    line_search: Literal["backtracking", "none"] = "backtracking"
    max_line_search_steps: int = 10
    failure: Literal["raise", "record_and_stop"] = "raise"
    gmin_start: float = 0.0
    gmin_steps: int = 0

    def __post_init__(self):
        for name in (
            "relative_tolerance",
            "absolute_tolerance",
            "update_relative_tolerance",
            "update_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not (value > 0.0):
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be a positive integer.")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        if int(self.max_line_search_steps) < 1:
            raise ValueError("max_line_search_steps must be a positive integer.")
        object.__setattr__(self, "max_line_search_steps", int(self.max_line_search_steps))
        if self.line_search not in ("backtracking", "none"):
            raise ValueError("line_search must be 'backtracking' or 'none'.")
        if self.failure not in ("raise", "record_and_stop"):
            raise ValueError("failure must be 'raise' or 'record_and_stop'.")
        if float(self.gmin_start) < 0.0:
            raise ValueError("gmin_start must be non-negative.")
        object.__setattr__(self, "gmin_start", float(self.gmin_start))
        if int(self.gmin_steps) < 0:
            raise ValueError("gmin_steps must be non-negative.")
        object.__setattr__(self, "gmin_steps", int(self.gmin_steps))


@dataclass(frozen=True)
class Diode(_CircuitDeviceContract):
    """Controlled Shockley junction diode.

    Conduction current ``i(v) = Is * (exp(v / (n * Vt)) - 1)`` with the thermal
    voltage ``Vt = k * T / q`` frozen from ``temperature`` at compile time. The
    optional junction capacitance contributes a stored charge and the optional
    series resistance is retained for the transient/coupled slices; both default
    to zero so the standalone conduction model is exactly analytic.
    """

    _parameter_names = (
        "saturation_current",
        "ideality",
        "series_resistance",
        "junction_capacitance",
        "temperature",
    )
    name: str
    positive: CircuitNode
    negative: CircuitNode
    saturation_current: torch.Tensor
    ideality: torch.Tensor
    series_resistance: torch.Tensor
    junction_capacitance: torch.Tensor
    temperature: torch.Tensor
    kind: str = field(default="diode", init=False)

    def __init__(
        self,
        name,
        positive,
        negative,
        *,
        saturation_current,
        ideality=1.0,
        series_resistance=0.0,
        junction_capacitance=0.0,
        temperature=300.15,
    ):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(
            self,
            "saturation_current",
            _positive_scalar(saturation_current, name="saturation_current", allow_zero=False),
        )
        object.__setattr__(self, "ideality", _positive_scalar(ideality, name="ideality", allow_zero=False))
        object.__setattr__(
            self,
            "series_resistance",
            _positive_scalar(series_resistance, name="series_resistance", allow_zero=True),
        )
        object.__setattr__(
            self,
            "junction_capacitance",
            _positive_scalar(junction_capacitance, name="junction_capacitance", allow_zero=True),
        )
        object.__setattr__(self, "temperature", _positive_scalar(temperature, name="temperature", allow_zero=False))
        object.__setattr__(self, "kind", "diode")


@dataclass(frozen=True)
class PiecewiseLinearIV(_CircuitDeviceContract):
    """Behavioral device whose current is a continuous piecewise-linear I-V curve.

    ``voltages`` must be strictly increasing; the current is linearly interpolated
    between the knots and linearly extrapolated with the end-segment slopes.
    Negative slopes are permitted (an active/incremental-negative-resistance
    region) but flagged by the compiler for multistability diagnostics.
    """

    _parameter_names = ("voltages", "currents")
    name: str
    positive: CircuitNode
    negative: CircuitNode
    voltages: torch.Tensor
    currents: torch.Tensor
    kind: str = field(default="piecewise_linear_iv", init=False)

    def __init__(self, name, positive, negative, *, voltages, currents):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        v = _vector_tensor(voltages, name="voltages")
        i = _vector_tensor(currents, name="currents")
        if v.numel() < 2:
            raise ValueError("PiecewiseLinearIV requires at least two knots.")
        if v.numel() != i.numel():
            raise ValueError("PiecewiseLinearIV voltages and currents must have equal length.")
        if not bool(torch.all(v[1:] > v[:-1])):
            raise ValueError("PiecewiseLinearIV voltages must be strictly increasing.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "voltages", v)
        object.__setattr__(self, "currents", i)
        object.__setattr__(self, "kind", "piecewise_linear_iv")


@dataclass(frozen=True)
class PolynomialIV(_CircuitDeviceContract):
    """Behavioral device with a polynomial current ``i(v) = sum_k c[k] * v**k``.

    ``coefficients`` is ordered from the constant term upward.
    """

    _parameter_names = ("coefficients",)
    name: str
    positive: CircuitNode
    negative: CircuitNode
    coefficients: torch.Tensor
    kind: str = field(default="polynomial_iv", init=False)

    def __init__(self, name, positive, negative, *, coefficients):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        c = _vector_tensor(coefficients, name="coefficients")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "coefficients", c)
        object.__setattr__(self, "kind", "polynomial_iv")


@dataclass(frozen=True)
class VoltageDependentCapacitor(_CircuitDeviceContract):
    """Voltage-dependent capacitor declared by a single-valued charge ``Q(V)``.

    The model is declared as a charge polynomial ``q(v) = sum_k a[k] * v**k`` so
    the differential capacitance ``C(v) = dq/dv`` is always the derivative of a
    consistent single-valued charge; a bare ``C(V)`` without an integrable charge
    is rejected by construction. The device carries no DC conduction current.
    """

    _parameter_names = ("q_coefficients",)
    name: str
    positive: CircuitNode
    negative: CircuitNode
    q_coefficients: torch.Tensor
    kind: str = field(default="voltage_dependent_capacitor", init=False)

    def __init__(self, name, positive, negative, *, q_coefficients):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        a = _vector_tensor(q_coefficients, name="q_coefficients")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "q_coefficients", a)
        object.__setattr__(self, "kind", "voltage_dependent_capacitor")


class BJT:
    """Reserved bipolar-junction-transistor surface (fails closed until Phase 5)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Transistor device BJT is gated behind the independent Phase 5 go/no-go "
            "transistor evaluation; the diode/behavioral nonlinear device family does "
            "not define transistor terminal physics, charge conservation, or gradients, "
            "so this surface fails closed until that gate passes."
        )


class MOSFET:
    """Reserved MOSFET surface (fails closed until Phase 5)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Transistor device MOSFET is gated behind the independent Phase 5 go/no-go "
            "transistor evaluation; the diode/behavioral nonlinear device family does "
            "not define transistor terminal physics, charge conservation, or gradients, "
            "so this surface fails closed until that gate passes."
        )


NONLINEAR_DEVICE_TYPES = (
    Diode,
    PiecewiseLinearIV,
    PolynomialIV,
    VoltageDependentCapacitor,
)


__all__ = [
    "BJT",
    "Diode",
    "MOSFET",
    "NonlinearSolveConfig",
    "PiecewiseLinearIV",
    "PolynomialIV",
    "VoltageDependentCapacitor",
]
