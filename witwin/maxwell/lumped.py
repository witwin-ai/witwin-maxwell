from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .sources import SourceTime, _require_length3


def _parameter_tensor(value, *, name: str, allow_zero: bool) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.tensor(value, dtype=torch.get_default_dtype())
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor or scalar value.")
    if tensor.is_complex():
        raise ValueError(f"{name} must be real.")
    if not bool(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite.")
    if allow_zero:
        if bool(tensor < 0.0):
            raise ValueError(f"{name} must be non-negative.")
    elif bool(tensor <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return tensor


def _terminals(name, positive, negative):
    resolved_name = str(name)
    if not resolved_name:
        raise ValueError("Lumped element name must not be empty.")
    resolved_positive = _require_length3("positive", positive)
    resolved_negative = _require_length3("negative", negative)
    if resolved_positive == resolved_negative:
        raise ValueError("Lumped element positive and negative terminals must be distinct.")
    return resolved_name, resolved_positive, resolved_negative


def _angular_frequency(value, reference: torch.Tensor) -> torch.Tensor:
    frequency = torch.as_tensor(value, device=reference.device)
    if frequency.is_complex():
        raise ValueError("angular_frequency must be real.")
    dtype = torch.promote_types(reference.dtype, frequency.dtype)
    return frequency.to(dtype=dtype)


def _jw(angular_frequency, reference: torch.Tensor) -> torch.Tensor:
    omega = _angular_frequency(angular_frequency, reference)
    return torch.complex(torch.zeros_like(omega), omega)


@dataclass(frozen=True)
class Resistor:
    name: str
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    resistance: torch.Tensor
    kind: str = "resistor"

    def __init__(self, name, positive, negative, resistance):
        resolved_name, resolved_positive, resolved_negative = _terminals(name, positive, negative)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", resolved_positive)
        object.__setattr__(self, "negative", resolved_negative)
        object.__setattr__(self, "resistance", _parameter_tensor(resistance, name="resistance", allow_zero=False))
        object.__setattr__(self, "kind", "resistor")

    @property
    def value(self) -> torch.Tensor:
        return self.resistance

    def impedance(self, angular_frequency) -> torch.Tensor:
        jw = _jw(angular_frequency, self.resistance)
        return self.resistance.to(dtype=jw.dtype) + torch.zeros_like(jw)


@dataclass(frozen=True)
class Capacitor:
    name: str
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    capacitance: torch.Tensor
    kind: str = "capacitor"

    def __init__(self, name, positive, negative, capacitance):
        resolved_name, resolved_positive, resolved_negative = _terminals(name, positive, negative)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", resolved_positive)
        object.__setattr__(self, "negative", resolved_negative)
        object.__setattr__(self, "capacitance", _parameter_tensor(capacitance, name="capacitance", allow_zero=False))
        object.__setattr__(self, "kind", "capacitor")

    @property
    def value(self) -> torch.Tensor:
        return self.capacitance

    def impedance(self, angular_frequency) -> torch.Tensor:
        return 1.0 / (_jw(angular_frequency, self.capacitance) * self.capacitance)


@dataclass(frozen=True)
class Inductor:
    name: str
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    inductance: torch.Tensor
    kind: str = "inductor"

    def __init__(self, name, positive, negative, inductance):
        resolved_name, resolved_positive, resolved_negative = _terminals(name, positive, negative)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", resolved_positive)
        object.__setattr__(self, "negative", resolved_negative)
        object.__setattr__(self, "inductance", _parameter_tensor(inductance, name="inductance", allow_zero=False))
        object.__setattr__(self, "kind", "inductor")

    @property
    def value(self) -> torch.Tensor:
        return self.inductance

    def impedance(self, angular_frequency) -> torch.Tensor:
        return _jw(angular_frequency, self.inductance) * self.inductance


def _optional_parameter(value, *, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = _parameter_tensor(value, name=name, allow_zero=True)
    return None if bool(tensor == 0.0) else tensor


def _is_effective(value: torch.Tensor | None) -> bool:
    return value is not None


def _validate_rlc(resistance, inductance, capacitance):
    resistance = _optional_parameter(resistance, name="r")
    inductance = _optional_parameter(inductance, name="l")
    capacitance = _optional_parameter(capacitance, name="c")
    values = tuple(value for value in (resistance, inductance, capacitance) if value is not None)
    if not values:
        raise ValueError("RLC descriptor requires at least one positive component.")
    devices = {value.device for value in values}
    if len(devices) > 1:
        raise ValueError("RLC component tensors must be on the same device.")
    return resistance, inductance, capacitance


def _rlc_reference(resistance, inductance, capacitance) -> torch.Tensor:
    values = tuple(value for value in (resistance, inductance, capacitance) if value is not None)
    reference = values[0]
    dtype = reference.dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    return reference.to(dtype=dtype)


def _rlc_impedance(resistance, inductance, capacitance, angular_frequency, *, parallel: bool) -> torch.Tensor:
    reference = _rlc_reference(resistance, inductance, capacitance)
    jw = _jw(angular_frequency, reference)
    complex_dtype = jw.dtype
    if parallel:
        admittance = torch.zeros_like(jw)
        if _is_effective(resistance):
            admittance = admittance + 1.0 / resistance.to(dtype=complex_dtype)
        if _is_effective(inductance):
            admittance = admittance + 1.0 / (jw * inductance.to(dtype=complex_dtype))
        if _is_effective(capacitance):
            admittance = admittance + jw * capacitance.to(dtype=complex_dtype)
        return 1.0 / admittance

    impedance = torch.zeros_like(jw)
    if _is_effective(resistance):
        impedance = impedance + resistance.to(dtype=complex_dtype)
    if _is_effective(inductance):
        impedance = impedance + jw * inductance.to(dtype=complex_dtype)
    if _is_effective(capacitance):
        impedance = impedance + 1.0 / (jw * capacitance.to(dtype=complex_dtype))
    return impedance


@dataclass(frozen=True)
class SeriesRLC:
    r: torch.Tensor | None = None
    l: torch.Tensor | None = None  # noqa: E741 - public circuit notation
    c: torch.Tensor | None = None
    kind: str = "series_rlc"

    def __init__(self, r=None, l=None, c=None):  # noqa: E741 - public circuit notation
        resistance, inductance, capacitance = _validate_rlc(r, l, c)
        object.__setattr__(self, "r", resistance)
        object.__setattr__(self, "l", inductance)
        object.__setattr__(self, "c", capacitance)
        object.__setattr__(self, "kind", "series_rlc")

    def impedance(self, angular_frequency) -> torch.Tensor:
        return _rlc_impedance(self.r, self.l, self.c, angular_frequency, parallel=False)

    def resonance_frequency(self) -> torch.Tensor:
        if not _is_effective(self.l) or not _is_effective(self.c):
            raise ValueError("resonance_frequency requires positive l and c.")
        return 1.0 / (2.0 * math.pi * torch.sqrt(self.l * self.c))


@dataclass(frozen=True)
class ParallelRLC:
    r: torch.Tensor | None = None
    l: torch.Tensor | None = None  # noqa: E741 - public circuit notation
    c: torch.Tensor | None = None
    kind: str = "parallel_rlc"

    def __init__(self, r=None, l=None, c=None):  # noqa: E741 - public circuit notation
        resistance, inductance, capacitance = _validate_rlc(r, l, c)
        object.__setattr__(self, "r", resistance)
        object.__setattr__(self, "l", inductance)
        object.__setattr__(self, "c", capacitance)
        object.__setattr__(self, "kind", "parallel_rlc")

    def impedance(self, angular_frequency) -> torch.Tensor:
        return _rlc_impedance(self.r, self.l, self.c, angular_frequency, parallel=True)

    def resonance_frequency(self) -> torch.Tensor:
        if not _is_effective(self.l) or not _is_effective(self.c):
            raise ValueError("resonance_frequency requires positive l and c.")
        return 1.0 / (2.0 * math.pi * torch.sqrt(self.l * self.c))


def _amplitude_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError("amplitude must be a scalar tensor or scalar value.")
        if not bool(torch.isfinite(value.real)) or (value.is_complex() and not bool(torch.isfinite(value.imag))):
            raise ValueError("amplitude must be finite.")
        return value
    dtype = torch.complex64 if isinstance(value, complex) else torch.get_default_dtype()
    return torch.tensor(value, dtype=dtype)


def _source_impedance(value):
    if isinstance(value, str):
        if value.strip().lower() != "matched":
            raise ValueError("source_impedance must be 'matched' or have a positive real part.")
        return "matched"
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError("source_impedance must be scalar.")
        if not bool(torch.isfinite(value.real)) or (value.is_complex() and not bool(torch.isfinite(value.imag))):
            raise ValueError("source_impedance must be finite.")
        if not bool(value.real > 0.0):
            raise ValueError("source_impedance real part must be positive.")
        return value
    impedance = complex(value)
    if not math.isfinite(impedance.real) or not math.isfinite(impedance.imag):
        raise ValueError("source_impedance must be finite.")
    if impedance.real <= 0.0:
        raise ValueError("source_impedance real part must be positive.")
    return float(impedance.real) if impedance.imag == 0.0 else impedance


@dataclass(frozen=True)
class PortExcitation:
    port_name: str
    amplitude: torch.Tensor
    source_impedance: str | complex | float | torch.Tensor = "matched"
    source_time: SourceTime | None = None
    kind: str = "port_excitation"

    def __init__(self, port_name, amplitude=1.0, source_impedance="matched", source_time=None):
        resolved_name = str(port_name)
        if not resolved_name:
            raise ValueError("port_name must not be empty.")
        object.__setattr__(self, "port_name", resolved_name)
        object.__setattr__(self, "amplitude", _amplitude_tensor(amplitude))
        object.__setattr__(self, "source_impedance", _source_impedance(source_impedance))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "kind", "port_excitation")


@dataclass(frozen=True)
class PortSweep:
    """Request one independent single-device FDTD run per input port."""

    ports: tuple[str, ...] | None
    amplitude: torch.Tensor
    source_time: SourceTime | None = None
    kind: str = "port_sweep"

    def __init__(self, ports=None, amplitude=1.0, source_time=None):
        if ports is None:
            resolved_ports = None
        elif isinstance(ports, str):
            resolved_ports = (ports,)
        else:
            resolved_ports = tuple(str(port) for port in ports)
        if resolved_ports is not None:
            if not resolved_ports or any(not port for port in resolved_ports):
                raise ValueError("PortSweep ports must contain non-empty names.")
            if len(set(resolved_ports)) != len(resolved_ports):
                raise ValueError("PortSweep ports must be unique.")
        if source_time is not None and not isinstance(source_time, SourceTime):
            raise TypeError("PortSweep source_time must be a SourceTime or None.")
        object.__setattr__(self, "ports", resolved_ports)
        object.__setattr__(self, "amplitude", _amplitude_tensor(amplitude))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "kind", "port_sweep")


@dataclass(frozen=True)
class RLCBranchState:
    current: torch.Tensor
    capacitor_voltage: torch.Tensor
    previous_voltage: torch.Tensor

    @classmethod
    def zeros(cls, *, dtype=torch.float32, device=None) -> "RLCBranchState":
        zero = torch.zeros((), dtype=dtype, device=device)
        return cls(current=zero, capacitor_voltage=zero.clone(), previous_voltage=zero.clone())


def _as_series_model(model) -> SeriesRLC:
    if isinstance(model, SeriesRLC):
        return model
    if isinstance(model, Resistor):
        return SeriesRLC(r=model.value)
    if isinstance(model, Capacitor):
        return SeriesRLC(c=model.value)
    if isinstance(model, Inductor):
        return SeriesRLC(l=model.value)
    raise TypeError("trapezoidal_branch_step expects R, C, L, or SeriesRLC.")


def _state_value(value, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _time_step(value, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError("dt must be a scalar.")
        if value.is_complex():
            raise ValueError("dt must be real.")
        return value.to(device=reference.device, dtype=reference.real.dtype)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("dt must be a positive finite scalar.")
    return torch.tensor(resolved, device=reference.device, dtype=reference.real.dtype)


def trapezoidal_branch_step(
    model,
    state: RLCBranchState,
    *,
    voltage,
    dt,
) -> tuple[RLCBranchState, torch.Tensor]:
    """Advance one passive series branch and return its midpoint current."""

    series = _as_series_model(model)
    reference = state.current
    next_voltage = _state_value(voltage, reference)
    step = _time_step(dt, reference)
    previous_voltage = state.previous_voltage.to(device=reference.device, dtype=reference.dtype)
    voltage_midpoint = 0.5 * (previous_voltage + next_voltage)

    resistance = (
        None
        if not _is_effective(series.r)
        else series.r.to(device=reference.device, dtype=reference.real.dtype)
    )
    inductance = (
        None
        if not _is_effective(series.l)
        else series.l.to(device=reference.device, dtype=reference.real.dtype)
    )
    capacitance = (
        None
        if not _is_effective(series.c)
        else series.c.to(device=reference.device, dtype=reference.real.dtype)
    )

    if resistance is not None and inductance is None and capacitance is None:
        midpoint_current = voltage_midpoint / resistance
        next_current = next_voltage / resistance
        next_capacitor_voltage = torch.zeros_like(next_voltage)
    elif capacitance is not None and resistance is None and inductance is None:
        midpoint_current = capacitance * (next_voltage - previous_voltage) / step
        next_current = midpoint_current
        next_capacitor_voltage = next_voltage
    elif inductance is not None and resistance is None and capacitance is None:
        next_current = state.current + step * voltage_midpoint / inductance
        midpoint_current = 0.5 * (state.current + next_current)
        next_capacitor_voltage = torch.zeros_like(next_voltage)
    elif capacitance is None:
        inductive = (
            torch.zeros((), device=reference.device, dtype=reference.real.dtype)
            if inductance is None
            else inductance / step
        )
        resistive = torch.zeros_like(inductive) if resistance is None else 0.5 * resistance
        denominator = inductive + resistive
        next_current = (
            voltage_midpoint
            + (inductive - resistive) * state.current
        ) / denominator
        midpoint_current = 0.5 * (state.current + next_current)
        next_capacitor_voltage = torch.zeros_like(next_voltage)
    else:
        inductive = (
            torch.zeros((), device=reference.device, dtype=reference.real.dtype)
            if inductance is None
            else inductance / step
        )
        resistive = torch.zeros_like(inductive) if resistance is None else 0.5 * resistance
        capacitive = step / (4.0 * capacitance)
        denominator = inductive + resistive + capacitive
        next_current = (
            voltage_midpoint
            + (inductive - resistive - capacitive) * state.current
            - state.capacitor_voltage
        ) / denominator
        midpoint_current = 0.5 * (state.current + next_current)
        next_capacitor_voltage = state.capacitor_voltage + step * midpoint_current / capacitance

    return (
        RLCBranchState(
            current=next_current,
            capacitor_voltage=next_capacitor_voltage,
            previous_voltage=next_voltage,
        ),
        midpoint_current,
    )


def branch_energy(model, state: RLCBranchState) -> torch.Tensor:
    series = _as_series_model(model)
    energy = torch.zeros((), device=state.current.device, dtype=state.current.real.dtype)
    if _is_effective(series.l):
        energy = energy + 0.5 * series.l.to(device=energy.device, dtype=energy.dtype) * torch.abs(state.current).square()
    if _is_effective(series.c):
        energy = energy + 0.5 * series.c.to(device=energy.device, dtype=energy.dtype) * torch.abs(
            state.capacitor_voltage
        ).square()
    return energy


def trapezoidal_impedance(model, angular_frequency, dt) -> torch.Tensor:
    """Return the bilinear-transform impedance of a passive lumped descriptor."""

    if isinstance(model, (Resistor, Capacitor, Inductor)):
        reference = model.value
    elif isinstance(model, (SeriesRLC, ParallelRLC)):
        reference = _rlc_reference(model.r, model.l, model.c)
    else:
        raise TypeError("trapezoidal_impedance expects R, C, L, SeriesRLC, or ParallelRLC.")
    omega = _angular_frequency(angular_frequency, reference)
    step = _time_step(dt, reference)
    mapped_omega = (2.0 / step) * torch.tan(0.5 * omega * step)
    return model.impedance(mapped_omega)


__all__ = [
    "Capacitor",
    "Inductor",
    "ParallelRLC",
    "PortExcitation",
    "RLCBranchState",
    "Resistor",
    "SeriesRLC",
    "branch_energy",
    "trapezoidal_branch_step",
    "trapezoidal_impedance",
]
