from __future__ import annotations

import ast
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import shlex
from typing import ClassVar, Literal, Protocol, runtime_checkable

import torch


def _name(value, *, kind: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{kind} name must not be empty.")
    if any(character.isspace() for character in resolved):
        raise ValueError(f"{kind} name must not contain whitespace.")
    return resolved


def _scalar_tensor(value, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.get_default_dtype())
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor or scalar value.")
    if tensor.is_complex():
        raise ValueError(f"{name} must be real.")
    if not bool(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite.")
    return tensor


@dataclass(frozen=True)
class CircuitNode:
    """Named circuit node; node ``0`` is the unique ground node."""

    name: str

    def __post_init__(self):
        object.__setattr__(self, "name", _name(self.name, kind="Circuit node"))

    @property
    def is_ground(self) -> bool:
        return self.name == "0"


@dataclass(frozen=True)
class PortBinding:
    port_name: str
    positive: CircuitNode
    negative: CircuitNode

    def __post_init__(self):
        object.__setattr__(self, "port_name", _name(self.port_name, kind="Port"))
        if not isinstance(self.positive, CircuitNode) or not isinstance(self.negative, CircuitNode):
            raise TypeError("Circuit port bindings require CircuitNode terminals.")
        if self.positive == self.negative:
            raise ValueError("Circuit port binding terminals must be distinct.")


@dataclass(frozen=True)
class MNAConfig:
    integration: Literal["trapezoidal", "backward_euler"] = "trapezoidal"
    pivot_tolerance: float = 1.0e-12
    regularization: Literal["error"] = "error"
    initialization: Literal["dc", "zero"] = "dc"
    diagnostics: Literal["basic", "full"] = "basic"
    dense_unknown_limit: int = 256

    def __post_init__(self):
        if self.integration not in ("trapezoidal", "backward_euler"):
            raise ValueError("integration must be 'trapezoidal' or 'backward_euler'.")
        if not math.isfinite(self.pivot_tolerance) or self.pivot_tolerance <= 0.0:
            raise ValueError("pivot_tolerance must be finite and positive.")
        if self.regularization != "error":
            raise ValueError("regularization currently supports only the explicit 'error' policy.")
        if self.initialization not in ("dc", "zero"):
            raise ValueError("initialization must be 'dc' or 'zero'.")
        if self.diagnostics not in ("basic", "full"):
            raise ValueError("diagnostics must be 'basic' or 'full'.")
        if not isinstance(self.dense_unknown_limit, int) or isinstance(self.dense_unknown_limit, bool) or self.dense_unknown_limit < 1:
            raise ValueError("dense_unknown_limit must be a positive integer.")


@dataclass(frozen=True)
class PulseWaveform:
    initial: torch.Tensor
    pulsed: torch.Tensor
    delay: torch.Tensor
    rise: torch.Tensor
    fall: torch.Tensor
    width: torch.Tensor
    period: torch.Tensor

    def __init__(self, initial, pulsed, delay=0.0, rise=0.0, fall=0.0, width=0.0, period=0.0):
        values = {
            "initial": initial,
            "pulsed": pulsed,
            "delay": delay,
            "rise": rise,
            "fall": fall,
            "width": width,
            "period": period,
        }
        for name, value in values.items():
            tensor = _scalar_tensor(value, name=name)
            if name not in ("initial", "pulsed") and bool(tensor < 0.0):
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, tensor)


@dataclass(frozen=True)
class SineWaveform:
    offset: torch.Tensor
    amplitude: torch.Tensor
    frequency: torch.Tensor
    delay: torch.Tensor
    damping: torch.Tensor
    phase_degrees: torch.Tensor

    def __init__(self, offset, amplitude, frequency, delay=0.0, damping=0.0, phase_degrees=0.0):
        values = {
            "offset": offset,
            "amplitude": amplitude,
            "frequency": frequency,
            "delay": delay,
            "damping": damping,
            "phase_degrees": phase_degrees,
        }
        for name, value in values.items():
            tensor = _scalar_tensor(value, name=name)
            if name in ("frequency", "delay", "damping") and bool(tensor < 0.0):
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, tensor)


@dataclass(frozen=True)
class PiecewiseLinearWaveform:
    times: torch.Tensor
    values: torch.Tensor

    def __init__(self, times, values):
        times_tensor = torch.as_tensor(times)
        values_tensor = torch.as_tensor(values, device=times_tensor.device)
        if times_tensor.ndim != 1 or values_tensor.ndim != 1 or times_tensor.numel() != values_tensor.numel():
            raise ValueError("PWL times and values must be one-dimensional arrays with equal length.")
        if times_tensor.numel() < 2:
            raise ValueError("PWL requires at least two time/value pairs.")
        if times_tensor.is_complex() or values_tensor.is_complex():
            raise ValueError("PWL times and values must be real.")
        if not bool(torch.all(torch.isfinite(times_tensor))) or not bool(torch.all(torch.isfinite(values_tensor))):
            raise ValueError("PWL times and values must be finite.")
        if not bool(torch.all(times_tensor[1:] > times_tensor[:-1])):
            raise ValueError("PWL times must be strictly increasing.")
        object.__setattr__(self, "times", times_tensor)
        object.__setattr__(self, "values", values_tensor)


SourceWaveform = PulseWaveform | SineWaveform | PiecewiseLinearWaveform


def _device_terminals(name, positive, negative):
    resolved_name = _name(name, kind="Circuit device")
    if not isinstance(positive, CircuitNode) or not isinstance(negative, CircuitNode):
        raise TypeError("Circuit device terminals must be CircuitNode instances.")
    if positive == negative:
        raise ValueError("Circuit device terminals must be distinct.")
    return resolved_name, positive, negative


class _CircuitDeviceContract:
    _terminal_names = ("positive", "negative")
    _parameter_names: tuple[str, ...] = ()

    @property
    def terminals(self) -> tuple[CircuitNode, ...]:
        return tuple(getattr(self, name) for name in self._terminal_names)

    @property
    def parameters(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._parameter_names}

    @property
    def initial_condition(self):
        return None


@runtime_checkable
class CircuitDevice(Protocol):
    name: str

    @property
    def terminals(self) -> tuple[CircuitNode, ...]: ...

    @property
    def parameters(self) -> dict[str, object]: ...

    @property
    def initial_condition(self): ...


@dataclass(frozen=True)
class VoltageSource(_CircuitDeviceContract):
    _parameter_names = ("voltage", "waveform")
    name: str
    positive: CircuitNode
    negative: CircuitNode
    voltage: torch.Tensor
    waveform: SourceWaveform | None = None
    kind: str = field(default="voltage_source", init=False)

    def __init__(self, name, positive, negative, voltage=0.0, *, waveform=None):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        if waveform is not None and not isinstance(waveform, (PulseWaveform, SineWaveform, PiecewiseLinearWaveform)):
            raise TypeError("waveform must be PulseWaveform, SineWaveform, or PiecewiseLinearWaveform.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "voltage", _scalar_tensor(voltage, name="voltage"))
        object.__setattr__(self, "waveform", waveform)
        object.__setattr__(self, "kind", "voltage_source")


@dataclass(frozen=True)
class CurrentSource(_CircuitDeviceContract):
    _parameter_names = ("current", "waveform")
    name: str
    positive: CircuitNode
    negative: CircuitNode
    current: torch.Tensor
    waveform: SourceWaveform | None = None
    kind: str = field(default="current_source", init=False)

    def __init__(self, name, positive, negative, current=0.0, *, waveform=None):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        if waveform is not None and not isinstance(waveform, (PulseWaveform, SineWaveform, PiecewiseLinearWaveform)):
            raise TypeError("waveform must be PulseWaveform, SineWaveform, or PiecewiseLinearWaveform.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "current", _scalar_tensor(current, name="current"))
        object.__setattr__(self, "waveform", waveform)
        object.__setattr__(self, "kind", "current_source")


@dataclass(frozen=True)
class VoltageControlledVoltageSource(_CircuitDeviceContract):
    _terminal_names = ("positive", "negative", "control_positive", "control_negative")
    _parameter_names = ("gain",)
    name: str
    positive: CircuitNode
    negative: CircuitNode
    control_positive: CircuitNode
    control_negative: CircuitNode
    gain: torch.Tensor
    kind: str = field(default="vcvs", init=False)

    def __init__(self, name, positive, negative, control_positive, control_negative, gain):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        if not isinstance(control_positive, CircuitNode) or not isinstance(control_negative, CircuitNode):
            raise TypeError("VCVS control terminals must be CircuitNode instances.")
        if control_positive == control_negative:
            raise ValueError("VCVS control terminals must be distinct.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "control_positive", control_positive)
        object.__setattr__(self, "control_negative", control_negative)
        object.__setattr__(self, "gain", _scalar_tensor(gain, name="gain"))
        object.__setattr__(self, "kind", "vcvs")


@dataclass(frozen=True)
class VoltageControlledCurrentSource(_CircuitDeviceContract):
    _terminal_names = ("positive", "negative", "control_positive", "control_negative")
    _parameter_names = ("transconductance",)
    name: str
    positive: CircuitNode
    negative: CircuitNode
    control_positive: CircuitNode
    control_negative: CircuitNode
    transconductance: torch.Tensor
    kind: str = field(default="vccs", init=False)

    def __init__(self, name, positive, negative, control_positive, control_negative, transconductance):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        if not isinstance(control_positive, CircuitNode) or not isinstance(control_negative, CircuitNode):
            raise TypeError("VCCS control terminals must be CircuitNode instances.")
        if control_positive == control_negative:
            raise ValueError("VCCS control terminals must be distinct.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "control_positive", control_positive)
        object.__setattr__(self, "control_negative", control_negative)
        object.__setattr__(self, "transconductance", _scalar_tensor(transconductance, name="transconductance"))
        object.__setattr__(self, "kind", "vccs")


@dataclass(frozen=True)
class CurrentControlledVoltageSource(_CircuitDeviceContract):
    _parameter_names = ("control_source", "transresistance")
    name: str
    positive: CircuitNode
    negative: CircuitNode
    control_source: str
    transresistance: torch.Tensor
    kind: str = field(default="ccvs", init=False)

    def __init__(self, name, positive, negative, control_source, transresistance):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "control_source", _name(control_source, kind="Control source"))
        object.__setattr__(self, "transresistance", _scalar_tensor(transresistance, name="transresistance"))
        object.__setattr__(self, "kind", "ccvs")


@dataclass(frozen=True)
class CurrentControlledCurrentSource(_CircuitDeviceContract):
    _parameter_names = ("control_source", "gain")
    name: str
    positive: CircuitNode
    negative: CircuitNode
    control_source: str
    gain: torch.Tensor
    kind: str = field(default="cccs", init=False)

    def __init__(self, name, positive, negative, control_source, gain):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "control_source", _name(control_source, kind="Control source"))
        object.__setattr__(self, "gain", _scalar_tensor(gain, name="gain"))
        object.__setattr__(self, "kind", "cccs")


@dataclass(frozen=True)
class MutualInductor(_CircuitDeviceContract):
    _terminal_names = ()
    _parameter_names = ("first_inductor", "second_inductor", "coupling")
    name: str
    first_inductor: str
    second_inductor: str
    coupling: torch.Tensor
    kind: str = field(default="mutual_inductor", init=False)

    def __init__(self, name, first_inductor, second_inductor, coupling):
        first = _name(first_inductor, kind="Inductor")
        second = _name(second_inductor, kind="Inductor")
        if first.casefold() == second.casefold():
            raise ValueError("MutualInductor requires two distinct inductors.")
        coefficient = _scalar_tensor(coupling, name="coupling")
        if bool(torch.abs(coefficient) > 1.0):
            raise ValueError("Mutual inductance coupling must satisfy |k| <= 1.")
        object.__setattr__(self, "name", _name(name, kind="Circuit device"))
        object.__setattr__(self, "first_inductor", first)
        object.__setattr__(self, "second_inductor", second)
        object.__setattr__(self, "coupling", coefficient)
        object.__setattr__(self, "kind", "mutual_inductor")


@dataclass(frozen=True)
class TimedSwitch(_CircuitDeviceContract):
    _parameter_names = (
        "transition_times",
        "initially_closed",
        "on_resistance",
        "off_resistance",
    )
    name: str
    positive: CircuitNode
    negative: CircuitNode
    transition_times: torch.Tensor
    initially_closed: bool = False
    on_resistance: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0e-6))
    off_resistance: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0e12))
    kind: str = field(default="timed_switch", init=False)

    def __init__(self, name, positive, negative, transition_times, *, initially_closed=False, on_resistance=1.0e-6, off_resistance=1.0e12):
        resolved_name, positive, negative = _device_terminals(name, positive, negative)
        times = torch.as_tensor(transition_times)
        if times.ndim != 1 or times.is_complex() or not bool(torch.all(torch.isfinite(times))):
            raise ValueError("transition_times must be a finite real one-dimensional tensor.")
        if times.numel() and (bool(torch.any(times < 0.0)) or not bool(torch.all(times[1:] > times[:-1]))):
            raise ValueError("transition_times must be non-negative and strictly increasing.")
        on = _scalar_tensor(on_resistance, name="on_resistance")
        off = _scalar_tensor(off_resistance, name="off_resistance")
        if bool(on <= 0.0) or bool(off <= on):
            raise ValueError("TimedSwitch resistances require 0 < on_resistance < off_resistance.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "transition_times", times)
        object.__setattr__(self, "initially_closed", bool(initially_closed))
        object.__setattr__(self, "on_resistance", on)
        object.__setattr__(self, "off_resistance", off)
        object.__setattr__(self, "kind", "timed_switch")


_CIRCUIT_DEVICE_TYPES = (
    VoltageSource,
    CurrentSource,
    VoltageControlledVoltageSource,
    VoltageControlledCurrentSource,
    CurrentControlledVoltageSource,
    CurrentControlledCurrentSource,
    MutualInductor,
    TimedSwitch,
)


@dataclass(frozen=True)
class CircuitData:
    """Torch-native sampled voltages, currents, power, and solver diagnostics."""

    schema_version: ClassVar[int] = 1

    circuit_name: str
    times: torch.Tensor
    node_names: tuple[str, ...]
    node_voltages: torch.Tensor
    branch_names: tuple[str, ...]
    branch_currents: torch.Tensor
    device_powers: dict[str, torch.Tensor] = field(default_factory=dict)
    energy_balance: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "circuit_name", _name(self.circuit_name, kind="Circuit"))
        times = torch.as_tensor(self.times)
        node_voltages = torch.as_tensor(self.node_voltages)
        branch_currents = torch.as_tensor(self.branch_currents)
        node_names = tuple(_name(name, kind="Circuit node") for name in self.node_names)
        branch_names = tuple(_name(name, kind="Circuit branch") for name in self.branch_names)
        if times.ndim != 1:
            raise ValueError("CircuitData.times must be one-dimensional.")
        if times.dtype not in (torch.float32, torch.float64) or times.is_complex():
            raise ValueError("CircuitData time-domain tensors must use a real floating dtype.")
        if not bool(torch.all(torch.isfinite(times))):
            raise ValueError("CircuitData.times must be finite.")
        if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
            raise ValueError("CircuitData.times must be strictly increasing.")
        if node_voltages.shape != (times.numel(), len(node_names)):
            raise ValueError("CircuitData.node_voltages must have shape [time, node].")
        if branch_currents.shape != (times.numel(), len(branch_names)):
            raise ValueError("CircuitData.branch_currents must have shape [time, branch].")
        if node_voltages.device != times.device or branch_currents.device != times.device:
            raise ValueError("CircuitData tensors must share one device.")
        if node_voltages.dtype != times.dtype or branch_currents.dtype != times.dtype:
            raise ValueError("CircuitData tensors must share one dtype.")
        if len(set(node_names)) != len(node_names):
            raise ValueError("CircuitData.node_names must be unique.")
        if len(set(branch_names)) != len(branch_names):
            raise ValueError("CircuitData.branch_names must be unique.")
        powers = {}
        for name, value in self.device_powers.items():
            resolved_name = _name(name, kind="Circuit device")
            if resolved_name in powers:
                raise ValueError("CircuitData device power names must be unique.")
            tensor = torch.as_tensor(value)
            if tensor.shape != times.shape:
                raise ValueError(f"Device power {name!r} must have shape [time].")
            if tensor.device != times.device or tensor.dtype != times.dtype:
                raise ValueError(f"Device power {name!r} must share CircuitData device and dtype.")
            powers[resolved_name] = tensor
        balance = self.energy_balance
        if balance is not None:
            balance = torch.as_tensor(balance)
            if balance.shape != times.shape:
                raise ValueError("CircuitData.energy_balance must have shape [time].")
            if balance.device != times.device or balance.dtype != times.dtype:
                raise ValueError("CircuitData.energy_balance must share CircuitData device and dtype.")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "node_names", node_names)
        object.__setattr__(self, "node_voltages", node_voltages)
        object.__setattr__(self, "branch_names", branch_names)
        object.__setattr__(self, "branch_currents", branch_currents)
        object.__setattr__(self, "device_powers", powers)
        object.__setattr__(self, "energy_balance", balance)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def node_voltage(self, name: str) -> torch.Tensor:
        try:
            index = self.node_names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"Circuit node {name!r} is not available; choices are {self.node_names}.") from exc
        return self.node_voltages[:, index]

    def branch_current(self, name: str) -> torch.Tensor:
        try:
            index = self.branch_names.index(str(name))
        except ValueError as exc:
            raise KeyError(
                f"Circuit branch {name!r} is not available; choices are {self.branch_names}."
            ) from exc
        return self.branch_currents[:, index]

    def to(self, device) -> "CircuitData":
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value

        return CircuitData(
            circuit_name=self.circuit_name,
            times=self.times.to(device=device),
            node_names=self.node_names,
            node_voltages=self.node_voltages.to(device=device),
            branch_names=self.branch_names,
            branch_currents=self.branch_currents.to(device=device),
            device_powers={name: value.to(device=device) for name, value in self.device_powers.items()},
            energy_balance=None if self.energy_balance is None else self.energy_balance.to(device=device),
            diagnostics=move(self.diagnostics),
        )

    def _snapshot(self) -> dict[str, object]:
        from .network import _detach_to_cpu, _validate_safe_persistence

        _validate_safe_persistence(self.diagnostics, path="CircuitData.diagnostics")
        return {
            "schema_version": self.schema_version,
            "data_type": "CircuitData",
            "circuit_name": self.circuit_name,
            "times": _detach_to_cpu(self.times),
            "node_names": self.node_names,
            "node_voltages": _detach_to_cpu(self.node_voltages),
            "branch_names": self.branch_names,
            "branch_currents": _detach_to_cpu(self.branch_currents),
            "device_powers": _detach_to_cpu(self.device_powers),
            "energy_balance": _detach_to_cpu(self.energy_balance),
            "diagnostics": _detach_to_cpu(self.diagnostics),
        }

    @classmethod
    def _from_snapshot(cls, payload) -> "CircuitData":
        from .network import _validate_safe_persistence

        if not isinstance(payload, dict):
            raise ValueError("CircuitData snapshot must contain a mapping payload.")
        if payload.get("data_type") != "CircuitData":
            raise ValueError("CircuitData snapshot has an invalid data_type.")
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError(
                f"Unsupported CircuitData schema_version {payload.get('schema_version')!r}."
            )
        required = {
            "circuit_name",
            "times",
            "node_names",
            "node_voltages",
            "branch_names",
            "branch_currents",
            "device_powers",
            "energy_balance",
            "diagnostics",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "CircuitData snapshot is missing required keys: "
                + ", ".join(sorted(missing))
            )
        if not isinstance(payload["device_powers"], dict):
            raise ValueError("CircuitData snapshot device_powers must be a mapping.")
        if not isinstance(payload["diagnostics"], dict):
            raise ValueError("CircuitData snapshot diagnostics must be a mapping.")
        _validate_safe_persistence(
            payload["diagnostics"],
            path="CircuitData.diagnostics",
        )
        return cls(
            circuit_name=payload["circuit_name"],
            times=payload["times"],
            node_names=tuple(payload["node_names"]),
            node_voltages=payload["node_voltages"],
            branch_names=tuple(payload["branch_names"]),
            branch_currents=payload["branch_currents"],
            device_powers=dict(payload["device_powers"]),
            energy_balance=payload["energy_balance"],
            diagnostics=dict(payload["diagnostics"]),
        )

    def save(self, path: str | Path) -> None:
        """Save a detached CPU data snapshot without the live autograd graph."""

        output_path = Path(path)
        payload = self._snapshot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "CircuitData":
        """Load a detached versioned snapshot using safe tensor-only unpickling."""

        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=True,
        )
        return cls._from_snapshot(payload)


class Circuit:
    """Declarative linear circuit graph owned by a Maxwell ``Scene``."""

    def __init__(
        self,
        name: str,
        *,
        parameters=None,
        config: MNAConfig | None = None,
        metadata=None,
    ):
        self.name = _name(name, kind="Circuit")
        self.ground = CircuitNode("0")
        self._nodes: dict[str, CircuitNode] = {"0": self.ground}
        self._node_keys: dict[str, CircuitNode] = {"0": self.ground}
        self._generated_node_keys: set[str] = set()
        self._devices: list[object] = []
        self._device_names: set[str] = set()
        self._bindings: list[PortBinding] = []
        self._binding_names: set[str] = set()
        self.parameters = dict(parameters or {})
        self.initial_conditions: dict[str, tuple[torch.Tensor, bool]] = {}
        self.config = MNAConfig() if config is None else config
        if not isinstance(self.config, MNAConfig):
            raise TypeError("config must be an MNAConfig instance.")
        self.metadata = dict(metadata or {})

    @property
    def nodes(self) -> tuple[CircuitNode, ...]:
        return tuple(self._nodes.values())

    @property
    def devices(self) -> tuple[object, ...]:
        return tuple(self._devices)

    @property
    def bindings(self) -> tuple[PortBinding, ...]:
        return tuple(self._bindings)

    def node(self, name: str) -> CircuitNode:
        resolved = _name(name, kind="Circuit node")
        if ":" in resolved:
            raise ValueError("Circuit node names must not contain the reserved ':' hierarchy separator.")
        return self._intern_node(resolved)

    def _intern_node(self, resolved: str) -> CircuitNode:
        if resolved == "0":
            return self.ground
        key = resolved.casefold()
        existing = self._node_keys.get(key)
        if existing is not None:
            return existing
        node = CircuitNode(resolved)
        self._nodes[resolved] = node
        self._node_keys[key] = node
        return node

    def _intern_generated_node(self, resolved: str) -> CircuitNode:
        key = resolved.casefold()
        existing = self._node_keys.get(key)
        if existing is not None and key not in self._generated_node_keys:
            raise ValueError(
                f"Flattened subcircuit node {resolved!r} collides with an explicit node name."
            )
        node = self._intern_node(resolved)
        self._generated_node_keys.add(key)
        return node

    def _resolve_node(self, value) -> CircuitNode:
        if isinstance(value, str):
            return self.node(value)
        if not isinstance(value, CircuitNode):
            raise TypeError("Circuit terminals must be CircuitNode instances or node names.")
        canonical = self._node_keys.get(value.name.casefold())
        if canonical is not value:
            raise ValueError(
                f"Circuit node {value.name!r} does not belong to circuit {self.name!r}."
            )
        return value

    def add(self, device):
        from .lumped import Capacitor, Inductor, Resistor
        from .circuit_devices import NONLINEAR_DEVICE_TYPES

        if not isinstance(
            device,
            (Resistor, Capacitor, Inductor, *_CIRCUIT_DEVICE_TYPES, *NONLINEAR_DEVICE_TYPES),
        ):
            raise TypeError("Circuit devices must be supported linear or nonlinear circuit device instances.")
        key = device.name.casefold()
        if key in self._device_names:
            raise ValueError(f"Circuit device name {device.name!r} is already present.")
        if isinstance(device, MutualInductor):
            terminals = ()
        else:
            terminals = tuple(
                getattr(device, terminal_name)
                for terminal_name in (
                    "positive",
                    "negative",
                    "control_positive",
                    "control_negative",
                )
                if hasattr(device, terminal_name)
            )
        for terminal in terminals:
            self._resolve_node(terminal)
        self._devices.append(device)
        self._device_names.add(key)
        return self

    def bind_port(self, port_name: str, *, positive, negative):
        resolved_name = _name(port_name, kind="Port")
        key = resolved_name
        if key in self._binding_names:
            raise ValueError(f"Port {resolved_name!r} is already bound in circuit {self.name!r}.")
        binding = PortBinding(
            port_name=resolved_name,
            positive=self._resolve_node(positive),
            negative=self._resolve_node(negative),
        )
        self._bindings.append(binding)
        self._binding_names.add(key)
        return self

    def set_initial_condition(self, node, value, *, constraint: bool = True):
        resolved = self._resolve_node(node)
        if resolved.is_ground and bool(_scalar_tensor(value, name="initial condition") != 0.0):
            raise ValueError("Ground initial voltage must be zero.")
        self.initial_conditions[resolved.name] = (
            _scalar_tensor(value, name="initial condition"),
            bool(constraint),
        )
        return self

    def compile(self, *, available_ports=None):
        from .compiler.circuits import compile_circuit_graph

        return compile_circuit_graph(self, available_ports=available_ports)

    @classmethod
    def from_spice(
        cls,
        source,
        *,
        name: str | None = None,
        parameters=None,
        include_root=None,
    ) -> "Circuit":
        path = Path(source).expanduser().resolve()
        text, root = _read_netlist_file(path, include_root=include_root)
        return parse_spice(
            text,
            name=path.stem if name is None else name,
            parameters=parameters,
            source_path=path,
            include_root=root,
        )

    def to_spice(self) -> str:
        lines = [f"* Circuit {self.name}"]
        for name, value in self.parameters.items():
            if _PARAMETER_NAME.fullmatch(str(name)) is None:
                raise ValueError(f"Circuit parameter name {name!r} cannot be serialized to SPICE.")
            lines.append(f".param {name}={_format_scalar(value)}")
        for device in self.devices:
            lines.append(_serialize_device(device))
        for node_name, (value, constraint) in self.initial_conditions.items():
            qualifier = "" if constraint else " guess"
            lines.append(f".ic V({node_name})={_format_scalar(value)}{qualifier}")
        lines.append(".end")
        return "\n".join(lines) + "\n"


_SUFFIXES = {
    "t": 1.0e12,
    "g": 1.0e9,
    "meg": 1.0e6,
    "k": 1.0e3,
    "m": 1.0e-3,
    "u": 1.0e-6,
    "n": 1.0e-9,
    "p": 1.0e-12,
    "f": 1.0e-15,
    "mil": 25.4e-6,
}
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>[A-Za-z]+)?"
)
_PARAM_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)\s*=\s*(\{[^{}]*\}|[^\s]+)")
_MAX_NETLIST_BYTES = 4 * 1024 * 1024
_MAX_LOGICAL_LINE_LENGTH = 16 * 1024
_MAX_INCLUDE_DEPTH = 32
_MAX_EXPANDED_STATEMENTS = 100_000
_MAX_INCLUDED_FILES = 1_024
_PARAMETER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _replace_spice_number(match: re.Match) -> str:
    number = match.group("number")
    suffix = match.group("suffix")
    if suffix is None:
        return number
    lowered = suffix.casefold()
    selected = next((key for key in ("meg", "mil", "t", "g", "k", "m", "u", "n", "p", "f") if lowered.startswith(key)), None)
    if selected is None:
        raise ValueError(f"Unsupported numeric suffix {suffix!r}.")
    return f"({number}*{_SUFFIXES[selected]!r})"


def _safe_expression(expression: str, parameters: dict[str, object]):
    source = expression.strip()
    if source.startswith("{") and source.endswith("}"):
        source = source[1:-1].strip()
    if not source:
        raise ValueError("SPICE expression must not be empty.")
    normalized = _NUMBER_TOKEN.sub(_replace_spice_number, source)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid SPICE expression {expression!r}.") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("SPICE expression is too complex.")
    names = {str(name).casefold(): value for name, value in parameters.items()}

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            key = node.id.casefold()
            if key not in names:
                raise ValueError(f"Unknown SPICE parameter {node.id!r}.")
            return names[key]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            exponent = float(right.detach().cpu()) if isinstance(right, torch.Tensor) else float(right)
            if not math.isfinite(exponent) or abs(exponent) > 1024.0:
                raise ValueError("SPICE exponent is outside the supported range.")
            return left**right
        raise ValueError(f"Unsupported syntax in SPICE expression {expression!r}.")

    value = evaluate(tree)
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float64)
    if tensor.ndim != 0 or tensor.is_complex() or not bool(torch.isfinite(tensor)):
        raise ValueError(f"SPICE expression {expression!r} must evaluate to a finite real scalar.")
    return value


def _logical_lines(text: str) -> list[str]:
    if "\x00" in text:
        raise ValueError("SPICE netlists must not contain NUL bytes.")
    if len(text.encode("utf-8")) > _MAX_NETLIST_BYTES:
        raise ValueError("SPICE netlist exceeds the 4 MiB input limit.")
    lines: list[str] = []
    for raw in text.splitlines():
        if len(raw) > _MAX_LOGICAL_LINE_LENGTH:
            raise ValueError("SPICE logical line exceeds the 16 KiB limit.")
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            if not lines:
                raise ValueError("SPICE continuation line has no preceding statement.")
            lines[-1] += " " + stripped[1:].strip()
            if len(lines[-1]) > _MAX_LOGICAL_LINE_LENGTH:
                raise ValueError("SPICE logical line exceeds the 16 KiB limit.")
            continue
        for marker in ("$", ";"):
            if marker in stripped:
                stripped = stripped.split(marker, 1)[0].rstrip()
        if stripped:
            lines.append(stripped)
    return lines


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _parse_assignments(text: str, parameters: dict[str, object]) -> dict[str, object]:
    matches = list(_PARAM_ASSIGNMENT.finditer(text))
    if not matches or "".join(match.group(0) for match in matches).replace(" ", "") == "":
        raise ValueError(f"Invalid SPICE parameter assignment {text!r}.")
    result = {}
    seen: set[str] = set()
    for match in matches:
        key = match.group(1)
        folded = key.casefold()
        if folded in seen:
            raise ValueError(f"Duplicate SPICE parameter assignment {key!r}.")
        seen.add(folded)
        value = _safe_expression(match.group(2), {**parameters, **result})
        result[key] = value
    remainder = _PARAM_ASSIGNMENT.sub("", text).strip()
    if remainder:
        raise ValueError(f"Invalid SPICE parameter assignment near {remainder!r}.")
    return result


@dataclass(frozen=True)
class _Subcircuit:
    name: str
    terminals: tuple[str, ...]
    defaults: dict[str, object]
    lines: tuple[str, ...]


def _validate_subcircuit_statement(line: str, parameters: dict[str, object]) -> None:
    tokens = _tokens(line)
    directive = tokens[0].casefold()
    if directive == ".param":
        parameters.update(_parse_assignments(line[len(tokens[0]):].strip(), parameters))
        return
    if directive.startswith("."):
        raise ValueError(f"Unsupported directive {tokens[0]!r} inside subcircuit.")
    designator = tokens[0][0].casefold()
    if designator in ("r", "l", "c"):
        if len(tokens) != 4:
            raise ValueError(f"Device {tokens[0]!r} requires exactly two nodes and one value.")
        _safe_expression(tokens[3], parameters)
        return
    if designator in ("v", "i"):
        if len(tokens) < 4:
            raise ValueError(f"Source {tokens[0]!r} requires two nodes and a value or waveform.")
        source_text = " ".join(tokens[3:])
        if source_text.casefold().startswith("dc "):
            source_text = source_text[3:].strip()
        if _parse_waveform(source_text, parameters) is None:
            _safe_expression(source_text, parameters)
        return
    if designator in ("e", "g"):
        if len(tokens) != 6:
            raise ValueError(f"Controlled source {tokens[0]!r} requires four nodes and one gain.")
        _safe_expression(tokens[5], parameters)
        return
    if designator in ("f", "h"):
        if len(tokens) != 5:
            raise ValueError(
                f"Controlled source {tokens[0]!r} requires two nodes, a control source, and one gain."
            )
        _safe_expression(tokens[4], parameters)
        return
    if designator == "k":
        if len(tokens) != 4:
            raise ValueError(
                f"Mutual inductor {tokens[0]!r} requires two inductor names and a coupling coefficient."
            )
        _safe_expression(tokens[3], parameters)
        return
    if designator == "x":
        assignment_start = next(
            (
                index
                for index, token in enumerate(tokens[1:], start=1)
                if "=" in token or token.casefold() == "params:"
            ),
            len(tokens),
        )
        if len(tokens[1:assignment_start]) < 2:
            raise ValueError(f"Subcircuit instance {tokens[0]!r} is incomplete.")
        assignment_text = " ".join(
            token for token in tokens[assignment_start:] if token.casefold() != "params:"
        )
        if assignment_text:
            _parse_assignments(assignment_text, parameters)
        return
    raise ValueError(f"Unsupported SPICE device {tokens[0]!r} inside subcircuit.")


def _extract_subcircuits(lines: list[str], parameters: dict[str, object]):
    top_level: list[str] = []
    subcircuits: dict[str, _Subcircuit] = {}
    scoped_parameters = dict(parameters)
    override_names = {str(name).casefold() for name in parameters}
    index = 0
    while index < len(lines):
        line = lines[index]
        tokens = _tokens(line)
        directive = tokens[0].casefold()
        if directive == ".end":
            top_level.extend(lines[index:])
            break
        if directive != ".subckt":
            top_level.append(line)
            if directive == ".param":
                parsed = _parse_assignments(line[len(tokens[0]):].strip(), scoped_parameters)
                for parameter_name, value in parsed.items():
                    if parameter_name.casefold() not in override_names:
                        scoped_parameters[parameter_name] = value
            index += 1
            continue
        if len(tokens) < 3:
            raise ValueError(".subckt requires a name and at least one terminal.")
        name = tokens[1]
        key = name.casefold()
        if key in subcircuits:
            raise ValueError(f"Duplicate subcircuit name {name!r}.")
        assignment_start = next((i for i, token in enumerate(tokens[2:], start=2) if "=" in token or token.casefold() == "params:"), len(tokens))
        terminals = tuple(tokens[2:assignment_start])
        if not terminals or len({terminal.casefold() for terminal in terminals}) != len(terminals):
            raise ValueError(f"Subcircuit {name!r} terminals must be non-empty and unique.")
        assignment_text = " ".join(token for token in tokens[assignment_start:] if token.casefold() != "params:")
        defaults = _parse_assignments(assignment_text, scoped_parameters) if assignment_text else {}
        body_parameters = {**scoped_parameters, **defaults}
        body: list[str] = []
        index += 1
        while index < len(lines):
            body_tokens = _tokens(lines[index])
            body_directive = body_tokens[0].casefold()
            if body_directive == ".ends":
                break
            if body_directive == ".subckt":
                raise ValueError("Nested .subckt declarations are not supported.")
            try:
                _validate_subcircuit_statement(lines[index], body_parameters)
            except ValueError as exc:
                raise ValueError(f"Subcircuit {name!r}: {exc}") from exc
            body.append(lines[index])
            index += 1
        if index >= len(lines):
            raise ValueError(f"Subcircuit {name!r} is missing .ends.")
        ends = _tokens(lines[index])
        if ends[0].casefold() != ".ends" or len(ends) > 2 or (len(ends) == 2 and ends[1].casefold() != key):
            raise ValueError(f".ends does not match subcircuit {name!r}.")
        subcircuits[key] = _Subcircuit(name, terminals, defaults, tuple(body))
        index += 1
    return top_level, subcircuits


def _mapped_node(circuit: Circuit, name: str, node_map: dict[str, str], prefix: str) -> CircuitNode:
    key = name.casefold()
    if key in node_map:
        return circuit._intern_node(node_map[key])
    if name == "0":
        return circuit.ground
    if prefix:
        return circuit._intern_generated_node(f"{prefix}:{name}")
    return circuit._intern_node(name)


def _hierarchical_device_name(prefix: str, local_name: str) -> str:
    return local_name if not prefix else f"{local_name[0]}{prefix}.{local_name}"


def _parse_waveform(text: str, parameters: dict[str, object]):
    match = re.fullmatch(r"(?is)\s*(PULSE|SIN|PWL)\s*\((.*)\)\s*", text)
    if match is None:
        return None
    kind = match.group(1).casefold()
    arguments = [token for token in re.split(r"[\s,]+", match.group(2).strip()) if token]
    values = [_safe_expression(token, parameters) for token in arguments]
    if kind == "pulse":
        if not 2 <= len(values) <= 7:
            raise ValueError("PULSE requires between two and seven arguments.")
        values += [0.0] * (7 - len(values))
        return PulseWaveform(*values)
    if kind == "sin":
        if not 3 <= len(values) <= 6:
            raise ValueError("SIN requires between three and six arguments.")
        values += [0.0] * (6 - len(values))
        return SineWaveform(*values)
    if len(values) < 4 or len(values) % 2:
        raise ValueError("PWL requires at least two time/value pairs.")
    return PiecewiseLinearWaveform(values[0::2], values[1::2])


def _instantiate_device(
    circuit: Circuit,
    line: str,
    *,
    parameters: dict[str, object],
    subcircuits: dict[str, _Subcircuit],
    node_map: dict[str, str],
    prefix: str,
    depth: int,
    expansion_budget: dict[str, int],
):
    from .lumped import Capacitor, Inductor, Resistor

    tokens = _tokens(line)
    if not tokens:
        return
    designator = tokens[0][0].casefold()
    local_name = tokens[0]
    if ":" in local_name:
        raise ValueError(
            f"SPICE device name {local_name!r} uses a reserved hierarchy separator."
        )
    name = _hierarchical_device_name(prefix, local_name)

    if designator == "x":
        expansion_budget["instances"] += 1
        if expansion_budget["instances"] > _MAX_EXPANDED_STATEMENTS:
            raise ValueError("Subcircuit expansion exceeds the 100000-instance limit.")
        if depth >= 32:
            raise ValueError("Subcircuit nesting exceeds the supported depth of 32.")
        assignment_start = next((i for i, token in enumerate(tokens[1:], start=1) if "=" in token or token.casefold() == "params:"), len(tokens))
        positional = tokens[1:assignment_start]
        if len(positional) < 2:
            raise ValueError(f"Subcircuit instance {name!r} is incomplete.")
        subckt_key = positional[-1].casefold()
        definition = subcircuits.get(subckt_key)
        if definition is None:
            raise ValueError(f"Subcircuit instance {name!r} references unknown subcircuit {positional[-1]!r}.")
        actual_nodes = positional[:-1]
        if len(actual_nodes) != len(definition.terminals):
            raise ValueError(
                f"Subcircuit instance {name!r} expects {len(definition.terminals)} terminals, got {len(actual_nodes)}."
            )
        child_map = {
            formal.casefold(): _mapped_node(circuit, actual, node_map, prefix).name
            for formal, actual in zip(definition.terminals, actual_nodes)
        }
        assignment_text = " ".join(token for token in tokens[assignment_start:] if token.casefold() != "params:")
        overrides = _parse_assignments(assignment_text, parameters) if assignment_text else {}
        child_parameters = {**parameters, **definition.defaults, **overrides}
        instance_path = f"{prefix}.{local_name}" if prefix else local_name
        for body_line in definition.lines:
            if body_line.startswith("."):
                if body_line.casefold().startswith(".param"):
                    child_parameters.update(_parse_assignments(body_line[6:].strip(), child_parameters))
                    continue
                raise ValueError(f"Unsupported directive {body_line.split()[0]!r} inside subcircuit {definition.name!r}.")
            _instantiate_device(
                circuit,
                body_line,
                parameters=child_parameters,
                subcircuits=subcircuits,
                node_map=child_map,
                prefix=instance_path,
                depth=depth + 1,
                expansion_budget=expansion_budget,
            )
        return

    expansion_budget["devices"] += 1
    if expansion_budget["devices"] > _MAX_EXPANDED_STATEMENTS:
        raise ValueError("Subcircuit expansion exceeds the 100000-device limit.")

    if designator in ("r", "c", "l"):
        if len(tokens) != 4:
            raise ValueError(f"Device {name!r} requires exactly two nodes and one value.")
        positive = _mapped_node(circuit, tokens[1], node_map, prefix)
        negative = _mapped_node(circuit, tokens[2], node_map, prefix)
        value = _safe_expression(tokens[3], parameters)
        device_type = {"r": Resistor, "c": Capacitor, "l": Inductor}[designator]
        keyword = {"r": "resistance", "c": "capacitance", "l": "inductance"}[designator]
        circuit.add(device_type(name, positive, negative, **{keyword: value}))
        return

    if designator in ("v", "i"):
        if len(tokens) < 4:
            raise ValueError(f"Source {name!r} requires two nodes and a value or waveform.")
        positive = _mapped_node(circuit, tokens[1], node_map, prefix)
        negative = _mapped_node(circuit, tokens[2], node_map, prefix)
        source_text = " ".join(tokens[3:])
        if source_text.casefold().startswith("dc "):
            source_text = source_text[3:].strip()
        waveform = _parse_waveform(source_text, parameters)
        dc_value = _safe_expression(source_text, parameters) if waveform is None else (
            waveform.initial if isinstance(waveform, PulseWaveform) else waveform.offset if isinstance(waveform, SineWaveform) else waveform.values[0]
        )
        source_type = VoltageSource if designator == "v" else CurrentSource
        keyword = "voltage" if designator == "v" else "current"
        circuit.add(source_type(name, positive, negative, **{keyword: dc_value}, waveform=waveform))
        return

    if designator in ("e", "g"):
        if len(tokens) != 6:
            raise ValueError(f"Controlled source {name!r} requires four nodes and one gain.")
        nodes = tuple(_mapped_node(circuit, token, node_map, prefix) for token in tokens[1:5])
        value = _safe_expression(tokens[5], parameters)
        if designator == "e":
            circuit.add(VoltageControlledVoltageSource(name, *nodes, value))
        else:
            circuit.add(VoltageControlledCurrentSource(name, *nodes, value))
        return

    if designator in ("f", "h"):
        if len(tokens) != 5:
            raise ValueError(f"Controlled source {name!r} requires two nodes, a control source, and one gain.")
        positive = _mapped_node(circuit, tokens[1], node_map, prefix)
        negative = _mapped_node(circuit, tokens[2], node_map, prefix)
        control = _hierarchical_device_name(prefix, tokens[3])
        value = _safe_expression(tokens[4], parameters)
        if designator == "f":
            circuit.add(CurrentControlledCurrentSource(name, positive, negative, control, value))
        else:
            circuit.add(CurrentControlledVoltageSource(name, positive, negative, control, value))
        return

    if designator == "k":
        if len(tokens) != 4:
            raise ValueError(f"Mutual inductor {name!r} requires two inductor names and a coupling coefficient.")
        first = _hierarchical_device_name(prefix, tokens[1])
        second = _hierarchical_device_name(prefix, tokens[2])
        circuit.add(MutualInductor(name, first, second, _safe_expression(tokens[3], parameters)))
        return

    raise ValueError(f"Unsupported SPICE device {tokens[0]!r}.")


def parse_spice(
    text: str,
    *,
    name: str = "circuit",
    parameters=None,
    source_path: Path | None = None,
    include_root: Path | None = None,
) -> Circuit:
    """Parse the supported deterministic, non-executable netlist subset."""

    if not isinstance(text, str):
        raise TypeError("SPICE input must be text.")
    lines = _logical_lines(text)
    if any(line.casefold().startswith(".include") for line in lines):
        if source_path is None or include_root is None:
            raise ValueError(".include requires a file-backed netlist and an include root sandbox.")
        source_path = Path(source_path)
        include_root = Path(include_root)
        lines = _expand_includes(
            lines,
            source_path=source_path.resolve(),
            include_root=include_root.resolve(),
            stack=(source_path.resolve(),),
            depth=0,
        )

    resolved_parameters = dict(parameters or {})
    if len({str(key).casefold() for key in resolved_parameters}) != len(resolved_parameters):
        raise ValueError("SPICE parameter overrides must have unique case-insensitive names.")
    for key, value in tuple(resolved_parameters.items()):
        if _PARAMETER_NAME.fullmatch(str(key)) is None:
            raise ValueError(f"Invalid SPICE parameter override name {key!r}.")
        resolved_parameters[key] = _scalar_tensor(value, name=f"parameter {key}")
    top_level, subcircuits = _extract_subcircuits(lines, resolved_parameters)
    circuit = Circuit(name, parameters=resolved_parameters)
    ended = False
    declared_parameters: set[str] = set()
    override_names = {str(key).casefold() for key in resolved_parameters}
    expansion_budget = {"instances": 0, "devices": 0}
    for line in top_level:
        tokens = _tokens(line)
        directive = tokens[0].casefold()
        if directive == ".param":
            if ended:
                raise ValueError("Statements after .end are not allowed.")
            parsed = _parse_assignments(line[len(tokens[0]):].strip(), circuit.parameters)
            for key, value in parsed.items():
                folded = key.casefold()
                if folded in declared_parameters:
                    raise ValueError(f"Duplicate SPICE parameter declaration {key!r}.")
                declared_parameters.add(folded)
                if folded not in override_names:
                    circuit.parameters[key] = value
            continue
        if directive == ".ic":
            if ended:
                raise ValueError("Statements after .end are not allowed.")
            match = re.fullmatch(r"(?is)\.ic\s+V\(([^)]+)\)\s*=\s*(\S+)(?:\s+(guess))?", line)
            if match is None:
                raise ValueError(".ic supports only '.ic V(node)=value' with an optional 'guess'.")
            circuit.set_initial_condition(
                circuit.node(match.group(1).strip()),
                _safe_expression(match.group(2), circuit.parameters),
                constraint=match.group(3) is None,
            )
            continue
        if directive == ".end":
            if len(tokens) != 1:
                raise ValueError(".end does not accept arguments.")
            ended = True
            continue
        if directive.startswith("."):
            raise ValueError(f"Unsupported SPICE directive {tokens[0]!r}.")
        if ended:
            raise ValueError("Statements after .end are not allowed.")
        _instantiate_device(
            circuit,
            line,
            parameters=circuit.parameters,
            subcircuits=subcircuits,
            node_map={"0": "0"},
            prefix="",
            depth=0,
            expansion_budget=expansion_budget,
        )
    return circuit


def _expand_includes(
    lines: list[str],
    *,
    source_path: Path,
    include_root: Path,
    stack: tuple[Path, ...],
    depth: int,
    budget: dict[str, int] | None = None,
) -> list[str]:
    if budget is None:
        budget = {
            "bytes": source_path.stat().st_size,
            "files": 1,
            "statements": len(lines),
        }
    if depth >= _MAX_INCLUDE_DEPTH:
        raise ValueError(f"SPICE include nesting exceeds {_MAX_INCLUDE_DEPTH} levels.")
    expanded: list[str] = []
    for line in lines:
        tokens = _tokens(line)
        if tokens[0].casefold() != ".include":
            expanded.append(line)
            continue
        if len(tokens) != 2:
            raise ValueError(".include requires exactly one relative path.")
        relative = Path(tokens[1])
        if relative.is_absolute() or relative.drive:
            raise ValueError("SPICE include paths must be relative to the including file.")
        include_path = (source_path.parent / relative).resolve()
        try:
            include_path.relative_to(include_root)
        except ValueError as exc:
            raise ValueError(f"SPICE include {tokens[1]!r} escapes the include root sandbox.") from exc
        if include_path in stack:
            chain = " -> ".join(path.name for path in (*stack, include_path))
            raise ValueError(f"SPICE include cycle detected: {chain}.")
        include_size = include_path.stat().st_size
        budget["bytes"] += include_size
        budget["files"] += 1
        if budget["bytes"] > _MAX_NETLIST_BYTES:
            raise ValueError("Expanded SPICE includes exceed the 4 MiB aggregate limit.")
        if budget["files"] > _MAX_INCLUDED_FILES:
            raise ValueError("SPICE include expansion exceeds the 1024-file limit.")
        include_text, _ = _read_netlist_file(include_path, include_root=include_root)
        included_lines = _logical_lines(include_text)
        budget["statements"] += len(included_lines)
        if budget["statements"] > _MAX_EXPANDED_STATEMENTS:
            raise ValueError("Expanded SPICE includes exceed the 100000-statement limit.")
        expanded.extend(
            _expand_includes(
                included_lines,
                source_path=include_path,
                include_root=include_root,
                stack=(*stack, include_path),
                depth=depth + 1,
                budget=budget,
            )
        )
    return expanded


def _read_netlist_file(path: Path, *, include_root=None):
    if not path.is_file():
        raise FileNotFoundError(f"SPICE netlist does not exist: {path}")
    root = path.parent.resolve() if include_root is None else Path(include_root).expanduser().resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"SPICE netlist {path} lies outside the include root sandbox {root}.") from exc
    if path.stat().st_size > _MAX_NETLIST_BYTES:
        raise ValueError("SPICE netlist exceeds the 4 MiB input limit.")
    return path.read_text(encoding="utf-8"), root


def _format_scalar(value) -> str:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 0:
        raise ValueError("Only scalar circuit values can be serialized.")
    return format(float(tensor.detach().cpu()), ".17g")


def _format_waveform(waveform: SourceWaveform) -> str:
    if isinstance(waveform, PulseWaveform):
        values = (waveform.initial, waveform.pulsed, waveform.delay, waveform.rise, waveform.fall, waveform.width, waveform.period)
        return "PULSE(" + " ".join(_format_scalar(value) for value in values) + ")"
    if isinstance(waveform, SineWaveform):
        values = (waveform.offset, waveform.amplitude, waveform.frequency, waveform.delay, waveform.damping, waveform.phase_degrees)
        return "SIN(" + " ".join(_format_scalar(value) for value in values) + ")"
    pairs = [item for pair in zip(waveform.times, waveform.values) for item in pair]
    return "PWL(" + " ".join(_format_scalar(value) for value in pairs) + ")"


def _serialize_device(device) -> str:
    from .lumped import Capacitor, Inductor, Resistor

    expected_designator = None
    if isinstance(device, Resistor):
        expected_designator = "r"
    elif isinstance(device, Capacitor):
        expected_designator = "c"
    elif isinstance(device, Inductor):
        expected_designator = "l"
    elif isinstance(device, VoltageSource):
        expected_designator = "v"
    elif isinstance(device, CurrentSource):
        expected_designator = "i"
    elif isinstance(device, VoltageControlledVoltageSource):
        expected_designator = "e"
    elif isinstance(device, VoltageControlledCurrentSource):
        expected_designator = "g"
    elif isinstance(device, CurrentControlledCurrentSource):
        expected_designator = "f"
    elif isinstance(device, CurrentControlledVoltageSource):
        expected_designator = "h"
    elif isinstance(device, MutualInductor):
        expected_designator = "k"
    if expected_designator is not None and not device.name.casefold().startswith(expected_designator):
        raise ValueError(
            f"Circuit device {device.name!r} must start with designator "
            f"{expected_designator.upper()!r} for canonical serialization."
        )

    if isinstance(device, (Resistor, Capacitor, Inductor)):
        return f"{device.name} {device.positive.name} {device.negative.name} {_format_scalar(device.value)}"
    if isinstance(device, (VoltageSource, CurrentSource)):
        value = device.voltage if isinstance(device, VoltageSource) else device.current
        source = _format_scalar(value) if device.waveform is None else _format_waveform(device.waveform)
        return f"{device.name} {device.positive.name} {device.negative.name} {source}"
    if isinstance(device, (VoltageControlledVoltageSource, VoltageControlledCurrentSource)):
        value = device.gain if isinstance(device, VoltageControlledVoltageSource) else device.transconductance
        return f"{device.name} {device.positive.name} {device.negative.name} {device.control_positive.name} {device.control_negative.name} {_format_scalar(value)}"
    if isinstance(device, (CurrentControlledVoltageSource, CurrentControlledCurrentSource)):
        value = device.transresistance if isinstance(device, CurrentControlledVoltageSource) else device.gain
        return f"{device.name} {device.positive.name} {device.negative.name} {device.control_source} {_format_scalar(value)}"
    if isinstance(device, MutualInductor):
        return f"{device.name} {device.first_inductor} {device.second_inductor} {_format_scalar(device.coupling)}"
    if isinstance(device, TimedSwitch):
        raise ValueError("TimedSwitch has no standard statement in the supported netlist subset.")
    raise TypeError(f"Cannot serialize unsupported circuit device {type(device).__name__}.")


__all__ = [
    "Circuit",
    "CircuitData",
    "CircuitDevice",
    "CircuitNode",
    "CurrentControlledCurrentSource",
    "CurrentControlledVoltageSource",
    "CurrentSource",
    "MNAConfig",
    "MutualInductor",
    "PiecewiseLinearWaveform",
    "PortBinding",
    "PulseWaveform",
    "SineWaveform",
    "TimedSwitch",
    "VoltageControlledCurrentSource",
    "VoltageControlledVoltageSource",
    "VoltageSource",
    "parse_spice",
]
