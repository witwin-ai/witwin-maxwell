from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping

import torch

from .network_math import (
    _broadcast_reference_impedance,
    _connect_s_matrix,
    _reflection_from_impedance,
    _require_real_reference,
    _return_loss_db,
    _s_to_y,
    _s_to_z,
    _terminate_s_matrix,
    _validate_complex_pair,
    _validate_reference_impedance,
    _vswr,
    _y_to_s,
    _z_to_s,
    power_waves_to_voltage_current,
    voltage_current_to_power_waves,
)

if TYPE_CHECKING:
    from .rational import FitReport, RationalFitConfig, RationalModel, StateSpaceNetwork


PHASOR_CONVENTION = "peak phasor with exp(-i*omega*t) time dependence"
POWER_WAVE_CONVENTION = "Kurokawa power waves normalized to sqrt(watt)"
PERSISTENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NetworkPhysicalityReport:
    """Sampled passivity and finite-band causality diagnostics.

    ``causal`` is ``None`` when the sweep does not start at DC on a uniform
    grid.  Even when available, the negative-time energy test is a finite-band
    diagnostic rather than an all-frequency causality certificate.
    """

    frequency_band: tuple[float, float]
    sample_count: int
    passive: bool
    passivity_margin: float
    max_passivity_violation: float
    stable: bool | None
    causal: bool | None
    negative_time_energy_ratio: float | None
    passivity_tolerance: float
    causality_tolerance: float
    warnings: tuple[str, ...] = ()


def _detach_to_cpu(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    return value


def _validate_safe_persistence(value: Any, *, path: str = "metadata") -> None:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, torch.Tensor)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (bool, int, float, complex, str, bytes)):
                raise TypeError(f"{path} contains an unsupported mapping key type {type(key).__name__}.")
            _validate_safe_persistence(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_persistence(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains unsupported persistence type {type(value).__name__}; "
        "use primitive values, tensors, mappings, lists, or tuples."
    )


def _load_persisted_payload(path, *, map_location, expected_type: str) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Persisted {expected_type} payload must be a mapping.")

    actual_type = payload.get("data_type")
    if actual_type != expected_type:
        raise ValueError(f"Persisted file contains {actual_type}, not {expected_type}.")

    schema_version = payload.get("schema_version")
    if schema_version != PERSISTENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {expected_type} schema_version {schema_version!r}; "
            f"expected {PERSISTENCE_SCHEMA_VERSION}."
        )
    return payload


def _validate_frequencies(frequencies: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if not isinstance(frequencies, torch.Tensor):
        raise TypeError("frequencies must be a torch.Tensor.")
    if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
        raise TypeError("frequencies must be a real floating-point tensor.")
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if frequencies.device != device:
        raise ValueError("frequencies and network data must be on the same device.")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("frequencies must contain only finite values.")
    if not bool(torch.all(frequencies >= 0.0)):
        raise ValueError("frequencies must be non-negative.")
    return frequencies


@dataclass(frozen=True)
class PortData:
    """Frequency-domain voltage/current data for one physical port.

    The final dimension of every signal is frequency. Leading dimensions may
    represent independent excitations while preserving the same contract.
    """

    port_name: str
    frequencies: torch.Tensor
    voltage: torch.Tensor
    current: torch.Tensor
    z0: Any
    direction: str = "+"
    reference_plane: float | None = None
    available_power: torch.Tensor | None = None
    mode_names: tuple[str, ...] | None = None
    beta: torch.Tensor | None = None
    characteristic_impedance: torch.Tensor | None = None
    tracking_confidence: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    phasor_convention: str = PHASOR_CONVENTION
    power_wave_convention: str = POWER_WAVE_CONVENTION

    schema_version: ClassVar[int] = PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self):
        voltage, current = _validate_complex_pair(
            self.voltage,
            self.current,
            first_name="voltage",
            second_name="current",
        )
        frequencies = _validate_frequencies(self.frequencies, device=voltage.device)
        if voltage.shape[-1] != frequencies.numel():
            raise ValueError(
                "voltage/current last dimension must equal the number of frequencies."
            )

        name = str(self.port_name)
        if not name:
            raise ValueError("port_name must not be empty.")
        direction = str(self.direction)
        if direction not in {"+", "-"}:
            raise ValueError("direction must be '+' or '-'.")

        reference = _broadcast_reference_impedance(
            self.z0,
            shape=voltage.shape,
            device=voltage.device,
            dtype=voltage.dtype,
        )
        available_power = self.available_power
        if available_power is not None:
            if not isinstance(available_power, torch.Tensor):
                available_power = torch.as_tensor(
                    available_power,
                    device=voltage.device,
                    dtype=voltage.real.dtype,
                )
            if available_power.is_complex():
                raise TypeError("available_power must be a real tensor.")
            available_power = available_power.to(
                device=voltage.device,
                dtype=voltage.real.dtype,
            )
            try:
                available_power = torch.broadcast_to(available_power, voltage.shape)
            except RuntimeError as exc:
                raise ValueError("available_power must be broadcastable to the signal shape.") from exc

        mode_names = self.mode_names
        beta = self.beta
        characteristic_impedance = self.characteristic_impedance
        tracking_confidence = self.tracking_confidence
        modal_values = (beta, characteristic_impedance, tracking_confidence)
        if mode_names is None:
            if any(value is not None for value in modal_values):
                raise ValueError(
                    "mode_names is required when modal beta, impedance, or tracking data is provided."
                )
        else:
            mode_names = tuple(str(mode_name) for mode_name in mode_names)
            if voltage.ndim < 2:
                raise ValueError("Modal PortData signals must have a mode axis before frequency.")
            if len(mode_names) != voltage.shape[-2]:
                raise ValueError("mode_names must match the signal mode dimension.")
            if any(not mode_name for mode_name in mode_names):
                raise ValueError("mode_names must not contain empty names.")
            if len(set(mode_names)) != len(mode_names):
                raise ValueError("mode_names must be unique.")
            modal_shape = (len(mode_names), frequencies.numel())

            def normalize_modal(value, *, field_name: str, real: bool):
                if value is None:
                    return None
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value, device=voltage.device)
                if value.device != voltage.device:
                    raise ValueError(f"{field_name} and port signals must be on the same device.")
                if tuple(value.shape) != modal_shape:
                    raise ValueError(f"{field_name} must have shape [M, F] = {modal_shape}.")
                if real and value.is_complex():
                    raise TypeError(f"{field_name} must be real.")
                if not value.dtype.is_floating_point and not value.is_complex():
                    raise TypeError(f"{field_name} must be floating point.")
                if not bool(torch.all(torch.isfinite(value))):
                    raise ValueError(f"{field_name} must contain only finite values.")
                return value

            beta = normalize_modal(beta, field_name="beta", real=False)
            characteristic_impedance = normalize_modal(
                characteristic_impedance,
                field_name="characteristic_impedance",
                real=False,
            )
            if characteristic_impedance is not None:
                characteristic_impedance = _validate_reference_impedance(
                    characteristic_impedance,
                    name="characteristic_impedance",
                )
            tracking_confidence = normalize_modal(
                tracking_confidence,
                field_name="tracking_confidence",
                real=True,
            )
            if tracking_confidence is not None and not bool(
                torch.all((tracking_confidence >= 0.0) & (tracking_confidence <= 1.0))
            ):
                raise ValueError("tracking_confidence must lie in [0, 1].")

        object.__setattr__(self, "port_name", name)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "voltage", voltage)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "z0", reference)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self,
            "reference_plane",
            None if self.reference_plane is None else float(self.reference_plane),
        )
        object.__setattr__(self, "available_power", available_power)
        object.__setattr__(self, "mode_names", mode_names)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "characteristic_impedance", characteristic_impedance)
        object.__setattr__(self, "tracking_confidence", tracking_confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_power_waves(
        cls,
        *,
        port_name: str,
        frequencies: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        z0,
        **kwargs,
    ) -> "PortData":
        voltage, current = power_waves_to_voltage_current(a, b, z0)
        return cls(
            port_name=port_name,
            frequencies=frequencies,
            voltage=voltage,
            current=current,
            z0=z0,
            **kwargs,
        )

    @property
    def a(self) -> torch.Tensor:
        return voltage_current_to_power_waves(self.voltage, self.current, self.z0)[0]

    @property
    def b(self) -> torch.Tensor:
        return voltage_current_to_power_waves(self.voltage, self.current, self.z0)[1]

    @property
    def z_in(self) -> torch.Tensor:
        return self.voltage / self.current

    @property
    def incident_power(self) -> torch.Tensor:
        return torch.abs(self.a).square()

    @property
    def reflected_power(self) -> torch.Tensor:
        return torch.abs(self.b).square()

    @property
    def accepted_power(self) -> torch.Tensor:
        return self.incident_power - self.reflected_power

    @property
    def delivered_power(self) -> torch.Tensor:
        return 0.5 * torch.real(self.voltage * torch.conj(self.current))

    @property
    def reflection_coefficient(self) -> torch.Tensor:
        return self.b / self.a

    @property
    def return_loss_db(self) -> torch.Tensor:
        return _return_loss_db(self.reflection_coefficient)

    @property
    def return_loss(self) -> torch.Tensor:
        return self.return_loss_db

    @property
    def vswr(self) -> torch.Tensor:
        return _vswr(self.reflection_coefficient)

    def save(self, path: str | Path):
        """Save a detached CPU snapshot.

        File persistence intentionally does not preserve the live autograd graph.
        Loaded tensors are detached and may be placed with ``load(map_location=...)``.
        Metadata may contain tensors and primitive dict/list/tuple values accepted
        by safe ``torch.load(weights_only=True)``; arbitrary Python objects are not
        part of this persistence contract.
        """

        _validate_safe_persistence(self.metadata)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": self.schema_version,
                "data_type": type(self).__name__,
                "port_name": self.port_name,
                "frequencies": _detach_to_cpu(self.frequencies),
                "voltage": _detach_to_cpu(self.voltage),
                "current": _detach_to_cpu(self.current),
                "z0": _detach_to_cpu(self.z0),
                "direction": self.direction,
                "reference_plane": self.reference_plane,
                "available_power": _detach_to_cpu(self.available_power),
                "mode_names": self.mode_names,
                "beta": _detach_to_cpu(self.beta),
                "characteristic_impedance": _detach_to_cpu(self.characteristic_impedance),
                "tracking_confidence": _detach_to_cpu(self.tracking_confidence),
                "metadata": _detach_to_cpu(self.metadata),
                "phasor_convention": self.phasor_convention,
                "power_wave_convention": self.power_wave_convention,
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "PortData":
        """Load a detached snapshot; no active autograd graph is reconstructed."""

        payload = _load_persisted_payload(
            Path(path),
            map_location=map_location,
            expected_type=cls.__name__,
        )
        return cls(
            port_name=payload["port_name"],
            frequencies=payload["frequencies"],
            voltage=payload["voltage"],
            current=payload["current"],
            z0=payload["z0"],
            direction=payload["direction"],
            reference_plane=payload["reference_plane"],
            available_power=payload["available_power"],
            mode_names=payload.get("mode_names"),
            beta=payload.get("beta"),
            characteristic_impedance=payload.get("characteristic_impedance"),
            tracking_confidence=payload.get("tracking_confidence"),
            metadata=payload["metadata"],
            phasor_convention=payload["phasor_convention"],
            power_wave_convention=payload["power_wave_convention"],
        )


def _validate_network_matrix(
    matrix: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not matrix.is_complex():
        raise TypeError(f"{name} must be a complex tensor.")
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"{name} must have shape [F, N, N].")
    frequency_count, port_count, _ = matrix.shape
    if frequency_count == 0 or port_count == 0:
        raise ValueError(f"{name} must have non-empty shape [F, N, N].")
    if not bool(torch.all(torch.isfinite(torch.real(matrix)))) or not bool(
        torch.all(torch.isfinite(torch.imag(matrix)))
    ):
        raise ValueError(f"{name} must contain only finite values.")
    resolved_frequencies = _validate_frequencies(frequencies, device=matrix.device)
    if resolved_frequencies.numel() != frequency_count:
        raise ValueError(f"{name} frequency dimension must match frequencies shape [F].")
    return matrix, resolved_frequencies, frequency_count, port_count


def _normalize_port_names(port_names, *, port_count: int) -> tuple[str, ...]:
    names = tuple(str(name) for name in port_names)
    if len(names) != port_count:
        raise ValueError(f"port_names must contain exactly {port_count} entries.")
    if any(not name for name in names):
        raise ValueError("port_names must not contain empty names.")
    if any(name != name.strip() for name in names):
        raise ValueError("port_names must not contain leading or trailing whitespace.")
    if len(set(names)) != len(names):
        raise ValueError("port_names must be unique.")
    return names


def _normalize_network_z0(
    z0,
    *,
    frequency_count: int,
    port_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(z0, device=device).to(dtype=dtype)
    target = (frequency_count, port_count)
    if value.ndim == 0:
        value = value.expand(target)
    elif tuple(value.shape) == (port_count,):
        value = value.unsqueeze(0).expand(target)
    elif tuple(value.shape) == (1, port_count):
        value = value.expand(target)
    elif tuple(value.shape) == (frequency_count, 1):
        value = value.expand(target)
    elif tuple(value.shape) != target:
        raise ValueError(f"z0 must be scalar or broadcastable to shape [F, N] = {target}.")
    return _validate_reference_impedance(value)


def _normalize_valid_columns(valid_columns, *, port_count: int, device: torch.device) -> torch.Tensor:
    if valid_columns is None:
        return torch.ones((port_count,), device=device, dtype=torch.bool)
    if not isinstance(valid_columns, torch.Tensor):
        valid_columns = torch.as_tensor(valid_columns, device=device)
    if valid_columns.dtype != torch.bool:
        raise TypeError("valid_columns must be a boolean tensor.")
    if tuple(valid_columns.shape) != (port_count,):
        raise ValueError(f"valid_columns must have shape [N] = ({port_count},).")
    return valid_columns.to(device=device)


def _resolve_single_port(port, *, port_names: tuple[str, ...], role: str = "port") -> int:
    """Resolve a port name or zero-based integer index to its index."""

    if isinstance(port, bool):
        raise TypeError(f"{role} must be a port name or a zero-based integer index.")
    if isinstance(port, int):
        if port < 0 or port >= len(port_names):
            raise ValueError(f"{role} index {port} is outside 0..{len(port_names) - 1}.")
        return port
    if isinstance(port, str):
        try:
            return port_names.index(port)
        except ValueError as exc:
            raise ValueError(f"Unknown {role} name {port!r}.") from exc
    raise TypeError(f"{role} must be a port name or a zero-based integer index.")


def _normalize_port_distances(
    distances,
    *,
    port_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(distances, device=device)
    if value.is_complex():
        raise TypeError("distances must be real.")
    value = value.to(dtype=dtype)
    if value.ndim == 0:
        value = value.expand(port_count)
    elif tuple(value.shape) != (port_count,):
        raise ValueError(f"distances must be scalar or have shape [N] = ({port_count},).")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("distances must contain only finite values.")
    return value


def _normalize_propagation_constants(
    propagation_constants,
    *,
    frequency_count: int,
    port_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(propagation_constants, device=device).to(dtype=dtype)
    target = (frequency_count, port_count)
    if value.ndim == 0:
        value = value.expand(target)
    elif tuple(value.shape) == (port_count,):
        value = value.unsqueeze(0).expand(target)
    elif tuple(value.shape) == (frequency_count, 1):
        value = value.expand(target)
    elif tuple(value.shape) != target:
        raise ValueError(
            "propagation_constants must be scalar or broadcastable to "
            f"shape [F, N] = {target}."
        )
    if not bool(torch.all(torch.isfinite(torch.real(value)))) or not bool(
        torch.all(torch.isfinite(torch.imag(value)))
    ):
        raise ValueError("propagation_constants must contain only finite values.")
    return value


def _append_transform_metadata(
    metadata: Mapping[str, Any],
    transform: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(metadata)
    history = resolved.get("network_transform_history", ())
    if not isinstance(history, (list, tuple)):
        raise TypeError("metadata['network_transform_history'] must be a list or tuple.")
    resolved["network_transform_history"] = (*history, dict(transform))
    return resolved


def _resolve_mixed_mode_pairs(
    pairs,
    *,
    port_names: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    try:
        requested = tuple(tuple(pair) for pair in pairs)
    except TypeError as exc:
        raise TypeError("pairs must be an iterable of two-port pairs.") from exc
    if not requested:
        raise ValueError("pairs must contain at least one port pair.")

    name_to_index = {name: index for index, name in enumerate(port_names)}
    resolved: list[tuple[int, int]] = []
    used: set[int] = set()
    for pair in requested:
        if len(pair) != 2:
            raise ValueError("Each mixed-mode pair must contain exactly two ports.")
        indices: list[int] = []
        for port in pair:
            if isinstance(port, str):
                if port not in name_to_index:
                    raise ValueError(f"Unknown port name {port!r} in mixed-mode pairs.")
                index = name_to_index[port]
            elif isinstance(port, int) and not isinstance(port, bool):
                index = port
                if index < 0 or index >= len(port_names):
                    raise ValueError(f"Mixed-mode port index {index} is out of range.")
            else:
                raise TypeError("Mixed-mode ports must be names or integer indices.")
            indices.append(index)
        positive, negative = indices
        if positive == negative:
            raise ValueError("A mixed-mode pair must contain two distinct ports.")
        if positive in used or negative in used:
            raise ValueError("A port may appear in only one mixed-mode pair.")
        used.update((positive, negative))
        resolved.append((positive, negative))
    return tuple(resolved)


@dataclass(frozen=True)
class NetworkData:
    """Complex N-port scattering data with fixed ``[frequency, out, in]`` order."""

    frequencies: torch.Tensor
    s: torch.Tensor
    z0: Any
    port_names: tuple[str, ...]
    valid_columns: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    phasor_convention: str = PHASOR_CONVENTION
    power_wave_convention: str = POWER_WAVE_CONVENTION

    schema_version: ClassVar[int] = PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self):
        s, frequencies, frequency_count, port_count = _validate_network_matrix(
            self.s,
            self.frequencies,
            name="s",
        )
        names = _normalize_port_names(self.port_names, port_count=port_count)
        reference = _normalize_network_z0(
            self.z0,
            frequency_count=frequency_count,
            port_count=port_count,
            device=s.device,
            dtype=s.dtype,
        )
        valid_columns = _normalize_valid_columns(
            self.valid_columns,
            port_count=port_count,
            device=s.device,
        )
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "z0", reference)
        object.__setattr__(self, "port_names", names)
        object.__setattr__(self, "valid_columns", valid_columns)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_touchstone(
        cls,
        path: str | Path,
        *,
        device=None,
        dtype: torch.dtype = torch.complex128,
    ) -> "NetworkData":
        """Read a Touchstone 1.x or 2.0 network file.

        Parsing is a non-differentiable control-plane operation. The returned
        tensors are created directly on ``device`` and use ``dtype``.
        """

        from .touchstone import read_touchstone

        return read_touchstone(path, device=device, dtype=dtype)

    @classmethod
    def from_z(
        cls,
        *,
        frequencies: torch.Tensor,
        z: torch.Tensor,
        z0,
        port_names,
        valid_columns: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
        phasor_convention: str = PHASOR_CONVENTION,
        power_wave_convention: str = POWER_WAVE_CONVENTION,
    ) -> "NetworkData":
        z, resolved_frequencies, frequency_count, port_count = _validate_network_matrix(
            z,
            frequencies,
            name="z",
        )
        names = _normalize_port_names(port_names, port_count=port_count)
        reference = _normalize_network_z0(
            z0,
            frequency_count=frequency_count,
            port_count=port_count,
            device=z.device,
            dtype=z.dtype,
        )
        valid = _normalize_valid_columns(
            valid_columns,
            port_count=port_count,
            device=z.device,
        )
        if not bool(torch.all(valid)):
            raise RuntimeError("S/Z conversion requires complete excitation columns.")
        return cls(
            frequencies=resolved_frequencies,
            s=_z_to_s(z, reference),
            z0=reference,
            port_names=names,
            valid_columns=valid,
            metadata={} if metadata is None else metadata,
            phasor_convention=phasor_convention,
            power_wave_convention=power_wave_convention,
        )

    @classmethod
    def from_y(
        cls,
        *,
        frequencies: torch.Tensor,
        y: torch.Tensor,
        z0,
        port_names,
        valid_columns: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
        phasor_convention: str = PHASOR_CONVENTION,
        power_wave_convention: str = POWER_WAVE_CONVENTION,
    ) -> "NetworkData":
        y, resolved_frequencies, frequency_count, port_count = _validate_network_matrix(
            y,
            frequencies,
            name="y",
        )
        names = _normalize_port_names(port_names, port_count=port_count)
        reference = _normalize_network_z0(
            z0,
            frequency_count=frequency_count,
            port_count=port_count,
            device=y.device,
            dtype=y.dtype,
        )
        valid = _normalize_valid_columns(
            valid_columns,
            port_count=port_count,
            device=y.device,
        )
        if not bool(torch.all(valid)):
            raise RuntimeError("S/Y conversion requires complete excitation columns.")
        return cls(
            frequencies=resolved_frequencies,
            s=_y_to_s(y, reference),
            z0=reference,
            port_names=names,
            valid_columns=valid,
            metadata={} if metadata is None else metadata,
            phasor_convention=phasor_convention,
            power_wave_convention=power_wave_convention,
        )

    @property
    def is_complete(self) -> bool:
        return bool(torch.all(self.valid_columns))

    def _require_complete(self, operation: str):
        if not self.is_complete:
            raise RuntimeError(f"{operation} requires complete excitation columns.")

    @property
    def reflection_coefficient(self) -> torch.Tensor:
        self._require_complete("reflection diagnostics")
        return torch.diagonal(self.s, dim1=-2, dim2=-1)

    @property
    def return_loss_db(self) -> torch.Tensor:
        return _return_loss_db(self.reflection_coefficient)

    @property
    def return_loss(self) -> torch.Tensor:
        return self.return_loss_db

    @property
    def vswr(self) -> torch.Tensor:
        return _vswr(self.reflection_coefficient)

    def to_z(self) -> torch.Tensor:
        self._require_complete("S/Z conversion")
        return _s_to_z(self.s, self.z0)

    def to_y(self) -> torch.Tensor:
        self._require_complete("S/Y conversion")
        return _s_to_y(self.s, self.z0)

    def validate_physicality(
        self,
        *,
        band: tuple[float, float] | None = None,
        passivity_tolerance: float = 1e-9,
        causality_tolerance: float = 1e-2,
    ) -> NetworkPhysicalityReport:
        """Return sampled passivity and finite-band causality diagnostics."""

        from .rational import check_sampled_passivity

        self._require_complete("physicality validation")
        if passivity_tolerance < 0.0 or causality_tolerance < 0.0:
            raise ValueError("physicality tolerances must be non-negative.")
        if band is None:
            mask = torch.ones_like(self.frequencies, dtype=torch.bool)
        else:
            if len(band) != 2 or not (0.0 <= band[0] < band[1]):
                raise ValueError("band must be an increasing non-negative frequency pair.")
            mask = (self.frequencies >= band[0]) & (self.frequencies <= band[1])
        if int(torch.count_nonzero(mask).item()) < 2:
            raise ValueError("physicality validation requires at least two in-band samples.")

        frequencies = self.frequencies[mask]
        response = self.s[mask]
        passivity = check_sampled_passivity(
            response,
            representation="S",
            tolerance=passivity_tolerance,
        )
        warnings: list[str] = []
        causal: bool | None = None
        negative_ratio: float | None = None
        spacing = frequencies[1:] - frequencies[:-1]
        spacing_tolerance = 64.0 * torch.finfo(frequencies.dtype).eps
        starts_at_dc = bool(
            torch.isclose(
                frequencies[0],
                torch.zeros((), dtype=frequencies.dtype, device=frequencies.device),
                rtol=0.0,
                atol=spacing_tolerance * torch.max(frequencies[-1], torch.ones_like(frequencies[-1])),
            )
        )
        uniformly_spaced = bool(
            torch.allclose(
                spacing,
                spacing[0].expand_as(spacing),
                rtol=spacing_tolerance,
                atol=spacing_tolerance * max(1.0, float(spacing[0].item())),
            )
        )
        if starts_at_dc and uniformly_spaced:
            mirrored = torch.conj(torch.flip(response[1:-1], dims=(0,)))
            spectrum = torch.cat((response, mirrored), dim=0)
            impulse = torch.fft.fft(spectrum, dim=0)
            energy = torch.abs(impulse) ** 2
            total_energy = torch.sum(energy)
            if bool(total_energy == 0.0):
                negative_ratio = 0.0
            else:
                negative_start = spectrum.shape[0] // 2 + 1
                negative_ratio = float(
                    (torch.sum(energy[negative_start:]) / total_energy).item()
                )
            if negative_ratio <= causality_tolerance:
                causal = True
            elif negative_ratio >= max(0.5, 10.0 * causality_tolerance):
                causal = False
            else:
                causal = None
                warnings.append(
                    "Causality is indeterminate because finite-band truncation leaves "
                    "ambiguous negative-time energy."
                )
            warnings.append(
                "Causality uses a finite-band negative-time energy heuristic, not an "
                "all-frequency certificate."
            )
        else:
            warnings.append(
                "Causality is indeterminate because the selected sweep must start at DC "
                "and be uniformly spaced."
            )
        return NetworkPhysicalityReport(
            frequency_band=(float(frequencies[0].item()), float(frequencies[-1].item())),
            sample_count=frequencies.numel(),
            passive=passivity.passive,
            passivity_margin=passivity.margin,
            max_passivity_violation=passivity.max_violation,
            stable=None,
            causal=causal,
            negative_time_energy_ratio=negative_ratio,
            passivity_tolerance=float(passivity_tolerance),
            causality_tolerance=float(causality_tolerance),
            warnings=tuple(warnings),
        )

    def fit_rational(
        self,
        config: RationalFitConfig | None = None,
        *,
        representation: str = "Y",
        initial_poles: torch.Tensor | None = None,
    ) -> RationalModel:
        """Fit a shared-pole rational model in the requested representation."""

        from .rational import fit_rational

        self._require_complete("rational fitting")
        representation = representation.upper()
        if representation == "S":
            values = self.s
        elif representation == "Y":
            values = self.to_y()
        elif representation == "Z":
            values = self.to_z()
        else:
            raise ValueError("representation must be 'Y', 'Z', or 'S'.")
        return fit_rational(
            self.frequencies,
            values,
            config=config,
            representation=representation,
            initial_poles=initial_poles,
        )

    def renormalize(self, z0) -> "NetworkData":
        self._require_complete("renormalization")
        reference = _normalize_network_z0(
            z0,
            frequency_count=self.s.shape[0],
            port_count=self.s.shape[1],
            device=self.s.device,
            dtype=self.s.dtype,
        )
        metadata = _append_transform_metadata(
            self.metadata,
            {
                "operation": "renormalize",
                "source_z0": self.z0.detach().clone(),
                "target_z0": reference.detach().clone(),
            },
        )
        return type(self).from_z(
            frequencies=self.frequencies,
            z=self.to_z(),
            z0=reference,
            port_names=self.port_names,
            valid_columns=self.valid_columns,
            metadata=metadata,
            phasor_convention=self.phasor_convention,
            power_wave_convention=self.power_wave_convention,
        )

    def terminate(self, port, *, gamma=None, impedance=None) -> "NetworkData":
        """Close one port with a reflection coefficient or load impedance.

        Exactly one of ``gamma`` (a reflection coefficient at the port reference
        plane) or ``impedance`` (a load impedance) must be given. Each may be a
        scalar or a length-``F`` frequency-dependent tensor. The remaining ports
        keep their order and reference impedances. For a two-port terminated on
        its second port this reduces to the closed-form input reflection
        ``S11 + S12 * gamma * S21 / (1 - S22 * gamma)``.
        """

        self._require_complete("termination")
        if (gamma is None) == (impedance is None):
            raise ValueError("Provide exactly one of gamma= or impedance=.")

        index = _resolve_single_port(port, port_names=self.port_names, role="terminated port")
        port_z0 = self.z0[:, index]

        if gamma is not None:
            reflection = torch.as_tensor(gamma, device=self.s.device).to(dtype=self.s.dtype)
        else:
            load = torch.as_tensor(impedance, device=self.s.device).to(dtype=self.s.dtype)
            reflection = _reflection_from_impedance(load, port_z0)
        if reflection.ndim == 0:
            reflection = reflection.expand(self.s.shape[0])
        elif tuple(reflection.shape) != (self.s.shape[0],):
            raise ValueError("gamma/impedance must be scalar or have shape [F].")
        if not bool(torch.all(torch.isfinite(torch.real(reflection)))) or not bool(
            torch.all(torch.isfinite(torch.imag(reflection)))
        ):
            raise ValueError("The terminating reflection coefficient must be finite.")

        keep = [i for i in range(len(self.port_names)) if i != index]
        reduced = _terminate_s_matrix(self.s, index=index, keep=keep, reflection=reflection)

        metadata = _append_transform_metadata(
            self.metadata,
            {
                "operation": "terminate",
                "terminated_port": self.port_names[index],
                "reflection": reflection.detach().clone(),
            },
        )
        return type(self)(
            frequencies=self.frequencies,
            s=reduced,
            z0=self.z0[:, keep],
            port_names=tuple(self.port_names[i] for i in keep),
            metadata=metadata,
            phasor_convention=self.phasor_convention,
            power_wave_convention=self.power_wave_convention,
        )

    def cascade(self, other: "NetworkData", *, port_map: Mapping) -> "NetworkData":
        """Connect ports of this network to ports of ``other`` and reduce them out.

        ``port_map`` maps each connected self port (name or zero-based index) to a
        connected ``other`` port. Every mapped pair is joined at a matched junction
        and removed; the result keeps this network's remaining ports first (in
        order) followed by ``other``'s remaining ports. Connected ports must share
        a real reference impedance. This is the general N-port star connection, so
        cascading a two-port through an ideal matched thru returns the original,
        and cascading two attenuators multiplies their transmissions.
        """

        if not isinstance(other, NetworkData):
            raise TypeError("cascade requires another NetworkData instance.")
        self._require_complete("cascade")
        other._require_complete("cascade")
        if self.s.device != other.s.device or self.s.dtype != other.s.dtype:
            raise ValueError("cascade requires matching device and dtype.")
        if self.frequencies.shape != other.frequencies.shape or not bool(
            torch.equal(self.frequencies, other.frequencies)
        ):
            raise ValueError("cascade requires an identical frequency grid on both networks.")
        if not isinstance(port_map, Mapping) or len(port_map) == 0:
            raise ValueError("port_map must map at least one self port to an other port.")

        na = len(self.port_names)
        nb = len(other.port_names)
        self_pairs: list[int] = []
        other_pairs: list[int] = []
        for self_port, other_port in port_map.items():
            i = _resolve_single_port(self_port, port_names=self.port_names, role="self port")
            j = _resolve_single_port(other_port, port_names=other.port_names, role="other port")
            if i in self_pairs:
                raise ValueError(f"self port {self.port_names[i]!r} is connected more than once.")
            if j in other_pairs:
                raise ValueError(f"other port {other.port_names[j]!r} is connected more than once.")
            _require_real_reference(self.z0[:, i], context="cascade")
            _require_real_reference(other.z0[:, j], context="cascade")
            tolerance = 1.0e-6 * torch.clamp(torch.abs(torch.real(self.z0[:, i])), min=1.0)
            if bool(
                torch.any(torch.abs(self.z0[:, i] - other.z0[:, j]) > tolerance)
            ):
                raise ValueError(
                    f"connected ports {self.port_names[i]!r} and {other.port_names[j]!r} "
                    "must share the same reference impedance; renormalize first."
                )
            self_pairs.append(i)
            other_pairs.append(j)

        combined = torch.zeros(
            (self.s.shape[0], na + nb, na + nb),
            device=self.s.device,
            dtype=self.s.dtype,
        )
        combined[:, :na, :na] = self.s
        combined[:, na:, na:] = other.s

        external_self = [i for i in range(na) if i not in self_pairs]
        external_other = [j for j in range(nb) if j not in other_pairs]
        external = external_self + [na + j for j in external_other]
        connected: list[int] = []
        for i, j in zip(self_pairs, other_pairs):
            connected.extend((i, na + j))

        result_names = tuple(self.port_names[i] for i in external_self) + tuple(
            other.port_names[j] for j in external_other
        )
        if len(set(result_names)) != len(result_names):
            raise ValueError(
                "cascade result would have duplicate external port names; rename ports first."
            )
        result_z0 = torch.cat(
            (
                self.z0[:, external_self],
                other.z0[:, external_other],
            ),
            dim=-1,
        )
        reduced = _connect_s_matrix(combined, external=external, connected=connected)
        metadata = _append_transform_metadata(
            self.metadata,
            {
                "operation": "cascade",
                "self_ports": tuple(self.port_names[i] for i in self_pairs),
                "other_ports": tuple(other.port_names[j] for j in other_pairs),
                "result_port_names": result_names,
            },
        )
        return type(self)(
            frequencies=self.frequencies,
            s=reduced,
            z0=result_z0,
            port_names=result_names,
            metadata=metadata,
            phasor_convention=self.phasor_convention,
            power_wave_convention=self.power_wave_convention,
        )

    def shift_reference_planes(
        self,
        distances,
        *,
        propagation_constants=None,
    ) -> "NetworkData":
        """Shift port reference planes with explicit complex propagation constants.

        ``distances`` follows the outward-positive convention. Under the package's
        ``exp(-i*omega*t)`` phasor convention, each one-way wave is multiplied by
        ``exp(1j * propagation_constants * distances)``. Consequently, each
        scattering entry receives both its output- and input-port factors.
        """

        self._require_complete("reference-plane shifting")
        distance = _normalize_port_distances(
            distances,
            port_count=self.s.shape[1],
            device=self.s.device,
            dtype=self.s.real.dtype,
        )
        resolved_propagation = propagation_constants
        if resolved_propagation is None:
            resolved_propagation = self.metadata.get("propagation_constants")
            if resolved_propagation is None:
                raise ValueError(
                    "propagation_constants must be provided when NetworkData metadata "
                    "does not contain modal propagation constants."
                )
        propagation = _normalize_propagation_constants(
            resolved_propagation,
            frequency_count=self.s.shape[0],
            port_count=self.s.shape[1],
            device=self.s.device,
            dtype=self.s.dtype,
        )
        one_way = torch.exp(1j * propagation * distance.unsqueeze(0))
        shifted = one_way.unsqueeze(-1) * self.s * one_way.unsqueeze(-2)
        metadata = _append_transform_metadata(
            self.metadata,
            {
                "operation": "shift_reference_planes",
                "distances": distance.detach().clone(),
                "propagation_constants": propagation.detach().clone(),
                "distance_convention": "outward-positive",
            },
        )
        return type(self)(
            frequencies=self.frequencies,
            s=shifted,
            z0=self.z0,
            port_names=self.port_names,
            valid_columns=self.valid_columns,
            metadata=metadata,
            phasor_convention=self.phasor_convention,
            power_wave_convention=self.power_wave_convention,
        )

    def to_mixed_mode(self, pairs, z0=None) -> "NetworkData":
        """Convert selected single-ended port pairs to differential/common modes.

        Pair outputs are ordered differential then common in the order supplied;
        unpaired ports follow in their original order. The voltage basis is
        ``Vd = Vp - Vn`` and ``Vc = (Vp + Vn) / 2``. Its inverse-transpose
        current basis gives ``Id = (Ip - In) / 2`` and ``Ic = Ip + In``, preserving
        complex power. With no explicit mixed-mode ``z0``, paired single-ended
        impedances must be equal and map to ``Zd = 2*z0`` and ``Zc = z0/2``.
        """

        self._require_complete("mixed-mode conversion")
        resolved_pairs = _resolve_mixed_mode_pairs(pairs, port_names=self.port_names)
        paired_indices = {index for pair in resolved_pairs for index in pair}
        unpaired_indices = tuple(
            index for index in range(len(self.port_names)) if index not in paired_indices
        )

        rows: list[torch.Tensor] = []
        names: list[str] = []
        default_references: list[torch.Tensor] = []
        for positive, negative in resolved_pairs:
            differential = torch.zeros(
                (len(self.port_names),),
                device=self.s.device,
                dtype=self.s.dtype,
            )
            common = torch.zeros_like(differential)
            differential[positive] = 1.0
            differential[negative] = -1.0
            common[positive] = 0.5
            common[negative] = 0.5
            rows.extend((differential, common))

            positive_name = self.port_names[positive]
            negative_name = self.port_names[negative]
            names.extend(
                (
                    f"d({positive_name},{negative_name})",
                    f"c({positive_name},{negative_name})",
                )
            )
            if z0 is None:
                positive_z0 = self.z0[:, positive]
                negative_z0 = self.z0[:, negative]
                tolerance = 10.0 * torch.finfo(self.s.real.dtype).eps
                if not bool(
                    torch.allclose(
                        positive_z0,
                        negative_z0,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                ):
                    raise ValueError(
                        "Paired single-ended reference impedances must be equal when "
                        "mixed-mode z0 is omitted; provide z0 explicitly for unequal ports."
                    )
                default_references.extend((2.0 * positive_z0, 0.5 * positive_z0))

        for index in unpaired_indices:
            row = torch.zeros(
                (len(self.port_names),),
                device=self.s.device,
                dtype=self.s.dtype,
            )
            row[index] = 1.0
            rows.append(row)
            names.append(self.port_names[index])
            if z0 is None:
                default_references.append(self.z0[:, index])

        voltage_basis = torch.stack(rows)
        source_z = self.to_z()
        mixed_z = voltage_basis @ source_z @ voltage_basis.transpose(-2, -1)
        if z0 is None:
            mixed_reference = torch.stack(default_references, dim=-1)
        else:
            mixed_reference = _normalize_network_z0(
                z0,
                frequency_count=self.s.shape[0],
                port_count=self.s.shape[1],
                device=self.s.device,
                dtype=self.s.dtype,
            )
        pair_names = tuple(
            (self.port_names[positive], self.port_names[negative])
            for positive, negative in resolved_pairs
        )
        metadata = _append_transform_metadata(
            self.metadata,
            {
                "operation": "to_mixed_mode",
                "pairs": pair_names,
                "source_port_names": self.port_names,
                "target_port_names": tuple(names),
            },
        )
        return type(self).from_z(
            frequencies=self.frequencies,
            z=mixed_z,
            z0=mixed_reference,
            port_names=tuple(names),
            valid_columns=self.valid_columns,
            metadata=metadata,
            phasor_convention=self.phasor_convention,
            power_wave_convention=self.power_wave_convention,
        )

    def to_touchstone(
        self,
        path,
        *,
        format="ri",
        frequency_unit="hz",
        version="auto",
        parameter="s",
    ):
        """Export this complete network to a Touchstone file."""

        from .touchstone import write_touchstone

        return write_touchstone(
            self,
            path,
            format=format,
            frequency_unit=frequency_unit,
            version=version,
            parameter=parameter,
        )

    def save(self, path: str | Path):
        """Save a detached CPU snapshot.

        File persistence intentionally does not preserve the live autograd graph.
        Loaded tensors are detached and may be placed with ``load(map_location=...)``.
        Metadata may contain tensors and primitive dict/list/tuple values accepted
        by safe ``torch.load(weights_only=True)``; arbitrary Python objects are not
        part of this persistence contract.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_network_snapshot(self), output_path)

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "NetworkData":
        """Load a detached snapshot; no active autograd graph is reconstructed."""

        payload = _load_persisted_payload(
            Path(path),
            map_location=map_location,
            expected_type=cls.__name__,
        )
        return _network_from_snapshot(payload)


def _network_snapshot(data: NetworkData) -> dict[str, Any]:
    """Single safe serializer shared by standalone and nested NetworkData."""

    _validate_safe_persistence(data.metadata)
    return {
        "schema_version": data.schema_version,
        "data_type": type(data).__name__,
        "frequencies": _detach_to_cpu(data.frequencies),
        "s": _detach_to_cpu(data.s),
        "z0": _detach_to_cpu(data.z0),
        "port_names": data.port_names,
        "valid_columns": _detach_to_cpu(data.valid_columns),
        "metadata": _detach_to_cpu(data.metadata),
        "phasor_convention": data.phasor_convention,
        "power_wave_convention": data.power_wave_convention,
    }


def _network_from_snapshot(payload) -> NetworkData:
    if not isinstance(payload, dict):
        raise ValueError("Persisted NetworkData payload must be a mapping.")
    if payload.get("data_type") != NetworkData.__name__:
        raise ValueError(
            f"Persisted file contains {payload.get('data_type')}, not NetworkData."
        )
    if payload.get("schema_version") != PERSISTENCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported NetworkData schema_version "
            f"{payload.get('schema_version')!r}; expected {PERSISTENCE_SCHEMA_VERSION}."
        )
    return NetworkData(
        frequencies=payload["frequencies"],
        s=payload["s"],
        z0=payload["z0"],
        port_names=tuple(payload["port_names"]),
        valid_columns=payload["valid_columns"],
        metadata=payload["metadata"],
        phasor_convention=payload["phasor_convention"],
        power_wave_convention=payload["power_wave_convention"],
    )


@dataclass(frozen=True)
class NetworkBlock:
    """Declarative passive network connected to named Scene terminal ports.

    ``connections`` maps every network port (by name or one-based index) to a
    unique Scene port name. Automatic fitting is a prepare-time operation. Set
    ``fit=False`` and provide ``model=`` for a pre-fitted, differentiable model.
    """

    name: str
    network: NetworkData
    connections: Mapping[str | int, str]
    fit: RationalFitConfig | Literal[False] = field(default_factory=lambda: _default_fit_config())
    model: RationalModel | StateSpaceNetwork | None = None
    extrapolation: Literal["reject"] = "reject"
    delay_seconds: Literal["auto"] | tuple[float, ...] | None = None
    max_delay_steps: int = 65536

    def __post_init__(self) -> None:
        from .rational import RationalFitConfig, RationalModel, StateSpaceNetwork

        name = str(self.name)
        if not name or name.strip() != name:
            raise ValueError("NetworkBlock name must be non-empty without surrounding whitespace.")
        if not isinstance(self.network, NetworkData):
            raise TypeError("network must be a NetworkData instance.")
        if not isinstance(self.connections, Mapping):
            raise TypeError("connections must map network ports to Scene port names.")
        if self.extrapolation != "reject":
            raise ValueError("The initial network embedding API only supports extrapolation='reject'.")
        if (
            not isinstance(self.max_delay_steps, int)
            or isinstance(self.max_delay_steps, bool)
            or self.max_delay_steps < 1
        ):
            raise ValueError("max_delay_steps must be a positive integer.")

        normalized: dict[str, str] = {}
        for source, target in self.connections.items():
            if isinstance(source, bool):
                raise TypeError("Network connection keys must be port names or one-based indices.")
            if isinstance(source, int):
                if not 1 <= source <= len(self.network.port_names):
                    raise ValueError(f"Network port index {source} is outside 1..{len(self.network.port_names)}.")
                source_name = self.network.port_names[source - 1]
            elif isinstance(source, str):
                source_name = source
                if source_name not in self.network.port_names:
                    raise ValueError(f"Unknown network port {source_name!r}.")
            else:
                raise TypeError("Network connection keys must be port names or one-based indices.")
            target_name = str(target)
            if not target_name or target_name.strip() != target_name:
                raise ValueError("Scene port names in connections must be non-empty without surrounding whitespace.")
            if source_name in normalized:
                raise ValueError(f"Network port {source_name!r} is connected more than once.")
            normalized[source_name] = target_name

        missing = tuple(port for port in self.network.port_names if port not in normalized)
        if missing:
            raise ValueError(f"connections must include every network port; missing {missing!r}.")
        targets = tuple(normalized[port] for port in self.network.port_names)
        if len(set(targets)) != len(targets):
            raise ValueError("Each network port must connect to a distinct Scene port.")

        if self.fit is not False and not isinstance(self.fit, RationalFitConfig):
            raise TypeError("fit must be a RationalFitConfig or False.")
        if self.model is not None and not isinstance(self.model, (RationalModel, StateSpaceNetwork)):
            raise TypeError("model must be a RationalModel, StateSpaceNetwork, or None.")
        if self.model is None and self.fit is False:
            raise ValueError("fit=False requires a pre-fitted RationalModel or StateSpaceNetwork in model=.")
        if self.model is not None and self.fit is not False:
            raise ValueError("A pre-fitted model requires fit=False so automatic fitting is unambiguous.")
        if self.model is None and any(
            tensor.requires_grad for tensor in (self.network.frequencies, self.network.s, self.network.z0)
        ):
            raise RuntimeError(
                "Automatic network fitting is non-differentiable; provide a pre-fitted model and set fit=False."
            )

        port_count = len(self.network.port_names)
        delay_seconds = self.delay_seconds
        if delay_seconds is not None and delay_seconds != "auto":
            if isinstance(delay_seconds, (str, bytes)):
                raise ValueError("delay_seconds must be None, 'auto', or one value per network port.")
            try:
                delay_seconds = tuple(float(value) for value in delay_seconds)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "delay_seconds must be None, 'auto', or a finite non-negative sequence."
                ) from exc
            if len(delay_seconds) != port_count:
                raise ValueError("delay_seconds must contain one one-way delay per network port.")
            delay_tensor = torch.tensor(delay_seconds, dtype=torch.float64)
            if not bool(torch.all(torch.isfinite(delay_tensor))) or not bool(
                torch.all(delay_tensor >= 0.0)
            ):
                raise ValueError("delay_seconds must contain finite non-negative values.")
        elif delay_seconds not in (None, "auto"):
            raise ValueError("delay_seconds must be None, 'auto', or one value per network port.")
        if delay_seconds == "auto" and any(
            tensor.requires_grad for tensor in (self.network.frequencies, self.network.s, self.network.z0)
        ):
            raise RuntimeError(
                "Automatic delay extraction is non-differentiable; declare fixed delay_seconds instead."
            )
        required_representation = "S" if delay_seconds is not None else "Y"
        if isinstance(self.model, RationalModel):
            if self.model.representation != required_representation:
                raise ValueError(
                    "Embedded RationalModel instances must use the "
                    f"{required_representation} representation when delay_seconds="
                    f"{delay_seconds!r}."
                )
            if self.model.input_count != port_count or self.model.output_count != port_count:
                raise ValueError("The pre-fitted model dimensions must match the NetworkData port count.")
        elif isinstance(self.model, StateSpaceNetwork):
            if self.model.representation != required_representation:
                raise ValueError(
                    "Embedded StateSpaceNetwork instances must use the "
                    f"{required_representation} representation when delay_seconds="
                    f"{delay_seconds!r}."
                )
            if self.model.input_count != port_count or self.model.output_count != port_count:
                raise ValueError("The pre-fitted model dimensions must match the NetworkData port count.")
            if self.model.port_order and tuple(self.model.port_order) != self.network.port_names:
                raise ValueError("StateSpaceNetwork.port_order must match NetworkData.port_names.")

        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "connections",
            {port: normalized[port] for port in self.network.port_names},
        )
        object.__setattr__(self, "delay_seconds", delay_seconds)

    @property
    def port_order(self) -> tuple[str, ...]:
        return self.network.port_names

    @property
    def connected_port_names(self) -> tuple[str, ...]:
        return tuple(self.connections[port] for port in self.port_order)


def _default_fit_config():
    from .rational import RationalFitConfig

    return RationalFitConfig()


class TouchstoneNetwork(NetworkBlock):
    """NetworkBlock convenience constructor accepting a file or NetworkData."""

    def __init__(
        self,
        name: str,
        *,
        connections: Mapping[str | int, str],
        network: NetworkData | None = None,
        path: str | Path | None = None,
        fit=None,
        model=None,
        extrapolation: Literal["reject"] = "reject",
        delay_seconds: Literal["auto"] | tuple[float, ...] | None = None,
        max_delay_steps: int = 65536,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.complex128,
    ) -> None:
        if (network is None) == (path is None):
            raise ValueError("Provide exactly one of network= or path=.")
        if network is None:
            network = NetworkData.from_touchstone(path, device=device, dtype=dtype)
        resolved_fit = _default_fit_config() if fit is None else fit
        super().__init__(
            name=name,
            network=network,
            connections=connections,
            fit=resolved_fit,
            model=model,
            extrapolation=extrapolation,
            delay_seconds=delay_seconds,
            max_delay_steps=max_delay_steps,
        )


@dataclass(frozen=True)
class EmbeddedNetworkData:
    """Structured, tensor-native diagnostics for one embedded network."""

    name: str
    frequencies: torch.Tensor
    port_names: tuple[str, ...]
    voltage: torch.Tensor
    current: torch.Tensor
    port_power: torch.Tensor
    absorbed_power: torch.Tensor
    generated_power: torch.Tensor
    state_norm: torch.Tensor
    model_id: str
    fit_report: FitReport | None = None
    runtime_warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("EmbeddedNetworkData name must not be empty.")
        frequencies = self.frequencies
        if not isinstance(frequencies, torch.Tensor) or frequencies.ndim != 1:
            raise TypeError("frequencies must be a one-dimensional torch.Tensor.")
        if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
            raise TypeError("frequencies must be real floating point.")
        if not bool(torch.all(torch.isfinite(frequencies))) or not bool(torch.all(frequencies >= 0.0)):
            raise ValueError("frequencies must be finite and non-negative.")
        voltage, current = _validate_complex_pair(
            self.voltage,
            self.current,
            first_name="voltage",
            second_name="current",
        )
        names = tuple(str(port) for port in self.port_names)
        if voltage.ndim != 2 or voltage.shape != (len(names), frequencies.numel()):
            raise ValueError("voltage/current must have shape [Nport, F].")
        if voltage.device != frequencies.device:
            raise ValueError("frequencies and network diagnostics must share a device.")
        for field_name, value in (
            ("port_power", self.port_power),
            ("absorbed_power", self.absorbed_power),
            ("generated_power", self.generated_power),
            ("state_norm", self.state_norm),
        ):
            if not isinstance(value, torch.Tensor) or value.is_complex() or not value.dtype.is_floating_point:
                raise TypeError(f"{field_name} must be a real floating-point torch.Tensor.")
            if value.device != frequencies.device or not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{field_name} must be finite and on the diagnostics device.")
        expected_port_power_shape = (len(names), frequencies.numel())
        if self.port_power.shape != expected_port_power_shape:
            raise ValueError("port_power must have shape [Nport, F].")
        expected_power_shape = (frequencies.numel(),)
        if self.absorbed_power.shape != expected_power_shape:
            raise ValueError("absorbed_power must have shape [F].")
        if self.generated_power.shape != expected_power_shape:
            raise ValueError("generated_power must have shape [F].")
        if self.state_norm.ndim != 0:
            raise ValueError("state_norm must be a scalar tensor.")
        model_id = str(self.model_id)
        if not model_id:
            raise ValueError("EmbeddedNetworkData model_id must not be empty.")
        if self.fit_report is not None:
            from .rational import FitReport

            if not isinstance(self.fit_report, FitReport):
                raise TypeError("fit_report must be a FitReport or None.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "port_names", names)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "runtime_warnings", tuple(str(item) for item in self.runtime_warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def net_power(self) -> torch.Tensor:
        """Signed total power entering the network at each frequency."""

        return torch.sum(self.port_power, dim=0)


__all__ = [
    "EmbeddedNetworkData",
    "NetworkBlock",
    "NetworkData",
    "NetworkPhysicalityReport",
    "PHASOR_CONVENTION",
    "PERSISTENCE_SCHEMA_VERSION",
    "POWER_WAVE_CONVENTION",
    "PortData",
    "TouchstoneNetwork",
    "power_waves_to_voltage_current",
    "voltage_current_to_power_waves",
]
