from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import torch


PHASOR_CONVENTION = "peak phasor with exp(-i*omega*t) time dependence"
POWER_WAVE_CONVENTION = "Kurokawa power waves normalized to sqrt(watt)"
PERSISTENCE_SCHEMA_VERSION = 1


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
    if not bool(torch.all(frequencies > 0.0)):
        raise ValueError("frequencies must be strictly positive.")
    return frequencies


def _validate_complex_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_name: str,
    second_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"{first_name} and {second_name} must be torch.Tensor instances.")
    if not first.is_complex() or not second.is_complex():
        raise TypeError(f"{first_name} and {second_name} must be complex tensors.")
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} must have identical shapes.")
    if first.ndim == 0 or first.numel() == 0:
        raise ValueError(f"{first_name} and {second_name} must be non-empty arrays.")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError(f"{first_name} and {second_name} must have the same device and dtype.")
    return first, second


def _validate_reference_impedance(z0: torch.Tensor, *, name: str = "z0") -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(torch.real(z0)))) or not bool(
        torch.all(torch.isfinite(torch.imag(z0)))
    ):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool(torch.all(torch.real(z0) > 0.0)):
        raise ValueError(f"Re({name}) must be strictly positive.")
    return z0


def _broadcast_reference_impedance(
    z0,
    *,
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(z0, device=device).to(dtype=dtype)
    try:
        value = torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        raise ValueError(f"z0 must be broadcastable to signal shape {tuple(shape)}.") from exc
    return _validate_reference_impedance(value)


def voltage_current_to_power_waves(
    voltage: torch.Tensor,
    current: torch.Tensor,
    z0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert peak-phasor voltage/current to Kurokawa power waves.

    The normalization includes the peak-to-average factor, so ``abs(a)**2`` and
    ``abs(b)**2`` are incident and reflected powers in watts and
    ``abs(a)**2 - abs(b)**2 == 0.5 * Re(V * conj(I))``.
    """

    voltage, current = _validate_complex_pair(
        voltage,
        current,
        first_name="voltage",
        second_name="current",
    )
    reference = _broadcast_reference_impedance(
        z0,
        shape=voltage.shape,
        device=voltage.device,
        dtype=voltage.dtype,
    )
    scale = 2.0 * torch.sqrt(2.0 * torch.real(reference))
    incident = (voltage + reference * current) / scale
    reflected = (voltage - torch.conj(reference) * current) / scale
    return incident, reflected


def power_waves_to_voltage_current(
    a: torch.Tensor,
    b: torch.Tensor,
    z0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Kurokawa power waves to peak-phasor voltage/current."""

    a, b = _validate_complex_pair(a, b, first_name="a", second_name="b")
    reference = _broadcast_reference_impedance(
        z0,
        shape=a.shape,
        device=a.device,
        dtype=a.dtype,
    )
    factor = torch.sqrt(2.0 / torch.real(reference))
    voltage = factor * (torch.conj(reference) * a + reference * b)
    current = factor * (a - b)
    return voltage, current


def _return_loss_db(reflection: torch.Tensor) -> torch.Tensor:
    return -20.0 * torch.log10(torch.abs(reflection))


def _vswr(reflection: torch.Tensor) -> torch.Tensor:
    magnitude = torch.abs(reflection)
    finite_ratio = (1.0 + magnitude) / (1.0 - magnitude)
    return torch.where(
        magnitude < 1.0,
        finite_ratio,
        torch.full_like(magnitude, torch.inf),
    )


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


def _identity_batch(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    return torch.eye(size, device=matrix.device, dtype=matrix.dtype).expand(
        matrix.shape[:-2] + (size, size)
    )


def _right_solve(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``left @ inv(right)`` using a batched linear solve."""

    return torch.linalg.solve(
        right.transpose(-2, -1),
        left.transpose(-2, -1),
    ).transpose(-2, -1)


def _s_to_z(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    resistance_root = torch.sqrt(torch.real(z0))
    normalized_s = (
        resistance_root.unsqueeze(-1)
        * s
        / resistance_root.unsqueeze(-2)
    )
    identity = _identity_batch(s)
    z0_matrix = torch.diag_embed(z0)
    rhs = torch.diag_embed(torch.conj(z0)) + normalized_s @ z0_matrix
    return torch.linalg.solve(identity - normalized_s, rhs)


def _z_to_s(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    z0_matrix = torch.diag_embed(z0)
    normalized_s = _right_solve(
        z - torch.diag_embed(torch.conj(z0)),
        z + z0_matrix,
    )
    resistance_root = torch.sqrt(torch.real(z0))
    return (
        normalized_s
        * resistance_root.unsqueeze(-2)
        / resistance_root.unsqueeze(-1)
    )


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
        y, resolved_frequencies, _, port_count = _validate_network_matrix(
            y,
            frequencies,
            name="y",
        )
        valid = _normalize_valid_columns(
            valid_columns,
            port_count=port_count,
            device=y.device,
        )
        if not bool(torch.all(valid)):
            raise RuntimeError("S/Y conversion requires complete excitation columns.")
        z = torch.linalg.solve(y, _identity_batch(y))
        return cls.from_z(
            frequencies=resolved_frequencies,
            z=z,
            z0=z0,
            port_names=port_names,
            valid_columns=valid,
            metadata=metadata,
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
        z = _s_to_z(self.s, self.z0)
        return torch.linalg.solve(z, _identity_batch(z))

    def renormalize(self, z0) -> "NetworkData":
        self._require_complete("renormalization")
        return type(self).from_z(
            frequencies=self.frequencies,
            z=self.to_z(),
            z0=z0,
            port_names=self.port_names,
            valid_columns=self.valid_columns,
            metadata=self.metadata,
            phasor_convention=self.phasor_convention,
            power_wave_convention=self.power_wave_convention,
        )

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
                "frequencies": _detach_to_cpu(self.frequencies),
                "s": _detach_to_cpu(self.s),
                "z0": _detach_to_cpu(self.z0),
                "port_names": self.port_names,
                "valid_columns": _detach_to_cpu(self.valid_columns),
                "metadata": _detach_to_cpu(self.metadata),
                "phasor_convention": self.phasor_convention,
                "power_wave_convention": self.power_wave_convention,
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "NetworkData":
        """Load a detached snapshot; no active autograd graph is reconstructed."""

        payload = _load_persisted_payload(
            Path(path),
            map_location=map_location,
            expected_type=cls.__name__,
        )
        return cls(
            frequencies=payload["frequencies"],
            s=payload["s"],
            z0=payload["z0"],
            port_names=tuple(payload["port_names"]),
            valid_columns=payload["valid_columns"],
            metadata=payload["metadata"],
            phasor_convention=payload["phasor_convention"],
            power_wave_convention=payload["power_wave_convention"],
        )


__all__ = [
    "NetworkData",
    "PHASOR_CONVENTION",
    "PERSISTENCE_SCHEMA_VERSION",
    "POWER_WAVE_CONVENTION",
    "PortData",
    "power_waves_to_voltage_current",
    "voltage_current_to_power_waves",
]
