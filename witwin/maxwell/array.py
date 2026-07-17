from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import torch

from .antenna import Ludwig3, _power_normalized_antenna_metrics
from .network import (
    PHASOR_CONVENTION,
    POWER_WAVE_CONVENTION,
    NetworkData,
    _network_from_snapshot,
    _network_snapshot,
    _validate_safe_persistence,
    power_waves_to_voltage_current,
)


ARRAY_POWER_NORMALIZATION = (
    "Kurokawa incident power waves in sqrt(watt); each embedded pattern column "
    "is normalized to a_n=1 sqrt(watt) with all other ports matched"
)
ARRAY_FIELD_BASIS = "spherical_theta_phi"
ARRAY_PERSISTENCE_SCHEMA_VERSION = 1


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


def _validate_persisted_payload(
    payload,
    *,
    expected_type: str,
    expected_schema_version: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Persisted {expected_type} payload must be a mapping.")
    actual_type = payload.get("data_type")
    if actual_type != expected_type:
        raise ValueError(f"Persisted file contains {actual_type}, not {expected_type}.")
    schema_version = payload.get("schema_version")
    if schema_version != expected_schema_version:
        raise ValueError(
            f"Unsupported {expected_type} schema_version {schema_version!r}; "
            f"expected {expected_schema_version}."
        )
    return payload


def _load_persisted_payload(path, *, map_location, expected_type: str):
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    return _validate_persisted_payload(
        payload,
        expected_type=expected_type,
        expected_schema_version=ARRAY_PERSISTENCE_SCHEMA_VERSION,
    )


@dataclass(frozen=True)
class AcceptanceBudget:
    analytic_rtol: float = 1.0e-6
    analytic_atol: float = 1.0e-10
    cuda_complex64_rtol: float = 2.0e-5
    cuda_complex64_atol: float = 1.0e-6
    fdtd_complex_l2: float = 0.03
    fdtd_phase_rms_deg: float = 3.0
    fdtd_phase_support_fraction: float = 0.10
    port_power_relative_error: float = 0.01
    physical_power_residual: float = 0.01
    reference_gain_error_db: float = 0.25
    reference_ecc_error: float = 0.02
    active_impedance_magnitude_error: float = 0.05
    active_impedance_phase_error_deg: float = 3.0
    gradient_relative_error: float = 0.02
    gradient_absolute_floor: float = 1.0e-8
    distributed_field_max_abs: float = 2.0e-6
    distributed_field_max_rel: float = 2.0e-5
    task_s_rtol: float = 2.0e-5
    task_s_atol: float = 1.0e-6
    distributed_result_rtol: float = 5.0e-5
    distributed_result_atol: float = 5.0e-6
    phase1_grid_shape: tuple[int, int, int] = (96, 96, 96)
    phase1_pml_cells: int = 8
    phase1_steps: int = 4096
    phase1_beams: int = 16
    phase1_angular_shape: tuple[int, int] = (181, 361)
    phase1_basis_direct_time_ratio: float = 0.40
    phase1_combine_solve_time_ratio: float = 0.10
    timing_warmups: int = 3
    timing_samples: int = 5
    timing_order_rounds: int = 4
    task_basis_count: int = 16
    two_gpu_parallel_efficiency: float = 0.80
    four_gpu_parallel_efficiency: float = 0.70
    scaling_hardware: str = (
        "4x NVIDIA RTX A6000 48 GiB, PCIe Gen4 x16, pairwise peer access enabled"
    )
    local_hardware: str = (
        "NVIDIA GeForce RTX 5080 16303 MiB, driver 596.49, PCI 00000000:01:00.0"
    )


ARRAY_ACCEPTANCE_BUDGET = AcceptanceBudget()


def _finite_complex(value: torch.Tensor) -> bool:
    return bool(torch.all(torch.isfinite(value.real))) and bool(
        torch.all(torch.isfinite(value.imag))
    )


def _trapz_weights(points: torch.Tensor) -> torch.Tensor:
    differences = points[1:] - points[:-1]
    weights = torch.empty_like(points)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    if points.numel() > 2:
        weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return weights


def _solid_angle_weights(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    theta_vector = theta[:, 0]
    phi_vector = phi[0, :]
    return (
        torch.sin(theta)
        * _trapz_weights(theta_vector)[:, None]
        * _trapz_weights(phi_vector)[None, :]
    )


def _validate_angular_grid(
    theta: torch.Tensor,
    phi: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(theta, torch.Tensor) or not isinstance(phi, torch.Tensor):
        raise TypeError("theta and phi must be torch.Tensor instances.")
    if theta.is_complex() or phi.is_complex():
        raise TypeError("theta and phi must be real floating-point tensors.")
    if not theta.dtype.is_floating_point or not phi.dtype.is_floating_point:
        raise TypeError("theta and phi must be real floating-point tensors.")
    if theta.device != device or phi.device != device:
        raise ValueError(f"theta and phi must be on device {device}.")
    if theta.dtype != phi.dtype:
        raise TypeError("theta and phi must have the same dtype.")
    if theta.ndim != 2 or theta.shape != phi.shape:
        raise ValueError("theta and phi must have identical shapes [T, P].")
    if theta.shape[0] < 3 or theta.shape[1] < 2:
        raise ValueError("The angular grid requires at least three theta and two phi samples.")
    theta_vector = theta[:, 0]
    phi_vector = phi[0, :]
    if not torch.allclose(theta, theta_vector[:, None]):
        raise ValueError("theta must be constant along the phi axis.")
    if not torch.allclose(phi, phi_vector[None, :]):
        raise ValueError("phi must be constant along the theta axis.")
    if not bool(torch.all(theta_vector[1:] > theta_vector[:-1])):
        raise ValueError("theta samples must be strictly increasing.")
    if not bool(torch.all(phi_vector[1:] > phi_vector[:-1])):
        raise ValueError("phi samples must be strictly increasing.")
    tolerance = 128.0 * torch.finfo(theta.dtype).eps
    if not torch.isclose(
        theta_vector[0], torch.zeros((), device=device, dtype=theta.dtype), atol=tolerance, rtol=0.0
    ) or not torch.isclose(
        theta_vector[-1],
        torch.full((), math.pi, device=device, dtype=theta.dtype),
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError("theta must span [0, pi] for full-sphere integration.")
    if not torch.isclose(
        phi_vector[-1] - phi_vector[0],
        torch.full((), 2.0 * math.pi, device=device, dtype=phi.dtype),
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError("phi must span exactly 2*pi for full-sphere integration.")
    return theta, phi


def _broadcast_pattern_parameter(
    value,
    *,
    name: str,
    shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}.")
        if value.is_complex() or not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be real floating-point data.")
        tensor = value.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device)
        if tensor.is_complex() or not tensor.dtype.is_floating_point:
            raise TypeError(f"{name} must be real floating-point data.")
        tensor = tensor.to(dtype=dtype)
    if tensor.ndim == 1 and tensor.shape == (shape[0],):
        tensor = tensor[:, None, None]
    elif tensor.ndim == 2 and tensor.shape == shape[1:]:
        tensor = tensor[None, ...]
    try:
        tensor = torch.broadcast_to(tensor, shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must be scalar or broadcastable to [F, T, P].") from exc
    if not bool(torch.all(torch.isfinite(tensor))) or not bool(torch.all(tensor > 0.0)):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return tensor


@dataclass(frozen=True)
class BeamWeights:
    """Complex incident power-wave amplitudes with the port axis last."""

    values: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("BeamWeights.values must be a torch.Tensor.")
        if not self.values.is_complex():
            raise TypeError("BeamWeights.values must be a complex tensor.")
        if self.values.ndim not in (1, 2, 3) or self.values.numel() == 0:
            raise ValueError("BeamWeights.values must have shape [N], [F, N], or [B, F, N].")
        if not _finite_complex(self.values):
            raise ValueError("BeamWeights.values must contain only finite values.")


@dataclass(frozen=True)
class EmbeddedElementPatternData:
    """Power-normalized embedded element patterns in ``[F, N, T, P]`` order."""

    frequencies: torch.Tensor
    port_names: tuple[str, ...]
    theta: torch.Tensor
    phi: torch.Tensor
    e_theta: torch.Tensor
    e_phi: torch.Tensor
    phase_center: torch.Tensor
    frame: torch.Tensor
    observation_radius: Any = 1.0
    wave_impedance: Any = 376.730313668
    polarization_basis: Ludwig3 = field(default_factory=Ludwig3)
    phase_center_source: str = "explicit"
    field_basis: str = ARRAY_FIELD_BASIS
    power_normalization: str = ARRAY_POWER_NORMALIZATION
    phasor_convention: str = PHASOR_CONVENTION
    power_wave_convention: str = POWER_WAVE_CONVENTION
    field_units: str = "V/m per sqrt(W)"

    schema_version: ClassVar[int] = ARRAY_PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.e_theta, torch.Tensor) or not isinstance(self.e_phi, torch.Tensor):
            raise TypeError("e_theta and e_phi must be torch.Tensor instances.")
        if not self.e_theta.is_complex() or not self.e_phi.is_complex():
            raise TypeError("e_theta and e_phi must be complex tensors.")
        if self.e_theta.shape != self.e_phi.shape or self.e_theta.ndim != 4:
            raise ValueError("e_theta and e_phi must have identical shape [F, N, T, P].")
        if self.e_theta.shape[0] == 0 or self.e_theta.shape[1] == 0:
            raise ValueError("Embedded element patterns require non-empty frequency and port axes.")
        if self.e_theta.dtype != self.e_phi.dtype or self.e_theta.device != self.e_phi.device:
            raise ValueError("e_theta and e_phi must have the same dtype and device.")
        if not _finite_complex(self.e_theta) or not _finite_complex(self.e_phi):
            raise ValueError("Embedded element fields must contain only finite values.")
        device = self.e_theta.device
        if not isinstance(self.frequencies, torch.Tensor):
            raise TypeError("frequencies must be a torch.Tensor.")
        if self.frequencies.device != device or self.frequencies.ndim != 1:
            raise ValueError("frequencies must have shape [F] on the pattern device.")
        if self.frequencies.is_complex() or not self.frequencies.dtype.is_floating_point:
            raise TypeError("frequencies must be a real floating-point tensor.")
        if self.frequencies.shape[0] != self.e_theta.shape[0]:
            raise ValueError("frequencies length must match the embedded-pattern frequency axis.")
        if not bool(torch.all(torch.isfinite(self.frequencies))) or not bool(
            torch.all(self.frequencies > 0.0)
        ):
            raise ValueError("frequencies must be finite and strictly positive.")
        if self.frequencies.numel() > 1 and not bool(
            torch.all(self.frequencies[1:] > self.frequencies[:-1])
        ):
            raise ValueError("frequencies must be strictly increasing.")
        names = tuple(str(name) for name in self.port_names)
        if len(names) != self.e_theta.shape[1] or any(not name for name in names):
            raise ValueError("port_names must contain one non-empty name per pattern column.")
        if len(set(names)) != len(names):
            raise ValueError("port_names must be unique.")
        theta, phi = _validate_angular_grid(self.theta, self.phi, device=device)
        if tuple(theta.shape) != tuple(self.e_theta.shape[-2:]):
            raise ValueError("theta/phi shape must match the pattern angular axes.")
        if not isinstance(self.phase_center, torch.Tensor) or not isinstance(
            self.frame, torch.Tensor
        ):
            raise TypeError("phase_center and frame must be torch.Tensor instances.")
        if self.phase_center.shape != (3,) or self.phase_center.device != device:
            raise ValueError("phase_center must have shape [3] on the pattern device.")
        if self.frame.shape != (3, 3) or self.frame.device != device:
            raise ValueError("frame must have shape [3, 3] on the pattern device.")
        if (
            self.phase_center.is_complex()
            or self.frame.is_complex()
            or not self.phase_center.dtype.is_floating_point
            or not self.frame.dtype.is_floating_point
        ):
            raise TypeError("phase_center and frame must be real floating-point tensors.")
        if not bool(torch.all(torch.isfinite(self.phase_center))) or not bool(
            torch.all(torch.isfinite(self.frame))
        ):
            raise ValueError("phase_center and frame must contain only finite values.")
        identity = torch.eye(3, device=device, dtype=self.frame.dtype)
        tolerance = 256.0 * torch.finfo(self.frame.dtype).eps
        if not torch.allclose(self.frame.transpose(0, 1) @ self.frame, identity, atol=tolerance, rtol=tolerance):
            raise ValueError("frame columns must be orthonormal.")
        if not bool(torch.linalg.det(self.frame) > 0.0):
            raise ValueError("frame must be right-handed.")
        if self.field_basis != ARRAY_FIELD_BASIS:
            raise ValueError(f"field_basis must be {ARRAY_FIELD_BASIS!r}.")
        if self.power_normalization != ARRAY_POWER_NORMALIZATION:
            raise ValueError(
                "power_normalization must describe unit incident Kurokawa power-wave EEP columns."
            )
        if self.phasor_convention != PHASOR_CONVENTION:
            raise ValueError(f"phasor_convention must be {PHASOR_CONVENTION!r}.")
        if self.power_wave_convention != POWER_WAVE_CONVENTION:
            raise ValueError(f"power_wave_convention must be {POWER_WAVE_CONVENTION!r}.")
        if self.field_units != "V/m per sqrt(W)":
            raise ValueError("field_units must be 'V/m per sqrt(W)'.")
        if not isinstance(self.polarization_basis, Ludwig3):
            raise TypeError("polarization_basis must be a Ludwig3 instance.")
        if self.phase_center_source not in {"explicit", "array_aabb"}:
            raise ValueError("phase_center_source must be 'explicit' or 'array_aabb'.")
        parameter_shape = (self.e_theta.shape[0], *self.e_theta.shape[-2:])
        radius = _broadcast_pattern_parameter(
            self.observation_radius,
            name="observation_radius",
            shape=parameter_shape,
            device=device,
            dtype=self.e_theta.real.dtype,
        )
        impedance = _broadcast_pattern_parameter(
            self.wave_impedance,
            name="wave_impedance",
            shape=parameter_shape,
            device=device,
            dtype=self.e_theta.real.dtype,
        )
        object.__setattr__(self, "port_names", names)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "observation_radius", radius)
        object.__setattr__(self, "wave_impedance", impedance)

    @property
    def device(self) -> torch.device:
        return self.e_theta.device

    @property
    def dtype(self) -> torch.dtype:
        return self.e_theta.dtype

    @property
    def E_theta(self) -> torch.Tensor:
        return self.e_theta

    @property
    def E_phi(self) -> torch.Tensor:
        return self.e_phi

    def save(self, path: str | Path):
        """Save a detached CPU snapshot.

        The snapshot omits the live autograd graph and loads through the safe
        tensor-only ``torch.load(weights_only=True)`` contract.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_embedded_pattern_snapshot(self), output_path)

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "EmbeddedElementPatternData":
        """Load a detached safe snapshot onto ``map_location``."""

        payload = _load_persisted_payload(
            path,
            map_location=map_location,
            expected_type=cls.__name__,
        )
        return _embedded_pattern_from_snapshot(payload)


def _embedded_pattern_snapshot(data: EmbeddedElementPatternData) -> dict[str, Any]:
    return {
        "schema_version": data.schema_version,
        "data_type": type(data).__name__,
        "frequencies": _detach_to_cpu(data.frequencies),
        "port_names": data.port_names,
        "theta": _detach_to_cpu(data.theta),
        "phi": _detach_to_cpu(data.phi),
        "e_theta": _detach_to_cpu(data.e_theta),
        "e_phi": _detach_to_cpu(data.e_phi),
        "phase_center": _detach_to_cpu(data.phase_center),
        "frame": _detach_to_cpu(data.frame),
        "observation_radius": _detach_to_cpu(data.observation_radius),
        "wave_impedance": _detach_to_cpu(data.wave_impedance),
        "polarization_basis": {
            "kind": type(data.polarization_basis).__name__,
            "reference_angle": data.polarization_basis.reference_angle,
        },
        "phase_center_source": data.phase_center_source,
        "field_basis": data.field_basis,
        "power_normalization": data.power_normalization,
        "phasor_convention": data.phasor_convention,
        "power_wave_convention": data.power_wave_convention,
        "field_units": data.field_units,
    }


def _embedded_pattern_from_snapshot(payload) -> EmbeddedElementPatternData:
    payload = _validate_persisted_payload(
        payload,
        expected_type=EmbeddedElementPatternData.__name__,
        expected_schema_version=EmbeddedElementPatternData.schema_version,
    )
    polarization = payload["polarization_basis"]
    if not isinstance(polarization, Mapping) or polarization.get("kind") != Ludwig3.__name__:
        raise ValueError("EmbeddedElementPatternData has an unsupported polarization basis.")
    return EmbeddedElementPatternData(
        frequencies=payload["frequencies"],
        port_names=tuple(payload["port_names"]),
        theta=payload["theta"],
        phi=payload["phi"],
        e_theta=payload["e_theta"],
        e_phi=payload["e_phi"],
        phase_center=payload["phase_center"],
        frame=payload["frame"],
        observation_radius=payload["observation_radius"],
        wave_impedance=payload["wave_impedance"],
        polarization_basis=Ludwig3(reference_angle=polarization["reference_angle"]),
        phase_center_source=payload["phase_center_source"],
        field_basis=payload["field_basis"],
        power_normalization=payload["power_normalization"],
        phasor_convention=payload["phasor_convention"],
        power_wave_convention=payload["power_wave_convention"],
        field_units=payload["field_units"],
    )


@dataclass(frozen=True)
class _BeamNetworkData:
    a: torch.Tensor
    b: torch.Tensor
    incident_power_per_port: torch.Tensor
    reflected_power_per_port: torch.Tensor
    accepted_power_per_port: torch.Tensor
    incident_power: torch.Tensor
    reflected_power: torch.Tensor
    accepted_power: torch.Tensor
    active_reflection: torch.Tensor
    active_impedance: torch.Tensor
    active_mask: torch.Tensor
    z0: torch.Tensor
    port_names: tuple[str, ...]
    frequencies: torch.Tensor
    phasor_convention: str
    power_wave_convention: str


@dataclass(frozen=True)
class _BeamFarFieldData:
    e_theta: torch.Tensor
    e_phi: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    phase_center: torch.Tensor
    frame: torch.Tensor
    observation_radius: torch.Tensor
    wave_impedance: torch.Tensor
    polarization_basis: Ludwig3
    phase_center_source: str
    power_normalization: str
    phasor_convention: str
    field_units: str = "V/m"
    field_basis: str = ARRAY_FIELD_BASIS

    @property
    def E_theta(self) -> torch.Tensor:
        return self.e_theta

    @property
    def E_phi(self) -> torch.Tensor:
        return self.e_phi


@dataclass(frozen=True)
class _BeamAntennaData:
    radiation_intensity: torch.Tensor
    p_rad: torch.Tensor
    directivity: torch.Tensor
    gain: torch.Tensor
    realized_gain: torch.Tensor
    radiation_efficiency: torch.Tensor
    mismatch_efficiency: torch.Tensor
    system_efficiency: torch.Tensor
    eirp: torch.Tensor
    radiation_valid: torch.Tensor
    accepted_power_valid: torch.Tensor


@dataclass(frozen=True)
class BeamData:
    weights: torch.Tensor
    network: _BeamNetworkData
    far_field: _BeamFarFieldData
    antenna: _BeamAntennaData
    frequencies: torch.Tensor
    names: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    power_normalization: str = ARRAY_POWER_NORMALIZATION
    phasor_convention: str = PHASOR_CONVENTION
    power_wave_convention: str = POWER_WAVE_CONVENTION
    units: Mapping[str, str] = field(
        default_factory=lambda: {
            "weights": "sqrt(W)",
            "power": "W",
            "active_impedance": "ohm",
            "far_field": "V/m",
            "radiation_intensity": "W/sr",
            "gain": "1",
            "eirp": "W",
        }
    )

    def __post_init__(self):
        if not isinstance(self.weights, torch.Tensor) or not self.weights.is_complex():
            raise TypeError("BeamData.weights must be a complex torch.Tensor.")
        if not isinstance(self.network, _BeamNetworkData):
            raise TypeError("BeamData.network has an invalid internal result contract.")
        if not isinstance(self.far_field, _BeamFarFieldData):
            raise TypeError("BeamData.far_field has an invalid internal result contract.")
        if not isinstance(self.antenna, _BeamAntennaData):
            raise TypeError("BeamData.antenna has an invalid internal result contract.")
        if self.frequencies.device != self.weights.device:
            raise ValueError("BeamData frequencies and weights must be on the same device.")
        if self.power_normalization != ARRAY_POWER_NORMALIZATION:
            raise ValueError("BeamData power_normalization does not match the array contract.")
        if self.phasor_convention != PHASOR_CONVENTION:
            raise ValueError("BeamData phasor_convention does not match the RF contract.")
        if self.power_wave_convention != POWER_WAVE_CONVENTION:
            raise ValueError("BeamData power_wave_convention does not match the RF contract.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "units", dict(self.units))

    @property
    def device(self) -> torch.device:
        return self.weights.device


def _restore_single(value: torch.Tensor, *, single: bool) -> torch.Tensor:
    return value[0] if single else value


def _normalize_weights(
    weights: BeamWeights | torch.Tensor,
    *,
    frequency_count: int,
    port_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    values = weights.values if isinstance(weights, BeamWeights) else weights
    if not isinstance(values, torch.Tensor):
        raise TypeError("weights must be a complex torch.Tensor or BeamWeights.")
    if not values.is_complex():
        raise TypeError("weights must be a complex tensor of incident power-wave amplitudes.")
    if values.device != device:
        raise ValueError(f"weights must be on device {device}.")
    if values.dtype != dtype:
        raise TypeError(f"weights must use dtype {dtype}.")
    if not _finite_complex(values):
        raise ValueError("weights must contain only finite values.")
    if values.ndim == 1:
        if values.shape != (port_count,):
            raise ValueError(f"weights with rank 1 must have shape [N] = ({port_count},).")
        return values[None, None, :].expand(1, frequency_count, port_count), True
    if values.ndim == 2:
        if values.shape != (frequency_count, port_count):
            raise ValueError(
                "rank-2 weights are frequency-dependent and must have exact shape "
                f"[F, N] = ({frequency_count}, {port_count}); frequency interpolation is forbidden."
            )
        return values[None, ...], True
    if values.ndim == 3:
        if values.shape[1:] != (frequency_count, port_count):
            raise ValueError(
                f"rank-3 weights must have shape [B, F, N] with [F, N] = ({frequency_count}, {port_count})."
            )
        if values.shape[0] == 0:
            raise ValueError("weights batch dimension must be non-empty.")
        return values, False
    raise ValueError("weights must have shape [N], [F, N], or [B, F, N].")


@dataclass(frozen=True)
class ArrayBasisData:
    """Reusable N-port network and embedded-pattern basis for one physical scene."""

    network: NetworkData
    embedded_patterns: EmbeddedElementPatternData
    fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    radiated_power_matrix: torch.Tensor | None = None

    schema_version: ClassVar[int] = ARRAY_PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.network, NetworkData):
            raise TypeError("network must be a NetworkData instance.")
        if not self.network.is_complete:
            raise ValueError("ArrayBasisData requires complete NetworkData excitation columns.")
        if not isinstance(self.embedded_patterns, EmbeddedElementPatternData):
            raise TypeError("embedded_patterns must be EmbeddedElementPatternData.")
        patterns = self.embedded_patterns
        if self.network.s.device != patterns.device:
            raise ValueError("NetworkData and embedded patterns must be on the same device.")
        if self.network.s.dtype != patterns.dtype:
            raise TypeError("NetworkData and embedded patterns must use the same complex dtype.")
        if self.network.port_names != patterns.port_names:
            raise ValueError("NetworkData and embedded patterns must use identical port order.")
        if (
            self.network.frequencies.shape != patterns.frequencies.shape
            or self.network.frequencies.dtype != patterns.frequencies.dtype
            or not torch.equal(self.network.frequencies, patterns.frequencies)
        ):
            raise ValueError("NetworkData and embedded patterns must use identical frequencies.")
        if self.network.phasor_convention != patterns.phasor_convention:
            raise ValueError("NetworkData and embedded patterns must use the same phasor convention.")
        if self.network.power_wave_convention != patterns.power_wave_convention:
            raise ValueError(
                "NetworkData and embedded patterns must use the same power-wave convention."
            )
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("fingerprint must be a non-empty string.")
        if self.radiated_power_matrix is not None:
            matrix = self.radiated_power_matrix
            expected_shape = self.network.s.shape
            if not isinstance(matrix, torch.Tensor) or not matrix.is_complex():
                raise TypeError("radiated_power_matrix must be a complex torch.Tensor.")
            if matrix.shape != expected_shape:
                raise ValueError(
                    "radiated_power_matrix must have shape [F, N, N] matching NetworkData."
                )
            if matrix.device != self.device or matrix.dtype != self.dtype:
                raise ValueError(
                    "radiated_power_matrix must share NetworkData dtype and device."
                )
            if not _finite_complex(matrix):
                raise ValueError("radiated_power_matrix must contain only finite values.")
            tolerance = 512.0 * torch.finfo(matrix.real.dtype).eps
            if not torch.allclose(
                matrix,
                matrix.mH,
                rtol=tolerance,
                atol=tolerance * max(1.0, float(torch.amax(torch.abs(matrix)).item())),
            ):
                raise ValueError("radiated_power_matrix must be Hermitian at every frequency.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def frequencies(self) -> torch.Tensor:
        return self.network.frequencies

    @property
    def port_names(self) -> tuple[str, ...]:
        return self.network.port_names

    @property
    def device(self) -> torch.device:
        return self.network.s.device

    @property
    def dtype(self) -> torch.dtype:
        return self.network.s.dtype

    @property
    def eep(self) -> EmbeddedElementPatternData:
        return self.embedded_patterns

    def save(self, path: str | Path):
        """Save a detached CPU snapshot.

        The snapshot preserves the complete network, embedded-pattern contract,
        and metadata but not their live autograd graphs. Metadata follows the
        same safe primitive/tensor contract as ``NetworkData``.
        """

        _validate_safe_persistence(self.metadata)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": self.schema_version,
                "data_type": type(self).__name__,
                "network": _network_snapshot(self.network),
                "embedded_patterns": _embedded_pattern_snapshot(self.embedded_patterns),
                "fingerprint": self.fingerprint,
                "metadata": _detach_to_cpu(self.metadata),
                "radiated_power_matrix": _detach_to_cpu(self.radiated_power_matrix),
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "ArrayBasisData":
        """Load a detached safe snapshot onto ``map_location``."""

        payload = _load_persisted_payload(
            path,
            map_location=map_location,
            expected_type=cls.__name__,
        )
        if "radiated_power_matrix" not in payload:
            raise ValueError(
                "Persisted ArrayBasisData is missing radiated_power_matrix."
            )
        return cls(
            network=_network_from_snapshot(payload["network"]),
            embedded_patterns=_embedded_pattern_from_snapshot(payload["embedded_patterns"]),
            fingerprint=payload["fingerprint"],
            metadata=payload["metadata"],
            radiated_power_matrix=payload["radiated_power_matrix"],
        )

    def combine(self, weights: BeamWeights | torch.Tensor) -> BeamData:
        """Combine incident power-wave weights without rerunning the field solver."""

        frequency_count, port_count, _ = self.network.s.shape
        a, single = _normalize_weights(
            weights,
            frequency_count=frequency_count,
            port_count=port_count,
            device=self.device,
            dtype=self.dtype,
        )
        incident_power_per_port = torch.abs(a).square()
        incident_power = torch.sum(incident_power_per_port, dim=-1)
        if bool(torch.any(incident_power <= 0.0)):
            raise ValueError("Every beam/frequency entry must have strictly positive incident power.")
        b = torch.einsum("fij,kfj->kfi", self.network.s, a)
        reflected_power_per_port = torch.abs(b).square()
        accepted_power_per_port = incident_power_per_port - reflected_power_per_port
        reflected_power = torch.sum(reflected_power_per_port, dim=-1)
        accepted_power = incident_power - reflected_power

        active_mask = torch.abs(a) > 0.0
        safe_a = torch.where(active_mask, a, torch.ones_like(a))
        active_reflection = b / safe_a
        complex_nan = torch.full_like(active_reflection, complex(float("nan"), float("nan")))
        active_reflection = torch.where(active_mask, active_reflection, complex_nan)
        z0 = self.network.z0.unsqueeze(0).expand_as(a)
        impedance_a = torch.where(active_mask, a, torch.ones_like(a))
        impedance_b = torch.where(active_mask, b, torch.zeros_like(b))
        voltage, current = power_waves_to_voltage_current(impedance_a, impedance_b, z0)
        active_impedance = voltage / current
        active_impedance = torch.where(active_mask, active_impedance, complex_nan)

        patterns = self.embedded_patterns
        e_theta = torch.einsum("kfn,fntp->kftp", a, patterns.e_theta)
        e_phi = torch.einsum("kfn,fntp->kftp", a, patterns.e_phi)
        angular_weights = _solid_angle_weights(patterns.theta, patterns.phi)
        radiated_power = None
        if self.radiated_power_matrix is not None:
            radiated_power = torch.real(
                torch.einsum(
                    "kfm,fmn,kfn->kf",
                    torch.conj(a),
                    self.radiated_power_matrix,
                    a,
                )
            )
        metrics = _power_normalized_antenna_metrics(
            e_theta=e_theta,
            e_phi=e_phi,
            observation_radius=patterns.observation_radius[None, ...],
            wave_impedance=patterns.wave_impedance[None, ...],
            solid_angle_weights=angular_weights[None, None, ...],
            incident_power=incident_power,
            accepted_power=accepted_power,
            radiated_power=radiated_power,
        )

        network = _BeamNetworkData(
            a=_restore_single(a, single=single),
            b=_restore_single(b, single=single),
            incident_power_per_port=_restore_single(incident_power_per_port, single=single),
            reflected_power_per_port=_restore_single(reflected_power_per_port, single=single),
            accepted_power_per_port=_restore_single(accepted_power_per_port, single=single),
            incident_power=_restore_single(incident_power, single=single),
            reflected_power=_restore_single(reflected_power, single=single),
            accepted_power=_restore_single(accepted_power, single=single),
            active_reflection=_restore_single(active_reflection, single=single),
            active_impedance=_restore_single(active_impedance, single=single),
            active_mask=_restore_single(active_mask, single=single),
            z0=_restore_single(z0, single=single),
            port_names=self.port_names,
            frequencies=self.frequencies,
            phasor_convention=self.network.phasor_convention,
            power_wave_convention=self.network.power_wave_convention,
        )
        far_field = _BeamFarFieldData(
            e_theta=_restore_single(e_theta, single=single),
            e_phi=_restore_single(e_phi, single=single),
            theta=patterns.theta,
            phi=patterns.phi,
            phase_center=patterns.phase_center,
            frame=patterns.frame,
            observation_radius=patterns.observation_radius,
            wave_impedance=patterns.wave_impedance,
            polarization_basis=patterns.polarization_basis,
            phase_center_source=patterns.phase_center_source,
            power_normalization=patterns.power_normalization,
            phasor_convention=patterns.phasor_convention,
            field_units="V/m",
        )
        antenna = _BeamAntennaData(
            radiation_intensity=_restore_single(metrics["radiation_intensity"], single=single),
            p_rad=_restore_single(metrics["p_rad"], single=single),
            directivity=_restore_single(metrics["directivity"], single=single),
            gain=_restore_single(metrics["gain"], single=single),
            realized_gain=_restore_single(metrics["realized_gain"], single=single),
            radiation_efficiency=_restore_single(metrics["radiation_efficiency"], single=single),
            mismatch_efficiency=_restore_single(metrics["mismatch_efficiency"], single=single),
            system_efficiency=_restore_single(metrics["system_efficiency"], single=single),
            eirp=_restore_single(metrics["eirp"], single=single),
            radiation_valid=_restore_single(metrics["radiation_valid"], single=single),
            accepted_power_valid=_restore_single(
                metrics["accepted_power_valid"], single=single
            ),
        )
        stored_weights = _restore_single(a, single=single)
        return BeamData(
            weights=stored_weights,
            network=network,
            far_field=far_field,
            antenna=antenna,
            frequencies=self.frequencies,
            metadata={
                "basis_fingerprint": self.fingerprint,
                "solver_rerun": False,
                "radiated_power_source": (
                    "closed_surface_complex_poynting_quadratic"
                    if self.radiated_power_matrix is not None
                    else "far_field_angular_integral"
                ),
            },
        )


__all__ = [
    "ARRAY_FIELD_BASIS",
    "ARRAY_PERSISTENCE_SCHEMA_VERSION",
    "ARRAY_POWER_NORMALIZATION",
    "ARRAY_ACCEPTANCE_BUDGET",
    "ArrayBasisData",
    "BeamData",
    "BeamWeights",
    "EmbeddedElementPatternData",
]
