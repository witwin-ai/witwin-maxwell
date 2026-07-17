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
    """Frozen Phase 0 acceptance thresholds.

    Basis-versus-direct comparison is a linearity check on a single solver, so it
    must agree to solver precision rather than to an engineering tolerance. The
    two scenes that run it do not share a truncation error, so they do not share
    a threshold:

    - the converged Phase 1 benchmark scene (96^3 cells, 4096 steps) drives the
      ``phase1_fdtd_*`` gates;
    - the coarse contract scene (192 steps, 4 absorbing cells) truncates a
      Gaussian pulse long before it decays and drives the ``contract_fdtd_*``
      gates.

    Threshold changes must retain the previous value, the measured evidence, and
    the technical reason. See ``docs/plans/array-active-s-mimo-implementation.md``.
    """

    analytic_rtol: float = 1.0e-6
    analytic_atol: float = 1.0e-10
    cuda_complex64_rtol: float = 2.0e-5
    cuda_complex64_atol: float = 1.0e-6
    # Was 0.03 for both scenes. Measured worst case on the coarse 4-PML-layer
    # contract scene is 1.433e-4 (four-element endfire), so 0.03 could not
    # discriminate a real superposition regression from noise.
    contract_fdtd_complex_l2: float = 5.0e-3
    # Was 3.0 deg. Measured worst case on the coarse 4-PML-layer contract scene is
    # 6.776e-3 deg (four-element endfire).
    contract_fdtd_phase_rms_deg: float = 0.5
    # Was 0.03. Recorded Phase 1 benchmark worst case is 2.219e-6 (endfire).
    phase1_fdtd_complex_l2: float = 1.0e-4
    # Was 3.0 deg. Recorded Phase 1 benchmark worst case is 1.518e-4 deg (endfire).
    phase1_fdtd_phase_rms_deg: float = 1.0e-2
    fdtd_phase_support_fraction: float = 0.10
    # Was 0.01. Measured worst case on the coarse 4-PML-layer contract scene is
    # 9.015e-6 (two-element endfire accepted power).
    port_power_relative_error: float = 5.0e-3
    physical_power_residual: float = 0.01
    # Q_rad is the Hermitian (real-power) part of the closed-surface complex
    # Poynting operator, so it is positive semidefinite in exact arithmetic. The
    # earlier -1e-3 floor was masking PML under-absorption: at 2 PML layers the
    # NF2FF box sits ~1 cell from the boundary and reflected field contaminates
    # the closed-surface Poynting integral, driving min_eig negative (-2.833e-5
    # ratio, four-element). Raising the contract scene to 4 PML layers restores a
    # positive-definite spectrum (measured worst min/max ratio +1.449e-6,
    # four-element; +4.215e-2 two-element), so the gate now enforces PSD with only
    # a floating-point roundoff allowance. eigvalsh backward error is ~1e-16*max
    # eigenvalue, so -1e-9*max is a conservative roundoff band that still rejects
    # the 2-layer artifact (four decades below the floor) while the 4-layer
    # spectrum clears it by three decades.
    radiated_power_psd_relative_floor: float = 1.0e-9
    reference_gain_error_db: float = 0.25
    reference_ecc_error: float = 0.02
    active_impedance_magnitude_error: float = 0.05
    active_impedance_phase_error_deg: float = 3.0
    gradient_relative_error: float = 0.02
    gradient_absolute_floor: float = 1.0e-8
    distributed_field_max_abs: float = 2.0e-6
    distributed_field_max_rel: float = 2.0e-5
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
class BeamCodebook:
    """A named set of incident power-wave weight vectors over one basis.

    ``weights`` has the port axis last and carries a leading beam axis:
    ``[B, N]`` (frequency-flat) or ``[B, F, N]`` (frequency-dependent). The
    codebook holds no solver state; it is combined against an
    :class:`ArrayBasisData` with zero additional field-solver steps.
    """

    weights: torch.Tensor
    names: tuple[str, ...]
    target_angles: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.weights, torch.Tensor):
            raise TypeError("BeamCodebook.weights must be a torch.Tensor.")
        if not self.weights.is_complex():
            raise TypeError("BeamCodebook.weights must be a complex tensor.")
        if self.weights.ndim not in (2, 3) or self.weights.numel() == 0:
            raise ValueError("BeamCodebook.weights must have shape [B, N] or [B, F, N].")
        if not _finite_complex(self.weights):
            raise ValueError("BeamCodebook.weights must contain only finite values.")
        names = tuple(str(name) for name in self.names)
        if len(names) != self.weights.shape[0] or any(not name for name in names):
            raise ValueError("BeamCodebook.names must contain one non-empty name per beam.")
        if len(set(names)) != len(names):
            raise ValueError("BeamCodebook.names must be unique.")
        if self.target_angles is not None:
            angles = self.target_angles
            if not isinstance(angles, torch.Tensor) or angles.is_complex():
                raise TypeError("target_angles must be a real torch.Tensor.")
            if angles.shape != (self.weights.shape[0], 2):
                raise ValueError("target_angles must have shape [B, 2] as (theta, phi).")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def beam_count(self) -> int:
        return self.weights.shape[0]

    def to_weights(
        self,
        *,
        frequency_count: int,
        port_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return frequency-explicit ``[B, F, N]`` weights for one basis."""

        values = self.weights
        if values.device != device:
            raise ValueError(f"BeamCodebook.weights must be on device {device}.")
        if values.dtype != dtype:
            raise TypeError(f"BeamCodebook.weights must use dtype {dtype}.")
        if values.ndim == 2:
            if values.shape[1] != port_count:
                raise ValueError(
                    f"frequency-flat codebook weights must have shape [B, N] with N={port_count}."
                )
            return values[:, None, :].expand(values.shape[0], frequency_count, port_count)
        if values.shape[1:] != (frequency_count, port_count):
            raise ValueError(
                f"codebook weights must have shape [B, F, N] with [F, N]=({frequency_count}, {port_count})."
            )
        return values

    @classmethod
    def from_scan_angles(
        cls,
        *,
        element_positions: torch.Tensor,
        frequencies: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        amplitude: torch.Tensor | float = 1.0,
        speed_of_light: float = 299792458.0,
        name_prefix: str = "scan",
    ) -> "BeamCodebook":
        """Build progressive-phase steering weights for a grid of scan angles.

        For each scan direction ``(theta_b, phi_b)`` the port weight is
        ``a_{b,n} = amplitude_n * exp(+j k (r_n . d_b))`` with ``d_b`` the unit
        pointing vector, so the co-phased main beam is steered toward ``d_b``.
        Element positions are supplied explicitly; no geometry is inferred.
        """

        if not isinstance(element_positions, torch.Tensor) or element_positions.ndim != 2:
            raise TypeError("element_positions must be a real tensor of shape [N, 3].")
        if element_positions.shape[1] != 3 or element_positions.is_complex():
            raise ValueError("element_positions must have shape [N, 3] with real coordinates.")
        device = element_positions.device
        real_dtype = element_positions.dtype
        for name, value in (("frequencies", frequencies), ("theta", theta), ("phi", phi)):
            if not isinstance(value, torch.Tensor) or value.is_complex():
                raise TypeError(f"{name} must be a real torch.Tensor.")
            if value.device != device:
                raise ValueError(f"{name} must be on device {device}.")
        if theta.ndim != 1 or phi.ndim != 1 or theta.shape != phi.shape:
            raise ValueError("theta and phi must be 1-D tensors of equal length (one per scan angle).")
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        port_count = element_positions.shape[0]
        amplitude_tensor = torch.as_tensor(amplitude, device=device, dtype=complex_dtype)
        if amplitude_tensor.ndim == 0:
            amplitude_tensor = amplitude_tensor.expand(port_count)
        if amplitude_tensor.shape != (port_count,):
            raise ValueError("amplitude must be a scalar or a per-port vector of length N.")
        directions = torch.stack(
            (
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ),
            dim=-1,
        )
        wave_number = (2.0 * math.pi * frequencies / speed_of_light).to(real_dtype)
        projection = torch.einsum("bd,nd->bn", directions, element_positions)
        phase = wave_number[None, :, None] * projection[:, None, :]
        weights = amplitude_tensor[None, None, :] * torch.exp(1j * phase.to(complex_dtype))
        names = tuple(
            f"{name_prefix}_theta{float(theta[index]):.4f}_phi{float(phi[index]):.4f}"
            for index in range(theta.shape[0])
        )
        return cls(
            weights=weights,
            names=names,
            target_angles=torch.stack((theta, phi), dim=-1),
            metadata={"builder": "from_scan_angles", "steering": "progressive_phase"},
        )


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

    @property
    def is_batched(self) -> bool:
        return self.weights.ndim == 3

    def max_hold(self, metric: str = "realized_gain") -> "MaxHoldComposite":
        """Envelope (per-direction maximum) of a metric across the beam axis.

        The envelope value carries a subgradient through the winning beam; the
        discrete winning-beam index is not differentiable, as the plan requires.
        Angular metrics reduce ``[B, F, T, P]`` to an ``[F, T, P]`` envelope with
        an ``[F, T, P]`` winning-beam index; the scalar ``eirp`` reduces
        ``[B, F]`` to an ``[F]`` envelope with an ``[F]`` index.
        """

        if not self.is_batched:
            raise ValueError("max_hold requires a batched beam result (rank-3 weights [B, F, N]).")
        angular_metrics = {
            "realized_gain": self.antenna.realized_gain,
            "gain": self.antenna.gain,
            "directivity": self.antenna.directivity,
            "radiation_intensity": self.antenna.radiation_intensity,
        }
        if metric == "eirp":
            source = self.antenna.eirp
        elif metric in angular_metrics:
            source = angular_metrics[metric]
        else:
            raise ValueError(
                "metric must be one of 'realized_gain', 'gain', 'directivity', "
                "'radiation_intensity', or 'eirp'."
            )
        if bool(torch.any(torch.isnan(source))):
            raise ValueError(
                f"max_hold({metric!r}) is undefined because at least one beam has a masked "
                "(NaN) metric; drive every beam with positive power before taking an envelope."
            )
        envelope = torch.amax(source, dim=0)
        winning_beam = torch.argmax(source.detach(), dim=0)
        return MaxHoldComposite(
            metric=metric,
            envelope=envelope,
            winning_beam=winning_beam,
            frequencies=self.frequencies,
            names=self.names,
        )


@dataclass(frozen=True)
class MaxHoldComposite:
    """Per-direction envelope of a beam metric across a codebook."""

    metric: str
    envelope: torch.Tensor
    winning_beam: torch.Tensor
    frequencies: torch.Tensor
    names: tuple[str, ...] | None = None

    @property
    def device(self) -> torch.device:
        return self.envelope.device


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

    def combine(
        self, weights: BeamWeights | BeamCodebook | torch.Tensor
    ) -> BeamData:
        """Combine incident power-wave weights without rerunning the field solver."""

        frequency_count, port_count, _ = self.network.s.shape
        beam_names: tuple[str, ...] | None = None
        codebook_metadata: dict[str, Any] = {}
        if isinstance(weights, BeamCodebook):
            beam_names = weights.names
            codebook_metadata = {
                "codebook": True,
                "beam_count": weights.beam_count,
                "codebook_metadata": dict(weights.metadata),
            }
            weights = weights.to_weights(
                frequency_count=frequency_count,
                port_count=port_count,
                device=self.device,
                dtype=self.dtype,
            )
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
            names=beam_names,
            metadata={
                "basis_fingerprint": self.fingerprint,
                "solver_rerun": False,
                "radiated_power_source": (
                    "closed_surface_complex_poynting_quadratic"
                    if self.radiated_power_matrix is not None
                    else "far_field_angular_integral"
                ),
                **codebook_metadata,
            },
        )

    @property
    def cache_key(self) -> str:
        """Content fingerprint used as the basis reuse key.

        The fingerprint is computed at extraction from scene physical content,
        resolved grid, boundaries, ports, terminations, frequencies, monitor
        surface, angular grid, polarization, phase center, and dtype. Combining
        different beam weights never changes this key (weights are not part of
        the basis); any geometry, material, port, frequency, or surface change
        does, because those tensors feed the digest.
        """

        return self.fingerprint

    def scene_gradient_vjp(self, *args, **kwargs):
        """Aggregated scene-parameter adjoint through the N-column basis.

        Fail-closed: the retained-column basis stores detached embedded-pattern
        tensors, so it cannot back-propagate to scene materials or geometry. The
        aggregated per-column adjoint envelope (plan 06 Phase 4, gated on the
        plan 02 Phase 7 distributed result-aggregation contract) is not wired to
        this single-device basis. Weight gradients through :meth:`combine` are
        fully supported and require no solver rerun.
        """

        raise NotImplementedError(
            "Scene-parameter gradients through the array basis require the aggregated "
            "per-column adjoint envelope (plan 06 Phase 4 / plan 02 Phase 7); this "
            "single-device retained-column basis only supports weight gradients through "
            "combine()."
        )

    def _environment_spectra(
        self, environment: "MultipathEnvironment"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patterns = self.embedded_patterns
        if not torch.equal(environment.theta, patterns.theta) or not torch.equal(
            environment.phi, patterns.phi
        ):
            raise ValueError(
                "MultipathEnvironment angular grid must match the embedded-pattern grid exactly."
            )
        angular_weights = _solid_angle_weights(patterns.theta, patterns.phi)
        density = environment.resolved_power_density(
            device=self.device, dtype=patterns.theta.dtype
        )
        total = torch.sum(density * angular_weights)
        if not bool(total > 0.0):
            raise ValueError("MultipathEnvironment power_density must integrate to a positive value.")
        normalized = density / total
        xpr = environment.cross_polar_ratio(device=self.device, dtype=patterns.theta.dtype)
        return normalized, angular_weights, xpr, total

    def mimo(self, environment: "MultipathEnvironment") -> "MIMOData":
        """Field-based MIMO metrics from the dual-polarized embedded patterns.

        The complex correlation matrix, ECC, apparent diversity gain, and mean
        effective gain are integrated over the full sphere against the supplied
        angular power spectrum and cross-polar ratio (Taga/Vaughan convention).
        """

        patterns = self.embedded_patterns
        normalized, angular_weights, xpr, _ = self._environment_spectra(environment)
        rho = environment.polarization_correlation_value(
            device=self.device, dtype=self.dtype
        )
        # P_theta and P_phi are each normalized to integrate to unity; the XPR
        # multiplier carries the polarization power imbalance explicitly.
        coef_theta = (xpr * normalized * angular_weights).to(self.dtype)
        coef_phi = (normalized * angular_weights).to(self.dtype)
        coef_cross = (torch.sqrt(xpr) * normalized * angular_weights).to(self.dtype)
        e_theta = patterns.e_theta
        e_phi = patterns.e_phi
        correlation = (
            torch.einsum("fitp,fjtp,tp->fij", e_theta, torch.conj(e_theta), coef_theta)
            + torch.einsum("fitp,fjtp,tp->fij", e_phi, torch.conj(e_phi), coef_phi)
            + rho
            * torch.einsum("fitp,fjtp,tp->fij", e_theta, torch.conj(e_phi), coef_cross)
            + torch.conj(rho)
            * torch.einsum("fitp,fjtp,tp->fij", e_phi, torch.conj(e_theta), coef_cross)
        )
        # Symmetrize away Hermitian roundoff from the independent einsums.
        correlation = 0.5 * (correlation + correlation.mH)
        diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1).real
        denominator = diagonal[..., :, None] * diagonal[..., None, :]
        ecc = torch.abs(correlation).square() / denominator
        diversity_gain = 10.0 * torch.sqrt(torch.clamp(1.0 - ecc, min=0.0))

        # Mean effective gain: realized-gain patterns (per unit incident power,
        # so the EEP columns already carry P_incident = 1 W) split by polarization
        # and integrated against the XPR-weighted angular spectra.
        radius_sq = patterns.observation_radius.square()
        u_theta = radius_sq * torch.abs(e_theta).square() / (2.0 * patterns.wave_impedance)
        u_phi = radius_sq * torch.abs(e_phi).square() / (2.0 * patterns.wave_impedance)
        gain_theta = 4.0 * math.pi * u_theta
        gain_phi = 4.0 * math.pi * u_phi
        weighted = normalized * angular_weights
        xpr_ratio = xpr / (1.0 + xpr)
        mean_effective_gain = (
            xpr_ratio * torch.einsum("fitp,tp->fi", gain_theta, weighted)
            + (1.0 - xpr_ratio) * torch.einsum("fitp,tp->fi", gain_phi, weighted)
        )
        return MIMOData(
            correlation=correlation,
            ecc=ecc,
            diversity_gain=diversity_gain,
            mean_effective_gain=mean_effective_gain,
            frequencies=self.frequencies,
            port_names=self.port_names,
            environment_metadata=environment.metadata_snapshot(),
            source="dual_polarized_far_field_integral",
        )

    def ecc_from_scattering(self) -> torch.Tensor:
        """S-parameter ECC approximation (Blanch/Thaysen), lossless assumption.

        Uses ``ecc_ij = |sum_n conj(S_ni) S_nj|^2 / prod_k (1 - sum_n |S_nk|^2)``.
        This is only valid for a lossless, high-radiation-efficiency array; it is
        deliberately a different method from the field-based :meth:`mimo` ECC and
        must not be used when material or ohmic loss is significant.
        """

        s = self.network.s
        column_power = torch.sum(torch.abs(s).square(), dim=-2)  # [F, N]
        available = 1.0 - column_power
        if bool(torch.any(available <= 0.0)):
            raise ValueError(
                "S-parameter ECC approximation requires 1 - sum_n |S_ni|^2 > 0 for every port; "
                "the array is not lossless/high-efficiency enough for this approximation."
            )
        numerator = torch.abs(
            torch.einsum("fni,fnj->fij", torch.conj(s), s)
        ).square()
        denominator = available[..., :, None] * available[..., None, :]
        return numerator / denominator


@dataclass(frozen=True)
class MultipathEnvironment:
    """Angular power spectrum and polarization statistics of an incident field."""

    theta: torch.Tensor
    phi: torch.Tensor
    power_density: torch.Tensor | float = 1.0
    cross_polar_ratio_db: float = 0.0
    polarization_correlation: complex = 0.0

    def __post_init__(self):
        if not isinstance(self.theta, torch.Tensor) or not isinstance(self.phi, torch.Tensor):
            raise TypeError("theta and phi must be torch.Tensor instances.")
        _validate_angular_grid(self.theta, self.phi, device=self.theta.device)
        if abs(complex(self.polarization_correlation)) > 1.0:
            raise ValueError("polarization_correlation magnitude must be <= 1.")
        if not math.isfinite(float(self.cross_polar_ratio_db)):
            raise ValueError("cross_polar_ratio_db must be finite.")

    def resolved_power_density(self, *, device, dtype) -> torch.Tensor:
        if isinstance(self.power_density, torch.Tensor):
            if self.power_density.shape != self.theta.shape:
                raise ValueError("power_density tensor must match the [T, P] angular grid.")
            density = self.power_density.to(device=device, dtype=dtype)
        else:
            density = torch.full(
                self.theta.shape, float(self.power_density), device=device, dtype=dtype
            )
        if not bool(torch.all(torch.isfinite(density))) or not bool(torch.all(density >= 0.0)):
            raise ValueError("power_density must be finite and non-negative.")
        return density

    def cross_polar_ratio(self, *, device, dtype) -> torch.Tensor:
        return torch.tensor(
            10.0 ** (float(self.cross_polar_ratio_db) / 10.0), device=device, dtype=dtype
        )

    def polarization_correlation_value(self, *, device, dtype) -> torch.Tensor:
        return torch.tensor(complex(self.polarization_correlation), device=device, dtype=dtype)

    def metadata_snapshot(self) -> dict[str, Any]:
        return {
            "cross_polar_ratio_db": float(self.cross_polar_ratio_db),
            "polarization_correlation": complex(self.polarization_correlation),
            "angular_shape": tuple(int(v) for v in self.theta.shape),
            "power_density": (
                "uniform"
                if not isinstance(self.power_density, torch.Tensor)
                else "custom_angular"
            ),
        }


@dataclass(frozen=True)
class MIMOData:
    """MIMO correlation metrics with the environment that produced them."""

    correlation: torch.Tensor
    ecc: torch.Tensor
    diversity_gain: torch.Tensor
    mean_effective_gain: torch.Tensor
    frequencies: torch.Tensor
    port_names: tuple[str, ...]
    environment_metadata: Mapping[str, Any]
    source: str

    def __post_init__(self):
        object.__setattr__(self, "environment_metadata", dict(self.environment_metadata))

    @property
    def device(self) -> torch.device:
        return self.correlation.device


__all__ = [
    "ARRAY_FIELD_BASIS",
    "ARRAY_PERSISTENCE_SCHEMA_VERSION",
    "ARRAY_POWER_NORMALIZATION",
    "ARRAY_ACCEPTANCE_BUDGET",
    "ArrayBasisData",
    "BeamCodebook",
    "BeamData",
    "BeamWeights",
    "EmbeddedElementPatternData",
    "MaxHoldComposite",
    "MIMOData",
    "MultipathEnvironment",
]
