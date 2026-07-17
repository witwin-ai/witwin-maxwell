from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _power_normalized_antenna_metrics(
    *,
    e_theta: torch.Tensor,
    e_phi: torch.Tensor,
    observation_radius: torch.Tensor,
    wave_impedance: torch.Tensor,
    solid_angle_weights: torch.Tensor,
    incident_power: torch.Tensor,
    accepted_power: torch.Tensor,
    radiated_power: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Shared torch kernel for absolute-power antenna metrics.

    Field tensors use ``[..., F, T, P]`` and power tensors use ``[..., F]``.
    Undefined directivity/gain quantities carry explicit validity masks and NaNs;
    realized gain and EIRP remain defined for dark or fully reflected beams when
    incident power is positive.
    """

    intensity = (
        observation_radius.square()
        * (torch.abs(e_theta).square() + torch.abs(e_phi).square())
        / (2.0 * wave_impedance)
    )
    angular_p_rad = torch.sum(intensity * solid_angle_weights, dim=(-2, -1))
    p_rad = angular_p_rad if radiated_power is None else radiated_power
    if p_rad.shape != incident_power.shape:
        raise ValueError("radiated_power must have the same shape as incident_power.")
    if p_rad.device != incident_power.device or p_rad.dtype != incident_power.dtype:
        raise ValueError("radiated_power must share incident_power dtype and device.")
    radiation_valid = torch.isfinite(p_rad) & (p_rad > 0.0)
    accepted_valid = torch.isfinite(accepted_power) & (accepted_power > 0.0)
    incident_valid = torch.isfinite(incident_power) & (incident_power > 0.0)
    safe_p_rad = torch.where(radiation_valid, p_rad, torch.ones_like(p_rad))
    safe_accepted = torch.where(
        accepted_valid, accepted_power, torch.ones_like(accepted_power)
    )
    safe_incident = torch.where(
        incident_valid, incident_power, torch.ones_like(incident_power)
    )
    four_pi_intensity = 4.0 * math.pi * intensity
    real_nan = torch.full_like(intensity, float("nan"))
    spectral_nan = torch.full_like(p_rad, float("nan"))
    directivity = torch.where(
        radiation_valid[..., None, None],
        four_pi_intensity / safe_p_rad[..., None, None],
        real_nan,
    )
    gain = torch.where(
        accepted_valid[..., None, None],
        four_pi_intensity / safe_accepted[..., None, None],
        real_nan,
    )
    realized_gain = torch.where(
        incident_valid[..., None, None],
        four_pi_intensity / safe_incident[..., None, None],
        real_nan,
    )
    radiation_efficiency = torch.where(
        accepted_valid,
        p_rad / safe_accepted,
        spectral_nan,
    )
    mismatch_efficiency = torch.where(
        incident_valid,
        accepted_power / safe_incident,
        spectral_nan,
    )
    system_efficiency = torch.where(
        incident_valid,
        p_rad / safe_incident,
        spectral_nan,
    )
    return {
        "radiation_intensity": intensity,
        "p_rad": p_rad,
        "directivity": directivity,
        "gain": gain,
        "realized_gain": realized_gain,
        "radiation_efficiency": radiation_efficiency,
        "mismatch_efficiency": mismatch_efficiency,
        "system_efficiency": system_efficiency,
        "eirp": torch.amax(four_pi_intensity, dim=(-2, -1)),
        "radiation_valid": radiation_valid,
        "accepted_power_valid": accepted_valid,
        "incident_power_valid": incident_valid,
    }


@dataclass(frozen=True)
class Ludwig3:
    """Ludwig-3 co/cross-polarization basis.

    ``reference_angle=0`` makes the co-polarized direction the Cartesian
    x-axis of ``AntennaData.frame`` at boresight. Angles are in radians.
    """

    reference_angle: float = 0.0

    def __post_init__(self):
        angle = float(self.reference_angle)
        if not math.isfinite(angle):
            raise ValueError("reference_angle must be finite.")
        object.__setattr__(self, "reference_angle", angle)

    def project(
        self,
        e_theta: torch.Tensor,
        e_phi: torch.Tensor,
        phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project spherical field components onto the Ludwig-3 basis."""

        angle = phi - self.reference_angle
        cosine = torch.cos(angle)[None, ...]
        sine = torch.sin(angle)[None, ...]
        co_polarized = e_theta * cosine - e_phi * sine
        cross_polarized = e_theta * sine + e_phi * cosine
        return co_polarized, cross_polarized


@dataclass(frozen=True)
class AntennaData:
    """Device-resident frequency-domain antenna engineering results.

    Pattern tensors use fixed ``[frequency, theta, phi]`` order. The frequency
    dimension remains explicit even when it has length one.
    """

    frequencies: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    e_theta: torch.Tensor | None
    e_phi: torch.Tensor | None
    observation_radius: torch.Tensor | None
    wave_impedance: torch.Tensor | None
    radiation_intensity: torch.Tensor
    p_rad: torch.Tensor
    p_accepted: torch.Tensor
    p_incident: torch.Tensor
    directivity: torch.Tensor
    gain: torch.Tensor
    realized_gain: torch.Tensor
    radiation_efficiency: torch.Tensor
    mismatch_efficiency: torch.Tensor
    system_efficiency: torch.Tensor
    eirp: torch.Tensor
    co_polarized: torch.Tensor | None
    cross_polarized: torch.Tensor | None
    axial_ratio: torch.Tensor | None
    phase_center: torch.Tensor
    frame: torch.Tensor
    polarization_basis: Ludwig3
    driven_port_name: str
    surface_currents: tuple[object, ...] | None = None
    field_basis: str = "spherical_theta_phi"
    power_normalization: str = (
        "peak phasor; U=r^2*(|E_theta|^2+|E_phi|^2)/(2*eta); "
        "gain referenced to accepted power; realized gain referenced to incident power"
    )
    phasor_convention: str = "peak phasor with exp(-i*omega*t) time dependence"

    def __post_init__(self):
        if not isinstance(self.frequencies, torch.Tensor):
            raise TypeError("frequencies must be a torch.Tensor.")
        if self.frequencies.ndim != 1 or self.frequencies.numel() == 0:
            raise ValueError("frequencies must have non-empty shape [F].")
        if (
            self.frequencies.is_complex()
            or not self.frequencies.dtype.is_floating_point
        ):
            raise TypeError("frequencies must be a real floating-point tensor.")

        frequency_count = self.frequencies.numel()
        if self.theta.ndim != 2 or self.phi.shape != self.theta.shape:
            raise ValueError("theta and phi must have identical 2D shapes [T, P].")
        pattern_shape = (frequency_count, *self.theta.shape)
        pattern_names = (
            "radiation_intensity",
            "directivity",
            "gain",
            "realized_gain",
        )
        for name in pattern_names:
            value = getattr(self, name)
            if value.shape != pattern_shape:
                raise ValueError(f"{name} must have shape [F, T, P].")

        spectral_names = (
            "p_rad",
            "p_accepted",
            "p_incident",
            "radiation_efficiency",
            "mismatch_efficiency",
            "system_efficiency",
            "eirp",
        )
        for name in spectral_names:
            value = getattr(self, name)
            if value.shape != (frequency_count,):
                raise ValueError(f"{name} must have shape [F].")

        optional_patterns = (
            "e_theta",
            "e_phi",
            "observation_radius",
            "wave_impedance",
            "co_polarized",
            "cross_polarized",
            "axial_ratio",
        )
        present = [getattr(self, name) is not None for name in optional_patterns]
        if any(present) and not all(present):
            raise ValueError(
                "Polarization field and axial-ratio tensors must be present together."
            )
        for name in optional_patterns:
            value = getattr(self, name)
            if value is not None and value.shape != pattern_shape:
                raise ValueError(f"{name} must have shape [F, T, P].")

        if self.phase_center.shape != (3,):
            raise ValueError("phase_center must have shape [3].")
        if self.frame.shape != (3, 3):
            raise ValueError("frame must have shape [3, 3].")

        device = self.frequencies.device
        tensor_names = (
            "theta",
            "phi",
            *pattern_names,
            *spectral_names,
            *optional_patterns,
            "phase_center",
            "frame",
        )
        for name in tensor_names:
            value = getattr(self, name)
            if value is not None and value.device != device:
                raise ValueError("All AntennaData tensors must be on one device.")

        if self.field_basis != "spherical_theta_phi":
            raise ValueError("field_basis must be 'spherical_theta_phi'.")
        if not isinstance(self.polarization_basis, Ludwig3):
            raise TypeError("polarization_basis must be a Ludwig3 instance.")
        if not isinstance(self.driven_port_name, str) or not self.driven_port_name:
            raise ValueError("driven_port_name must be a non-empty string.")
        if self.surface_currents is not None:
            currents = tuple(self.surface_currents)
            if len(currents) != frequency_count:
                raise ValueError("surface_currents must contain one entry per frequency.")
            for current in currents:
                current_device = getattr(current, "device", None)
                if current_device is None:
                    raise TypeError(
                        "surface_currents entries must expose a device property."
                    )
                if torch.device(current_device) != device:
                    raise ValueError(
                        "surface_currents entries must be on the AntennaData device."
                    )
            object.__setattr__(self, "surface_currents", currents)

    @property
    def device(self) -> torch.device:
        return self.frequencies.device

    @property
    def directivity_db(self) -> torch.Tensor:
        return 10.0 * torch.log10(self.directivity)

    @property
    def gain_db(self) -> torch.Tensor:
        return 10.0 * torch.log10(self.gain)

    @property
    def realized_gain_db(self) -> torch.Tensor:
        return 10.0 * torch.log10(self.realized_gain)

    @property
    def axial_ratio_db(self) -> torch.Tensor | None:
        if self.axial_ratio is None:
            return None
        return 20.0 * torch.log10(self.axial_ratio)

    @property
    def directivity_max(self) -> torch.Tensor:
        return torch.amax(self.directivity, dim=(-2, -1))

    @property
    def gain_max(self) -> torch.Tensor:
        return torch.amax(self.gain, dim=(-2, -1))

    @property
    def realized_gain_max(self) -> torch.Tensor:
        return torch.amax(self.realized_gain, dim=(-2, -1))


__all__ = ["AntennaData", "Ludwig3"]
