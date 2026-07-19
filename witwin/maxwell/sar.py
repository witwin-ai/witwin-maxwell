from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

# The single versioned mass-averaging profile. It is IEEE/IEC-inspired but NOT
# certified: see docs and SARAveraging for the documented differences from
# IEEE/IEC 62704-1. Do not rename it to imply certification.
AVERAGING_PROFILE = "cubical-prefix-v1"
_SUPPORTED_PROFILES = (AVERAGING_PROFILE,)
_SUPPORTED_CONNECTIVITY = ("cube",)
_SUPPORTED_BOUNDARY_POLICY = ("strict-interior",)

SAR_UNIT = "W/kg"
POWER_UNIT = "W"

_SAR_RESULT_SCHEMA_VERSION = 1


def _coerce_mass_tuple(mass) -> tuple[float, ...]:
    if isinstance(mass, (int, float)):
        values = (float(mass),)
    else:
        values = tuple(float(m) for m in mass)
    if not values:
        raise ValueError("SARAveraging.mass must contain at least one target mass.")
    if any(not (m > 0.0) for m in values):
        raise ValueError("SARAveraging.mass values must be strictly positive (kg).")
    return values


@dataclass(frozen=True)
class SARAveraging:
    """Local mass-averaging request for SAR.

    ``mass`` is one or more target averaging masses in kg (e.g. ``1e-3`` for 1 g,
    ``10e-3`` for 10 g). ``profile`` selects the versioned averaging algorithm;
    only ``"cubical-prefix-v1"`` is supported. ``connectivity`` fixes how a cube's
    tissue is gathered (only ``"cube"``; tissue flood-fill is not in v1).
    ``boundary_policy`` fixes cube-clipping behaviour at the monitor edge
    (``"strict-interior"`` marks clipped centers invalid rather than padding).
    ``min_tissue_fraction`` is the minimum tissue fill of a valid averaging cube
    (the "no air-mass makeup" rule).

    The averaging computation itself is delivered by the mass-averaging stage; this
    spec is a validated, serializable request object.
    """

    mass: tuple[float, ...]
    profile: str = AVERAGING_PROFILE
    connectivity: str = "cube"
    boundary_policy: str = "strict-interior"
    min_tissue_fraction: float = 0.1

    def __init__(
        self,
        mass,
        profile: str = AVERAGING_PROFILE,
        connectivity: str = "cube",
        boundary_policy: str = "strict-interior",
        min_tissue_fraction: float = 0.1,
    ):
        object.__setattr__(self, "mass", _coerce_mass_tuple(mass))
        if profile not in _SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported SAR averaging profile {profile!r}; choices are {_SUPPORTED_PROFILES}."
            )
        if connectivity not in _SUPPORTED_CONNECTIVITY:
            raise NotImplementedError(
                f"SAR averaging connectivity {connectivity!r} is not implemented in "
                f"{AVERAGING_PROFILE}; only {_SUPPORTED_CONNECTIVITY} (no tissue flood-fill) is supported."
            )
        if boundary_policy not in _SUPPORTED_BOUNDARY_POLICY:
            raise NotImplementedError(
                f"SAR averaging boundary_policy {boundary_policy!r} is not implemented; "
                f"only {_SUPPORTED_BOUNDARY_POLICY} is supported."
            )
        fraction = float(min_tissue_fraction)
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("SARAveraging.min_tissue_fraction must lie in [0, 1].")
        object.__setattr__(self, "profile", str(profile))
        object.__setattr__(self, "connectivity", str(connectivity))
        object.__setattr__(self, "boundary_policy", str(boundary_policy))
        object.__setattr__(self, "min_tissue_fraction", fraction)

    def payload(self) -> dict[str, Any]:
        return {
            "mass": list(self.mass),
            "profile": self.profile,
            "version": AVERAGING_PROFILE,
            "connectivity": self.connectivity,
            "boundary_policy": self.boundary_policy,
            "min_tissue_fraction": self.min_tissue_fraction,
        }


@dataclass(frozen=True)
class PowerNormalization:
    """Multiplicative power scaling applied to SAR, with provenance.

    Constructed through the ``source`` / ``accepted_power`` / ``input_power``
    factories. SAR is linear in absorbed power, so a target/measured power ratio
    scales SAR directly; ``source(amplitude=a)`` scales by ``a**2`` (fields scale
    by ``a``). The scale is resolved against a ``Result`` at reduction time.
    """

    kind: str
    amplitude: float | None = None
    port: str | None = None
    watts: float | None = None

    def __post_init__(self):
        if self.kind not in ("source", "accepted_power", "input_power"):
            raise ValueError(f"Unknown PowerNormalization kind {self.kind!r}.")

    @classmethod
    def source(cls, amplitude: float = 1.0) -> "PowerNormalization":
        amplitude = float(amplitude)
        if not (amplitude > 0.0):
            raise ValueError("PowerNormalization.source amplitude must be > 0.")
        return cls(kind="source", amplitude=amplitude)

    @classmethod
    def none(cls) -> "PowerNormalization":
        return cls(kind="source", amplitude=1.0)

    @classmethod
    def accepted_power(cls, port: str, watts: float) -> "PowerNormalization":
        watts = float(watts)
        if not (watts > 0.0):
            raise ValueError("PowerNormalization.accepted_power watts must be > 0.")
        return cls(kind="accepted_power", port=str(port), watts=watts)

    @classmethod
    def input_power(cls, watts: float) -> "PowerNormalization":
        watts = float(watts)
        if not (watts > 0.0):
            raise ValueError("PowerNormalization.input_power watts must be > 0.")
        return cls(kind="input_power", watts=watts)

    def resolve_scale(self, result=None, *, measured_power=None):
        """Return the multiplicative power scale (SAR is linear in this scale).

        ``source`` resolves to ``amplitude**2`` with no solver data. The
        port-accepted-power and total-input-power scalings need a measured power
        from the run and are wired in the normalization stage; until then they
        fail closed rather than silently returning an unnormalized scale.
        """
        if self.kind == "source":
            return float(self.amplitude) ** 2
        raise NotImplementedError(
            f"PowerNormalization.{self.kind} scaling is resolved in the SAR normalization "
            "stage (it consumes the run's measured accepted/input power); it is not yet "
            "available from this reducer."
        )

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "amplitude": self.amplitude,
            "port": self.port,
            "watts": self.watts,
        }


@dataclass(frozen=True)
class SARResult:
    """Typed SAR reduction with units, coordinates, and algorithm provenance.

    ``point`` maps each electric loss channel plus ``"total"`` to the per-cell
    point SAR field ``[F, nx, ny, nz]`` in W/kg (NaN at invalid cells). ``averaged``
    and ``peaks`` are populated by the mass-averaging stage. ``statistics`` holds
    per-tissue absorbed power and SAR summaries. Large tensors keep their device
    and autograd graph; discrete masks/ids are stop-grad.
    """

    point: Mapping[str, torch.Tensor]
    statistics: Mapping[int, Mapping[str, Any]]
    normalization: PowerNormalization
    provenance: Mapping[str, Any]
    frequencies: torch.Tensor
    coordinates: Mapping[str, torch.Tensor]
    valid: torch.Tensor
    occupancy: torch.Tensor
    rho_cell: torch.Tensor
    cell_volume: torch.Tensor
    tissue_id: torch.Tensor
    tissue_names: Mapping[int, str]
    absorbed_power_density: Mapping[str, torch.Tensor] = field(default_factory=dict)
    averaged: Mapping[str, Any] = field(default_factory=dict)
    peaks: Mapping[str, Any] = field(default_factory=dict)
    sar_unit: str = SAR_UNIT
    power_unit: str = POWER_UNIT

    @property
    def device(self) -> torch.device:
        return self.rho_cell.device

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.point)

    def point_sar(self, channel: str = "total") -> torch.Tensor:
        channel_name = str(channel)
        if channel_name not in self.point:
            raise KeyError(
                f"SAR channel {channel_name!r} is unavailable; choices are {self.channels}."
            )
        return self.point[channel_name]

    def peak(self, mass: float):
        raise NotImplementedError(
            "Mass-averaged peak SAR is produced by the mass-averaging stage; this result "
            "carries only point SAR and per-tissue statistics."
        )

    def soft_peak(self, temperature: float):
        raise NotImplementedError(
            "Differentiable soft_peak is produced by the differentiable-workflow stage; it is "
            "explicitly non-regulatory and not available from this reducer."
        )

    def payload(self) -> dict[str, Any]:
        """Detached CPU serialization payload (typed data, not anonymous metadata)."""

        def _cpu(mapping):
            return {name: tensor.detach().to("cpu") for name, tensor in mapping.items()}

        return {
            "schema_version": _SAR_RESULT_SCHEMA_VERSION,
            "data_type": "SARResult",
            "sar_unit": self.sar_unit,
            "power_unit": self.power_unit,
            "values": _cpu(self.point),
            "absorbed_power_density": _cpu(self.absorbed_power_density),
            "coordinates": _cpu(self.coordinates),
            "frequencies": self.frequencies.detach().to("cpu"),
            "field_convention": self.provenance.get("field_convention"),
            "normalization": self.normalization.payload(),
            "averaging_profile": self.provenance.get("averaging_profile"),
            "grid_hash": self.provenance.get("grid_hash"),
            "provenance": dict(self.provenance),
            "valid": self.valid.detach().to("cpu"),
            "occupancy": self.occupancy.detach().to("cpu"),
            "rho_cell": self.rho_cell.detach().to("cpu"),
            "cell_volume": self.cell_volume.detach().to("cpu"),
            "tissue_id": self.tissue_id.detach().to("cpu"),
            "tissue_names": dict(self.tissue_names),
            "statistics": {
                int(tid): {
                    key: (value.detach().to("cpu") if torch.is_tensor(value) else value)
                    for key, value in stats.items()
                }
                for tid, stats in self.statistics.items()
            },
        }

    def save(self, path: str | Path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload(), output_path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> "SARResult":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or payload.get("data_type") != "SARResult":
            raise ValueError("File does not contain a SARResult payload.")
        norm = payload["normalization"]
        normalization = PowerNormalization(
            kind=norm["kind"],
            amplitude=norm.get("amplitude"),
            port=norm.get("port"),
            watts=norm.get("watts"),
        )
        statistics = {
            int(tid): MappingProxyType(dict(stats))
            for tid, stats in payload.get("statistics", {}).items()
        }
        return cls(
            point=MappingProxyType(dict(payload["values"])),
            statistics=MappingProxyType(statistics),
            normalization=normalization,
            provenance=MappingProxyType(dict(payload.get("provenance", {}))),
            frequencies=payload["frequencies"],
            coordinates=MappingProxyType(dict(payload["coordinates"])),
            valid=payload["valid"],
            occupancy=payload["occupancy"],
            rho_cell=payload["rho_cell"],
            cell_volume=payload["cell_volume"],
            tissue_id=payload["tissue_id"],
            tissue_names=MappingProxyType(dict(payload.get("tissue_names", {}))),
            absorbed_power_density=MappingProxyType(dict(payload.get("absorbed_power_density", {}))),
        )


__all__ = [
    "SARAveraging",
    "PowerNormalization",
    "SARResult",
    "AVERAGING_PROFILE",
    "SAR_UNIT",
    "POWER_UNIT",
]
