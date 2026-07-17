from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, ClassVar, Literal

import torch


_SNAP_POLICIES = {"continuous", "nearest", "strict"}
_WIRE_QUANTITIES = {"current", "charge", "ohmic_loss"}


def _nonempty_name(value, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{field_name} must not be empty.")
    return resolved


def _require_real_floating_tensor(value: torch.Tensor, *, field_name: str) -> None:
    if value.is_complex() or not value.dtype.is_floating_point:
        raise TypeError(f"{field_name} tensor must be real floating point.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{field_name} tensor must contain only finite values.")


def _normalize_points(points) -> tuple[tuple[float, float, float], ...] | torch.Tensor:
    if isinstance(points, torch.Tensor):
        _require_real_floating_tensor(points, field_name="points")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("points tensor must have shape [P, 3] with P >= 2.")
        if not bool(
            torch.all(torch.linalg.vector_norm(torch.diff(points, dim=0), dim=1) > 0.0)
        ):
            raise ValueError(
                "points must not contain zero-length consecutive segments."
            )
        unique_count = int(torch.unique(points.detach(), dim=0).shape[0])
        closed = bool(torch.equal(points.detach()[0], points.detach()[-1]))
        if closed and points.shape[0] < 4:
            raise ValueError(
                "duplicate nodes are valid only for a closed loop with at least three segments."
            )
        expected_unique = int(points.shape[0]) - (1 if closed else 0)
        if unique_count != expected_unique:
            raise ValueError("points may repeat only the first node as the final node of a closed loop.")
        return points

    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError(
            "points must be a torch.Tensor or a sequence of 3D coordinates."
        )
    resolved = []
    for point in points:
        if (
            isinstance(point, (str, bytes))
            or not isinstance(point, Sequence)
            or len(point) != 3
        ):
            raise ValueError("each wire point must contain exactly three coordinates.")
        coordinates = []
        for coordinate in point:
            if isinstance(coordinate, bool) or isinstance(coordinate, complex):
                raise TypeError("wire point coordinates must be real scalars.")
            value = float(coordinate)
            if not math.isfinite(value):
                raise ValueError("wire point coordinates must be finite.")
            coordinates.append(value)
        resolved.append(tuple(coordinates))
    if len(resolved) < 2:
        raise ValueError("points must contain at least two coordinates.")
    if any(left == right for left, right in zip(resolved, resolved[1:])):
        raise ValueError("points must not contain zero-length consecutive segments.")
    closed = resolved[0] == resolved[-1]
    if closed and len(resolved) < 4:
        raise ValueError(
            "duplicate nodes are valid only for a closed loop with at least three segments."
        )
    expected_unique = len(resolved) - (1 if closed else 0)
    if len(set(resolved)) != expected_unique:
        raise ValueError("points may repeat only the first node as the final node of a closed loop.")
    return tuple(resolved)


def _segment_count(points) -> int:
    return (
        int(points.shape[0] - 1)
        if isinstance(points, torch.Tensor)
        else len(points) - 1
    )


def _normalize_radius(radius, *, segment_count: int):
    if isinstance(radius, torch.Tensor):
        _require_real_floating_tensor(radius, field_name="radius")
        if radius.ndim not in {0, 1}:
            raise ValueError("radius tensor must be scalar or have shape [S].")
        if radius.ndim == 1 and radius.shape != (segment_count,):
            raise ValueError(
                f"per-segment radius tensor must have shape [{segment_count}]."
            )
        if not bool(torch.all(radius > 0.0)):
            raise ValueError("radius must be positive.")
        return radius

    if isinstance(radius, bool) or isinstance(radius, complex):
        raise TypeError(
            "radius must be a positive real scalar or per-segment sequence."
        )
    if isinstance(radius, Sequence) and not isinstance(radius, (str, bytes)):
        values = tuple(float(value) for value in radius)
        if len(values) != segment_count:
            raise ValueError(f"per-segment radius must contain {segment_count} values.")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("radius values must be finite and positive.")
        return values
    value = float(radius)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("radius must be finite and positive.")
    return value


@dataclass(frozen=True)
class WireConductor:
    """Thin-wire conductor law. The current implementation exposes lossless PEC."""

    kind: Literal["pec"] = "pec"

    def __init__(self, kind: str = "pec"):
        resolved = str(kind).strip().lower()
        if resolved != "pec":
            raise ValueError("WireConductor currently supports the 'pec' conductor law.")
        object.__setattr__(self, "kind", "pec")

    @classmethod
    def pec(cls) -> "WireConductor":
        return cls("pec")


@dataclass(frozen=True)
class WireEnd:
    """Boundary condition for one physical endpoint of a thin wire."""

    kind: Literal["open", "grounded", "node"]
    structure: str | None = None
    node_name: str | None = None

    def __init__(
        self,
        kind: str = "open",
        *,
        structure: str | None = None,
        node_name: str | None = None,
    ):
        resolved = str(kind).strip().lower()
        if resolved not in {"open", "grounded", "node"}:
            raise ValueError("WireEnd kind must be 'open', 'grounded', or 'node'.")
        if resolved == "open":
            if structure is not None or node_name is not None:
                raise ValueError("WireEnd.open() cannot reference a structure or node name.")
            resolved_structure = None
            resolved_node_name = None
        elif resolved == "grounded":
            if node_name is not None:
                raise ValueError("WireEnd.grounded() cannot reference a node name.")
            resolved_structure = _nonempty_name(
                structure, field_name="grounded structure"
            )
            resolved_node_name = None
        else:
            if structure is not None:
                raise ValueError("WireEnd.node() cannot reference a structure.")
            resolved_structure = None
            resolved_node_name = _nonempty_name(node_name, field_name="wire node name")
            if resolved_node_name.startswith("__closed__:"):
                raise ValueError(
                    "wire node names beginning with '__closed__:' are reserved for internal loop identity."
                )
        object.__setattr__(self, "kind", resolved)
        object.__setattr__(self, "structure", resolved_structure)
        object.__setattr__(self, "node_name", resolved_node_name)

    @classmethod
    def open(cls) -> "WireEnd":
        return cls("open")

    @classmethod
    def grounded(cls, *, structure: str) -> "WireEnd":
        return cls("grounded", structure=structure)

    @classmethod
    def node(cls, name: str) -> "WireEnd":
        return cls("node", node_name=name)


@dataclass(frozen=True, eq=False)
class ThinWire:
    """Immutable centerline definition for a subgrid thin wire.

    Tensor ``radius`` inputs retain their PyTorch autograd graph. Tensor
    ``points`` retain coordinate gradients when ``snap="continuous"``; the
    compiled cell stencil remains a fixed, discrete decision.
    """

    name: str
    points: tuple[tuple[float, float, float], ...] | torch.Tensor
    radius: float | tuple[float, ...] | torch.Tensor
    conductor: WireConductor
    endpoints: tuple[WireEnd, ...]
    snap: Literal["continuous", "nearest", "strict"] = "strict"

    def __init__(
        self,
        name,
        points,
        radius,
        conductor,
        endpoints=None,
        snap="strict",
    ):
        resolved_name = _nonempty_name(name, field_name="ThinWire name")
        resolved_points = _normalize_points(points)
        resolved_radius = _normalize_radius(
            radius,
            segment_count=_segment_count(resolved_points),
        )
        if not isinstance(conductor, WireConductor):
            raise TypeError("conductor must be a WireConductor.")
        if endpoints is None:
            if self_closed := (
                bool(torch.equal(resolved_points[0], resolved_points[-1]))
                if isinstance(resolved_points, torch.Tensor)
                else resolved_points[0] == resolved_points[-1]
            ):
                resolved_endpoints = ()
            else:
                resolved_endpoints = (WireEnd.open(), WireEnd.open())
        else:
            self_closed = (
                bool(torch.equal(resolved_points[0], resolved_points[-1]))
                if isinstance(resolved_points, torch.Tensor)
                else resolved_points[0] == resolved_points[-1]
            )
            if self_closed:
                raise ValueError("closed-loop ThinWire points must not specify endpoints.")
            if not isinstance(endpoints, Sequence) or len(endpoints) != 2:
                raise ValueError("endpoints must contain exactly two WireEnd values.")
            resolved_endpoints = tuple(endpoints)
            if any(
                not isinstance(endpoint, WireEnd) for endpoint in resolved_endpoints
            ):
                raise TypeError("endpoints must contain only WireEnd values.")
        resolved_snap = str(snap).strip().lower()
        if resolved_snap not in _SNAP_POLICIES:
            raise ValueError("snap must be 'continuous', 'nearest', or 'strict'.")
        if (
            isinstance(resolved_points, torch.Tensor)
            and resolved_points.requires_grad
            and resolved_snap != "continuous"
        ):
            raise ValueError(
                "trainable points require snap='continuous' so the fixed-stencil "
                "coordinate-gradient contract is explicit."
            )

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "points", resolved_points)
        object.__setattr__(self, "radius", resolved_radius)
        object.__setattr__(self, "conductor", conductor)
        object.__setattr__(self, "endpoints", resolved_endpoints)
        object.__setattr__(self, "snap", resolved_snap)

    @property
    def segment_count(self) -> int:
        return _segment_count(self.points)

    @property
    def is_closed(self) -> bool:
        if isinstance(self.points, torch.Tensor):
            return bool(torch.equal(self.points.detach()[0], self.points.detach()[-1]))
        return self.points[0] == self.points[-1]


@dataclass(frozen=True, eq=False)
class WireData:
    """Frequency-domain current, charge, and loss for one wire monitor.

    Requested current and ohmic loss use shape ``[F, S]``; requested charge
    uses ``[F, N]``. Unrequested quantities are ``None``. Tensors remain
    device-resident and retain any live autograd graph.
    """

    monitor_name: str
    wire_name: str
    frequencies: torch.Tensor
    current: torch.Tensor | None
    charge: torch.Tensor | None
    ohmic_loss: torch.Tensor | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[int] = 1

    def __post_init__(self):
        monitor_name = _nonempty_name(self.monitor_name, field_name="monitor_name")
        wire_name = _nonempty_name(self.wire_name, field_name="wire_name")
        if not isinstance(self.frequencies, torch.Tensor):
            raise TypeError("WireData frequencies must be a torch tensor.")
        frequencies = self.frequencies
        if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
            raise TypeError("WireData frequencies must be real floating point.")
        if frequencies.ndim != 1 or frequencies.numel() == 0:
            raise ValueError("WireData frequencies must have non-empty shape [F].")
        if not bool(torch.all(torch.isfinite(frequencies))) or not bool(
            torch.all(frequencies > 0.0)
        ):
            raise ValueError(
                "WireData frequencies must be finite and strictly positive."
            )

        present = tuple(
            value
            for value in (self.current, self.charge, self.ohmic_loss)
            if value is not None
        )
        if not present:
            raise ValueError("WireData must contain at least one requested quantity.")
        if any(not isinstance(value, torch.Tensor) for value in present):
            raise TypeError("WireData quantities must be torch tensors or None.")
        if any(value.device != frequencies.device for value in present):
            raise ValueError("WireData tensors must reside on one device.")

        if self.current is not None:
            current = self.current
            if not current.is_complex():
                raise TypeError("WireData current must be complex floating point.")
            if (
                current.ndim != 2
                or current.shape[0] != frequencies.numel()
                or current.shape[1] == 0
            ):
                raise ValueError("WireData current must have non-empty shape [F, S].")
            if current.real.dtype != frequencies.dtype or not bool(
                torch.all(torch.isfinite(current))
            ):
                raise ValueError(
                    "WireData current must be finite and match frequency precision."
                )
        if self.charge is not None:
            charge = self.charge
            if not charge.is_complex():
                raise TypeError("WireData charge must be complex floating point.")
            if (
                charge.ndim != 2
                or charge.shape[0] != frequencies.numel()
                or charge.shape[1] == 0
            ):
                raise ValueError("WireData charge must have non-empty shape [F, N].")
            if charge.real.dtype != frequencies.dtype or not bool(
                torch.all(torch.isfinite(charge))
            ):
                raise ValueError(
                    "WireData charge must be finite and match frequency precision."
                )
        if (
            self.current is not None
            and self.charge is not None
            and self.current.dtype != self.charge.dtype
        ):
            raise ValueError(
                "WireData current and charge must share one complex dtype."
            )
        if self.ohmic_loss is not None:
            ohmic_loss = self.ohmic_loss
            if ohmic_loss.is_complex() or not ohmic_loss.dtype.is_floating_point:
                raise TypeError("WireData ohmic_loss must be real floating point.")
            if (
                ohmic_loss.ndim != 2
                or ohmic_loss.shape[0] != frequencies.numel()
                or ohmic_loss.shape[1] == 0
            ):
                raise ValueError("WireData ohmic_loss must have non-empty shape [F, S].")
            if self.current is not None and ohmic_loss.shape != self.current.shape:
                raise ValueError(
                    "WireData ohmic_loss must match current shape when both are present."
                )
            if ohmic_loss.dtype != frequencies.dtype:
                raise ValueError("WireData ohmic_loss must match frequency precision.")
            if not bool(
                torch.all(torch.isfinite(ohmic_loss) & (ohmic_loss >= 0.0))
            ):
                raise ValueError("WireData ohmic_loss must be finite and non-negative.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("WireData metadata must be a mapping.")

        object.__setattr__(self, "monitor_name", monitor_name)
        object.__setattr__(self, "wire_name", wire_name)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def device(self) -> torch.device:
        return self.frequencies.device


def normalize_wire_quantities(quantities) -> tuple[str, ...]:
    if isinstance(quantities, str):
        values = (quantities,)
    else:
        values = tuple(quantities)
    if not values:
        raise ValueError("WireMonitor quantities must not be empty.")
    normalized = tuple(str(value).strip().lower() for value in values)
    unknown = tuple(value for value in normalized if value not in _WIRE_QUANTITIES)
    if unknown:
        raise ValueError(f"Unsupported WireMonitor quantities {unknown}.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("WireMonitor quantities must be unique.")
    return normalized


__all__ = [
    "ThinWire",
    "WireConductor",
    "WireData",
    "WireEnd",
]
