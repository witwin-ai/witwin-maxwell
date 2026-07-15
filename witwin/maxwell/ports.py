from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from witwin.core import Box

from .monitors import ModeMonitor, _normalize_frequencies, _normalize_normal_direction
from .sources import (
    ModeSource,
    SourceTime,
    _normalize_bend,
    _normalize_mode_direction,
    _require_length3,
    _require_mode_source_size,
    _resolve_mode_source_normal_axis,
    _resolve_mode_source_polarization_axis,
    polarization_vector,
)


@dataclass(frozen=True)
class AxisPath:
    """Axis-aligned voltage integration path.

    The path orientation is supplied by the owning port's ``negative`` and
    ``positive`` terminals; this object only fixes the Cartesian axis.
    """

    axis: str

    def __init__(self, axis):
        if not isinstance(axis, str):
            raise TypeError("AxisPath axis must be one of 'x', 'y', or 'z'.")
        resolved_axis = axis.strip().lower()
        if resolved_axis not in {"x", "y", "z"}:
            raise ValueError("AxisPath axis must be one of 'x', 'y', or 'z'.")
        object.__setattr__(self, "axis", resolved_axis)


@dataclass(frozen=True)
class LumpedPort:
    """Declare a two-terminal lumped port with explicit discrete V/I geometry.

    Voltage is the line integral of E from ``negative`` to ``positive``.
    Current is the right-hand circulation of H around ``current_surface`` with
    the same orientation. Phasors are peak-valued, so average power is
    ``0.5 * Re(V * conj(I))``.
    """

    name: str
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    voltage_path: AxisPath
    current_surface: Box
    reference_impedance: complex | float | torch.Tensor
    reference_plane: float | None = None
    kind: str = "lumped_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"

    def __init__(
        self,
        name,
        *,
        positive,
        negative,
        voltage_path,
        current_surface,
        reference_impedance=50.0,
        reference_plane=None,
    ):
        if not isinstance(voltage_path, AxisPath):
            raise TypeError("voltage_path must be an AxisPath.")
        if not isinstance(current_surface, Box):
            raise TypeError("current_surface must be a witwin.core.Box.")

        resolved_positive = _require_length3("positive", positive)
        resolved_negative = _require_length3("negative", negative)
        if resolved_positive == resolved_negative:
            raise ValueError("LumpedPort positive and negative terminals must be distinct.")

        axis_index = "xyz".index(voltage_path.axis)
        transverse_indices = tuple(index for index in range(3) if index != axis_index)
        if any(
            not math.isclose(
                resolved_positive[index],
                resolved_negative[index],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for index in transverse_indices
        ):
            raise ValueError(
                "LumpedPort positive and negative terminals must define an axis-aligned voltage path."
            )
        if math.isclose(
            resolved_positive[axis_index],
            resolved_negative[axis_index],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("LumpedPort positive and negative terminals must be distinct along voltage_path.axis.")

        if torch.is_tensor(reference_impedance):
            if reference_impedance.numel() != 1:
                raise ValueError("reference_impedance tensor must contain one value.")
            real = torch.real(reference_impedance)
            imag = torch.imag(reference_impedance) if reference_impedance.is_complex() else torch.zeros_like(real)
            if not bool(torch.isfinite(real)) or not bool(torch.isfinite(imag)):
                raise ValueError("reference_impedance must be finite.")
            if not bool(real > 0.0):
                raise ValueError("reference_impedance real part must be positive.")
            stored_impedance = reference_impedance
        else:
            resolved_impedance = complex(reference_impedance)
            if not math.isfinite(resolved_impedance.real) or not math.isfinite(resolved_impedance.imag):
                raise ValueError("reference_impedance must be finite.")
            if resolved_impedance.real <= 0.0:
                raise ValueError("reference_impedance real part must be positive.")
            stored_impedance = (
                float(resolved_impedance.real)
                if resolved_impedance.imag == 0.0
                else resolved_impedance
            )

        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("LumpedPort name must not be empty.")

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", resolved_positive)
        object.__setattr__(self, "negative", resolved_negative)
        object.__setattr__(self, "voltage_path", voltage_path)
        object.__setattr__(self, "current_surface", current_surface)
        object.__setattr__(self, "reference_impedance", stored_impedance)
        object.__setattr__(
            self,
            "reference_plane",
            None if reference_plane is None else float(reference_plane),
        )
        object.__setattr__(self, "kind", "lumped_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")


@dataclass(frozen=True)
class ModePort:
    """Couple a source and monitor using a polarization-family mode index."""

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    mode_index: int
    direction: str
    polarization: tuple[float, float, float]
    frequencies: tuple[float, ...] | None = None
    source_time: SourceTime | None = None
    monitor_offset: float = 0.0
    normal_direction: str = "+"
    normal_axis: str = "z"
    polarization_axis: str = "x"
    bend_radius: float | None = None
    bend_axis: str | None = None
    kind: str = "mode_port"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        mode_index=0,
        direction="+",
        polarization="auto",
        frequencies=None,
        source_time=None,
        monitor_offset=0.0,
        normal_direction=None,
        bend_radius=None,
        bend_axis=None,
    ):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_mode_source_size(size)
        normal_axis = _resolve_mode_source_normal_axis(resolved_size)
        polarization_axis = _resolve_mode_source_polarization_axis(normal_axis, resolved_size, polarization)
        resolved_bend_radius, resolved_bend_axis = _normalize_bend(bend_radius, bend_axis, normal_axis)
        resolved_direction = _normalize_mode_direction(direction)
        resolved_normal_direction = (
            resolved_direction if normal_direction is None else _normalize_normal_direction(normal_direction)
        )
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "mode_index", int(mode_index))
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "polarization", polarization_vector(f"E{polarization_axis}"))
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "monitor_offset", float(monitor_offset))
        object.__setattr__(self, "normal_direction", resolved_normal_direction)
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "polarization_axis", polarization_axis)
        object.__setattr__(self, "bend_radius", resolved_bend_radius)
        object.__setattr__(self, "bend_axis", resolved_bend_axis)
        object.__setattr__(self, "kind", "mode_port")
        if self.mode_index < 0:
            raise ValueError("mode_index must be >= 0.")

    @property
    def source_name(self) -> str:
        return f"{self.name}::source"

    def to_mode_source(self) -> ModeSource | None:
        if self.source_time is None:
            return None
        return ModeSource(
            position=self.position,
            size=self.size,
            mode_index=self.mode_index,
            direction=self.direction,
            polarization=f"E{self.polarization_axis}",
            source_time=self.source_time,
            name=self.source_name,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
        )

    def to_mode_monitor(self) -> ModeMonitor:
        position = list(self.position)
        axis_index = "xyz".index(self.normal_axis)
        position[axis_index] += float(self.monitor_offset)
        return ModeMonitor(
            name=self.name,
            position=tuple(position),
            size=self.size,
            mode_index=self.mode_index,
            direction=self.direction,
            polarization=f"E{self.polarization_axis}",
            frequencies=self.frequencies,
            normal_direction=self.normal_direction,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
        )
