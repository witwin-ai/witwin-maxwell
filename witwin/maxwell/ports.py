from __future__ import annotations

from dataclasses import dataclass

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
