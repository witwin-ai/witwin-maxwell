from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .sources import (
    _normalize_bend,
    _normalize_mode_direction,
    _require_length3,
    _require_mode_source_size,
    _resolve_mode_source_normal_axis,
    _resolve_mode_source_polarization_axis,
    polarization_vector,
)

_AXES = ("x", "y", "z")


def _normalize_fields(fields):
    if isinstance(fields, str):
        return (fields,)
    if not fields:
        raise ValueError("fields must not be empty.")
    return tuple(str(field) for field in fields)


def normalize_component(component):
    component_name = str(component).lower()
    if component_name not in {"ex", "ey", "ez", "hx", "hy", "hz"}:
        raise ValueError("field component must be one of: Ex, Ey, Ez, Hx, Hy, Hz.")
    return component_name


def _normalize_frequencies(frequencies):
    if frequencies is None:
        return None
    if isinstance(frequencies, Iterable) and not isinstance(frequencies, (str, bytes)):
        values = tuple(float(freq) for freq in frequencies)
    else:
        values = (float(frequencies),)
    if not values:
        raise ValueError("frequencies must not be empty.")
    return values


def normalize_axis(axis):
    axis_name = str(axis).lower()
    if axis_name not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    return axis_name


def _normalize_normal_direction(normal_direction):
    direction = str(normal_direction)
    if direction not in {"+", "-"}:
        raise ValueError("normal_direction must be '+' or '-'.")
    return direction


def required_flux_fields(axis):
    axis_name = normalize_axis(axis)
    if axis_name == "x":
        return ("Ey", "Ez", "Hy", "Hz")
    if axis_name == "y":
        return ("Ex", "Ez", "Hx", "Hz")
    return ("Ex", "Ey", "Hx", "Hy")


def _validate_flux_fields(axis: str, fields: tuple[str, ...], *, monitor_name: str) -> None:
    required_fields = required_flux_fields(axis)
    missing_fields = tuple(field for field in required_fields if field not in fields)
    if missing_fields:
        raise ValueError(
            f"{monitor_name} compute_flux=True requires tangential fields {required_fields}, "
            f"missing {missing_fields}."
        )


def _require_nonnegative_length3(name: str, values) -> tuple[float, float, float]:
    resolved = _require_length3(name, values)
    if any(value < 0.0 for value in resolved):
        raise ValueError(f"{name} must contain non-negative lengths.")
    return resolved


def _resolve_finite_plane_geometry(
    position,
    size,
) -> tuple[tuple[float, float, float], tuple[float, float, float], str, float, dict[str, tuple[float, float]]]:
    resolved_position = _require_length3("position", position)
    resolved_size = _require_nonnegative_length3("size", size)
    zero_axes = [index for index, length in enumerate(resolved_size) if abs(length) <= 1e-12]
    if len(zero_axes) != 1:
        raise ValueError("FinitePlaneMonitor size must have exactly one zero-thickness axis.")

    axis_index = zero_axes[0]
    axis = _AXES[axis_index]
    plane_position = float(resolved_position[axis_index])
    tangential_bounds: dict[str, tuple[float, float]] = {}
    for index, axis_name in enumerate(_AXES):
        if index == axis_index:
            continue
        half_extent = 0.5 * float(resolved_size[index])
        tangential_bounds[axis_name] = (
            float(resolved_position[index] - half_extent),
            float(resolved_position[index] + half_extent),
        )
    return resolved_position, resolved_size, axis, plane_position, tangential_bounds


def _face_area(face: "FinitePlaneMonitor") -> float:
    area = 1.0
    axis_index = _AXES.index(face.axis)
    for index, extent in enumerate(face.size):
        if index == axis_index:
            continue
        area *= float(extent)
    return area


def _validate_closed_surface_area_balance(faces: tuple["FinitePlaneMonitor", ...]) -> None:
    for axis in _AXES:
        positive_area = 0.0
        negative_area = 0.0
        for face in faces:
            if face.axis != axis:
                continue
            if face.normal_direction == "+":
                positive_area += _face_area(face)
            else:
                negative_area += _face_area(face)
        if positive_area <= 0.0 or negative_area <= 0.0:
            raise ValueError(
                f"ClosedSurfaceMonitor requires outward faces on both +/-{axis} directions."
            )
        if abs(positive_area - negative_area) > 1e-9 * max(positive_area, negative_area, 1.0):
            raise ValueError(
                f"ClosedSurfaceMonitor projected area is unbalanced on {axis}-normal faces: "
                f"+={positive_area}, -={negative_area}."
            )


@dataclass(frozen=True)
class PointMonitor:
    name: str
    position: tuple[float, float, float]
    fields: tuple[str, ...]
    kind: str = "point"

    def __init__(self, name, position, fields=("Ez",)):
        if len(position) != 3:
            raise ValueError("position must contain exactly three values.")
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", tuple(float(v) for v in position))
        object.__setattr__(self, "fields", _normalize_fields(fields))
        object.__setattr__(self, "kind", "point")


@dataclass(frozen=True)
class PlaneMonitor:
    name: str
    axis: str
    position: float
    fields: tuple[str, ...]
    frequencies: tuple[float, ...] | None = None
    compute_flux: bool = False
    normal_direction: str = "+"
    kind: str = "plane"

    def __init__(
        self,
        name,
        axis="z",
        position=0.0,
        fields=("Ez",),
        frequencies=None,
        compute_flux=False,
        normal_direction="+",
    ):
        axis_name = normalize_axis(axis)
        normalized_fields = _normalize_fields(fields)
        flux_enabled = bool(compute_flux)
        if flux_enabled:
            _validate_flux_fields(axis_name, normalized_fields, monitor_name="PlaneMonitor")
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "axis", axis_name)
        object.__setattr__(self, "position", float(position))
        object.__setattr__(self, "fields", normalized_fields)
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "compute_flux", flux_enabled)
        object.__setattr__(self, "normal_direction", _normalize_normal_direction(normal_direction))
        object.__setattr__(self, "kind", "plane")


class FluxMonitor(PlaneMonitor):
    def __init__(self, name, axis="z", position=0.0, frequencies=None, normal_direction="+"):
        super().__init__(
            name=name,
            axis=axis,
            position=position,
            fields=required_flux_fields(axis),
            frequencies=frequencies,
            compute_flux=True,
            normal_direction=normal_direction,
        )


@dataclass(frozen=True)
class FinitePlaneMonitor:
    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    fields: tuple[str, ...]
    frequencies: tuple[float, ...] | None = None
    compute_flux: bool = False
    normal_direction: str = "+"
    axis: str = "z"
    plane_position: float = 0.0
    tangential_bounds: dict[str, tuple[float, float]] | None = None
    face_label: str | None = None
    surface_name: str | None = None
    kind: str = "finite_plane"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        fields=("Ez",),
        frequencies=None,
        compute_flux=False,
        normal_direction="+",
        *,
        face_label=None,
        surface_name=None,
    ):
        resolved_position, resolved_size, axis, plane_position, tangential_bounds = _resolve_finite_plane_geometry(
            position,
            size,
        )
        normalized_fields = _normalize_fields(fields)
        flux_enabled = bool(compute_flux)
        if flux_enabled:
            _validate_flux_fields(axis, normalized_fields, monitor_name="FinitePlaneMonitor")

        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "fields", normalized_fields)
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "compute_flux", flux_enabled)
        object.__setattr__(self, "normal_direction", _normalize_normal_direction(normal_direction))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane_position", plane_position)
        object.__setattr__(self, "tangential_bounds", tangential_bounds)
        object.__setattr__(self, "face_label", None if face_label is None else str(face_label))
        object.__setattr__(self, "surface_name", None if surface_name is None else str(surface_name))
        object.__setattr__(self, "kind", "finite_plane")


@dataclass(frozen=True)
class ClosedSurfaceMonitor:
    name: str
    faces: tuple[FinitePlaneMonitor, ...]
    frequencies: tuple[float, ...] | None = None
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    kind: str = "closed_surface"

    def __init__(self, name, faces, frequencies=None):
        surface_name = str(name)
        face_defs = tuple(faces)
        if not face_defs:
            raise ValueError("ClosedSurfaceMonitor requires at least one finite face.")

        resolved_frequencies = _normalize_frequencies(frequencies)
        normalized_faces = []
        face_labels = set()
        for face in face_defs:
            if not isinstance(face, FinitePlaneMonitor):
                raise TypeError(
                    "ClosedSurfaceMonitor expects FinitePlaneMonitor faces, "
                    f"got {type(face).__name__}."
                )
            face_label = face.face_label or face.name
            if face_label in face_labels:
                raise ValueError(f"ClosedSurfaceMonitor face labels must be unique, got duplicate {face_label!r}.")
            face_labels.add(face_label)

            face_frequencies = face.frequencies if face.frequencies is not None else resolved_frequencies
            if resolved_frequencies is not None and face_frequencies is not None and face_frequencies != resolved_frequencies:
                raise ValueError("ClosedSurfaceMonitor face frequencies must match the surface frequencies.")

            normalized_faces.append(
                FinitePlaneMonitor(
                    name=f"{surface_name}::{face_label}",
                    position=face.position,
                    size=face.size,
                    fields=face.fields,
                    frequencies=face_frequencies,
                    compute_flux=face.compute_flux,
                    normal_direction=face.normal_direction,
                    face_label=face_label,
                    surface_name=surface_name,
                )
            )

        normalized_faces_tuple = tuple(normalized_faces)
        _validate_closed_surface_area_balance(normalized_faces_tuple)

        if resolved_frequencies is None:
            reference_frequencies = normalized_faces_tuple[0].frequencies
            for face in normalized_faces_tuple[1:]:
                if face.frequencies != reference_frequencies:
                    raise ValueError("ClosedSurfaceMonitor faces must share the same frequencies.")
            resolved_frequencies = reference_frequencies

        object.__setattr__(self, "name", surface_name)
        object.__setattr__(self, "faces", normalized_faces_tuple)
        object.__setattr__(self, "frequencies", resolved_frequencies)
        object.__setattr__(self, "bounds", self._compute_bounds(normalized_faces_tuple))
        object.__setattr__(self, "kind", "closed_surface")

    @staticmethod
    def _compute_bounds(
        faces: tuple[FinitePlaneMonitor, ...],
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        lower = [float("inf"), float("inf"), float("inf")]
        upper = [float("-inf"), float("-inf"), float("-inf")]
        for face in faces:
            for axis_index, axis_name in enumerate(_AXES):
                if axis_name == face.axis:
                    coordinate = float(face.plane_position)
                    lower[axis_index] = min(lower[axis_index], coordinate)
                    upper[axis_index] = max(upper[axis_index], coordinate)
                    continue
                axis_lower, axis_upper = face.tangential_bounds[axis_name]
                lower[axis_index] = min(lower[axis_index], float(axis_lower))
                upper[axis_index] = max(upper[axis_index], float(axis_upper))
        return tuple((lower[index], upper[index]) for index in range(3))

    @property
    def face_monitor_names(self) -> tuple[str, ...]:
        return tuple(face.name for face in self.faces)

    def resolved_monitors(self) -> tuple[FinitePlaneMonitor, ...]:
        return self.faces

    @classmethod
    def box(
        cls,
        name,
        *,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        frequencies=None,
    ) -> "ClosedSurfaceMonitor":
        center = _require_length3("position", position)
        extents = _require_nonnegative_length3("size", size)
        if any(length <= 0.0 for length in extents):
            raise ValueError("ClosedSurfaceMonitor.box size must contain strictly positive extents.")

        half = tuple(0.5 * length for length in extents)
        faces = (
            FinitePlaneMonitor(
                "x_neg",
                position=(center[0] - half[0], center[1], center[2]),
                size=(0.0, extents[1], extents[2]),
                fields=required_flux_fields("x"),
                frequencies=frequencies,
                normal_direction="-",
            ),
            FinitePlaneMonitor(
                "x_pos",
                position=(center[0] + half[0], center[1], center[2]),
                size=(0.0, extents[1], extents[2]),
                fields=required_flux_fields("x"),
                frequencies=frequencies,
                normal_direction="+",
            ),
            FinitePlaneMonitor(
                "y_neg",
                position=(center[0], center[1] - half[1], center[2]),
                size=(extents[0], 0.0, extents[2]),
                fields=required_flux_fields("y"),
                frequencies=frequencies,
                normal_direction="-",
            ),
            FinitePlaneMonitor(
                "y_pos",
                position=(center[0], center[1] + half[1], center[2]),
                size=(extents[0], 0.0, extents[2]),
                fields=required_flux_fields("y"),
                frequencies=frequencies,
                normal_direction="+",
            ),
            FinitePlaneMonitor(
                "z_neg",
                position=(center[0], center[1], center[2] - half[2]),
                size=(extents[0], extents[1], 0.0),
                fields=required_flux_fields("z"),
                frequencies=frequencies,
                normal_direction="-",
            ),
            FinitePlaneMonitor(
                "z_pos",
                position=(center[0], center[1], center[2] + half[2]),
                size=(extents[0], extents[1], 0.0),
                fields=required_flux_fields("z"),
                frequencies=frequencies,
                normal_direction="+",
            ),
        )
        return cls(name, faces, frequencies=frequencies)


def _material_monitor_bounds(
    position: tuple[float, float, float],
    size: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bounds = []
    for center, extent in zip(position, size):
        half = 0.5 * float(extent)
        bounds.append((float(center) - half, float(center) + half))
    return tuple(bounds)


@dataclass(frozen=True)
class PermittivityMonitor:
    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    frequencies: tuple[float, ...] | None
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    kind: str = "permittivity"

    def __init__(self, name, position=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), frequencies=None):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_nonnegative_length3("size", size)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "bounds", _material_monitor_bounds(resolved_position, resolved_size))
        object.__setattr__(self, "kind", "permittivity")


@dataclass(frozen=True)
class MediumMonitor:
    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    frequencies: tuple[float, ...] | None
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    kind: str = "medium"

    def __init__(self, name, position=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), frequencies=None):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_nonnegative_length3("size", size)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "bounds", _material_monitor_bounds(resolved_position, resolved_size))
        object.__setattr__(self, "kind", "medium")


@dataclass(frozen=True)
class DipoleEmissionMonitor:
    """Measure the power a named ``PointDipole`` delivers to the field.

    The co-located electric field is sampled in the frequency domain at the
    dipole cell, and ``P = -(1/2) Re(conj(J) . E)`` is formed from the known
    dipole current spectrum. Normalizing this against the same dipole run in
    vacuum yields the Purcell factor (local density of states). The vacuum
    normalization is preferred over an analytic free-space formula because the
    discrete Yee-grid effective source volume has no reliable closed form.
    """

    name: str
    source_name: str
    frequencies: tuple[float, ...] | None = None
    kind: str = "dipole_emission"

    def __init__(self, name, source_name, frequencies=None):
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "source_name", str(source_name))
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "kind", "dipole_emission")


def _validate_time_sampling(start, stop, interval) -> tuple[int, int | None, int]:
    resolved_start = int(start)
    if resolved_start < 0:
        raise ValueError("start must be >= 0.")
    resolved_interval = int(interval)
    if resolved_interval < 1:
        raise ValueError("interval must be >= 1.")
    if stop is None:
        resolved_stop = None
    else:
        resolved_stop = int(stop)
        if resolved_stop <= resolved_start:
            raise ValueError("stop must be greater than start.")
    return resolved_start, resolved_stop, resolved_interval


@dataclass(frozen=True)
class FieldTimeMonitor:
    name: str
    components: tuple[str, ...]
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    start: int
    stop: int | None
    interval: int
    region_kind: str
    axis: str | None = None
    plane_position: float | None = None
    tangential_bounds: dict[str, tuple[float, float]] | None = None
    kind: str = "field_time"

    def __init__(
        self,
        name,
        components=("Ez",),
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        start=0,
        stop=None,
        interval=1,
    ):
        normalized_components = tuple(normalize_component(component).capitalize() for component in _normalize_fields(components))
        resolved_position = _require_length3("position", position)
        resolved_size = _require_nonnegative_length3("size", size)
        resolved_start, resolved_stop, resolved_interval = _validate_time_sampling(start, stop, interval)
        zero_axes = [index for index, length in enumerate(resolved_size) if abs(length) <= 1e-12]
        axis = None
        plane_position = None
        tangential_bounds = None
        if len(zero_axes) == 3:
            region_kind = "point"
        elif len(zero_axes) == 1:
            region_kind = "plane"
            _, _, axis, plane_position, tangential_bounds = _resolve_finite_plane_geometry(
                resolved_position,
                resolved_size,
            )
        else:
            region_kind = "volume"

        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "components", normalized_components)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "start", resolved_start)
        object.__setattr__(self, "stop", resolved_stop)
        object.__setattr__(self, "interval", resolved_interval)
        object.__setattr__(self, "region_kind", region_kind)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane_position", plane_position)
        object.__setattr__(self, "tangential_bounds", tangential_bounds)
        object.__setattr__(self, "kind", "field_time")


@dataclass(frozen=True)
class FluxTimeMonitor:
    name: str
    axis: str
    position: float
    fields: tuple[str, ...]
    normal_direction: str
    start: int
    stop: int | None
    interval: int
    kind: str = "flux_time"

    def __init__(
        self,
        name,
        axis="z",
        position=0.0,
        start=0,
        stop=None,
        interval=1,
        normal_direction="+",
    ):
        axis_name = normalize_axis(axis)
        resolved_start, resolved_stop, resolved_interval = _validate_time_sampling(start, stop, interval)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "axis", axis_name)
        object.__setattr__(self, "position", float(position))
        object.__setattr__(self, "fields", required_flux_fields(axis_name))
        object.__setattr__(self, "normal_direction", _normalize_normal_direction(normal_direction))
        object.__setattr__(self, "start", resolved_start)
        object.__setattr__(self, "stop", resolved_stop)
        object.__setattr__(self, "interval", resolved_interval)
        object.__setattr__(self, "kind", "flux_time")


@dataclass(frozen=True)
class ModeMonitor:
    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    mode_index: int
    direction: str
    polarization: tuple[float, float, float]
    frequencies: tuple[float, ...] | None = None
    normal_direction: str = "+"
    axis: str = "z"
    plane_position: float = 0.0
    fields: tuple[str, ...] = ()
    compute_flux: bool = True
    normal_axis: str = "z"
    polarization_axis: str = "x"
    bend_radius: float | None = None
    bend_axis: str | None = None
    kind: str = "mode"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        mode_index=0,
        direction="+",
        polarization="auto",
        frequencies=None,
        normal_direction=None,
        bend_radius=None,
        bend_axis=None,
    ):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_mode_source_size(size)
        normal_axis = _resolve_mode_source_normal_axis(resolved_size)
        polarization_axis = _resolve_mode_source_polarization_axis(normal_axis, resolved_size, polarization)
        resolved_bend_radius, resolved_bend_axis = _normalize_bend(bend_radius, bend_axis, normal_axis)
        axis_index = "xyz".index(normal_axis)
        resolved_direction = _normalize_mode_direction(direction)
        resolved_normal_direction = (
            resolved_direction if normal_direction is None else _normalize_normal_direction(normal_direction)
        )
        axis = normal_axis
        fields = required_flux_fields(axis)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "mode_index", int(mode_index))
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "polarization", polarization_vector(f"E{polarization_axis}"))
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "normal_direction", resolved_normal_direction)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane_position", float(resolved_position[axis_index]))
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "compute_flux", True)
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "polarization_axis", polarization_axis)
        object.__setattr__(self, "bend_radius", resolved_bend_radius)
        object.__setattr__(self, "bend_axis", resolved_bend_axis)
        object.__setattr__(self, "kind", "mode")
        if self.mode_index < 0:
            raise ValueError("mode_index must be >= 0.")

    def mode_spec(self) -> dict[str, object]:
        return {
            "name": self.name,
            "position": self.position,
            "size": self.size,
            "mode_index": int(self.mode_index),
            "direction": self.direction,
            "polarization": self.polarization,
            "polarization_axis": self.polarization_axis,
            "normal_axis": self.normal_axis,
            "bend_radius": self.bend_radius,
            "bend_axis": self.bend_axis,
        }


@dataclass(frozen=True)
class DiffractionMonitor:
    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    frequencies: tuple[float, ...] | None
    normal_direction: str
    orders: int | None
    axis: str
    plane_position: float
    tangential_axes: tuple[str, str]
    periods: dict[str, float]
    fields: tuple[str, ...]
    compute_flux: bool = True
    kind: str = "diffraction"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        frequencies=None,
        normal_direction="+",
        orders=None,
    ):
        resolved_position, resolved_size, axis, plane_position, _ = _resolve_finite_plane_geometry(
            position,
            size,
        )
        transverse_axes = tuple(axis_name for axis_name in _AXES if axis_name != axis)
        periods = {
            axis_name: float(resolved_size[_AXES.index(axis_name)]) for axis_name in transverse_axes
        }
        if any(period <= 0.0 for period in periods.values()):
            raise ValueError("DiffractionMonitor requires strictly positive transverse periods.")
        resolved_orders = None if orders is None else int(orders)
        if resolved_orders is not None and resolved_orders < 0:
            raise ValueError("orders must be a non-negative integer or None.")

        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "normal_direction", _normalize_normal_direction(normal_direction))
        object.__setattr__(self, "orders", resolved_orders)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane_position", plane_position)
        object.__setattr__(self, "tangential_axes", transverse_axes)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "fields", required_flux_fields(axis))
        object.__setattr__(self, "compute_flux", True)
        object.__setattr__(self, "kind", "diffraction")

    def diffraction_spec(self) -> dict[str, object]:
        return {
            "name": self.name,
            "position": self.position,
            "size": self.size,
            "normal_axis": self.axis,
            "tangential_axes": self.tangential_axes,
            "periods": dict(self.periods),
            "orders": self.orders,
            "normal_direction": self.normal_direction,
        }
