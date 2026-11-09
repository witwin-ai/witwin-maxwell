from __future__ import annotations

from ._fixed_inputs import require_fixed_input

import math
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
from .thin_wire import normalize_wire_quantities

_AXES = ("x", "y", "z")
POWER_LOSS_CHANNELS = (
    "conduction",
    "electric_dispersion",
    "magnetic_dispersion",
    "nonlinear",
    "circuit",
    "surface",
    "wire",
)


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
        raw_values = tuple(frequencies)
    else:
        raw_values = (frequencies,)
    require_fixed_input(raw_values, name="Monitor frequencies")
    values = tuple(float(freq) for freq in raw_values)
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


def _normalize_closure_planes(closure_planes, tangential_bounds):
    if closure_planes is None:
        return None
    require_fixed_input(closure_planes, name="FinitePlaneMonitor.closure_planes")
    planes = dict(closure_planes)
    if set(planes) != set(tangential_bounds):
        raise ValueError(
            f"FinitePlaneMonitor closure_planes must name the tangential axes {tuple(tangential_bounds)}."
        )
    resolved = {}
    for axis_name in tangential_bounds:
        lower, upper = (float(value) for value in planes[axis_name])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("FinitePlaneMonitor closure_planes must be finite increasing pairs.")
        resolved[axis_name] = (lower, upper)
    return resolved


def _surface_closure_planes(faces) -> list[dict[str, tuple[float, float]]]:
    """Adjacent-face normal coordinates that bound every face of a closed surface.

    Each tangential edge of a face lies on the plane of the face that meets it
    there. That face's normal coordinate, which is snapped to the grid exactly
    as the face itself is, bounds the edge. An edge without a declared
    neighbour (an excluded projection face) keeps its own coordinate, which is
    snapped by the same rule.
    """
    scale = max(
        [1.0]
        + [abs(float(value)) for face in faces for value in face.position]
        + [float(value) for face in faces for value in face.size]
    )
    tolerance = 1e-9 * scale
    planes = []
    for face in faces:
        resolved = {}
        for axis_name, bounds in face.tangential_bounds.items():
            edges = []
            for bound in bounds:
                matches = [
                    float(other.plane_position)
                    for other in faces
                    if other.axis == axis_name and abs(float(other.plane_position) - bound) <= tolerance
                ]
                edges.append(min(matches, key=lambda value: abs(value - bound)) if matches else float(bound))
            resolved[axis_name] = (edges[0], edges[1])
        planes.append(resolved)
    return planes


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
        require_fixed_input(position, name="PointMonitor.position")
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", tuple(float(v) for v in position))
        object.__setattr__(self, "fields", _normalize_fields(fields))
        object.__setattr__(self, "kind", "point")


@dataclass(frozen=True)
class WireMonitor:
    """Sample frequency-domain state from one declared thin wire."""

    name: str
    wire: str
    frequencies: tuple[float, ...]
    quantities: tuple[str, ...]
    kind: str = "wire"

    def __init__(
        self,
        name,
        wire,
        frequencies,
        quantities=("current", "charge", "ohmic_loss"),
    ):
        monitor_name = str(name).strip()
        if not monitor_name:
            raise ValueError("WireMonitor name must not be empty.")
        wire_name = str(wire).strip()
        if not wire_name:
            raise ValueError("WireMonitor wire must not be empty.")
        resolved_frequencies = _normalize_frequencies(frequencies)
        if resolved_frequencies is None or any(
            not math.isfinite(frequency) or frequency <= 0.0
            for frequency in resolved_frequencies
        ):
            raise ValueError(
                "WireMonitor frequencies must be finite and strictly positive."
            )

        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "wire", wire_name)
        object.__setattr__(self, "frequencies", resolved_frequencies)
        object.__setattr__(self, "quantities", normalize_wire_quantities(quantities))
        object.__setattr__(self, "kind", "wire")


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
        require_fixed_input(compute_flux, name="Monitor.compute_flux")
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


# The single versioned spatial-averaging window for incident power density. It is
# an engineering convenience for exposure reporting and does NOT claim compliance
# with any specific standard averaging area or window shape.
INCIDENT_SPATIAL_AVERAGE_VERSION = "spatial-average-v1"


class IncidentPowerDensityMonitor(PlaneMonitor):
    """Time-averaged incident power density ``|S.n|`` on a plane (W/m^2).

    A plane monitor variant that carries the tangential fields required to form
    the normal Poynting component ``S.n = 0.5 * Re((E x conj(H)).n_hat)``. The
    result-domain accessor :meth:`Result.incident_power_density` returns the
    signed per-cell normal Poynting, its magnitude ``|S.n|`` (the exposure
    incident power density), the plane-integrated flux (identically equal to the
    :class:`FluxMonitor` integral on the same plane), and, when
    ``spatial_average`` is set, an area-weighted moving-window average of
    ``|S.n|``.

    ``spatial_average`` is the moving-window **area** in m^2 (for example
    ``4e-4`` for a 4 cm^2-class window). The window is an axis-aligned square of
    side ``sqrt(area)`` centred on each cell; near the plane edge it is truncated
    to the in-domain cells (``edge_policy="truncate"``). The averaging is
    versioned ``spatial-average-v1`` and is NOT a certified standards window.

    The monitor reports whatever field is present at its plane; place it in the
    incident region (ahead of any scatterer) to read the exposure incident power
    density.
    """

    spatial_average: float | None
    spatial_average_version: str | None

    def __init__(
        self,
        name,
        axis="z",
        position=0.0,
        frequencies=None,
        normal_direction="+",
        spatial_average=None,
    ):
        super().__init__(
            name=name,
            axis=axis,
            position=position,
            fields=required_flux_fields(axis),
            frequencies=frequencies,
            compute_flux=True,
            normal_direction=normal_direction,
        )
        resolved_area = None
        version = None
        if spatial_average is not None:
            resolved_area = float(spatial_average)
            if not math.isfinite(resolved_area) or resolved_area <= 0.0:
                raise ValueError(
                    "IncidentPowerDensityMonitor spatial_average must be a positive "
                    "window area in m^2, or None to disable."
                )
            version = INCIDENT_SPATIAL_AVERAGE_VERSION
        object.__setattr__(self, "spatial_average", resolved_area)
        object.__setattr__(self, "spatial_average_version", version)


@dataclass(frozen=True)
class FinitePlaneMonitor:
    """Finite planar observation with optional physical quadrature spacing.

    ``sample_spacing`` sets a maximum rectangular midpoint quadrature interval
    in metres, independently of the field grid. All components are linearly
    interpolated from their native Yee locations to those common points.
    Converge this observation quadrature separately from the field mesh. The
    default retains the full native staggered plane representation.

    A standalone plane integrates exactly over its declared rectangle, including
    partial edge cells. Faces built by ``ClosedSurfaceMonitor`` and by box
    projection surfaces carry ``closure_planes`` instead: the normal coordinates
    of the adjacent faces that bound each tangential axis. Such a face is placed
    on its nearest tangential-E Yee node like a flux plane, and its in-plane
    rectangle is bounded by the same snapped node of each adjacent face, so the
    discrete faces enclose one closed region exactly.
    """
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
    sample_spacing: float | None = None
    closure_planes: dict[str, tuple[float, float]] | None = None
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
        sample_spacing=None,
        closure_planes=None,
    ):
        resolved_position, resolved_size, axis, plane_position, tangential_bounds = _resolve_finite_plane_geometry(
            position,
            size,
        )
        normalized_fields = _normalize_fields(fields)
        require_fixed_input(compute_flux, name="Monitor.compute_flux")
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
        require_fixed_input(sample_spacing, name="FinitePlaneMonitor.sample_spacing")
        spacing = None if sample_spacing is None else float(sample_spacing)
        if spacing is not None and (not math.isfinite(spacing) or spacing <= 0):
            raise ValueError("FinitePlaneMonitor sample_spacing must be finite and positive.")
        object.__setattr__(self, "sample_spacing", spacing)
        object.__setattr__(self, "closure_planes", _normalize_closure_planes(closure_planes, tangential_bounds))
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
        closure_planes = _surface_closure_planes(face_defs)
        for face, face_closure in zip(face_defs, closure_planes):
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
                    sample_spacing=face.sample_spacing,
                    closure_planes=face_closure,
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
        sample_spacing=None,
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
                sample_spacing=sample_spacing,
                normal_direction="-",
            ),
            FinitePlaneMonitor(
                "x_pos",
                position=(center[0] + half[0], center[1], center[2]),
                size=(0.0, extents[1], extents[2]),
                fields=required_flux_fields("x"),
                frequencies=frequencies,
                sample_spacing=sample_spacing,
                normal_direction="+",
            ),
            FinitePlaneMonitor(
                "y_neg",
                position=(center[0], center[1] - half[1], center[2]),
                size=(extents[0], 0.0, extents[2]),
                fields=required_flux_fields("y"),
                frequencies=frequencies,
                sample_spacing=sample_spacing,
                normal_direction="-",
            ),
            FinitePlaneMonitor(
                "y_pos",
                position=(center[0], center[1] + half[1], center[2]),
                size=(extents[0], 0.0, extents[2]),
                fields=required_flux_fields("y"),
                frequencies=frequencies,
                sample_spacing=sample_spacing,
                normal_direction="+",
            ),
            FinitePlaneMonitor(
                "z_neg",
                position=(center[0], center[1], center[2] - half[2]),
                size=(extents[0], extents[1], 0.0),
                fields=required_flux_fields("z"),
                frequencies=frequencies,
                sample_spacing=sample_spacing,
                normal_direction="-",
            ),
            FinitePlaneMonitor(
                "z_pos",
                position=(center[0], center[1], center[2] + half[2]),
                size=(extents[0], extents[1], 0.0),
                fields=required_flux_fields("z"),
                frequencies=frequencies,
                sample_spacing=sample_spacing,
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
class PowerLossMonitor:
    """Select an axis-aligned volume and explicit physical loss channels.

    Automatic FDTD producers support ``conduction`` from static conductivity
    and ``electric_dispersion`` from native passive bulk polarization work.
    Other channels require an explicit physical input during postprocess.

    ``native_cells`` assigns complete native control volumes to selected edge
    centers. ``physical_intersection`` integrates the compiled homogenized
    cell density over the intersection with the requested box, including cells
    whose centers lie outside it. It does not recover subcell material fill;
    clipping a material interface requires joint material/window integration
    for a stronger continuum interpretation.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    frequencies: tuple[float, ...] | None
    channels: tuple[str, ...]
    quadrature: str
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    kind: str = "power_loss"

    def __init__(
        self,
        name,
        position,
        size,
        frequencies=None,
        channels=("conduction",),
        quadrature="native_cells",
    ):
        monitor_name = str(name)
        if not monitor_name:
            raise ValueError("PowerLossMonitor name must not be empty.")
        resolved_position = _require_length3("position", position)
        resolved_size = _require_nonnegative_length3("size", size)
        if not all(math.isfinite(value) for value in (*resolved_position, *resolved_size)):
            raise ValueError("PowerLossMonitor position and size must be finite.")
        if any(value <= 0.0 for value in resolved_size):
            raise ValueError("PowerLossMonitor size must contain three positive lengths.")

        resolved_frequencies = _normalize_frequencies(frequencies)
        if resolved_frequencies is not None and any(
            not math.isfinite(value) or value <= 0.0 for value in resolved_frequencies
        ):
            raise ValueError("PowerLossMonitor frequencies must be finite and strictly positive.")

        if isinstance(channels, str):
            resolved_channels = (channels,)
        else:
            resolved_channels = tuple(str(channel) for channel in channels)
        if not resolved_channels:
            raise ValueError("PowerLossMonitor channels must not be empty.")
        if len(set(resolved_channels)) != len(resolved_channels):
            raise ValueError("PowerLossMonitor channels must be unique.")
        unsupported = tuple(
            channel for channel in resolved_channels if channel not in POWER_LOSS_CHANNELS
        )
        if unsupported:
            raise ValueError(
                f"Unsupported power-loss channels {unsupported}; choices are {POWER_LOSS_CHANNELS}."
            )
        if quadrature not in ("native_cells", "physical_intersection"):
            raise ValueError("PowerLossMonitor quadrature must be 'native_cells' or 'physical_intersection'.")

        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "frequencies", resolved_frequencies)
        object.__setattr__(self, "channels", resolved_channels)
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(
            self,
            "bounds",
            _material_monitor_bounds(resolved_position, resolved_size),
        )
        object.__setattr__(self, "kind", "power_loss")


BREAKDOWN_QUANTITIES = ("electric_field", "exposure", "dissipated_energy", "damage")


@dataclass(frozen=True)
class BreakdownMonitor:
    """Non-feedback dielectric-stress accumulator over an axis-aligned region.

    Capability level: **stress-only**. During a standard FDTD run this records,
    per cell and entirely on device, the running ``max|E|``, the exceedance time
    ``integral H(|E| - critical_field) dt``, the longest contiguous exceedance
    duration and (when ``damage_exponent`` is set) a damage integral
    ``integral (|E|/critical_field)^k dt``. It performs NO feedback into the
    field solve and predicts nothing about device failure; a recorded exceedance
    is a stress indicator only. ``minimum_duration`` is the reporting threshold
    for a qualifying sustained exceedance and does not alter accumulation. Only
    target-material occupancy of partially filled cells counts.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    quantities: tuple[str, ...]
    critical_field: float
    minimum_duration: float
    damage_exponent: float | None
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    kind: str = "breakdown"

    def __init__(
        self,
        name,
        region=None,
        *,
        position=None,
        size=None,
        quantities=("electric_field", "exposure", "dissipated_energy"),
        critical_field=None,
        minimum_duration=0.0,
        damage_exponent=None,
    ):
        monitor_name = str(name).strip()
        if not monitor_name:
            raise ValueError("BreakdownMonitor name must not be empty.")
        resolved_position, resolved_size = _resolve_region_geometry(region, position, size)
        if not all(math.isfinite(value) for value in (*resolved_position, *resolved_size)):
            raise ValueError("BreakdownMonitor position and size must be finite.")
        if any(value <= 0.0 for value in resolved_size):
            raise ValueError("BreakdownMonitor size must contain three positive lengths.")
        if critical_field is None:
            raise ValueError("BreakdownMonitor requires an explicit critical_field.")
        resolved_field = float(critical_field)
        if not math.isfinite(resolved_field) or resolved_field <= 0.0:
            raise ValueError("critical_field must be a positive finite field strength.")
        resolved_duration = float(minimum_duration)
        if not math.isfinite(resolved_duration) or resolved_duration < 0.0:
            raise ValueError("minimum_duration must be a non-negative finite time.")
        resolved_exponent = None if damage_exponent is None else float(damage_exponent)
        if resolved_exponent is not None and (
            not math.isfinite(resolved_exponent) or resolved_exponent <= 0.0
        ):
            raise ValueError("damage_exponent must be a positive finite exponent or None.")

        if isinstance(quantities, str):
            resolved_quantities = (quantities,)
        else:
            resolved_quantities = tuple(str(quantity) for quantity in quantities)
        if not resolved_quantities:
            raise ValueError("BreakdownMonitor quantities must not be empty.")
        if len(set(resolved_quantities)) != len(resolved_quantities):
            raise ValueError("BreakdownMonitor quantities must be unique.")
        unsupported = tuple(q for q in resolved_quantities if q not in BREAKDOWN_QUANTITIES)
        if unsupported:
            raise ValueError(
                f"Unsupported BreakdownMonitor quantities {unsupported}; choices are {BREAKDOWN_QUANTITIES}."
            )
        if "damage" in resolved_quantities and resolved_exponent is None:
            raise ValueError("BreakdownMonitor 'damage' quantity requires damage_exponent.")

        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "quantities", resolved_quantities)
        object.__setattr__(self, "critical_field", resolved_field)
        object.__setattr__(self, "minimum_duration", resolved_duration)
        object.__setattr__(self, "damage_exponent", resolved_exponent)
        object.__setattr__(
            self,
            "bounds",
            _material_monitor_bounds(resolved_position, resolved_size),
        )
        object.__setattr__(self, "kind", "breakdown")


@dataclass(frozen=True)
class ComponentStressMonitor:
    """Bind a :class:`ComponentRating` to a port's recorded V(t)/I(t) series.

    Capability level: **stress-only**. ``voltage_series`` and ``current_series``
    name time-series monitors present in the run result (for example a
    ``FieldTimeMonitor`` sampling the gap voltage and a current time series);
    the result reduces them to ``P = V I``, cumulative ``integral P dt`` and an
    exceedance summary versus the rating envelope. It flags stress against the
    declared absolute-maximum envelope, never device failure.
    """

    name: str
    port: str
    rating: object
    voltage_series: str
    current_series: str
    kind: str = "component_stress"

    def __init__(self, name, *, port, rating, voltage_series, current_series):
        from .breakdown_stress import ComponentRating

        monitor_name = str(name).strip()
        if not monitor_name:
            raise ValueError("ComponentStressMonitor name must not be empty.")
        port_name = str(port).strip()
        if not port_name:
            raise ValueError("ComponentStressMonitor port must not be empty.")
        if not isinstance(rating, ComponentRating):
            raise TypeError("rating must be a ComponentRating instance.")
        voltage_name = str(voltage_series).strip()
        current_name = str(current_series).strip()
        if not voltage_name or not current_name:
            raise ValueError("voltage_series and current_series must name time-series monitors.")
        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "port", port_name)
        object.__setattr__(self, "rating", rating)
        object.__setattr__(self, "voltage_series", voltage_name)
        object.__setattr__(self, "current_series", current_name)
        object.__setattr__(self, "kind", "component_stress")


def _resolve_region_geometry(region, position, size):
    """Resolve a monitor region from an explicit box or a bounded scene object."""

    if region is not None:
        if position is not None or size is not None:
            raise ValueError("Provide either region or position/size, not both.")
        bounds = getattr(region, "bounds", None)
        if bounds is not None and len(bounds) == 3:
            resolved_position = tuple(
                0.5 * (float(lower) + float(upper)) for lower, upper in bounds
            )
            resolved_size = tuple(float(upper) - float(lower) for lower, upper in bounds)
            return resolved_position, resolved_size
        region_position = getattr(region, "position", None)
        region_size = getattr(region, "size", None)
        if region_position is not None and region_size is not None:
            return (
                _require_length3("region.position", region_position),
                _require_nonnegative_length3("region.size", region_size),
            )
        raise TypeError("region must expose bounds or position/size.")
    if position is None or size is None:
        raise ValueError("Provide a region or an explicit position and size.")
    return (
        _require_length3("position", position),
        _require_nonnegative_length3("size", size),
    )


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
    """Record detached raw Yee fields; use spectral monitors for adjoint objectives."""
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
    """Record detached native-cell flux with the declared Yee time staggering.

    Paired native products use E at trace time t and H at t - dt/2 before
    tangential projection and integration with physical cell widths. Exported
    metadata records these offsets; no common-clock time interpolation is
    applied. Traces remain detached in differentiable simulations and do not
    introduce time-monitor gradients.
    """
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
    """Decompose a guided mode indexed within the requested polarization family."""

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
        spec = {
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
        wave_family = getattr(self, "_wave_family", None)
        if wave_family is not None:
            spec["wave_family"] = wave_family
        conductor_model=getattr(self,"_conductor_model",None)
        if conductor_model is not None:
            spec["conductor_model"]=conductor_model
        terminal_mode = getattr(self, "_terminal_mode", None)
        if terminal_mode is not None:
            spec["terminal_mode"] = terminal_mode
        return spec


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


# ---------------------------------------------------------------------------
# In-simulation field projection, Gaussian-overlap, surface and auxiliary time
# monitors.
# ---------------------------------------------------------------------------

SURFACE_FACE_LABELS = ("x_neg", "x_pos", "y_neg", "y_pos", "z_neg", "z_pos")


def _normalize_float_grid(name: str, values) -> tuple[float, ...]:
    """Flatten a fixed 1-D sample grid (scalar, sequence, array or tensor)."""

    require_fixed_input(values, name=name)
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = (values,)
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{name} must not be empty.")
    if not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} must contain finite values.")
    return resolved


def _normalize_surface_exclusions(exclude_surfaces) -> tuple[str, ...]:
    if exclude_surfaces is None:
        return ()
    if isinstance(exclude_surfaces, str):
        exclude_surfaces = (exclude_surfaces,)
    labels = tuple(str(label) for label in exclude_surfaces)
    unknown = tuple(label for label in labels if label not in SURFACE_FACE_LABELS)
    if unknown:
        raise ValueError(
            f"exclude_surfaces entries must be face labels {SURFACE_FACE_LABELS}, got {unknown}."
        )
    if len(set(labels)) != len(labels):
        raise ValueError("exclude_surfaces must not repeat a face label.")
    return labels


def _resolve_declared_surface(
    monitor_name: str,
    position,
    size,
    *,
    owner: str,
    normal_direction=None,
    exclude_surfaces=None,
):
    """Resolve a declared plane patch or (open) box into axis-aligned face boxes.

    Returns ``(surface_kind, center, extents, normal_direction, exclusions,
    faces)`` where every face is ``(label, axis, normal_direction,
    face_position, face_size)``. A plane patch (exactly one zero extent) yields
    one ``"plane"`` face oriented by ``normal_direction`` (default ``"+"``). A
    box (three positive extents) yields its outward faces minus
    ``exclude_surfaces``.
    """

    center = _require_length3("position", position)
    extents = _require_nonnegative_length3("size", size)
    if not all(math.isfinite(value) for value in (*center, *extents)):
        raise ValueError(f"{owner} position and size must be finite.")
    zero_axes = [index for index, length in enumerate(extents) if abs(length) <= 1e-12]
    exclusions = _normalize_surface_exclusions(exclude_surfaces)
    if len(zero_axes) == 1:
        if exclusions:
            raise ValueError(f"{owner} exclude_surfaces applies to box surfaces only.")
        direction = "+" if normal_direction is None else _normalize_normal_direction(normal_direction)
        axis = _AXES[zero_axes[0]]
        return "planar", center, extents, direction, (), (("plane", axis, direction, center, extents),)
    if zero_axes:
        raise ValueError(
            f"{owner} size must describe a plane patch (one zero extent) or a box (three positive extents)."
        )
    if normal_direction is not None:
        raise ValueError(f"{owner} normal_direction applies to planar surfaces only; box faces point outward.")
    faces = []
    for axis_index, axis in enumerate(_AXES):
        for sign, suffix in ((-1.0, "neg"), (1.0, "pos")):
            label = f"{axis}_{suffix}"
            if label in exclusions:
                continue
            face_position = list(center)
            face_position[axis_index] = center[axis_index] + sign * 0.5 * extents[axis_index]
            face_size = list(extents)
            face_size[axis_index] = 0.0
            faces.append((label, axis, "+" if sign > 0 else "-", tuple(face_position), tuple(face_size)))
    if not faces:
        raise ValueError(f"{owner} must keep at least one face after exclude_surfaces.")
    return "box", center, extents, None, exclusions, tuple(faces)


class _ProjectionSurfaceMonitor:
    """Shared Huygens-surface behaviour of the projection monitors.

    The monitor records the tangential running-DFT fields on its faces as
    ordinary finite-plane observations (``<name>::<face label>``) and projects
    them when ``Result.monitor(name)`` is requested.
    """

    def _init_surface(
        self,
        name,
        position,
        size,
        frequencies,
        origin,
        normal_direction,
        exclude_surfaces,
        window_size,
        sample_spacing,
        kind,
    ):
        monitor_name = str(name).strip()
        if not monitor_name:
            raise ValueError(f"{type(self).__name__} name must not be empty.")
        owner = type(self).__name__
        surface_kind, center, extents, direction, exclusions, face_specs = _resolve_declared_surface(
            monitor_name,
            position,
            size,
            owner=owner,
            normal_direction=normal_direction,
            exclude_surfaces=exclude_surfaces,
        )
        resolved_frequencies = _normalize_frequencies(frequencies)
        if resolved_frequencies is not None and any(
            not math.isfinite(value) or value <= 0.0 for value in resolved_frequencies
        ):
            raise ValueError(f"{owner} frequencies must be finite and strictly positive.")
        resolved_origin = center if origin is None else _require_length3("origin", origin)
        resolved_window = None
        if window_size is not None:
            if surface_kind != "planar":
                raise ValueError(f"{owner} window_size applies to planar surfaces only.")
            require_fixed_input(window_size, name=f"{owner}.window_size")
            if len(window_size) != 2:
                raise ValueError("window_size must contain two fractions.")
            resolved_window = tuple(float(value) for value in window_size)
            if any(not 0.0 <= value <= 1.0 for value in resolved_window):
                raise ValueError("window_size fractions must lie in [0, 1].")
        faces = tuple(
            FinitePlaneMonitor(
                f"{monitor_name}::{label}",
                position=face_position,
                size=face_size,
                fields=required_flux_fields(axis),
                frequencies=resolved_frequencies,
                normal_direction=face_direction,
                face_label=label,
                surface_name=monitor_name,
                sample_spacing=sample_spacing,
            )
            for label, axis, face_direction, face_position, face_size in face_specs
        )
        if surface_kind == "box":
            # Huygens boxes, open ones included, are watertight on the grid.
            faces = tuple(
                FinitePlaneMonitor(
                    face.name,
                    position=face.position,
                    size=face.size,
                    fields=face.fields,
                    frequencies=face.frequencies,
                    normal_direction=face.normal_direction,
                    face_label=face.face_label,
                    surface_name=face.surface_name,
                    sample_spacing=face.sample_spacing,
                    closure_planes=planes,
                )
                for face, planes in zip(faces, _surface_closure_planes(faces))
            )
        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "position", center)
        object.__setattr__(self, "size", extents)
        object.__setattr__(self, "frequencies", resolved_frequencies)
        object.__setattr__(self, "origin", resolved_origin)
        object.__setattr__(self, "normal_direction", direction)
        object.__setattr__(self, "exclude_surfaces", exclusions)
        object.__setattr__(self, "window_size", resolved_window)
        object.__setattr__(self, "sample_spacing", faces[0].sample_spacing)
        object.__setattr__(self, "surface_kind", surface_kind)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "bounds", ClosedSurfaceMonitor._compute_bounds(faces))
        object.__setattr__(self, "kind", kind)

    @property
    def face_monitor_names(self) -> tuple[str, ...]:
        return tuple(face.name for face in self.faces)

    def resolved_monitors(self) -> tuple[FinitePlaneMonitor, ...]:
        return self.faces


@dataclass(frozen=True)
class AngleProjectionMonitor(_ProjectionSurfaceMonitor):
    """Project the fields on a Huygens surface onto a ``(theta, phi)`` grid.

    ``position``/``size`` declare either a plane patch (one zero extent,
    oriented by ``normal_direction``) or a box whose outward faces form the
    surface; ``exclude_surfaces`` drops box faces by label (``"x_neg"`` ...
    ``"z_pos"``), e.g. a face lying on a substrate. ``Result.monitor(name)``
    returns the :class:`~witwin.maxwell.postprocess.ProjectedFieldData` that
    ``FieldProjector.angles(theta, phi, radius=radius,
    far_field_approx=far_field_approx)`` returns for the same surface data,
    with the projection origin at ``origin`` (default: the surface center).
    The exterior medium immediately outside every face must be homogeneous.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    theta: tuple[float, ...]
    phi: tuple[float, ...]
    frequencies: tuple[float, ...] | None
    radius: float
    far_field_approx: bool
    origin: tuple[float, float, float]
    normal_direction: str | None
    exclude_surfaces: tuple[str, ...]
    window_size: tuple[float, float] | None
    sample_spacing: float | None
    surface_kind: str
    faces: tuple[FinitePlaneMonitor, ...]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    kind: str = "angle_projection"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        *,
        theta,
        phi,
        frequencies=None,
        radius=1.0,
        far_field_approx=True,
        origin=None,
        normal_direction=None,
        exclude_surfaces=None,
        window_size=None,
        sample_spacing=None,
    ):
        require_fixed_input(radius, name="AngleProjectionMonitor.radius")
        resolved_radius = float(radius)
        if not math.isfinite(resolved_radius) or resolved_radius <= 0.0:
            raise ValueError("AngleProjectionMonitor radius must be finite and positive.")
        object.__setattr__(self, "theta", _normalize_float_grid("theta", theta))
        object.__setattr__(self, "phi", _normalize_float_grid("phi", phi))
        object.__setattr__(self, "radius", resolved_radius)
        object.__setattr__(self, "far_field_approx", bool(far_field_approx))
        self._init_surface(
            name, position, size, frequencies, origin, normal_direction,
            exclude_surfaces, window_size, sample_spacing, "angle_projection",
        )


@dataclass(frozen=True)
class CartesianProjectionMonitor(_ProjectionSurfaceMonitor):
    """Project the fields on a Huygens surface onto a Cartesian observation plane.

    The plane is normal to ``axis`` at signed ``distance`` from ``origin``
    (default: the surface center); ``x``/``y`` are local in-plane coordinates
    along the two remaining world axes in ascending order, exactly as in
    ``FieldProjector.cartesian``. ``far_field_approx=False`` (the default)
    evaluates the exact Stratton-Chu field. Surface declaration and Result
    access follow :class:`AngleProjectionMonitor`.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    x: tuple[float, ...]
    y: tuple[float, ...]
    axis: str
    distance: float
    frequencies: tuple[float, ...] | None
    far_field_approx: bool
    origin: tuple[float, float, float]
    normal_direction: str | None
    exclude_surfaces: tuple[str, ...]
    window_size: tuple[float, float] | None
    sample_spacing: float | None
    surface_kind: str
    faces: tuple[FinitePlaneMonitor, ...]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    kind: str = "cartesian_projection"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        *,
        x,
        y,
        distance,
        axis="z",
        frequencies=None,
        far_field_approx=False,
        origin=None,
        normal_direction=None,
        exclude_surfaces=None,
        window_size=None,
        sample_spacing=None,
    ):
        require_fixed_input(distance, name="CartesianProjectionMonitor.distance")
        resolved_distance = float(distance)
        if not math.isfinite(resolved_distance):
            raise ValueError("CartesianProjectionMonitor distance must be finite.")
        object.__setattr__(self, "x", _normalize_float_grid("x", x))
        object.__setattr__(self, "y", _normalize_float_grid("y", y))
        object.__setattr__(self, "axis", normalize_axis(axis))
        object.__setattr__(self, "distance", resolved_distance)
        object.__setattr__(self, "far_field_approx", bool(far_field_approx))
        self._init_surface(
            name, position, size, frequencies, origin, normal_direction,
            exclude_surfaces, window_size, sample_spacing, "cartesian_projection",
        )


@dataclass(frozen=True)
class KSpaceProjectionMonitor(_ProjectionSurfaceMonitor):
    """Project the fields on a Huygens surface onto direction cosines ``(ux, uy)``.

    Directions are ``ux u_hat + uy v_hat + s sqrt(1 - ux^2 - uy^2) n_hat`` about
    ``axis`` with the in-plane axes ordered as in ``FieldProjector.kspace``;
    ``hemisphere`` selects the sign ``s`` (default: the orientation of a planar
    surface normal to ``axis``, else ``"+"``). Evanescent samples are masked and
    zero. The far-field approximation is always used. Surface declaration and
    Result access follow :class:`AngleProjectionMonitor`.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    ux: tuple[float, ...]
    uy: tuple[float, ...]
    axis: str
    hemisphere: str
    radius: float
    frequencies: tuple[float, ...] | None
    origin: tuple[float, float, float]
    normal_direction: str | None
    exclude_surfaces: tuple[str, ...]
    window_size: tuple[float, float] | None
    sample_spacing: float | None
    surface_kind: str
    faces: tuple[FinitePlaneMonitor, ...]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    far_field_approx: bool = True
    kind: str = "kspace_projection"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        *,
        ux,
        uy,
        axis="z",
        hemisphere=None,
        radius=1.0,
        frequencies=None,
        origin=None,
        normal_direction=None,
        exclude_surfaces=None,
        window_size=None,
        sample_spacing=None,
    ):
        require_fixed_input(radius, name="KSpaceProjectionMonitor.radius")
        resolved_radius = float(radius)
        if not math.isfinite(resolved_radius) or resolved_radius <= 0.0:
            raise ValueError("KSpaceProjectionMonitor radius must be finite and positive.")
        axis_name = normalize_axis(axis)
        object.__setattr__(self, "ux", _normalize_float_grid("ux", ux))
        object.__setattr__(self, "uy", _normalize_float_grid("uy", uy))
        object.__setattr__(self, "axis", axis_name)
        object.__setattr__(self, "radius", resolved_radius)
        object.__setattr__(self, "far_field_approx", True)
        self._init_surface(
            name, position, size, frequencies, origin, normal_direction,
            exclude_surfaces, window_size, sample_spacing, "kspace_projection",
        )
        if hemisphere is None:
            planar_face = self.faces[0] if self.surface_kind == "planar" else None
            hemisphere = (
                planar_face.normal_direction
                if planar_face is not None and planar_face.axis == axis_name
                else "+"
            )
        object.__setattr__(self, "hemisphere", _normalize_normal_direction(hemisphere))


PROJECTION_MONITOR_TYPES = (AngleProjectionMonitor, CartesianProjectionMonitor, KSpaceProjectionMonitor)


class _GaussianOverlapPlaneMonitor:
    """Shared plane handling of the Gaussian-overlap monitors.

    The monitor records its plane as one finite-plane observation carrying the
    monitor's own name and decomposes it when ``Result.monitor(name)`` is
    requested.
    """

    def _init_plane(self, name, position, size, frequencies, direction, polarization, focus, kind):
        monitor_name = str(name).strip()
        owner = type(self).__name__
        if not monitor_name:
            raise ValueError(f"{owner} name must not be empty.")
        resolved_position, resolved_size, axis, plane_position, tangential_bounds = _resolve_finite_plane_geometry(
            position,
            size,
        )
        axis_index = _AXES.index(axis)
        if direction is None:
            direction = tuple(1.0 if index == axis_index else 0.0 for index in range(3))
        from .sources import _validate_transverse_polarization

        resolved_direction, resolved_polarization = _validate_transverse_polarization(direction, polarization)
        normal_component = resolved_direction[axis_index]
        if abs(normal_component) <= 1e-6:
            raise ValueError(f"{owner} direction must cross the monitor plane.")
        resolved_frequencies = _normalize_frequencies(frequencies)
        if resolved_frequencies is not None and any(
            not math.isfinite(value) or value <= 0.0 for value in resolved_frequencies
        ):
            raise ValueError(f"{owner} frequencies must be finite and strictly positive.")
        resolved_focus = resolved_position if focus is None else _require_length3("focus", focus)
        normal_direction = "+" if normal_component > 0.0 else "-"
        plane = FinitePlaneMonitor(
            monitor_name,
            position=resolved_position,
            size=resolved_size,
            fields=required_flux_fields(axis),
            frequencies=resolved_frequencies,
            normal_direction=normal_direction,
            face_label="plane",
            surface_name=monitor_name,
        )
        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "frequencies", resolved_frequencies)
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "polarization", resolved_polarization)
        object.__setattr__(self, "focus", resolved_focus)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane_position", plane_position)
        object.__setattr__(self, "normal_direction", normal_direction)
        object.__setattr__(self, "tangential_bounds", tangential_bounds)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "kind", kind)

    def resolved_monitors(self) -> tuple[FinitePlaneMonitor, ...]:
        return (self.plane,)


@dataclass(frozen=True)
class GaussianOverlapMonitor(_GaussianOverlapPlaneMonitor):
    """Decompose the fields on a plane into a Gaussian beam profile, per frequency.

    The beam is declared with the same vocabulary as :class:`GaussianBeam`:
    propagation ``direction`` (default: the plane normal), transverse
    ``polarization``, ``beam_waist`` radius and the waist center ``focus``
    (default: the monitor center). ``Result.monitor(name)`` returns, per
    frequency, the forward amplitude (along ``direction``) and the backward
    amplitude of the tangentially mirrored profile ``(E_t, -H_t)``, in
    ``sqrt(W)`` with the reference profile normalized to unit power, together
    with the matching powers and their fractions of the net plane flux. The
    paraxial profile is evaluated in the medium at the monitor center.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    frequencies: tuple[float, ...] | None
    direction: tuple[float, float, float]
    polarization: tuple[float, float, float]
    beam_waist: float
    focus: tuple[float, float, float]
    axis: str
    plane_position: float
    normal_direction: str
    tangential_bounds: dict[str, tuple[float, float]]
    plane: FinitePlaneMonitor
    kind: str = "gaussian_overlap"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        *,
        beam_waist,
        frequencies=None,
        direction=None,
        polarization=(1.0, 0.0, 0.0),
        focus=None,
    ):
        require_fixed_input(beam_waist, name="GaussianOverlapMonitor.beam_waist")
        waist = float(beam_waist)
        if not math.isfinite(waist) or waist <= 0.0:
            raise ValueError("beam_waist must be > 0.")
        object.__setattr__(self, "beam_waist", waist)
        self._init_plane(name, position, size, frequencies, direction, polarization, focus, "gaussian_overlap")

    @property
    def beam_waists(self) -> tuple[float, float]:
        return (self.beam_waist, self.beam_waist)

    @property
    def focus_offsets(self) -> tuple[float, float]:
        return (0.0, 0.0)


@dataclass(frozen=True)
class AstigmaticGaussianOverlapMonitor(_GaussianOverlapPlaneMonitor):
    """Astigmatic variant of :class:`GaussianOverlapMonitor`.

    Mirrors :class:`AstigmaticGaussianBeam`: ``beam_waist=(w_u, w_v)`` along
    the polarization ``u`` and the binormal ``v = direction x polarization``,
    with per-axis waist planes displaced by ``focus_u``/``focus_v`` along the
    propagation direction from ``focus``.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    frequencies: tuple[float, ...] | None
    direction: tuple[float, float, float]
    polarization: tuple[float, float, float]
    beam_waist: tuple[float, float]
    focus: tuple[float, float, float]
    focus_u: float
    focus_v: float
    axis: str
    plane_position: float
    normal_direction: str
    tangential_bounds: dict[str, tuple[float, float]]
    plane: FinitePlaneMonitor
    kind: str = "astigmatic_gaussian_overlap"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        *,
        beam_waist,
        frequencies=None,
        direction=None,
        polarization=(1.0, 0.0, 0.0),
        focus=None,
        focus_u=0.0,
        focus_v=0.0,
    ):
        from .sources import _require_beam_waist_pair

        require_fixed_input(beam_waist, name="AstigmaticGaussianOverlapMonitor.beam_waist")
        require_fixed_input((focus_u, focus_v), name="AstigmaticGaussianOverlapMonitor.focus offsets")
        object.__setattr__(self, "beam_waist", _require_beam_waist_pair(beam_waist))
        object.__setattr__(self, "focus_u", float(focus_u))
        object.__setattr__(self, "focus_v", float(focus_v))
        self._init_plane(
            name, position, size, frequencies, direction, polarization, focus, "astigmatic_gaussian_overlap",
        )

    @property
    def beam_waists(self) -> tuple[float, float]:
        return self.beam_waist

    @property
    def focus_offsets(self) -> tuple[float, float]:
        return (self.focus_u, self.focus_v)


GAUSSIAN_OVERLAP_MONITOR_TYPES = (GaussianOverlapMonitor, AstigmaticGaussianOverlapMonitor)


def _normalize_time_components(components, allowed, *, owner):
    if isinstance(components, str):
        components = (components,)
    resolved = []
    lookup = {name.lower(): name for name in allowed}
    for component in components:
        key = str(component).lower()
        if key not in lookup:
            raise ValueError(f"{owner} components must be drawn from {allowed}, got {component!r}.")
        resolved.append(lookup[key])
    if not resolved:
        raise ValueError(f"{owner} components must not be empty.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{owner} components must be unique.")
    return tuple(resolved)


@dataclass(frozen=True)
class SurfaceFieldTimeMonitor:
    """Record detached time-domain Yee fields on a plane patch or an (open) box surface.

    ``position``/``size`` declare one plane patch (one zero extent) or a box
    whose six faces (minus ``exclude_surfaces``) form the surface. On every
    face each requested component is copied from its nearest native Yee plane
    over the face's tangential extent, without interpolation, on the
    ``start``/``stop``/``interval`` step schedule of :class:`FieldTimeMonitor`.
    Traces are detached, like :class:`FieldTimeMonitor`; use spectral monitors
    for adjoint objectives.
    """

    name: str
    components: tuple[str, ...]
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    start: int
    stop: int | None
    interval: int
    surface_kind: str
    exclude_surfaces: tuple[str, ...]
    face_specs: tuple[tuple[str, str, str, tuple[float, float, float], tuple[float, float, float]], ...]
    kind: str = "surface_field_time"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        start=0,
        stop=None,
        interval=1,
        *,
        exclude_surfaces=None,
    ):
        monitor_name = str(name).strip()
        if not monitor_name:
            raise ValueError("SurfaceFieldTimeMonitor name must not be empty.")
        normalized_components = tuple(
            normalize_component(component).capitalize() for component in _normalize_fields(components)
        )
        if len(set(normalized_components)) != len(normalized_components):
            raise ValueError("SurfaceFieldTimeMonitor components must be unique.")
        surface_kind, center, extents, _, exclusions, face_specs = _resolve_declared_surface(
            monitor_name,
            position,
            size,
            owner="SurfaceFieldTimeMonitor",
            exclude_surfaces=exclude_surfaces,
        )
        resolved_start, resolved_stop, resolved_interval = _validate_time_sampling(start, stop, interval)
        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "components", normalized_components)
        object.__setattr__(self, "position", center)
        object.__setattr__(self, "size", extents)
        object.__setattr__(self, "start", resolved_start)
        object.__setattr__(self, "stop", resolved_stop)
        object.__setattr__(self, "interval", resolved_interval)
        object.__setattr__(self, "surface_kind", surface_kind)
        object.__setattr__(self, "exclude_surfaces", exclusions)
        object.__setattr__(self, "face_specs", face_specs)
        object.__setattr__(self, "kind", "surface_field_time")

    @property
    def face_labels(self) -> tuple[str, ...]:
        return tuple(spec[0] for spec in self.face_specs)


# Auxiliary ADE state recorded by AuxFieldTimeMonitor, keyed by public name:
# (field family, Yee component it is co-located with, state tensor name).
AUX_FIELD_COMPONENTS = {
    "Px": ("electric", "Ex", "polarization"),
    "Py": ("electric", "Ey", "polarization"),
    "Pz": ("electric", "Ez", "polarization"),
    "Jpx": ("electric", "Ex", "current"),
    "Jpy": ("electric", "Ey", "current"),
    "Jpz": ("electric", "Ez", "current"),
    "Mx": ("magnetic", "Hx", "polarization"),
    "My": ("magnetic", "Hy", "polarization"),
    "Mz": ("magnetic", "Hz", "polarization"),
    "Jmx": ("magnetic", "Hx", "current"),
    "Jmy": ("magnetic", "Hy", "current"),
    "Jmz": ("magnetic", "Hz", "current"),
}


@dataclass(frozen=True)
class AuxFieldTimeMonitor:
    """Record detached time-domain auxiliary dispersive-material state.

    Components are the native auxiliary-differential-equation (ADE) state the
    FDTD runtime keeps for Debye/Drude/Lorentz poles, summed over all poles of
    the co-located Yee component:

    - ``Px, Py, Pz``: electric polarization (C/m^2) of Debye and Lorentz poles
      (Drude poles carry no stored polarization);
    - ``Jpx, Jpy, Jpz``: electric polarization current density (A/m^2) of all
      electric poles;
    - ``Mx, My, Mz`` and ``Jmx, Jmy, Jmz``: the magnetic-pole counterparts
      (``mu0``-scaled magnetic polarization, V*s/m^2, and its rate, V/m^2).

    A component whose ADE state does not exist in the run (no such pole in
    the scene) records zeros and is flagged in the payload ``present``
    mapping. Samples follow the :class:`FieldTimeMonitor` schedule and clock
    (electric state with E, magnetic state with H); the state recorded at a
    sample was advanced with the co-located field of the previous sample
    (payload ``drive_lag_samples=1``). The region is a point, an axis-aligned patch or a volume; along a
    zero extent the nearest native Yee index is used. Instantaneous Kerr,
    chi2 and two-photon-absorption media keep no auxiliary state, and
    free-carrier densities are not modeled, so neither is recordable.
    Traces are detached; use spectral monitors for adjoint objectives.
    """

    name: str
    components: tuple[str, ...]
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    start: int
    stop: int | None
    interval: int
    region_kind: str
    kind: str = "aux_field_time"

    def __init__(
        self,
        name,
        components=("Px", "Py", "Pz"),
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        start=0,
        stop=None,
        interval=1,
    ):
        monitor_name = str(name).strip()
        if not monitor_name:
            raise ValueError("AuxFieldTimeMonitor name must not be empty.")
        resolved_components = _normalize_time_components(
            components, tuple(AUX_FIELD_COMPONENTS), owner="AuxFieldTimeMonitor"
        )
        resolved_position = _require_length3("position", position)
        resolved_size = _require_nonnegative_length3("size", size)
        resolved_start, resolved_stop, resolved_interval = _validate_time_sampling(start, stop, interval)
        zero_axes = sum(1 for length in resolved_size if abs(length) <= 1e-12)
        region_kind = {3: "point", 2: "line", 1: "plane"}.get(zero_axes, "volume")
        object.__setattr__(self, "name", monitor_name)
        object.__setattr__(self, "components", resolved_components)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "start", resolved_start)
        object.__setattr__(self, "stop", resolved_stop)
        object.__setattr__(self, "interval", resolved_interval)
        object.__setattr__(self, "region_kind", region_kind)
        object.__setattr__(self, "kind", "aux_field_time")


IN_SIMULATION_MONITOR_TYPES = (
    *PROJECTION_MONITOR_TYPES,
    *GAUSSIAN_OVERLAP_MONITOR_TYPES,
    SurfaceFieldTimeMonitor,
    AuxFieldTimeMonitor,
)
