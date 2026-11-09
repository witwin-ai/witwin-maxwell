"""Custom 2D and composite voltage/current integral declarations for RF ports.

These are declarative specifications only. They are lowered onto the Yee grid by
``compiler/path_integrals.py`` when a :class:`~witwin.maxwell.TerminalWavePort` is
compiled:

* a voltage path follows primal electric edges node to node, so the discrete line
  integral of ``E = -grad(phi)`` is exactly a potential difference;
* a current loop is the boundary of the enclosed dual cells, so its discrete
  circulation is exactly the enclosed discrete Ampere current.

All coordinates are fixed (non-trainable) inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ._fixed_inputs import require_fixed_input


def _point3(label: str, value) -> tuple[float, float, float]:
    require_fixed_input(value, name=label)
    try:
        items = tuple(float(item) for item in value)
    except TypeError as error:
        raise TypeError(f"{label} must be a length-3 point.") from error
    if len(items) != 3:
        raise ValueError(f"{label} must be a length-3 point.")
    if any(not math.isfinite(item) for item in items):
        raise ValueError(f"{label} must be finite.")
    return items


def _varying_axes(first, second) -> list[int]:
    return [
        axis
        for axis in range(3)
        if not math.isclose(first[axis], second[axis], rel_tol=0.0, abs_tol=1.0e-12)
    ]


@dataclass(frozen=True)
class AxisSegment:
    """One axis-aligned straight path from ``start`` to ``end``."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]

    def __init__(self, start, end):
        resolved_start = _point3("AxisSegment start", start)
        resolved_end = _point3("AxisSegment end", end)
        if len(_varying_axes(resolved_start, resolved_end)) != 1:
            raise ValueError("AxisSegment must be a nonzero axis-aligned segment.")
        object.__setattr__(self, "start", resolved_start)
        object.__setattr__(self, "end", resolved_end)

    @property
    def points(self) -> tuple[tuple[float, float, float], ...]:
        return (self.start, self.end)


@dataclass(frozen=True)
class PolylinePath:
    """An open in-plane polyline ``points[0] -> points[-1]``.

    Every vertex must lie on a Yee node of the port plane. Each leg is lowered to
    a deterministic node-to-node staircase of primal electric edges whose signed
    lengths sum to the leg's projections, so the integral of a gradient field is
    exactly the potential difference between the end vertices.
    """

    points: tuple[tuple[float, float, float], ...]

    def __init__(self, points):
        try:
            resolved = tuple(_point3("PolylinePath point", point) for point in points)
        except TypeError as error:
            raise TypeError("PolylinePath points must be an iterable of 3D points.") from error
        if len(resolved) < 2:
            raise ValueError("PolylinePath requires at least two points.")
        for first, second in zip(resolved, resolved[1:]):
            if not _varying_axes(first, second):
                raise ValueError("PolylinePath consecutive points must be distinct.")
        object.__setattr__(self, "points", resolved)


@dataclass(frozen=True)
class RectangleLoop:
    """A planar rectangular current loop; ``size`` has exactly one zero component."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def __init__(self, center, size):
        resolved_center = _point3("RectangleLoop center", center)
        resolved_size = _point3("RectangleLoop size", size)
        zero = [axis for axis, value in enumerate(resolved_size) if abs(value) <= 1.0e-12]
        if len(zero) != 1:
            raise ValueError("RectangleLoop size must contain exactly one zero component.")
        if any(value < 0.0 for value in resolved_size):
            raise ValueError("RectangleLoop size components must be nonnegative.")
        object.__setattr__(self, "center", resolved_center)
        object.__setattr__(self, "size", resolved_size)

    @property
    def normal_index(self) -> int:
        return next(axis for axis, value in enumerate(self.size) if abs(value) <= 1.0e-12)

    def vertices(self) -> tuple[tuple[float, float, float], ...]:
        normal = self.normal_index
        first, second = (axis for axis in range(3) if axis != normal)
        corners = []
        for sign_first, sign_second in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            point = list(self.center)
            point[first] += 0.5 * sign_first * self.size[first]
            point[second] += 0.5 * sign_second * self.size[second]
            corners.append(tuple(point))
        return tuple(corners)


@dataclass(frozen=True)
class PolygonLoop:
    """A closed planar polygon current loop (vertices in order, not repeated)."""

    points: tuple[tuple[float, float, float], ...]

    def __init__(self, points):
        try:
            resolved = tuple(_point3("PolygonLoop point", point) for point in points)
        except TypeError as error:
            raise TypeError("PolygonLoop points must be an iterable of 3D points.") from error
        if len(resolved) < 3:
            raise ValueError("PolygonLoop requires at least three points.")
        object.__setattr__(self, "points", resolved)

    def vertices(self) -> tuple[tuple[float, float, float], ...]:
        return self.points


@dataclass(frozen=True)
class CircleLoop:
    """A circular current loop approximated by a regular ``segments``-gon."""

    center: tuple[float, float, float]
    radius: float
    axis: str
    segments: int = 64

    def __init__(self, center, radius, axis, segments=64):
        resolved_center = _point3("CircleLoop center", center)
        require_fixed_input(radius, name="CircleLoop radius")
        resolved_radius = float(radius)
        if not math.isfinite(resolved_radius) or resolved_radius <= 0.0:
            raise ValueError("CircleLoop radius must be finite and positive.")
        resolved_axis = str(axis).strip().lower()
        if resolved_axis not in {"x", "y", "z"}:
            raise ValueError("CircleLoop axis must be 'x', 'y', or 'z'.")
        if isinstance(segments, bool) or int(segments) != segments or int(segments) < 8:
            raise ValueError("CircleLoop segments must be an integer >= 8.")
        object.__setattr__(self, "center", resolved_center)
        object.__setattr__(self, "radius", resolved_radius)
        object.__setattr__(self, "axis", resolved_axis)
        object.__setattr__(self, "segments", int(segments))

    def vertices(self) -> tuple[tuple[float, float, float], ...]:
        normal = "xyz".index(self.axis)
        first, second = (axis for axis in range(3) if axis != normal)
        points = []
        for index in range(self.segments):
            angle = 2.0 * math.pi * index / self.segments
            point = list(self.center)
            point[first] += self.radius * math.cos(angle)
            point[second] += self.radius * math.sin(angle)
            points.append(tuple(point))
        return tuple(points)


@dataclass(frozen=True)
class CustomVoltageIntegral:
    """``V = integral of E . dl`` along ``path`` in its declared point order.

    For a terminal port declare the path from the terminal conductor to the
    reference conductor, so ``V`` is the terminal potential above the reference.
    """

    path: AxisSegment | PolylinePath

    def __init__(self, path):
        if not isinstance(path, (AxisSegment, PolylinePath)):
            raise TypeError("CustomVoltageIntegral path must be an AxisSegment or PolylinePath.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class CustomCurrentIntegral:
    """``I = circulation of H`` around a closed in-plane loop.

    The circulation is right-handed about the owning port's propagation
    direction, so a forward wave on the enclosed conductor has positive current.
    """

    loop: RectangleLoop | PolygonLoop | CircleLoop

    def __init__(self, loop):
        if not isinstance(loop, (RectangleLoop, PolygonLoop, CircleLoop)):
            raise TypeError("CustomCurrentIntegral loop must be a RectangleLoop, PolygonLoop, or CircleLoop.")
        object.__setattr__(self, "loop", loop)


def _composite_terms(label, terms, allowed):
    try:
        resolved = tuple(tuple(term) for term in terms)
    except TypeError as error:
        raise TypeError(f"{label} terms must be an iterable of (coefficient, integral) pairs.") from error
    if not resolved:
        raise ValueError(f"{label} requires at least one term.")
    output = []
    for term in resolved:
        if len(term) != 2:
            raise ValueError(f"{label} terms must be (coefficient, integral) pairs.")
        coefficient, integral = term
        require_fixed_input(coefficient, name=f"{label} coefficient")
        value = float(coefficient)
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"{label} coefficients must be finite and nonzero.")
        if not isinstance(integral, allowed):
            raise TypeError(f"{label} terms must contain {' or '.join(item.__name__ for item in allowed)} integrals.")
        output.append((value, integral))
    return tuple(output)


@dataclass(frozen=True)
class CompositeVoltageIntegral:
    """A real linear combination of voltage integrals."""

    terms: tuple[tuple[float, CustomVoltageIntegral], ...]

    def __init__(self, terms):
        object.__setattr__(
            self,
            "terms",
            _composite_terms("CompositeVoltageIntegral", terms, (CustomVoltageIntegral, CompositeVoltageIntegral)),
        )


@dataclass(frozen=True)
class CompositeCurrentIntegral:
    """A real linear combination of current integrals."""

    terms: tuple[tuple[float, CustomCurrentIntegral], ...]

    def __init__(self, terms):
        object.__setattr__(
            self,
            "terms",
            _composite_terms("CompositeCurrentIntegral", terms, (CustomCurrentIntegral, CompositeCurrentIntegral)),
        )


VOLTAGE_INTEGRAL_TYPES = (CustomVoltageIntegral, CompositeVoltageIntegral)
CURRENT_INTEGRAL_TYPES = (CustomCurrentIntegral, CompositeCurrentIntegral)

__all__ = [
    "AxisSegment",
    "CircleLoop",
    "CompositeCurrentIntegral",
    "CompositeVoltageIntegral",
    "CustomCurrentIntegral",
    "CustomVoltageIntegral",
    "PolygonLoop",
    "PolylinePath",
    "RectangleLoop",
]
