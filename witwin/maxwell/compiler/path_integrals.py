"""Lower custom 2D / composite voltage and current integrals onto the Yee grid.

A compiled integral is a sparse multi-term row ``sum_t sum_k w_tk F_t[idx_tk]``
over native field components:

* voltage paths follow primal electric edges node to node (a deterministic
  staircase), so the integral of ``E = -grad(phi)`` is exactly ``phi(start) -
  phi(end)`` independent of the chosen staircase;
* current loops are the boundary of the union of dual cells of the enclosed
  nodes, so the circulation of ``H`` is exactly the enclosed discrete Ampere
  current, independent of how far the loop sits from the conductor edges.

These rows are evaluated on modal transverse profiles; they never alter the
existing ``AxisPath`` / ``Box`` lowering used by the other RF ports.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..path_integrals import (
    AxisSegment,
    CompositeCurrentIntegral,
    CompositeVoltageIntegral,
    CustomCurrentIntegral,
    CustomVoltageIntegral,
    PolylinePath,
)
from .ports import _axis_values, _axis_widths, _snap_index

_AXES = "xyz"


@dataclass(frozen=True)
class PortPlane:
    """Grid description of one planar modal aperture."""

    normal_axis: str
    tangential_axes: tuple[str, str]
    electric_plane_index: int
    magnetic_plane_index: int
    lower: tuple[int, int]
    upper: tuple[int, int]
    reference_plane: float
    direction_sign: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.upper[0] - self.lower[0] + 1, self.upper[1] - self.lower[1] + 1)

    @property
    def handedness(self) -> int:
        # (u, v, n) is right-handed except for the y normal, where (x, z) gives -y.
        return -1 if self.normal_axis == "y" else 1


@dataclass(frozen=True)
class CompiledPathIntegral:
    """Grid-resolved custom/composite integral.

    ``terms`` are ``(component, indices[K, 3], weights[K])`` rows. ``paths`` keeps
    ``(coefficient, start_local, end_local)`` for every voltage leaf and ``loops``
    keeps ``(coefficient, enclosed_flat_local_nodes)`` for every current leaf, so the
    resolved conductor topology can audit the integral before it is used.
    """

    kind: str
    terms: tuple[tuple[str, torch.Tensor, torch.Tensor], ...]
    paths: tuple = ()
    loops: tuple = ()
    composite: bool = False


def _plane_tolerance(plane: PortPlane) -> float:
    return 1.0e-9 * max(1.0, abs(plane.reference_plane))


def _snap_plane_point(scene, plane: PortPlane, point, *, label: str) -> tuple[int, int]:
    normal_index = _AXES.index(plane.normal_axis)
    if abs(float(point[normal_index]) - plane.reference_plane) > _plane_tolerance(plane):
        raise ValueError(f"{label} must lie in the port plane {plane.normal_axis}={plane.reference_plane:g}.")
    local = []
    for slot, axis in enumerate(plane.tangential_axes):
        index = _snap_index(
            _axis_values(scene, axis, half=False),
            float(point[_AXES.index(axis)]),
            label=f"{label} {axis} node",
        )
        if index < plane.lower[slot] or index > plane.upper[slot]:
            raise ValueError(f"{label} lies outside the port aperture along {axis}.")
        local.append(index - plane.lower[slot])
    return tuple(local)


def _staircase(scene, plane: PortPlane, start, end):
    """Node-to-node primal-edge staircase from ``start`` to ``end`` (local indices)."""
    coords = [
        _axis_values(scene, axis, half=False)[plane.lower[slot]:plane.upper[slot] + 1]
        for slot, axis in enumerate(plane.tangential_axes)
    ]
    origin = np.array([coords[0][start[0]], coords[1][start[1]]])
    target = np.array([coords[0][end[0]], coords[1][end[1]]])
    direction = target - origin
    length = float(np.linalg.norm(direction))
    steps = []
    current = list(start)
    signs = [int(np.sign(end[0] - start[0])), int(np.sign(end[1] - start[1]))]
    while tuple(current) != tuple(end):
        candidates = []
        for slot in (0, 1):
            if current[slot] == end[slot]:
                continue
            trial = list(current)
            trial[slot] += signs[slot]
            point = np.array([coords[0][trial[0]], coords[1][trial[1]]])
            offset = point - origin
            distance = abs(direction[0] * offset[1] - direction[1] * offset[0]) / length
            candidates.append((distance, slot, trial))
        _distance, slot, trial = min(candidates, key=lambda item: (item[0], item[1]))
        steps.append((slot, current[slot], trial[slot], tuple(current)))
        current = trial
    return steps


def _accumulate(rows: dict, component: str, index: tuple[int, int, int], weight: float) -> None:
    key = (component, index)
    rows[key] = rows.get(key, 0.0) + weight


def _voltage_leaves(spec, coefficient=1.0):
    if isinstance(spec, CustomVoltageIntegral):
        yield coefficient, spec
        return
    for value, item in spec.terms:
        yield from _voltage_leaves(item, coefficient * value)


def _current_leaves(spec, coefficient=1.0):
    if isinstance(spec, CustomCurrentIntegral):
        yield coefficient, spec
        return
    for value, item in spec.terms:
        yield from _current_leaves(item, coefficient * value)


def _rows_to_terms(rows: dict, *, device) -> tuple:
    by_component: dict[str, tuple[list, list]] = {}
    for (component, index), weight in sorted(rows.items()):
        if weight == 0.0:
            continue
        indices, weights = by_component.setdefault(component, ([], []))
        indices.append(index)
        weights.append(weight)
    if not by_component:
        raise ValueError("integral reduces to an empty discrete row.")
    return tuple(
        (
            component,
            torch.as_tensor(np.asarray(indices, dtype=np.int64), device=device),
            torch.as_tensor(np.asarray(weights, dtype=np.float64), device=device),
        )
        for component, (indices, weights) in by_component.items()
    )


def compile_voltage_integral(scene, spec, plane: PortPlane, *, device) -> CompiledPathIntegral:
    if not isinstance(spec, (CustomVoltageIntegral, CompositeVoltageIntegral)):
        raise TypeError("voltage integral must be a CustomVoltageIntegral or CompositeVoltageIntegral.")
    normal_index = _AXES.index(plane.normal_axis)
    rows: dict = {}
    paths = []
    for coefficient, leaf in _voltage_leaves(spec):
        path = leaf.path
        points = path.points if isinstance(path, (AxisSegment, PolylinePath)) else ()
        vertices = [_snap_plane_point(scene, plane, point, label="voltage path vertex") for point in points]
        if vertices[0] == vertices[-1]:
            raise ValueError("voltage path starts and ends on the same Yee node.")
        for first, second in zip(vertices, vertices[1:]):
            for slot, low, high, node in _staircase(scene, plane, first, second):
                axis = plane.tangential_axes[slot]
                edge = min(low, high) + plane.lower[slot]
                width = float(_axis_widths(scene, axis, dual=False)[edge])
                index = [0, 0, 0]
                index[normal_index] = plane.electric_plane_index
                other = 1 - slot
                index[_AXES.index(axis)] = edge
                index[_AXES.index(plane.tangential_axes[other])] = node[other] + plane.lower[other]
                _accumulate(rows, f"E{axis}", tuple(index), coefficient * (high - low) * width)
        paths.append((coefficient, vertices[0], vertices[-1]))
    return CompiledPathIntegral(
        kind="voltage",
        terms=_rows_to_terms(rows, device=device),
        paths=tuple(paths),
        composite=isinstance(spec, CompositeVoltageIntegral),
    )


def _loop_vertices(scene, plane: PortPlane, loop) -> np.ndarray:
    normal_index = _AXES.index(plane.normal_axis)
    vertices = loop.vertices()
    if getattr(loop, "axis", None) is not None and loop.axis != plane.normal_axis:
        raise ValueError("current loop axis must match the port normal.")
    if getattr(loop, "normal_index", normal_index) != normal_index:
        raise ValueError("current loop normal must match the port normal.")
    output = []
    for point in vertices:
        if abs(float(point[normal_index]) - plane.reference_plane) > _plane_tolerance(plane):
            raise ValueError(
                f"current loop must lie in the port plane {plane.normal_axis}={plane.reference_plane:g}."
            )
        output.append([float(point[_AXES.index(axis)]) for axis in plane.tangential_axes])
    return np.asarray(output, dtype=np.float64)


def _enclosed_nodes(scene, plane: PortPlane, polygon: np.ndarray) -> np.ndarray:
    coords = [
        _axis_values(scene, axis, half=False)[plane.lower[slot]:plane.upper[slot] + 1]
        for slot, axis in enumerate(plane.tangential_axes)
    ]
    spacing = min(float(np.min(np.diff(values))) for values in coords)
    u, v = np.meshgrid(coords[0], coords[1], indexing="ij")
    inside = np.zeros(u.shape, dtype=bool)
    edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
    for (ax, ay), (bx, by) in edges:
        segment = np.array([bx - ax, by - ay])
        length = float(np.linalg.norm(segment))
        if length == 0.0:
            continue
        along = ((u - ax) * segment[0] + (v - ay) * segment[1]) / length**2
        across = np.abs((u - ax) * segment[1] - (v - ay) * segment[0]) / length
        if bool(np.any((across <= 1.0e-6 * spacing) & (along >= -1.0e-9) & (along <= 1.0 + 1.0e-9))):
            raise ValueError("current loop passes through a Yee node; offset it so every node is strictly inside or outside.")
        crosses = (ay > v) != (by > v)
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection = ax + (v - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (u < intersection)
    return inside


def compile_current_integral(scene, spec, plane: PortPlane, *, device) -> CompiledPathIntegral:
    if not isinstance(spec, (CustomCurrentIntegral, CompositeCurrentIntegral)):
        raise TypeError("current integral must be a CustomCurrentIntegral or CompositeCurrentIntegral.")
    normal_index = _AXES.index(plane.normal_axis)
    axis_u, axis_v = plane.tangential_axes
    dual_u = _axis_widths(scene, axis_u, dual=True)
    dual_v = _axis_widths(scene, axis_v, dual=True)
    orientation = plane.handedness * plane.direction_sign
    rows: dict = {}
    loops = []
    for coefficient, leaf in _current_leaves(spec):
        inside = _enclosed_nodes(scene, plane, _loop_vertices(scene, plane, leaf.loop))
        if not bool(np.any(inside)):
            raise ValueError("current loop encloses no Yee node.")
        if bool(np.any(inside[0, :]) or np.any(inside[-1, :]) or np.any(inside[:, 0]) or np.any(inside[:, -1])):
            raise ValueError("current loop must stay strictly inside the port aperture.")
        weight = coefficient * orientation
        # u-neighbour pairs share a dual edge along v at u half index i.
        change_u = inside[1:, :].astype(np.int8) - inside[:-1, :].astype(np.int8)
        for i, j in zip(*np.nonzero(change_u)):
            sign = 1.0 if change_u[i, j] < 0 else -1.0  # inside at i, outside at i+1 -> +v
            index = [0, 0, 0]
            index[normal_index] = plane.magnetic_plane_index
            index[_AXES.index(axis_u)] = int(i) + plane.lower[0]
            index[_AXES.index(axis_v)] = int(j) + plane.lower[1]
            _accumulate(rows, f"H{axis_v}", tuple(index), weight * sign * float(dual_v[int(j) + plane.lower[1]]))
        # v-neighbour pairs share a dual edge along u at v half index j.
        change_v = inside[:, 1:].astype(np.int8) - inside[:, :-1].astype(np.int8)
        for i, j in zip(*np.nonzero(change_v)):
            sign = -1.0 if change_v[i, j] < 0 else 1.0  # inside at j, outside at j+1 -> -u
            index = [0, 0, 0]
            index[normal_index] = plane.magnetic_plane_index
            index[_AXES.index(axis_u)] = int(i) + plane.lower[0]
            index[_AXES.index(axis_v)] = int(j) + plane.lower[1]
            _accumulate(rows, f"H{axis_u}", tuple(index), weight * sign * float(dual_u[int(i) + plane.lower[0]]))
        loops.append((coefficient, tuple(int(value) for value in np.flatnonzero(inside.reshape(-1)))))
    return CompiledPathIntegral(
        kind="current",
        terms=_rows_to_terms(rows, device=device),
        loops=tuple(loops),
        composite=isinstance(spec, CompositeCurrentIntegral),
    )


def audit_terminal_integral(compiled: CompiledPathIntegral, label_map: np.ndarray, terminal: int) -> dict:
    """Check a compiled terminal integral against the resolved conductor labels.

    ``label_map`` uses ``-1`` for dielectric, ``0`` for the reference and ``k + 1``
    for terminal ``k``; the aperture boundary is grounded. A plain voltage path must
    run from terminal ``k`` to the reference; a plain current loop must enclose
    terminal ``k`` entirely and no other conductor. Composite leaves must start and
    end on conductors and enclose only whole conductors. Returns an audit record.
    """
    labels = np.asarray(label_map)
    shape = labels.shape
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    grounded = (labels == 0) | boundary

    def node_label(node):
        if grounded[node]:
            return 0
        return int(labels[node])

    if compiled.kind == "voltage":
        for _coefficient, start, end in compiled.paths:
            start_label, end_label = node_label(start), node_label(end)
            if start_label < 0 or end_label < 0:
                raise ValueError(
                    f"terminal {terminal + 1} voltage path endpoints must lie on conductor nodes "
                    f"(start label {start_label}, end label {end_label})."
                )
            if not compiled.composite and (start_label != terminal + 1 or end_label != 0):
                raise ValueError(
                    f"terminal {terminal + 1} voltage path must run from that terminal to the reference "
                    f"(it runs from conductor {start_label} to conductor {end_label}; 0 is the reference)."
                )
        return {"kind": "voltage", "checked": "composite_endpoints" if compiled.composite else "terminal_to_reference"}
    flat = labels.reshape(-1)
    for _coefficient, enclosed in compiled.loops:
        enclosed = np.asarray(enclosed, dtype=np.int64)
        mask = np.zeros(flat.shape, dtype=bool)
        mask[enclosed] = True
        conductor_labels = set(int(value) for value in np.unique(flat[mask]) if value >= 0)
        for value in conductor_labels:
            if bool(np.any((flat == value) & ~mask)):
                raise ValueError(
                    f"terminal {terminal + 1} current loop cuts through conductor {value} "
                    "(0 is the reference); a loop must enclose whole conductors."
                )
        if not compiled.composite and conductor_labels != {terminal + 1}:
            raise ValueError(
                f"terminal {terminal + 1} current loop must enclose exactly that terminal; it encloses "
                f"conductors {sorted(conductor_labels)} (0 is the reference)."
            )
    return {"kind": "current", "checked": "composite_whole_conductors" if compiled.composite else "encloses_terminal"}


__all__ = [
    "CompiledPathIntegral",
    "PortPlane",
    "audit_terminal_integral",
    "compile_current_integral",
    "compile_voltage_integral",
]
