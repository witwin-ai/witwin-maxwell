from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch


EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
_AXES = "xyz"
_COMPONENTS = ("Ex", "Ey", "Ez")
_MAX_RADIUS_TO_SPACING = 0.2


@dataclass(frozen=True)
class CompiledWireNetwork:
    """Device-native sparse graph and Yee coupling for Phase 1 thin wires."""

    wire_names: tuple[str, ...]
    field_shapes: tuple[tuple[int, int, int], ...]
    node_positions: torch.Tensor
    node_grid_indices: torch.Tensor
    node_ids: torch.Tensor
    node_wire_ids: torch.Tensor
    wire_node_offsets: torch.Tensor
    node_offsets: torch.Tensor
    node_segments: torch.Tensor
    node_signs: torch.Tensor
    node_capacitance: torch.Tensor
    open_endpoints: torch.Tensor
    grounded: torch.Tensor
    segment_ids: torch.Tensor
    segment_wire_ids: torch.Tensor
    wire_segment_offsets: torch.Tensor
    tail: torch.Tensor
    head: torch.Tensor
    segment_axes: torch.Tensor
    segment_directions: torch.Tensor
    radius: torch.Tensor
    length: torch.Tensor
    coupling_distance: torch.Tensor
    radius_to_spacing: torch.Tensor
    inductance: torch.Tensor
    capacitance_per_length: torch.Tensor
    segment_offsets: torch.Tensor
    edge_components: torch.Tensor
    edge_offsets: torch.Tensor
    weights: torch.Tensor
    edge_group_offsets: torch.Tensor
    target_components: torch.Tensor
    target_offsets: torch.Tensor
    contribution_segments: torch.Tensor
    contribution_weights: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def device(self) -> torch.device:
        return self.node_positions.device

    @property
    def node_count(self) -> int:
        return int(self.node_ids.numel())

    @property
    def segment_count(self) -> int:
        return int(self.segment_ids.numel())

    def tensor_fields(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            value
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), torch.Tensor)
        )


@dataclass(frozen=True)
class CompiledWireMonitor:
    """Resolved node and segment indices for one wire monitor."""

    name: str
    wire_name: str
    wire_id: int
    node_indices: torch.Tensor
    segment_indices: torch.Tensor
    quantities: tuple[str, ...]
    frequencies: tuple[float, ...]


def _kind_name(value) -> str:
    resolved = getattr(value, "kind", value)
    resolved = getattr(resolved, "value", resolved)
    return str(resolved).strip().lower()


def _grid_tolerance(coords: np.ndarray) -> float:
    spacing = np.diff(coords)
    minimum = float(np.min(spacing)) if spacing.size else 1.0
    return max(1.0e-12, minimum * 1.0e-6)


def _snap_index(coords: np.ndarray, value: float, *, strict: bool, label: str) -> int:
    index = int(np.argmin(np.abs(coords - value)))
    if strict and abs(float(coords[index]) - value) > _grid_tolerance(coords):
        raise ValueError(f"{label}={value} must lie on a grid node when snap='strict'.")
    return index


def _point_tensor(points, *, wire_name: str) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        if points.requires_grad:
            raise ValueError(f"ThinWire {wire_name!r} trainable points are not supported in Phase 1.")
        tensor = points
    else:
        for point in points:
            for coordinate in point:
                if isinstance(coordinate, torch.Tensor) and coordinate.requires_grad:
                    raise ValueError(
                        f"ThinWire {wire_name!r} trainable points are not supported in Phase 1."
                    )
        tensor = torch.as_tensor(points)
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"ThinWire {wire_name!r} points must be real coordinates.")
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] < 2:
        raise ValueError(f"ThinWire {wire_name!r} points must have shape (N, 3) with N >= 2.")
    tensor = tensor.detach().cpu().to(dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"ThinWire {wire_name!r} points must be finite.")
    return tensor


def _radius_dtype(value) -> torch.dtype:
    if not isinstance(value, torch.Tensor):
        return torch.float64
    if not value.is_floating_point() or value.is_complex():
        raise TypeError("ThinWire radius must be a real floating-point tensor or scalar.")
    return value.dtype


def _radius_tensor(
    value,
    *,
    source_segment_count: int,
    device: torch.device,
    dtype: torch.dtype,
    wire_name: str,
) -> torch.Tensor:
    radius = torch.as_tensor(value, device=device, dtype=dtype)
    if radius.ndim == 0 or radius.numel() == 1:
        radius = radius.reshape(()).expand(source_segment_count)
    elif radius.ndim == 1 and radius.numel() == source_segment_count:
        radius = radius.reshape(source_segment_count)
    else:
        raise ValueError(
            f"ThinWire {wire_name!r} radius must be scalar or have one value per polyline segment."
        )
    if not bool(torch.all(torch.isfinite(radius))):
        raise ValueError(f"ThinWire {wire_name!r} radius must be finite.")
    if not bool(torch.all(radius > 0)):
        raise ValueError(f"ThinWire {wire_name!r} radius must be positive.")
    if not bool(torch.all(radius >= torch.finfo(dtype).tiny)):
        raise ValueError(
            f"ThinWire {wire_name!r} radius must remain in the normal floating-point range."
        )
    return radius


@lru_cache(maxsize=512)
def _bspline_coupling_distance(du: float, dv: float) -> float:
    """Geometric-mean radius of the BS1 x BS1 transverse kernel."""

    from scipy.integrate import quad

    du = float(du)
    dv = float(dv)
    if not math.isfinite(du) or not math.isfinite(dv) or du <= 0.0 or dv <= 0.0:
        raise ValueError("Transverse grid spacings must be positive and finite.")

    def integrate_v(u: float) -> float:
        axial = du * u
        axial2 = axial * axial
        dv2 = dv * dv
        if axial == 0.0:
            integral_log = 2.0 * math.log(dv) - 2.0
            integral_v_log = math.log(dv) - 0.5
        else:
            total = axial2 + dv2
            integral_log = math.log(total) - 2.0 + 2.0 * axial * math.atan(dv / axial) / dv
            upper = total * math.log(total) - total
            lower = axial2 * math.log(axial2) - axial2
            integral_v_log = (upper - lower) / (2.0 * dv2)
        return 0.5 * (integral_log - integral_v_log)

    log_mean = 4.0 * quad(
        lambda u: (1.0 - u) * integrate_v(u),
        0.0,
        1.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=100,
    )[0]
    return math.exp(log_mean)


def _grid_arrays(scene) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    nodes = tuple(np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64) for axis in _AXES)
    primal = tuple(np.asarray(getattr(scene, f"d{axis}_primal64"), dtype=np.float64) for axis in _AXES)
    dual = tuple(np.asarray(getattr(scene, f"d{axis}_dual64"), dtype=np.float64) for axis in _AXES)
    for axis, axis_nodes, axis_primal, axis_dual in zip(_AXES, nodes, primal, dual):
        if axis_nodes.ndim != 1 or axis_nodes.size < 2 or not np.all(np.diff(axis_nodes) > 0.0):
            raise ValueError(f"Prepared scene {axis}-axis nodes must be strictly increasing.")
        if axis_primal.shape != (axis_nodes.size - 1,) or axis_dual.shape != axis_nodes.shape:
            raise ValueError(f"Prepared scene {axis}-axis spacing arrays do not match its nodes.")
    return nodes, primal, dual


def _field_shapes(nodes: tuple[np.ndarray, ...]) -> tuple[tuple[int, int, int], ...]:
    nx, ny, nz = (int(axis.size) for axis in nodes)
    return ((nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1))


def _flat_offset(component: int, edge: tuple[int, int, int], shapes) -> int:
    i, j, k = edge
    shape = shapes[component]
    if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
        raise ValueError("Compiled thin-wire edge lies outside its Yee component array.")
    return (i * shape[1] + j) * shape[2] + k


def _hash_arrays(*entries) -> str:
    digest = hashlib.sha256()
    for label, value in entries:
        digest.update(str(label).encode("utf8"))
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _grid_fingerprint(scene, nodes, primal, dual) -> str:
    entries = []
    for axis, axis_nodes, axis_primal, axis_dual in zip(_AXES, nodes, primal, dual):
        entries.extend(
            (
                (f"{axis}_nodes", axis_nodes),
                (f"d{axis}_primal", axis_primal),
                (f"d{axis}_dual", axis_dual),
            )
        )
    boundary = getattr(scene, "boundary", None)
    if boundary is not None:
        face_kinds = tuple(
            _kind_name(boundary.face_kind(axis, side))
            for axis in _AXES
            for side in ("low", "high")
        )
        entries.append(("boundary", np.asarray(face_kinds, dtype="S16")))
    return _hash_arrays(*entries)


def _domain_error(scene, point: np.ndarray, *, wire_name: str) -> None:
    bounds = tuple(tuple(float(value) for value in pair) for pair in scene.domain.bounds)
    tolerances = tuple(_grid_tolerance(np.asarray(getattr(scene, f"{axis}_nodes64"))) for axis in _AXES)
    if all(lo - tolerance <= value <= hi + tolerance for value, (lo, hi), tolerance in zip(point, bounds, tolerances)):
        return
    computational = tuple(
        (float(getattr(scene, f"{axis}_nodes64")[0]), float(getattr(scene, f"{axis}_nodes64")[-1]))
        for axis in _AXES
    )
    if all(lo <= value <= hi for value, (lo, hi) in zip(point, computational)):
        raise ValueError(f"ThinWire {wire_name!r} enters the PML outside the physical domain.")
    raise ValueError(f"ThinWire {wire_name!r} lies outside the physical domain.")


def _expanded_path(
    scene,
    wire,
    nodes: tuple[np.ndarray, ...],
) -> tuple[list[tuple[int, int, int]], list[int], int]:
    name = str(wire.name)
    points = _point_tensor(wire.points, wire_name=name).numpy()
    snap = _kind_name(getattr(wire, "snap", "strict"))
    if snap not in {"strict", "nearest"}:
        raise ValueError(f"ThinWire {name!r} snap must be 'strict' or 'nearest'.")
    snapped = []
    for point_index, point in enumerate(points):
        _domain_error(scene, point, wire_name=name)
        snapped.append(
            tuple(
                _snap_index(
                    axis_nodes,
                    float(value),
                    strict=snap == "strict",
                    label=f"ThinWire {name!r} point {point_index} {_AXES[axis_index]}",
                )
                for axis_index, (axis_nodes, value) in enumerate(zip(nodes, point))
            )
        )

    path = [snapped[0]]
    source_segments = []
    for source_segment, (start, end) in enumerate(zip(snapped, snapped[1:])):
        differing = [axis for axis in range(3) if start[axis] != end[axis]]
        if not differing:
            raise ValueError(f"ThinWire {name!r} contains a zero-length segment after snapping.")
        if len(differing) != 1:
            raise ValueError(f"ThinWire {name!r} must contain only axis-aligned segments in Phase 1.")
        axis = differing[0]
        direction = 1 if end[axis] > start[axis] else -1
        current = list(start)
        while current[axis] != end[axis]:
            current[axis] += direction
            path.append(tuple(current))
            source_segments.append(source_segment)

    if len(set(path)) != len(path):
        raise ValueError(
            f"ThinWire {name!r} repeats a grid node; loops, branches, and self-intersections are not supported in Phase 1."
        )
    return path, source_segments, len(snapped) - 1


def _geometry_signed_distance(structure, point: tuple[float, float, float]) -> torch.Tensor:
    geometry = getattr(structure, "geometry", None)
    if geometry is None or not callable(getattr(geometry, "signed_distance", None)):
        raise ValueError(
            f"Grounded PEC structure {getattr(structure, 'name', None)!r} "
            "must provide signed-distance geometry."
        )
    reference = torch.as_tensor(getattr(geometry, "position", point))
    dtype = reference.dtype if reference.is_floating_point() else torch.float64
    coordinates = torch.as_tensor(point, device=reference.device, dtype=dtype)
    return geometry.signed_distance(
        coordinates[0],
        coordinates[1],
        coordinates[2],
    )


def _endpoint_kinds(
    scene,
    wire,
    path: list[tuple[int, int, int]],
    nodes: tuple[np.ndarray, ...],
) -> tuple[str, str]:
    endpoints = getattr(wire, "endpoints", None)
    if endpoints is None:
        return ("open", "open")
    endpoints = tuple(endpoints)
    if len(endpoints) != 2:
        raise ValueError(f"ThinWire {wire.name!r} endpoints must contain exactly two entries.")
    kinds = tuple(_kind_name(endpoint) for endpoint in endpoints)
    if any(kind not in {"open", "grounded"} for kind in kinds):
        raise ValueError(
            f"ThinWire {wire.name!r} endpoints must be open or grounded in Phase 1."
        )
    tolerance = max(
        1.0e-12,
        min(float(np.min(np.diff(axis_nodes))) for axis_nodes in nodes) * 1.0e-6,
    )
    for endpoint_index, (endpoint, kind, grid_index) in enumerate(
        zip(endpoints, kinds, (path[0], path[-1]))
    ):
        if kind != "grounded":
            continue
        structure_name = getattr(endpoint, "structure", None)
        if not isinstance(structure_name, str) or not structure_name.strip():
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} "
                "must reference a named PEC structure."
            )
        matches = [
            structure
            for structure in scene.structures
            if getattr(structure, "name", None) == structure_name
        ]
        if not matches:
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} "
                f"structure {structure_name!r} does not exist."
            )
        if len(matches) != 1:
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} "
                f"structure name {structure_name!r} is not unique."
            )
        structure = matches[0]
        if not bool(getattr(structure, "enabled", True)):
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} "
                f"structure {structure_name!r} is disabled."
            )
        if not bool(getattr(getattr(structure, "material", None), "is_pec", False)):
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} "
                f"structure {structure_name!r} must be PEC."
            )
        point = tuple(float(nodes[axis][grid_index[axis]]) for axis in range(3))
        distance = _geometry_signed_distance(structure, point)
        if distance.numel() != 1 or not bool(torch.isfinite(distance).all()):
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} "
                f"structure {structure_name!r} returned an invalid signed distance."
            )
        if float(distance.detach().cpu()) > tolerance:
            raise ValueError(
                f"ThinWire {wire.name!r} grounded endpoint {endpoint_index} must lie "
                f"on or inside PEC structure {structure_name!r}."
            )
    return kinds


def _validate_boundary_contacts(scene, wire, path, nodes) -> None:
    boundary = getattr(scene, "boundary", None)
    if boundary is None:
        return
    bounds = tuple(tuple(float(value) for value in pair) for pair in scene.domain.bounds)
    for axis_index, axis in enumerate(_AXES):
        tolerance = _grid_tolerance(nodes[axis_index])
        for side_index, side in enumerate(("low", "high")):
            if _kind_name(boundary.face_kind(axis, side)) != "pec":
                continue
            face = bounds[axis_index][side_index]
            if any(
                abs(float(nodes[axis_index][grid_index[axis_index]]) - face) <= tolerance
                for grid_index in path
            ):
                raise ValueError(
                    f"ThinWire {wire.name!r} touches the PEC {axis}-{side} boundary; "
                    "Phase 1 grounding requires a named PEC structure."
                )


def _empty_tensor(device, dtype, shape) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=dtype)


def _pec_structures(scene) -> tuple[Any, ...]:
    return tuple(
        structure
        for structure in getattr(scene, "structures", ())
        if bool(getattr(structure, "enabled", True))
        and bool(getattr(getattr(structure, "material", None), "is_pec", False))
    )


def _signed_distance_value(structure, point) -> float:
    distance = _geometry_signed_distance(structure, tuple(float(value) for value in point))
    if distance.numel() != 1 or not bool(torch.isfinite(distance).all()):
        raise ValueError(
            f"PEC structure {getattr(structure, 'name', None)!r} returned an invalid signed distance."
        )
    return float(distance.detach().cpu())


def _segment_intersects_geometry(structure, start, end, tolerance: float) -> bool:
    """Conservatively test a segment against 1-Lipschitz signed-distance geometry."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    start_distance = _signed_distance_value(structure, start)
    end_distance = _signed_distance_value(structure, end)
    if start_distance <= tolerance or end_distance <= tolerance:
        return True
    stack = [(start, end, start_distance, end_distance, 0)]
    while stack:
        left, right, left_distance, right_distance, depth = stack.pop()
        interval_length = float(np.linalg.norm(right - left))
        if min(left_distance, right_distance) > interval_length + tolerance:
            continue
        midpoint = 0.5 * (left + right)
        midpoint_distance = _signed_distance_value(structure, midpoint)
        if midpoint_distance <= tolerance:
            return True
        if depth >= 24:
            # The signed-distance lower bound still cannot exclude contact at
            # sub-nanometre relative scale, so reject the ambiguous geometry.
            return True
        half_length = 0.5 * interval_length
        if min(midpoint_distance, right_distance) <= half_length + tolerance:
            stack.append((midpoint, right, midpoint_distance, right_distance, depth + 1))
        if min(left_distance, midpoint_distance) <= half_length + tolerance:
            stack.append((left, midpoint, left_distance, midpoint_distance, depth + 1))
    return False


def _validate_pec_contacts(scene, wire, path, nodes, endpoint_kinds) -> None:
    structures = _pec_structures(scene)
    if not structures:
        return
    endpoint_structures = tuple(
        getattr(endpoint, "structure", None)
        for endpoint in tuple(getattr(wire, "endpoints", ()) or ())
    )
    tolerance = max(
        1.0e-12,
        min(float(np.min(np.diff(axis_nodes))) for axis_nodes in nodes) * 1.0e-6,
    )
    positions = [
        np.asarray(
            tuple(float(nodes[axis][grid_index[axis]]) for axis in range(3)),
            dtype=np.float64,
        )
        for grid_index in path
    ]
    last_node = len(path) - 1
    for node_index, point in enumerate(positions):
        for structure in structures:
            if _signed_distance_value(structure, point) > tolerance:
                continue
            allowed = False
            for endpoint_index, path_node in enumerate((0, last_node)):
                if (
                    node_index == path_node
                    and endpoint_kinds[endpoint_index] == "grounded"
                    and endpoint_index < len(endpoint_structures)
                    and endpoint_structures[endpoint_index] == getattr(structure, "name", None)
                ):
                    allowed = True
            if not allowed:
                raise ValueError(
                    f"ThinWire {wire.name!r} overlaps PEC structure "
                    f"{getattr(structure, 'name', None)!r} away from a declared grounded endpoint."
                )
    for segment_index, (start, end) in enumerate(zip(positions, positions[1:])):
        for structure in structures:
            structure_name = getattr(structure, "name", None)
            grounded_attachment = (
                segment_index == 0
                and endpoint_kinds[0] == "grounded"
                and endpoint_structures[0] == structure_name
            ) or (
                segment_index == len(positions) - 2
                and endpoint_kinds[1] == "grounded"
                and endpoint_structures[1] == structure_name
            )
            if grounded_attachment:
                opposite = end if segment_index == 0 else start
                if _signed_distance_value(structure, opposite) <= tolerance:
                    raise ValueError(
                        f"ThinWire {wire.name!r} grounded PEC contact consumes an entire segment."
                    )
                continue
            if _segment_intersects_geometry(structure, start, end, tolerance):
                raise ValueError(
                    f"ThinWire {wire.name!r} overlaps PEC structure {structure_name!r} "
                    "away from a declared grounded endpoint."
                )


def _segment_distance(a0, a1, b0, b1) -> float:
    """Return the shortest distance between two finite line segments."""

    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    aa = float(np.dot(u, u))
    bb = float(np.dot(u, v))
    cc = float(np.dot(v, v))
    dd = float(np.dot(u, w))
    ee = float(np.dot(v, w))
    denominator = aa * cc - bb * bb
    small = np.finfo(np.float64).eps * max(aa * cc, 1.0)
    if denominator < small:
        s = 0.0
        t = 0.0 if cc == 0.0 else np.clip(ee / cc, 0.0, 1.0)
    else:
        s = np.clip((bb * ee - cc * dd) / denominator, 0.0, 1.0)
        t = np.clip((aa * ee - bb * dd) / denominator, 0.0, 1.0)
    # Clamping one parameter can move the optimum for the other.
    s = 0.0 if aa == 0.0 else np.clip((bb * t - dd) / aa, 0.0, 1.0)
    t = 0.0 if cc == 0.0 else np.clip((bb * s + ee) / cc, 0.0, 1.0)
    return float(np.linalg.norm(w + s * u - t * v))


def _validate_wire_proximity(
    node_positions: np.ndarray,
    tail: list[int],
    head: list[int],
    wire_ids: list[int],
    source_segment_ids: list[int],
    radii: torch.Tensor,
) -> float | None:
    minimum = None
    radii_cpu = radii.detach().cpu().to(dtype=torch.float64).numpy()
    for first in range(len(tail)):
        for second in range(first + 1, len(tail)):
            if (
                wire_ids[first] == wire_ids[second]
                and abs(source_segment_ids[first] - source_segment_ids[second]) <= 1
            ):
                continue
            distance = _segment_distance(
                node_positions[tail[first]],
                node_positions[head[first]],
                node_positions[tail[second]],
                node_positions[head[second]],
            )
            minimum = distance if minimum is None else min(minimum, distance)
            if distance <= radii_cpu[first] + radii_cpu[second]:
                raise ValueError("ThinWire conductors overlap or touch before a declared junction.")
    return minimum


def _validate_port_ownership(prepared_scene, target_device, shapes, targets) -> None:
    if not tuple(getattr(prepared_scene, "ports", ())):
        return
    from .ports import compile_ports

    for port in compile_ports(prepared_scene, device=target_device):
        component = _COMPONENTS.index(port.voltage_component)
        for index in port.voltage_indices.detach().cpu().tolist():
            key = (component, _flat_offset(component, tuple(index), shapes))
            if key in targets:
                raise ValueError(
                    f"Port {port.port_name!r} overlaps a ThinWire coupling edge; "
                    "wire-port binding is introduced in Phase 3."
                )


def compile_thin_wires(
    prepared_scene,
    *,
    device: str | torch.device | None = None,
) -> CompiledWireNetwork:
    """Compile Phase 1 PEC axis-aligned wires into sparse graph tensors."""

    nodes, primal, dual = _grid_arrays(prepared_scene)
    shapes = _field_shapes(nodes)
    target_device = torch.device(prepared_scene.device if device is None else device)
    wires = tuple(sorted(getattr(prepared_scene, "thin_wires", ()), key=lambda wire: str(wire.name)))
    names = tuple(str(wire.name) for wire in wires)
    if len(names) != len(set(names)):
        raise ValueError("ThinWire names must be unique.")
    coefficient_dtype = None
    for wire in wires:
        radius_dtype = _radius_dtype(wire.radius)
        coefficient_dtype = (
            radius_dtype
            if coefficient_dtype is None
            else torch.promote_types(coefficient_dtype, radius_dtype)
        )
    if coefficient_dtype is None:
        coefficient_dtype = torch.float64

    material_model = prepared_scene.compile_materials() if wires else None
    eps_components = None if material_model is None else material_model["eps_components"]
    mu_components = None if material_model is None else material_model["mu_components"]
    eps_fields = (
        None
        if eps_components is None
        else {
            axis: eps_components[axis].to(device=target_device, dtype=coefficient_dtype)
            for axis in _AXES
        }
    )
    mu_fields = (
        None
        if mu_components is None
        else {
            axis: mu_components[axis].to(device=target_device, dtype=coefficient_dtype)
            for axis in _AXES
        }
    )

    grid_fingerprint = _grid_fingerprint(prepared_scene, nodes, primal, dual)
    global_node_owner: dict[tuple[int, int, int], str] = {}
    node_grid = []
    node_wire_ids = []
    open_flags = []
    grounded_flags = []
    tail = []
    head = []
    segment_wire_ids = []
    segment_source_ids = []
    segment_axes = []
    segment_directions = []
    segment_lengths = []
    segment_radii = []
    coupling_distances = []
    radius_to_spacing = []
    inductance_per_segment = []
    capacitance_per_length = []
    local_permittivity = []
    local_permeability = []
    edge_offsets = []
    wire_node_offsets = [0]
    wire_segment_offsets = [0]

    for wire_id, wire in enumerate(wires):
        if _kind_name(wire.conductor) != "pec":
            raise ValueError(f"ThinWire {wire.name!r} must use a PEC conductor in Phase 1.")
        path, source_segments, source_count = _expanded_path(prepared_scene, wire, nodes)
        radii = _radius_tensor(
            wire.radius,
            source_segment_count=source_count,
            device=target_device,
            dtype=coefficient_dtype,
            wire_name=str(wire.name),
        )
        endpoint_kinds = _endpoint_kinds(prepared_scene, wire, path, nodes)
        _validate_boundary_contacts(prepared_scene, wire, path, nodes)
        _validate_pec_contacts(prepared_scene, wire, path, nodes, endpoint_kinds)
        local_start = len(node_grid)
        for local_node, grid_index in enumerate(path):
            owner = global_node_owner.get(grid_index)
            if owner is not None:
                raise ValueError(
                    f"ThinWire {wire.name!r} shares grid node {grid_index} with {owner!r}; junctions are Phase 2."
                )
            global_node_owner[grid_index] = str(wire.name)
            node_grid.append(grid_index)
            node_wire_ids.append(wire_id)
            open_flags.append(
                (local_node == 0 and endpoint_kinds[0] == "open")
                or (local_node == len(path) - 1 and endpoint_kinds[1] == "open")
            )
            grounded_flags.append(
                (local_node == 0 and endpoint_kinds[0] == "grounded")
                or (local_node == len(path) - 1 and endpoint_kinds[1] == "grounded")
            )

        for local_segment, (start, end, source_segment) in enumerate(
            zip(path, path[1:], source_segments)
        ):
            axis = next(index for index in range(3) if start[index] != end[index])
            direction = 1 if end[axis] > start[axis] else -1
            lower = min(start[axis], end[axis])
            edge = list(start)
            edge[axis] = lower
            length = float(primal[axis][lower])
            transverse = tuple(index for index in range(3) if index != axis)
            du = float(dual[transverse[0]][start[transverse[0]]])
            dv = float(dual[transverse[1]][start[transverse[1]]])
            distance = _bspline_coupling_distance(du, dv)
            radius = radii[source_segment]
            spacing = min(du, dv)
            ratio = radius / spacing
            distance_t = torch.as_tensor(distance, device=target_device, dtype=coefficient_dtype)
            if not bool(radius < distance_t):
                raise ValueError(
                    f"ThinWire {wire.name!r} radius must remain below the local coupling distance."
                )
            if not bool(ratio <= _MAX_RADIUS_TO_SPACING):
                raise ValueError(
                    f"ThinWire {wire.name!r} exceeds the Phase 1 a/delta_perp <= {_MAX_RADIUS_TO_SPACING} validity band."
                )
            eps_local = torch.stack(
                [0.5 * (eps_fields[name][start] + eps_fields[name][end]) for name in _AXES]
            )
            mu_local = torch.stack(
                [0.5 * (mu_fields[name][start] + mu_fields[name][end]) for name in _AXES]
            )
            if not torch.allclose(
                eps_local,
                eps_local[0].expand_as(eps_local),
                rtol=1.0e-6,
                atol=1.0e-7,
            ) or not torch.allclose(
                mu_local,
                mu_local[0].expand_as(mu_local),
                rtol=1.0e-6,
                atol=1.0e-7,
            ):
                raise NotImplementedError(
                    f"ThinWire {wire.name!r} Phase 1 requires a locally isotropic host material."
                )
            eps_r = eps_local.mean()
            mu_r = mu_local.mean()
            if not bool(torch.isfinite(eps_r)) or not bool(torch.isfinite(mu_r)):
                raise ValueError(f"ThinWire {wire.name!r} local host material must be finite.")
            if not bool(eps_r > 0.0) or not bool(mu_r > 0.0):
                raise ValueError(f"ThinWire {wire.name!r} local host material must be positive.")
            log_ratio = torch.log(distance_t) - torch.log(radius)
            if not bool(torch.isfinite(log_ratio)) or not bool(log_ratio > 0.0):
                raise ValueError(
                    f"ThinWire {wire.name!r} logarithmic self term is ill-conditioned."
                )
            l_prime = MU_0 * mu_r * log_ratio / (2.0 * math.pi)
            c_prime = MU_0 * mu_r * EPSILON_0 * eps_r / l_prime
            if (
                not bool(torch.isfinite(l_prime) & (l_prime > 0.0))
                or not bool(torch.isfinite(c_prime) & (c_prime > 0.0))
            ):
                raise ValueError(
                    f"ThinWire {wire.name!r} local line coefficients must be finite and positive."
                )
            tail.append(local_start + local_segment)
            head.append(local_start + local_segment + 1)
            segment_wire_ids.append(wire_id)
            segment_source_ids.append(source_segment)
            segment_axes.append(axis)
            segment_directions.append(direction)
            segment_lengths.append(length)
            segment_radii.append(radius)
            coupling_distances.append(distance_t)
            radius_to_spacing.append(ratio)
            inductance_per_segment.append(l_prime * length)
            capacitance_per_length.append(c_prime)
            local_permittivity.append(eps_r)
            local_permeability.append(mu_r)
            edge_offsets.append(_flat_offset(axis, tuple(edge), shapes))
        wire_node_offsets.append(len(node_grid))
        wire_segment_offsets.append(len(tail))

    node_count = len(node_grid)
    segment_count = len(tail)
    node_grid_array = np.asarray(node_grid, dtype=np.int64).reshape(node_count, 3)
    node_position_array = np.empty((node_count, 3), dtype=np.float64)
    for axis in range(3):
        if node_count:
            node_position_array[:, axis] = nodes[axis][node_grid_array[:, axis]]

    tail_t = torch.as_tensor(tail, device=target_device, dtype=torch.int64)
    head_t = torch.as_tensor(head, device=target_device, dtype=torch.int64)
    if segment_count:
        radius_t = torch.stack(segment_radii)
        length_t = torch.as_tensor(segment_lengths, device=target_device, dtype=coefficient_dtype)
        distance_t = torch.stack(coupling_distances)
        ratio_t = torch.stack(radius_to_spacing)
        inductance_t = torch.stack(inductance_per_segment)
        capacitance_t = torch.stack(capacitance_per_length)
        local_permittivity_t = torch.stack(local_permittivity)
        local_permeability_t = torch.stack(local_permeability)
        contributions = 0.5 * capacitance_t * length_t
        node_capacitance = torch.zeros(node_count, device=target_device, dtype=coefficient_dtype)
        node_capacitance = node_capacitance.index_add(
            0,
            torch.cat((tail_t, head_t)),
            torch.cat((contributions, contributions)),
        )
    else:
        radius_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        length_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        distance_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        ratio_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        inductance_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        capacitance_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        local_permittivity_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        local_permeability_t = _empty_tensor(target_device, coefficient_dtype, (0,))
        node_capacitance = _empty_tensor(target_device, coefficient_dtype, (0,))

    minimum_neighbor_distance = _validate_wire_proximity(
        node_position_array,
        tail,
        head,
        segment_wire_ids,
        segment_source_ids,
        radius_t,
    )

    incidence = []
    for segment, (tail_node, head_node) in enumerate(zip(tail, head)):
        incidence.append((tail_node, segment, 1.0))
        incidence.append((head_node, segment, -1.0))
    incidence.sort(key=lambda entry: (entry[0], entry[1]))
    node_counts = np.bincount([entry[0] for entry in incidence], minlength=node_count)
    node_offsets = np.concatenate(([0], np.cumsum(node_counts, dtype=np.int64)))

    sampling_components = np.asarray(segment_axes, dtype=np.int32)
    sampling_offsets = np.asarray(edge_offsets, dtype=np.int64)
    sampling_weights = np.asarray(segment_directions, dtype=np.float64) * np.asarray(
        segment_lengths, dtype=np.float64
    )
    sparse_target_owner = {}
    for segment, (component, offset) in enumerate(
        zip(sampling_components, sampling_offsets)
    ):
        key = (int(component), int(offset))
        previous = sparse_target_owner.get(key)
        if previous is not None and segment_wire_ids[previous] != segment_wire_ids[segment]:
            raise ValueError(
                "ThinWire conductors overlap one compiled sparse Yee coupling target."
            )
        sparse_target_owner[key] = segment
    deposition = sorted(
        (
            int(component),
            int(offset),
            segment,
            float(weight),
        )
        for segment, (component, offset, weight) in enumerate(
            zip(sampling_components, sampling_offsets, sampling_weights)
        )
    )
    target_components = []
    target_offsets = []
    previous_key = None
    for component, offset, _segment, _weight in deposition:
        key = (component, offset)
        if key != previous_key:
            target_components.append(component)
            target_offsets.append(offset)
            previous_key = key
    if target_components:
        counts = []
        for component, offset in zip(target_components, target_offsets):
            counts.append(sum(1 for entry in deposition if entry[:2] == (component, offset)))
        edge_group_offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).tolist()
    else:
        edge_group_offsets = [0]

    grid_entries = (
        ("node_grid", node_grid_array),
        ("tail", np.asarray(tail, dtype=np.int64)),
        ("head", np.asarray(head, dtype=np.int64)),
        ("wire_ids", np.asarray(segment_wire_ids, dtype=np.int64)),
        ("radius", radius_t.detach().cpu().to(dtype=torch.float64).numpy()),
        (
            "local_permittivity",
            local_permittivity_t.detach().cpu().to(dtype=torch.float64).numpy(),
        ),
        (
            "local_permeability",
            local_permeability_t.detach().cpu().to(dtype=torch.float64).numpy(),
        ),
        ("open_endpoints", np.asarray(open_flags, dtype=np.bool_)),
        ("grounded", np.asarray(grounded_flags, dtype=np.bool_)),
    )
    compile_digest = hashlib.sha256()
    compile_digest.update(grid_fingerprint.encode("ascii"))
    for name in names:
        encoded_name = name.encode("utf8")
        compile_digest.update(len(encoded_name).to_bytes(8, byteorder="little"))
        compile_digest.update(encoded_name)
    compile_digest.update(_hash_arrays(*grid_entries).encode("ascii"))
    compile_fingerprint = compile_digest.hexdigest()
    cache_enabled = not any(
        tensor.requires_grad for tensor in (radius_t, local_permittivity_t, local_permeability_t)
    )
    metadata = MappingProxyType(
        {
            "grid_fingerprint": grid_fingerprint,
            "compile_fingerprint": compile_fingerprint,
            "component_order": _COMPONENTS,
            "max_radius_to_spacing": _MAX_RADIUS_TO_SPACING,
            "wire_node_ranges": tuple(
                (name, wire_node_offsets[index], wire_node_offsets[index + 1])
                for index, name in enumerate(names)
            ),
            "wire_segment_ranges": tuple(
                (name, wire_segment_offsets[index], wire_segment_offsets[index + 1])
                for index, name in enumerate(names)
            ),
            "validity": MappingProxyType(
                {
                    "phase": 1,
                    "conductor": "pec",
                    "topology": "unbranched_open_path",
                    "coupling_kernel": "BS1xBS1",
                    "max_radius_to_spacing": _MAX_RADIUS_TO_SPACING,
                    "minimum_neighbor_distance": minimum_neighbor_distance,
                    "proximity_criterion": "physical_radius_and_sparse_target_ownership",
                }
            ),
            "cache_enabled": cache_enabled,
        }
    )

    target_keys = set(zip(target_components, target_offsets))
    _validate_port_ownership(prepared_scene, target_device, shapes, target_keys)

    cache_key = (str(target_device), compile_fingerprint)
    if cache_enabled:
        cache = getattr(prepared_scene, "_thin_wire_network_cache", None)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

    network = CompiledWireNetwork(
        wire_names=names,
        field_shapes=shapes,
        node_positions=torch.as_tensor(node_position_array, device=target_device, dtype=torch.float64),
        node_grid_indices=torch.as_tensor(node_grid_array, device=target_device, dtype=torch.int64),
        node_ids=torch.arange(node_count, device=target_device, dtype=torch.int64),
        node_wire_ids=torch.as_tensor(node_wire_ids, device=target_device, dtype=torch.int64),
        wire_node_offsets=torch.as_tensor(wire_node_offsets, device=target_device, dtype=torch.int64),
        node_offsets=torch.as_tensor(node_offsets, device=target_device, dtype=torch.int64),
        node_segments=torch.as_tensor(
            [entry[1] for entry in incidence], device=target_device, dtype=torch.int64
        ),
        node_signs=torch.as_tensor(
            [int(entry[2]) for entry in incidence], device=target_device, dtype=torch.int32
        ),
        node_capacitance=node_capacitance,
        open_endpoints=torch.as_tensor(open_flags, device=target_device, dtype=torch.bool),
        grounded=torch.as_tensor(grounded_flags, device=target_device, dtype=torch.bool),
        segment_ids=torch.arange(segment_count, device=target_device, dtype=torch.int64),
        segment_wire_ids=torch.as_tensor(segment_wire_ids, device=target_device, dtype=torch.int64),
        wire_segment_offsets=torch.as_tensor(wire_segment_offsets, device=target_device, dtype=torch.int64),
        tail=tail_t,
        head=head_t,
        segment_axes=torch.as_tensor(segment_axes, device=target_device, dtype=torch.int32),
        segment_directions=torch.as_tensor(segment_directions, device=target_device, dtype=torch.int8),
        radius=radius_t,
        length=length_t,
        coupling_distance=distance_t,
        radius_to_spacing=ratio_t,
        inductance=inductance_t,
        capacitance_per_length=capacitance_t,
        segment_offsets=torch.arange(segment_count + 1, device=target_device, dtype=torch.int64),
        edge_components=torch.as_tensor(sampling_components, device=target_device, dtype=torch.int32),
        edge_offsets=torch.as_tensor(sampling_offsets, device=target_device, dtype=torch.int64),
        weights=torch.as_tensor(sampling_weights, device=target_device, dtype=coefficient_dtype),
        edge_group_offsets=torch.as_tensor(edge_group_offsets, device=target_device, dtype=torch.int64),
        target_components=torch.as_tensor(target_components, device=target_device, dtype=torch.int32),
        target_offsets=torch.as_tensor(target_offsets, device=target_device, dtype=torch.int64),
        contribution_segments=torch.as_tensor(
            [entry[2] for entry in deposition], device=target_device, dtype=torch.int64
        ),
        contribution_weights=torch.as_tensor(
            [entry[3] for entry in deposition], device=target_device, dtype=coefficient_dtype
        ),
        metadata=metadata,
    )
    if cache_enabled:
        cache = getattr(prepared_scene, "_thin_wire_network_cache", None)
        if cache is None:
            cache = {}
            prepared_scene._thin_wire_network_cache = cache
        cache[cache_key] = network
    return network


def compile_wire_monitors(
    prepared_scene,
    network: CompiledWireNetwork | None = None,
    monitors=None,
    *,
    device: str | torch.device | None = None,
) -> tuple[CompiledWireMonitor, ...]:
    """Resolve wire-monitor names to contiguous compiled graph ranges."""

    if network is None:
        network = compile_thin_wires(prepared_scene, device=device)
    elif device is not None and network.device != torch.device(device):
        raise ValueError("Wire monitor device must match the compiled wire network device.")
    selected = tuple(getattr(prepared_scene, "monitors", ()) if monitors is None else monitors)
    name_to_id = {name: index for index, name in enumerate(network.wire_names)}
    compiled = []
    for monitor in selected:
        if not hasattr(monitor, "wire"):
            continue
        wire_name = str(monitor.wire)
        if wire_name not in name_to_id:
            raise ValueError(f"Wire monitor {monitor.name!r} references unknown wire {wire_name!r}.")
        wire_id = name_to_id[wire_name]
        node_start = int(network.wire_node_offsets[wire_id])
        node_end = int(network.wire_node_offsets[wire_id + 1])
        segment_start = int(network.wire_segment_offsets[wire_id])
        segment_end = int(network.wire_segment_offsets[wire_id + 1])
        quantities = tuple(str(value) for value in getattr(monitor, "quantities", ("current", "charge")))
        frequencies = tuple(float(value) for value in getattr(monitor, "frequencies", ()))
        if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
            raise ValueError(f"Wire monitor {monitor.name!r} frequencies must be positive and finite.")
        compiled.append(
            CompiledWireMonitor(
                name=str(monitor.name),
                wire_name=wire_name,
                wire_id=wire_id,
                node_indices=torch.arange(node_start, node_end, device=network.device, dtype=torch.int64),
                segment_indices=torch.arange(
                    segment_start, segment_end, device=network.device, dtype=torch.int64
                ),
                quantities=quantities,
                frequencies=frequencies,
            )
        )
    return tuple(compiled)


__all__ = [
    "CompiledWireMonitor",
    "CompiledWireNetwork",
    "compile_thin_wires",
    "compile_wire_monitors",
]
