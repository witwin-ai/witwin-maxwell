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
    """Device-native sparse graph and conservative Yee coupling for thin wires."""

    wire_names: tuple[str, ...]
    field_shapes: tuple[tuple[int, int, int], ...]
    node_positions: torch.Tensor
    node_grid_indices: torch.Tensor
    node_ids: torch.Tensor
    node_wire_ids: torch.Tensor
    wire_node_offsets: torch.Tensor
    wire_node_indices: torch.Tensor
    wire_source_point_offsets: torch.Tensor
    source_point_node_ids: torch.Tensor
    junction_names: tuple[str, ...]
    junction_node_ids: torch.Tensor
    node_offsets: torch.Tensor
    node_segments: torch.Tensor
    node_signs: torch.Tensor
    node_capacitance: torch.Tensor
    open_endpoints: torch.Tensor
    grounded: torch.Tensor
    segment_ids: torch.Tensor
    segment_wire_ids: torch.Tensor
    segment_source_ids: torch.Tensor
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
    local_permittivity: torch.Tensor
    local_permeability: torch.Tensor
    segment_offsets: torch.Tensor
    fragment_offsets: torch.Tensor
    fragment_segment_ids: torch.Tensor
    fragment_cell_indices: torch.Tensor
    fragment_lengths: torch.Tensor
    edge_components: torch.Tensor
    edge_offsets: torch.Tensor
    weights: torch.Tensor
    edge_group_offsets: torch.Tensor
    target_components: torch.Tensor
    target_offsets: torch.Tensor
    contribution_segments: torch.Tensor
    contribution_weights: torch.Tensor
    port_binding_names: tuple[str, ...]
    port_binding_kinds: tuple[str, ...]
    port_negative_node_ids: torch.Tensor
    port_positive_node_ids: torch.Tensor
    port_gap_offsets: torch.Tensor
    port_gap_edge_components: torch.Tensor
    port_gap_edge_offsets: torch.Tensor
    port_gap_weights: torch.Tensor
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


def _point_tensor(
    points,
    *,
    wire_name: str,
    allow_trainable: bool = False,
) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        if points.requires_grad and not allow_trainable:
            raise ValueError(
                f"ThinWire {wire_name!r} trainable points are discrete fixed-stencil "
                "compilation inputs unless snap='continuous'."
            )
        tensor = points
    else:
        for point in points:
            for coordinate in point:
                if (
                    isinstance(coordinate, torch.Tensor)
                    and coordinate.requires_grad
                    and not allow_trainable
                ):
                    raise ValueError(
                        f"ThinWire {wire_name!r} trainable points are discrete fixed-stencil "
                        "compilation inputs unless snap='continuous'."
                    )
        tensor = torch.as_tensor(points)
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"ThinWire {wire_name!r} points must be real coordinates.")
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] < 2:
        raise ValueError(f"ThinWire {wire_name!r} points must have shape (N, 3) with N >= 2.")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"ThinWire {wire_name!r} points must be finite.")
    return tensor


def _point_dtype(value) -> torch.dtype:
    if not isinstance(value, torch.Tensor):
        return torch.float64
    if not value.is_floating_point() or value.is_complex():
        raise TypeError("ThinWire points must be a real floating-point tensor or coordinates.")
    return value.dtype


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


def _bspline_coupling_distance_tensor(du: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
    """Differentiable quadrature for the BS1 x BS1 geometric-mean radius."""

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(48)
    u = torch.as_tensor(
        0.5 * (quadrature_nodes + 1.0), device=du.device, dtype=du.dtype
    )
    weights = torch.as_tensor(
        0.5 * quadrature_weights, device=du.device, dtype=du.dtype
    )
    axial = du * u
    axial2 = axial.square()
    dv2 = dv.square()
    total = axial2 + dv2
    integral_log = (
        torch.log(total)
        - 2.0
        + 2.0 * axial * torch.atan(dv / axial) / dv
    )
    integral_v_log = (
        total * torch.log(total)
        - total
        - axial2 * torch.log(axial2)
        + axial2
    ) / (2.0 * dv2)
    integrate_v = 0.5 * (integral_log - integral_v_log)
    return torch.exp(4.0 * torch.sum(weights * (1.0 - u) * integrate_v))


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
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    list[tuple[float, float, float]],
    list[torch.Tensor],
    list[int],
    int,
    list[int],
]:
    name = str(wire.name)
    snap = _kind_name(getattr(wire, "snap", "strict"))
    if snap not in {"continuous", "strict", "nearest"}:
        raise ValueError(
            f"ThinWire {name!r} snap must be 'continuous', 'strict', or 'nearest'."
        )
    point_tensor = _point_tensor(
        wire.points,
        wire_name=name,
        allow_trainable=snap == "continuous",
    ).to(device=device, dtype=dtype)
    points = point_tensor.detach().cpu().to(dtype=torch.float64).numpy()
    resolved_points = []
    for point_index, point in enumerate(points):
        _domain_error(scene, point, wire_name=name)
        if snap == "continuous":
            resolved_points.append(tuple(float(value) for value in point))
        else:
            snapped_indices = tuple(
                _snap_index(
                    axis_nodes,
                    float(value),
                    strict=snap == "strict",
                    label=f"ThinWire {name!r} point {point_index} {_AXES[axis_index]}",
                )
                for axis_index, (axis_nodes, value) in enumerate(zip(nodes, point))
            )
            resolved_points.append(
                tuple(
                    float(nodes[axis][index])
                    for axis, index in enumerate(snapped_indices)
                )
            )

    source_tensors = (
        [point_tensor[index] for index in range(point_tensor.shape[0])]
        if snap == "continuous"
        else [
            torch.as_tensor(point, device=device, dtype=dtype)
            for point in resolved_points
        ]
    )
    path = [resolved_points[0]]
    path_tensors = [source_tensors[0]]
    source_segments = []
    source_point_path_indices = [0]
    for source_segment, (start, end) in enumerate(
        zip(resolved_points, resolved_points[1:])
    ):
        if start == end:
            raise ValueError(f"ThinWire {name!r} contains a zero-length segment after snapping.")
        # A user polyline span is one physical circuit segment. Grid-plane
        # clipping is coupling-only and is performed later when constructing G.
        # Keeping those two topologies separate makes I/q ownership independent
        # of grid orientation and preserves a fixed state shape for coordinates.
        path.append(end)
        path_tensors.append(source_tensors[source_segment + 1])
        source_segments.append(source_segment)
        source_point_path_indices.append(len(path) - 1)

    closed = path[0] == path[-1]
    unique_path = path[:-1] if closed else path
    if len(set(unique_path)) != len(unique_path):
        raise ValueError(
            f"ThinWire {name!r} repeats an internal grid node; only a first-to-last closed loop is valid."
        )
    return (
        path,
        path_tensors,
        source_segments,
        len(resolved_points) - 1,
        source_point_path_indices,
    )


def _point_grid_index(
    point: tuple[float, float, float],
    nodes: tuple[np.ndarray, ...],
) -> tuple[int, int, int]:
    """Return the nearest grid-node index used only for ownership metadata."""

    return tuple(
        int(np.argmin(np.abs(axis_nodes - coordinate)))
        for axis_nodes, coordinate in zip(nodes, point)
    )


def _fragment_cell(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_tensor: torch.Tensor,
    end_tensor: torch.Tensor,
    nodes: tuple[np.ndarray, ...],
    primal: tuple[np.ndarray, ...],
) -> tuple[tuple[int, int, int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve one grid-plane-clipped fragment to a cell and local coordinates."""

    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    midpoint = 0.5 * (start_array + end_array)
    cell = tuple(
        int(np.clip(np.searchsorted(axis_nodes, value, side="right") - 1, 0, len(axis_nodes) - 2))
        for axis_nodes, value in zip(nodes, midpoint)
    )

    origin = torch.as_tensor(
        [nodes[axis][cell[axis]] for axis in range(3)],
        device=start_tensor.device,
        dtype=start_tensor.dtype,
    )
    spacing = torch.as_tensor(
        [primal[axis][cell[axis]] for axis in range(3)],
        device=start_tensor.device,
        dtype=start_tensor.dtype,
    )

    def local(point: torch.Tensor) -> torch.Tensor:
        return (point - origin) / spacing

    return (
        cell,
        local(start_tensor),
        local(end_tensor),
        local(0.5 * (start_tensor + end_tensor)),
    )


def _transverse_bits(component: int) -> tuple[tuple[int, int, int], ...]:
    transverse = tuple(axis for axis in range(3) if axis != component)
    return tuple(
        tuple(
            bit_u if axis == transverse[0] else bit_v if axis == transverse[1] else 0
            for axis in range(3)
        )
        for bit_u in (0, 1)
        for bit_v in (0, 1)
    )


def _point_edge_weights(
    component: int,
    local: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    transverse = tuple(axis for axis in range(3) if axis != component)
    return tuple(
        (local[transverse[0]] if bits[transverse[0]] else 1.0 - local[transverse[0]])
        * (local[transverse[1]] if bits[transverse[1]] else 1.0 - local[transverse[1]])
        for bits in _transverse_bits(component)
    )


def _fragment_edge_coupling(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_tensor: torch.Tensor,
    end_tensor: torch.Tensor,
    nodes: tuple[np.ndarray, ...],
    primal: tuple[np.ndarray, ...],
    shapes: tuple[tuple[int, int, int], ...],
) -> tuple[
    tuple[int, int, int],
    torch.Tensor,
    list[tuple[int, int, torch.Tensor]],
]:
    """Integrate tensor-product Yee edge basis functions along one fragment."""

    cell, local_start, local_end, local_midpoint = _fragment_cell(
        start,
        end,
        start_tensor,
        end_tensor,
        nodes,
        primal,
    )
    displacement = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    tensor_displacement = end_tensor - start_tensor
    samples: list[tuple[int, int, torch.Tensor]] = []
    for component in range(3):
        if displacement[component] == 0.0:
            continue
        transverse = tuple(axis for axis in range(3) if axis != component)
        for bits in _transverse_bits(component):
            first_start = (
                local_start[transverse[0]]
                if bits[transverse[0]]
                else 1.0 - local_start[transverse[0]]
            )
            second_start = (
                local_start[transverse[1]]
                if bits[transverse[1]]
                else 1.0 - local_start[transverse[1]]
            )
            first_end = (
                local_end[transverse[0]]
                if bits[transverse[0]]
                else 1.0 - local_end[transverse[0]]
            )
            second_end = (
                local_end[transverse[1]]
                if bits[transverse[1]]
                else 1.0 - local_end[transverse[1]]
            )
            first_slope = first_end - first_start
            second_slope = second_end - second_start
            basis_integral = (
                first_start * second_start
                + 0.5
                * (first_start * second_slope + second_start * first_slope)
                + first_slope * second_slope / 3.0
            )
            weight = tensor_displacement[component] * basis_integral
            if abs(float(weight.detach().cpu())) <= (
                64.0 * np.finfo(np.float64).eps * abs(displacement[component])
            ):
                continue
            edge = tuple(cell[axis] + bits[axis] for axis in range(3))
            samples.append((component, _flat_offset(component, edge, shapes), weight))
    if not samples:
        raise RuntimeError("Thin-wire fragment produced no conservative Yee coupling samples.")
    return cell, local_midpoint, samples


def _segment_coupling_fragments(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_tensor: torch.Tensor,
    end_tensor: torch.Tensor,
    nodes: tuple[np.ndarray, ...],
    primal: tuple[np.ndarray, ...],
    shapes: tuple[tuple[int, int, int], ...],
    *,
    continuous: bool,
) -> list[
    tuple[
        torch.Tensor,
        tuple[int, int, int],
        torch.Tensor,
        list[tuple[int, int, torch.Tensor]],
    ]
]:
    """Split one circuit segment into cell-local conservative coupling fragments."""

    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    delta = end_array - start_array
    crossings: list[tuple[float, int | None, float | None]] = [
        (0.0, None, None),
        (1.0, None, None),
    ]
    for axis, axis_nodes in enumerate(nodes):
        if delta[axis] == 0.0:
            continue
        lower = min(start_array[axis], end_array[axis])
        upper = max(start_array[axis], end_array[axis])
        tolerance = _grid_tolerance(axis_nodes)
        interior = axis_nodes[
            (axis_nodes > lower + tolerance) & (axis_nodes < upper - tolerance)
        ]
        crossings.extend(
            (
                float((coordinate - start_array[axis]) / delta[axis]),
                axis,
                float(coordinate),
            )
            for coordinate in interior
        )
    crossings.sort(key=lambda entry: (entry[0], -1 if entry[1] is None else entry[1]))
    crossing_tolerance = 256.0 * np.finfo(np.float64).eps
    unique = [crossings[0]]
    for crossing in crossings[1:]:
        if crossing[0] - unique[-1][0] > crossing_tolerance * max(
            1.0, abs(crossing[0]), abs(unique[-1][0])
        ):
            unique.append(crossing)
        elif (
            continuous
            and 0.0 < crossing[0] < 1.0
            and crossing[1] != unique[-1][1]
        ):
            raise ValueError(
                "ThinWire snap='continuous' crosses multiple grid planes at one point; "
                "that stencil is not differentiable. Perturb the centerline or use a "
                "discrete snap policy."
            )

    tensor_delta = end_tensor - start_tensor
    points = [start]
    point_tensors = [start_tensor]
    for fraction_value, crossing_axis, crossing_coordinate in unique[1:]:
        if crossing_axis is None:
            point = end
            point_tensor = end_tensor
        else:
            point_array = start_array + fraction_value * delta
            point = tuple(float(value) for value in point_array)
            plane = torch.as_tensor(
                crossing_coordinate,
                device=start_tensor.device,
                dtype=start_tensor.dtype,
            )
            fraction = (plane - start_tensor[crossing_axis]) / tensor_delta[
                crossing_axis
            ]
            point_tensor = torch.stack(
                tuple(
                    plane
                    if axis == crossing_axis
                    else start_tensor[axis] + fraction * tensor_delta[axis]
                    for axis in range(3)
                )
            )
        points.append(point)
        point_tensors.append(point_tensor)

    fragments = []
    for left, right, left_tensor, right_tensor in zip(
        points, points[1:], point_tensors, point_tensors[1:]
    ):
        cell, local_midpoint, samples = _fragment_edge_coupling(
            left,
            right,
            left_tensor,
            right_tensor,
            nodes,
            primal,
            shapes,
        )
        fragments.append(
            (
                torch.linalg.vector_norm(right_tensor - left_tensor),
                cell,
                local_midpoint,
                samples,
            )
        )
    return fragments


def _local_transverse_spacing(
    *,
    cell: tuple[int, int, int],
    local_midpoint: torch.Tensor,
    tangent: torch.Tensor,
    dual: tuple[np.ndarray, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    local_spacing = torch.stack(
        [
            (1.0 - local_midpoint[axis])
            * torch.as_tensor(
                dual[axis][cell[axis]],
                device=local_midpoint.device,
                dtype=local_midpoint.dtype,
            )
            + local_midpoint[axis]
            * torch.as_tensor(
                dual[axis][cell[axis] + 1],
                device=local_midpoint.device,
                dtype=local_midpoint.dtype,
            )
            for axis in range(3)
        ]
    )
    projector = torch.eye(
        3, device=local_midpoint.device, dtype=local_midpoint.dtype
    ) - torch.outer(tangent, tangent)
    projected_metric = projector @ torch.diag(local_spacing.square()) @ projector
    eigenvalues = torch.linalg.eigvalsh(projected_metric)
    transverse = torch.sqrt(torch.clamp_min(eigenvalues[-2:], torch.finfo(eigenvalues.dtype).tiny))
    return transverse[0], transverse[1]


def _interpolate_material_component(
    field: torch.Tensor,
    component: int,
    cell: tuple[int, int, int],
    local_midpoint: torch.Tensor,
) -> torch.Tensor:
    values = []
    for bits, weight in zip(
        _transverse_bits(component),
        _point_edge_weights(component, local_midpoint),
    ):
        edge = tuple(cell[axis] + bits[axis] for axis in range(3))
        values.append(weight * field[edge])
    return torch.stack(values).sum()


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


def _endpoint_specs(
    scene,
    wire,
    path: list[tuple[float, float, float]],
    nodes: tuple[np.ndarray, ...],
) -> tuple[tuple[str, str | None], tuple[str, str | None]]:
    endpoints = getattr(wire, "endpoints", None)
    endpoints = tuple(endpoints or ())
    if not endpoints:
        if path[0] != path[-1]:
            return (("open", None), ("open", None))
        closure_name = f"__closed__:{wire.name}"
        return (("node", closure_name), ("node", closure_name))
    if len(endpoints) != 2:
        raise ValueError(f"ThinWire {wire.name!r} endpoints must contain exactly two entries.")
    kinds = tuple(_kind_name(endpoint) for endpoint in endpoints)
    if any(kind not in {"open", "grounded", "node"} for kind in kinds):
        raise ValueError(
            f"ThinWire {wire.name!r} endpoints must be open, grounded, or named nodes."
        )
    tolerance = max(
        1.0e-12,
        min(float(np.min(np.diff(axis_nodes))) for axis_nodes in nodes) * 1.0e-6,
    )
    resolved_node_names: dict[int, str] = {}
    for endpoint_index, (endpoint, kind, point) in enumerate(
        zip(endpoints, kinds, (path[0], path[-1]))
    ):
        if kind == "node":
            node_name = getattr(endpoint, "node_name", None)
            if not isinstance(node_name, str) or not node_name.strip():
                raise ValueError(
                    f"ThinWire {wire.name!r} named endpoint {endpoint_index} must have a node name."
                )
            resolved_node_name = node_name.strip()
            if resolved_node_name.startswith("__closed__:"):
                raise ValueError(
                    f"ThinWire {wire.name!r} named endpoint {endpoint_index} uses the reserved "
                    "'__closed__:' internal loop namespace."
                )
            resolved_node_names[endpoint_index] = resolved_node_name
            continue
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
    return tuple(
        (
            kind,
            resolved_node_names[index] if kind == "node" else None,
        )
        for index, kind in enumerate(kinds)
    )


def _validate_boundary_contacts(scene, wire, path, nodes) -> None:
    boundary = getattr(scene, "boundary", None)
    if boundary is None:
        return
    bounds = tuple(tuple(float(value) for value in pair) for pair in scene.domain.bounds)
    for axis_index, axis in enumerate(_AXES):
        tolerance = _grid_tolerance(nodes[axis_index])
        low_kind = _kind_name(boundary.face_kind(axis, "low"))
        high_kind = _kind_name(boundary.face_kind(axis, "high"))
        if "bloch" in {low_kind, high_kind}:
            raise NotImplementedError(
                f"ThinWire {wire.name!r} with a Bloch boundary requires phase-aware wire "
                "topology, which is not yet supported."
            )
        for side_index, side in enumerate(("low", "high")):
            kind = _kind_name(boundary.face_kind(axis, side))
            face = bounds[axis_index][side_index]
            touches = any(abs(float(point[axis_index]) - face) <= tolerance for point in path)
            if kind == "pec" and touches:
                raise ValueError(
                    f"ThinWire {wire.name!r} touches the PEC {axis}-{side} boundary; "
                    "Thin-wire grounding requires a named PEC structure."
                )
            if kind == "periodic" and touches:
                raise ValueError(
                    f"ThinWire {wire.name!r} touches the periodic {axis}-{side} boundary. "
                    "Wire paths use absolute coordinates inside one fundamental cell; "
                    "implicit periodic wrap is not supported."
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
    positions = [np.asarray(point, dtype=np.float64) for point in path]
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
                and source_segment_ids[first] == source_segment_ids[second]
            ) or ({tail[first], head[first]} & {tail[second], head[second]}):
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


def _validate_unique_physical_segments(
    node_positions: np.ndarray,
    tail: list[int],
    head: list[int],
) -> None:
    seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for tail_node, head_node in zip(tail, head):
        endpoints = sorted(
            (
                tuple(float(value) for value in node_positions[tail_node]),
                tuple(float(value) for value in node_positions[head_node]),
            )
        )
        key = (endpoints[0], endpoints[1])
        if key in seen:
            raise ValueError("ThinWire conductors overlap on one physical fragment.")
        seen.add(key)


def _validate_port_ownership(prepared_scene, target_device, shapes, targets) -> None:
    field_ports = tuple(
        port
        for port in getattr(prepared_scene, "ports", ())
        if getattr(port, "wire_binding", None) is None
    )
    if not field_ports:
        return
    from .ports import compile_ports

    for port in compile_ports(prepared_scene, field_ports, device=target_device):
        component = _COMPONENTS.index(port.voltage_component)
        for index in port.voltage_indices.detach().cpu().tolist():
            key = (component, _flat_offset(component, tuple(index), shapes))
            if key in targets:
                raise ValueError(
                    f"Port {port.port_name!r} overlaps a ThinWire coupling edge; "
                    "wire-port binding is introduced in Phase 3."
                )


def _resolve_port_bindings(prepared_scene, wires) -> tuple[dict[str, Any], ...]:
    wire_ids = {str(wire.name): index for index, wire in enumerate(wires)}
    point_counts = {
        str(wire.name): int(
            _point_tensor(
                wire.points,
                wire_name=str(wire.name),
                allow_trainable=True,
            ).shape[0]
        )
        for wire in wires
    }
    records = []
    seen_bindings = set()
    seen_gap_segments = set()
    ports = sorted(
        (
            port
            for port in getattr(prepared_scene, "ports", ())
            if getattr(port, "wire_binding", None) is not None
        ),
        key=lambda port: str(port.name),
    )
    for port in ports:
        binding = port.wire_binding
        kind = _kind_name(binding)
        if kind not in {"nodes", "gap"}:
            raise ValueError(
                f"Port {port.name!r} has unsupported wire binding kind {kind!r}."
            )
        references = []
        for terminal, reference in (
            ("negative", binding.negative),
            ("positive", binding.positive),
        ):
            wire_name = str(reference.wire)
            if wire_name not in wire_ids:
                raise ValueError(
                    f"Port {port.name!r} {terminal} wire reference names unknown "
                    f"ThinWire {wire_name!r}."
                )
            point = int(reference.point)
            if point < 0:
                point += point_counts[wire_name]
            if point < 0 or point >= point_counts[wire_name]:
                raise ValueError(
                    f"Port {port.name!r} {terminal} source-point index {point} is out of "
                    f"range for ThinWire {wire_name!r}."
                )
            references.append((wire_name, point))
        negative, positive = references
        binding_key = (kind, tuple(sorted((negative, positive))))
        if binding_key in seen_bindings:
            raise ValueError(
                f"Port {port.name!r} duplicates an existing thin-wire binding."
            )
        seen_bindings.add(binding_key)

        source_wire = None
        source_segment = None
        orientation = 0
        if kind == "gap":
            if negative[0] != positive[0]:
                raise ValueError(
                    f"Port {port.name!r} gap terminals must reference one ThinWire."
                )
            if abs(positive[1] - negative[1]) != 1:
                raise ValueError(
                    f"Port {port.name!r} gap terminals must be adjacent source points."
                )
            source_wire = negative[0]
            source_segment = min(negative[1], positive[1])
            orientation = 1 if positive[1] > negative[1] else -1
            gap_key = (source_wire, source_segment)
            if gap_key in seen_gap_segments:
                raise ValueError(
                    f"Port {port.name!r} duplicates a feed gap on ThinWire "
                    f"{source_wire!r} source segment {source_segment}."
                )
            seen_gap_segments.add(gap_key)
        records.append(
            {
                "port_name": str(port.name),
                "kind": kind,
                "negative_wire": negative[0],
                "negative_point": negative[1],
                "positive_wire": positive[0],
                "positive_point": positive[1],
                "source_wire": source_wire,
                "source_segment": source_segment,
                "orientation": orientation,
            }
        )
    return tuple(records)


def compile_thin_wires(
    prepared_scene,
    *,
    device: str | torch.device | None = None,
) -> CompiledWireNetwork:
    """Compile PEC wire graphs into sparse device tensors."""

    nodes, primal, dual = _grid_arrays(prepared_scene)
    shapes = _field_shapes(nodes)
    target_device = torch.device(prepared_scene.device if device is None else device)
    wires = tuple(sorted(getattr(prepared_scene, "thin_wires", ()), key=lambda wire: str(wire.name)))
    names = tuple(str(wire.name) for wire in wires)
    if len(names) != len(set(names)):
        raise ValueError("ThinWire names must be unique.")
    port_binding_records = _resolve_port_bindings(prepared_scene, wires)
    gap_segments = {
        (record["source_wire"], record["source_segment"])
        for record in port_binding_records
        if record["kind"] == "gap"
    }
    coefficient_dtype = None
    for wire in wires:
        radius_dtype = _radius_dtype(wire.radius)
        wire_dtype = radius_dtype
        if _kind_name(getattr(wire, "snap", "strict")) == "continuous":
            wire_dtype = torch.promote_types(wire_dtype, _point_dtype(wire.points))
        coefficient_dtype = (
            wire_dtype
            if coefficient_dtype is None
            else torch.promote_types(coefficient_dtype, wire_dtype)
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
    wire_records = []
    occurrences: dict[
        tuple[float, float, float],
        list[tuple[int, int, tuple[str, str | None] | None]],
    ] = {}
    position_tensor_occurrences: dict[
        tuple[float, float, float], list[torch.Tensor]
    ] = {}
    named_coordinates: dict[str, tuple[float, float, float]] = {}
    named_occurrences: dict[str, int] = {}
    internal_closed_names: set[str] = set()
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
    segment_offsets = [0]
    fragment_offsets = [0]
    fragment_segment_ids = []
    fragment_cell_indices = []
    fragment_lengths = []
    sampling_components = []
    sampling_offsets = []
    sampling_weights = []
    wire_node_offsets = [0]
    wire_node_indices = []
    wire_source_point_offsets = [0]
    source_point_positions = []
    wire_segment_offsets = [0]
    gap_samples: dict[
        tuple[str, int], list[tuple[int, int, torch.Tensor]]
    ] = {}

    for wire_id, wire in enumerate(wires):
        if _kind_name(wire.conductor) != "pec":
            raise NotImplementedError(
                f"ThinWire {wire.name!r} uses a finite conductor. The per-unit-length "
                "series-impedance model is available via "
                "witwin.maxwell.compiler.wire_impedance.fit_series_impedance, but the "
                "lossy current recurrence is not yet wired into the FDTD runtime; use a "
                "PEC conductor to run a thin-wire FDTD simulation."
            )
        (
            path,
            path_tensors,
            source_segments,
            source_count,
            source_point_path_indices,
        ) = _expanded_path(
            prepared_scene,
            wire,
            nodes,
            device=target_device,
            dtype=coefficient_dtype,
        )
        radii = _radius_tensor(
            wire.radius,
            source_segment_count=source_count,
            device=target_device,
            dtype=coefficient_dtype,
            wire_name=str(wire.name),
        )
        endpoint_specs = _endpoint_specs(prepared_scene, wire, path, nodes)
        if path[0] == path[-1] and not tuple(getattr(wire, "endpoints", ()) or ()):
            internal_closed_names.add(str(endpoint_specs[0][1]))
        endpoint_kinds = tuple(spec[0] for spec in endpoint_specs)
        _validate_boundary_contacts(prepared_scene, wire, path, nodes)
        _validate_pec_contacts(prepared_scene, wire, path, nodes, endpoint_kinds)
        gap_source_segments = {
            source_segment
            for wire_name, source_segment in gap_segments
            if wire_name == str(wire.name)
        }
        included_path_nodes = set(source_point_path_indices)
        for local_segment, source_segment in enumerate(source_segments):
            if source_segment not in gap_source_segments:
                included_path_nodes.update((local_segment, local_segment + 1))
        for local_node, grid_index in enumerate(path):
            if local_node not in included_path_nodes:
                continue
            endpoint_spec = None
            if local_node == 0:
                endpoint_spec = endpoint_specs[0]
            if local_node == len(path) - 1:
                endpoint_spec = endpoint_specs[1]
            occurrences.setdefault(grid_index, []).append(
                (wire_id, local_node, endpoint_spec)
            )
            position_tensor_occurrences.setdefault(grid_index, []).append(
                path_tensors[local_node]
            )
            if endpoint_spec is not None and endpoint_spec[0] == "node":
                node_name = endpoint_spec[1]
                if node_name is None:
                    raise RuntimeError("Named wire endpoint lost its node name.")
                previous = named_coordinates.get(node_name)
                if previous is not None and previous != grid_index:
                    raise ValueError(
                        f"Wire node {node_name!r} resolves to multiple grid coordinates "
                        f"{previous} and {grid_index}."
                    )
                named_coordinates[node_name] = grid_index
                named_occurrences[node_name] = named_occurrences.get(node_name, 0) + 1

        wire_records.append(
            {
                "wire": wire,
                "wire_id": wire_id,
                "path": path,
                "path_tensors": path_tensors,
                "source_segments": source_segments,
                "source_point_path_indices": source_point_path_indices,
                "gap_source_segments": gap_source_segments,
                "radii": radii,
            }
        )

    for node_name, count in named_occurrences.items():
        if node_name not in internal_closed_names and count < 2:
            raise ValueError(
                f"Wire node {node_name!r} is unresolved; a named junction requires at least two endpoints."
            )

    node_name_by_grid: dict[tuple[float, float, float], str] = {}
    for grid_index, entries in occurrences.items():
        if len(entries) <= 1:
            continue
        labels = [
            spec[1] if spec is not None and spec[0] == "node" else None
            for _wire_id, _local_node, spec in entries
        ]
        if any(label is None for label in labels) or len(set(labels)) != 1:
            participants = tuple(names[wire_id] for wire_id, _local_node, _spec in entries)
            raise ValueError(
                f"ThinWire paths touch at position {grid_index} without one shared named node; "
                f"participants are {participants}."
            )
        node_name = labels[0]
        if node_name is None:
            raise RuntimeError("Shared wire node validation lost its node name.")
        if node_name not in internal_closed_names:
            node_name_by_grid[grid_index] = node_name

    node_positions = sorted(occurrences)
    node_position_tensors = [
        torch.stack(position_tensor_occurrences[position]).mean(dim=0)
        for position in node_positions
    ]
    position_to_node = {
        position: index for index, position in enumerate(node_positions)
    }
    node_wire_ids = [
        min(wire_id for wire_id, _local_node, _spec in occurrences[position])
        for position in node_positions
    ]
    open_flags = []
    grounded_flags = []
    for position in node_positions:
        specs = [spec for _wire_id, _local_node, spec in occurrences[position] if spec is not None]
        open_flags.append(any(spec[0] == "open" for spec in specs))
        grounded_flags.append(any(spec[0] == "grounded" for spec in specs))

    for record in wire_records:
        wire = record["wire"]
        wire_id = record["wire_id"]
        path = record["path"]
        path_tensors = record["path_tensors"]
        source_segments = record["source_segments"]
        source_point_path_indices = record["source_point_path_indices"]
        gap_source_segments = record["gap_source_segments"]
        radii = record["radii"]
        wire_source_positions = [path[index] for index in source_point_path_indices]
        source_point_positions.extend(wire_source_positions)
        wire_source_point_offsets.append(len(source_point_positions))
        active_node_ids = []

        for local_segment, (
            start,
            end,
            start_tensor,
            end_tensor,
            source_segment,
        ) in enumerate(
            zip(
                path,
                path[1:],
                path_tensors,
                path_tensors[1:],
                source_segments,
            )
        ):
            displacement = np.asarray(end, dtype=np.float64) - np.asarray(
                start, dtype=np.float64
            )
            tensor_displacement = end_tensor - start_tensor
            length = torch.linalg.vector_norm(tensor_displacement)
            tangent = tensor_displacement / length
            axis = int(np.argmax(np.abs(displacement)))
            direction = 1 if displacement[axis] > 0.0 else -1
            continuous = _kind_name(getattr(wire, "snap", "strict")) == "continuous"
            coupling_fragments = _segment_coupling_fragments(
                start,
                end,
                start_tensor,
                end_tensor,
                nodes,
                primal,
                shapes,
                continuous=continuous,
            )
            if source_segment in gap_source_segments:
                gap_row = gap_samples.setdefault((str(wire.name), source_segment), [])
                for _micro_length, _cell, _local_midpoint, fragment_samples in coupling_fragments:
                    gap_row.extend(fragment_samples)
                continue
            radius = radii[source_segment]
            micro_distances = []
            micro_ratios = []
            micro_inductances = []
            micro_capacitances = []
            micro_permittivities = []
            micro_permeabilities = []
            for micro_length, cell, local_midpoint, micro_samples in coupling_fragments:
                du, dv = _local_transverse_spacing(
                    cell=cell,
                    local_midpoint=local_midpoint,
                    tangent=tangent,
                    dual=dual,
                )
                distance_t = (
                    _bspline_coupling_distance_tensor(du, dv)
                    if continuous
                    else torch.as_tensor(
                        _bspline_coupling_distance(
                            float(du.detach().cpu()), float(dv.detach().cpu())
                        ),
                        device=target_device,
                        dtype=coefficient_dtype,
                    )
                )
                spacing = torch.minimum(du, dv)
                ratio = radius / spacing
                if not bool(radius < distance_t):
                    raise ValueError(
                        f"ThinWire {wire.name!r} radius must remain below the local coupling distance."
                    )
                if not bool(ratio <= _MAX_RADIUS_TO_SPACING):
                    raise ValueError(
                        f"ThinWire {wire.name!r} exceeds the accepted a/delta_perp <= "
                        f"{_MAX_RADIUS_TO_SPACING} validity band."
                    )
                eps_local = torch.stack(
                    [
                        _interpolate_material_component(
                            eps_fields[name], component, cell, local_midpoint
                        )
                        for component, name in enumerate(_AXES)
                    ]
                )
                mu_local = torch.stack(
                    [
                        _interpolate_material_component(
                            mu_fields[name], component, cell, local_midpoint
                        )
                        for component, name in enumerate(_AXES)
                    ]
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
                        f"ThinWire {wire.name!r} requires a locally isotropic host material."
                    )
                eps_r = eps_local.mean()
                mu_r = mu_local.mean()
                if not bool(torch.isfinite(eps_r)) or not bool(torch.isfinite(mu_r)):
                    raise ValueError(
                        f"ThinWire {wire.name!r} local host material must be finite."
                    )
                if not bool(eps_r > 0.0) or not bool(mu_r > 0.0):
                    raise ValueError(
                        f"ThinWire {wire.name!r} local host material must be positive."
                    )
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
                micro_distances.append(distance_t)
                micro_ratios.append(ratio)
                micro_inductances.append(l_prime * micro_length)
                micro_capacitances.append(c_prime * micro_length)
                micro_permittivities.append(eps_r * micro_length)
                micro_permeabilities.append(mu_r * micro_length)

            inductance = torch.stack(micro_inductances).sum()
            capacitance = torch.stack(micro_capacitances).sum()
            c_prime = capacitance / length
            eps_r = torch.stack(micro_permittivities).sum() / length
            mu_r = torch.stack(micro_permeabilities).sum() / length
            ratio = torch.stack(micro_ratios).max()
            distance_t = torch.exp(
                torch.sum(
                    torch.stack(
                        [
                            micro_length * torch.log(distance)
                            for (micro_length, *_), distance in zip(
                                coupling_fragments, micro_distances
                            )
                        ]
                    )
                )
                / length
            )
            tail_node = position_to_node[start]
            head_node = position_to_node[end]
            tail.append(tail_node)
            head.append(head_node)
            active_node_ids.extend((tail_node, head_node))
            segment_wire_ids.append(wire_id)
            segment_source_ids.append(source_segment)
            segment_axes.append(axis)
            segment_directions.append(direction)
            segment_lengths.append(length)
            segment_radii.append(radius)
            coupling_distances.append(distance_t)
            radius_to_spacing.append(ratio)
            inductance_per_segment.append(inductance)
            capacitance_per_length.append(c_prime)
            local_permittivity.append(eps_r)
            local_permeability.append(mu_r)
            segment_id = len(tail) - 1
            for micro_length, cell, _local_midpoint, micro_samples in coupling_fragments:
                for component, offset, weight in micro_samples:
                    sampling_components.append(component)
                    sampling_offsets.append(offset)
                    sampling_weights.append(weight)
                fragment_offsets.append(len(sampling_weights))
                fragment_segment_ids.append(segment_id)
                fragment_cell_indices.append(cell)
                fragment_lengths.append(micro_length)
            segment_offsets.append(len(sampling_weights))
        active_node_set = set(active_node_ids)
        for source_segment in gap_source_segments:
            gap_endpoint_ids = {
                position_to_node[path[source_point_path_indices[source_segment]]],
                position_to_node[path[source_point_path_indices[source_segment + 1]]],
            }
            if not gap_endpoint_ids.issubset(active_node_set):
                raise ValueError(
                    f"ThinWire {wire.name!r} feed gap source segment {source_segment} "
                    "must leave an active wire fragment on both terminals."
                )
        wire_node_indices.extend(tuple(dict.fromkeys(active_node_ids)))
        wire_node_offsets.append(len(wire_node_indices))
        wire_segment_offsets.append(len(tail))

    node_count = len(node_positions)
    segment_count = len(tail)
    node_position_array = np.asarray(node_positions, dtype=np.float64).reshape(node_count, 3)
    node_position_t = (
        torch.stack(node_position_tensors)
        if node_position_tensors
        else _empty_tensor(target_device, coefficient_dtype, (0, 3))
    )
    node_grid_array = np.asarray(
        [_point_grid_index(position, nodes) for position in node_positions],
        dtype=np.int64,
    ).reshape(node_count, 3)
    wire_node_indices_array = np.asarray(wire_node_indices, dtype=np.int64)
    source_point_node_ids_array = np.asarray(
        [position_to_node[position] for position in source_point_positions],
        dtype=np.int64,
    )
    public_junction_names = tuple(
        sorted(name for name in named_coordinates if name not in internal_closed_names)
    )
    junction_node_ids = [
        position_to_node[named_coordinates[name]] for name in public_junction_names
    ]
    wire_id_by_name = {name: index for index, name in enumerate(names)}

    def source_point_node_id(wire_name: str, point: int) -> int:
        wire_id = wire_id_by_name[wire_name]
        return int(
            source_point_node_ids_array[wire_source_point_offsets[wire_id] + point]
        )

    port_binding_names = []
    port_binding_kinds = []
    port_negative_node_ids = []
    port_positive_node_ids = []
    port_gap_offsets = [0]
    port_gap_components = []
    port_gap_edge_offsets = []
    port_gap_weights = []
    port_binding_metadata = []
    for record in port_binding_records:
        negative_node = source_point_node_id(
            record["negative_wire"], record["negative_point"]
        )
        positive_node = source_point_node_id(
            record["positive_wire"], record["positive_point"]
        )
        if negative_node == positive_node:
            raise ValueError(
                f"Port {record['port_name']!r} wire-binding terminals resolve to the same "
                "global thin-wire node."
            )
        port_binding_names.append(record["port_name"])
        port_binding_kinds.append(record["kind"])
        port_negative_node_ids.append(negative_node)
        port_positive_node_ids.append(positive_node)
        if record["kind"] == "gap":
            key = (record["source_wire"], record["source_segment"])
            combined: dict[tuple[int, int], torch.Tensor] = {}
            for component, offset, weight in gap_samples.get(key, ()):
                edge_key = (component, offset)
                contribution = record["orientation"] * weight
                combined[edge_key] = (
                    contribution
                    if edge_key not in combined
                    else combined[edge_key] + contribution
                )
            for (component, offset), weight in sorted(combined.items()):
                if float(weight.detach().cpu()) == 0.0:
                    continue
                port_gap_components.append(component)
                port_gap_edge_offsets.append(offset)
                port_gap_weights.append(weight)
            if port_gap_offsets[-1] == len(port_gap_weights):
                raise RuntimeError(
                    f"Port {record['port_name']!r} feed gap produced an empty conservative row."
                )
        port_gap_offsets.append(len(port_gap_weights))
        port_binding_metadata.append(
            MappingProxyType(
                {
                    "port_name": record["port_name"],
                    "kind": record["kind"],
                    "negative": (
                        record["negative_wire"],
                        record["negative_point"],
                    ),
                    "positive": (
                        record["positive_wire"],
                        record["positive_point"],
                    ),
                    "negative_node_id": negative_node,
                    "positive_node_id": positive_node,
                    "source_wire": record["source_wire"],
                    "source_segment": record["source_segment"],
                }
            )
        )

    tail_t = torch.as_tensor(tail, device=target_device, dtype=torch.int64)
    head_t = torch.as_tensor(head, device=target_device, dtype=torch.int64)
    if segment_count:
        radius_t = torch.stack(segment_radii)
        length_t = torch.stack(segment_lengths)
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

    _validate_unique_physical_segments(node_position_array, tail, head)
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
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for tail_node, head_node in zip(tail, head):
        tail_root = find(tail_node)
        head_root = find(head_node)
        if tail_root != head_root:
            parent[head_root] = tail_root
    component_count = len({find(node) for node in range(node_count)}) if node_count else 0
    cycle_rank = segment_count - node_count + component_count
    branch_node_count = int(np.count_nonzero(node_counts > 2))

    sampling_components_array = np.asarray(sampling_components, dtype=np.int32)
    sampling_offsets_array = np.asarray(sampling_offsets, dtype=np.int64)
    sampling_weights_array = np.asarray(
        [float(weight.detach().cpu()) for weight in sampling_weights],
        dtype=np.float64,
    )
    deposition = sorted(
        (
            int(sampling_components_array[sample]),
            int(sampling_offsets_array[sample]),
            segment,
            sample,
        )
        for segment in range(segment_count)
        for sample in range(segment_offsets[segment], segment_offsets[segment + 1])
    )
    target_components = []
    target_offsets = []
    previous_key = None
    for component, offset, _segment, _sample in deposition:
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
        ("node_positions", node_position_array),
        ("wire_node_indices", wire_node_indices_array),
        (
            "wire_source_point_offsets",
            np.asarray(wire_source_point_offsets, dtype=np.int64),
        ),
        ("source_point_node_ids", source_point_node_ids_array),
        ("tail", np.asarray(tail, dtype=np.int64)),
        ("head", np.asarray(head, dtype=np.int64)),
        ("wire_ids", np.asarray(segment_wire_ids, dtype=np.int64)),
        ("source_segment_ids", np.asarray(segment_source_ids, dtype=np.int64)),
        (
            "segment_lengths",
            length_t.detach().cpu().to(dtype=torch.float64).numpy(),
        ),
        ("segment_offsets", np.asarray(segment_offsets, dtype=np.int64)),
        ("fragment_offsets", np.asarray(fragment_offsets, dtype=np.int64)),
        (
            "fragment_segment_ids",
            np.asarray(fragment_segment_ids, dtype=np.int64),
        ),
        (
            "fragment_cell_indices",
            np.asarray(fragment_cell_indices, dtype=np.int64).reshape(-1, 3),
        ),
        (
            "fragment_lengths",
            np.asarray(
                [float(value.detach().cpu()) for value in fragment_lengths],
                dtype=np.float64,
            ),
        ),
        ("sampling_components", sampling_components_array),
        ("sampling_offsets", sampling_offsets_array),
        ("sampling_weights", sampling_weights_array),
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
        ("junction_node_ids", np.asarray(junction_node_ids, dtype=np.int64)),
        (
            "port_negative_node_ids",
            np.asarray(port_negative_node_ids, dtype=np.int64),
        ),
        (
            "port_positive_node_ids",
            np.asarray(port_positive_node_ids, dtype=np.int64),
        ),
        ("port_gap_offsets", np.asarray(port_gap_offsets, dtype=np.int64)),
        (
            "port_gap_components",
            np.asarray(port_gap_components, dtype=np.int32),
        ),
        (
            "port_gap_edge_offsets",
            np.asarray(port_gap_edge_offsets, dtype=np.int64),
        ),
        (
            "port_gap_weights",
            np.asarray(
                [float(weight.detach().cpu()) for weight in port_gap_weights],
                dtype=np.float64,
            ),
        ),
    )
    compile_digest = hashlib.sha256()
    compile_digest.update(grid_fingerprint.encode("ascii"))
    for name in names:
        encoded_name = name.encode("utf8")
        compile_digest.update(len(encoded_name).to_bytes(8, byteorder="little"))
        compile_digest.update(encoded_name)
    for name in public_junction_names:
        encoded_name = name.encode("utf8")
        compile_digest.update(len(encoded_name).to_bytes(8, byteorder="little"))
        compile_digest.update(encoded_name)
    for record in port_binding_records:
        for value in (
            record["port_name"],
            record["kind"],
            record["negative_wire"],
            record["negative_point"],
            record["positive_wire"],
            record["positive_point"],
            record["source_wire"],
            record["source_segment"],
        ):
            encoded_value = str(value).encode("utf8")
            compile_digest.update(len(encoded_value).to_bytes(8, byteorder="little"))
            compile_digest.update(encoded_value)
    compile_digest.update(_hash_arrays(*grid_entries).encode("ascii"))
    compile_fingerprint = compile_digest.hexdigest()
    cache_enabled = not any(
        tensor.requires_grad
        for tensor in (
            node_position_t,
            radius_t,
            local_permittivity_t,
            local_permeability_t,
        )
    )
    boundary = getattr(prepared_scene, "boundary", None)
    periodic_axes = tuple(
        axis
        for axis in _AXES
        if boundary is not None and _kind_name(boundary.face_kind(axis, "low")) == "periodic"
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
            "wire_node_index_semantics": "CSR membership into wire_node_indices",
            "wire_source_point_ranges": tuple(
                (name, wire_source_point_offsets[index], wire_source_point_offsets[index + 1])
                for index, name in enumerate(names)
            ),
            "junction_nodes": tuple(zip(public_junction_names, junction_node_ids)),
            "wire_segment_ranges": tuple(
                (name, wire_segment_offsets[index], wire_segment_offsets[index + 1])
                for index, name in enumerate(names)
            ),
            "segment_semantics": "physical_polyline_span",
            "fragment_semantics": "cell_local_coupling_without_state",
            "port_bindings": tuple(port_binding_metadata),
            "validity": MappingProxyType(
                {
                    "phase": 3,
                    "conductor": "pec",
                    "topology": "arbitrary_direction_graph",
                    "junction_count": len(public_junction_names),
                    "branch_node_count": branch_node_count,
                    "cycle_rank": cycle_rank,
                    "coupling_kernel": "BS1xBS1",
                    "yee_coupling": "cell_local_tensor_product_line_integral",
                    "fragmentation": "exact_grid_plane_clipping",
                    "coordinate_gradient": "fixed_stencil_only",
                    "cross_cell_gradient": "discontinuous",
                    "periodic_axes": periodic_axes,
                    "periodic_path_semantics": "absolute_interior_no_wrap",
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
        node_positions=node_position_t,
        node_grid_indices=torch.as_tensor(node_grid_array, device=target_device, dtype=torch.int64),
        node_ids=torch.arange(node_count, device=target_device, dtype=torch.int64),
        node_wire_ids=torch.as_tensor(node_wire_ids, device=target_device, dtype=torch.int64),
        wire_node_offsets=torch.as_tensor(wire_node_offsets, device=target_device, dtype=torch.int64),
        wire_node_indices=torch.as_tensor(
            wire_node_indices_array, device=target_device, dtype=torch.int64
        ),
        wire_source_point_offsets=torch.as_tensor(
            wire_source_point_offsets, device=target_device, dtype=torch.int64
        ),
        source_point_node_ids=torch.as_tensor(
            source_point_node_ids_array, device=target_device, dtype=torch.int64
        ),
        junction_names=public_junction_names,
        junction_node_ids=torch.as_tensor(
            junction_node_ids, device=target_device, dtype=torch.int64
        ),
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
        segment_source_ids=torch.as_tensor(
            segment_source_ids, device=target_device, dtype=torch.int64
        ),
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
        local_permittivity=local_permittivity_t,
        local_permeability=local_permeability_t,
        segment_offsets=torch.as_tensor(
            segment_offsets, device=target_device, dtype=torch.int64
        ),
        fragment_offsets=torch.as_tensor(
            fragment_offsets, device=target_device, dtype=torch.int64
        ),
        fragment_segment_ids=torch.as_tensor(
            fragment_segment_ids, device=target_device, dtype=torch.int64
        ),
        fragment_cell_indices=torch.as_tensor(
            fragment_cell_indices, device=target_device, dtype=torch.int64
        ).reshape(-1, 3),
        fragment_lengths=(
            torch.stack(fragment_lengths)
            if fragment_lengths
            else _empty_tensor(target_device, coefficient_dtype, (0,))
        ),
        edge_components=torch.as_tensor(
            sampling_components_array, device=target_device, dtype=torch.int32
        ),
        edge_offsets=torch.as_tensor(
            sampling_offsets_array, device=target_device, dtype=torch.int64
        ),
        weights=(
            torch.stack(sampling_weights)
            if sampling_weights
            else _empty_tensor(target_device, coefficient_dtype, (0,))
        ),
        edge_group_offsets=torch.as_tensor(edge_group_offsets, device=target_device, dtype=torch.int64),
        target_components=torch.as_tensor(target_components, device=target_device, dtype=torch.int32),
        target_offsets=torch.as_tensor(target_offsets, device=target_device, dtype=torch.int64),
        contribution_segments=torch.as_tensor(
            [entry[2] for entry in deposition], device=target_device, dtype=torch.int64
        ),
        contribution_weights=(
            torch.stack([sampling_weights[entry[3]] for entry in deposition])
            if deposition
            else _empty_tensor(target_device, coefficient_dtype, (0,))
        ),
        port_binding_names=tuple(port_binding_names),
        port_binding_kinds=tuple(port_binding_kinds),
        port_negative_node_ids=torch.as_tensor(
            port_negative_node_ids, device=target_device, dtype=torch.int64
        ),
        port_positive_node_ids=torch.as_tensor(
            port_positive_node_ids, device=target_device, dtype=torch.int64
        ),
        port_gap_offsets=torch.as_tensor(
            port_gap_offsets, device=target_device, dtype=torch.int64
        ),
        port_gap_edge_components=torch.as_tensor(
            port_gap_components, device=target_device, dtype=torch.int32
        ),
        port_gap_edge_offsets=torch.as_tensor(
            port_gap_edge_offsets, device=target_device, dtype=torch.int64
        ),
        port_gap_weights=(
            torch.stack(port_gap_weights)
            if port_gap_weights
            else _empty_tensor(target_device, coefficient_dtype, (0,))
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
                node_indices=network.wire_node_indices[node_start:node_end],
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
