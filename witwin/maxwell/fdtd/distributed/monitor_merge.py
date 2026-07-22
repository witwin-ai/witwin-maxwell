from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from ...fdtd_parallel import FDTDShardLayout
from ..observers import _align_plane_monitor_payload, _compute_plane_flux
from .output import move_tensors_to_device


_PLANE_COORD_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}


def _validate_matching_metadata(entries: list[tuple[int, Mapping[str, Any]]]) -> None:
    first = entries[0][1]
    for rank, payload in entries[1:]:
        for key in (
            "kind",
            "monitor_type",
            "fields",
            "frequencies",
            "axis",
            "position",
            "compute_flux",
            "normal_direction",
        ):
            if payload.get(key) != first.get(key):
                raise ValueError(
                    f"Shard {rank} monitor metadata {key!r} does not match rank "
                    f"{entries[0][0]}."
                )


def _normalize_shard_layouts(
    shard_layouts: Mapping[int, FDTDShardLayout] | Iterable[FDTDShardLayout] | None,
) -> dict[int, FDTDShardLayout]:
    if shard_layouts is None:
        return {}
    values = shard_layouts.values() if isinstance(shard_layouts, Mapping) else shard_layouts
    normalized: dict[int, FDTDShardLayout] = {}
    for layout in values:
        if not isinstance(layout, FDTDShardLayout):
            raise TypeError("shard_layouts must contain FDTDShardLayout instances.")
        if layout.rank in normalized:
            raise ValueError(f"shard_layouts contains duplicate rank {layout.rank}.")
        normalized[layout.rank] = layout
    return normalized


def _slice_tensor_x(tensor: torch.Tensor, x_slice: slice) -> torch.Tensor:
    index = [slice(None)] * tensor.ndim
    index[-2] = x_slice
    return tensor[tuple(index)]


def _stitch_owned_component(
    entries: list[tuple[int, Mapping[str, Any]]],
    component: str,
    *,
    shard_layouts: Mapping[int, FDTDShardLayout],
    result_device: torch.device,
) -> dict[str, Any]:
    owned_tiles = []
    expected_transverse = None
    expected_global_extent = None
    for rank, monitor_payload in entries:
        component_payloads = monitor_payload.get("components", {})
        if component not in component_payloads:
            raise ValueError(f"Shard {rank} is missing plane component {component!r}.")
        component_payload = component_payloads[component]
        data = component_payload.get("data")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Shard {rank} plane component {component!r} data must be a tensor.")
        if data.ndim < 2:
            raise ValueError(
                f"Shard {rank} plane component {component!r} data must have at least two dimensions."
            )
        coords = component_payload.get("coords", ())
        if len(coords) != 2:
            raise ValueError(
                f"Shard {rank} component {component!r} requires two plane coordinate arrays."
            )
        x_values = np.asarray(coords[0], dtype=np.float64)
        transverse = np.asarray(coords[1], dtype=np.float64)

        component_layout = shard_layouts[rank].component(component)
        local_x_extent = int(component_layout.local_shape[0])
        if x_values.size != local_x_extent or int(data.shape[-2]) != local_x_extent:
            raise ValueError(
                f"Shard {rank} component {component!r} local x extent does not match "
                f"its layout: coords={x_values.size}, data={int(data.shape[-2])}, "
                f"layout={local_x_extent}."
            )
        if int(data.shape[-1]) != transverse.size:
            raise ValueError(
                f"Shard {rank} component {component!r} transverse coordinates do not "
                f"match data shape {tuple(data.shape)}."
            )
        if expected_transverse is None:
            expected_transverse = np.array(transverse, copy=True)
        elif not np.array_equal(transverse, expected_transverse):
            raise ValueError(
                f"Shard {rank} component {component!r} transverse coordinates do not match."
            )

        owned_local = component_layout.owned_local_slice[0]
        owned_global = component_layout.owned_global_slice[0]
        owned_count = int(owned_global.stop) - int(owned_global.start)
        cropped_x = np.array(x_values[owned_local], copy=True)
        cropped_data = _slice_tensor_x(data, owned_local)
        if cropped_x.size != owned_count or int(cropped_data.shape[-2]) != owned_count:
            raise RuntimeError(
                f"Shard {rank} component {component!r} owned local/global x extents disagree."
            )
        global_extent = int(component_layout.global_shape[0])
        if expected_global_extent is None:
            expected_global_extent = global_extent
        elif expected_global_extent != global_extent:
            raise ValueError(
                f"Shard {rank} component {component!r} global x extent is inconsistent."
            )
        owned_tiles.append(
            (
                int(owned_global.start),
                int(owned_global.stop),
                rank,
                cropped_data.to(device=result_device, non_blocking=True),
                cropped_x,
                component_payload,
            )
        )

    owned_tiles.sort(key=lambda item: (item[0], item[2]))
    cursor = 0
    for begin, end, rank, _data, _x, _payload in owned_tiles:
        if begin != cursor:
            relation = "overlap" if begin < cursor else "gap"
            raise ValueError(
                f"Component {component!r} owned global x slices contain a {relation} "
                f"before shard {rank}: expected {cursor}, got {begin}."
            )
        cursor = end
    if expected_global_extent is None or cursor != expected_global_extent:
        raise ValueError(
            f"Component {component!r} owned global x coverage ends at {cursor}, "
            f"expected {expected_global_extent}."
        )

    merged_data = torch.cat([item[3] for item in owned_tiles], dim=-2)
    merged_x = np.concatenate([item[4] for item in owned_tiles])
    if merged_x.size > 1 and not np.all(np.diff(merged_x) > 0.0):
        raise ValueError(
            f"Component {component!r} owned global x coordinates are not strictly increasing."
        )
    template = {
        key: value
        for key, value in owned_tiles[0][5].items()
        if key not in {"data", "coords"}
    }
    merged = move_tensors_to_device(template, result_device)
    merged["data"] = merged_data
    merged["coords"] = (merged_x, expected_transverse)
    return merged


def _merge_owned_components(
    entries: list[tuple[int, Mapping[str, Any]]],
    *,
    shard_layouts: Mapping[int, FDTDShardLayout],
    result_device: torch.device,
) -> dict[str, Any]:
    first_components = entries[0][1].get("components", {})
    if not isinstance(first_components, Mapping) or not first_components:
        raise ValueError("A tiled plane monitor requires raw Yee component payloads.")
    expected = tuple(first_components)
    for rank, payload in entries[1:]:
        if tuple(payload.get("components", {})) != expected:
            raise ValueError(
                f"Shard {rank} plane component order/set does not match rank {entries[0][0]}."
            )
    return {
        component: _stitch_owned_component(
            entries,
            component,
            shard_layouts=shard_layouts,
            result_device=result_device,
        )
        for component in expected
    }


def _plane_metadata_template(payload: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "components",
        "component",
        "data",
        "coords",
        # ``cell_widths`` is a per-shard local quadrature table; the tiled merge
        # reassembles the global plane coordinates below, so a shard-local copy
        # here would be stale (wrong length). Dropping it lets the flux
        # integration derive cell-center weights from the merged coordinates.
        "cell_widths",
        "x",
        "y",
        "z",
        "flux",
        "power",
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _normalize_physical_bounds(physical_bounds):
    if physical_bounds is None or len(physical_bounds) != 3:
        raise ValueError(
            "physical_bounds must contain the three logical-domain (low, high) pairs."
        )
    normalized = []
    for bounds in physical_bounds:
        if len(bounds) != 2:
            raise ValueError("Each physical_bounds entry must be a (low, high) pair.")
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            raise ValueError("physical_bounds entries must be finite and ordered.")
        normalized.append((low, high))
    return tuple(normalized)


def _physical_coord_indices(values, bounds: tuple[float, float]) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float64)
    low, high = bounds
    coordinate_magnitude = max(
        abs(low),
        abs(high),
        float(np.max(np.abs(coords))) if coords.size else 0.0,
    )
    domain_span = abs(high - low)
    unique = np.unique(coords)
    positive_spacing = np.diff(unique)
    positive_spacing = positive_spacing[positive_spacing > 0.0]
    local_spacing = (
        float(np.min(positive_spacing)) if positive_spacing.size else domain_span
    )
    scale = max(
        coordinate_magnitude,
        domain_span,
        local_spacing,
        np.finfo(np.float64).tiny,
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    indices = np.flatnonzero(
        (coords >= low - tolerance) & (coords <= high + tolerance)
    )
    if indices.size == 0:
        raise ValueError("FluxMonitor plane has no samples inside the physical domain.")
    return indices


def _index_plane_values(values, first: np.ndarray, second: np.ndarray):
    if isinstance(values, torch.Tensor):
        first_index = torch.as_tensor(first, device=values.device, dtype=torch.long)
        second_index = torch.as_tensor(second, device=values.device, dtype=torch.long)
        selected = torch.index_select(values, values.ndim - 2, first_index)
        return torch.index_select(selected, selected.ndim - 1, second_index)
    selected = np.take(np.asarray(values), first, axis=np.ndim(values) - 2)
    return np.take(selected, second, axis=np.ndim(selected) - 1)


def _crop_aligned_to_physical_bounds(axis: str, aligned, physical_bounds):
    bounds = _normalize_physical_bounds(physical_bounds)
    coord_names = _PLANE_COORD_NAMES[axis]
    axis_index = {"x": 0, "y": 1, "z": 2}
    first_coord = np.asarray(aligned[coord_names[0]], dtype=np.float64)
    second_coord = np.asarray(aligned[coord_names[1]], dtype=np.float64)
    first_indices = _physical_coord_indices(
        first_coord,
        bounds[axis_index[coord_names[0]]],
    )
    second_indices = _physical_coord_indices(
        second_coord,
        bounds[axis_index[coord_names[1]]],
    )
    return {
        coord_names[0]: np.array(first_coord[first_indices], copy=True),
        coord_names[1]: np.array(second_coord[second_indices], copy=True),
        "fields": {
            component: _index_plane_values(values, first_indices, second_indices)
            for component, values in aligned["fields"].items()
        },
    }


def _merge_tiled_plane(
    entries: list[tuple[int, Mapping[str, Any]]],
    *,
    shard_layouts: Mapping[int, FDTDShardLayout],
    physical_bounds,
    result_device: torch.device,
) -> dict[str, Any]:
    _validate_matching_metadata(entries)
    first = entries[0][1]
    axis = str(first.get("axis", "")).lower()
    if axis not in {"y", "z"}:
        raise ValueError(f"Only y/z-normal planes are tiled across x, got axis={axis!r}.")

    missing_layouts = sorted(rank for rank, _payload in entries if rank not in shard_layouts)
    if missing_layouts:
        ranks = ", ".join(str(rank) for rank in missing_layouts)
        raise ValueError(f"Tiled plane merge is missing shard layouts for rank {ranks}.")

    merged = move_tensors_to_device(_plane_metadata_template(first), result_device)
    components = _merge_owned_components(
        entries,
        shard_layouts=shard_layouts,
        result_device=result_device,
    )
    merged["components"] = components

    fields = tuple(first.get("fields", ()))
    coord_names = _PLANE_COORD_NAMES[axis]
    if len(fields) == 1:
        if bool(first.get("compute_flux", False)):
            raise ValueError("FluxMonitor requires multiple tangential E/H components.")
        component = fields[0]
        component_payload = components[component]
        merged["component"] = component.lower()
        merged["data"] = component_payload["data"]
        merged[component] = component_payload["data"]
        merged["coords"] = component_payload["coords"]
        merged[coord_names[0]] = component_payload["coords"][0]
        merged[coord_names[1]] = component_payload["coords"][1]
        return merged

    aligned = _align_plane_monitor_payload(axis, components)
    if aligned is None:
        if bool(first.get("compute_flux", False)):
            raise ValueError(
                "FluxMonitor is missing aligned tangential fields required for integration."
            )
        return merged
    if bool(first.get("compute_flux", False)):
        aligned = _crop_aligned_to_physical_bounds(axis, aligned, physical_bounds)

    merged[coord_names[0]] = aligned[coord_names[0]]
    merged[coord_names[1]] = aligned[coord_names[1]]
    merged["coords"] = (aligned[coord_names[0]], aligned[coord_names[1]])
    for component, values in aligned["fields"].items():
        merged[component] = values

    if bool(first.get("compute_flux", False)):
        flux = _compute_plane_flux(merged)
        merged["flux"] = flux
        merged["power"] = flux
    return merged


def merge_sharded_monitor_payloads(
    monitor_order: Iterable[str],
    shard_payloads: Iterable[tuple[int, Mapping[str, Mapping[str, Any]]]],
    *,
    result_device: str | torch.device,
    shard_layouts: (
        Mapping[int, FDTDShardLayout] | Iterable[FDTDShardLayout] | None
    ) = None,
    physical_bounds=None,
) -> dict[str, Any]:
    """Merge rank-local monitor payloads in scene declaration order.

    Point monitors and x-normal planes have exactly one owner. For y/z-normal
    planes, every raw Yee component is first cropped by its component-specific
    owned-local x slice and concatenated by owned-global x slice. Multi-component
    planes are aligned only after global assembly. Flux is recomputed on the
    result device after cropping to the logical physical bounds.
    """

    device = torch.device(result_device)
    layouts = _normalize_shard_layouts(shard_layouts)
    by_name: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for rank, payloads in sorted(shard_payloads, key=lambda item: int(item[0])):
        for name, payload in payloads.items():
            by_name.setdefault(str(name), []).append((int(rank), payload))

    merged = {}
    for name in monitor_order:
        entries = by_name.pop(str(name), [])
        if not entries:
            continue
        first = entries[0][1]
        if first.get("kind") == "plane" and first.get("axis") in {"y", "z"}:
            merged[str(name)] = _merge_tiled_plane(
                entries,
                shard_layouts=layouts,
                physical_bounds=physical_bounds,
                result_device=device,
            )
        else:
            if len(entries) != 1:
                axis = first.get("axis")
                descriptor = "x-normal plane" if axis == "x" else "monitor"
                raise RuntimeError(f"{descriptor} {name!r} has more than one shard owner.")
            merged[str(name)] = move_tensors_to_device(entries[0][1], device)

    if by_name:
        undeclared = ", ".join(sorted(by_name))
        raise ValueError(f"Shard payloads contain undeclared monitors: {undeclared}.")
    return merged


__all__ = ["merge_sharded_monitor_payloads"]
