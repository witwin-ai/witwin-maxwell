from __future__ import annotations

import torch
from collections.abc import Mapping
from dataclasses import dataclass, asdict

from ..fdtd.distributed.capacity import local_dft_working_set_bytes
from ..fdtd.distributed.output import electric_field_output_bytes
from ..scene import Scene

# Conservative multiplier over the three staggered E-field element counts that
# approximates one FDTD shard's persistent device state: the six Yee field
# components plus the per-component update coefficients, material inverses and
# CPML/auxiliary buffers all scale with the same cell count. This is a placement
# preflight heuristic (fail before OOM), not an exact allocator accounting.
_FIELD_STATE_MULTIPLIER = 8


@dataclass(frozen=True)
class TensorAllocation:
    """A unique live storage and the tensor views reachable from audit roots."""

    device: str
    storage_bytes: int
    owners: tuple[str, ...]
    views: tuple[dict, ...]


def tensor_memory_manifest(roots: Mapping[str, object]) -> dict:
    """Audit reachable tensor storage without reading tensor values.

    This measures storage, not allocator reservations, graph-private pools or
    opaque native allocations. Attributes are read from instance dictionaries;
    properties, callables and autograd graphs are deliberately not traversed.
    Shared containers are visited once, with their aliases reported separately.
    """
    storages = {}
    visited = {}
    aliases = {}
    unsupported = []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.device.type == "meta":
                unsupported.append(path)
                return
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            # Empty allocations carry no bytes and need no unique pointer.
            record = storages.setdefault(key, {
                "device": str(value.device), "storage_bytes": storage.nbytes(),
                "owners": [], "views": [],
            })
            record["owners"].append(path)
            record["views"].append({
                "path": path, "shape": list(value.shape),
                "stride": list(value.stride()), "dtype": str(value.dtype),
                "offset_elements": value.storage_offset(),
                "logical_bytes": value.numel() * value.element_size(),
                "requires_grad": value.requires_grad,
            })
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)) or callable(value):
            return
        if id(value) in visited:
            aliases[path] = visited[id(value)]
            return
        visited[id(value)] = path
        if isinstance(value, Mapping):
            children = value.items()
        elif isinstance(value, (list, tuple)):
            children = enumerate(value)
        elif type(value).__module__.startswith(("witwin.", "types")):
            children = vars(value).items() if hasattr(value, "__dict__") else ()
        else:
            return
        for name, child in children:
            visit(child, f"{path}.{name}")

    for name, value in roots.items():
        visit(value, str(name))
    allocations = [asdict(TensorAllocation(
        device=r["device"], storage_bytes=r["storage_bytes"],
        owners=tuple(r["owners"]), views=tuple(r["views"]),
    )) for r in storages.values()]
    totals = {}
    for allocation in allocations:
        device = allocation["device"]
        totals[device] = totals.get(device, 0) + allocation["storage_bytes"]
    return {
        "schema_version": 1, "storage_bytes_by_device": totals,
        "allocations": allocations, "container_aliases": aliases,
        "unsupported_tensors": unsupported,
        "excludes": ["allocator_reserved", "opaque_native_allocations", "autograd_saved_tensors"],
    }


def _node_counts(scene: Scene) -> tuple[int, int, int] | None:
    """Cheaply derive the Yee node counts from a uniform grid.

    Only uniform grids expose their shape from ``Domain`` bounds and spacing
    alone. Custom and auto grids would require a full ``prepare_scene`` on the
    coordinator (allocating on the scene's device and discarding the result), so
    they deliberately return ``None`` and let the scheduler fall back to
    order-based placement instead of doing placement-time GPU work.
    """

    grid = scene.grid
    if grid.is_custom or grid.is_auto:
        return None
    bounds = scene.domain.bounds
    spacing = grid.spacing
    counts = []
    for (low, high), delta in zip(bounds, spacing):
        extent = float(high) - float(low)
        if delta is None or float(delta) <= 0.0 or extent <= 0.0:
            return None
        counts.append(int(round(extent / float(delta))) + 1)
    return tuple(counts)


def estimate_scene_footprint_bytes(
    scene: Scene,
    *,
    frequencies=None,
    full_field_dft: bool = False,
) -> int | None:
    """Estimate one Simulation's resident device footprint for placement.

    Returns ``None`` when the grid shape cannot be resolved cheaply, in which
    case the scheduler falls back to order-based placement for that task.
    """

    counts = _node_counts(scene)
    if counts is None:
        return None
    field_bytes = electric_field_output_bytes(counts, frequency_count=1, complex_output=False)
    resident = _FIELD_STATE_MULTIPLIER * field_bytes
    dft_bytes = local_dft_working_set_bytes(
        counts,
        dft_frequency=tuple(frequencies) if frequencies else None,
        full_field_dft=full_field_dft,
    )
    return int(resident + dft_bytes)


def estimate_simulation_footprint_bytes(simulation) -> int | None:
    """Estimate the resident footprint of a prepared ``Simulation`` task."""

    scene = getattr(simulation, "scene", None)
    if not isinstance(scene, Scene):
        return None
    config = getattr(simulation, "config", None)
    full_field_dft = bool(getattr(config, "full_field_dft", False))
    return estimate_scene_footprint_bytes(
        scene,
        frequencies=getattr(simulation, "frequencies", None),
        full_field_dft=full_field_dft,
    )


__all__ = [
    "tensor_memory_manifest",
    "estimate_scene_footprint_bytes",
    "estimate_simulation_footprint_bytes",
]
