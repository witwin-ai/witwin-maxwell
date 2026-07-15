from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch


_COMPONENT_SHAPE_OFFSETS = {
    "Ex": (-1, 0, 0),
    "Ey": (0, -1, 0),
    "Ez": (0, 0, -1),
    "Hx": (0, -1, -1),
    "Hy": (-1, 0, -1),
    "Hz": (-1, -1, 0),
}
_CELL_EXTENT_COMPONENTS = frozenset(("Ex", "Hy", "Hz"))
_NODE_EXTENT_COMPONENTS = frozenset(("Ey", "Ez", "Hx"))
_ELECTRIC_HALO_COMPONENTS = frozenset(("Ey", "Ez"))
_MAGNETIC_HALO_COMPONENTS = frozenset(("Hy", "Hz"))
_PHYSICAL_FACES = frozenset(
    ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")
)

Index3 = tuple[slice, slice, slice]


def _require_positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _require_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def _normalize_cuda_device(value, *, name: str) -> str:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a CUDA device, got {value!r}.") from exc
    if device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA device, got {device}.")
    if device.index is None:
        raise ValueError(f"{name} must include an explicit CUDA index, got {device}.")
    return str(device)


def _normalize_devices(devices, *, minimum: int) -> tuple[str, ...]:
    if isinstance(devices, (str, bytes, torch.device)):
        raise TypeError("devices must be an ordered iterable of CUDA devices.")
    try:
        values = tuple(devices)
    except TypeError as exc:
        raise TypeError("devices must be an ordered iterable of CUDA devices.") from exc
    if len(values) < minimum:
        noun = "device" if minimum == 1 else "unique CUDA devices"
        raise ValueError(f"devices must contain at least {minimum} {noun}.")
    normalized = tuple(
        _normalize_cuda_device(value, name=f"devices[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("devices must contain unique CUDA devices.")
    return normalized


def _normalize_axis(value: str) -> str:
    axis = str(value).strip().lower()
    if axis != "x":
        raise ValueError("decomposition_axis must be 'x' for FDTD joint solves.")
    return axis


@dataclass(frozen=True)
class FDTDParallelConfig:
    """Public execution descriptor for one FDTD solve spanning CUDA devices.

    This type performs only structural validation. CUDA availability, peer access,
    topology, homogeneity, and memory capacity are intentionally prepare-time
    checks so the public configuration remains usable in CPU-only API tests.
    """

    devices: tuple[str | torch.device, ...]
    decomposition_axis: str = "x"
    transport: str = "auto"
    overlap: bool = True
    gather_fields: bool = False
    result_device: str | torch.device | None = None

    def __post_init__(self) -> None:
        devices = _normalize_devices(self.devices, minimum=2)
        axis = _normalize_axis(self.decomposition_axis)
        transport = str(self.transport).strip().lower()
        if transport not in {"auto", "cuda_p2p", "nccl"}:
            raise ValueError("transport must be one of: auto, cuda_p2p, nccl.")
        if not isinstance(self.overlap, bool):
            raise TypeError("overlap must be a bool.")
        if not isinstance(self.gather_fields, bool):
            raise TypeError("gather_fields must be a bool.")

        result_device = devices[0]
        if self.result_device is not None:
            result_device = _normalize_cuda_device(self.result_device, name="result_device")
        if result_device not in devices:
            raise ValueError("result_device must be one of the participating devices.")

        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "decomposition_axis", axis)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "result_device", result_device)


@dataclass(frozen=True)
class FDTDHaloRegion:
    """A persistent receive-halo or send region for one component and neighbor."""

    side: str
    neighbor_rank: int
    global_slice: Index3
    local_slice: Index3

    def __post_init__(self) -> None:
        if self.side not in {"low", "high"}:
            raise ValueError("side must be 'low' or 'high'.")
        if isinstance(self.neighbor_rank, bool) or not isinstance(self.neighbor_rank, int):
            raise TypeError("neighbor_rank must be an integer.")

    @property
    def width(self) -> int:
        axis_slice = self.global_slice[0]
        return int(axis_slice.stop) - int(axis_slice.start)


@dataclass(frozen=True)
class FDTDComponentLayout:
    """Owned and allocated extents of one staggered Yee component on one shard."""

    component: str
    global_shape: tuple[int, int, int]
    owned_global_slice: Index3
    allocation_global_slice: Index3
    owned_local_slice: Index3
    global_origin: tuple[int, int, int]
    receive_halos: tuple[FDTDHaloRegion, ...] = ()
    send_regions: tuple[FDTDHaloRegion, ...] = ()

    def __post_init__(self) -> None:
        if self.component not in _COMPONENT_SHAPE_OFFSETS:
            raise ValueError(f"Unsupported Yee component {self.component!r}.")

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return tuple(
            int(axis_slice.stop) - int(axis_slice.start)
            for axis_slice in self.owned_global_slice
        )

    @property
    def local_shape(self) -> tuple[int, int, int]:
        return tuple(
            int(axis_slice.stop) - int(axis_slice.start)
            for axis_slice in self.allocation_global_slice
        )

    @property
    def low_halo(self) -> FDTDHaloRegion | None:
        return next((region for region in self.receive_halos if region.side == "low"), None)

    @property
    def high_halo(self) -> FDTDHaloRegion | None:
        return next((region for region in self.receive_halos if region.side == "high"), None)

    @property
    def low_send(self) -> FDTDHaloRegion | None:
        return next((region for region in self.send_regions if region.side == "low"), None)

    @property
    def high_send(self) -> FDTDHaloRegion | None:
        return next((region for region in self.send_regions if region.side == "high"), None)

    def owns(self, global_index: tuple[int, int, int]) -> bool:
        return _index_in_slice(global_index, self.owned_global_slice)

    def contains(self, global_index: tuple[int, int, int]) -> bool:
        """Whether the local allocation contains an owned or receive-halo value."""

        return _index_in_slice(global_index, self.allocation_global_slice)

    def global_to_local(
        self,
        global_index: tuple[int, int, int],
        *,
        include_halo: bool = False,
    ) -> tuple[int, int, int]:
        selected = self.allocation_global_slice if include_halo else self.owned_global_slice
        if not _index_in_slice(global_index, selected):
            extent = "local allocation" if include_halo else "owned extent"
            raise IndexError(
                f"Global index {global_index!r} is outside {self.component} {extent}."
            )
        return tuple(
            int(index) - int(origin)
            for index, origin in zip(global_index, self.global_origin)
        )

    def local_to_global(
        self,
        local_index: tuple[int, int, int],
        *,
        include_halo: bool = False,
    ) -> tuple[int, int, int]:
        if len(local_index) != 3:
            raise ValueError("local_index must have three entries.")
        global_index = tuple(
            int(index) + int(origin)
            for index, origin in zip(local_index, self.global_origin)
        )
        selected = self.allocation_global_slice if include_halo else self.owned_global_slice
        if not _index_in_slice(global_index, selected):
            extent = "local allocation" if include_halo else "owned extent"
            raise IndexError(
                f"Local index {local_index!r} is outside {self.component} {extent}."
            )
        return global_index


def _index_in_slice(index: tuple[int, int, int], extent: Index3) -> bool:
    if len(index) != 3:
        raise ValueError("A Yee-grid index must have three entries.")
    return all(
        int(axis_slice.start) <= int(value) < int(axis_slice.stop)
        for value, axis_slice in zip(index, extent)
    )


@dataclass(frozen=True)
class FDTDShardLayout:
    """Immutable ownership, halo, and physical-face metadata for one CUDA shard."""

    rank: int
    device: str
    global_shape: tuple[int, int, int]
    physical_cell_begin: int
    physical_cell_end: int
    global_cell_owned: slice
    global_node_owned: slice
    storage_cell_owned: slice
    storage_node_owned: slice
    halo_width: int
    component_layouts: tuple[FDTDComponentLayout, ...]
    physical_faces: frozenset[str]

    @property
    def cell_begin(self) -> int:
        return self.physical_cell_begin

    @property
    def cell_end(self) -> int:
        return self.physical_cell_end

    @property
    def cell_interval(self) -> tuple[int, int]:
        """Tuple form of the global physical-cell interval."""

        return self.physical_cell_begin, self.physical_cell_end

    @property
    def physical_owned_cell_count(self) -> int:
        return self.physical_cell_end - self.physical_cell_begin

    @property
    def owned_cell_count(self) -> int:
        return int(self.global_cell_owned.stop) - int(self.global_cell_owned.start)

    @property
    def owned_node_count(self) -> int:
        return int(self.global_node_owned.stop) - int(self.global_node_owned.start)

    @property
    def storage_cell_count(self) -> int:
        return self.owned_cell_count + int(self.storage_cell_owned.start)

    @property
    def storage_node_count(self) -> int:
        return self.storage_cell_count + 1

    @property
    def components(self) -> Mapping[str, FDTDComponentLayout]:
        return MappingProxyType(
            {layout.component: layout for layout in self.component_layouts}
        )

    def component(self, name: str) -> FDTDComponentLayout:
        normalized = str(name).strip()
        normalized = normalized[0].upper() + normalized[1:].lower() if normalized else normalized
        for layout in self.component_layouts:
            if layout.component == normalized:
                return layout
        raise KeyError(f"Unknown Yee component {name!r}.")

    def owns_physical_face(self, axis: str, side: str | None = None) -> bool:
        face = str(axis).strip().lower()
        if side is not None:
            face = f"{face}_{str(side).strip().lower()}"
        if face not in _PHYSICAL_FACES:
            choices = ", ".join(sorted(_PHYSICAL_FACES))
            raise ValueError(f"Physical face must be one of: {choices}.")
        return face in self.physical_faces


@dataclass(frozen=True)
class FDTDPartitionPlan:
    """Balanced x-cell domain decomposition for all six staggered components."""

    global_shape: tuple[int, int, int]
    devices: tuple[str | torch.device, ...]
    decomposition_axis: str = "x"
    halo_width: int = 1
    low_pml_cells: int = 0
    high_pml_cells: int = 0
    shard_layouts: tuple[FDTDShardLayout, ...] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.global_shape) != 3:
            raise ValueError("global_shape must be an (Nx, Ny, Nz) tuple.")
        shape = tuple(
            _require_positive_int(value, name=f"global_shape[{axis}]")
            for axis, value in enumerate(self.global_shape)
        )
        if any(value < 2 for value in shape):
            raise ValueError("FDTD global_shape dimensions must all be >= 2.")
        devices = _normalize_devices(self.devices, minimum=1)
        axis = _normalize_axis(self.decomposition_axis)
        halo_width = _require_positive_int(self.halo_width, name="halo_width")
        if halo_width != 1:
            raise ValueError("halo_width must be 1 for the current second-order Yee stencil.")
        low_pml_cells = _require_nonnegative_int(self.low_pml_cells, name="low_pml_cells")
        high_pml_cells = _require_nonnegative_int(self.high_pml_cells, name="high_pml_cells")

        cell_count = shape[0] - 1
        physical_cell_count = cell_count - low_pml_cells - high_pml_cells
        if physical_cell_count <= 0:
            raise ValueError(
                "low_pml_cells + high_pml_cells must leave at least one physical x cell."
            )
        if len(devices) > physical_cell_count:
            raise ValueError(
                f"Cannot partition {physical_cell_count} physical x cells across "
                f"{len(devices)} devices."
            )
        intervals = _balanced_intervals(physical_cell_count, len(devices))
        if len(devices) > 1 and any(end - begin < halo_width for begin, end in intervals):
            raise ValueError(
                "Every x partition must own at least halo_width cells; "
                f"got intervals={intervals} and halo_width={halo_width}."
            )

        layouts = tuple(
            _build_shard_layout(
                rank=rank,
                device=device,
                global_shape=shape,
                physical_cell_interval=intervals[rank],
                shard_count=len(devices),
                halo_width=halo_width,
                low_pml_cells=low_pml_cells,
                high_pml_cells=high_pml_cells,
            )
            for rank, device in enumerate(devices)
        )
        object.__setattr__(self, "global_shape", shape)
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "decomposition_axis", axis)
        object.__setattr__(self, "halo_width", halo_width)
        object.__setattr__(self, "low_pml_cells", low_pml_cells)
        object.__setattr__(self, "high_pml_cells", high_pml_cells)
        object.__setattr__(self, "shard_layouts", layouts)

    @property
    def num_shards(self) -> int:
        return len(self.shard_layouts)

    @property
    def cell_count(self) -> int:
        return self.global_shape[0] - 1

    @property
    def physical_cell_count(self) -> int:
        return self.cell_count - self.low_pml_cells - self.high_pml_cells

    @property
    def cell_intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(layout.cell_interval for layout in self.shard_layouts)

    @property
    def global_cell_intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                int(layout.global_cell_owned.start),
                int(layout.global_cell_owned.stop),
            )
            for layout in self.shard_layouts
        )

    @property
    def component_shapes(self) -> Mapping[str, tuple[int, int, int]]:
        return MappingProxyType(_component_shapes(self.global_shape))

    def layout(self, rank: int) -> FDTDShardLayout:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer.")
        if rank < 0 or rank >= self.num_shards:
            raise IndexError(f"rank={rank} is out of range for {self.num_shards} shards.")
        return self.shard_layouts[rank]

    def layout_for_device(self, device: str | torch.device) -> FDTDShardLayout:
        normalized = _normalize_cuda_device(device, name="device")
        try:
            rank = self.devices.index(normalized)
        except ValueError as exc:
            raise KeyError(f"Device {normalized} does not participate in this plan.") from exc
        return self.shard_layouts[rank]

    def owner_of_cell(self, cell_index: int) -> int:
        if isinstance(cell_index, bool) or not isinstance(cell_index, int):
            raise TypeError("cell_index must be an integer.")
        if cell_index < 0 or cell_index >= self.cell_count:
            raise IndexError(
                f"cell_index={cell_index} is out of range for extent {self.cell_count}."
            )
        for layout in self.shard_layouts:
            if int(layout.global_cell_owned.start) <= cell_index < int(layout.global_cell_owned.stop):
                return layout.rank
        raise RuntimeError(f"No owner found for cell_index={cell_index}; invalid partition.")

    def owner_of_node(self, node_index: int) -> int:
        if isinstance(node_index, bool) or not isinstance(node_index, int):
            raise TypeError("node_index must be an integer.")
        if node_index < 0 or node_index >= self.global_shape[0]:
            raise IndexError(
                f"node_index={node_index} is out of range for {self.global_shape[0]} x nodes."
            )
        if node_index == self.global_shape[0] - 1:
            return self.num_shards - 1
        return self.owner_of_cell(node_index)

    def owner_of_component_x(self, component: str, x_index: int) -> int:
        normalized = str(component).strip()
        normalized = normalized[0].upper() + normalized[1:].lower() if normalized else normalized
        if normalized in _CELL_EXTENT_COMPONENTS:
            return self.owner_of_cell(x_index)
        if normalized in _NODE_EXTENT_COMPONENTS:
            return self.owner_of_node(x_index)
        raise KeyError(f"Unknown Yee component {component!r}.")




def _balanced_intervals(cell_count: int, shard_count: int) -> tuple[tuple[int, int], ...]:
    quotient, remainder = divmod(cell_count, shard_count)
    intervals = []
    begin = 0
    for rank in range(shard_count):
        length = quotient + (1 if rank < remainder else 0)
        end = begin + length
        intervals.append((begin, end))
        begin = end
    return tuple(intervals)


def _component_shapes(global_shape: tuple[int, int, int]) -> dict[str, tuple[int, int, int]]:
    return {
        component: tuple(size + offset for size, offset in zip(global_shape, offsets))
        for component, offsets in _COMPONENT_SHAPE_OFFSETS.items()
    }


def _slice3(x_begin: int, x_end: int, shape: tuple[int, int, int]) -> Index3:
    return (slice(x_begin, x_end), slice(0, shape[1]), slice(0, shape[2]))


def _build_component_layout(
    *,
    component: str,
    component_shape: tuple[int, int, int],
    rank: int,
    shard_count: int,
    global_cell_owned: slice,
    global_node_owned: slice,
    storage_cell_owned: slice,
    storage_node_owned: slice,
    halo_width: int,
) -> FDTDComponentLayout:
    cell_extent = component in _CELL_EXTENT_COMPONENTS
    global_owned = global_cell_owned if cell_extent else global_node_owned
    storage_owned = storage_cell_owned if cell_extent else storage_node_owned
    owned_begin = int(global_owned.start)
    owned_end = int(global_owned.stop)
    allocation_begin = owned_begin - int(storage_owned.start)
    storage_cell_count = (
        int(global_cell_owned.stop)
        - int(global_cell_owned.start)
        + int(storage_cell_owned.start)
    )
    storage_count = storage_cell_count if cell_extent else storage_cell_count + 1
    allocation_end = allocation_begin + storage_count
    receive_halos: list[FDTDHaloRegion] = []
    send_regions: list[FDTDHaloRegion] = []

    if component in _ELECTRIC_HALO_COMPONENTS:
        if rank < shard_count - 1:
            receive_halos.append(
                FDTDHaloRegion(
                    side="high",
                    neighbor_rank=rank + 1,
                    global_slice=_slice3(
                        owned_end,
                        owned_end + halo_width,
                        component_shape,
                    ),
                    local_slice=_slice3(
                        int(storage_node_owned.stop),
                        int(storage_node_owned.stop) + halo_width,
                        component_shape,
                    ),
                )
            )
        if rank > 0:
            send_regions.append(
                FDTDHaloRegion(
                    side="low",
                    neighbor_rank=rank - 1,
                    global_slice=_slice3(
                        owned_begin,
                        owned_begin + halo_width,
                        component_shape,
                    ),
                    local_slice=_slice3(
                        int(storage_node_owned.start),
                        int(storage_node_owned.start) + halo_width,
                        component_shape,
                    ),
                )
            )
    elif component in _MAGNETIC_HALO_COMPONENTS:
        if rank > 0:
            receive_halos.append(
                FDTDHaloRegion(
                    side="low",
                    neighbor_rank=rank - 1,
                    global_slice=_slice3(
                        owned_begin - halo_width,
                        owned_begin,
                        component_shape,
                    ),
                    local_slice=_slice3(0, halo_width, component_shape),
                )
            )
        if rank < shard_count - 1:
            send_regions.append(
                FDTDHaloRegion(
                    side="high",
                    neighbor_rank=rank + 1,
                    global_slice=_slice3(owned_end - halo_width, owned_end, component_shape),
                    local_slice=_slice3(
                        int(storage_cell_owned.stop) - halo_width,
                        int(storage_cell_owned.stop),
                        component_shape,
                    ),
                )
            )

    allocation_global = _slice3(allocation_begin, allocation_end, component_shape)
    owned_global = _slice3(owned_begin, owned_end, component_shape)
    owned_local = _slice3(
        int(storage_owned.start),
        int(storage_owned.stop),
        component_shape,
    )
    return FDTDComponentLayout(
        component=component,
        global_shape=component_shape,
        owned_global_slice=owned_global,
        allocation_global_slice=allocation_global,
        owned_local_slice=owned_local,
        global_origin=(allocation_begin, 0, 0),
        receive_halos=tuple(receive_halos),
        send_regions=tuple(send_regions),
    )


def _build_shard_layout(
    *,
    rank: int,
    device: str,
    global_shape: tuple[int, int, int],
    physical_cell_interval: tuple[int, int],
    shard_count: int,
    halo_width: int,
    low_pml_cells: int,
    high_pml_cells: int,
) -> FDTDShardLayout:
    shapes = _component_shapes(global_shape)
    physical_cell_begin, physical_cell_end = physical_cell_interval
    total_cell_count = global_shape[0] - 1
    if rank == 0 and physical_cell_begin != 0:
        raise RuntimeError(
            "The first physical partition must start at physical cell zero."
        )
    if (
        rank == shard_count - 1
        and physical_cell_end
        != total_cell_count - low_pml_cells - high_pml_cells
    ):
        raise RuntimeError(
            "The final physical partition must end at the physical cell count."
        )
    global_cell_begin = 0 if rank == 0 else physical_cell_begin + low_pml_cells
    global_cell_end = (
        total_cell_count
        if rank == shard_count - 1
        else physical_cell_end + low_pml_cells
    )
    global_cell_owned = slice(global_cell_begin, global_cell_end)
    terminal_node_count = 1 if rank == shard_count - 1 else 0
    global_node_owned = slice(global_cell_begin, global_cell_end + terminal_node_count)
    storage_owned_start = 0 if rank == 0 else halo_width
    owned_cell_count = (
        int(global_cell_owned.stop) - int(global_cell_owned.start)
    )
    storage_cell_owned = slice(
        storage_owned_start, storage_owned_start + owned_cell_count
    )
    storage_node_owned = slice(
        storage_owned_start,
        storage_owned_start + owned_cell_count + terminal_node_count,
    )

    components = tuple(
        _build_component_layout(
            component=component,
            component_shape=shapes[component],
            rank=rank,
            shard_count=shard_count,
            global_cell_owned=global_cell_owned,
            global_node_owned=global_node_owned,
            storage_cell_owned=storage_cell_owned,
            storage_node_owned=storage_node_owned,
            halo_width=halo_width,
        )
        for component in _COMPONENT_SHAPE_OFFSETS
    )
    physical_faces = {"y_low", "y_high", "z_low", "z_high"}
    if rank == 0:
        physical_faces.add("x_low")
    if rank == shard_count - 1:
        physical_faces.add("x_high")
    return FDTDShardLayout(
        rank=rank,
        device=device,
        global_shape=global_shape,
        physical_cell_begin=physical_cell_begin,
        physical_cell_end=physical_cell_end,
        global_cell_owned=global_cell_owned,
        global_node_owned=global_node_owned,
        storage_cell_owned=storage_cell_owned,
        storage_node_owned=storage_node_owned,
        halo_width=halo_width,
        component_layouts=components,
        physical_faces=frozenset(physical_faces),
    )


__all__ = [
    "FDTDComponentLayout",
    "FDTDHaloRegion",
    "FDTDParallelConfig",
    "FDTDPartitionPlan",
    "FDTDShardLayout",
]
