from __future__ import annotations

import numpy as np
import pytest
import torch

from witwin.maxwell.fdtd.distributed.monitor_merge import (
    _crop_aligned_to_physical_bounds,
    merge_sharded_monitor_payloads,
)
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan, FDTDShardLayout


_FREQUENCY = 1.0e9
_X_NODES = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
_Y_NODES = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
_Z_NODES = np.linspace(-0.3, 0.3, 7, dtype=np.float64)
_PHYSICAL_BOUNDS = ((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))
_NODE_X_COMPONENTS = frozenset(("Ey", "Ez", "Hx"))


def _plan() -> FDTDPartitionPlan:
    return FDTDPartitionPlan(
        global_shape=(_X_NODES.size, _Y_NODES.size, _Z_NODES.size),
        devices=("cuda:0", "cuda:1"),
        low_pml_cells=1,
        high_pml_cells=1,
    )


def _layout_map(plan: FDTDPartitionPlan) -> dict[int, FDTDShardLayout]:
    return {layout.rank: layout for layout in plan.shard_layouts}


def _global_component_coords(component: str, axis: str) -> np.ndarray:
    if axis == "x":
        nodes = _X_NODES
        is_node = component in _NODE_X_COMPONENTS
    elif axis == "z":
        nodes = _Z_NODES
        is_node = component in {"Ex", "Ey", "Hz"}
    else:
        raise ValueError(f"Unsupported test axis {axis!r}.")
    return nodes if is_node else 0.5 * (nodes[:-1] + nodes[1:])


def _local_coords(
    layout: FDTDShardLayout,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    component_layout = layout.component(component)
    x = _global_component_coords(component, "x")
    allocation = component_layout.allocation_global_slice[0]
    return np.array(x[allocation], copy=True), _global_component_coords(component, "z")


def _tagged_component(
    layout: FDTDShardLayout,
    component: str,
    *,
    device: str | torch.device = "cpu",
    offset: float = 0.0,
) -> dict[str, object]:
    component_layout = layout.component(component)
    x, z = _local_coords(layout, component)
    data = torch.full(
        (x.size, z.size),
        complex(9000.0 + layout.rank, 0.0),
        device=device,
        dtype=torch.complex64,
    )
    owned_local = component_layout.owned_local_slice[0]
    owned_global = component_layout.owned_global_slice[0]
    values = torch.arange(
        int(owned_global.start),
        int(owned_global.stop),
        device=device,
        dtype=torch.float32,
    )
    data[owned_local] = (values + float(offset))[:, None].to(torch.complex64)
    return {"data": data, "coords": (x, z)}


def _tagged_plane(layout: FDTDShardLayout) -> dict[str, object]:
    fields = ("Ex", "Ez")
    return {
        "kind": "plane",
        "monitor_type": "plane",
        "fields": fields,
        "components": {
            "Ex": _tagged_component(layout, "Ex"),
            "Ez": _tagged_component(layout, "Ez", offset=100.0),
        },
        "samples": 8,
        "frequency": _FREQUENCY,
        "frequencies": (_FREQUENCY,),
        "axis": "y",
        "position": 0.0,
        "compute_flux": False,
        "normal_direction": "+",
        "Ex": torch.full((1, 1), complex(7000.0 + layout.rank, 0.0)),
        "Ez": torch.full((1, 1), complex(7000.0 + layout.rank, 0.0)),
    }


def _constant_vector_plane(
    layout: FDTDShardLayout,
    *,
    device: str | torch.device,
    monitor_type: str,
    compute_flux: bool,
) -> dict[str, object]:
    constants = {"Ex": 0.0, "Ez": 1.0, "Hx": 2.0, "Hz": 0.0}
    components = {}
    for component, value in constants.items():
        component_layout = layout.component(component)
        x, z = _local_coords(layout, component)
        data = torch.full(
            (x.size, z.size),
            complex(5000.0 + layout.rank, 0.0),
            device=device,
            dtype=torch.complex64,
        )
        data[component_layout.owned_local_slice[0]] = complex(value, 0.0)
        components[component] = {"data": data, "coords": (x, z)}

    payload: dict[str, object] = {
        "kind": "plane",
        "monitor_type": monitor_type,
        "fields": tuple(constants),
        "components": components,
        "samples": 8,
        "frequency": _FREQUENCY,
        "frequencies": (_FREQUENCY,),
        "axis": "y",
        "position": 0.0,
        "compute_flux": compute_flux,
        "normal_direction": "+",
        "mode_spec": {"mode_index": 0} if monitor_type == "mode" else None,
        "flux": torch.tensor(1234.0, device=device),
        "power": torch.tensor(1234.0, device=device),
    }
    for component in constants:
        payload[component] = torch.full(
            (1, 1),
            complex(6000.0 + layout.rank, 0.0),
            device=device,
            dtype=torch.complex64,
        )
    return payload


def test_tiled_plane_discards_node_high_ghost_and_cell_low_ghost():
    plan = _plan()
    layouts = _layout_map(plan)
    payloads = tuple(
        (layout.rank, {"plane": _tagged_plane(layout)})
        for layout in reversed(plan.shard_layouts)
    )

    output = merge_sharded_monitor_payloads(
        ("plane",),
        payloads,
        shard_layouts=layouts,
        physical_bounds=_PHYSICAL_BOUNDS,
        result_device="cpu",
    )["plane"]

    ex = output["components"]["Ex"]
    ez = output["components"]["Ez"]
    np.testing.assert_array_equal(
        ex["coords"][0],
        0.5 * (_X_NODES[:-1] + _X_NODES[1:]),
    )
    np.testing.assert_array_equal(ez["coords"][0], _X_NODES)
    torch.testing.assert_close(
        ex["data"].real[:, 0],
        torch.arange(0, 10, dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        ez["data"].real[:, 0],
        torch.arange(0, 11, dtype=torch.float32) + 100.0,
        rtol=0.0,
        atol=0.0,
    )
    assert float(ez["data"][5, 0].real) == 105.0


def test_flux_recomputes_from_global_owned_components_on_result_gpu(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    plan = _plan()
    layouts = _layout_map(plan)
    x_plane = {
        "kind": "plane",
        "monitor_type": "plane",
        "fields": ("Ez",),
        "axis": "x",
        "position": 0.1,
        "compute_flux": False,
        "normal_direction": "+",
        "data": torch.ones((2, 2), device=cuda_p2p_devices[1]),
    }
    payloads = tuple(
        (
            layout.rank,
            {
                "flux": _constant_vector_plane(
                    layout,
                    device=cuda_p2p_devices[layout.rank],
                    monitor_type="plane",
                    compute_flux=True,
                )
            }
            | ({"x_plane": x_plane} if layout.rank == 1 else {}),
        )
        for layout in reversed(plan.shard_layouts)
    )

    merged = merge_sharded_monitor_payloads(
        ("x_plane", "flux"),
        payloads,
        shard_layouts=layouts,
        physical_bounds=_PHYSICAL_BOUNDS,
        result_device=cuda_p2p_devices[0],
    )
    assert tuple(merged) == ("x_plane", "flux")
    assert merged["x_plane"]["data"].device == cuda_p2p_devices[0]
    output = merged["flux"]

    assert output["flux"].device == cuda_p2p_devices[0]
    assert output["power"].device == cuda_p2p_devices[0]
    torch.testing.assert_close(
        output["flux"],
        torch.tensor(0.32, device=cuda_p2p_devices[0]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(output["power"], output["flux"])
    np.testing.assert_allclose(output["x"], np.linspace(-0.35, 0.35, 8))
    np.testing.assert_allclose(output["z"], np.linspace(-0.15, 0.15, 4))
    assert output["Ez"].shape == (8, 4)
    assert torch.all(output["Ez"] == 1.0)
    assert torch.all(output["Hx"] == 2.0)


def test_tiled_mode_realigns_global_raw_components_after_owned_stitch():
    plan = _plan()
    layouts = _layout_map(plan)
    payloads = tuple(
        (
            layout.rank,
            {
                "mode": _constant_vector_plane(
                    layout,
                    device="cpu",
                    monitor_type="mode",
                    compute_flux=False,
                )
            },
        )
        for layout in reversed(plan.shard_layouts)
    )

    output = merge_sharded_monitor_payloads(
        ("mode",),
        payloads,
        shard_layouts=layouts,
        physical_bounds=_PHYSICAL_BOUNDS,
        result_device="cpu",
    )["mode"]

    assert output["monitor_type"] == "mode"
    assert output["mode_spec"] == {"mode_index": 0}
    assert output["Ez"].shape == (10, 6)
    assert torch.all(output["Ez"] == 1.0)
    assert torch.all(output["Hx"] == 2.0)
    assert float(output["Ez"].real.max()) < 10.0


def test_tiled_plane_requires_layout_for_every_contributing_rank():
    plan = _plan()
    left, right = plan.shard_layouts
    payloads = (
        (left.rank, {"plane": _tagged_plane(left)}),
        (right.rank, {"plane": _tagged_plane(right)}),
    )
    with pytest.raises(ValueError, match="missing shard layouts.*rank 1"):
        merge_sharded_monitor_payloads(
            ("plane",),
            payloads,
            shard_layouts={0: left},
            physical_bounds=_PHYSICAL_BOUNDS,
            result_device="cpu",
        )


@pytest.mark.parametrize(
    ("monitor_type", "compute_flux"),
    (("plane", True), ("mode", False)),
)
def test_x_normal_flux_and_mode_require_exactly_one_owner(monitor_type, compute_flux):
    payload = {
        "kind": "plane",
        "monitor_type": monitor_type,
        "fields": ("Ez",),
        "axis": "x",
        "position": 0.0,
        "compute_flux": compute_flux,
        "normal_direction": "+",
        "data": torch.ones((2, 2)),
    }
    with pytest.raises(RuntimeError, match="x-normal plane.*more than one shard owner"):
        merge_sharded_monitor_payloads(
            ("plane",),
            ((0, {"plane": payload}), (1, {"plane": payload})),
            shard_layouts=_layout_map(_plan()),
            physical_bounds=_PHYSICAL_BOUNDS,
            result_device="cpu",
        )



def test_nm_scale_physical_crop_does_not_admit_pml_samples():
    nm = 1.0e-9
    coords = np.asarray((-150.0, -50.0, 50.0, 150.0), dtype=np.float64) * nm
    aligned = {
        "x": coords,
        "z": coords,
        "fields": {
            "Ez": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        },
    }

    cropped = _crop_aligned_to_physical_bounds(
        "y",
        aligned,
        ((-100.0 * nm, 100.0 * nm), (0.0, 0.0), (-100.0 * nm, 100.0 * nm)),
    )

    np.testing.assert_array_equal(cropped["x"], coords[1:3])
    np.testing.assert_array_equal(cropped["z"], coords[1:3])
    torch.testing.assert_close(
        cropped["fields"]["Ez"],
        torch.tensor(((5.0, 6.0), (9.0, 10.0))),
        rtol=0.0,
        atol=0.0,
    )
