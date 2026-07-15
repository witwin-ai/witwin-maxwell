from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from witwin.maxwell.fdtd_parallel import FDTDParallelConfig, FDTDPartitionPlan


def _plan(*, shape=(11, 6, 5), devices=("cuda:0", "cuda:1", "cuda:2")):
    return FDTDPartitionPlan(global_shape=shape, devices=devices)


def _x_extent(index3):
    return index3[0].start, index3[0].stop


def test_parallel_config_normalizes_without_touching_cuda_runtime():
    config = FDTDParallelConfig(
        devices=(torch.device("cuda:1"), "cuda:0"),
        decomposition_axis=" X ",
        transport=" CUDA_P2P ",
        gather_fields=True,
    )

    assert config.devices == ("cuda:1", "cuda:0")
    assert config.decomposition_axis == "x"
    assert config.transport == "cuda_p2p"
    assert config.result_device == "cuda:1"
    assert config.gather_fields is True
    with pytest.raises(FrozenInstanceError):
        config.transport = "auto"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"devices": ("cuda:0",)}, ValueError, "at least 2"),
        ({"devices": ("cuda:0", "cuda:0")}, ValueError, "unique"),
        ({"devices": ("cuda:0", "cpu")}, ValueError, "CUDA"),
        ({"devices": ("cuda", "cuda:1")}, ValueError, "explicit CUDA index"),
        ({"devices": ("cuda:0", "cuda:1"), "decomposition_axis": "y"}, ValueError, "must be 'x'"),
        ({"devices": ("cuda:0", "cuda:1"), "transport": "gloo"}, ValueError, "transport"),
        (
            {"devices": ("cuda:0", "cuda:1"), "result_device": "cuda:2"},
            ValueError,
            "participating",
        ),
        ({"devices": ("cuda:0", "cuda:1"), "overlap": 1}, TypeError, "overlap"),
    ],
)
def test_parallel_config_rejects_invalid_structure(kwargs, error, message):
    with pytest.raises(error, match=message):
        FDTDParallelConfig(**kwargs)


def test_balanced_partition_puts_remainder_on_low_ranks():
    plan = _plan()

    assert plan.cell_count == 10
    assert plan.cell_intervals == ((0, 4), (4, 7), (7, 10))
    assert tuple(layout.owned_cell_count for layout in plan.shard_layouts) == (4, 3, 3)
    assert tuple(layout.device for layout in plan.shard_layouts) == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
    )


def test_padded_storage_slices_keep_low_ghost_and_terminal_node_explicit():
    plan = _plan()

    first, middle, last = plan.shard_layouts
    assert first.global_cell_owned == slice(0, 4)
    assert first.global_node_owned == slice(0, 4)
    assert first.storage_cell_owned == slice(0, 4)
    assert first.storage_node_owned == slice(0, 4)

    assert middle.global_cell_owned == slice(4, 7)
    assert middle.global_node_owned == slice(4, 7)
    assert middle.storage_cell_owned == slice(1, 4)
    assert middle.storage_node_owned == slice(1, 4)

    assert last.global_cell_owned == slice(7, 10)
    assert last.global_node_owned == slice(7, 11)
    assert last.storage_cell_owned == slice(1, 4)
    assert last.storage_node_owned == slice(1, 5)

    assert _x_extent(first.component("Ex").allocation_global_slice) == (0, 4)
    assert _x_extent(first.component("Hx").allocation_global_slice) == (0, 5)
    assert _x_extent(middle.component("Ex").allocation_global_slice) == (3, 7)
    assert _x_extent(middle.component("Hx").allocation_global_slice) == (3, 8)
    assert _x_extent(last.component("Ex").allocation_global_slice) == (6, 10)
    assert _x_extent(last.component("Hx").allocation_global_slice) == (6, 11)


def test_outer_shards_include_pml_while_balancing_only_physical_cells():
    plan = FDTDPartitionPlan(
        global_shape=(15, 6, 5),
        devices=("cuda:0", "cuda:1", "cuda:2"),
        low_pml_cells=2,
        high_pml_cells=2,
    )

    assert plan.physical_cell_count == 10
    assert plan.cell_intervals == ((0, 4), (4, 7), (7, 10))
    assert plan.global_cell_intervals == ((0, 6), (6, 9), (9, 14))

    first, middle, last = plan.shard_layouts
    assert (first.physical_cell_begin, first.physical_cell_end) == (0, 4)
    assert first.global_cell_owned == slice(0, 6)
    assert first.global_node_owned == slice(0, 6)

    assert (middle.physical_cell_begin, middle.physical_cell_end) == (4, 7)
    assert middle.global_cell_owned == slice(6, 9)
    assert middle.global_node_owned == slice(6, 9)

    assert (last.physical_cell_begin, last.physical_cell_end) == (7, 10)
    assert last.global_cell_owned == slice(9, 14)
    assert last.global_node_owned == slice(9, 15)

    assert plan.owner_of_cell(0) == 0
    assert plan.owner_of_cell(13) == 2
    assert _x_extent(first.component("Ex").owned_global_slice) == (0, 6)
    assert _x_extent(last.component("Ey").owned_global_slice) == (9, 15)


def test_all_six_component_shapes_and_owned_extents_follow_yee_staggering():
    plan = _plan(shape=(11, 6, 5))
    assert dict(plan.component_shapes) == {
        "Ex": (10, 6, 5),
        "Ey": (11, 5, 5),
        "Ez": (11, 6, 4),
        "Hx": (11, 5, 4),
        "Hy": (10, 6, 4),
        "Hz": (10, 5, 5),
    }

    first, middle, last = plan.shard_layouts
    for component in ("Ex", "Hy", "Hz"):
        assert _x_extent(first.component(component).owned_global_slice) == (0, 4)
        assert _x_extent(middle.component(component).owned_global_slice) == (4, 7)
        assert _x_extent(last.component(component).owned_global_slice) == (7, 10)
    for component in ("Ey", "Ez", "Hx"):
        assert _x_extent(first.component(component).owned_global_slice) == (0, 4)
        assert _x_extent(middle.component(component).owned_global_slice) == (4, 7)
        assert _x_extent(last.component(component).owned_global_slice) == (7, 11)


def test_interface_and_terminal_values_have_exactly_one_owner():
    plan = _plan()

    for component, x_extent in plan.component_shapes.items():
        for x_index in range(x_extent[0]):
            owners = [
                shard.rank
                for shard in plan.shard_layouts
                if shard.component(component).owns((x_index, 0, 0))
            ]
            assert owners == [plan.owner_of_component_x(component, x_index)]

    assert plan.owner_of_node(4) == 1
    assert plan.owner_of_node(7) == 2
    assert plan.owner_of_node(10) == 2
    assert plan.owner_of_cell(3) == 0
    assert plan.owner_of_cell(4) == 1


def test_tangential_halos_and_send_regions_match_half_step_dependencies():
    first, middle, last = _plan().shard_layouts

    for component in ("Ey", "Ez"):
        first_component = first.component(component)
        middle_component = middle.component(component)
        last_component = last.component(component)
        assert _x_extent(first_component.high_halo.global_slice) == (4, 5)
        assert first_component.high_halo.neighbor_rank == 1
        assert first_component.low_halo is None
        assert _x_extent(middle_component.low_send.global_slice) == (4, 5)
        assert middle_component.low_send.neighbor_rank == 0
        assert _x_extent(middle_component.high_halo.global_slice) == (7, 8)
        assert last_component.high_halo is None
        assert _x_extent(last_component.low_send.global_slice) == (7, 8)

    for component in ("Hy", "Hz"):
        first_component = first.component(component)
        middle_component = middle.component(component)
        last_component = last.component(component)
        assert first_component.low_halo is None
        assert _x_extent(first_component.high_send.global_slice) == (3, 4)
        assert first_component.high_send.neighbor_rank == 1
        assert _x_extent(middle_component.low_halo.global_slice) == (3, 4)
        assert middle_component.low_halo.neighbor_rank == 0
        assert _x_extent(middle_component.high_send.global_slice) == (6, 7)
        assert _x_extent(last_component.low_halo.global_slice) == (6, 7)
        assert last_component.high_send is None

    for shard in (first, middle, last):
        for component in ("Ex", "Hx"):
            layout = shard.component(component)
            assert layout.receive_halos == ()
            assert layout.send_regions == ()
            assert layout.local_shape[0] >= layout.owned_shape[0]


def test_global_local_round_trip_includes_read_only_halo_when_requested():
    middle = _plan().layout(1)
    ey = middle.component("Ey")
    hy = middle.component("Hy")

    assert ey.global_origin == (3, 0, 0)
    assert ey.global_to_local((4, 2, 3)) == (1, 2, 3)
    assert ey.local_to_global((2, 1, 1)) == (5, 1, 1)
    assert ey.global_to_local((7, 1, 1), include_halo=True) == (4, 1, 1)
    with pytest.raises(IndexError, match="owned extent"):
        ey.global_to_local((7, 1, 1))

    assert hy.global_origin == (3, 0, 0)
    assert hy.global_to_local((4, 2, 2)) == (1, 2, 2)
    assert hy.global_to_local((3, 2, 2), include_halo=True) == (0, 2, 2)
    assert hy.local_to_global((0, 2, 2), include_halo=True) == (3, 2, 2)


def test_only_outer_x_shards_own_x_physical_faces():
    first, middle, last = _plan().shard_layouts

    assert first.owns_physical_face("x", "low")
    assert not first.owns_physical_face("x_high")
    assert not middle.owns_physical_face("x_low")
    assert not middle.owns_physical_face("x_high")
    assert last.owns_physical_face("x", "high")
    for shard in (first, middle, last):
        assert shard.owns_physical_face("y_low")
        assert shard.owns_physical_face("y", "high")
        assert shard.owns_physical_face("z_low")
        assert shard.owns_physical_face("z", "high")


def test_single_shard_plan_owns_both_x_faces_and_has_no_halos():
    plan = _plan(devices=("cuda:4",))
    shard = plan.layout_for_device("cuda:4")

    assert shard.cell_interval == (0, 10)
    assert shard.owns_physical_face("x_low")
    assert shard.owns_physical_face("x_high")
    assert all(not layout.receive_halos for layout in shard.component_layouts)
    assert all(not layout.send_regions for layout in shard.component_layouts)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"global_shape": (1, 4, 4), "devices": ("cuda:0",)}, ">= 2"),
        ({"global_shape": (4, 4), "devices": ("cuda:0",)}, "global_shape"),
        (
            {"global_shape": (3, 4, 4), "devices": ("cuda:0", "cuda:1", "cuda:2")},
            "Cannot partition",
        ),
        (
            {
                "global_shape": (6, 4, 4),
                "devices": ("cuda:0", "cuda:1", "cuda:2"),
                "halo_width": 2,
            },
            "halo_width",
        ),
    ],
)
def test_partition_rejects_illegal_extents(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FDTDPartitionPlan(**kwargs)


def test_partition_and_nested_layouts_are_frozen():
    plan = _plan()
    with pytest.raises(FrozenInstanceError):
        plan.halo_width = 2
    with pytest.raises(FrozenInstanceError):
        plan.layout(0).cell_interval = (1, 4)
    with pytest.raises(FrozenInstanceError):
        plan.layout(0).component("Ex").global_origin = (1, 0, 0)
