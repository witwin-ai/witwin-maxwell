from __future__ import annotations

from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.distributed.transport import CudaP2PHaloTransport
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan


_SENTINEL = -91.0


def _make_transport_shards(devices: tuple[torch.device, torch.device]):
    plan = FDTDPartitionPlan(global_shape=(10, 7, 6), devices=devices)
    shards = []
    for layout in plan.shard_layouts:
        device = torch.device(layout.device)
        with torch.cuda.device(device):
            solver = SimpleNamespace(
                Ey=torch.full(
                    layout.component("Ey").local_shape,
                    _SENTINEL,
                    device=device,
                    dtype=torch.float32,
                ),
                Ez=torch.full(
                    layout.component("Ez").local_shape,
                    _SENTINEL,
                    device=device,
                    dtype=torch.float32,
                ),
                Hy=torch.full(
                    layout.component("Hy").local_shape,
                    _SENTINEL,
                    device=device,
                    dtype=torch.float32,
                ),
                Hz=torch.full(
                    layout.component("Hz").local_shape,
                    _SENTINEL,
                    device=device,
                    dtype=torch.float32,
                ),
            )
            compute_stream = torch.cuda.Stream(device=device)
            communication_stream = torch.cuda.Stream(device=device, priority=-1)
            shards.append(
                SimpleNamespace(
                    rank=layout.rank,
                    device=device,
                    layout=layout,
                    solver=solver,
                    compute_stream=compute_stream,
                    communication_stream=communication_stream,
                    electric_ready=torch.cuda.Event(),
                    electric_received=torch.cuda.Event(),
                    magnetic_ready=torch.cuda.Event(),
                    magnetic_received=torch.cuda.Event(),
                    halo_hy_low=solver.Hy[0] if layout.rank > 0 else None,
                    halo_hz_low=solver.Hz[0] if layout.rank > 0 else None,
                )
            )
    return plan, tuple(shards)


def _tag_plane(plane: torch.Tensor, *, offset: float) -> torch.Tensor:
    values = torch.arange(plane.numel(), device=plane.device, dtype=plane.dtype)
    return (values.reshape_as(plane) + offset).contiguous()


def test_cuda_p2p_transport_round_trips_tagged_asymmetric_y_halos(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()
    plan, shards = _make_transport_shards(cuda_p2p_devices)
    left, right = shards

    right_node = right.layout.storage_node_owned.start
    left_ghost = left.layout.storage_node_owned.stop
    ey_tag = _tag_plane(right.solver.Ey[right_node], offset=1_000.0)
    ez_tag = _tag_plane(right.solver.Ez[right_node], offset=2_000.0)
    right.solver.Ey[right_node].copy_(ey_tag)
    right.solver.Ez[right_node].copy_(ez_tag)
    untouched_left_ey = left.solver.Ey[left_ghost - 1].clone()

    transport.exchange_electric(shards)
    left.electric_received.synchronize()

    torch.testing.assert_close(
        left.solver.Ey[left_ghost], ey_tag.to(left.device), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        left.solver.Ez[left_ghost], ez_tag.to(left.device), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        left.solver.Ey[left_ghost - 1], untouched_left_ey, rtol=0.0, atol=0.0
    )

    left_last = left.layout.storage_cell_owned.stop - 1
    hy_tag = _tag_plane(left.solver.Hy[left_last], offset=3_000.0)
    hz_tag = _tag_plane(left.solver.Hz[left_last], offset=4_000.0)
    left.solver.Hy[left_last].copy_(hy_tag)
    left.solver.Hz[left_last].copy_(hz_tag)
    untouched_right_hy = right.solver.Hy[1].clone()

    transport.exchange_magnetic(shards)
    right.magnetic_received.synchronize()

    torch.testing.assert_close(right.halo_hy_low, hy_tag.to(right.device), rtol=0.0, atol=0.0)
    torch.testing.assert_close(right.halo_hz_low, hz_tag.to(right.device), rtol=0.0, atol=0.0)
    torch.testing.assert_close(right.solver.Hy[1], untouched_right_hy, rtol=0.0, atol=0.0)

    assert plan.layout(0).component("Ey").high_halo is not None
    assert plan.layout(1).component("Hy").low_halo is not None


def test_cuda_p2p_transport_reuses_storage_without_torch_allocator_growth(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()
    _, shards = _make_transport_shards(cuda_p2p_devices)
    left, right = shards

    # Warm up peer mappings and event dependencies before taking allocator snapshots.
    transport.exchange_electric(shards)
    transport.exchange_magnetic(shards)
    left.electric_received.synchronize()
    right.magnetic_received.synchronize()

    pointers = (
        left.solver.Ey.data_ptr(),
        left.solver.Ez.data_ptr(),
        right.solver.Hy.data_ptr(),
        right.solver.Hz.data_ptr(),
    )
    allocated_before = tuple(torch.cuda.memory_allocated(device) for device in cuda_p2p_devices)
    for _ in range(32):
        transport.exchange_electric(shards)
        transport.exchange_magnetic(shards)
    left.electric_received.synchronize()
    right.magnetic_received.synchronize()
    allocated_after = tuple(torch.cuda.memory_allocated(device) for device in cuda_p2p_devices)

    assert allocated_after == allocated_before
    assert pointers == (
        left.solver.Ey.data_ptr(),
        left.solver.Ez.data_ptr(),
        right.solver.Hy.data_ptr(),
        right.solver.Hz.data_ptr(),
    )


def test_cuda_p2p_transport_waits_for_pending_destination_writes(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    """A delayed compute write must finish before the receive copy overwrites it."""

    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()
    _, shards = _make_transport_shards(cuda_p2p_devices)
    left, right = shards

    right_node = right.layout.storage_node_owned.start
    left_ghost = left.layout.storage_node_owned.stop
    ey_tag = _tag_plane(right.solver.Ey[right_node], offset=11_000.0)
    ez_tag = _tag_plane(right.solver.Ez[right_node], offset=12_000.0)
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)

    with torch.cuda.device(right.device), torch.cuda.stream(right.compute_stream):
        right.solver.Ey[right_node].copy_(ey_tag)
        right.solver.Ez[right_node].copy_(ez_tag)
    with torch.cuda.device(left.device), torch.cuda.stream(left.compute_stream):
        torch.cuda._sleep(20_000_000)
        left.solver.Ey[left_ghost].fill_(-17.0)
        left.solver.Ez[left_ghost].fill_(-17.0)

    transport.exchange_electric(shards)
    left.electric_received.synchronize()
    torch.testing.assert_close(
        left.solver.Ey[left_ghost], ey_tag.to(left.device), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        left.solver.Ez[left_ghost], ez_tag.to(left.device), rtol=0.0, atol=0.0
    )

    left_last = left.layout.storage_cell_owned.stop - 1
    hy_tag = _tag_plane(left.solver.Hy[left_last], offset=13_000.0)
    hz_tag = _tag_plane(left.solver.Hz[left_last], offset=14_000.0)
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)

    with torch.cuda.device(left.device), torch.cuda.stream(left.compute_stream):
        left.solver.Hy[left_last].copy_(hy_tag)
        left.solver.Hz[left_last].copy_(hz_tag)
    with torch.cuda.device(right.device), torch.cuda.stream(right.compute_stream):
        torch.cuda._sleep(20_000_000)
        right.halo_hy_low.fill_(-23.0)
        right.halo_hz_low.fill_(-23.0)

    transport.exchange_magnetic(shards)
    right.magnetic_received.synchronize()
    torch.testing.assert_close(right.halo_hy_low, hy_tag.to(right.device), rtol=0.0, atol=0.0)
    torch.testing.assert_close(right.halo_hz_low, hz_tag.to(right.device), rtol=0.0, atol=0.0)


def test_cuda_p2p_transport_reports_bidirectional_neighbor_capability(cuda_p2p_devices):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()

    assert transport.name == "cuda_p2p"
    assert len(transport.links) == 2
    assert {link.direction for link in transport.links} == {
        "electric_right_to_left",
        "magnetic_left_to_right",
    }
    topology = transport.topology
    assert topology["kind"] == "cuda_p2p"
    assert topology["neighbor_pairs"] == (
        {
            "devices": ("cuda:0", "cuda:1"),
            "peer_left_to_right": True,
            "peer_right_to_left": True,
        },
    )
