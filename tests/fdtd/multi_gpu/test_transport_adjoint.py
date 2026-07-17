"""Reverse-halo transport unit tests (synthetic tensors, two CUDA devices).

The forward Yee x halos are plain plane copies (owner -> neighbour ghost). Their
transpose must accumulate the neighbour's ghost adjoint plane back into the owner
and then zero the ghost, so the fused reverse kernels keep the ghost-adjoint-zero
invariant. These tests pin (1) the discrete-transpose pairing identity against the
forward exchange, (2) bitwise determinism across repeats, and (3) that no
allocation happens per reverse step after warmup.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.distributed.transport import CudaP2PHaloTransport
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan


_SENTINEL = -91.0


def _make_shards(devices):
    plan = FDTDPartitionPlan(global_shape=(10, 7, 6), devices=devices)
    shards = []
    for layout in plan.shard_layouts:
        device = torch.device(layout.device)
        with torch.cuda.device(device):
            solver = SimpleNamespace(
                Ey=torch.full(layout.component("Ey").local_shape, _SENTINEL, device=device),
                Ez=torch.full(layout.component("Ez").local_shape, _SENTINEL, device=device),
                Hy=torch.full(layout.component("Hy").local_shape, _SENTINEL, device=device),
                Hz=torch.full(layout.component("Hz").local_shape, _SENTINEL, device=device),
            )
            shards.append(
                SimpleNamespace(
                    rank=layout.rank,
                    device=device,
                    layout=layout,
                    solver=solver,
                    compute_stream=torch.cuda.Stream(device=device),
                    communication_stream=torch.cuda.Stream(device=device, priority=-1),
                    electric_ready=torch.cuda.Event(),
                    electric_received=torch.cuda.Event(),
                    magnetic_ready=torch.cuda.Event(),
                    magnetic_received=torch.cuda.Event(),
                    halo_hy_low=solver.Hy[0] if layout.rank > 0 else None,
                    halo_hz_low=solver.Hz[0] if layout.rank > 0 else None,
                )
            )
    return plan, tuple(shards)


def _order_producers_into_compute(shards):
    """Order the transport's compute-stream events after this test's producer writes.

    The forward/adjoint exchanges record ``magnetic_ready``/``electric_ready`` on
    each shard's compute stream, matching the production contract that every field
    producer runs on that stream. These tests instead write their synthetic input
    planes (owner cells, ghost cotangents) on the device default stream, so the
    recorded events do not order after those writes; under non-blocking streams the
    comm-stream staging copy could read stale planes. Making each compute stream
    wait on the default stream restores the producer -> event ordering the
    transport relies on, without changing the (correct) transport itself.
    """

    for shard in shards:
        with torch.cuda.device(shard.device):
            shard.compute_stream.wait_stream(torch.cuda.current_stream(shard.device))


def _make_adjoint_states(shards, *, seed):
    """Independent random per-shard adjoint field planes (padded local shapes)."""

    states = []
    for shard in shards:
        device = shard.device
        gen = torch.Generator(device=device).manual_seed(seed + shard.rank)
        with torch.cuda.device(device):
            states.append(
                {
                    name: torch.rand(
                        getattr(shard.solver, name).shape, generator=gen, device=device
                    )
                    for name in ("Ey", "Ez", "Hy", "Hz")
                }
            )
    return tuple(states)


def test_magnetic_adjoint_is_discrete_transpose_of_forward_exchange(
    cuda_p2p_devices, cuda_memory_cleanup
):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()
    _, shards = _make_shards(cuda_p2p_devices)
    left, right = shards

    # Forward input x lives on the left owner cell plane; run the forward halo and
    # read the received ghost plane on the right neighbour.
    left_last = left.layout.storage_cell_owned.stop - 1
    with torch.cuda.device(left.device):
        x_hy = torch.rand(left.solver.Hy[left_last].shape, device=left.device)
        x_hz = torch.rand(left.solver.Hz[left_last].shape, device=left.device)
    left.solver.Hy[left_last].copy_(x_hy)
    left.solver.Hz[left_last].copy_(x_hz)
    _order_producers_into_compute(shards)
    transport.exchange_magnetic(shards)
    right.magnetic_received.synchronize()
    fx_hy = right.halo_hy_low.clone()
    fx_hz = right.halo_hz_low.clone()

    # Adjoint cotangent y lives on the right ghost adjoint plane; the owner adjoint
    # starts at zero so the accumulation output is exactly the transpose action.
    adjoint_states = _make_adjoint_states(shards, seed=17)
    for state in adjoint_states:
        state["Hy"].zero_()
        state["Hz"].zero_()
    with torch.cuda.device(right.device):
        y_hy = torch.rand(adjoint_states[right.rank]["Hy"][0].shape, device=right.device)
        y_hz = torch.rand(adjoint_states[right.rank]["Hz"][0].shape, device=right.device)
    adjoint_states[right.rank]["Hy"][0].copy_(y_hy)
    adjoint_states[right.rank]["Hz"][0].copy_(y_hz)

    _order_producers_into_compute(shards)
    transport.exchange_magnetic_adjoint(shards, adjoint_states)
    left.compute_stream.synchronize()
    right.compute_stream.synchronize()

    lhs = torch.dot(fx_hy.flatten().double(), y_hy.to(fx_hy.device).flatten().double()) + torch.dot(
        fx_hz.flatten().double(), y_hz.to(fx_hz.device).flatten().double()
    )
    owner_adj_hy = adjoint_states[left.rank]["Hy"][left_last]
    owner_adj_hz = adjoint_states[left.rank]["Hz"][left_last]
    rhs = torch.dot(x_hy.flatten().double(), owner_adj_hy.flatten().double()) + torch.dot(
        x_hz.flatten().double(), owner_adj_hz.flatten().double()
    )
    torch.testing.assert_close(lhs.item(), rhs.item(), rtol=0.0, atol=0.0)

    # The ghost adjoint plane must be zeroed after the transposed send.
    assert torch.count_nonzero(adjoint_states[right.rank]["Hy"][0]) == 0
    assert torch.count_nonzero(adjoint_states[right.rank]["Hz"][0]) == 0


def test_electric_adjoint_is_discrete_transpose_of_forward_exchange(
    cuda_p2p_devices, cuda_memory_cleanup
):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()
    _, shards = _make_shards(cuda_p2p_devices)
    left, right = shards

    right_node = right.layout.storage_node_owned.start
    left_ghost = left.layout.storage_node_owned.stop
    with torch.cuda.device(right.device):
        x_ey = torch.rand(right.solver.Ey[right_node].shape, device=right.device)
        x_ez = torch.rand(right.solver.Ez[right_node].shape, device=right.device)
    right.solver.Ey[right_node].copy_(x_ey)
    right.solver.Ez[right_node].copy_(x_ez)
    _order_producers_into_compute(shards)
    transport.exchange_electric(shards)
    left.electric_received.synchronize()
    fx_ey = left.solver.Ey[left_ghost].clone()
    fx_ez = left.solver.Ez[left_ghost].clone()

    adjoint_states = _make_adjoint_states(shards, seed=29)
    for state in adjoint_states:
        state["Ey"].zero_()
        state["Ez"].zero_()
    with torch.cuda.device(left.device):
        y_ey = torch.rand(adjoint_states[left.rank]["Ey"][left_ghost].shape, device=left.device)
        y_ez = torch.rand(adjoint_states[left.rank]["Ez"][left_ghost].shape, device=left.device)
    adjoint_states[left.rank]["Ey"][left_ghost].copy_(y_ey)
    adjoint_states[left.rank]["Ez"][left_ghost].copy_(y_ez)

    _order_producers_into_compute(shards)
    transport.exchange_electric_adjoint(shards, adjoint_states)
    left.compute_stream.synchronize()
    right.compute_stream.synchronize()

    lhs = torch.dot(fx_ey.flatten().double(), y_ey.to(fx_ey.device).flatten().double()) + torch.dot(
        fx_ez.flatten().double(), y_ez.to(fx_ez.device).flatten().double()
    )
    owner_adj_ey = adjoint_states[right.rank]["Ey"][right_node]
    owner_adj_ez = adjoint_states[right.rank]["Ez"][right_node]
    rhs = torch.dot(x_ey.flatten().double(), owner_adj_ey.flatten().double()) + torch.dot(
        x_ez.flatten().double(), owner_adj_ez.flatten().double()
    )
    torch.testing.assert_close(lhs.item(), rhs.item(), rtol=0.0, atol=0.0)

    assert torch.count_nonzero(adjoint_states[left.rank]["Ey"][left_ghost]) == 0
    assert torch.count_nonzero(adjoint_states[left.rank]["Ez"][left_ghost]) == 0


def _run_reverse_pair(transport, shards, adjoint_states):
    _order_producers_into_compute(shards)
    transport.exchange_magnetic_adjoint(shards, adjoint_states)
    transport.exchange_electric_adjoint(shards, adjoint_states)
    for shard in shards:
        shard.compute_stream.synchronize()
        shard.communication_stream.synchronize()


def test_reverse_halos_are_bitwise_deterministic_across_repeats(
    cuda_p2p_devices, cuda_memory_cleanup
):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()

    def one_pass():
        _, shards = _make_shards(cuda_p2p_devices)
        adjoint_states = _make_adjoint_states(shards, seed=101)
        _run_reverse_pair(transport, shards, adjoint_states)
        left, right = shards
        left_last = left.layout.storage_cell_owned.stop - 1
        right_node = right.layout.storage_node_owned.start
        return (
            adjoint_states[left.rank]["Hy"][left_last].clone(),
            adjoint_states[left.rank]["Hz"][left_last].clone(),
            adjoint_states[right.rank]["Ey"][right_node].clone(),
            adjoint_states[right.rank]["Ez"][right_node].clone(),
        )

    first = one_pass()
    second = one_pass()
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_reverse_halos_reuse_staging_without_allocator_growth(
    cuda_p2p_devices, cuda_memory_cleanup
):
    transport = CudaP2PHaloTransport(cuda_p2p_devices)
    transport.preflight()
    _, shards = _make_shards(cuda_p2p_devices)
    adjoint_states = _make_adjoint_states(shards, seed=7)

    # Warm up peer mappings and staging allocation before snapshotting.
    transport.prepare_adjoint_staging(shards, adjoint_states)
    _run_reverse_pair(transport, shards, adjoint_states)

    allocated_before = tuple(torch.cuda.memory_allocated(device) for device in cuda_p2p_devices)
    for _ in range(32):
        _run_reverse_pair(transport, shards, adjoint_states)
    allocated_after = tuple(torch.cuda.memory_allocated(device) for device in cuda_p2p_devices)
    assert allocated_after == allocated_before
