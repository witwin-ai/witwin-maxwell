from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed.solver import DistributedFDTD
from witwin.maxwell.fdtd.distributed.wire import (
    _decode_edge,
    compile_distributed_wire_plan,
)
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig, FDTDPartitionPlan
from witwin.maxwell.scene import prepare_scene


# --------------------------------------------------------------------------- #
# Pure ownership units (no CUDA): a synthetic compiled wire network is enough  #
# to exercise fragment/state ownership, boundary splitting, and completeness.  #
# --------------------------------------------------------------------------- #


def _ex_flat(shape, i, j, k) -> int:
    return (i * shape[1] + j) * shape[2] + k


def _straight_ex_wire_network(*, ny=5, nz=5, nx=9, cells=(2, 3, 4, 5)):
    """One physical Ex segment split into one fragment per crossed x cell.

    ``cells`` are the x cell indices the (x-directed) wire occupies. Every crossed
    Ex edge is both a sampling entry and a deposition target, mirroring the real
    compiler contract, so the synthetic network drives the same ownership code the
    GPU path uses.
    """

    ex_shape = (nx - 1, ny, nz)
    field_shapes = (ex_shape, (nx, ny - 1, nz), (nx, ny, nz - 1))
    offsets = [_ex_flat(ex_shape, cell, 2, 2) for cell in cells]
    count = len(offsets)
    return SimpleNamespace(
        field_shapes=field_shapes,
        segment_count=1,
        edge_components=torch.zeros(count, dtype=torch.int32),
        edge_offsets=torch.tensor(offsets, dtype=torch.int64),
        weights=torch.ones(count, dtype=torch.float64),
        segment_offsets=torch.tensor([0, count], dtype=torch.int64),
        edge_group_offsets=torch.arange(count + 1, dtype=torch.int64),
        target_components=torch.zeros(count, dtype=torch.int32),
        target_offsets=torch.tensor(offsets, dtype=torch.int64),
        contribution_segments=torch.zeros(count, dtype=torch.int64),
        contribution_weights=torch.ones(count, dtype=torch.float64),
        node_capacitance=torch.ones(2, dtype=torch.float64),
    )


def _plan(cells):
    network = _straight_ex_wire_network(cells=cells)
    partition = FDTDPartitionPlan(
        global_shape=(9, 5, 5), devices=("cuda:0", "cuda:1")
    )
    return compile_distributed_wire_plan(network, partition), partition


def test_owner_is_the_shard_of_the_minimum_global_sampling_edge():
    plan, partition = _plan(cells=(2, 3, 4, 5))
    # The minimum (i, j, k, component) edge is Ex cell 2, owned by rank 0.
    assert plan.owner_reference_edge == (2, 2, 2, 0)
    assert plan.owner_rank == partition.owner_of_component_x("Ex", 2) == 0


def test_owner_selection_is_independent_of_sampling_entry_declaration_order():
    network = _straight_ex_wire_network(cells=(2, 3, 4, 5))
    partition = FDTDPartitionPlan(global_shape=(9, 5, 5), devices=("cuda:0", "cuda:1"))
    forward = compile_distributed_wire_plan(network, partition)

    # A different fragment emission order (e.g. the wire declared tip-to-tail)
    # must not move the owner or the per-shard edge sets: ownership is a function
    # of geometry, not declaration order.
    permutation = torch.tensor([3, 1, 0, 2])
    shuffled = SimpleNamespace(
        field_shapes=network.field_shapes,
        segment_count=1,
        edge_components=network.edge_components[permutation],
        edge_offsets=network.edge_offsets[permutation],
        weights=network.weights[permutation],
        segment_offsets=network.segment_offsets,
        edge_group_offsets=network.edge_group_offsets,
        target_components=network.target_components[permutation],
        target_offsets=network.target_offsets[permutation],
        contribution_segments=network.contribution_segments,
        contribution_weights=network.contribution_weights,
        node_capacitance=network.node_capacitance,
    )
    reverse = compile_distributed_wire_plan(shuffled, partition)

    assert forward.owner_rank == reverse.owner_rank
    assert forward.owner_reference_edge == reverse.owner_reference_edge

    def _edge_sets(plan):
        return {
            sample.rank: {int(v) for v in sample.edge_offsets.tolist()}
            for sample in plan.sample_plans
        }

    assert _edge_sets(forward) == _edge_sets(reverse)


def test_segment_spanning_shards_is_split_into_owned_fragments():
    plan, partition = _plan(cells=(2, 3, 4, 5))
    # The physical segment crosses the x split at cell 4 (rank boundary), so it is
    # split into one fragment group per shard while staying a single segment.
    assert plan.num_segments == 1
    assert plan.cross_shard_segment_count == 1
    by_rank = {sample.rank: sample for sample in plan.sample_plans}
    assert set(by_rank) == {0, 1}
    # rank 0 owns cells 2,3; rank 1 owns cells 4,5 (owner_of_cell splits at 4).
    assert int(by_rank[0].edge_offsets.numel()) == 2
    assert int(by_rank[1].edge_offsets.numel()) == 2
    for sample in plan.sample_plans:
        # Every shard's CSR still spans the full segment count with the total
        # length equal to that shard's owned entry count.
        assert sample.segment_offsets.numel() == plan.num_segments + 1
        assert int(sample.segment_offsets[-1]) == int(sample.edge_offsets.numel())


def test_every_yee_edge_has_exactly_one_owner_and_is_never_dropped():
    cells = (2, 3, 4, 5)
    plan, partition = _plan(cells=cells)
    ex_shape = (8, 5, 5)
    global_targets = {_ex_flat(ex_shape, cell, 2, 2): cell for cell in cells}

    owners: dict[tuple[int, int], int] = {}
    for deposit in plan.deposit_plans:
        layout = partition.shard_layouts[deposit.rank].component("Ex")
        for component, local_offset in zip(
            deposit.target_components.tolist(), deposit.target_offsets.tolist()
        ):
            # Map the shard-local flat offset back to a global (i, j, k) and confirm
            # it decodes to a real crossed cell owned by exactly one shard.
            lx, ly, lz = layout.local_shape
            li = local_offset // (ly * lz)
            gi = li + int(layout.global_origin[0])
            key = (int(component), gi)
            assert key not in owners, "a Yee edge was claimed by two shards"
            owners[key] = deposit.rank

    # Exactly the four crossed cells, each owned once.
    owned_cells = sorted(gi for (_component, gi) in owners)
    assert owned_cells == list(cells)


def test_owner_reference_edge_prefers_lower_x_then_component():
    # Two co-located transverse edges at the same (i, j, k): the component tiebreak
    # must be deterministic (Ex < Ey < Ez), matching the circuit owner convention.
    ex_shape = (8, 5, 5)
    ey_shape = (9, 4, 5)
    network = SimpleNamespace(
        field_shapes=(ex_shape, ey_shape, (9, 5, 4)),
        segment_count=1,
        edge_components=torch.tensor([1, 0], dtype=torch.int32),
        edge_offsets=torch.tensor(
            [_ex_flat(ey_shape, 2, 1, 2), _ex_flat(ex_shape, 2, 1, 2)],
            dtype=torch.int64,
        ),
        weights=torch.ones(2, dtype=torch.float64),
        segment_offsets=torch.tensor([0, 2], dtype=torch.int64),
        edge_group_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
        target_components=torch.tensor([1, 0], dtype=torch.int32),
        target_offsets=torch.tensor(
            [_ex_flat(ey_shape, 2, 1, 2), _ex_flat(ex_shape, 2, 1, 2)],
            dtype=torch.int64,
        ),
        contribution_segments=torch.zeros(2, dtype=torch.int64),
        contribution_weights=torch.ones(2, dtype=torch.float64),
        node_capacitance=torch.ones(2, dtype=torch.float64),
    )
    partition = FDTDPartitionPlan(global_shape=(9, 5, 5), devices=("cuda:0", "cuda:1"))
    plan = compile_distributed_wire_plan(network, partition)
    assert plan.owner_reference_edge == (2, 1, 2, 0)


def test_decode_edge_round_trips_flat_offsets():
    shape = (8, 5, 5)
    assert _decode_edge(0, _ex_flat(shape, 3, 2, 4), ((8, 5, 5), (9, 4, 5), (9, 5, 4))) == (
        3,
        2,
        4,
    )


# --------------------------------------------------------------------------- #
# Fail-closed guards (no CUDA hardware required: rejection is static).         #
# --------------------------------------------------------------------------- #

_FREQUENCY = 2.0e9


def _wire_scene(*, boundary=None, trainable_radius=False, circuit=None, device="cuda:0"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none() if boundary is None else boundary,
        device=device,
        circuits=() if circuit is None else (circuit,),
    )
    radius = (
        torch.tensor(2.0e-3, requires_grad=True) if trainable_radius else 2.0e-3
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((-0.08, 0.0, 0.0), (0.08, 0.0, 0.0)),
            radius=radius,
            conductor=mw.WireConductor.pec(),
            snap="strict",
        )
    )
    return scene


def _parallel(devices=("cuda:0", "cuda:1")):
    return FDTDParallelConfig(devices=devices, transport="cuda_p2p", result_device=devices[0])


def test_distributed_wire_with_pml_boundary_is_rejected():
    scene = _wire_scene(boundary=mw.BoundarySpec.pml(num_layers=4))
    with pytest.raises(NotImplementedError, match="distributed CPML"):
        DistributedFDTD(scene, frequency=_FREQUENCY, parallel=_parallel())


def test_distributed_wire_with_circuit_is_rejected():
    circuit = mw.Circuit("c")
    node = circuit.node("n")
    circuit.add(mw.Resistor("R", node, circuit.ground, 50.0))
    circuit.bind_port("p", positive=node, negative=circuit.ground)
    scene = _wire_scene(circuit=circuit)
    with pytest.raises(NotImplementedError, match="embedded network or lumped circuit"):
        DistributedFDTD(scene, frequency=_FREQUENCY, parallel=_parallel())


def test_trainable_wire_plus_parallel_is_rejected_at_construction():
    scene = _wire_scene(trainable_radius=True)
    with pytest.raises(ValueError, match="forward solve only"):
        DistributedFDTD(scene, frequency=_FREQUENCY, parallel=_parallel())


def test_public_prepare_rejects_trainable_wire_parallel():
    scene = _wire_scene(trainable_radius=True)
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=4),
        parallel=_parallel(),
    )
    with pytest.raises(ValueError, match="thin-wire state"):
        simulation.prepare()


# --------------------------------------------------------------------------- #
# Physical single-vs-two-GPU parity (requires two peer CUDA devices).          #
# --------------------------------------------------------------------------- #

# The deterministic (sorted-by-rank) cross-shard EMF reduction and the per-edge
# deposition are bit-exact against the single-GPU kernels, so a wire crossing the
# split reproduces the single-GPU trajectory bitwise for a short horizon (measured
# below: identical to 0 ULP through 12 steps while the wire already carries ~36 A
# across the split). Beyond ~13 steps the coupled Maxwell+wire system amplifies the
# unavoidable float32 halo-curl reordering chaotically -- a property of any
# distributed FDTD, not of the wire coupling -- so the bitwise gate runs at the
# short horizon and the energy gate relies on the discrete invariant, which stays
# tight (~2e-9) even after pointwise field phases drift.
_TIME_STEPS = 12
_ENERGY_TIME_STEPS = 16


def _straight_crossing_scene():
    """An x-directed PEC wire that crosses the x=0 partition split, plus monitor."""

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((-0.08, 0.0, 0.0), (0.08, 0.0, 0.0)),
            radius=2.0e-3,
            conductor=mw.WireConductor.pec(),
            snap="strict",
        )
    )
    scene.add_source(
        mw.PointDipole(
            name="drive",
            position=(-0.04, 0.0, 0.0),
            polarization="Ex",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
        )
    )
    scene.add_monitor(
        mw.WireMonitor(name="wire_state", wire="wire", frequencies=(_FREQUENCY,))
    )
    return scene


def _closed_loop_crossing_scene():
    """A closed lossless PEC loop in the z=0 plane crossing the x=0 split."""

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="loop",
            points=(
                (-0.08, -0.08, 0.0),
                (0.08, -0.08, 0.0),
                (0.08, 0.08, 0.0),
                (-0.08, 0.08, 0.0),
                (-0.08, -0.08, 0.0),
            ),
            radius=2.0e-3,
            conductor=mw.WireConductor.pec(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            name="drive",
            position=(-0.04, -0.08, 0.0),
            polarization="Ex",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
        )
    )
    scene.add_monitor(
        mw.WireMonitor(name="loop_state", wire="loop", frequencies=(_FREQUENCY,))
    )
    return scene


def _run_single(scene, *, time_steps=_TIME_STEPS):
    solver = FDTD(prepare_scene(scene), frequency=_FREQUENCY)
    solver.init_field()
    output = solver.solve(
        time_steps=time_steps,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=False,
    )
    return solver, output


def _run_distributed(scene, devices, *, time_steps=_TIME_STEPS):
    solver = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=FDTDParallelConfig(
            devices=devices,
            transport="cuda_p2p",
            overlap=True,
            gather_fields=True,
            result_device=devices[0],
        ),
    )
    solver.init_field()
    output = solver.solve(
        time_steps=time_steps,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=False,
    )
    return solver, output


def _combined_energy(fields, wire_data, dtype=torch.float64):
    total = torch.zeros((), dtype=dtype)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        total = total + fields[name].to(dtype).square().sum().cpu()
    if wire_data is not None:
        total = total + wire_data.current.abs().to(dtype).square().sum().cpu()
        total = total + wire_data.charge.abs().to(dtype).square().sum().cpu()
    return float(total)


def _gather_fields(distributed_solver, distributed_output):
    return {
        name: (
            distributed_output[name]
            if name in distributed_output
            else distributed_solver._gather_component(
                name,
                tuple(getattr(shard.solver, name) for shard in distributed_solver.shards),
            )
        )
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def test_two_gpu_straight_wire_crossing_split_is_bitwise_identical_to_single_gpu(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _straight_crossing_scene()
    single_solver, single_output = _run_single(scene)
    distributed_solver, distributed_output = _run_distributed(scene, cuda_p2p_devices)

    stats = distributed_solver.parallel_stats["thin_wire"]
    assert stats["enabled"] is True
    assert stats["cross_shard_segment_count"] >= 1
    assert stats["sample_shard_count"] == 2
    assert stats["deposit_shard_count"] == 2
    assert stats["reduction_order"] == "sorted_by_rank_deterministic"

    distributed_wire = distributed_output["observers"]["wire_state"]
    # The wire is genuinely active across the split: a trivial zero-current match
    # would not exercise the cross-shard EMF reduction / current broadcast.
    assert float(distributed_wire.current.abs().max()) > 1.0

    single_fields = {
        name: getattr(single_solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    distributed_fields = _gather_fields(distributed_solver, distributed_output)
    for name in single_fields:
        # Deterministic reductions -> exact bitwise parity at the short horizon.
        torch.testing.assert_close(
            distributed_fields[name],
            single_fields[name].to(cuda_p2p_devices[0]),
            rtol=0.0,
            atol=0.0,
        )

    single_wire = single_output["observers"]["wire_state"]
    torch.testing.assert_close(
        distributed_wire.current, single_wire.current.to(cuda_p2p_devices[0]),
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        distributed_wire.charge, single_wire.charge.to(cuda_p2p_devices[0]),
        rtol=0.0, atol=0.0,
    )
    assert distributed_wire.current.device == cuda_p2p_devices[0]
    assert distributed_wire.metadata["distributed_wire_owner_rank"] == stats["owner_rank"]


def test_two_gpu_closed_loop_crossing_split_conserves_energy_like_single_gpu(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    import math

    scene = _closed_loop_crossing_scene()
    single_solver, single_output = _run_single(scene, time_steps=_ENERGY_TIME_STEPS)
    distributed_solver, distributed_output = _run_distributed(
        scene, cuda_p2p_devices, time_steps=_ENERGY_TIME_STEPS
    )

    stats = distributed_solver.parallel_stats["thin_wire"]
    # The loop has two x-directed arms crossing the split -> two cross-shard segments.
    assert stats["cross_shard_segment_count"] == 2

    single_fields = {
        name: getattr(single_solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    distributed_fields = _gather_fields(distributed_solver, distributed_output)

    single_wire = single_output["observers"]["loop_state"]
    distributed_wire = distributed_output["observers"]["loop_state"]
    assert float(distributed_wire.current.abs().max()) > 1.0

    # Wire current/charge stay tight across the split (measured ~3e-8 relative).
    torch.testing.assert_close(
        distributed_wire.current, single_wire.current.to(cuda_p2p_devices[0]),
        rtol=2.0e-5, atol=2.0e-9,
    )
    torch.testing.assert_close(
        distributed_wire.charge, single_wire.charge.to(cuda_p2p_devices[0]),
        rtol=2.0e-5, atol=2.0e-20,
    )

    single_energy = _combined_energy(single_fields, single_wire)
    distributed_energy = _combined_energy(distributed_fields, distributed_wire)
    assert distributed_energy > 0.0 and math.isfinite(distributed_energy)
    # The discrete energy invariant of a lossless loop crossing the split is
    # preserved essentially bitwise (measured ~2e-9 relative) even where individual
    # field phases drift by chaotic float32 accumulation over the run: the single-GPU
    # per-step conservation gate lives in tests/fdtd/thin_wire/test_thin_wire_forward.
    assert abs(distributed_energy - single_energy) <= 1.0e-6 * single_energy
    # Pointwise fields remain close and bounded (documented chaotic FP phase drift).
    for name in single_fields:
        torch.testing.assert_close(
            distributed_fields[name],
            single_fields[name].to(cuda_p2p_devices[0]),
            rtol=2.0e-4,
            atol=1.0e-6,
        )
