from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import CompiledPortGeometry
from witwin.maxwell.fdtd.distributed.networks import (
    DistributedNetworkRuntime,
    compile_distributed_network_plan,
)
from witwin.maxwell.fdtd.distributed.solver import DistributedFDTD
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig, FDTDPartitionPlan
from witwin.maxwell.scene import prepare_scene


def _geometry(name: str, component: str, indices) -> CompiledPortGeometry:
    return CompiledPortGeometry(
        port_name=name,
        axis=component[-1].lower(),
        direction=1,
        voltage_component=component,
        voltage_indices=torch.as_tensor(indices, dtype=torch.int64),
        voltage_weights=torch.ones((len(indices),), dtype=torch.float64),
        current_components=(),
        current_indices=(),
        current_weights=(),
        reference_impedance=50.0,
    )


def _network_block(connection_names) -> SimpleNamespace:
    # compile_distributed_network_plan only reads ``name`` and
    # ``connected_port_names``, so a lightweight stand-in exercises the ownership
    # resolver without compiling a rational model.
    return SimpleNamespace(
        name="distributed",
        connected_port_names=tuple(connection_names),
    )


def _structural_plan(connection_names, geometries):
    partition = FDTDPartitionPlan(
        global_shape=(9, 5, 5),
        devices=("cuda:0", "cuda:1"),
    )
    scene = SimpleNamespace(networks=(_network_block(connection_names),))
    return compile_distributed_network_plan(
        scene,
        partition,
        geometries=tuple(geometries),
    )


def test_network_owner_is_minimum_global_reference_edge_independent_of_connection_order():
    geometries = (
        _geometry("left", "Ez", ((1, 2, 1),)),
        _geometry("right", "Ez", ((6, 2, 1),)),
    )
    forward = _structural_plan(("left", "right"), geometries)
    reverse = _structural_plan(("right", "left"), tuple(reversed(geometries)))

    assert forward.owner_rank == reverse.owner_rank == 0
    assert forward.port_owners == reverse.port_owners == {"left": 0, "right": 1}
    assert forward.owner_reference_port == reverse.owner_reference_port == "left"


def test_component_order_breaks_equal_coordinate_reference_ties_deterministically():
    geometries = (
        _geometry("ey", "Ey", ((4, 1, 2),)),
        _geometry("ex", "Ex", ((4, 1, 2),)),
    )
    forward = _structural_plan(("ey", "ex"), geometries)
    reverse = _structural_plan(("ex", "ey"), tuple(reversed(geometries)))

    assert forward.owner_rank == reverse.owner_rank
    assert forward.port_owners == reverse.port_owners
    assert forward.owner_reference_port == reverse.owner_reference_port == "ex"


def test_one_connected_port_cannot_span_multiple_shards():
    geometry = _geometry("split", "Ez", ((3, 2, 1), (4, 2, 1)))

    with pytest.raises(ValueError, match="crosses the multi-GPU x partition"):
        _structural_plan(("split",), (geometry,))


def test_connected_ports_cannot_overlap_the_same_global_yee_edge():
    geometries = (
        _geometry("first", "Ex", ((2, 1, 1),)),
        _geometry("second", "Ex", ((2, 1, 1),)),
    )

    with pytest.raises(ValueError, match="overlap"):
        _structural_plan(("first", "second"), geometries)


_FREQUENCY = 3.0e9
_TIME_STEPS = 200


def _port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.004),
        negative=(x, 0.0, -0.004),
        voltage_path=mw.AxisPath("z"),
        # See tests/fdtd/multi_gpu/test_circuit_owner.py for the identical
        # z-normal Ampere-loop placement rationale (z=-0.002 = -0.5*dz).
        current_surface=mw.Box(
            position=(x, 0.0, -0.002),
            size=(0.012, 0.012, 0.0),
        ),
        reference_impedance=50.0,
    )


def _two_port_network() -> mw.NetworkBlock:
    resistance = 50.0
    capacitance = 5.0e-12
    shunt = 2.0e-3
    incidence = torch.tensor((1.0, -1.0), dtype=torch.float64)
    direct = torch.outer(incidence, incidence) / resistance + shunt * torch.eye(
        2, dtype=torch.float64
    )
    model = mw.StateSpaceNetwork(
        A=torch.tensor(((-1.0 / (resistance * capacitance),),), dtype=torch.float64),
        B=(incidence / (resistance * capacitance)).reshape(1, 2),
        C=(-incidence / resistance).reshape(2, 1),
        D=direct,
        representation="Y",
        port_order=("net_0", "net_1"),
    )
    frequencies = torch.linspace(0.2e9, 6.0e9, 30, dtype=torch.float64)
    data = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies),
        z0=50.0,
        port_names=model.port_order,
    )
    return mw.NetworkBlock(
        name="split_net",
        network=data,
        connections={"net_0": "left_port", "net_1": "right_port"},
        fit=False,
        model=model,
    )


def _two_shard_network_scene(*, device: str = "cuda:0") -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.8e9),
            ),
        ),
        ports=(_port("left_port", -0.008), _port("right_port", 0.008)),
        networks=(_two_port_network(),),
        device=device,
    )


def _trainable_network_scene() -> tuple[mw.Scene, torch.Tensor]:
    # A pre-fitted RationalModel with a trainable direct term is accepted by the
    # single-GPU network adjoint, so it exercises the distributed-adjoint boundary
    # (rejected only because ``parallel`` is set) rather than an unrelated guard.
    poles = torch.tensor((-8.0e9 + 0.0j,), dtype=torch.complex128)
    residues = torch.tensor((((1.0e8 + 0.0j,),),), dtype=torch.complex128)
    direct = torch.tensor(((0.02 + 0.0j,),), dtype=torch.complex128, requires_grad=True)
    model = mw.RationalModel(
        poles=poles, residues=residues, direct=direct, representation="Y"
    )
    frequencies = torch.linspace(0.25e9, 6.0e9, 33, dtype=torch.float64)
    data = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies).detach(),
        z0=50.0,
        port_names=("load",),
    )
    block = mw.NetworkBlock(
        name="trainable_load",
        network=data,
        connections={"load": "feed"},
        fit=False,
        model=model,
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.8e9),
            ),
        ),
        ports=(_port("feed", 0.008),),
        networks=(block,),
        device="cuda:0",
    )
    return scene, direct


def test_public_simulation_validation_accepts_distributed_network_scene():
    simulation = mw.Simulation.fdtd(
        _two_shard_network_scene(),
        frequency=_FREQUENCY,
        parallel=mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1")),
    )

    # The FDTD-only network guard must accept the multi-GPU network scene.
    simulation._validate_network_solver()


def test_trainable_network_plus_parallel_is_rejected_before_solver_build():
    scene, direct = _trainable_network_scene()
    assert direct.requires_grad
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        parallel=mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1")),
    )

    assert simulation.has_trainable_parameters
    with pytest.raises(ValueError, match="does not support trainable scene parameters"):
        simulation.prepare()


def _run_single(scene):
    solver = FDTD(prepare_scene(scene), frequency=_FREQUENCY)
    solver._requested_port_frequencies = (_FREQUENCY,)
    solver.init_field()
    output = solver.solve(
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=False,
    )
    return solver, output


def _run_distributed(scene, devices):
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
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=False,
    )
    return solver, output


def test_network_plus_circuit_scene_is_rejected(cuda_p2p_devices, cuda_memory_cleanup):
    circuit = mw.Circuit("stray")
    node = circuit.node("n")
    circuit.add(mw.Resistor("R", node, circuit.ground, 50.0))
    scene = _two_shard_network_scene(device=str(cuda_p2p_devices[0]))
    scene = scene.clone(circuits=(circuit,))

    with pytest.raises(ValueError, match="both embedded networks"):
        DistributedFDTD(
            scene,
            frequency=_FREQUENCY,
            parallel=FDTDParallelConfig(devices=cuda_p2p_devices),
        )


def test_physical_two_gpu_network_matches_single_gpu_and_reports_scalar_contract(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _two_shard_network_scene(device=str(cuda_p2p_devices[0]))
    single_solver, single_output = _run_single(scene)
    distributed_solver, distributed_output = _run_distributed(
        scene,
        cuda_p2p_devices,
    )

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        distributed_field = (
            distributed_output[name]
            if name in distributed_output
            else distributed_solver._gather_component(
                name,
                tuple(
                    getattr(shard.solver, name)
                    for shard in distributed_solver.shards
                ),
            )
        )
        torch.testing.assert_close(
            distributed_field,
            getattr(single_solver, name).to(cuda_p2p_devices[0]),
            rtol=2.0e-5,
            atol=2.0e-7,
        )

    for name in ("left_port", "right_port"):
        actual = distributed_output["ports"][name]
        expected = single_output["ports"][name]
        torch.testing.assert_close(actual.voltage, expected.voltage, rtol=2.0e-5, atol=2.0e-8)
        torch.testing.assert_close(actual.current, expected.current, rtol=2.0e-5, atol=2.0e-10)
        assert actual.voltage.device == cuda_p2p_devices[0]

    actual_network = distributed_output["embedded_networks"]["split_net"]
    expected_network = single_output["embedded_networks"]["split_net"]
    torch.testing.assert_close(
        actual_network.voltage,
        expected_network.voltage,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    torch.testing.assert_close(
        actual_network.current,
        expected_network.current,
        rtol=2.0e-5,
        atol=2.0e-10,
    )
    assert actual_network.voltage.device == cuda_p2p_devices[0]

    # Network state parity: the owner shard advances the sole authoritative state
    # recurrence, and it must track the single-GPU run.
    distributed_state = distributed_solver._distributed_network.network_runtime.state
    single_state = single_solver._network_runtimes[0].state
    torch.testing.assert_close(
        distributed_state.to(cuda_p2p_devices[0]),
        single_state.to(cuda_p2p_devices[0]),
        rtol=2.0e-5,
        atol=1.0e-9,
    )

    stats = distributed_solver.parallel_stats["network"]
    assert stats["network_owner_rank"] == 0
    assert stats["network_owner_reference_port"] == "left_port"
    assert stats["port_owner_ranks"] == {"left_port": 0, "right_port": 1}
    assert stats["connected_port_count"] == 2
    assert stats["remote_port_count"] == 1
    assert stats["same_shard_fast_path_count"] == 1
    assert stats["scalar_transfers_per_step"] == 2
    assert stats["owner_copy_acknowledgements_per_step"] == 1
    assert stats["communication_order"] == "O(connected_ports)"
    assert stats["communication_bytes_per_step"] == 2 * torch.tensor(
        [], dtype=torch.float32
    ).element_size()

    # A single-shard network must never allocate cross-device events.
    for port_runtime in distributed_solver._distributed_network.port_runtimes:
        remote = port_runtime.plan.owner_rank != 0
        assert (port_runtime.voltage_ready is not None) == remote

    checkpoint = distributed_solver.network_checkpoint_tensors()
    assert checkpoint["network_state_split_net"].device == cuda_p2p_devices[0]
