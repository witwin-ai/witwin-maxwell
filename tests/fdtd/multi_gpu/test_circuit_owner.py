from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import CompiledPortGeometry
from witwin.maxwell.fdtd.distributed.circuits import (
    _OwnerCurrentReuseState,
    compile_distributed_circuit_plan,
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


def _circuit_with_bindings(names) -> mw.Circuit:
    circuit = mw.Circuit("distributed")
    for name in names:
        node = circuit.node(f"{name}_node")
        circuit.add(mw.Resistor(f"R_{name}", node, circuit.ground, 50.0))
        circuit.bind_port(name, positive=node, negative=circuit.ground)
    return circuit


def _structural_plan(circuit, geometries):
    partition = FDTDPartitionPlan(
        global_shape=(9, 5, 5),
        devices=("cuda:0", "cuda:1"),
    )
    scene = SimpleNamespace(circuits=(circuit,))
    return compile_distributed_circuit_plan(
        scene,
        partition,
        geometries=tuple(geometries),
    )


def test_circuit_owner_is_minimum_global_reference_edge_independent_of_binding_order():
    geometries = (
        _geometry("left", "Ez", ((1, 2, 1),)),
        _geometry("right", "Ez", ((6, 2, 1),)),
    )
    forward = _structural_plan(
        _circuit_with_bindings(("left", "right")),
        geometries,
    )
    reverse = _structural_plan(
        _circuit_with_bindings(("right", "left")),
        tuple(reversed(geometries)),
    )

    assert forward.owner_rank == reverse.owner_rank == 0
    assert forward.port_owners == reverse.port_owners == {"left": 0, "right": 1}


def test_component_order_breaks_equal_coordinate_reference_ties_deterministically():
    geometries = (
        _geometry("ey", "Ey", ((4, 1, 2),)),
        _geometry("ex", "Ex", ((4, 1, 2),)),
    )
    forward = _structural_plan(_circuit_with_bindings(("ey", "ex")), geometries)
    reverse = _structural_plan(
        _circuit_with_bindings(("ex", "ey")),
        tuple(reversed(geometries)),
    )

    assert forward.owner_rank == reverse.owner_rank
    assert forward.port_owners == reverse.port_owners
    assert forward.owner_reference_port == reverse.owner_reference_port == "ex"


def test_one_bound_port_cannot_span_multiple_shards():
    circuit = _circuit_with_bindings(("split",))
    geometry = _geometry("split", "Ez", ((3, 2, 1), (4, 2, 1)))

    with pytest.raises(ValueError, match="crosses the multi-GPU x partition"):
        _structural_plan(circuit, (geometry,))


def test_bound_ports_cannot_overlap_the_same_global_yee_edge():
    circuit = _circuit_with_bindings(("first", "second"))
    geometries = (
        _geometry("first", "Ex", ((2, 1, 1),)),
        _geometry("second", "Ex", ((2, 1, 1),)),
    )

    with pytest.raises(ValueError, match="overlap"):
        _structural_plan(circuit, geometries)


def test_remote_owner_current_reuse_requires_one_copy_acknowledgement():
    state = _OwnerCurrentReuseState()

    assert state.begin_owner_write() is False
    state.finish_remote_read_schedule()
    with pytest.raises(RuntimeError, match="remote copy acknowledgement"):
        state.finish_remote_read_schedule()
    assert state.begin_owner_write() is True
    assert state.begin_owner_write() is False


_FREQUENCY = 3.0e9
_TIME_STEPS = 16


def _port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.004),
        negative=(x, 0.0, -0.004),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, 0.0),
            size=(0.008, 0.008, 0.0),
        ),
        reference_impedance=50.0,
    )


def _two_shard_circuit_scene() -> mw.Scene:
    circuit = mw.Circuit("two_shard")
    left = circuit.node("left")
    right = circuit.node("right")
    circuit.add(
        mw.CurrentSource(
            "Iexcite",
            left,
            circuit.ground,
            0.0,
            waveform=mw.SineWaveform(0.0, 1.0e-3, _FREQUENCY),
        )
    )
    circuit.add(mw.Resistor("Rlink", left, right, 75.0))
    circuit.add(mw.Resistor("Rload", right, circuit.ground, 50.0))
    circuit.bind_port("left_port", positive=left, negative=circuit.ground)
    circuit.bind_port("right_port", positive=right, negative=circuit.ground)
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        ports=(_port("left_port", -0.008), _port("right_port", 0.008)),
        circuits=(circuit,),
        device="cuda:0",
    )


def test_public_simulation_validation_accepts_distributed_circuit_scene():
    simulation = mw.Simulation.fdtd(
        _two_shard_circuit_scene(),
        frequency=_FREQUENCY,
        parallel=mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1")),
    )

    simulation._validate_circuit_execution()


def _run_single(scene):
    solver = FDTD(prepare_scene(scene), frequency=_FREQUENCY)
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


def test_physical_two_gpu_circuit_matches_single_gpu_and_reports_scalar_contract(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _two_shard_circuit_scene()
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

    actual_circuit = distributed_output["circuits"]["two_shard"]
    expected_circuit = single_output["circuits"]["two_shard"]
    torch.testing.assert_close(
        actual_circuit.node_voltages,
        expected_circuit.node_voltages,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    torch.testing.assert_close(
        actual_circuit.branch_currents,
        expected_circuit.branch_currents,
        rtol=2.0e-5,
        atol=2.0e-10,
    )
    assert actual_circuit.node_voltages.device == cuda_p2p_devices[0]

    stats = distributed_solver.parallel_stats["circuit"]
    assert stats["circuit_owner_rank"] == 0
    assert stats["port_owner_ranks"] == {"left_port": 0, "right_port": 1}
    assert stats["remote_port_count"] == 1
    assert stats["same_shard_fast_path_count"] == 1
    assert stats["scalar_transfers_per_step"] == 2
    assert stats["owner_copy_acknowledgements_per_step"] == 1
    assert stats["communication_bytes_per_step"] == 2 * torch.tensor([], dtype=torch.float32).element_size()
    checkpoint = distributed_solver.circuit_checkpoint_tensors()
    assert checkpoint["step"].device == cuda_p2p_devices[0]
    assert int(checkpoint["step"].item()) == _TIME_STEPS
