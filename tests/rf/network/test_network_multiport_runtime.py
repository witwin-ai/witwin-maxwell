from types import SimpleNamespace

import numpy as np
import pytest
import torch

from witwin.maxwell.compiler.networks import CompiledNetworkBlock
from witwin.maxwell.compiler.ports import CompiledPortGeometry
from witwin.maxwell.fdtd.lumped import prepare_lumped_runtime
from witwin.maxwell.fdtd.networks import (
    apply_network_runtimes,
    finalize_embedded_networks,
    make_network_runner,
    prepare_network_runtimes,
)
from witwin.maxwell.rational import (
    DiscreteStateSpaceNetwork,
    StateSpaceNetwork,
)


def _matrices(port_count: int, *, device: torch.device):
    dtype = torch.float64
    state_count = 3
    A = torch.diag(
        torch.tensor((0.61, 0.72, 0.83), device=device, dtype=dtype)
    )
    B = 0.03 * torch.arange(
        1,
        state_count * port_count + 1,
        device=device,
        dtype=dtype,
    ).reshape(state_count, port_count)
    C = 0.02 * torch.arange(
        1,
        port_count * state_count + 1,
        device=device,
        dtype=dtype,
    ).reshape(port_count, state_count)
    indices = torch.arange(port_count, device=device, dtype=dtype)
    separation = torch.abs(indices[:, None] - indices[None, :])
    D = 0.04 / (1.0 + separation)
    D.diagonal().add_(0.18)
    return A, B, C, D


def _geometry(
    physical_index: int,
    *,
    name: str,
    device: torch.device,
) -> CompiledPortGeometry:
    return CompiledPortGeometry(
        port_name=name,
        axis="x",
        direction=1,
        voltage_component="Ex",
        voltage_indices=torch.tensor(
            ((physical_index, 0, 0),), device=device, dtype=torch.int64
        ),
        voltage_weights=torch.tensor((1.0,), device=device, dtype=torch.float64),
        current_components=(),
        current_indices=(),
        current_weights=(),
        reference_impedance=50.0,
    )


def _solver(
    port_count: int,
    device: torch.device,
    *,
    permutation: tuple[int, ...] | None = None,
    direct_matrix: torch.Tensor | None = None,
):
    dtype = torch.float64
    permutation = permutation or tuple(range(port_count))
    base_names = tuple(f"p{index}" for index in range(port_count))
    base_network_names = tuple(f"n{index}" for index in range(port_count))
    geometries = tuple(
        _geometry(index, name=name, device=device)
        for index, name in enumerate(base_names)
    )
    field = torch.linspace(
        0.7,
        1.9,
        port_count,
        device=device,
        dtype=dtype,
    ).reshape(port_count, 1, 1)
    eps = torch.linspace(
        0.8,
        1.4,
        port_count,
        device=device,
        dtype=dtype,
    ).reshape(port_count, 1, 1)
    port_runtimes = []
    for geometry in geometries:
        lumped = prepare_lumped_runtime(
            geometry,
            dt=torch.tensor(0.2, device=device, dtype=dtype),
            eps_edge=eps,
            yee_control_volume=torch.ones_like(eps),
            resistance=0.0,
        )
        port_runtimes.append(
            SimpleNamespace(
                port=SimpleNamespace(name=geometry.port_name),
                geometry=geometry,
                field_name="Ex",
                lumped=lumped,
                embedded_network_name="multi",
            )
        )

    A, B, C, D = _matrices(port_count, device=device)
    if direct_matrix is not None:
        D = direct_matrix.to(device=device, dtype=dtype)
    selection = torch.tensor(permutation, device=device, dtype=torch.int64)
    B = torch.index_select(B, 1, selection)
    C = torch.index_select(C, 0, selection)
    D = torch.index_select(torch.index_select(D, 0, selection), 1, selection)
    ordered_connections = tuple(base_names[index] for index in permutation)
    ordered_network_names = tuple(base_network_names[index] for index in permutation)
    continuous = StateSpaceNetwork(
        A=-2.0e9 * torch.eye(3, device=device, dtype=dtype),
        B=torch.zeros((3, port_count), device=device, dtype=dtype),
        C=torch.zeros((port_count, 3), device=device, dtype=dtype),
        D=D,
        representation="Y",
        port_order=ordered_network_names,
    )
    discrete = DiscreteStateSpaceNetwork(
        A=A,
        B=B,
        C=C,
        D=D,
        dt=0.2,
        representation="Y",
        port_order=ordered_network_names,
        pole_radius=0.83,
    )
    compiled = CompiledNetworkBlock(
        name="multi",
        port_order=ordered_network_names,
        connection_names=ordered_connections,
        ports=tuple(geometries[index] for index in permutation),
        continuous=continuous,
        discrete=discrete,
        fit_report=None,
        model_id="multi-oracle",
        frequency_band=(0.0, 10.0),
    )
    solver = SimpleNamespace(
        scene=SimpleNamespace(
            compile_networks=lambda *, dt, device: (compiled,)
        ),
        device=device,
        dt=0.2,
        Ex=field,
        _port_runtimes=tuple(port_runtimes),
    )
    prepare_network_runtimes(solver)
    solver._network_runtimes[0].state.copy_(
        torch.tensor((0.13, -0.27, 0.41), device=device, dtype=dtype)
    )
    return solver


@pytest.mark.parametrize("port_count", (2, 4))
def test_multiport_direct_loop_matches_independent_matrix_oracle(port_count: int) -> None:
    solver = _solver(port_count, torch.device("cpu"))
    runtime = solver._network_runtimes[0]
    before = solver.Ex.detach().numpy().copy()
    state = runtime.state.detach().numpy().copy()
    free_voltage = np.array(
        [
            before[int(port_runtime.port.name[1:]), 0, 0]
            for port_runtime in runtime.port_runtimes
        ]
    )
    A = runtime.A.detach().numpy()
    B = runtime.B.detach().numpy()
    C = runtime.C.detach().numpy()
    D = runtime.D.detach().numpy()
    feedback = runtime.feedback_impedance.detach().numpy()
    # Slice U2 (coordinator unification ruling): the network solve consumes the
    # trapezoidal half-step voltage 0.5*(V_after_prev + V_free); at this cold-start
    # apply V_after_prev = 0, so the direct loop and midpoint observable use
    # 0.5 * free_voltage. The loop denominator/feedback is unchanged.
    coupling_voltage = 0.5 * free_voltage
    expected_current = np.linalg.solve(
        np.eye(port_count) + D @ np.diag(feedback),
        C @ state + D @ coupling_voltage,
    )
    expected_voltage = coupling_voltage - feedback * expected_current
    expected_state = A @ state + B @ expected_voltage
    expected_field = before.copy()
    for index, port_runtime in enumerate(runtime.port_runtimes):
        physical = int(port_runtime.port.name[1:])
        injection = float(port_runtime.lumped.injection[0])
        expected_field[physical, 0, 0] -= injection * expected_current[index]

    apply_network_runtimes(solver)

    np.testing.assert_allclose(runtime.branch_current.numpy(), expected_current)
    np.testing.assert_allclose(runtime.network_voltage.numpy(), expected_voltage)
    np.testing.assert_allclose(runtime.state.numpy(), expected_state)
    np.testing.assert_allclose(solver.Ex.numpy(), expected_field)
    assert abs(D[0, 1]) > 0.0


def test_port_permutation_preserves_physical_fields_currents_and_state() -> None:
    original = _solver(4, torch.device("cpu"))
    permuted = _solver(4, torch.device("cpu"), permutation=(2, 0, 3, 1))

    for _ in range(7):
        apply_network_runtimes(original)
        apply_network_runtimes(permuted)

    torch.testing.assert_close(permuted.Ex, original.Ex)
    torch.testing.assert_close(
        permuted._network_runtimes[0].state,
        original._network_runtimes[0].state,
    )
    original_currents = {
        runtime.port.name: runtime.lumped.last_branch_current
        for runtime in original._port_runtimes
    }
    permuted_currents = {
        runtime.port.name: runtime.lumped.last_branch_current
        for runtime in permuted._port_runtimes
    }
    for name in original_currents:
        torch.testing.assert_close(permuted_currents[name], original_currents[name])


def test_finalize_preserves_network_port_order_and_current_sign() -> None:
    solver = _solver(4, torch.device("cpu"), permutation=(2, 0, 3, 1))
    apply_network_runtimes(solver)
    frequencies = torch.tensor((1.0, 2.0), dtype=torch.float64)
    ports = {
        name: SimpleNamespace(
            frequencies=frequencies,
            voltage=torch.tensor(
                (physical + 1.0j, physical + 2.0j), dtype=torch.complex128
            ),
            current=torch.tensor(
                (-physical - 0.1j, -physical - 0.2j), dtype=torch.complex128
            ),
        )
        for physical, name in enumerate(f"p{index}" for index in range(4))
    }

    diagnostics = finalize_embedded_networks(solver, ports)["multi"]

    assert diagnostics.port_names == ("n2", "n0", "n3", "n1")
    torch.testing.assert_close(
        diagnostics.voltage,
        torch.stack(tuple(ports[name].voltage for name in ("p2", "p0", "p3", "p1"))),
    )
    torch.testing.assert_close(
        diagnostics.current,
        torch.stack(tuple(-ports[name].current for name in ("p2", "p0", "p3", "p1"))),
    )
    assert len(diagnostics.metadata["port_energy"]) == 4
    assert isinstance(diagnostics.metadata["absorbed_energy"], float)
    assert isinstance(diagnostics.metadata["generated_energy"], float)
    assert diagnostics.metadata["direct_loop_condition"] >= 1.0


def test_prepared_lu_matches_oracle_for_ill_conditioned_direct_loop() -> None:
    direct = torch.zeros((4, 4), dtype=torch.float64)
    direct[0, 0] = (1.0e-6 - 1.0) / 0.125
    solver = _solver(4, torch.device("cpu"), direct_matrix=direct)
    runtime = solver._network_runtimes[0]
    state = runtime.state.detach().numpy().copy()
    free_voltage = solver.Ex[:, 0, 0].detach().numpy().copy()
    # Slice U2: cold-start trapezoidal coupling voltage is 0.5 * free_voltage.
    coupling_voltage = 0.5 * free_voltage
    expected = np.linalg.solve(
        runtime.loop_denominator.detach().numpy(),
        runtime.C.detach().numpy() @ state
        + runtime.D.detach().numpy() @ coupling_voltage,
    )

    apply_network_runtimes(solver)

    np.testing.assert_allclose(runtime.branch_current.numpy(), expected, rtol=1.0e-9)
    assert runtime.loop_condition > 1.0e5


def test_passive_through_network_classifies_power_after_port_sum() -> None:
    solver = _solver(2, torch.device("cpu"))
    frequencies = torch.tensor((1.0, 2.0), dtype=torch.float64)
    ports = {
        "p0": SimpleNamespace(
            frequencies=frequencies,
            voltage=torch.ones(2, dtype=torch.complex128),
            current=-2.0 * torch.ones(2, dtype=torch.complex128),
        ),
        "p1": SimpleNamespace(
            frequencies=frequencies,
            voltage=torch.ones(2, dtype=torch.complex128),
            current=2.0 * torch.ones(2, dtype=torch.complex128),
        ),
    }

    diagnostics = finalize_embedded_networks(solver, ports)["multi"]

    torch.testing.assert_close(
        diagnostics.port_power,
        torch.tensor(((1.0, 1.0), (-1.0, -1.0)), dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.net_power,
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.absorbed_power,
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.generated_power,
        torch.zeros(2, dtype=torch.float64),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multiport_cuda_hot_path_has_no_alloc_scalar_sync_or_copy() -> None:
    solver = _solver(4, torch.device("cuda"))
    for _ in range(4):
        apply_network_runtimes(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        acc_events=True,
        profile_memory=True,
    ) as profile:
        for _ in range(16):
            apply_network_runtimes(solver)
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    assert "aten::item" not in event_names
    assert "aten::_local_scalar_dense" not in event_names
    assert all(
        event.self_cpu_memory_usage == 0 and event.self_device_memory_usage == 0
        for event in profile.key_averages()
    )
    assert not any(
        "Memcpy HtoD" in name or "Memcpy DtoH" in name for name in event_names
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multiport_cuda_graph_matches_eager_updates() -> None:
    eager = _solver(4, torch.device("cuda"))
    captured = _solver(4, torch.device("cuda"))
    runner = make_network_runner(captured, use_cuda_graph=True)
    assert captured._network_cuda_graph_active

    for _ in range(16):
        apply_network_runtimes(eager)
        runner()

    torch.testing.assert_close(captured.Ex, eager.Ex)
    torch.testing.assert_close(
        captured._network_runtimes[0].state,
        eager._network_runtimes[0].state,
    )
    torch.testing.assert_close(
        captured._network_runtimes[0].branch_current,
        eager._network_runtimes[0].branch_current,
    )
