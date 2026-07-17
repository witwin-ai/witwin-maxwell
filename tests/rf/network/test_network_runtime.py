from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.networks import CompiledNetworkBlock
from witwin.maxwell.compiler.ports import CompiledPortGeometry
from witwin.maxwell.fdtd.lumped import prepare_lumped_runtime
from witwin.maxwell.fdtd.networks import (
    apply_network_runtimes,
    make_network_runner,
    prepare_network_runtimes,
)
from witwin.maxwell.rational import (
    DiscreteStateSpaceNetwork,
    StateSpaceNetwork,
)


def _geometry(device: torch.device, dtype: torch.dtype) -> CompiledPortGeometry:
    return CompiledPortGeometry(
        port_name="feed",
        axis="x",
        direction=1,
        voltage_component="Ex",
        voltage_indices=torch.tensor(
            ((0, 0, 0), (1, 0, 0)), device=device, dtype=torch.int64
        ),
        voltage_weights=torch.tensor((0.4, 0.6), device=device, dtype=dtype),
        current_components=(),
        current_indices=(),
        current_weights=(),
        reference_impedance=50.0,
    )


def _solver(device: torch.device):
    dtype = torch.float64
    geometry = _geometry(device, dtype)
    field = torch.tensor([1.0, 2.0], device=device, dtype=dtype).reshape(2, 1, 1)
    lumped = prepare_lumped_runtime(
        geometry,
        dt=torch.tensor(0.2, device=device, dtype=dtype),
        eps_edge=torch.ones_like(field),
        yee_control_volume=torch.ones_like(field),
        resistance=0.0,
    )
    port_runtime = SimpleNamespace(
        port=SimpleNamespace(name="feed"),
        geometry=geometry,
        field_name="Ex",
        lumped=lumped,
        embedded_network_name="load",
    )
    continuous = StateSpaceNetwork(
        A=torch.tensor([[-1.0]], device=device, dtype=dtype),
        B=torch.tensor([[1.0]], device=device, dtype=dtype),
        C=torch.tensor([[1.0]], device=device, dtype=dtype),
        D=torch.tensor([[0.4]], device=device, dtype=dtype),
        representation="Y",
        port_order=("terminal",),
    )
    discrete = DiscreteStateSpaceNetwork(
        A=torch.tensor([[0.8]], device=device, dtype=dtype),
        B=torch.tensor([[0.1]], device=device, dtype=dtype),
        C=torch.tensor([[0.3]], device=device, dtype=dtype),
        D=torch.tensor([[0.4]], device=device, dtype=dtype),
        dt=0.2,
        representation="Y",
        port_order=("terminal",),
        pole_radius=0.8,
    )
    compiled = CompiledNetworkBlock(
        name="load",
        port_order=("terminal",),
        connection_names=("feed",),
        ports=(geometry,),
        continuous=continuous,
        discrete=discrete,
        fit_report=None,
        model_id="unit-load",
        frequency_band=(0.0, 1.0),
    )
    scene = SimpleNamespace(
        compile_networks=lambda *, dt, device: (compiled,),
    )
    solver = SimpleNamespace(
        scene=scene,
        device=device,
        dt=0.2,
        Ex=field,
        _port_runtimes=(port_runtime,),
    )
    prepare_network_runtimes(solver)
    return solver


def test_single_port_runtime_solves_direct_loop_without_one_step_delay() -> None:
    solver = _solver(torch.device("cpu"))
    runtime = solver._network_runtimes[0]
    runtime.state.fill_(0.5)
    before = solver.Ex.clone()
    free_voltage = torch.tensor(1.6, dtype=torch.float64)
    feedback_impedance = runtime.port_runtime.lumped.discrete_port_impedance
    expected_current = (0.3 * 0.5 + 0.4 * free_voltage) / (
        1.0 + 0.4 * feedback_impedance
    )
    expected_voltage = free_voltage - feedback_impedance * expected_current
    expected_state = 0.8 * 0.5 + 0.1 * expected_voltage

    apply_network_runtimes(solver)

    lumped = runtime.port_runtime.lumped
    torch.testing.assert_close(lumped.last_branch_current, expected_current)
    torch.testing.assert_close(lumped.last_voltage_midpoint, expected_voltage)
    torch.testing.assert_close(runtime.state, expected_state.reshape(1))
    expected_field = before.clone().reshape(-1)
    expected_field.index_add_(
        0,
        lumped.linear_indices,
        -lumped.injection * expected_current,
    )
    torch.testing.assert_close(solver.Ex.reshape(-1), expected_field)


def test_runtime_reuses_all_hot_path_buffers() -> None:
    solver = _solver(torch.device("cpu"))
    runtime = solver._network_runtimes[0]
    pointers = (
        runtime.state.data_ptr(),
        runtime.next_state.data_ptr(),
        runtime.state_drive.data_ptr(),
        runtime.output_buffer.data_ptr(),
        runtime.port_runtime.lumped.edge_buffer.data_ptr(),
        runtime.port_runtime.lumped.correction_buffer.data_ptr(),
    )

    for _ in range(8):
        apply_network_runtimes(solver)

    assert pointers == (
        runtime.state.data_ptr(),
        runtime.next_state.data_ptr(),
        runtime.state_drive.data_ptr(),
        runtime.output_buffer.data_ptr(),
        runtime.port_runtime.lumped.edge_buffer.data_ptr(),
        runtime.port_runtime.lumped.correction_buffer.data_ptr(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_runtime_has_no_scalar_sync_or_host_device_copy() -> None:
    solver = _solver(torch.device("cuda"))
    for _ in range(4):
        apply_network_runtimes(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        acc_events=True,
    ) as profile:
        for _ in range(16):
            apply_network_runtimes(solver)
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    assert "aten::item" not in event_names
    assert "aten::_local_scalar_dense" not in event_names
    assert not any("Memcpy HtoD" in name or "Memcpy DtoH" in name for name in event_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_network_cuda_graph_matches_eager_updates() -> None:
    eager = _solver(torch.device("cuda"))
    captured = _solver(torch.device("cuda"))
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bilinear_stability_check_preserves_transition_matrix() -> None:
    model = _series_rlc_state_space(
        resistance=25.0,
        inductance=0.5e-9,
        capacitance=1.0e-12,
    )
    model = mw.StateSpaceNetwork(
        A=model.A.cuda(),
        B=model.B.cuda(),
        C=model.C.cuda(),
        D=model.D.cuda(),
        representation=model.representation,
        port_order=model.port_order,
    )
    dt = 5.797101449275362e-12
    identity = torch.eye(model.state_count, device="cuda", dtype=model.A.dtype)
    expected = torch.linalg.solve(
        identity - 0.5 * dt * model.A,
        identity + 0.5 * dt * model.A,
    )

    discrete = model.discretize(dt)

    torch.testing.assert_close(discrete.A, expected)


def _series_rlc_state_space(
    *,
    resistance: float,
    inductance: float,
    capacitance: float,
) -> mw.StateSpaceNetwork:
    dtype = torch.float64
    return mw.StateSpaceNetwork(
        A=torch.tensor(
            (
                (-resistance / inductance, -1.0 / inductance),
                (1.0 / capacitance, 0.0),
            ),
            dtype=dtype,
        ),
        B=torch.tensor(((1.0 / inductance,), (0.0,)), dtype=dtype),
        C=torch.tensor(((1.0, 0.0),), dtype=dtype),
        D=torch.zeros((1, 1), dtype=dtype),
        representation="Y",
        port_order=("load",),
        passivity_margin=0.0,
    )


def _fdtd_scene(*, termination=None, network=None) -> mw.Scene:
    port = mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=50.0,
        termination=termination,
    )
    return mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.5e9),
            ),
        ),
        ports=(port,),
        networks=() if network is None else (network,),
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_touchstone_fitted_series_rlc_matches_native_fdtd_termination(tmp_path) -> None:
    resistance = 25.0
    inductance = 0.5e-9
    capacitance = 1.0e-12
    frequencies = torch.tensor((2.5e9, 3.0e9), dtype=torch.float64)
    file_frequencies = torch.linspace(1.0e9, 5.0e9, 41, dtype=torch.float64)
    model = _series_rlc_state_space(
        resistance=resistance,
        inductance=inductance,
        capacitance=capacitance,
    )
    source_data = mw.NetworkData.from_y(
        frequencies=file_frequencies,
        y=model.evaluate(file_frequencies),
        z0=50.0,
        port_names=("load",),
    )
    path = tmp_path / "series_rlc.s1p"
    source_data.to_touchstone(path, format="ri")
    block = mw.TouchstoneNetwork(
        name="load_network",
        path=path,
        connections={"load": "feed"},
        fit=mw.RationalFitConfig(
            order=2,
            iterations=2,
            relative_tolerance=1.0e-3,
        ),
        device="cuda",
    )
    torch.testing.assert_close(
        block.network.to_y().cpu(),
        model.evaluate(file_frequencies),
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    run_args = dict(
        frequencies=tuple(frequencies.tolist()),
        run_time=mw.TimeConfig(time_steps=96),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=True,
    )

    native = mw.Simulation.fdtd(
        _fdtd_scene(
            termination=mw.SeriesRLC(
                r=resistance,
                l=inductance,
                c=capacitance,
            )
        ),
        **run_args,
    ).run()
    embedded = mw.Simulation.fdtd(_fdtd_scene(network=block), **run_args).run()

    native_port = native.port("feed")
    embedded_port = embedded.port("feed")
    torch.testing.assert_close(
        embedded_port.voltage,
        native_port.voltage,
        rtol=2.0e-4,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        embedded_port.current,
        native_port.current,
        rtol=2.0e-4,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        embedded_port.reflection_coefficient,
        native_port.reflection_coefficient,
        rtol=2.0e-4,
        atol=1.0e-9,
    )
    for embedded_value, native_value in (
        (embedded_port.voltage, native_port.voltage),
        (embedded_port.current, native_port.current),
        (
            embedded_port.reflection_coefficient,
            native_port.reflection_coefficient,
        ),
    ):
        magnitude_error = torch.abs(
            torch.abs(embedded_value) - torch.abs(native_value)
        ) / torch.abs(native_value)
        phase_error_degrees = torch.abs(
            torch.angle(embedded_value / native_value)
        ) * (180.0 / torch.pi)
        assert torch.max(magnitude_error) < 0.01
        assert torch.max(phase_error_degrees) < 2.0
    diagnostics = embedded.embedded_network("load_network")
    torch.testing.assert_close(diagnostics.voltage[0], embedded_port.voltage)
    torch.testing.assert_close(diagnostics.current[0], -embedded_port.current)
    signed_power = 0.5 * torch.real(
        diagnostics.voltage * torch.conj(diagnostics.current)
    )
    incident_power = embedded_port.incident_power.unsqueeze(0)
    assert torch.all(signed_power >= -1.0e-5 * incident_power)
    assert torch.all(diagnostics.absorbed_power >= 0.0)
    assert torch.all(diagnostics.generated_power >= 0.0)
    assert diagnostics.model_id == "load_network:Y:2"
    assert diagnostics.fit_report is not None
    assert diagnostics.fit_report.relative_max_error < 1.0e-3
    assert diagnostics.metadata["current_convention"] == "entering_embedded_network"
    assert embedded.stats()["num_embedded_networks"] == 1
    assert embedded.stats()["network_cuda_graph_active"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_embedding_rejects_excitation_spectrum_outside_model_band() -> None:
    frequencies = torch.tensor((2.5e9, 3.0e9), dtype=torch.float64)
    model = _series_rlc_state_space(
        resistance=25.0,
        inductance=0.5e-9,
        capacitance=1.0e-12,
    )
    data = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies),
        z0=50.0,
        port_names=("load",),
    )
    block = mw.NetworkBlock(
        name="band_limited_load",
        network=data,
        connections={"load": "feed"},
        fit=False,
        model=model,
    )

    with pytest.raises(ValueError, match="excitation effective bands"):
        mw.Simulation.fdtd(
            _fdtd_scene(network=block),
            frequency=2.75e9,
            run_time=mw.TimeConfig(time_steps=8),
        ).prepare()
