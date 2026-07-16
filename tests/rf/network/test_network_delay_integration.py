import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.delay import delay_phase_matrix
from witwin.maxwell.fdtd.delay import (
    advance_bidirectional_delay,
    prepare_bidirectional_delay,
)
from witwin.maxwell.fdtd.networks import apply_network_runtimes
from witwin.maxwell.network import voltage_current_to_power_waves


def _delayed_two_port(
    frequencies: torch.Tensor,
    delays: tuple[float, float],
) -> tuple[mw.NetworkData, mw.StateSpaceNetwork]:
    core = torch.tensor(
        ((0.1, 0.25), (0.25, 0.1)),
        dtype=torch.float64,
        device=frequencies.device,
    )
    scattering = core.to(torch.complex128)[None, ...] * delay_phase_matrix(
        frequencies, delays
    )
    data = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("left", "right"),
    )
    model = mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64),
        B=torch.zeros((0, 2), dtype=torch.float64),
        C=torch.zeros((2, 0), dtype=torch.float64),
        D=core,
        representation="S",
        port_order=data.port_names,
    )
    return data, model


def _port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, -0.0025),
            size=(0.005, 0.005, 0.0),
        ),
        reference_impedance=50.0,
    )


def _scene(
    block: mw.NetworkBlock,
    *,
    device: str,
    sources: tuple[object, ...] | None = None,
) -> mw.Scene:
    if sources is None:
        sources = (
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.CW(frequency=2.0e9),
            ),
        )
    return mw.Scene(
        domain=mw.Domain(
            bounds=((-0.025, 0.025), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        sources=sources,
        ports=(_port("p_left", -0.01), _port("p_right", 0.01)),
        networks=(block,),
        device=device,
    )


def test_compiler_builds_passive_scattering_core_and_delay_descriptor() -> None:
    frequencies = torch.linspace(0.0, 4.0e9, 81, dtype=torch.float64)
    delays = (0.2e-9, 0.3e-9)
    data, model = _delayed_two_port(frequencies, delays)
    block = mw.NetworkBlock(
        name="delayed_link",
        network=data,
        connections={"left": "p_left", "right": "p_right"},
        fit=False,
        model=model,
        delay_seconds=delays,
        max_delay_steps=128,
    )

    (compiled,) = _scene(block, device="cpu").compile_networks(
        dt=0.01e-9,
        device="cpu",
    )

    assert compiled.continuous.representation == "S"
    assert compiled.delay is not None
    assert compiled.delay.delay_seconds == pytest.approx(delays)
    assert compiled.delay.phase_error_degrees < 3.0
    assert compiled.delay.reembedding_max_error < 1.0e-12
    torch.testing.assert_close(
        compiled.reference_impedance,
        torch.full((2,), 50.0, dtype=torch.float64),
    )

    automatic = mw.NetworkBlock(
        name="automatic_link",
        network=data,
        connections={"left": "p_left", "right": "p_right"},
        fit=mw.RationalFitConfig(
            order=1,
            iterations=2,
            relative_tolerance=1.0e-3,
        ),
        delay_seconds="auto",
        max_delay_steps=128,
    )
    (automatic_compiled,) = _scene(automatic, device="cpu").compile_networks(
        dt=0.01e-9,
        device="cpu",
    )
    assert automatic_compiled.delay is not None
    assert automatic_compiled.delay.delay_seconds == pytest.approx(delays, abs=1.0e-15)
    assert automatic_compiled.fit_report is not None
    assert automatic_compiled.fit_report.delay_estimation_rank == 2
    assert automatic_compiled.fit_report.delay_reembedding_max_error < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_delayed_two_port_fdtd_runs_bounded_graph_path() -> None:
    frequencies = torch.linspace(0.0, 4.0e9, 81, dtype=torch.float64)
    delays = (0.0, 0.3e-9)
    data, model = _delayed_two_port(frequencies, delays)
    block = mw.NetworkBlock(
        name="delayed_link",
        network=data,
        connections={"left": "p_left", "right": "p_right"},
        fit=False,
        model=model,
        delay_seconds=delays,
        max_delay_steps=128,
    )
    requested = (2.0e9,)

    result = mw.Simulation.fdtd(
        _scene(block, device="cuda"),
        frequencies=requested,
        run_time=mw.TimeConfig(time_steps=64),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=True,
    ).run()

    diagnostics = result.embedded_network("delayed_link")
    assert bool(torch.all(torch.isfinite(diagnostics.voltage)))
    assert bool(torch.all(torch.isfinite(diagnostics.current)))
    assert torch.max(torch.abs(diagnostics.voltage)) > 0.0
    assert result.stats()["network_cuda_graph_active"] is True
    assert diagnostics.metadata["delay_phase_error_degrees"] < 3.0
    assert diagnostics.metadata["delay_reembedding_max_error"] < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_delayed_network_block_fdtd_runtime_matches_frequency_response() -> None:
    frequencies = torch.linspace(0.0, 6.0e9, 121, dtype=torch.float64)
    delays = (0.0, 0.3e-9)
    data, model = _delayed_two_port(frequencies, delays)
    block = mw.NetworkBlock(
        name="delayed_link",
        network=data,
        connections={"left": "p_left", "right": "p_right"},
        fit=False,
        model=model,
        delay_seconds=delays,
        max_delay_steps=128,
    )
    requested = (2.0e9,)
    solver = mw.Simulation.fdtd(
        _scene(block, device="cuda", sources=()),
        frequencies=requested,
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=False,
    ).prepare().solver
    runtime = solver._network_runtimes[0]
    warmup_steps = 256
    sample_count = 2048
    total_steps = warmup_steps + sample_count
    step = torch.arange(total_steps, device="cuda", dtype=torch.float64)
    drive = torch.cos(2.0 * torch.pi * requested[0] * float(solver.dt) * step)
    voltage_samples = torch.empty(
        (sample_count, 2),
        device="cuda",
        dtype=runtime.network_voltage.dtype,
    )
    current_samples = torch.empty_like(voltage_samples)
    fields = {field.data_ptr(): field for field in runtime.electric_fields}
    denominators = tuple(
        torch.sum(port_runtime.lumped.voltage_weights.square())
        for port_runtime in runtime.port_runtimes
    )

    for sample_index in range(total_steps):
        for field in fields.values():
            field.zero_()
        for port_index, port_runtime in enumerate(runtime.port_runtimes):
            lumped = port_runtime.lumped
            target = drive[sample_index] if port_index == 0 else drive.new_zeros(())
            values = lumped.voltage_weights * target / denominators[port_index]
            runtime.electric_fields[port_index].view(-1).index_copy_(
                0,
                lumped.linear_indices,
                values,
            )
        apply_network_runtimes(solver)
        if sample_index >= warmup_steps:
            output_index = sample_index - warmup_steps
            voltage_samples[output_index].copy_(runtime.network_voltage)
            current_samples[output_index].copy_(runtime.branch_current)

    measured_steps = torch.arange(
        warmup_steps,
        total_steps,
        device="cuda",
        dtype=torch.float64,
    )
    angle = 2.0 * torch.pi * requested[0] * float(solver.dt) * measured_steps
    projection = torch.complex(torch.cos(angle), torch.sin(angle))
    voltage = torch.sum(voltage_samples.to(torch.complex128) * projection[:, None], dim=0)
    current = torch.sum(current_samples.to(torch.complex128) * projection[:, None], dim=0)
    incident, reflected = voltage_current_to_power_waves(
        voltage,
        current,
        50.0,
    )
    sample_index = int(torch.argmin(torch.abs(frequencies - requested[0])))
    expected = data.s[sample_index].to(device="cuda") @ incident
    scale = torch.max(torch.abs(expected))
    relative_error = torch.max(torch.abs(reflected - expected)) / scale
    valid = torch.abs(expected) > 1.0e-6 * scale
    phase_error = torch.max(torch.abs(torch.angle(reflected[valid] / expected[valid])))
    phase_error_degrees = torch.rad2deg(phase_error)

    assert relative_error < 0.02, float(relative_error)
    assert phase_error_degrees < 3.0, float(phase_error_degrees)


def test_long_fractional_delay_measured_phase_error_is_below_three_degrees() -> None:
    delay = 40.5
    angular_frequency = 2.0 * math.pi * 0.005
    runtime = prepare_bidirectional_delay(
        (delay,),
        dt=1.0,
        max_delay_steps=64,
        device="cpu",
        dtype=torch.float64,
    )
    forward_input = torch.empty(1, dtype=torch.float64)
    reverse_input = torch.zeros(1, dtype=torch.float64)
    forward_output = torch.empty_like(forward_input)
    reverse_output = torch.empty_like(forward_input)
    samples = []
    for step in range(3000):
        forward_input.fill_(math.cos(angular_frequency * step))
        advance_bidirectional_delay(
            runtime,
            forward_input,
            reverse_input,
            forward_output,
            reverse_output,
        )
        samples.append(forward_output.clone())
    output = torch.cat(samples)[1000:]
    indices = torch.arange(1000, 3000, dtype=torch.float64)
    projection = torch.exp(1j * angular_frequency * indices)
    input_amplitude = torch.sum(torch.cos(angular_frequency * indices) * projection)
    output_amplitude = torch.sum(output.to(torch.complex128) * projection)
    transfer = output_amplitude / input_amplitude
    target = complex(
        math.cos(angular_frequency * delay),
        math.sin(angular_frequency * delay),
    )
    phase_error = torch.rad2deg(torch.abs(torch.angle(transfer * target.conjugate())))
    assert phase_error < 3.0
    assert torch.abs(torch.abs(transfer) - 1.0) < 1.0e-3
