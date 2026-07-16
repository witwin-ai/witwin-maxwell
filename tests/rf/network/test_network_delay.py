import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.delay import (
    compile_network_delay,
    deembed_scattering,
    delay_phase_matrix,
    estimate_port_delays,
    reembed_scattering,
)
from witwin.maxwell.fdtd.delay import (
    advance_bidirectional_delay,
    prepare_bidirectional_delay,
)
from witwin.maxwell.rational import FitReport, NetworkFitReport, RationalModel, StateSpaceNetwork


def _matched_delayed_two_port(
    delay_seconds: tuple[float, float] = (1.25e-9, 1.25e-9),
    *,
    device: str = "cpu",
) -> mw.NetworkData:
    frequencies = torch.linspace(0.0, 1.0e9, 129, dtype=torch.float64, device=device)
    scattering = torch.zeros((129, 2, 2), dtype=torch.complex128, device=device)
    phase = delay_phase_matrix(frequencies, delay_seconds)
    scattering[:, 0, 1] = 0.8 * phase[:, 0, 1]
    scattering[:, 1, 0] = 0.8 * phase[:, 1, 0]
    return mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("left", "right"),
    )


def test_network_block_validates_explicit_and_automatic_delay_contract() -> None:
    network = _matched_delayed_two_port()
    block = mw.NetworkBlock(
        name="line",
        network=network,
        connections={"left": "p1", "right": "p2"},
        delay_seconds=[1.0e-9, 2.0e-9],
        max_delay_steps=200,
    )
    assert block.delay_seconds == (1.0e-9, 2.0e-9)
    auto = mw.NetworkBlock(
        name="auto_line",
        network=network,
        connections={1: "p1", 2: "p2"},
        delay_seconds="auto",
    )
    assert auto.delay_seconds == "auto"
    with pytest.raises(ValueError, match="one one-way delay"):
        mw.NetworkBlock(
            name="bad",
            network=network,
            connections={1: "p1", 2: "p2"},
            delay_seconds=(1.0e-9,),
        )
    with pytest.raises(ValueError, match="positive integer"):
        mw.NetworkBlock(
            name="bad_steps",
            network=network,
            connections={1: "p1", 2: "p2"},
            max_delay_steps=0,
        )
    trainable = mw.NetworkData(
        frequencies=network.frequencies,
        s=network.s.detach().clone().requires_grad_(),
        z0=network.z0,
        port_names=network.port_names,
    )
    fixed_model = StateSpaceNetwork(
        A=torch.empty((0, 0), dtype=torch.float64),
        B=torch.empty((0, 2), dtype=torch.float64),
        C=torch.empty((2, 0), dtype=torch.float64),
        D=0.01 * torch.eye(2, dtype=torch.float64),
        port_order=network.port_names,
    )
    with pytest.raises(RuntimeError, match="non-differentiable"):
        mw.NetworkBlock(
            name="trainable_auto",
            network=trainable,
            connections={1: "p1", 2: "p2"},
            fit=False,
            model=fixed_model,
            delay_seconds="auto",
        )


def test_auto_delay_uses_positive_exp_minus_iwt_sign_and_rank_deficient_split() -> None:
    network = _matched_delayed_two_port((1.0e-9, 2.0e-9))
    delay, rank, equation_count, residual, warnings = estimate_port_delays(network)
    assert delay == pytest.approx((1.5e-9, 1.5e-9), rel=2e-6, abs=1e-15)
    assert rank == 1
    assert equation_count == 2
    assert residual < 1e-15
    assert any("rank deficient" in warning for warning in warnings)

    phase = delay_phase_matrix(network.frequencies, delay)
    expected = torch.exp(
        1j * 2.0 * torch.pi * network.frequencies * sum(delay)
    )
    torch.testing.assert_close(phase[:, 0, 1], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rank_deficient_auto_delay_has_cpu_cuda_minimum_norm_parity() -> None:
    cpu = estimate_port_delays(_matched_delayed_two_port((1.0e-9, 2.0e-9)))[0]
    cuda = estimate_port_delays(
        _matched_delayed_two_port((1.0e-9, 2.0e-9), device="cuda")
    )[0]
    assert cpu == pytest.approx((1.5e-9, 1.5e-9), rel=2e-6, abs=1e-15)
    assert cuda == pytest.approx(cpu, rel=1e-12, abs=1e-16)


def test_delay_deembedding_round_trip_and_passivity_are_invariant() -> None:
    network = _matched_delayed_two_port()
    delay = (1.25e-9, 1.25e-9)
    core = deembed_scattering(network.s, network.frequencies, delay)
    reconstructed = reembed_scattering(core, network.frequencies, delay)
    torch.testing.assert_close(reconstructed, network.s, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.linalg.svdvals(core),
        torch.linalg.svdvals(network.s),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.max(torch.linalg.svdvals(core)).item() <= 0.8 + 1e-12


def test_compile_delay_records_fractional_phase_and_rejects_memory_limit() -> None:
    network = _matched_delayed_two_port()
    compiled, core = compile_network_delay(
        network,
        "auto",
        dt=0.1e-9,
        max_delay_steps=32,
    )
    assert compiled.delay_seconds == pytest.approx((1.25e-9, 1.25e-9), rel=2e-6)
    assert compiled.integer_steps == (12, 12)
    assert compiled.fractional_samples == pytest.approx((0.5, 0.5), abs=2e-5)
    assert compiled.phase_error_degrees < 3.0
    assert compiled.reembedding_max_error < 1e-12
    assert core.shape == network.s.shape
    report = compiled.update_report(
        FitReport(
            rms_error=0.0,
            max_error=0.0,
            relative_rms_error=0.0,
            relative_max_error=0.0,
            frequency_band=(0.0, 1.0e9),
            order=4,
            iterations=0,
        ),
        port_count=2,
    )
    assert report.delay_seconds == pytest.approx(compiled.delay_seconds)
    assert report.delay_estimation_rank == 1
    assert report.delay_equation_count == 2
    existing = NetworkFitReport(**report.__dict__)
    updated = compiled.update_report(existing, port_count=2)
    assert updated.delay_seconds == report.delay_seconds

    with pytest.raises(ValueError, match="exceeding max_delay_steps"):
        compile_network_delay(
            network,
            (1.25e-9, 1.25e-9),
            dt=0.1e-9,
            max_delay_steps=12,
        )
    with pytest.raises(ValueError, match="at least one FDTD step"):
        compile_network_delay(
            network,
            (0.05e-9, 1.25e-9),
            dt=0.1e-9,
            max_delay_steps=32,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        compile_network_delay(
            network,
            (1.25e-9, 1.25e-9),
            dt=0.1e-9,
            max_delay_steps=32,
            phase_tolerance_degrees=float("nan"),
        )


def test_fractional_phase_gate_applies_to_complete_reflection_path() -> None:
    frequencies = torch.linspace(0.0, 0.39, 65, dtype=torch.float64)
    delay = (1.01,)
    scattering = delay_phase_matrix(frequencies, delay)
    network = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("port",),
    )
    with pytest.raises(ValueError, match="phase error"):
        compile_network_delay(
            network,
            delay,
            dt=1.0,
            max_delay_steps=4,
            phase_tolerance_degrees=3.0,
        )


def test_delay_report_fields_survive_rational_model_persistence(tmp_path) -> None:
    network = _matched_delayed_two_port()
    compiled, _ = compile_network_delay(
        network,
        "auto",
        dt=0.1e-9,
        max_delay_steps=32,
    )
    report = compiled.update_report(
        FitReport(
            rms_error=0.0,
            max_error=0.0,
            relative_rms_error=0.0,
            relative_max_error=0.0,
            frequency_band=(0.0, 1.0e9),
            order=1,
            iterations=0,
        ),
        port_count=2,
    )
    model = RationalModel(
        poles=torch.tensor([-1.0], dtype=torch.complex128),
        residues=torch.zeros((2, 2, 1), dtype=torch.complex128),
        direct=0.01 * torch.eye(2, dtype=torch.float64),
        report=report,
    )
    path = tmp_path / "delayed_model.pt"
    model.save(path)
    loaded = RationalModel.load(path)
    assert isinstance(loaded.report, NetworkFitReport)
    assert loaded.report.delay_seconds == pytest.approx(compiled.delay_seconds)
    assert loaded.report.delay_phase_error_degrees == pytest.approx(
        compiled.phase_error_degrees
    )


def test_bidirectional_integer_fractional_delay_has_fixed_storage_and_unit_energy() -> None:
    runtime = prepare_bidirectional_delay(
        (2.0, 2.5, 0.0),
        dt=1.0,
        max_delay_steps=4,
        device="cpu",
        dtype=torch.float64,
    )
    assert runtime.forward_ring.shape == (3, 3)
    assert runtime.reverse_ring.shape == (3, 3)
    integer_runtime = prepare_bidirectional_delay(
        (4.0, 2.0),
        dt=1.0,
        max_delay_steps=4,
        device="cpu",
        dtype=torch.float64,
    )
    assert integer_runtime.forward_ring.shape == (2, 4)
    forward_input = torch.zeros(3, dtype=torch.float64)
    reverse_input = torch.zeros(3, dtype=torch.float64)
    forward_output = torch.empty_like(forward_input)
    reverse_output = torch.empty_like(reverse_input)
    pointers = (
        runtime.forward_ring.data_ptr(),
        runtime.reverse_ring.data_ptr(),
        runtime.forward_integer_sample.data_ptr(),
        runtime.reverse_integer_sample.data_ptr(),
    )
    forward_history = []
    reverse_history = []
    for step in range(96):
        forward_input.zero_()
        reverse_input.zero_()
        if step == 0:
            forward_input.fill_(1.0)
            reverse_input[1] = -1.0
        advance_bidirectional_delay(
            runtime,
            forward_input,
            reverse_input,
            forward_output,
            reverse_output,
        )
        forward_history.append(forward_output.clone())
        reverse_history.append(reverse_output.clone())
    forward = torch.stack(forward_history)
    reverse = torch.stack(reverse_history)
    assert forward[2, 0].item() == pytest.approx(1.0)
    assert forward[0, 2].item() == pytest.approx(1.0)
    assert reverse[2, 1].item() == pytest.approx(-1.0 / 3.0)
    assert torch.sum(forward[:, 1].square()).item() == pytest.approx(1.0, rel=1e-10)
    assert torch.sum(reverse[:, 1].square()).item() == pytest.approx(1.0, rel=1e-10)
    assert pointers == (
        runtime.forward_ring.data_ptr(),
        runtime.reverse_ring.data_ptr(),
        runtime.forward_integer_sample.data_ptr(),
        runtime.reverse_integer_sample.data_ptr(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_delay_hot_path_has_no_allocation_or_host_transfer() -> None:
    runtime = prepare_bidirectional_delay(
        (2.0, 2.5, 3.25),
        dt=1.0,
        max_delay_steps=8,
        device="cuda",
        dtype=torch.float32,
    )
    forward_input = torch.ones(3, device="cuda")
    reverse_input = -forward_input
    forward_output = torch.empty_like(forward_input)
    reverse_output = torch.empty_like(reverse_input)
    for _ in range(4):
        advance_bidirectional_delay(
            runtime, forward_input, reverse_input, forward_output, reverse_output
        )
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as profile:
        for _ in range(8):
            advance_bidirectional_delay(
                runtime, forward_input, reverse_input, forward_output, reverse_output
            )
        torch.cuda.synchronize()
    keys = {event.key for event in profile.key_averages()}
    assert not any("memcpy" in key.lower() or "item" in key.lower() for key in keys)
    assert sum(
        max(0, getattr(event, "self_device_memory_usage", 0))
        for event in profile.key_averages()
    ) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_delay_is_graph_replayable() -> None:
    runtime = prepare_bidirectional_delay(
        (2.0, 2.5),
        dt=1.0,
        max_delay_steps=4,
        device="cuda",
        dtype=torch.float32,
    )
    forward_input = torch.ones(2, device="cuda")
    reverse_input = -forward_input
    forward_output = torch.empty_like(forward_input)
    reverse_output = torch.empty_like(reverse_input)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            advance_bidirectional_delay(
                runtime, forward_input, reverse_input, forward_output, reverse_output
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        advance_bidirectional_delay(
            runtime, forward_input, reverse_input, forward_output, reverse_output
        )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.all(torch.isfinite(forward_output)))
    assert bool(torch.all(torch.isfinite(reverse_output)))
