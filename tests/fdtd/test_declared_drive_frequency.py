"""Every declared drive constrains sampling without changing source defaults."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.initialization import maximum_declared_frequency


def _scene(kind, *, default_field=False, device="cuda"):
    port = mw.LumpedPort(name="feed", positive=(0., 0., .005), negative=(0., 0., -.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)),
        reference_impedance=50.)
    circuits = ()
    if kind == "circuit":
        circuit = mw.Circuit("driven")
        node = circuit.node("input")
        circuit.add(mw.Resistor("R1", node, circuit.ground, 50.))
        circuit.add(mw.CurrentSource("I1", circuit.ground, node,
            waveform=mw.SineWaveform(0., .01, 3e12)))
        circuit.bind_port("feed", positive=node, negative=circuit.ground)
        circuits = (circuit,)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(),
        ports=(port,), circuits=circuits, device=device)
    if default_field:
        scene.add_source(mw.PointDipole(position=(.01, 0., 0.), polarization="Ez"))
    return scene


def test_declared_circuit_frequency_is_available_before_runtime_binding():
    scene = _scene("circuit", default_field=True, device="cpu")
    assert maximum_declared_frequency(scene, 3e9) == pytest.approx(3e12, rel=1e-7)
    assert scene.sources[0].source_time is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["port", "circuit"])
@pytest.mark.parametrize("default_field", [False, True])
def test_all_drive_carriers_limit_automatic_step_without_changing_default_source(kind, default_field):
    kwargs = {"excitations": mw.PortExcitation("feed", source_time=mw.CW(frequency=3e12))} if kind == "port" else {}
    solver = mw.Simulation.fdtd(_scene(kind, default_field=default_field), frequency=3e9,
        run_time=mw.TimeConfig(time_steps=8), cuda_graph=False, **kwargs).prepare().solver
    assert solver.dt <= (1 / (30 * 3e12)) * (1 + 1e-12)
    if default_field:
        assert solver._compiled_sources[0]["source_time"]["frequency"] == 3e9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_explicit_step_preserves_user_sampling_choice():
    solver = mw.Simulation.fdtd(_scene("port", default_field=True), frequency=3e9,
        excitations=mw.PortExcitation("feed", source_time=mw.CW(frequency=3e12)),
        run_time=mw.TimeConfig(time_steps=8, dt=1e-12), cuda_graph=False).prepare().solver
    assert solver.dt == 1e-12


def _finite_circuit_scene(waveform, *, device="cpu"):
    circuit = mw.Circuit("finite")
    node = circuit.node("input")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.))
    circuit.add(mw.CurrentSource("I1", circuit.ground, node, waveform=waveform))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    return _scene("port", device=device).clone(circuits=(circuit,))


@pytest.mark.parametrize("waveform,expected", [
    (mw.PulseWaveform(0., .01, width=2e-12), 5e11),
    (mw.PulseWaveform(0., .01, rise=2e-12, width=4e-12, fall=8e-12, period=1e-9), 5e11),
    (mw.PulseWaveform(0., .01, period=1e-9), 3e9),
    (mw.PulseWaveform(0., .01), 3e9),
    (mw.PiecewiseLinearWaveform(torch.tensor([0., 1e-9, 1.001e-9, 1.003e-9], dtype=torch.float64),
        torch.tensor([0., 0., .01, 0.], dtype=torch.float64)), 1e12),
    (mw.SineWaveform(0., .01, 1e9, damping=1e12), 1e12),
])
def test_finite_declared_time_scales_bound_resolution_without_infinite_jump_frequency(waveform, expected):
    assert maximum_declared_frequency(_finite_circuit_scene(waveform), 3e9) == pytest.approx(expected, rel=2e-7)


def test_pwl_resolution_is_invariant_under_collinear_resampling():
    # A 1 ns ramp remains a 1 ns physical feature with 4001 table samples.
    for samples in (2, 41, 4001):
        times = torch.linspace(0., 1e-9, samples, dtype=torch.float64)
        waveform = mw.PiecewiseLinearWaveform(times, times / 1e-9)
        assert maximum_declared_frequency(_finite_circuit_scene(waveform), 1e8) == pytest.approx(1e9)
    constant = mw.PiecewiseLinearWaveform([0., 1e-15, 1e-9], [3., 3., 3.])
    assert maximum_declared_frequency(_finite_circuit_scene(constant), 1e8) == 1e8


def test_dense_esd_table_uses_physical_rise_scale():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.)
    frequencies = [maximum_declared_frequency(
        _finite_circuit_scene(waveform.to_circuit_waveform(t_end=1e-9, samples=samples)), 1e8
    ) for samples in (401, 4001)]
    assert frequencies[1] == pytest.approx(frequencies[0], rel=1e-3)
    assert 1e9 < frequencies[1] < 1e10


def _short_waveform(kind):
    if kind == "pulse":
        return mw.PulseWaveform(0., .01, delay=2e-12, rise=2e-13, width=2e-13, fall=2e-13, period=1e-9)
    return mw.PiecewiseLinearWaveform(
        torch.tensor([0., 2e-12, 2.2e-12, 2.4e-12, 2.6e-12, 4e-12], dtype=torch.float64),
        torch.tensor([0., 0., .01, .01, 0., 0.], dtype=torch.float64),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["pulse", "pwl"])
def test_automatic_sampling_captures_finite_circuit_pulses_that_fit_between_coarse_steps(kind):
    scene = _finite_circuit_scene(_short_waveform(kind), device="cuda")
    results = []
    for dt, steps in ((None, 600), (2e-13 / 60, 1200), (8e-12, 4)):
        result = mw.Simulation.fdtd(scene, frequency=3e9,
            run_time=mw.TimeConfig(time_steps=steps, dt=dt),
            spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False).run()
        results.append(result)
    automatic, refined, coarse = results
    assert automatic.solver.dt <= (2e-13 / 30) * (1 + 2e-7)
    # This trapezoid has area I_peak * (width + (rise + fall) / 2).
    expected_area = .01 * 4e-13
    for result in (automatic, refined):
        samples = result.solver._circuit_runtimes[0].source_values["I1"]
        area = float(samples.sum() * result.solver.dt)
        assert area == pytest.approx(expected_area, rel=1e-3)
        assert float(result.circuit("finite").node_voltages.abs().max()) > .1
    torch.testing.assert_close(
        automatic.circuit("finite").node_voltages.abs().max(),
        refined.circuit("finite").node_voltages.abs().max(), rtol=.02, atol=1e-6,
    )
    # An explicitly requested stable dt remains authoritative, even when its
    # sample grid misses a short drive. This also makes the regression nonvacuous.
    assert coarse.solver.dt == 8e-12
    assert float(coarse.circuit("finite").node_voltages.abs().max()) == 0.
