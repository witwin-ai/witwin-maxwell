"""Cross-check public output consumers against independently sampled fields."""
import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.representation import spectral_scale


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("field_pulse,port_pulse", [(False, False), (True, False), (False, True), (True, True)])
def test_port_tables_consume_the_joint_cw_or_pulse_time_plan(field_pulse, port_pulse):
    from witwin.maxwell.sources import compile_source_time
    from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators
    from witwin.maxwell.fdtd.runtime.spectral import compute_spectral_start_step, resolve_spectral_window_type
    from witwin.maxwell.fdtd.runtime.timeplan import compile_fdtd_time_plan

    def waveform(pulse):
        return mw.GaussianPulse(1e9, .5e9) if pulse else mw.CW(1e9)

    runtime = SimpleNamespace(frequencies=torch.tensor([1e9], dtype=torch.float64),
        excitation=None, lumped=None, wire_provider=None, sheet_feed=None, field_name="Ez",
        window_weights=None, voltage_phase_weights=None, current_phase_weights=None,
        observer_current_buffer=None, observer_window_buffer=None,
        observer_voltage_kernel_buffer=None, observer_current_kernel_buffer=None,
        accumulator=None, drive_accumulator=None)
    solver = SimpleNamespace(dt=1e-10, c=299792458., device="cpu", Ez=torch.zeros(1),
        _port_runtimes=(runtime,),
        _compiled_sources=({"kind": "point_dipole", "source_time": compile_source_time(waveform(field_pulse), default_frequency=1e9),
                            "reference_provider_kind": "additive_J", "reference_waveform_kind": "gaussian_pulse" if field_pulse else "cw"},))
    solver._resolve_spectral_window_type = lambda window: resolve_spectral_window_type(solver, window)
    solver._compute_spectral_start_step = lambda frequency, window_type: compute_spectral_start_step(solver, frequency, window_type=window_type)
    scene = SimpleNamespace(domain=SimpleNamespace(bounds=((-0.1, .1),) * 3), circuits=())
    plan = compile_fdtd_time_plan(solver, scene, mw.TimeConfig.auto(transient_cycles=50, steady_cycles=5),
        [1e9], default_source_frequency=1e9, excitations=(SimpleNamespace(source_time=waveform(port_pulse)),))
    prepare_port_spectral_accumulators(solver, plan["total_steps"], "hanning")
    if field_pulse or port_pulse:
        assert runtime.window_type == "none"
        assert bool(torch.all(runtime.window_weights == 1))
    else:
        assert runtime.window_type == "hanning"
        assert plan["spectral_start_step"] == 500
        assert int(torch.nonzero(runtime.window_weights[:, 0] > 0)[0]) == 501


@pytest.mark.parametrize("compiled", [False, True])
def test_lower_monitor_frequency_does_not_change_default_incident_waveform(compiled):
    from witwin.maxwell.compiler.sources import compile_fdtd_sources
    from witwin.maxwell.fdtd.runtime.timeplan import compile_fdtd_time_plan

    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, .1),) * 3), device="cpu",
        sources=(mw.PointDipole((0., 0., 0.)),))
    solver = SimpleNamespace(dt=1e-11, c=299792458.)
    solver._compiled_sources = compile_fdtd_sources(scene, default_frequency=1e9)
    for source in solver._compiled_sources:
        source.update(reference_provider_kind="additive_J", reference_waveform_kind="cw")
    if not compiled:
        del solver._compiled_sources
        with pytest.raises(AttributeError, match="_compiled_sources"):
            compile_fdtd_time_plan(solver, scene, mw.TimeConfig(time_steps=20), [1e9, .5e9],
                default_source_frequency=1e9)
        return
    plan = compile_fdtd_time_plan(solver, scene, mw.TimeConfig(time_steps=20), [1e9, .5e9],
        default_source_frequency=1e9, excitations=(SimpleNamespace(source_time=None),), normalize_source=True)
    assert solver._normalization_source_time["frequency"] == 1e9
    assert plan["reference_frequency"] == .5e9


def _port_scene(*, field_source=True):
    port = mw.LumpedPort(
        "feed", positive=(0., 0., .005), negative=(0., 0., -.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)),
        reference_impedance=50.,
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.mur(),
        ports=(port,), device="cuda",
        sources=(mw.PointDipole(position=(.005, 0., 0.), polarization="Ez",
            source_time=mw.GaussianPulse(3e9, .6e9)),) if field_source else (),
        monitors=(mw.PointMonitor("point", (0., 0., 0.), fields=("Ez",)),),
    )


@requires_cuda
@pytest.mark.parametrize("graph", [False, True])
def test_port_pulse_window_and_field_line_integral_share_the_public_clock(graph):
    results = []
    for window in ("hanning", "none"):
        result = mw.Simulation.fdtd(
            _port_scene(), frequency=3e9, run_time=mw.TimeConfig(time_steps=500),
            spectral_sampler=mw.SpectralSampler(window=window),
            full_field_dft=True, cuda_graph=graph,
        ).run()
        results.append(result)
        assert result.stats()["time_plan"]["window"] == "none"
        assert result.solver._port_runtimes[0].window_type == "none"
        fields = {component: result.field(component) for component in ("Ex", "Ey", "Ez")}
        integrated = result.solver._port_runtimes[0].geometry.integrate_voltage(fields)
        torch.testing.assert_close(
            result.port("feed").voltage.squeeze(), integrated.to(torch.complex128),
            rtol=3e-5, atol=1e-7,
        )
    torch.testing.assert_close(results[0].port("feed").voltage, results[1].port("feed").voltage)


def _unit_pulse_spectrum(solver, steps):
    # Independent analytical waveform: source amplitude and phase are physical
    # controls, so the normalization reference uses unit amplitude, zero phase.
    sigma = 1. / (2. * math.pi * .6e9)
    delay = 6. * sigma
    t = torch.arange(steps, dtype=torch.float64) * solver.dt
    sample_time = t + .5 * solver.dt
    wave = torch.exp(-.5 * ((sample_time - delay) / sigma).square()) * torch.cos(2. * math.pi * 3e9 * (sample_time - delay))
    # Same per-reference-period pulse scaling as the library spectra.
    scale = spectral_scale(steps, integration_dt=solver._spectral_integration_dt,
                           reference_frequency=solver._spectral_reference_frequency, frequency=3e9)
    return (wave * torch.exp(2j * math.pi * 3e9 * sample_time)).sum() * scale


@requires_cuda
def test_public_default_drive_frequency_survives_lower_frequency_monitor():
    scene = _port_scene().clone(sources=(mw.PointDipole((.005, 0., 0.), polarization="Ez"),))
    scene.add_monitor(mw.PlaneMonitor("lower", axis="z", position=0., fields=("Ez",), frequencies=(1.5e9,)))
    solver = mw.Simulation.fdtd(scene, frequency=3e9, excitations=mw.PortExcitation("feed"),
        run_time=mw.TimeConfig(time_steps=10), spectral_sampler=mw.SpectralSampler(normalize_source=True),
        cuda_graph=False).prepare().solver
    assert solver._normalization_source_time["frequency"] == 3e9
    assert solver._compiled_sources[0]["source_time"]["frequency"] == 3e9
    assert solver._port_runtimes[0].source_frequency == 3e9
    assert solver._time_plan["reference_frequency"] == 1.5e9


@requires_cuda
@pytest.mark.parametrize("graph", [False, True])
def test_port_only_normalization_matches_independent_unit_pulse_spectrum(graph):
    results = []
    for normalize in (False, True):
        result = mw.Simulation.fdtd(
            _port_scene(field_source=False), frequency=3e9,
            excitations=(mw.PortExcitation("feed", source_time=mw.GaussianPulse(3e9, .6e9, amplitude=2., phase=.3)),),
            run_time=mw.TimeConfig(time_steps=500),
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=normalize),
            full_field_dft=True, cuda_graph=graph,
        ).run()
        results.append(result)
    reference = _unit_pulse_spectrum(results[0].solver, 500).to("cuda")
    for actual, raw in (
        (results[1].field("Ez"), results[0].field("Ez")),
        (results[1].monitor("point")["data"], results[0].monitor("point")["data"]),
    ):
        torch.testing.assert_close(actual.to(torch.complex128), raw.to(torch.complex128) / reference, rtol=3e-5, atol=1e-6)


@requires_cuda
def test_normalized_port_amplitude_gradient_preserves_physical_drive_scale():
    amplitude = torch.tensor(1.5, device="cuda", requires_grad=True)
    result = mw.Simulation.fdtd(
        _port_scene(field_source=False), frequency=3e9,
        excitations=(mw.PortExcitation("feed", amplitude=amplitude,
            source_time=mw.GaussianPulse(3e9, .6e9)),),
        run_time=mw.TimeConfig(time_steps=240),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=True),
        cuda_graph=False,
    ).run()
    loss = result.monitor("point")["data"].abs().square().sum()
    loss.backward()
    torch.testing.assert_close(amplitude.grad, 2. * loss.detach() / amplitude.detach(), rtol=3e-4, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("region", ["point", "plane", "volume"])
def test_complex_field_time_record_retains_both_bloch_quadratures(region):
    size = {"point": (0., 0., 0.), "plane": (.2, .2, 0.), "volume": (.2, .2, .2)}[region]
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, .4),) * 3), grid=mw.GridSpec.uniform(.1),
        boundary=mw.BoundarySpec.faces(default="mur", x="bloch", bloch_wavevector=(math.pi / 1.6, 0., 0.)),
        sources=(mw.PointDipole(position=(-.2, 0., 0.), polarization="Ez", width=.1, source_time=mw.CW(1e9)),),
        monitors=(mw.FieldTimeMonitor("trace", components=("Ez", "Hy"), position=(.2, 0., 0.), size=size),),
        device="cuda",
    )
    result = mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=150),
        spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False).run()
    payload = result.monitor("trace")
    record = result.solver.time_observers[0]
    for component in ("Ez", "Hy"):
        metadata = record["component_meta"][component]
        selection = metadata["field_index"] if region == "point" else metadata["field_slice"] if region == "plane" else metadata["slices"]
        actual = torch.complex(getattr(result.solver, component)[selection], getattr(result.solver, component + "_imag")[selection])
        if region == "plane":
            actual = actual.squeeze(metadata["axis_code"])
        trace = payload["components"][component]
        assert trace.is_complex() and not trace.requires_grad
        assert float(actual.imag.abs().max()) > 1e-4
        torch.testing.assert_close(trace[-1], actual, rtol=0, atol=0)


@requires_cuda
def test_flux_time_rejects_complex_bloch_fields():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.4, .4),) * 3), grid=mw.GridSpec.uniform(.1),
        boundary=mw.BoundarySpec.faces(default="mur", x="bloch", bloch_wavevector=(math.pi / 1.6, 0., 0.)),
        monitors=(mw.FluxTimeMonitor("flux"),), device="cuda")
    with pytest.raises(ValueError, match="FluxTimeMonitor requires real"):
        mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=10)).run()
