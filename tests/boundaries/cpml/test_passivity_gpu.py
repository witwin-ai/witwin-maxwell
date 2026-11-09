"""Long-record CPML gates for GPU review; never execute without CUDA.

These are qualification cases, not a claim of a passive CPML closure. Run with
python -m pytest tests/boundaries/cpml/test_passivity_gpu.py -v
"""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.passivity import FIT_WINDOWS, total_energy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CPML qualification requires CUDA")


def _slab(*, fixed_k=False, absorber="cpml", steps=240000):
    # The real slab matches the diagnosed 1.28 m period / eps3 / 12-layer cell.
    k = .55 * 2 * math.pi * 1e9 / 299792458.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((- .64, .64),) * 3),
        grid=mw.GridSpec.anisotropic(1.28, .025, .025),
        boundary=mw.BoundarySpec.faces(
            x="bloch" if fixed_k else "periodic", y="bloch" if fixed_k else "periodic",
            z="pml", num_layers=40 if absorber == "auto" else 12,
            bloch_wavevector=(0., k if fixed_k else 0., 0.)), device="cuda",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
    )
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0., 0., 0.), size=(1.28, 1.28, .1)),
        material=mw.Material(eps_r=4. if fixed_k else 3.)))
    if fixed_k:
        # A finite-support cutoff pulse excites the slab-guided TM tail below
        # the fixed-k vacuum light line (0.55 GHz). Endpoints are exactly zero.
        times = torch.linspace(0., 40e-9, 4001, dtype=torch.float64)
        pulse = torch.sin(2 * math.pi * .4e9 * times) * torch.sin(math.pi * times / times[-1]).square()
        pulse[0] = pulse[-1] = 0
        # The cell is one 1.28 m cell along x: use the ideal (point) dipole profile.
        scene.add_source(mw.PointDipole(position=(0., 0., .025), polarization="Ez", profile="ideal",
            source_time=mw.CustomSourceTime(times=tuple(times.tolist()), amplitudes=tuple(pulse.tolist()),
                                          frequency=1e9, characteristic_frequency=2e9)))
    scene.add_monitor(mw.PointMonitor("field", position=(0., 0., 0.), fields=("Ez",)))
    sim = mw.Simulation.fdtd(scene, frequencies=(.3e9, 1e9), absorber=absorber,
        run_time=mw.TimeConfig(time_steps=steps, dt=1.6666666666666667e-11), full_field_dft=False)
    return sim


def _seed_tm(solver):
    # Source-free transverse Floquet q=1 seed, identical on periodic endpoints.
    y = torch.as_tensor(solver.scene.y_nodes64, device=solver.device)
    wave = torch.cos(2 * math.pi * (y - y[0]) / 1.28)
    solver.Ez.copy_(1e-5 * wave[None, :, None].expand_as(solver.Ez))


def test_real_periodic_eps3_slab_tm_explicit_cpml_trips():
    prepared = _slab(steps=80000).prepare()
    _seed_tm(prepared.solver)
    with pytest.raises(mw.FDTDPassivityError, match="real-field closed-transverse PML"):
        prepared.run()


def test_auto_real_periodic_slab_does_not_trip():
    prepared = _slab(absorber="auto", steps=80000).prepare()
    assert prepared.solver.active_absorber_type == "absorber"
    _seed_tm(prepared.solver)
    prepared.run()
    assert torch.isfinite(prepared.solver.Ez).all()
    assert prepared.solver._passivity_state is None


def test_open_pml_dipole_does_not_trip():
    scene = mw.Scene(domain=mw.Domain(bounds=((- .15, .15),) * 3),
        grid=mw.GridSpec.uniform(.03), boundary=mw.BoundarySpec.pml(num_layers=12), device="cuda")
    scene.add_source(mw.PointDipole(position=(0., 0., 0.), polarization="Ez",
                                  source_time=mw.GaussianPulse(frequency=1e9, fwidth=.3e9)))
    scene.add_monitor(mw.PointMonitor("field", position=(0., .03, 0.), fields=("Ez",)))
    prepared = mw.Simulation.fdtd(scene, frequency=1e9,
        run_time=mw.TimeConfig(time_steps=40000), full_field_dft=False).prepare()
    result = prepared.run()
    assert result.metadata["absorber"] == "cpml"
    assert prepared.solver._passivity_state is None
    assert torch.isfinite(prepared.solver.Ez).all()


def test_bloch_lc_60_degrees_default_profile_does_not_trip():
    # Same split-patch geometry and LC as the acceptance fixture, with pulsed
    # point-current excitation admitted by the fixed-k source contract.
    f0 = 3e9
    k = 2 * math.pi * f0 / 299792458. * math.sin(math.radians(60))
    scene = mw.Scene(domain=mw.Domain(bounds=((- .01, .01), (- .01, .01), (- .06, .06))),
        grid=mw.GridSpec.uniform(.0005), device="cuda",
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=16,
                                       bloch_wavevector=(0., k, 0.)))
    for bounds in ((-.007, -.0005), (.0005, .007)):
        scene.add_structure(mw.Structure(
            geometry=mw.PlanarPEC.rectangle((bounds, (-.007, .007)), normal_axis="z", coordinate=0.),
            material=mw.Material.pec()))
    scene.add_port(mw.LumpedPort(name="load", negative=(-.0005, 0., 0.), positive=(.0005, 0., 0.),
        voltage_path=mw.AxisPath("x"),
        current_surface=mw.Box(position=(-.00025, 0., 0.), size=(0., .0015, .0015)),
        termination=mw.SeriesRLC(l=5e-9, c=.5e-12)))
    scene.add_source(mw.PointDipole(position=(0., 0., -.03), polarization="Ex",
                                  source_time=mw.GaussianPulse(frequency=f0, fwidth=.9e9)))
    prepared = mw.Simulation.fdtd(scene, frequencies=(2.7e9, 3e9, 3.3e9),
        run_time=mw.TimeConfig(time_steps=120000), full_field_dft=False).prepare()
    assert prepared.solver.active_absorber_type == "cpml"
    prepared.run()
    assert prepared.solver._passivity_state["window_count"] >= FIT_WINDOWS
    assert torch.isfinite(total_energy(prepared.solver))


@pytest.mark.parametrize("graph", (False, True))
def test_guard_checkpoint_resume_and_graph_match(graph, tmp_path):
    # Source-free TM seed keeps this equality gate independent of pulse tails.
    simulation = _slab(steps=16000)
    simulation.config.cuda_graph = graph
    full = simulation.prepare()
    _seed_tm(full.solver)
    full.run()
    split = simulation.prepare()
    _seed_tm(split.solver)
    checkpoint = split.run_until(5312)
    path = tmp_path / "cpml.pt"
    checkpoint.save(path)
    resumed = simulation.prepare()
    resumed.run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    for key in ("history", "peak", "window_max"):
        assert torch.equal(full.solver._passivity_state[key], resumed.solver._passivity_state[key])
    for key in ("sample_count", "window_count"):
        assert full.solver._passivity_state[key] == resumed.solver._passivity_state[key]


CW_GROWTH_HORIZON = 7.399178367515743e-6
CW_REPORTED_ONSET = 2.3e-6
CW_TRANSIENT = 1e-6


def _cw_modulated_slab(*, device="cuda"):
    from benchmark.scenes import SCENARIOS, build_scene

    spec = SCENARIOS["modulated_slab"]
    # Keep the benchmark grid, modulation and CW drive. Match the three point
    # traces in the archived growth.py, without full-volume spectral buffers.
    scene = build_scene("modulated_slab").clone(device=device, monitors=[
        mw.FieldTimeMonitor(name, components=("Ex",), position=position, interval=4)
        for name, position in (("slab", (0., 0., 0.)),
                               ("down", (0., 0., .3)), ("up", (0., 0., -.3)))
    ])
    transient_cycles = math.ceil(CW_TRANSIENT * min(spec.frequencies))
    return mw.Simulation.fdtd(scene, frequencies=spec.frequencies, absorber="cpml",
        full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=1, transient_cycles=transient_cycles))


def _cw_growth_report(windows, *, dt, start_step, planned_steps, trip_step, error):
    # Diagnostic onset only: exceed the settled baseline by 0.1%. The guard's
    # independent log-envelope slope/uncertainty rule is left unchanged.
    ordered = [windows[step] for step in sorted(windows)]
    baseline = max((row[1] for row in ordered[:FIT_WINDOWS]), default=0.)
    onset_step = next((int(row[0]) for row in ordered[FIT_WINDOWS:]
                       if row[1] > 1.001 * baseline), None)
    return dict(scene="modulated_slab", absorber="cpml", dt=dt,
        planned_steps=planned_steps, planned_horizon_s=planned_steps * dt,
        guard_start_step=start_step, reported_onset_time_s=CW_REPORTED_ONSET,
        onset_definition="first window maximum > 1.001 times first-16-window maximum",
        onset_step=onset_step, onset_time_s=None if onset_step is None else onset_step * dt,
        trip_step=trip_step, trip_time_s=None if trip_step is None else trip_step * dt,
        baseline_energy=baseline, error=error,
        windows=[dict(end_step=int(row[0]), maximum=row[1], peak=row[2]) for row in ordered])


GROWTH_RATE_AMPLITUDE = 8.11e6  # 1/s, static z-CPML grazing-TM mode (docs/memory/fdtd-real-periodic-cpml-growth-2026-11-01.md)


def _record_guard(monkeypatch):
    from witwin.maxwell.fdtd.runtime import passivity

    windows, outcome = {}, dict(trip_step=None, message=None)
    check = passivity.check_passivity

    def record_check(history, state):
        # history is already on the CPU: recording adds no device sync.
        windows.update({int(row[0]): row.tolist() for row in history})
        try:
            check(history, state)
        except mw.FDTDPassivityError as error:
            outcome.update(trip_step=int(history[-1, 0]), message=str(error))
            raise

    monkeypatch.setattr(passivity, "check_passivity", record_check)
    return windows, outcome


def _write_report(tmp_path, record_property, name, report):
    import json

    destination = tmp_path / f"{name}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    record_property("passivity_report", str(destination))
    for key in ("trip_step", "trip_time_s", "onset_step", "onset_time_s"):
        record_property(key, report.get(key))
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}), flush=True)


def test_cw_modulated_slab_explicit_cpml_is_a_no_trip_control(monkeypatch, tmp_path, record_property):
    # The 2026-11-01 modulated_slab growth predates the modulation clock (b156c42b) and
    # partial-cell (f928126d) fixes. On the current base the CW-driven benchmark cell stays
    # transversely uniform, so the static grazing-TM CPML mode is never seeded: a no-trip control.
    simulation = _cw_modulated_slab()
    prepared = simulation.prepare()
    solver = prepared.solver
    assert solver.active_absorber_type == "cpml"
    dt = float(solver.dt)
    steps = math.ceil(CW_GROWTH_HORIZON / dt)
    simulation.config.run_time = mw.TimeConfig(time_steps=steps,
        transient_cycles=simulation.config.run_time.transient_cycles)
    windows, outcome = _record_guard(monkeypatch)
    prepared.run()
    report = _cw_growth_report(windows, dt=dt, start_step=solver._passivity_state["start_step"],
                               planned_steps=steps, trip_step=None, error=None)
    _write_report(tmp_path, record_property, "cw_modulated_slab_control", report)
    assert outcome["trip_step"] is None
    maxima = [row["maximum"] for row in report["windows"]]
    assert max(maxima) <= 1.05 * maxima[0]


def _cw_eps3_slab():
    # The seeded real-periodic eps3 cell (_slab), CW-driven instead of seeded. An off-axis
    # point dipole breaks transverse uniformity, so its start-up transient seeds the
    # transverse Floquet q=1 grazing-TM CPML mode.
    simulation = _slab(steps=1)
    scene = simulation.scene.clone(sources=[mw.PointDipole(
        position=(0., .3, .025), polarization="Ez", profile="ideal",
        source_time=mw.CW(frequency=1e9))])
    return mw.Simulation.fdtd(scene, frequencies=(1e9,), absorber="cpml", full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=1, dt=simulation.config.run_time.dt,
                               transient_cycles=math.ceil(CW_TRANSIENT * 1e9)))


def test_cw_driven_eps3_slab_explicit_cpml_trips(monkeypatch, tmp_path, record_property):
    from witwin.maxwell.fdtd.runtime import passivity

    simulation = _cw_eps3_slab()
    prepared = simulation.prepare()
    solver = prepared.solver
    assert not solver.complex_fields_enabled and solver.active_absorber_type == "cpml"
    dt = float(solver.dt)
    steps = math.ceil(CW_GROWTH_HORIZON / dt)
    simulation.config.run_time = mw.TimeConfig(time_steps=steps, dt=dt,
        transient_cycles=simulation.config.run_time.transient_cycles)
    # Prediction, stated before the run: the mode's energy grows at 2*8.11e6 1/s
    # (2.70e-4 per step). For a start-up seed fraction s of the CW energy, the diagnostic
    # onset (+0.1 %) is ln(1e-3/s)/(2*rate) after the source starts: 0.43-1.84 us for
    # s in [1e-16, 1e-6]. The guard needs one complete 16-window fit after its start and
    # after the mode reaches the window maxima, so the predicted trip is <= onset + one fit.
    fit = passivity.FIT_WINDOWS * passivity.WINDOW_SAMPLES * passivity.SAMPLE_INTERVAL * dt
    predicted_latest_trip = max(CW_TRANSIENT, math.log(1e-3 / 1e-16) / (2 * GROWTH_RATE_AMPLITUDE)) + fit
    windows, outcome = _record_guard(monkeypatch)
    try:
        with pytest.raises(mw.FDTDPassivityError, match="real-field closed-transverse PML"):
            prepared.run()
    finally:
        state = getattr(solver, "_passivity_state", None)
        report = _cw_growth_report(windows, dt=dt, start_step=None if state is None else state["start_step"],
                                   planned_steps=steps, trip_step=outcome["trip_step"], error=outcome["message"])
        report.update(scene="cw_eps3_slab", reported_onset_time_s=None, growth_rate_amplitude=GROWTH_RATE_AMPLITUDE,
                      predicted_latest_trip_s=predicted_latest_trip)
        _write_report(tmp_path, record_property, "cw_eps3_slab_trip", report)
    assert report["trip_time_s"] <= predicted_latest_trip


# Measured 2026-11-04 on the guard branch (master d02ee013 base): this point-dipole fixed-k
# eps4 cell grows at 2.09e-7 per step in log energy (stderr 4.6e-10 over 101 windows), inside
# the guard's documented undetected band (< 4e-7 per step). Above that criterion,
# complex fixed-k runs warn once and continue (user decision D2).
FIXED_K_SUBRESOLUTION_BAND = (1.5e-7, 3e-7)


def test_complex_fixed_k_eps4_slab_growth_is_below_guard_resolution(monkeypatch, tmp_path, record_property):
    windows, outcome = _record_guard(monkeypatch)
    prepared = _slab(fixed_k=True).prepare()
    prepared.run()
    assert outcome["trip_step"] is None
    rows = [windows[step] for step in sorted(windows)]
    steps = torch.tensor([row[0] for row in rows], dtype=torch.float64)
    log_energy = torch.tensor([row[1] for row in rows], dtype=torch.float64).log()
    centered = steps - steps.mean()
    slope = float((centered * (log_energy - log_energy.mean())).sum() / centered.square().sum())
    report = dict(scene="fixed_k_eps4_point_dipole", windows_fitted=len(rows), slope_per_step=slope,
                  band=FIXED_K_SUBRESOLUTION_BAND, guard_limit=4e-7,
                  status=prepared.solver.passivity_guard_status)
    _write_report(tmp_path, record_property, "fixed_k_eps4_subresolution", report)
    assert prepared.solver.passivity_guard_status == "active"
    assert FIXED_K_SUBRESOLUTION_BAND[0] <= slope <= FIXED_K_SUBRESOLUTION_BAND[1] < 4e-7


@pytest.mark.cuda
def test_complex_fixed_k_cw_eps4_slab_completes_with_warning_metadata():
    import numpy as np

    # Match the measured TM CW cell: k_parallel=0.25 k0, 0.9 f0, ppw160.
    # Its measured log-energy growth is 8.2e-7/step, above the unchanged limit.
    f0, resolution = 1e9, 160
    wavelength = 299792458. / f0
    kpar = .25 * 2 * math.pi / wavelength
    frequency = .9 * f0
    sine = .25 / .9
    cosine = math.sqrt(1 - sine**2)
    xy = np.linspace(-.05 * wavelength, .05 * wavelength, resolution // 10 + 1)
    z = np.linspace(-wavelength, wavelength, 2 * resolution + 1)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((xy[0], xy[-1]), (xy[0], xy[-1]), (z[0], z[-1]))),
        grid=mw.GridSpec.custom(xy, xy, z), device="cuda",
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml",
            num_layers=resolution // 2, bloch_wavevector=(kpar, 0., 0.)),
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 128), averaging="polarized"))
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0., 0., .013 * wavelength),
                        size=(wavelength, wavelength, .37 * wavelength)),
        material=mw.Material(eps_r=4.), name="slab"))
    scene.add_source(mw.PlaneWave(direction=(sine, 0., cosine), polarization=(cosine, 0., -sine),
        source_time=mw.CW(frequency=frequency),
        injection=mw.TFSF.slab(axis="z", bounds=(-.65 * wavelength, .65 * wavelength))))
    scene.add_monitor(mw.PointMonitor("field", position=(0., 0., 0.), fields=("Ex",)))
    prepared = mw.Simulation.fdtd(scene, frequency=frequency, absorber="cpml", shutoff=0.,
        full_field_dft=False, spectral_sampler=mw.SpectralSampler(window="hanning"),
        run_time=mw.TimeConfig.auto(transient_cycles=300, steady_cycles=100)).prepare()
    assert prepared.solver.complex_fields_enabled
    with pytest.warns(RuntimeWarning, match="complex fixed-k Bloch.*slope=.*stderr=.*window steps") as caught:
        result = prepared.run()
    assert len(caught) == 1
    assert result.metadata["passivity_guard"] == "warned: complex fixed-k growth"
    record = result.metadata["passivity_guard_warning"]
    assert record == prepared.solver.passivity_guard_warning == result.solver_stats["passivity_guard_warning"]
    assert record["slope"] + 3 * record["stderr"] > 4e-7
    assert record["window"][1] == record["step"]
    assert record["step"] < result.stats()["steps_run"]
    assert torch.isfinite(total_energy(prepared.solver))
