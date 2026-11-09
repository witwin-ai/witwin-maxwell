import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.sources import (
    SOURCE_TIME_KIND_CUSTOM,
    compile_source_time,
    evaluate_source_time,
)


def _make_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    )


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Tier A: construction / interpolation / compilation (CPU, no solver)
# ---------------------------------------------------------------------------


def test_custom_source_time_interpolates_table_linearly():
    times = (0.0, 1.0, 2.0)
    amplitudes = (0.0, 10.0, 0.0)
    signal = mw.CustomSourceTime(times, amplitudes)

    assert signal.kind == "custom"
    assert signal.evaluate(0.5) == pytest.approx(5.0)
    assert signal.evaluate(1.5) == pytest.approx(5.0)
    assert signal.evaluate(1.0) == pytest.approx(10.0)
    # Zero outside the sampled range.
    assert signal.evaluate(-1.0) == 0.0
    assert signal.evaluate(3.0) == 0.0


def test_custom_source_time_amplitude_scales_table():
    signal = mw.CustomSourceTime((0.0, 2.0), (1.0, 3.0), amplitude=2.0)

    assert signal.evaluate(1.0) == pytest.approx(2.0 * 2.0)


def test_custom_source_time_sorts_unordered_samples():
    signal = mw.CustomSourceTime((2.0, 0.0, 1.0), (0.0, 0.0, 10.0))

    assert signal.evaluate(0.5) == pytest.approx(5.0)
    assert signal.settling_time == pytest.approx(2.0)
    assert signal.delay == pytest.approx(0.0)


def test_custom_source_time_reproduces_sampled_gaussian_pulse():
    pulse = mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9, amplitude=2.0)
    dt = 2.0e-12
    times = np.arange(0.0, 3.0e-9, dt)
    amplitudes = np.array([pulse.evaluate(float(t)) for t in times])

    signal = mw.CustomSourceTime(times, amplitudes)

    # Evaluate midway between samples: linear interpolation of a smooth,
    # well-resolved signal reproduces the analytic waveform tightly.
    probes = times[10:-10] + 0.5 * dt
    reconstructed = np.array([signal.evaluate(float(t)) for t in probes])
    analytic = np.array([pulse.evaluate(float(t)) for t in probes])
    np.testing.assert_allclose(reconstructed, analytic, atol=1e-4, rtol=1e-4)

    # FFT peak recovers the pulse center frequency to bin resolution.
    assert signal.characteristic_frequency == pytest.approx(1.0e9, abs=2.0e8)


def test_custom_source_time_compiles_and_round_trips_through_evaluate():
    pulse = mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9)
    dt = 2.0e-12
    times = np.arange(0.0, 3.0e-9, dt)
    amplitudes = np.array([pulse.evaluate(float(t)) for t in times])
    signal = mw.CustomSourceTime(times, amplitudes)

    compiled = compile_source_time(signal, default_frequency=1.0e9)

    assert compiled["kind"] == "custom"
    assert compiled["kind_code"] == SOURCE_TIME_KIND_CUSTOM
    assert compiled["times"] is not None
    for probe in (0.4e-9, 1.23e-9, 2.71e-9):
        assert evaluate_source_time(compiled, probe) == pytest.approx(signal.evaluate(probe))


def test_custom_source_time_from_callable_requires_characteristic_frequency():
    with pytest.raises(ValueError):
        mw.CustomSourceTime(lambda t: np.cos(2.0e9 * t))


def test_custom_source_time_from_callable_evaluates_and_compiles():
    fn = lambda t: np.cos(2.0 * np.pi * 1.0e9 * t)
    signal = mw.CustomSourceTime(fn, characteristic_frequency=1.0e9)

    assert signal.evaluate(0.25e-9) == pytest.approx(fn(0.25e-9))

    compiled = compile_source_time(signal, default_frequency=1.0e9)
    assert compiled["kind_code"] == SOURCE_TIME_KIND_CUSTOM
    assert evaluate_source_time(compiled, 0.7e-9) == pytest.approx(fn(0.7e-9))


def test_custom_source_time_rejects_mismatched_table():
    with pytest.raises(ValueError):
        mw.CustomSourceTime((0.0, 1.0, 2.0), (0.0, 1.0))


def test_point_dipole_compiles_custom_source_time_on_uniform_path():
    scene = _make_scene()
    signal = mw.CustomSourceTime((0.0, 1.0e-9, 2.0e-9), (0.0, 1.0, 0.0))
    scene.add_source(
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", source_time=signal)
    )

    compiled = compile_fdtd_sources(scene, default_frequency=1.0e9)

    assert compiled[0]["source_time"]["kind"] == "custom"
    assert compiled[0]["source_time"]["kind_code"] == SOURCE_TIME_KIND_CUSTOM


def test_surface_sources_compile_sampled_custom_source_time():
    table = mw.CustomSourceTime((0.0, 1.0e-9, 2.0e-9), (0.0, 1.0, 0.0), interpolation="cubic")
    scene = _make_scene()
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=table))
    scene.add_source(mw.GaussianBeam(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                     beam_waist=0.3, source_time=table))
    scene.add_source(mw.ModeSource(position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.0), source_time=table))
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=table,
                                  injection=mw.TFSF(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)))))

    compiled = compile_fdtd_sources(scene, default_frequency=1.0e9)

    for entry in compiled:
        assert entry["source_time"]["kind_code"] == SOURCE_TIME_KIND_CUSTOM
        assert entry["source_time"]["table"] is table.table
        assert entry["source_time"]["interpolation"] == "cubic"


@pytest.mark.parametrize(
    "make_source",
    [
        lambda st: mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=st),
        lambda st: mw.GaussianBeam(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                   beam_waist=0.3, source_time=st),
        lambda st: mw.ModeSource(position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.0), source_time=st),
    ],
)
def test_time_shifted_surface_sources_reject_callable_custom_source_time(make_source):
    scene = _make_scene()
    callable_time = mw.CustomSourceTime(lambda t: np.sin(2.0e9 * t), characteristic_frequency=1.0e9)
    scene.add_source(make_source(callable_time))
    with pytest.raises(ValueError, match="Tabulate it explicitly"):
        compile_fdtd_sources(scene, default_frequency=1.0e9)


def test_tfsf_plane_wave_keeps_callable_custom_source_time_on_auxiliary_line():
    # The TFSF plane wave drives a 1D auxiliary line on the solver time grid.
    scene = _make_scene()
    callable_time = mw.CustomSourceTime(lambda t: np.sin(2.0e9 * t), characteristic_frequency=1.0e9)
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                  source_time=callable_time,
                                  injection=mw.TFSF(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)))))
    compiled = compile_fdtd_sources(scene, default_frequency=1.0e9)
    assert compiled[0]["source_time"]["fn"] is not None


def test_time_shifted_source_term_rejects_callable_before_stepping():
    from witwin.maxwell.fdtd.excitation.temporal import build_source_term

    callable_time = compile_source_time(
        mw.CustomSourceTime(lambda t: 0.0, characteristic_frequency=1.0e9), default_frequency=1.0e9
    )
    patch = torch.ones((1, 2, 2))
    with pytest.raises(ValueError, match="Tabulate it explicitly"):
        build_source_term(None, field_name="Ex", offsets=(0, 0, 0), patch=patch,
                          delay_patch=torch.zeros_like(patch), source_time=callable_time)
    # Undelayed terms evaluate the callable on the solver grid.
    term = build_source_term(None, field_name="Ex", offsets=(0, 0, 0), patch=patch, source_time=callable_time)
    assert term["delay_patch"] is None


def test_custom_source_time_cubic_spline_is_natural_and_interpolates_knots():
    times = np.array([0.0, 0.3, 1.0, 1.4, 2.5])
    values = np.array([0.0, 1.0, -0.5, 0.25, 0.0])
    signal = mw.CustomSourceTime(times, values, interpolation="cubic")
    for t, v in zip(times, values):
        assert signal.evaluate(float(t)) == pytest.approx(float(v), abs=1e-12)
    from scipy.interpolate import CubicSpline

    reference = CubicSpline(times, values, bc_type="natural")
    probes = np.linspace(0.0, 2.5, 97)
    np.testing.assert_allclose([signal.evaluate(float(t)) for t in probes], reference(probes), atol=1e-12)
    assert signal.evaluate(-1.0e-9) == 0.0
    assert signal.evaluate(2.5 + 1.0e-9) == 0.0
    assert signal.table.uniform is False


def test_custom_source_time_table_torch_twin_matches_host_on_uniform_and_nonuniform_tables():
    rng = np.random.default_rng(7)
    uniform = np.linspace(-1.0, 3.0, 41)
    nonuniform = np.sort(np.concatenate(([-1.0, 3.0], rng.uniform(-1.0, 3.0, 60))))
    probes = np.linspace(-1.5, 3.5, 501)
    for times in (uniform, nonuniform):
        for interpolation in ("linear", "cubic"):
            signal = mw.CustomSourceTime(times, np.cos(2.0 * times) * np.exp(-0.2 * times ** 2),
                                         amplitude=1.5, interpolation=interpolation)
            host = np.array([signal.evaluate(float(t)) for t in probes])
            twin = 1.5 * signal.table.evaluate_torch(torch.as_tensor(probes, dtype=torch.float64)).numpy()
            np.testing.assert_allclose(twin, host, atol=1e-12)
        assert signal.table.uniform is (times is uniform)
    # Linear tables keep the previous zero-outside linear interpolation.
    linear = mw.CustomSourceTime(nonuniform, np.sin(nonuniform))
    np.testing.assert_allclose([linear.evaluate(float(t)) for t in probes],
                               np.interp(probes, nonuniform, np.sin(nonuniform), left=0.0, right=0.0), atol=1e-12)


def test_cubic_table_reproduces_gaussian_pulse_far_better_than_linear():
    pulse = mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9)
    h = 2.5e-11
    times = np.arange(0.0, pulse.settling_time, h)
    values = np.array([pulse.evaluate(float(t)) for t in times])
    probes = times[5:-5] + 0.37 * h
    analytic = np.array([pulse.evaluate(float(t)) for t in probes])
    errors = {}
    for interpolation in ("linear", "cubic"):
        signal = mw.CustomSourceTime(times, values, interpolation=interpolation)
        errors[interpolation] = np.max(np.abs([signal.evaluate(float(t)) for t in probes] - analytic))
    assert errors["cubic"] < 1e-5
    assert errors["cubic"] < 0.01 * errors["linear"]


def test_custom_source_time_validation():
    with pytest.raises(ValueError, match="interpolation"):
        mw.CustomSourceTime((0.0, 1.0), (0.0, 1.0), interpolation="spline")
    with pytest.raises(ValueError, match="strictly increasing"):
        mw.CustomSourceTime((0.0, 1.0, 1.0), (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        mw.CustomSourceTime((0.0, 1.0), (0.0, float("nan")))


# ---------------------------------------------------------------------------
# Tier B: real FDTD physics on CUDA
# ---------------------------------------------------------------------------


def _build_dipole_scene(source_time):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=source_time,
            name="src",
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("trace", components=("Ez",), position=(0.0, 0.0, 0.16), interval=1)
    )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_custom_source_time_matches_native_gaussian_pulse_in_fdtd():
    frequency = 1.0e9
    pulse = mw.GaussianPulse(frequency=frequency, fwidth=0.4e9, amplitude=100.0)

    run_time = mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15)
    sampler = mw.SpectralSampler(window="none")

    native_scene = _build_dipole_scene(pulse)
    native_result = mw.Simulation.fdtd(
        native_scene,
        frequencies=[frequency],
        run_time=run_time,
        spectral_sampler=sampler,
        full_field_dft=False,
    ).run()

    native_payload = native_result.monitor("trace")
    dt = float(native_result.solver.dt)
    native_t = _to_numpy(native_payload["t"])
    n_steps = int(round(float(native_t[-1]) / dt))

    # Sample the analytic pulse on the exact solver time grid, then drive an
    # identical run with the arbitrary-waveform table using a fixed step count.
    times = np.arange(n_steps + 2, dtype=np.float64) * dt
    amplitudes = np.array([pulse.evaluate(float(t)) for t in times])
    # Match the pulse characteristic frequency so both runs share the same dt
    # (and therefore the same time grid) for an element-wise trace comparison.
    custom = mw.CustomSourceTime(
        times, amplitudes, characteristic_frequency=pulse.characteristic_frequency
    )

    custom_scene = _build_dipole_scene(custom)
    custom_result = mw.Simulation.fdtd(
        custom_scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig(time_steps=n_steps),
        spectral_sampler=sampler,
        full_field_dft=False,
    ).run()

    native_trace = _to_numpy(native_result.monitor("trace")["field"])
    custom_trace = _to_numpy(custom_result.monitor("trace")["field"])

    n = min(native_trace.shape[0], custom_trace.shape[0])
    native_trace = native_trace[:n]
    custom_trace = custom_trace[:n]

    peak = np.max(np.abs(native_trace))
    assert peak > 0.0
    # The table sampled on the exact solver time grid must drive the field to
    # match the native analytic waveform to discretization precision.
    np.testing.assert_allclose(custom_trace, native_trace, atol=1e-3 * peak, rtol=2e-2)
