"""Arbitrary and broadband source times on time-shifted surface sources (CUDA).

PlaneWave, GaussianBeam, TFSF beams and ModeSource cells sample the waveform at a
per-cell delayed time. A ``CustomSourceTime`` table is evaluated natively on the
GPU at those delayed times, so a cubic table sampled from a ``GaussianPulse``
must reproduce the analytic GaussianPulse-driven fields to the table's
interpolation error (far below the float32 field noise of the solver). Every run
is small (< 0.2 GB).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")

_C0 = 299792458.0
_FREQUENCY = 1.0e9
_PULSE = mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.4e9, amplitude=1.0)
_STEPS = 320
# Max trace deviation relative to the E (or H) trace peak. Cubic tables sampled
# at dt/4 interpolate the pulse to ~1e-7 relative; measured deviations are
# 1e-6..4e-6 (float32 round-off in the native kernel and field updates).
_TOLERANCE = 2.0e-5


def _scene(source, *, monitors):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=10),
        device="cuda",
    )
    scene.add_source(source)
    for monitor in monitors:
        scene.add_monitor(monitor)
    return scene


def _traces():
    return [
        mw.FieldTimeMonitor("on_axis", components=("Ex", "Hy"), position=(0.0, 0.0, 0.12), interval=1),
        mw.FieldTimeMonitor("off_axis", components=("Ex", "Ez"), position=(0.08, -0.06, 0.04), interval=1),
    ]


def _table(pulse, dt, *, oversample=4, jitter=0.0, steps=_STEPS):
    # Cover every per-cell delayed time: delays are bounded by the domain transit time.
    margin = 0.9 * np.sqrt(3.0) / _C0 + 4.0 * dt
    h = dt / oversample
    times = np.arange(-margin, steps * dt + margin, h)
    if jitter:
        rng = np.random.default_rng(5)
        times[1:-1] += rng.uniform(-jitter, jitter, times.size - 2) * h
    values = np.array([pulse.evaluate(float(t)) / pulse.amplitude for t in times])
    # Same reference frequency (launch dispersion tuning, beam/mode profiles) and
    # the same time-step bound as the analytic pulse.
    return mw.CustomSourceTime(times, values, amplitude=pulse.amplitude, interpolation="cubic",
                               frequency=pulse.frequency,
                               characteristic_frequency=pulse.characteristic_frequency)


def _run(make_source, source_time, dt):
    scene = _scene(make_source(source_time), monitors=_traces())
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_FREQUENCY],
        run_time=mw.TimeConfig(time_steps=_STEPS, dt=dt),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    traces = {}
    for name in ("on_axis", "off_axis"):
        payload = result.monitor(name)
        for component, values in payload["components"].items():
            traces[(name, component)] = np.asarray(values.detach().cpu().numpy()).reshape(-1)
    return traces


def _reference_dt(make_source):
    scene = _scene(make_source(_PULSE), monitors=())
    return float(mw.Simulation.fdtd(scene, frequencies=[_FREQUENCY]).prepare().solver.dt)


def _relative_deviation(make_source, table_kwargs=None):
    dt = _reference_dt(make_source)
    native = _run(make_source, _PULSE, dt)
    custom = _run(make_source, _table(_PULSE, dt, **(table_kwargs or {})), dt)
    return _trace_deviation(native, custom)


def _trace_deviation(native, custom):
    """Worst trace max deviation relative to the peak of its field family (E or H).

    Normalizing by the family peak keeps cross-polarized traces that are pure
    round-off (e.g. Ez of an x-polarized plane wave) from dominating the metric.
    """
    family_peak = {}
    for (_, component), reference in native.items():
        family_peak[component[0]] = max(family_peak.get(component[0], 0.0), float(np.max(np.abs(reference))))
    assert all(peak > 0.0 for peak in family_peak.values())
    return max(
        float(np.max(np.abs(custom[key] - reference))) / family_peak[key[1][0]]
        for key, reference in native.items()
    )


def _plane_wave(source_time):
    return mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=source_time)


def _gaussian_beam(source_time):
    return mw.GaussianBeam(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), beam_waist=0.12,
                           focus=(0.0, 0.0, 0.05), source_time=source_time)


def _tfsf_gaussian_beam(source_time):
    return mw.GaussianBeam(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), beam_waist=0.12,
                           focus=(0.0, 0.0, 0.0), source_time=source_time,
                           injection=mw.TFSF(bounds=((-0.16, 0.16), (-0.16, 0.16), (-0.16, 0.16))))


def _tfsf_plane_wave(source_time):
    return mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=source_time,
                        injection=mw.TFSF(bounds=((-0.16, 0.16), (-0.16, 0.16), (-0.16, 0.16))))


@pytest.mark.parametrize(
    "make_source",
    [_plane_wave, _gaussian_beam, _tfsf_gaussian_beam, _tfsf_plane_wave],
    ids=["plane_wave", "gaussian_beam", "tfsf_gaussian_beam", "tfsf_plane_wave"],
)
def test_sampled_gaussian_table_reproduces_gaussian_pulse_fields(make_source):
    deviation = _relative_deviation(make_source)
    print(f"{make_source.__name__}: {deviation:.3e}")
    assert deviation <= _TOLERANCE


def test_nonuniform_sampled_table_reproduces_plane_wave_fields():
    # Jittered knots exercise the native binary-search segment lookup.
    deviation = _relative_deviation(_plane_wave, {"jitter": 0.3})
    print(f"nonuniform plane wave: {deviation:.3e}")
    assert deviation <= _TOLERANCE


def test_table_mode_source_matches_gaussian_pulse_mode_source():
    core = mw.Material(eps_r=4.0)

    def make_source(source_time):
        return mw.ModeSource(position=(0.0, 0.0, -0.12), size=(0.36, 0.36, 0.0), direction="+",
                             polarization="Ex", source_time=source_time)

    def run(source_time, dt):
        scene = _scene(make_source(source_time), monitors=_traces())
        scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.16, 0.12, 1.0)),
                                         material=core))
        result = mw.Simulation.fdtd(
            scene, frequencies=[_FREQUENCY], run_time=mw.TimeConfig(time_steps=_STEPS, dt=dt),
            spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=False,
        ).run()
        return {
            (name, component): np.asarray(values.detach().cpu().numpy()).reshape(-1)
            for name in ("on_axis", "off_axis")
            for component, values in result.monitor(name)["components"].items()
        }

    probe = _scene(make_source(_PULSE), monitors=())
    probe.add_structure(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.16, 0.12, 1.0)),
                                     material=core))
    dt = float(mw.Simulation.fdtd(probe, frequencies=[_FREQUENCY]).prepare().solver.dt)
    deviation = _trace_deviation(run(_PULSE, dt), run(_table(_PULSE, dt), dt))
    print(f"mode source: {deviation:.3e}")
    assert deviation <= _TOLERANCE


def test_broadband_pulse_plane_wave_flux_is_flat_over_the_band():
    band = (0.5e9, 2.0e9)
    minimum_amplitude = 0.3
    pulse = mw.BroadbandPulse(freq_range=band, minimum_amplitude=minimum_amplitude)
    frequencies = np.linspace(band[0], band[1], 13)

    def run(source_time, normalize, reference_frequency=None):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.4, 0.4))),
            grid=mw.GridSpec.uniform(0.015),
            boundary=mw.BoundarySpec.pml(num_layers=12),
            device="cuda",
        )
        scene.add_source(_plane_wave(source_time))
        scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.15, frequencies=frequencies))
        result = mw.Simulation.fdtd(
            scene, frequencies=frequencies, reference_frequency=reference_frequency,
            run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=4),
            spectral_sampler=mw.SpectralSampler(normalize_source=normalize),
            full_field_dft=False,
        ).run()
        return np.asarray(result.monitor("flux")["flux"].detach().cpu().numpy(), dtype=float)

    normalized = run(pulse, True)
    # A fixed reference period keeps raw spectra proportional to the Fourier transform;
    # the per-frequency default would add a factor f per bin (FT x f).
    raw = run(pulse, False, reference_frequency=1.25e9)
    # Independent geometry reference: a wideband GaussianPulse in the same scene.
    reference = run(mw.GaussianPulse(frequency=1.25e9, fwidth=0.6e9), True)

    # Source-normalized flux is waveform independent (unit-power plane wave over
    # the computational aperture; the physical plane holds ~0.39 of it and the
    # aperture geometry drifts it by a few percent across this octave-plus band).
    np.testing.assert_allclose(normalized, reference, rtol=2.0e-3)
    assert normalized.max() / normalized.min() < 1.06
    # The injected power spectrum measured through the solver follows the
    # analytic |S(f)|^2 and covers the band at minimum_amplitude^2.
    measured = raw / reference
    expected = np.abs(pulse.spectrum(frequencies)) ** 2
    np.testing.assert_allclose(measured / measured.mean(), expected / expected.mean(), rtol=5.0e-3)
    assert measured.min() / measured.max() >= minimum_amplitude ** 2 * 0.99
