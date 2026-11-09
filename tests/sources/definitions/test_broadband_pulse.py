import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.sources import (
    compile_source_time,
    evaluate_source_time,
    evaluate_source_time_normalization,
)
from witwin.maxwell.waveforms import (
    SOURCE_TIME_KIND_BROADBAND_PULSE,
    evaluate_time_shifted_waveform_torch,
)

_BANDS = [(1.0e9, 2.0e9), (1.0e9, 10.0e9), (0.1e9, 10.0e9), (190.0e12, 210.0e12)]


def _sampled_spectrum(pulse, frequencies):
    """Brute-force exp(+i omega t) transform of the time-domain waveform."""
    dt = 1.0 / (40.0 * pulse.characteristic_frequency)
    times = np.arange(0.0, 2.0 * pulse.settling_time, dt)
    samples = np.array([pulse.evaluate(float(t)) for t in times])
    return (samples[None, :] * np.exp(2j * np.pi * np.asarray(frequencies)[:, None] * times[None, :])).sum(1) * dt


@pytest.mark.parametrize("band", _BANDS)
@pytest.mark.parametrize("minimum_amplitude", [0.06, 0.3, 0.49])
def test_broadband_pulse_spectrum_covers_band_at_minimum_amplitude(band, minimum_amplitude):
    pulse = mw.BroadbandPulse(freq_range=band, minimum_amplitude=minimum_amplitude)
    frequencies = np.linspace(band[0], band[1], 257)
    sampled = np.abs(_sampled_spectrum(pulse, frequencies))
    # The coverage contract holds on the independently sampled time waveform.
    assert sampled.min() / sampled.max() >= minimum_amplitude * (1.0 - 1.0e-4)
    # ... and is tight: fwidth is the narrowest width meeting it (band edges hit it).
    assert sampled.min() / sampled.max() == pytest.approx(minimum_amplitude, rel=2.0e-3)
    # The analytic spectrum agrees with the sampled transform.
    analytic = pulse.spectrum(frequencies)
    assert np.max(np.abs(np.abs(analytic) - sampled)) <= 1.0e-6 * sampled.max()


@pytest.mark.parametrize("band", _BANDS)
def test_broadband_pulse_is_zero_mean_and_starts_switched_off(band):
    pulse = mw.BroadbandPulse(freq_range=band, phase=0.4)
    dt = 1.0 / (40.0 * pulse.characteristic_frequency)
    times = np.arange(0.0, pulse.settling_time, dt)
    samples = np.array([pulse.evaluate(float(t)) for t in times])
    in_band = np.abs(pulse.spectrum([0.5 * (band[0] + band[1])]))[0]
    assert abs(samples.sum() * dt) <= 1.0e-6 * in_band
    assert abs(pulse.spectrum([0.0])[0]) <= 1.0e-12 * in_band
    assert abs(pulse.evaluate(0.0)) <= 2.0e-8
    assert abs(pulse.evaluate(pulse.settling_time)) <= 2.0e-8
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=0.05)


def test_broadband_pulse_narrowband_limit_is_the_band_centred_gaussian_pulse():
    band = (190.0e12, 210.0e12)
    pulse = mw.BroadbandPulse(freq_range=band, minimum_amplitude=0.3, amplitude=2.0, phase=0.3)
    expected_fwidth = 0.5 * (band[1] - band[0]) / math.sqrt(2.0 * math.log(1.0 / 0.3))
    assert pulse.frequency == pytest.approx(200.0e12)
    assert pulse.fwidth == pytest.approx(expected_fwidth, rel=1.0e-9)
    assert pulse.pedestal == 0.0 or abs(pulse.pedestal) < 1.0e-100
    gaussian = mw.GaussianPulse(frequency=pulse.frequency, fwidth=pulse.fwidth, amplitude=2.0, phase=0.3,
                                delay=pulse.delay)
    for t in np.linspace(0.0, pulse.settling_time, 37):
        assert pulse.evaluate(float(t)) == pytest.approx(gaussian.evaluate(float(t)), abs=1.0e-12)
    assert pulse.characteristic_frequency == pytest.approx(gaussian.characteristic_frequency)
    assert pulse.delay == pytest.approx(6.0 * pulse.sigma_t)
    assert pulse.settling_time == pytest.approx(gaussian.settling_time)


def test_broadband_pulse_offset_shifts_delay_in_bandwidth_units():
    base = mw.BroadbandPulse(freq_range=(1.0e9, 3.0e9))
    shifted = mw.BroadbandPulse(freq_range=(1.0e9, 3.0e9), offset=5.0)
    assert shifted.delay - base.delay == pytest.approx(5.0 / (2.0 * math.pi * 2.0e9))
    assert shifted.fwidth == base.fwidth
    assert shifted.evaluate(base.delay + 1.0 / (2.0 * math.pi * 2.0e9) * 5.0) == pytest.approx(
        base.evaluate(base.delay)
    )


def test_broadband_pulse_compiles_and_every_evaluator_agrees():
    pulse = mw.BroadbandPulse(freq_range=(1.0e9, 10.0e9), minimum_amplitude=0.25, amplitude=3.0, phase=0.7)
    compiled = compile_source_time(pulse, default_frequency=1.0e9)
    assert compiled["kind"] == "broadband_pulse"
    assert compiled["kind_code"] == SOURCE_TIME_KIND_BROADBAND_PULSE
    assert compiled["fwidth"] == pulse.fwidth
    assert compiled["pedestal"] == pulse.pedestal
    times = np.linspace(0.0, pulse.settling_time, 211)
    host = np.array([pulse.evaluate(float(t)) for t in times])
    np.testing.assert_allclose([evaluate_source_time(compiled, float(t)) for t in times], host, atol=1e-12)
    twin = evaluate_time_shifted_waveform_torch(compiled, torch.as_tensor(times, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(twin, host, atol=1e-12)
    from witwin.maxwell.fdtd.adjoint.core import _evaluate_source_time_tensor

    adjoint = _evaluate_source_time_tensor(compiled, torch.as_tensor(times, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(adjoint, host, atol=1e-12)


def test_broadband_pulse_normalization_strips_amplitude_and_phase_consistently():
    pulse = mw.BroadbandPulse(freq_range=(1.0e9, 10.0e9), amplitude=3.0, phase=0.7)
    reference = mw.BroadbandPulse(freq_range=(1.0e9, 10.0e9))
    compiled = compile_source_time(pulse, default_frequency=1.0e9)
    for t in np.linspace(0.0, pulse.settling_time, 31):
        expected = reference.evaluate(float(t))
        assert evaluate_source_time_normalization(pulse, float(t)) == pytest.approx(expected, abs=1e-12)
        assert evaluate_source_time_normalization(compiled, float(t)) == pytest.approx(expected, abs=1e-12)


def test_broadband_pulse_validation():
    with pytest.raises(ValueError):
        mw.BroadbandPulse(freq_range=(2.0e9, 1.0e9))
    with pytest.raises(ValueError):
        mw.BroadbandPulse(freq_range=(0.0, 1.0e9))
    with pytest.raises(ValueError, match="minimum_amplitude"):
        mw.BroadbandPulse(freq_range=(1.0e9, 2.0e9), minimum_amplitude=0.5)
    with pytest.raises(ValueError, match="minimum_amplitude"):
        mw.BroadbandPulse(freq_range=(1.0e9, 2.0e9), minimum_amplitude=0.05)


def test_broadband_pulse_drives_surface_and_point_sources():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    )
    pulse = mw.BroadbandPulse(freq_range=(1.0e8, 3.0e8))
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=pulse))
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, 0.0), source_time=pulse))
    compiled = compile_fdtd_sources(scene, default_frequency=2.0e8)
    assert {entry["source_time"]["kind"] for entry in compiled} == {"broadband_pulse"}
