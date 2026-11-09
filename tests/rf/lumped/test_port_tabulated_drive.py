"""Baseband, broadband and sampled-table waveforms as lumped/coaxial port drives."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import erf

import witwin.maxwell as mw
from witwin.maxwell.fdtd.ports import _source_descriptor


def test_baseband_pulses_are_sampled_tables_matching_their_analytic_waveforms():
    gaussian = mw.BasebandGaussianPulse(50e-12, offset=6.0, amplitude=2.0)
    assert isinstance(gaussian, mw.CustomSourceTime) and gaussian.table is not None
    t = np.linspace(0.0, 0.9e-9, 997)
    expected = 2.0 * np.exp(-0.5 * ((t - 6.0 * 50e-12) / 50e-12) ** 2)
    assert np.max(np.abs([gaussian.evaluate(v) for v in t] - expected)) <= 2.0e-6
    assert gaussian.characteristic_frequency == pytest.approx(3.0 / (2 * math.pi * 50e-12))
    assert gaussian.t_center == pytest.approx(300e-12)

    rectangle = mw.BasebandRectangularPulse(20e-12, 200e-12)
    sigma = 20e-12 / 2.5631031310892006
    start, stop = 5 * 20e-12, 5 * 20e-12 + 200e-12
    expected = 0.5 * (erf((t - start) / (math.sqrt(2) * sigma)) - erf((t - stop) / (math.sqrt(2) * sigma)))
    inside = t <= rectangle.times[-1]
    assert np.max(np.abs(np.asarray([rectangle.evaluate(v) for v in t])[inside] - expected[inside])) <= 2.0e-6
    assert mw.PortExcitation("p", source_time=rectangle).source_time is rectangle


@pytest.mark.parametrize("kwargs", [dict(twidth=0.0), dict(twidth=1e-11, offset=1.0)])
def test_baseband_gaussian_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        mw.BasebandGaussianPulse(**kwargs)


def test_port_drive_descriptor_rejects_callable_tables_and_complex_amplitudes():
    callable_time = mw.CustomSourceTime(lambda t: 0.0, characteristic_frequency=1e9)
    with pytest.raises(NotImplementedError, match="sampled-table CustomSourceTime"):
        _source_descriptor(mw.PortExcitation("p", source_time=callable_time), default_frequency=1e9,
                           device=torch.device("cpu"), dtype=torch.float64)
    with pytest.raises(ValueError, match="amplitude must be real"):
        _source_descriptor(mw.PortExcitation("p", amplitude=1j, source_time=mw.BasebandGaussianPulse(1e-11)),
                           default_frequency=1e9, device=torch.device("cpu"), dtype=torch.float64)
    kind, *_rest = _source_descriptor(mw.PortExcitation("p", source_time=mw.BroadbandPulse(freq_range=(1e9, 3e9))),
                                      default_frequency=1e9, device=torch.device("cpu"), dtype=torch.float64)
    assert kind == "broadband_pulse"


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="port drive physics requires CUDA")


def _terminated_coax():
    # Round coax (Z_ref 56.9 ohm) terminated in 50 ohm; see test_coaxial_lumped_physics.
    from tests.rf.lumped.test_coaxial_lumped_physics import _scene

    return _scene(load=50.0, probes=False)


def _reflection(source_time, frequencies, steps=6000):
    from tests.rf.lumped.test_coaxial_lumped_physics import _run

    data = _run(_terminated_coax(), source_time=source_time, frequencies=frequencies, steps=steps).port("feed")
    assert data.available_power is not None and torch.all(data.available_power > 0)
    return data.reflection_coefficient.cpu().numpy()


@cuda
def test_tabulated_port_drives_reproduce_the_analytic_gaussian_network_response():
    frequencies = (1.0e9, 2.0e9, 3.0e9)
    gaussian = mw.GaussianPulse(frequency=2.0e9, fwidth=1.5e9)
    reference = _reflection(gaussian, frequencies)
    times = np.linspace(0.0, 2.0 * gaussian.delay, 4001)
    table = mw.CustomSourceTime(times, [gaussian.evaluate(t) for t in times], interpolation="cubic",
                                frequency=gaussian.frequency, characteristic_frequency=gaussian.characteristic_frequency)
    sampled = _reflection(table, frequencies)
    pulse = mw.BroadbandPulse(freq_range=(0.8e9, 3.2e9))
    # The zero-mean pedestal is slow; run until the whole pulse has decayed.
    broadband = _reflection(pulse, frequencies, steps=int(1.5 * pulse.settling_time / 4.333e-13))
    baseband = _reflection(mw.BasebandGaussianPulse(50e-12), (0.1e9, *frequencies))
    # S11 is a property of the linear network: every decayed drive must agree.
    assert np.max(np.abs(sampled - reference)) <= 1.0e-4
    assert np.max(np.abs(broadband - reference)) <= 2.0e-3
    assert np.max(np.abs(baseband[1:] - reference)) <= 2.0e-3
    # The DC-containing pulse also resolves a near-DC sample. The 15 mm line is electrically
    # short at 0.1 GHz, so the 50 ohm load reads |S11| -> 0 although the line is 56.9 ohm:
    # measured 0.0023, as ideal line theory with the two end capacitances predicts.
    assert abs(baseband[0]) <= 5.0e-3
