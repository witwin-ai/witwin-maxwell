"""Independent finite-window contracts for analytic complex CW sources."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

import witwin.maxwell as mw
from witwin.maxwell.sources import compile_source_time
from witwin.maxwell.fdtd.runtime.timeplan import compile_fdtd_time_plan
from witwin.maxwell.fdtd.runtime.representation import (
    energy_representation, normalization_signal, spectral_phasor_factor,
)


def solver_for(waveforms, complex_fields=True):
    solver = SimpleNamespace(dt=1e-11, c=299792458., complex_fields_enabled=complex_fields,
        _compiled_sources=[dict(kind="point_dipole", source_time=compile_source_time(w, default_frequency=1e9),
                                reference_provider_kind="additive_J", reference_waveform_kind=w.kind)
                           for w in waveforms])
    scene = SimpleNamespace(domain=SimpleNamespace(bounds=((-0.1, 0.1),)*3), circuits=())
    compile_fdtd_time_plan(solver, scene, mw.TimeConfig(time_steps=117), [1e9],
                          default_source_frequency=1e9)
    return solver


@pytest.mark.parametrize('complex_fields', (False, True))
def test_finite_window_unit_reference_preserves_declared_cw_amplitude_and_phase(complex_fields):
    waveform = mw.CW(1e9, amplitude=3., phase=.47)
    solver = solver_for([waveform], complex_fields)
    compiled = solver._compiled_sources[0]['source_time']
    time = np.arange(13, 117)*solver.dt
    weights = np.hanning(len(time))
    reference = np.array([normalization_signal(solver, compiled, t) for t in time])
    phase = np.exp(2j*np.pi*1e9*time)
    scale = spectral_phasor_factor(solver)/weights.sum()
    reference_spectrum = scale*np.sum(weights*reference*phase)
    if complex_fields:
        physical = 3*np.exp(-1j*(2*np.pi*1e9*time+.47))
        assert abs(reference_spectrum-1) < 1e-14
        spectrum = scale*np.sum(weights*physical*phase)/reference_spectrum
        assert abs(spectrum-3*np.exp(-.47j)) < 1e-14
    else:
        assert np.all(reference.imag == 0)
        assert spectral_phasor_factor(solver) == 2


@pytest.mark.parametrize('reverse', (False, True))
def test_complex_mixed_cw_and_real_pulse_requires_separate_runs(reverse):
    waveforms = [mw.CW(1e9), mw.GaussianPulse(1e9, .5e9)]
    if reverse:
        waveforms.reverse()
    with pytest.raises(NotImplementedError, match='mix analytic CW and real-pulse'):
        solver_for(waveforms)
    assert solver_for(waveforms, False)._time_plan['source_representation'] == 'real_projected'


def test_pulse_complex_amplitude_and_algorithmic_energy_are_distinct_contracts():
    solver = solver_for([mw.GaussianPulse(1e9, .5e9)])
    assert spectral_phasor_factor(solver) == 2
    assert solver._time_plan['source_representation'] == 'real_projected'
    assert energy_representation(solver) == 'combined_real_imaginary_algorithmic_state'
    assert energy_representation(solver_for([mw.CW(1e9)], False)) == 'physical_real_field'


def test_multiple_cw_carriers_share_analytic_representation_without_source_normalization():
    solver = solver_for([mw.CW(1e9), mw.CW(1.37e9)])
    assert spectral_phasor_factor(solver) == 1
    assert solver._time_plan['source_representation'] == 'analytic_cw'
