"""CPU spectrum diagnostics and unchanged prepared source terms."""
import math
import struct
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation import incident_line_spectrum
from witwin.maxwell.fdtd.excitation.incident_line_spectrum import (
    check_incident_line_spectrum,
    incident_line_band_edge,
)


def _check(waveform, *, steps=4096, dt=1e-12, f_edge=None, modulus=1.0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fractions = check_incident_line_spectrum(waveform, f_edge=f_edge, dt=dt,
            ds=.025, line_name="test-line", record_steps=steps, pole_period_modulus=modulus)
    return fractions, caught


@pytest.mark.parametrize("c,dt,ds", [(299792458., 1e-12, .025), (1., .3, 1.),
                                    (2., .5, 1.), (3., .5, 1.)])
def test_axial_band_edge_preserves_bits(c, dt, ds):
    previous = math.asin(min(1.0, c * dt / ds)) / (math.pi * dt)
    for edge in (incident_line_band_edge(c, dt, ds), incident_line_band_edge(c, dt, ds, k_t=0.0)):
        assert struct.pack("d", edge) == struct.pack("d", previous)


@pytest.mark.parametrize("k_t", [2., 8., 16.])
def test_fixed_k_band_edge_increases_and_saturates(k_t):
    c, dt, ds = 1., .25, .5
    edge = incident_line_band_edge(c, dt, ds, k_t)
    scaled = c * dt / ds * math.sqrt(1 + (k_t * ds / 2)**2)
    assert edge > incident_line_band_edge(c, dt, ds)
    assert edge == math.asin(min(1.0, scaled)) / (math.pi * dt)
    if scaled >= 1:
        assert edge == 1 / (2 * dt)
    else:
        assert edge < 1 / (2 * dt)


def test_sample_offset_changes_asymmetric_waveform_and_preserves_default_bits():
    class Waveform:
        def evaluate(self, t):
            return math.exp(-((t - 2.3) / 1.7)**2) * (1 + .2 * t) * math.cos(1.7 * t)

    waveform = Waveform()
    dt, steps, edge = .3, 32, .7
    kwargs = dict(f_edge=edge, dt=dt, ds=.025, line_name="test-line", record_steps=steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        default = check_incident_line_spectrum(waveform, **kwargs)
        explicit = check_incident_line_spectrum(waveform, **kwargs, sample_offset=1.0)
        shifted = check_incident_line_spectrum(waveform, **kwargs, sample_offset=.5)

    # Reproduce the original entry-E clock and spectrum arithmetic independently.
    times = (np.arange(steps, dtype=np.float64) + 1) * dt
    samples = np.array([waveform.evaluate(float(t)) for t in times], dtype=np.float64)
    samples /= float(np.max(np.abs(samples)))
    dc = abs(float(samples.sum()))
    edge_magnitude = float(abs(np.dot(samples, np.exp(-2j * np.pi * edge * times))))
    peak = max(float(np.abs(np.fft.rfft(samples, n=8 * steps)).max()), edge_magnitude, dc)
    previous = dict(edge_fraction=edge_magnitude / peak, dc_fraction=dc / peak)
    for key in previous:
        assert struct.pack("d", default[key]) == struct.pack("d", previous[key])
        assert struct.pack("d", explicit[key]) == struct.pack("d", previous[key])
        assert shifted[key] != default[key]


def test_prepared_line_band_edge_clock_and_cache(monkeypatch):
    line = SimpleNamespace(ds=.5)
    waveform = dict(kind="gaussian", frequency=1., amplitude=1.)
    solver = SimpleNamespace(c=1., dt=.25, _tfsf_state={"auxiliary_grid": line},
        _compiled_sources=[dict(kind="plane_wave", name="drive", source_time=waveform,
                                injection={"kind": "tfsf"})])
    calls = []

    def capture(source_time, **kwargs):
        assert source_time is waveform
        calls.append(kwargs)

    monkeypatch.setattr(incident_line_spectrum, "check_incident_line_spectrum", capture)
    checked = set()
    for k_t, offset in ((0., 1.), (2., 1.), (2., .5), (8., .5), (16., .5), (0., 1.)):
        if k_t != 0 or offset != 1:
            line.band_edge_transverse_wavenumber = k_t
            line.drive_sample_offset = offset
        else:
            line.__dict__.pop("band_edge_transverse_wavenumber", None)
            line.__dict__.pop("drive_sample_offset", None)
        for _ in range(2):
            incident_line_spectrum.check_prepared_source_spectra(solver, record_steps=32, checked=checked)
    assert len(calls) == len(checked) == 5
    for call, (k_t, offset) in zip(calls, ((0., 1.), (2., 1.), (2., .5), (8., .5), (16., .5))):
        assert call == dict(f_edge=incident_line_band_edge(solver.c, solver.dt, line.ds, k_t),
                           dt=solver.dt, ds=line.ds, line_name="drive", record_steps=32,
                           sample_offset=offset)


def test_exact_synthetic_bin_and_off_bin_magnitudes():
    class Waveform:
        def evaluate(self, t):
            return .2 + np.cos(2 * np.pi * 3 * t / 64)

    waveform = Waveform()
    samples = np.array([waveform.evaluate(n) for n in range(1, 65)])
    for edge in (3 / 64, 3.25 / 64):
        fractions, caught = _check(waveform, steps=64, dt=1, f_edge=edge)
        edge_mag = abs(np.dot(samples, np.exp(-2j * np.pi * edge * np.arange(1, 65))))
        peak = max(abs(np.fft.rfft(samples, n=512)).max(), edge_mag)
        assert fractions["edge_fraction"] == pytest.approx(edge_mag / peak, abs=1e-14)
        assert fractions["dc_fraction"] == pytest.approx(abs(samples.sum()) / peak, abs=1e-14)
        assert len(caught) == 2
        assert all(item.category is RuntimeWarning for item in caught)
        assert "test-line" in str(caught[0].message)
        assert "f_edge=" in str(caught[0].message)
        assert "static charge/field" in str(caught[1].message)
        assert "dc_fraction x (0.6-3)" in str(caught[1].message)


def test_zero_cw_and_pole_attenuation():
    for waveform in (mw.CW(frequency=2.4e9), mw.GaussianPulse(frequency=2.4e9, fwidth=1.92e9, amplitude=0)):
        fractions, caught = _check(waveform, f_edge=3.8e9)
        assert fractions == dict(edge_fraction=0.0, dc_fraction=0.0)
        assert not caught
    waveform = mw.RickerWavelet(frequency=3.8e9)
    assert any("axial band edge" in str(w.message) for w in _check(waveform, f_edge=3.8e9)[1])
    assert not _check(waveform, f_edge=3.8e9, modulus=.5)[1]


@pytest.mark.parametrize("steps,ratio,expected", [(16, 0.5e-6, False), (16, 2e-6, True),
                                                   (4096, 2e-6, False), (4096, 1e-5, True)])
def test_fp32_floor(steps, ratio, expected):
    class Waveform:
        def evaluate(self, t):
            # Coherent Nyquist carrier of unit magnitude, known DC ratio.
            return (-1.0) ** round(t) + ratio

    fractions, caught = _check(Waveform(), dt=1, steps=steps)
    assert fractions["dc_fraction"] == pytest.approx(ratio, abs=1e-14)
    assert bool(caught) is expected


def _simulation(waveform, *, direction=None, slab=False, source=None):
    if source is None:
        source = mw.PointDipole(position=(0., 0., 0.), polarization="Ez", source_time=waveform, name="drive")
    if direction is not None:
        injection = (mw.TFSF.slab(axis="z", bounds=(-.05, .05)) if slab else
                     mw.TFSF(bounds=((-0.05, .05),) * 3))
        source = mw.PlaneWave(direction=direction, polarization="Ex" if direction[0] == 0 else "Ey",
                              source_time=waveform, injection=injection, name="drive")
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.15, .15),) * 3),
        grid=mw.GridSpec.uniform(.025), boundary=mw.BoundarySpec.pml(num_layers=3),
        sources=[source], device="cpu")
    return mw.Simulation.fdtd(scene, frequency=2.4e9, backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=4096), full_field_dft=False,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))


@pytest.mark.parametrize("waveform,expect_dc", [
    (mw.GaussianPulse(frequency=2.4e9, fwidth=.8 * 2.4e9), True),
    (mw.RickerWavelet(frequency=2.4e9), False),
    (mw.GaussianPulse(frequency=2.4e9, fwidth=.1 * 2.4e9), False),
])
def test_point_dipole_dc_at_prepare(waveform, expect_dc):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _simulation(waveform).prepare()
    dc = [w for w in caught if "waveform has DC content" in str(w.message)]
    assert len(dc) == int(expect_dc)
    assert not any("axial band edge" in str(w.message) for w in caught)


@pytest.mark.parametrize("kind", ["uniform", "custom_J", "custom_M", "custom_JM", "plane", "beam"])
def test_other_current_and_soft_sources_receive_dc_only(kind):
    waveform = mw.GaussianPulse(frequency=2.4e9, fwidth=1.92e9)
    if kind == "uniform":
        source = mw.UniformCurrentSource(size=(.05, .05, .05), polarization="Ez", source_time=waveform)
    elif kind.startswith("custom"):
        axis = np.linspace(-.05, .05, 3)
        components = {}
        if "J" in kind:
            components["Jz"] = np.ones((3, 3, 3))
        if "M" in kind:
            components["My"] = np.ones((3, 3, 3))
        source = mw.CustomCurrentSource(mw.CurrentDataset((axis, axis, axis), components), source_time=waveform)
    elif kind == "plane":
        source = mw.PlaneWave(direction=(0., 0., 1.), polarization="Ex", source_time=waveform)
    else:
        source = mw.GaussianBeam(direction=(0., 0., 1.), polarization="Ex", beam_waist=.05, source_time=waveform)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _simulation(waveform, source=source).prepare()
    assert sum("waveform has DC content" in str(w.message) for w in caught) == 1
    assert not any("axial band edge" in str(w.message) for w in caught)


@pytest.mark.parametrize("direction,slab", [((0., 0., 1.), False), ((1., 0., 0.), False),
                                           ((.6, 0., .8), False), ((0., 0., 1.), True)])
@pytest.mark.parametrize("broad", [True, False])
def test_actual_auxiliary_line_edge_and_reprepare(direction, slab, broad):
    waveform = (mw.GaussianPulse(frequency=3.8e9, fwidth=3e9) if broad else
                mw.GaussianPulse(frequency=1e9, fwidth=.15e9))
    simulation = _simulation(waveform, direction=direction, slab=slab)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = simulation.prepare()
        simulation.prepare()
    edges = [w for w in caught if "axial band edge" in str(w.message)]
    dc = [w for w in caught if "waveform has DC content" in str(w.message)]
    assert len(edges) == int(broad)
    assert len(dc) <= 1
    aux = prepared.solver._tfsf_state["auxiliary_grid"]
    assert aux.ds > 0
    if broad:
        assert "edge_fraction=" in str(edges[0].message)


@pytest.mark.parametrize("direction", [None, (0., 0., 1.)])
def test_warning_filters_never_change_compiled_terms(direction):
    waveform = mw.GaussianPulse(frequency=2.4e9, fwidth=1.92e9)
    solvers = []
    for action in ("always", "ignore"):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter(action)
            solvers.append(_simulation(waveform, direction=direction).prepare().solver)

    def tensors(solver):
        terms = [*(solver._source_terms or ()), *(solver._electric_source_terms or ()),
                 *(solver._magnetic_source_terms or ())]
        if solver._tfsf_state:
            terms += solver._tfsf_state["electric_terms"] + solver._tfsf_state["magnetic_terms"]
        return [value for term in terms for value in term.values() if torch.is_tensor(value)]

    left, right = map(tensors, solvers)
    assert left and len(left) == len(right)
    assert all(torch.equal(a, b) for a, b in zip(left, right))
    assert solvers[0]._source_time == solvers[1]._source_time
