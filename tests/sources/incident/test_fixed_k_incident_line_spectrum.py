"""CPU fixed-k band edge, drive clock, and preparation diagnostics."""
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.bloch_floquet_acceptance import common
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.fdtd.boundary.cpml import DEFAULT_CPML_CONFIG
from witwin.maxwell.fdtd.excitation.fixed_k import FixedKIncidentLine
from witwin.maxwell.fdtd.excitation.incident_line_spectrum import (
    check_incident_line_spectrum, check_prepared_source_spectra, incident_line_band_edge,
)
from witwin.maxwell.sources import evaluate_source_time


def _line(resolution, fraction):
    ds = common.LAMBDA0 / resolution
    # Production vacuum time step: source-resolution bound and 3D Courant bound.
    dt = min(1 / (30 * common.F0), ds / (common.C_0 * math.sqrt(3)))
    solver = SimpleNamespace(Nx=8, Ny=8, Nz=8, Ex=torch.zeros(1, dtype=torch.float64),
        device="cpu", dt=dt, c=common.C_0, eps0=EPSILON_0, mu0=MU_0,
        _requested_port_frequencies=(common.F0,),
        resolved_bloch_wavevector=(fraction * common.K0, 0., 0.),
        cpml_config=dict(DEFAULT_CPML_CONFIG))
    source = dict(injection=dict(axis="z"), direction=(0., 0., 1.), polarization=(0., 1., 0.),
        source_time=dict(kind="cw", frequency=common.F0, amplitude=0., phase=0., delay=0.))
    return FixedKIncidentLine(solver, source, dict.fromkeys("xyz", ds), (0, 0, 0))


def _periodic_eigenvalues(line):
    # Independent periodic Yee operator. Scale H by impedance to condition the
    # dense eigenproblem; this similarity transform preserves its eigenvalues.
    count = 8
    eta = math.sqrt(line.ce / line.ch)

    def curl(fields, forward):
        def derivative(component, axis):
            if axis != line.axis:
                return 1j * line.k[axis] * fields[component]
            return ((np.roll(fields[component], -1) - fields[component]) / line.ds
                    if forward else
                    (fields[component] - np.roll(fields[component], 1)) / line.ds)
        return np.array([derivative((i + 2) % 3, (i + 1) % 3) -
                         derivative((i + 1) % 3, (i + 2) % 3) for i in range(3)])

    def step(vector):
        electric, magnetic = vector.reshape(2, 3, count).copy()
        magnetic -= line.ch * eta * curl(electric, True)
        electric += line.ce / eta * curl(magnetic, False)
        return np.concatenate((electric.ravel(), magnetic.ravel()))

    basis = np.eye(6 * count, dtype=np.complex128)
    operator = np.column_stack([step(column) for column in basis])
    return np.linalg.eigvals(operator)


@pytest.mark.parametrize("resolution", (20, 40))
@pytest.mark.parametrize("fraction", (.25, .55))
def test_band_edge_matches_independent_dense_periodic_yee_oracle(resolution, fraction, record_property):
    line = _line(resolution, fraction)
    eigenvalues = _periodic_eigenvalues(line)
    maximum = np.max(np.abs(np.angle(eigenvalues))) / (2 * np.pi * line.dt)
    actual = incident_line_band_edge(common.C_0, line.dt, line.ds,
                                    line.band_edge_transverse_wavenumber)
    relative = abs(actual / maximum - 1)
    record_property("oracle", f"ppw={resolution}, k/k0={fraction}, f_up={actual:.17g}, "
                    f"eig_max={maximum:.17g}, relative_error={relative:.17g}")
    assert relative <= 1e-12
    assert np.max(np.abs(eigenvalues)) <= 1 + 2e-14


@pytest.mark.parametrize("courant", (.01, .2, 1 / math.sqrt(3), .99, 1., 1.1))
def test_axial_formula_is_bitwise_unchanged_and_clips_to_nyquist(courant):
    c, ds = common.C_0, .025
    dt = courant * ds / c
    old = math.asin(min(1., c * dt / ds)) / (math.pi * dt)
    assert incident_line_band_edge(c, dt, ds).hex() == old.hex()
    assert incident_line_band_edge(c, dt, ds, 1e9) == pytest.approx(.5 / dt, rel=2e-16)


@pytest.fixture
def host_preparation(monkeypatch):
    # Native Bloch admission/loading is CUDA-only; setup and the real incident
    # line allocate CPU tensors here, with no solver stepping or GPU access.
    for name in ("witwin.maxwell.simulation._require_cuda_scene",
                 "witwin.maxwell.fdtd.solver._require_cuda_scene",
                 "witwin.maxwell.fdtd.runtime.initialization.require_cuda_scene"):
        monkeypatch.setattr(name, lambda *args, **kwargs: None)
    for name in ("witwin.maxwell.fdtd.solver._get_fdtd_module",
                 "witwin.maxwell.fdtd.runtime.initialization.get_fdtd_module"):
        monkeypatch.setattr(name, lambda: object())
    scene_type = mw.Scene
    monkeypatch.setattr(common.mw, "Scene", lambda **kwargs: scene_type(**dict(kwargs, device="cpu")))


def _prepare(waveform=None, polarization="TE", steps=4096):
    return common.make_simulation(20, .55 * common.K0, polarization, 1.,
                                 source_time=waveform, time_steps=steps)


def test_fixed_k_edge_warning_once_and_reprepare_cached(host_preparation):
    simulation = _prepare(mw.GaussianPulse(frequency=6e9, fwidth=4e9))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = simulation.prepare()
        simulation.prepare()
    line = prepared.solver._tfsf_state["auxiliary_grid"]
    expected = incident_line_band_edge(prepared.solver.c, line.dt, line.ds,
                                      math.sqrt(sum(k * k for k in line.k)))
    edges = [w for w in caught if "f_edge=" in str(w.message)]
    assert len(edges) == 1
    assert f"f_edge={expected:.9g} Hz" in str(edges[0].message)
    assert expected > incident_line_band_edge(prepared.solver.c, line.dt, line.ds)


@pytest.mark.parametrize("polarization", ("TE", "TM"))
def test_r3_acceptance_pulse_does_not_warn(host_preparation, polarization):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _prepare(polarization=polarization).prepare()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_tm_dc_clock_matches_drive_and_exported_reference(host_preparation, monkeypatch):
    waveform = mw.GaussianPulse(frequency=1e9, fwidth=.8e9, delay=0.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = _prepare(waveform, "TM", steps=96).prepare()
    assert sum("waveform has DC content" in str(w.message) for w in caught) == 1
    solver = prepared.solver
    line = solver._tfsf_state["auxiliary_grid"]
    sampled = []
    evaluate = evaluate_source_time

    def record(source_time, time):
        value = evaluate(source_time, time)
        sampled.append((time, value))
        return value

    monkeypatch.setattr("witwin.maxwell.fdtd.excitation.incident_line_spectrum.evaluate_source_time", record)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fractions = check_incident_line_spectrum(line.source_time, f_edge=None, dt=line.dt,
            ds=line.ds, line_name="TM", record_steps=96, sample_offset=line.drive_sample_offset)
    diagnostic_samples = sampled.copy()
    sampled.clear()
    monkeypatch.setattr("witwin.maxwell.fdtd.excitation.fixed_k.evaluate_source_time", record)
    reference = torch.zeros_like(line.spectrum)
    for step in range(96):
        line.advance_magnetic()
        line.advance_electric()
        reference += torch.exp(1j * line.omega * ((step + 1) * line.dt))[:, None] * line.electric[:, line.reference_index]
    assert sampled == diagnostic_samples
    assert [t for t, _ in sampled] == [step * line.dt for step in range(96)]
    values = np.array([v for _, v in sampled])
    assert fractions["dc_fraction"] == pytest.approx(abs(values.sum()) / abs(np.fft.rfft(values, n=8*96)).max())
    exported = line.incident_spectrum(solver)["electric"]
    scale = line.dt * line.omega / (2 * math.pi)
    assert reference.abs().max() > 0
    torch.testing.assert_close(exported, reference * scale[:, None], rtol=2e-13, atol=0.)


def test_cache_tracks_discrete_k_and_drive_offset(host_preparation):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = _prepare().prepare().solver
        checked = set()
        check_prepared_source_spectra(solver, record_steps=4096, checked=checked)
        check_prepared_source_spectra(solver, record_steps=4096, checked=checked)
        assert len(checked) == 1
        line = solver._tfsf_state["auxiliary_grid"]
        line.k = tuple(2 * k for k in line.k)
        check_prepared_source_spectra(solver, record_steps=4096, checked=checked)
        assert len(checked) == 2
        line.drive_sample_offset = 1.0
        check_prepared_source_spectra(solver, record_steps=4096, checked=checked)
        assert len(checked) == 3
