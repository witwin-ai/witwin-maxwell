"""Raw source timing against the closed finite Yee recurrence."""
import math
import struct
from types import SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.fdtd.excitation.temporal import InjectionKind, injection_time
from witwin.maxwell.fdtd.runtime.spectral import source_normalization_spectrum
from witwin.maxwell.fdtd.runtime.timeplan import compile_fdtd_time_plan


def test_injection_time_is_bitwise_identical_to_inline_expressions():
    for kind in get_args(InjectionKind):
        offset = 0.5 if kind in (
            "electric_current", "equivalent_electric_current", "port_drive"
        ) else 0.0
        for n in (0, 1, 3, 17, 2**20):
            for dt in (0.1, 9.5238e-12, 1e-20, math.nextafter(1e-12, math.inf)):
                expected = n * dt + offset * float(dt)
                actual = injection_time(kind, n * dt, dt)
                assert type(actual) is float
                assert struct.pack("d", actual) == struct.pack("d", expected)
                for dtype, bits in ((torch.float32, torch.int32), (torch.float64, torch.int64)):
                    origin = torch.tensor(n, dtype=dtype) * dt
                    expected_tensor = origin + offset * float(dt)
                    actual_tensor = injection_time(kind, origin, dt)
                    assert actual_tensor.dtype == dtype
                    assert torch.equal(actual_tensor.view(bits), expected_tensor.view(bits))


def _closed_record(matrix, forcing, omega, dt, count, alpha):
    """Finite zero-state solution, including startup and cosine leakage."""
    r = np.exp(-1j * omega * dt)
    response = np.linalg.solve(r * np.eye(2) - matrix, forcing)
    eigenvalues, eigenvectors = np.linalg.eig(matrix.astype(complex))
    inverse = np.linalg.inv(eigenvectors)
    return np.array([
        ((r ** (n + 1) * np.eye(2)
          - eigenvectors @ np.diag(eigenvalues ** (n + 1)) @ inverse)
         @ response * np.exp(-1j * alpha * omega * dt)).real
        for n in range(count)
    ])


def _record(kind, count=64, *, normalize=False):
    dx, length, frequency, dt = .05, .4, 2e9, 9.5238e-12
    if kind == "J":
        coords = np.arange(9, dtype=float) * dx
        profile = np.sin(math.pi * coords / length)
        component, alpha = "Jx", .5
    else:
        coords = (np.arange(10, dtype=float) - .5) * dx
        profile = np.cos(math.pi * coords / length)
        component, alpha = "My", 0.
    # Compensate the half-covered nodal y endpoint control cells so the
    # declared density compiles to a current uniform across x and y.
    values = np.broadcast_to(profile[None, None, :], (2, 3, len(profile))).copy()
    values *= np.array([2., 1., 2.])[None, :, None]
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0., .1), (0., .1), (0., length))),
        grid=mw.GridSpec.uniform(dx), device="cpu",
        boundary=mw.BoundarySpec.pec().with_faces(y_low="pmc", y_high="pmc"),
        sources=(mw.CustomCurrentSource(
            mw.CurrentDataset(([-.05, .15], [0., .05, .1], coords), {component: values}),
            source_time=mw.CW(frequency)),),
        monitors=(mw.PointMonitor("probe", (.025, .05, .1), fields=("Ex", "Hy")),),
    )
    prepared = mw.Simulation.fdtd(
        scene, frequencies=(frequency,), backend="cpu_reference", cuda_graph=False,
        field_dtype=torch.float32,
        run_time=mw.TimeConfig(time_steps=count, dt=dt, transient_cycles=0),
        full_field_dft=normalize,
        spectral_sampler=mw.SpectralSampler(
            window="none", normalize_source=normalize,
            observer_dtype=torch.float32 if normalize else torch.float64),
    ).prepare()
    result = prepared.run()
    solver = prepared.solver
    raw = result.raw_monitor("probe")
    eps, mu = float(solver.eps_Ex[0, 1, 2]), float(solver.mu_Hy[0, 1, 1])
    eta = math.sqrt(mu / eps)
    kappa = 2 * math.sin(math.pi * dx / (2 * length)) / dx
    a = kappa * solver.dt / math.sqrt(eps * mu)
    matrix = np.array([[1 - a * a, a], [-a, 1.]])
    forcing = (np.array([-solver.dt / eps, 0.]) if kind == "J" else
               np.array([-a * solver.dt * eta / mu, -solver.dt * eta / mu]))
    omega = 2 * math.pi * frequency
    record = _closed_record(matrix, forcing, omega, solver.dt, count, alpha)
    integer_record = _closed_record(matrix, forcing, omega, solver.dt, count, 0.)
    kernel = np.exp(1j * omega * (np.arange(count) + 1) * solver.dt)
    source_spectrum = 2 / count * np.sum(np.cos(omega * (np.arange(count) + alpha) * solver.dt) *
        np.exp(1j * omega * (np.arange(count) + alpha) * solver.dt))
    h_offset = np.exp(-.5j * omega * solver.dt)
    measurements = {}
    for name, index in raw["field_indices"].items():
        if name == "Ex":
            z = float(solver.scene.z_nodes64[index[2]])
            shape, channel, stagger = math.sin(math.pi * z / length), 0, 1.
        else:
            z = float(solver.scene.z_half64[index[2]])
            shape, channel, stagger = math.cos(math.pi * z / length) / eta, 1, h_offset
        scale = 2 / count * shape * stagger
        if normalize:
            scale /= source_spectrum
        measurements[name] = dict(
            observed=complex(raw["components"][name]),
            expected=scale * np.sum(record[:, channel] * kernel),
            integer_expected=scale * np.sum(integer_record[:, channel] * kernel),
            native=float(getattr(solver, name)[tuple(index)]),
            origin=stagger * np.exp(1j * omega * solver.dt),
        )
    return measurements


@pytest.fixture(scope="module")
def records():
    return {kind: _record(kind) for kind in ("J", "M")}


def test_additive_j_at_half_step(records):
    for measurement in records["J"].values():
        error = abs(measurement["observed"] - measurement["expected"]) / abs(measurement["expected"])
        integer_error = abs(measurement["observed"] - measurement["integer_expected"]) / abs(measurement["integer_expected"])
        assert error < 3e-6, (error, integer_error)


def test_m_at_integer_step(records):
    for measurement in records["M"].values():
        assert abs(measurement["observed"] - measurement["expected"]) / abs(measurement["expected"]) < 3e-6


def test_observer_origin_independent_of_source():
    for measurement in _record("M", 1).values():
        ratio = measurement["observed"] / (2 * measurement["native"])
        assert abs(ratio - measurement["origin"]) < 2e-14


@pytest.mark.parametrize("kind", ["J", "M"])
def test_normalized_recurrence_uses_the_same_source_instant(kind):
    for measurement in _record(kind, normalize=True).values():
        assert abs(measurement["observed"] - measurement["expected"]) / abs(measurement["expected"]) < 3e-6


@pytest.mark.parametrize("kind,alpha,waveform", [
    (kind, alpha, waveform)
    for kind, alpha in [
        ("dipole", .5), ("uniform", .5), ("J", .5), ("M", 0.),
        # Auxiliary entry E and its unit reference now share post-step time.
        ("custom_field", 0.), ("tfsf", 1.), ("mode", 0.), ("port", .5),
    ]
    for waveform in ("cw", "pulse", "callable", "analytic_cw")
    if waveform != "callable" or kind not in ("custom_field", "mode")
])
def test_source_spectrum_matches_injection_instant(kind, alpha, waveform):
    """Independent finite sums keep the waveform and observer clocks separate."""
    frequency, dt, count = 2e9, 9.5238e-12, 64
    if waveform == "pulse":
        source_time = mw.GaussianPulse(frequency, .5e9)
        sigma = 1 / (2 * math.pi * .5e9)
        envelope = lambda t: np.exp(-.5 * ((t - 6 * sigma) / sigma) ** 2) * np.cos(2 * math.pi * frequency * (t - 6 * sigma))
    elif waveform == "callable":
        envelope = lambda t: np.cos(2 * math.pi * frequency * t) * np.exp(-frequency * t)
        source_time = mw.CustomSourceTime(envelope, characteristic_frequency=frequency)
    else:
        source_time = mw.CW(frequency)
        envelope = (lambda t: np.exp(-2j * math.pi * frequency * t)) if waveform == "analytic_cw" else (lambda t: np.cos(2 * math.pi * frequency * t))
    coords = ([-.05, .05],) * 3
    sources = {
        "dipole": lambda: mw.PointDipole((0., 0., 0.), source_time=source_time),
        "uniform": lambda: mw.UniformCurrentSource((.1,) * 3, source_time=source_time),
        "J": lambda: mw.CustomCurrentSource(mw.CurrentDataset(coords, {"Jx": np.ones((2, 2, 2))}), source_time=source_time),
        "M": lambda: mw.CustomCurrentSource(mw.CurrentDataset(coords, {"My": np.ones((2, 2, 2))}), source_time=source_time),
        "custom_field": lambda: mw.CustomFieldSource(mw.FieldDataset((coords[0], coords[1], [0.]), {"Ex": np.ones((2, 2, 1))}), source_time=source_time),
        "tfsf": lambda: mw.PlaneWave(source_time=source_time, injection=mw.TFSF(bounds=((-0.05, .05),) * 3)),
        "mode": lambda: mw.ModeSource(position=(0., 0., 0.), source_time=source_time),
    }
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, .1),) * 3), device="cpu",
                     sources=() if kind == "port" else (sources[kind](),))
    solver = SimpleNamespace(
        dt=dt, c=299792458., device="cpu", _normalize_source=True,
        complex_fields_enabled=waveform == "analytic_cw", _spectral_reference_frequency=None,
        _compiled_sources=compile_fdtd_sources(scene, default_frequency=frequency),
    )
    # These scalar tests supply the dispatch descriptors of their prepared providers.
    providers = {"dipole": "additive_J", "uniform": "additive_J", "J": "additive_J",
                 "M": "magnetic_M", "custom_field": "custom_field_surface", "tfsf": "plane_wave_axis_aligned",
                 "mode": "mode_surface"}
    for source in solver._compiled_sources:
        source.update(reference_provider_kind=providers[kind], reference_waveform_kind=source["source_time"]["kind"])
        if kind == "tfsf" and waveform != "cw" and waveform != "analytic_cw":
            source.update(reference_incident_kind="physical_upstream_yee_entry_plane", reference_incident_projection=-.05)
    compile_fdtd_time_plan(solver, scene, mw.TimeConfig(time_steps=count), (frequency,),
        default_source_frequency=frequency, window="none", normalize_source=True,
        excitations=(mw.PortExcitation("feed", source_time=source_time),) if kind == "port" else ())
    actual = source_normalization_spectrum(solver, (frequency,), steps_run=count, window_type="none")
    n = np.arange(count)
    samples = envelope(n * dt + alpha * dt)
    scale = (1 if waveform == "analytic_cw" else 2) / count if waveform in ("cw", "analytic_cw") else dt * frequency
    beta = 1.0 if kind == "tfsf" else alpha
    expected = scale * np.sum(samples * np.exp(2j * math.pi * frequency * (n + beta) * dt))
    np.testing.assert_allclose(actual.numpy(), [expected], rtol=2e-13, atol=1e-20)
