"""Self-validation of the independent dipole-sphere series reference (CPU only)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from benchmark.sie_reference import sphere_fields

_spec = importlib.util.spec_from_file_location(
    'dipole_sphere_series', Path(__file__).parent/'reference'/'dipole_sphere_series.py')
series = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(series)

RADIUS, FREQUENCY = 0.1, 477e6
POSITION = np.array([0.05, -0.08, 0.13])
MOMENT = np.array([0.3, 1.0, -0.4+0.2j])
PROBES = np.array([[0.15, 0.02, 0.03], [-0.12, 0.1, -0.05], [0.03, 0.04, 0.05], [0.0, 0.13, 0.11]])


def _reference(n_max, epsilon_r=4+1j, position=POSITION, moment=MOMENT):
    return series.DipoleSphereSeries(radius=RADIUS, frequency=FREQUENCY, epsilon_r=epsilon_r,
                                     position=position, moment=moment, n_max=n_max)


def _relative(a, b):
    return np.abs(a-b).max()/np.abs(b).max()


@pytest.mark.host_validation
def test_series_converges_geometrically_in_number_of_terms():
    # The dipole sits at 1.61 sphere radii, the hardest distance used below.
    fields = [_reference(n).total(PROBES) for n in (10, 20, 30, 40, 50)]
    changes = [_relative(fields[i][0], fields[-1][0]) for i in range(4)]
    assert changes[0] > 1e-4 and changes[1] < 1e-2 and changes[2] < 1e-5 and changes[3] < 1e-8
    assert all(b < a/10 for a, b in zip(changes, changes[1:]))


@pytest.mark.host_validation
def test_regular_expansion_reproduces_the_closed_form_incident_dipole():
    reference = _reference(40)
    points = np.array([[0.02, 0.03, -0.04], [0.05, 0.01, 0.06], [-0.09, 0.02, 0.01]])
    expanded = reference.regular_incident(points)
    exact = reference.incident(points)
    for a, b in zip(expanded, exact):
        assert _relative(a, b) < 1e-7


@pytest.mark.host_validation
def test_series_satisfies_interface_conditions():
    reference = _reference(40)
    rng = np.random.default_rng(3)
    normal = rng.normal(size=(6, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    normal[np.abs(normal[:, 2]) > .99] = (0.6, 0.0, 0.8)
    outside = reference.total(normal*RADIUS*(1+1e-8))
    inside = reference.total(normal*RADIUS/(1+1e-8))

    def tangential(v): return v-normal*(v*normal).sum(-1)[:, None]
    for out_field, in_field in zip(outside, inside):
        assert _relative(tangential(in_field), tangential(out_field)) < 1e-6
    normal_d = (outside[0]*normal).sum(-1)-(4+1j)*(inside[0]*normal).sum(-1)
    assert np.abs(normal_d).max()/np.abs(outside[0]).max() < 1e-6


@pytest.mark.host_validation
def test_vacuum_sphere_scatters_nothing():
    reference = _reference(30, epsilon_r=1.0)
    scattered = reference.scattered(PROBES[[0, 1, 3]])
    incident = reference.incident(PROBES[[0, 1, 3]])
    assert np.abs(scattered[0]).max()/np.abs(incident[0]).max() < 1e-13
    for a, b in zip(reference.total(PROBES[2:3]), reference.incident(PROBES[2:3])):
        assert _relative(a, b) < 1e-8


@pytest.mark.host_validation
def test_distant_dipole_reduces_to_plane_wave_mie_with_first_order_error():
    k, omega = 2*np.pi*FREQUENCY/series.C0, 2*np.pi*FREQUENCY
    outside, inside = PROBES[np.linalg.norm(PROBES, axis=1) > RADIUS], PROBES[np.linalg.norm(PROBES, axis=1) < RADIUS]
    errors = []
    for distance in (1e2, 1e3, 1e4):
        reference = series.DipoleSphereSeries(radius=RADIUS, frequency=FREQUENCY, epsilon_r=4+1j,
            position=(0, 0, -distance), moment=(1, 0, 0), n_max=25)
        # Local plane-wave amplitude of the transverse far field at the sphere centre.
        amplitude = 1j*omega*series.MU0*np.exp(1j*k*distance)/(4*np.pi*distance)
        mie_scattered = sphere_fields(RADIUS, k, 4+1j, outside, field_kind='scattered')
        mie_total = sphere_fields(RADIUS, k, 4+1j, inside, field_kind='total')
        actual = (*reference.scattered(outside), reference.total(inside)[0])
        expected = (*(amplitude*f for f in mie_scattered), amplitude*mie_total[0])
        errors.append(max(_relative(a, b) for a, b in zip(actual, expected)))
    assert errors[-1] < 3e-5
    # Wavefront curvature and near-field corrections are O(radius/distance).
    for a, b in zip(errors, errors[1:]):
        assert 8 < a/b < 12


@pytest.mark.host_validation
def test_series_is_reciprocal():
    r1, r2 = np.array([0.05, -0.08, 0.13]), np.array([-0.14, 0.06, -0.02])
    p1, p2 = np.array([0.3, 1.0, -0.4]), np.array([1.0, -0.2, 0.5])
    forward = _reference(40, position=r1, moment=p1).scattered(r2[None])[0][0]@p2
    backward = _reference(40, position=r2, moment=p2).scattered(r1[None])[0][0]@p1
    assert abs(forward-backward) < 1e-9*abs(forward)
