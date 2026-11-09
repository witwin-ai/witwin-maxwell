"""Exterior electric PointDipole incident fields, carrier identity and scope (host checks)."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.carrier import CARRIER_VERSION, POINT_DIPOLE_CARRIER_VERSION, IncidentCarrier
from witwin.maxwell.mom.dipole import (DIPOLE_CLEARANCE_PANEL_RADII, point_dipole_fields,
                                       validate_point_dipole_clearance)
from witwin.maxwell.mom.physics import compile_medium, source_phasor, validate_scene_structure

_spec = importlib.util.spec_from_file_location(
    'dipole_sphere_series', Path(__file__).parent/'reference'/'dipole_sphere_series.py')
series = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(series)

FREQUENCY = 3e8
POSITION = (0.11, -0.04, 0.27)
MOMENT = (0.3-0.2j, 1.0, -0.4+0.7j)


def exterior():
    return compile_medium(mw.Material(), FREQUENCY, background=True)


def dipole(**kwargs):
    options = dict(position=POSITION, polarization=(0, 0, 1), profile='ideal',
                   source_time=mw.CW(frequency=FREQUENCY, amplitude=2.0, phase=.3))
    options.update(kwargs)
    return mw.PointDipole(**options)


def _points():
    rng = np.random.default_rng(7)
    # Distances from 0.002 to 2 wavelengths cover the near, intermediate and far zones.
    radius = np.geomspace(2e-3, 2.0, 12)
    direction = rng.normal(size=(12, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return torch.tensor(np.asarray(POSITION)+radius[:, None]*direction, dtype=torch.float64)


@pytest.mark.host_validation
def test_closed_form_matches_vector_potential_derivatives():
    """Independent of the closed form: E = i omega mu (g Il + grad(grad g . Il)/k^2), H = grad g x Il."""
    medium = exterior()
    points = _points()
    e, h = point_dipole_fields(points, POSITION, MOMENT, medium)
    k, origin = medium.k.real, torch.tensor(POSITION, dtype=torch.float64)
    current = torch.tensor(MOMENT, dtype=torch.complex128)

    def green_parts(r):
        distance = torch.linalg.vector_norm(r-origin)
        return torch.stack((torch.cos(k*distance), torch.sin(k*distance)))/(4*math.pi*distance)

    for point, actual_e, actual_h in zip(points, e, h):
        value = green_parts(point)
        gradient = torch.autograd.functional.jacobian(green_parts, point)
        hessian = torch.stack([torch.autograd.functional.hessian(lambda r, i=i: green_parts(r)[i], point)
                               for i in range(2)])
        g = torch.complex(value[0], value[1])
        grad = torch.complex(gradient[0], gradient[1])
        hess = torch.complex(hessian[0], hessian[1])
        expected_e = 1j*medium.omega*medium.mu*(g*current+hess@current/k**2)
        expected_h = torch.linalg.cross(grad, current)
        torch.testing.assert_close(actual_e, expected_e, rtol=1e-9, atol=0)
        torch.testing.assert_close(actual_h, expected_h, rtol=1e-9, atol=0)


@pytest.mark.host_validation
def test_closed_form_satisfies_maxwell_equations_and_matches_numpy_reference():
    medium = exterior()
    points = _points()[3:]
    e, h = point_dipole_fields(points, POSITION, MOMENT, medium)
    e_ref, h_ref = series.dipole_fields(points.numpy(), POSITION, MOMENT, medium.k.real, medium.omega, medium.mu.real)
    torch.testing.assert_close(e, torch.from_numpy(e_ref), rtol=1e-12, atol=0)
    torch.testing.assert_close(h, torch.from_numpy(h_ref), rtol=1e-12, atol=0)

    def curl(component, point):
        parts = [lambda r, i=i: component(r)[i] for i in range(3)]
        jac = torch.stack([torch.autograd.functional.jacobian(lambda r, f=f: torch.view_as_real(f(r)), point)
                           for f in parts])  # [field component, re/im, d/dx]
        jac = torch.complex(jac[:, 0], jac[:, 1])
        return torch.stack((jac[2, 1]-jac[1, 2], jac[0, 2]-jac[2, 0], jac[1, 0]-jac[0, 1]))

    for point, actual_e, actual_h in zip(points, e, h):
        curl_e = curl(lambda r: point_dipole_fields(r[None], POSITION, MOMENT, medium)[0][0], point)
        curl_h = curl(lambda r: point_dipole_fields(r[None], POSITION, MOMENT, medium)[1][0], point)
        torch.testing.assert_close(curl_e, 1j*medium.omega*medium.mu*actual_h, rtol=1e-9, atol=0)
        torch.testing.assert_close(curl_h, -1j*medium.omega*medium.epsilon*actual_e, rtol=1e-9, atol=0)


@pytest.mark.host_validation
def test_far_zone_power_matches_hertzian_radiation_formula():
    medium = exterior()
    radius = 400.0
    x, w = np.polynomial.legendre.leggauss(64)
    phi = 2*np.pi*np.arange(64)/64
    theta, phi = np.meshgrid(np.arccos(x), phi, indexing='ij')
    normal = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)), -1).reshape(-1, 3)
    weights = (w[:, None]*np.full(64, 2*np.pi/64)).ravel()*radius**2
    points = torch.tensor(np.asarray(POSITION)+radius*normal, dtype=torch.float64)
    e, h = point_dipole_fields(points, POSITION, MOMENT, medium)
    flux = 0.5*(torch.linalg.cross(e, h.conj()).real*torch.from_numpy(normal)).sum(-1)
    power = float((flux*torch.from_numpy(weights)).sum())
    moment_norm = sum(abs(c)**2 for c in MOMENT)
    expected = medium.eta.real*medium.k.real**2*moment_norm/(12*math.pi)
    assert abs(power-expected) < 1e-6*expected


@pytest.mark.host_validation
def test_field_is_rejected_at_the_dipole_position():
    with pytest.raises(ValueError, match='singular'):
        point_dipole_fields(torch.tensor([POSITION], dtype=torch.float64), POSITION, MOMENT, exterior())


@pytest.mark.host_validation
def test_plane_wave_carrier_identity_is_unchanged():
    """Identities pinned from the parent commit before point dipoles existed."""
    medium = exterior()
    sources = (mw.PlaneWave(source_time=mw.CW(frequency=FREQUENCY, phase=.37)),
               mw.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1),
                            source_time=mw.CW(frequency=FREQUENCY, phase=-.51, amplitude=.31)))
    carrier = IncidentCarrier.compile(sources, medium)
    assert carrier.identity == '6650e3aa8644d960e6fc4520ef5a023032a51d37d97905514646e9ed22eecb26'
    assert IncidentCarrier.compile((mw.PlaneWave(),), medium).identity == \
        '850463d58380f46ff2889a26e5b68a1b31a9d4c3ba59ad8327ab0b9e91b992d9'
    payload = carrier.payload()
    assert payload['version'] == CARRIER_VERSION and 'dipoles' not in payload and carrier.dipoles == ()


@pytest.mark.host_validation
def test_dipole_carrier_records_current_moment_and_roundtrips():
    medium = exterior()
    sources = (dipole(), mw.PlaneWave(source_time=mw.CW(frequency=FREQUENCY)),
               dipole(position=(-0.3, 0.0, 0.0), polarization=(1, 1, 0)))
    carrier = IncidentCarrier.compile(sources, medium)
    payload = json.loads(json.dumps(carrier.payload()))
    assert payload['version'] == POINT_DIPOLE_CARRIER_VERSION and payload['dipole_moment_units'] == 'A*m'
    moment = complex(2.0*math.cos(.3), -2.0*math.sin(.3))
    assert payload['dipoles'][0] == {'position': list(POSITION), 'moment': [[0., 0.], [0., 0.], [moment.real, moment.imag]]}
    assert len(carrier.waves) == 1 and len(carrier.dipoles) == 2
    restored = IncidentCarrier.restore(payload, expected_identity=carrier.identity, sources=sources, exterior=medium)
    assert restored == carrier
    moved = copy.deepcopy(payload)
    moved['dipoles'][0]['position'][2] += 1e-9
    with pytest.raises(ValueError, match='does not match'):
        IncidentCarrier.restore(moved, expected_identity=carrier.identity, sources=sources, exterior=medium)
    other = (dipole(position=(0.11, -0.04, 0.28)),) + sources[1:]
    assert IncidentCarrier.compile(other, medium).identity != carrier.identity


@pytest.mark.host_validation
def test_dipole_source_phasor_and_rejections():
    assert source_phasor(dipole(), FREQUENCY) == pytest.approx(2.0*complex(math.cos(.3), -math.sin(.3)))
    assert source_phasor(dipole(source_time=None), FREQUENCY) == 1
    with pytest.raises(ValueError, match="profile='ideal'"):
        source_phasor(dipole(profile='gaussian'), FREQUENCY)
    with pytest.raises(ValueError, match='CW or None'):
        source_phasor(dipole(source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=1e8)), FREQUENCY)
    with pytest.raises(ValueError, match='frequencies must match'):
        source_phasor(dipole(source_time=mw.CW(frequency=2*FREQUENCY)), FREQUENCY)
    magnetic = mw.CustomCurrentSource(mw.CurrentDataset(
        ([0.0, 1e-3], [0.0, 1e-3], [0.0, 1e-3]), {'Mz': np.ones((2, 2, 2))}))
    for source in (magnetic, mw.UniformCurrentSource(size=(0, 0, 0))):
        with pytest.raises(ValueError, match='magnetic dipoles'):
            source_phasor(source, FREQUENCY)


@pytest.mark.host_validation
def test_dipole_scope_is_explicit_for_every_scene_path():
    structure = mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2))
    scene = mw.Scene(device='cpu', structures=[structure], sources=[dipole()])
    assert validate_scene_structure(scene, FREQUENCY, point_dipoles=True)
    with pytest.raises(ValueError, match='linear dense free-space scalar runtime'):
        validate_scene_structure(scene, FREQUENCY)


def _cube(half=0.1, center=(0., 0., 0.), divisions=1):
    """Outward closed cube triangulation with divisions x divisions squares per face."""
    triangles = []
    grid = np.linspace(-half, half, divisions+1)
    for axis in range(3):
        for sign in (-1, 1):
            u, v = [(axis+1) % 3, (axis+2) % 3][::sign]
            for i in range(divisions):
                for j in range(divisions):
                    corners = []
                    for a, b in ((i, j), (i+1, j), (i+1, j+1), (i, j+1)):
                        point = np.zeros(3)
                        point[axis], point[u], point[v] = sign*half, grid[a], grid[b]
                        corners.append(point+center)
                    triangles += [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]]
    return torch.tensor(np.array(triangles), dtype=torch.float64)


@pytest.mark.host_validation
def test_clearance_rule_rejects_inside_on_and_near_surface_dipoles():
    coarse = _cube()
    radius = float(torch.linalg.vector_norm(coarse-coarse.mean(1)[:, None], dim=-1).amax())
    assert validate_point_dipole_clearance((0., 0., 0.1+3.01*radius), coarse) >= DIPOLE_CLEARANCE_PANEL_RADII
    with pytest.raises(ValueError, match='panel radii'):
        validate_point_dipole_clearance((0., 0., 0.1+2.9*radius), coarse, body_name='cube')
    with pytest.raises(ValueError, match='on the surface'):
        validate_point_dipole_clearance((0.02, 0.03, 0.1), coarse)
    for position in ((0., 0., 0.), (0., 0., 0.09)):
        with pytest.raises(ValueError, match='inside'):
            validate_point_dipole_clearance(position, coarse)
    # Clearance is local: refining the panels near the source admits a closer dipole.
    fine = _cube(divisions=8)
    fine_radius = float(torch.linalg.vector_norm(fine-fine.mean(1)[:, None], dim=-1).amax())
    position = (0.0, 0.0, 0.1+3.5*fine_radius)
    with pytest.raises(ValueError, match='panel radii'):
        validate_point_dipole_clearance(position, coarse)
    assert validate_point_dipole_clearance(position, fine) >= DIPOLE_CLEARANCE_PANEL_RADII


@pytest.mark.host_validation
def test_clearance_rule_matches_measured_quadrature_error():
    """At the admitted limit, a fixed order-4 panel rule integrates the dipole trace to 2e-4."""
    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes, weights = (nodes+1)/2, weights/2
    a, b = np.meshgrid(nodes, nodes, indexing='ij')
    rule_w = (weights[:, None]*weights[None]*a).ravel()
    a, b = a.ravel(), b.ravel()
    rho = 0.01
    triangle = np.array([[rho*math.cos(t), rho*math.sin(t), 0.] for t in (0, 2*math.pi/3, 4*math.pi/3)])
    medium = exterior()

    def integrate(tri, source, moment, order_points=None):
        v0, v1, v2 = tri
        if order_points is None:
            p = v0+a[:, None]*((1-b[:, None])*(v1-v0)+b[:, None]*(v2-v0))
            w = rule_w*np.linalg.norm(np.cross(v1-v0, v2-v0))
        else:
            p, w = order_points
        e, _ = point_dipole_fields(torch.tensor(p), source, moment, medium)
        weight = 1+(p[:, 0]-tri[0, 0])/rho
        return (torch.tensor(w*weight)[:, None]*e).sum(0)

    def fine_rule(tri, depth=5):
        triangles = [tri]
        for _ in range(depth):
            refined = []
            for t in triangles:
                m01, m12, m20 = (t[0]+t[1])/2, (t[1]+t[2])/2, (t[2]+t[0])/2
                refined += [np.array(x) for x in ([t[0], m01, m20], [m01, t[1], m12], [m20, m12, t[2]], [m01, m12, m20])]
            triangles = refined
        n12, w12 = np.polynomial.legendre.leggauss(12)
        n12, w12 = (n12+1)/2, w12/2
        aa, bb = np.meshgrid(n12, n12, indexing='ij')
        ww = (w12[:, None]*w12[None]*aa).ravel()
        aa, bb = aa.ravel(), bb.ravel()
        points, weights_out = [], []
        for v0, v1, v2 in triangles:
            points.append(v0+aa[:, None]*((1-bb[:, None])*(v1-v0)+bb[:, None]*(v2-v0)))
            weights_out.append(ww*np.linalg.norm(np.cross(v1-v0, v2-v0)))
        return np.concatenate(points), np.concatenate(weights_out)

    reference_rule = fine_rule(triangle)
    for base in (np.zeros(3), triangle[0], (triangle[0]+triangle[1])/2):
        source = tuple(base+np.array([0., 0., DIPOLE_CLEARANCE_PANEL_RADII*rho]))
        for moment in ((0, 0, 1), (1, 0, 0)):
            coarse = integrate(triangle, source, moment)
            exact = integrate(triangle, source, moment, reference_rule)
            assert float(torch.linalg.vector_norm(coarse-exact)/torch.linalg.vector_norm(exact)) < 2e-4
