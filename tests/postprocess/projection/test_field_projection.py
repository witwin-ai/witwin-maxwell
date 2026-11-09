"""Angle / Cartesian / k-space field projections of analytic Hertzian-dipole currents."""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.postprocess import (
    EquivalentCurrentsSurface,
    FieldProjector,
    ProjectedFieldData,
    StrattonChuPropagator,
    build_plane_points,
    equivalent_surface_currents_from_fields,
    equivalent_surface_currents_from_monitor,
)

_ETA0 = math.sqrt(MU_0 / EPSILON_0)
_FREQUENCY = 1.0e9
_HALF = 0.06


def _medium(frequency, eps_r=1.0):
    omega = 2.0 * math.pi * frequency
    n = math.sqrt(eps_r)
    return omega, n * omega / C_0, _ETA0 / n


def _dipole_fields(points, *, k, eta, omega):
    """Exact fields of a unit z-directed electric dipole moment (e^{-i w t})."""
    p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.complex128)
    eps = k / (eta * omega)
    c = omega / k
    r = torch.linalg.norm(points, dim=-1)
    n = (points / r[..., None]).to(torch.complex128)
    n_dot_p = n[..., 2]
    exp = torch.exp(1j * k * r)
    p_perp = p - n * n_dot_p[..., None]
    p_near = 3.0 * n * n_dot_p[..., None] - p
    e_field = (1.0 / (4.0 * math.pi * eps)) * (
        (k**2) * p_perp * (exp / r)[..., None]
        + p_near * ((1.0 / r**3 - 1j * k / r**2) * exp)[..., None]
    )
    n_cross_p = torch.cross(n, p.expand_as(n), dim=-1)
    h_field = (c * k**2 / (4.0 * math.pi)) * n_cross_p * (exp / r * (1.0 - 1.0 / (1j * k * r)))[..., None]
    return e_field, h_field


def _dipole_far_fields(directions, radius, *, k, eta, omega):
    """Leading 1/r term of the same dipole along unit ``directions`` at ``radius``."""
    p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.complex128)
    eps = k / (eta * omega)
    c = omega / k
    n = directions.to(torch.complex128)
    radius = torch.as_tensor(radius, dtype=torch.float64)
    phase = torch.exp(1j * k * radius) / radius
    p_perp = p - n * n[..., 2:3]
    e_field = (k**2 / (4.0 * math.pi * eps)) * p_perp * phase[..., None]
    h_field = (c * k**2 / (4.0 * math.pi)) * torch.cross(n, p.expand_as(n), dim=-1) * phase[..., None]
    return e_field, h_field


def _box_currents(frequency=_FREQUENCY, *, eps_r=1.0, half=_HALF, samples=41, scale=None):
    omega, k, eta = _medium(frequency, eps_r)
    coord = torch.linspace(-half, half, samples, dtype=torch.float64)
    faces = []
    for axis, position, normal in (
        ("x", -half, "-"), ("x", half, "+"),
        ("y", -half, "-"), ("y", half, "+"),
        ("z", -half, "-"), ("z", half, "+"),
    ):
        points = build_plane_points(axis, position, coord, coord).reshape(-1, 3)
        e_field, h_field = _dipole_fields(points, k=k, eta=eta, omega=omega)
        if scale is not None:
            e_field = e_field * scale
            h_field = h_field * scale
        components = {}
        for index, label in enumerate("xyz"):
            components[f"E{label}"] = e_field[:, index].reshape(samples, samples)
            components[f"H{label}"] = h_field[:, index].reshape(samples, samples)
        faces.append(
            equivalent_surface_currents_from_fields(
                axis=axis,
                position=position,
                frequency=frequency,
                u=coord,
                v=coord,
                fields=components,
                normal_direction=normal,
                background_eps_r=eps_r,
            )
        )
    return EquivalentCurrentsSurface(tuple(faces))


def _rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


@pytest.fixture(scope="module")
def currents():
    return _box_currents()


@pytest.fixture(scope="module")
def projector(currents):
    return FieldProjector(currents, device="cpu")


def test_angle_projection_matches_analytic_far_field(projector):
    omega, k, eta = _medium(_FREQUENCY)
    theta = torch.linspace(0.05, math.pi - 0.05, 37, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 25, dtype=torch.float64)
    data = projector.angles(theta, phi, radius=10.0)

    assert isinstance(data, ProjectedFieldData)
    assert data.kind == "angle" and data.far_field_approx
    assert data.E.shape == (1, 37, 25, 3)
    assert data.E_theta.shape == (1, 37, 25)
    assert data.frequencies.tolist() == [_FREQUENCY]
    assert bool(torch.all(data.mask))

    e_ref, h_ref = _dipole_far_fields(data.directions, 10.0, k=k, eta=eta, omega=omega)
    assert _rel(data.E[0], e_ref) < 2e-3
    assert _rel(data.H[0], h_ref) < 2e-3
    # Transverse far field: no radial components, E_phi is cross-polar zero.
    assert float(torch.max(torch.abs(data.E_r))) < 1e-9 * float(torch.max(torch.abs(data.E_theta)))
    assert float(torch.max(torch.abs(data.E_phi))) < 1e-2 * float(torch.max(torch.abs(data.E_theta)))
    # Radiation intensity follows U = eta k^4 c^2 |p|^2 sin^2(theta) / (32 pi^2).
    expected_u = eta * (k**4) * (omega / k) ** 2 * torch.sin(data.theta) ** 2 / (32.0 * math.pi**2)
    assert _rel(data.radiation_intensity[0], expected_u) < 2e-2
    torch.testing.assert_close(data.E_intensity, torch.sum(torch.abs(data.E) ** 2, dim=-1))
    # Far-field radiation intensity does not depend on the projection radius.
    farther = projector.angles(theta, phi, radius=1.0e4)
    torch.testing.assert_close(farther.radiation_intensity, data.radiation_intensity, rtol=1e-9, atol=0.0)


def test_far_field_phase_is_referenced_to_projection_origin(currents):
    omega, k, eta = _medium(_FREQUENCY)
    origin = (0.03, -0.02, 0.01)
    shifted = FieldProjector(currents, origin=origin, device="cpu")
    theta = torch.linspace(0.3, math.pi - 0.3, 9, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float64)
    radius = 2.0e3
    data = shifted.angles(theta, phi, radius=radius)
    torch.testing.assert_close(data.points, torch.tensor(origin, dtype=torch.float64) + radius * data.directions)
    e_exact, _ = _dipole_fields(data.points.reshape(-1, 3), k=k, eta=eta, omega=omega)
    assert _rel(data.E[0].reshape(-1, 3), e_exact) < 1e-2


def test_exact_angle_projection_matches_dipole_near_field(projector):
    omega, k, eta = _medium(_FREQUENCY)
    theta = torch.linspace(0.2, math.pi - 0.2, 11, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 9, dtype=torch.float64)
    radius = 0.25
    exact = projector.angles(theta, phi, radius=radius, far_field_approx=False)
    assert not exact.far_field_approx
    e_ref, h_ref = _dipole_fields(exact.points.reshape(-1, 3), k=k, eta=eta, omega=omega)
    assert _rel(exact.E[0].reshape(-1, 3), e_ref) < 2e-3
    assert _rel(exact.H[0].reshape(-1, 3), h_ref) < 2e-3
    # At a quarter wavelength the near-field radial component is significant and
    # the transverse far-field approximation is visibly wrong.
    assert float(torch.max(torch.abs(exact.E_r))) > 0.1 * float(torch.max(torch.abs(exact.E_theta)))
    approx = projector.angles(theta, phi, radius=radius)
    assert _rel(approx.E[0], exact.E[0]) > 0.1


def test_exact_cartesian_projection_matches_direct_stratton_chu(currents, projector):
    omega, k, eta = _medium(_FREQUENCY)
    x = torch.linspace(-0.15, 0.15, 7, dtype=torch.float64)
    y = torch.linspace(-0.1, 0.1, 5, dtype=torch.float64)
    data = projector.cartesian(x, y, axis="y", distance=0.2)
    assert data.kind == "cartesian" and not data.far_field_approx
    assert data.E.shape == (1, 7, 5, 3)
    # y-normal plane: local (x, y) are world (x, z).
    torch.testing.assert_close(data.points[..., 0], data.coords["x"])
    torch.testing.assert_close(data.points[..., 2], data.coords["y"])
    assert bool(torch.all(data.points[..., 1] == 0.2))

    direct = StrattonChuPropagator(currents, c=C_0, eps0=EPSILON_0, mu0=MU_0, device="cpu")
    reference = direct.propagate_points(data.points.reshape(-1, 3))
    torch.testing.assert_close(data.E[0].reshape(-1, 3), reference["E"], rtol=1e-12, atol=0.0)
    torch.testing.assert_close(data.H[0].reshape(-1, 3), reference["H"], rtol=1e-12, atol=0.0)
    e_ref, _ = _dipole_fields(data.points.reshape(-1, 3), k=k, eta=eta, omega=omega)
    assert _rel(data.E[0].reshape(-1, 3), e_ref) < 2e-2


def test_exact_cartesian_projection_converges_to_far_field_approximation(projector):
    x = torch.linspace(-0.5, 0.5, 5, dtype=torch.float64)
    errors = []
    for distance in (2.0, 20.0, 200.0):
        coords = x * distance
        exact = projector.cartesian(coords, coords, axis="x", distance=-distance)
        approx = projector.cartesian(coords, coords, axis="x", distance=-distance, far_field_approx=True)
        assert approx.far_field_approx
        errors.append(_rel(approx.E[0], exact.E[0]))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-3


def test_kspace_projection_masks_evanescent_directions_and_matches_analytic(projector):
    omega, k, eta = _medium(_FREQUENCY)
    ux = torch.linspace(-1.2, 1.2, 25, dtype=torch.float64)
    uy = torch.linspace(-1.2, 1.2, 21, dtype=torch.float64)
    data = projector.kspace(ux, uy, axis="z", radius=5.0)
    assert data.kind == "kspace" and data.far_field_approx
    ux_grid, uy_grid = torch.meshgrid(ux, uy, indexing="ij")
    expected_mask = ux_grid**2 + uy_grid**2 <= 1.0
    assert torch.equal(data.mask, expected_mask)
    assert 0 < int(expected_mask.sum()) < expected_mask.numel()
    assert bool(torch.all(data.E[0][~expected_mask] == 0))
    assert bool(torch.all(data.H[0][~expected_mask] == 0))
    assert bool(torch.all(data.power_density[0][~expected_mask] == 0))

    valid = data.directions[expected_mask]
    torch.testing.assert_close(valid[:, 0], ux_grid[expected_mask])
    torch.testing.assert_close(valid[:, 1], uy_grid[expected_mask])
    assert bool(torch.all(valid[:, 2] >= 0.0))
    e_ref, h_ref = _dipole_far_fields(valid, 5.0, k=k, eta=eta, omega=omega)
    assert _rel(data.E[0][expected_mask], e_ref) < 2e-3
    assert _rel(data.H[0][expected_mask], h_ref) < 2e-3

    # The k-space samples are the same far field as an angle projection along
    # those directions.
    # Column vectors broadcast pairwise instead of forming a theta x phi mesh.
    angles = projector.angles(data.theta[expected_mask][:, None], data.phi[expected_mask][:, None], radius=5.0)
    scale = float(torch.max(torch.abs(angles.E)))
    torch.testing.assert_close(data.E[0][expected_mask], angles.E[0, :, 0], rtol=1e-9, atol=1e-12 * scale)


def test_kspace_projection_negative_hemisphere_along_x(projector):
    ux = torch.tensor([0.0, 0.3, 0.6], dtype=torch.float64)
    uy = torch.tensor([0.0, -0.4], dtype=torch.float64)
    data = projector.kspace(ux, uy, axis="x", normal_direction="-", radius=5.0)
    # x-normal: local (ux, uy) are world (y, z) direction cosines.
    torch.testing.assert_close(data.directions[..., 1], data.coords["ux"])
    torch.testing.assert_close(data.directions[..., 2], data.coords["uy"])
    assert bool(torch.all(data.directions[..., 0] < 0.0))
    torch.testing.assert_close(torch.linalg.norm(data.directions, dim=-1), torch.ones(3, 2, dtype=torch.float64))


def test_projection_keeps_explicit_frequency_axis():
    frequencies = (0.8e9, 1.2e9)
    per_frequency = [_box_currents(frequency) for frequency in frequencies]
    projector = FieldProjector(per_frequency, device="cpu")
    theta = torch.linspace(0.2, math.pi - 0.2, 5, dtype=torch.float64)
    phi = torch.tensor([0.0, 1.0], dtype=torch.float64)
    data = projector.angles(theta, phi, radius=3.0)
    assert data.E.shape == (2, 5, 2, 3)
    torch.testing.assert_close(data.frequencies, torch.tensor(frequencies, dtype=torch.float64))
    for index, frequency in enumerate(frequencies):
        omega, k, eta = _medium(frequency)
        assert abs(complex(data.wavenumbers[index]) - k) < 1e-9 * k
        single = FieldProjector(per_frequency[index], device="cpu").angles(theta, phi, radius=3.0)
        torch.testing.assert_close(data.E[index], single.E[0])
        e_ref, _ = _dipole_far_fields(data.directions, 3.0, k=k, eta=eta, omega=omega)
        assert _rel(data.E[index], e_ref) < 1e-2


def test_projection_uses_homogeneous_background_medium():
    eps_r = 2.25
    omega, k, eta = _medium(_FREQUENCY, eps_r)
    projector = FieldProjector(_box_currents(eps_r=eps_r, half=0.04), device="cpu")
    assert abs(complex(projector.wavenumbers[0]) - k) < 1e-9 * k
    assert abs(complex(projector.impedances[0]) - eta) < 1e-9 * eta
    theta = torch.linspace(0.2, math.pi - 0.2, 9, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float64)
    far = projector.angles(theta, phi, radius=8.0)
    e_ref, _ = _dipole_far_fields(far.directions, 8.0, k=k, eta=eta, omega=omega)
    assert _rel(far.E[0], e_ref) < 2e-2
    exact = projector.cartesian(
        torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64),
        torch.tensor([0.0, 0.05], dtype=torch.float64),
        axis="z",
        distance=0.15,
    )
    e_near, _ = _dipole_fields(exact.points.reshape(-1, 3), k=k, eta=eta, omega=omega)
    assert _rel(exact.E[0].reshape(-1, 3), e_near) < 3e-2


def test_projection_gradients_flow_through_currents():
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    projector = FieldProjector(_box_currents(samples=21, scale=scale), device="cpu")
    theta = torch.linspace(0.2, math.pi - 0.2, 5, dtype=torch.float64)
    phi = torch.tensor([0.0, 2.0], dtype=torch.float64)
    ux = torch.linspace(-0.9, 0.9, 4, dtype=torch.float64)

    angle_power = projector.angles(theta, phi, radius=4.0).radiation_intensity.sum()
    kspace_power = projector.kspace(ux, ux, radius=4.0).radiation_intensity.sum()
    plane = projector.cartesian(ux * 0.1, ux * 0.1, distance=0.12).E_intensity.sum()
    for objective in (angle_power, kspace_power, plane):
        (grad,) = torch.autograd.grad(objective, scale, retain_graph=True)
        # All objectives are quadratic in the current amplitude.
        torch.testing.assert_close(grad, 2.0 * objective.detach() / scale.detach(), rtol=1e-9, atol=0.0)


def test_projection_validates_inputs(projector):
    theta = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="radius must be positive"):
        projector.angles(theta, theta, radius=0.0)
    with pytest.raises(ValueError, match="radius must be positive"):
        projector.kspace(theta, theta, radius=-1.0)
    with pytest.raises(ValueError, match="origin"):
        projector.cartesian(torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), distance=0.0)
    with pytest.raises(ValueError, match="normal_direction"):
        projector.kspace(theta, theta, normal_direction="up")


def _synthetic_two_frequency_result() -> mw.Result:
    frequencies = (1.0e9, 1.5e9)
    surface = mw.ClosedSurfaceMonitor.box(
        "huygens", position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4), frequencies=frequencies
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    ).add_monitor(surface)
    coords = np.linspace(-0.3, 0.3, 7)
    monitors = {}
    for face in surface.faces:
        names = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[face.axis]
        payload = {
            "kind": "plane",
            "fields": face.fields,
            "components": {},
            "samples": 8,
            "frequencies": frequencies,
            "axis": face.axis,
            "position": face.plane_position,
            "compute_flux": False,
            "normal_direction": face.normal_direction,
        }
        for component in face.fields:
            data = np.stack(
                [np.full((coords.size, coords.size), scale, dtype=np.complex64) for scale in (1.0, 2.0j)]
            )
            payload["components"][component] = {
                "data": data,
                "coords": (coords, coords),
                "plane_index": 0,
                "plane_indices": (0,),
                "plane_weights": (1.0,),
                "plane_positions": (face.plane_position,),
            }
            payload[component] = data
        payload[names[0]] = coords
        payload[names[1]] = coords
        payload["coords"] = (coords, coords)
        monitors[face.name] = payload
    return mw.Result(method="fdtd", scene=scene, frequencies=frequencies, monitors=monitors)


def test_projector_from_monitor_stacks_all_result_frequencies():
    result = _synthetic_two_frequency_result()
    projector = FieldProjector.from_monitor(result, "huygens", device="cpu")
    theta = torch.linspace(0.3, 2.8, 4, dtype=torch.float64)
    phi = torch.tensor([0.0, 1.5], dtype=torch.float64)
    data = projector.angles(theta, phi, radius=10.0)
    assert data.E.shape == (2, 4, 2, 3)
    assert data.frequencies.tolist() == [1.0e9, 1.5e9]
    for index, frequency in enumerate((1.0e9, 1.5e9)):
        currents = equivalent_surface_currents_from_monitor(result, "huygens", frequency=frequency)
        single = FieldProjector(currents, device="cpu").angles(theta, phi, radius=10.0)
        torch.testing.assert_close(data.E[index], single.E[0])
    assert float(torch.max(torch.abs(data.E[1]))) > 0.0
