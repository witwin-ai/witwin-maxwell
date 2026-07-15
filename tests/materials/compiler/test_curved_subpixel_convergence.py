import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


_BOUNDS = ((-0.8, 0.8),) * 3
_POSITION = (0.037, -0.023, 0.041)


def _primitive(kind, *, position=_POSITION, trainable=False):
    if kind == "sphere":
        radius = torch.tensor(0.35, requires_grad=True) if trainable else 0.35
        geometry = mw.Sphere(position=position, radius=radius)
        return geometry, (4.0 / 3.0) * math.pi * 0.35**3, (radius,)

    radius = torch.tensor(0.31, requires_grad=True) if trainable else 0.31
    height = torch.tensor(0.7, requires_grad=True) if trainable else 0.7
    geometry = mw.Cylinder(position=position, radius=radius, height=height)
    return geometry, math.pi * 0.31**2 * 0.7, (radius, height)


def _compile_components(geometry, grid, *, samples, averaging="arithmetic", eps_r=5.0):
    scene = mw.Scene(
        domain=mw.Domain(bounds=_BOUNDS),
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging),
    )
    scene.add_structure(
        mw.Structure(
            name="curved",
            geometry=geometry,
            material=mw.Material(eps_r=eps_r),
        )
    )
    prepared = prepare_scene(scene)
    eps, _ = prepared.compile_material_components()
    return prepared, eps


def _occupancy_and_weights(geometry, grid, *, samples):
    prepared, eps = _compile_components(geometry, grid, samples=samples, eps_r=2.0)
    occupancy = (eps["x"] - 1.0).double()
    weights = (
        torch.as_tensor(prepared.dx_dual64)[:, None, None]
        * torch.as_tensor(prepared.dy_dual64)[None, :, None]
        * torch.as_tensor(prepared.dz_dual64)[None, None, :]
    )
    return occupancy, weights


def _relative_volume_error(geometry, exact_volume, grid, *, samples):
    occupancy, weights = _occupancy_and_weights(geometry, grid, samples=samples)
    measured = float((occupancy * weights).sum())
    return abs(measured - exact_volume) / exact_volume


def _mesh_volume(vertices, faces):
    triangles = vertices[faces]
    signed = (
        torch.sum(
            triangles[:, 0] * torch.cross(triangles[:, 1], triangles[:, 2], dim=1)
        )
        / 6.0
    )
    return abs(float(signed))


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_primitive_volume_converges_across_three_yee_grids(kind):
    geometry, exact_volume, _ = _primitive(kind)
    errors = [
        _relative_volume_error(
            geometry, exact_volume, mw.GridSpec.uniform(dx), samples=3
        )
        for dx in (0.2, 0.1, 0.05)
    ]

    assert errors[0] > errors[1] > errors[2]
    observed_orders = [math.log2(errors[index] / errors[index + 1]) for index in (0, 1)]
    assert min(observed_orders) > 0.5
    assert errors[-1] < 5e-4


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_primitive_subcell_translation_and_sample_parity_are_stable(kind):
    _, exact_volume, _ = _primitive(kind)
    errors = {}
    for samples in (1, 2, 3, 4):
        errors[samples] = []
        for fraction in (0.0, 0.23, 0.47):
            geometry, _, _ = _primitive(
                kind,
                position=(fraction * 0.1, 0.017, -0.023),
            )
            errors[samples].append(
                _relative_volume_error(
                    geometry,
                    exact_volume,
                    mw.GridSpec.uniform(0.1),
                    samples=samples,
                )
            )

    multisample_errors = [error for samples in (2, 3, 4) for error in errors[samples]]
    assert max(multisample_errors) < max(errors[1])
    assert np.ptp(errors[4]) < np.ptp(errors[1])
    assert max(multisample_errors) < (0.012 if kind == "sphere" else 0.008)


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_primitive_volume_is_accurate_on_nonuniform_grid(kind):
    nodes = np.linspace(-0.8, 0.8, 17)
    half_wave = np.sin(np.linspace(0.0, math.pi, nodes.size))
    full_wave = np.sin(np.linspace(0.0, 2.0 * math.pi, nodes.size))
    x_nodes = nodes + 0.018 * half_wave
    y_nodes = nodes - 0.012 * half_wave
    z_nodes = nodes + 0.009 * full_wave
    for axis_nodes in (x_nodes, y_nodes, z_nodes):
        axis_nodes[0], axis_nodes[-1] = -0.8, 0.8

    geometry, exact_volume, _ = _primitive(kind)
    error = _relative_volume_error(
        geometry,
        exact_volume,
        mw.GridSpec.custom(x_nodes, y_nodes, z_nodes),
        samples=3,
    )
    assert error < 0.007


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_polarized_material_uses_radial_normal_component(kind):
    geometry = (
        mw.Sphere(radius=0.25)
        if kind == "sphere"
        else mw.Cylinder(radius=0.25, height=0.5)
    )
    grid = mw.GridSpec.uniform(0.25)
    prepared, arithmetic = _compile_components(
        geometry,
        grid,
        samples=3,
        averaging="arithmetic",
    )
    _, polarized = _compile_components(
        geometry,
        grid,
        samples=3,
        averaging="polarized",
    )
    node = (
        int(np.argmin(np.abs(prepared.x_nodes64 - 0.25))),
        int(np.argmin(np.abs(prepared.y_nodes64))),
        int(np.argmin(np.abs(prepared.z_nodes64))),
    )
    occupancy = (float(arithmetic["x"][node]) - 1.0) / 4.0
    harmonic = 1.0 / ((1.0 - occupancy) / 1.0 + occupancy / 5.0)

    assert float(polarized["x"][node]) == pytest.approx(harmonic, abs=5e-5)
    assert float(polarized["y"][node]) == pytest.approx(
        float(arithmetic["y"][node]), abs=5e-5
    )
    assert float(polarized["z"][node]) == pytest.approx(
        float(arithmetic["z"][node]), abs=5e-5
    )


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_polarized_material_has_finite_nonzero_geometry_gradients(kind):
    geometry, _, parameters = _primitive(kind, trainable=True)
    _, eps = _compile_components(
        geometry,
        mw.GridSpec.uniform(0.1),
        samples=3,
        averaging="polarized",
    )
    gradients = torch.autograd.grad(
        sum(component.sum() for component in eps.values()), parameters
    )

    for gradient in gradients:
        assert torch.isfinite(gradient)
        assert float(gradient.abs()) > 1e-6


def test_sphere_mesh_resolution_converges_to_analytic_occupancy():
    analytic, _, _ = _primitive("sphere", position=(0.0, 0.0, 0.0))
    analytic_occupancy, _ = _occupancy_and_weights(
        analytic,
        mw.GridSpec.uniform(0.2),
        samples=2,
    )
    shape_errors = []
    for segments in (4, 8, 12):
        vertices, faces = analytic.to_mesh(segments)
        mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
        mesh_occupancy, _ = _occupancy_and_weights(
            mesh,
            mw.GridSpec.uniform(0.2),
            samples=2,
        )
        shape_errors.append(
            float(
                torch.linalg.vector_norm(mesh_occupancy - analytic_occupancy)
                / torch.linalg.vector_norm(analytic_occupancy)
            )
        )

    assert shape_errors[0] > shape_errors[1] > shape_errors[2]
    assert shape_errors[-1] < 0.04


def test_fixed_sphere_mesh_volume_converges_across_three_yee_grids():
    sphere = mw.Sphere(radius=0.35, device="cpu")
    vertices, faces = sphere.to_mesh(12)
    mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
    exact_mesh_volume = _mesh_volume(vertices, faces)
    errors = [
        _relative_volume_error(
            mesh, exact_mesh_volume, mw.GridSpec.uniform(dx), samples=2
        )
        for dx in (0.4, 0.2, 0.1)
    ]

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_sphere_mesh_occupancy_is_orientation_invariant():
    sphere = mw.Sphere(radius=0.35, device="cpu")
    vertices, faces = sphere.to_mesh(8)
    forward = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
    reversed_mesh = mw.Mesh(
        vertices, faces[:, [0, 2, 1]], recenter=False, fill_mode="solid"
    )
    forward_occupancy, _ = _occupancy_and_weights(
        forward,
        mw.GridSpec.uniform(0.2),
        samples=2,
    )
    reversed_occupancy, _ = _occupancy_and_weights(
        reversed_mesh,
        mw.GridSpec.uniform(0.2),
        samples=2,
    )

    assert torch.allclose(forward_occupancy, reversed_occupancy, atol=1e-6, rtol=1e-6)
