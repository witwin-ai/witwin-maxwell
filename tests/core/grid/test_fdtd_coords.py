from __future__ import annotations

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import centered_cell_coords, component_coords
from witwin.maxwell.fdtd.excitation.injection import _ideal_axis_weights
from witwin.maxwell.scene import prepare_scene


def _scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=10),
    )


def test_component_coords_follow_actual_scene_grid_spacing():
    scene = _scene()
    prepared_scene = prepare_scene(scene, device="cpu")

    ex_x, ex_y, ex_z = component_coords(scene, "Ex", device="cpu")
    ey_x, ey_y, ey_z = component_coords(scene, "Ey", device="cpu")
    ez_x, ez_y, ez_z = component_coords(scene, "Ez", device="cpu")

    np.testing.assert_allclose(np.diff(ex_x), prepared_scene.dx, atol=1e-7)
    np.testing.assert_allclose(np.diff(ex_y), prepared_scene.dy, atol=1e-7)
    np.testing.assert_allclose(np.diff(ex_z), prepared_scene.dz, atol=1e-7)
    np.testing.assert_allclose(np.diff(ey_x), prepared_scene.dx, atol=1e-7)
    np.testing.assert_allclose(np.diff(ey_y), prepared_scene.dy, atol=1e-7)
    np.testing.assert_allclose(np.diff(ey_z), prepared_scene.dz, atol=1e-7)
    np.testing.assert_allclose(np.diff(ez_x), prepared_scene.dx, atol=1e-7)
    np.testing.assert_allclose(np.diff(ez_y), prepared_scene.dy, atol=1e-7)
    np.testing.assert_allclose(np.diff(ez_z), prepared_scene.dz, atol=1e-7)

    assert np.isclose(ex_x[0], prepared_scene.x[0].item() + 0.5 * prepared_scene.dx)
    assert np.isclose(ex_x[-1], prepared_scene.x[-2].item() + 0.5 * prepared_scene.dx)
    assert np.isclose(ey_y[0], prepared_scene.y[0].item() + 0.5 * prepared_scene.dy)
    assert np.isclose(ez_z[0], prepared_scene.z[0].item() + 0.5 * prepared_scene.dz)


def test_centered_cell_coords_use_scene_nodes_instead_of_domain_endpoint_linspace():
    scene = _scene()
    prepared_scene = prepare_scene(scene, device="cpu")
    x, y, z = centered_cell_coords(scene, device="cpu")

    np.testing.assert_allclose(np.diff(x), prepared_scene.dx, atol=1e-7)
    np.testing.assert_allclose(np.diff(y), prepared_scene.dy, atol=1e-7)
    np.testing.assert_allclose(np.diff(z), prepared_scene.dz, atol=1e-7)
    assert x.shape == (prepared_scene.Nx - 1,)
    assert y.shape == (prepared_scene.Ny - 1,)
    assert z.shape == (prepared_scene.Nz - 1,)
    assert np.isclose(x[-1], prepared_scene.x[-2].item() + 0.5 * prepared_scene.dx)


def test_ideal_axis_weights_use_trilinear_subcell_distribution():
    coords = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    indices, weights = _ideal_axis_weights(coords, 0.25)
    assert indices == [0, 1]
    np.testing.assert_allclose(weights, [0.75, 0.25])

    indices, weights = _ideal_axis_weights(coords, 1.0)
    assert indices == [1]
    np.testing.assert_allclose(weights, [1.0])

    indices, weights = _ideal_axis_weights(coords, -0.5)
    assert indices == [0]
    np.testing.assert_allclose(weights, [1.0])
