import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _interface_normals
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize("x", [(0., 1.), (0., .5, 1.), (0., .2, .6, 1.)])
def test_periodic_node_material_gradient_has_one_shared_seam(x):
    from witwin.maxwell.compiler.subpixel import _field_gradients

    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0., 1.),)*3),
        grid=mw.GridSpec.custom(x_coords=x, y_coords=(0., 1.), z_coords=(0., 1.)),
        boundary=mw.BoundarySpec.periodic(), device="cpu"))
    field = torch.arange(len(x), dtype=scene.x.dtype)[:, None, None].expand(-1, 2, 2).clone()
    field[-1] = field[0]
    original = _field_gradients(scene, field)["x"]
    gradients = _field_gradients(scene, field, periodic_nodes=True)
    if len(x) == 2:
        expected = torch.zeros_like(field[0])
    else:
        left, right = scene.x[-1]-scene.x[-2], scene.x[1]-scene.x[0]
        expected = (right*(field[0]-field[-2])/left
                    + left*(field[1]-field[0])/right)/(left+right)
    torch.testing.assert_close(gradients["x"][0], expected)
    torch.testing.assert_close(gradients["x"][0], gradients["x"][-1], rtol=0, atol=0)
    torch.testing.assert_close(gradients["x"][1:-1], original[1:-1], rtol=0, atol=0)
    assert not torch.any(gradients["y"]) and not torch.any(gradients["z"])


def test_interface_normals_match_tilted_plane():
    domain = mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))
    grid = mw.GridSpec.uniform(0.05)
    scene = prepare_scene(mw.Scene(domain=domain, grid=grid, device="cpu"))

    angle = math.radians(30.0)
    geometry = mw.Box(position=(0.0, 0.0, 0.0), size=(1.0, 4.0, 4.0), rotation=(0.0, 0.0, angle))
    normals = _interface_normals(scene, geometry)
    nx, ny, nz = normals["x"], normals["y"], normals["z"]
    magnitude = torch.sqrt(nx * nx + ny * ny + nz * nz)

    # Interface band: nodes near the tilted +/- x face where the SDF gradient is well defined.
    band = magnitude > 0.9
    assert int(band.sum()) > 0
    assert torch.isfinite(magnitude).all()
    assert float(magnitude.max()) <= 1.0 + 1e-4

    # The rotated slab face normal lies in the x-y plane along (cos, sin, 0).
    dot = (nx * math.cos(angle) + ny * math.sin(angle)).abs()
    assert float(dot[band].mean()) > 0.99

    # Degenerate deep-interior / far-field nodes yield a finite ~zero normal.
    degenerate = magnitude < 0.1
    assert int(degenerate.sum()) > 0
    assert torch.isfinite(nx[degenerate]).all()
