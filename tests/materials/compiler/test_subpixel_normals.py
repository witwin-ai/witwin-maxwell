import math

import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _interface_normals
from witwin.maxwell.scene import prepare_scene


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
