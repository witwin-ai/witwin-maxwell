"""Rotated analytic primitives must export with their rotation (point-in-solid vs witwin's SDF)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

td = pytest.importorskip("tidy3d")

import witwin.maxwell as mw  # noqa: E402
from witwin.maxwell.adapters.tidy3d import _convert_geometry  # noqa: E402

S = 1.0


def _zyx(z, y, x):
    """Unit quaternion for intrinsic z-y-x Euler angles in degrees."""
    def axis(angle, index):
        half = math.radians(angle) / 2
        q = [math.cos(half), 0.0, 0.0, 0.0]
        q[1 + index] = math.sin(half)
        return q

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]

    return tuple(mul(mul(axis(z, 2), axis(y, 1)), axis(x, 0)))


ROTATIONS = {
    "z_0p47rad": (math.cos(0.235), 0.0, 0.0, math.sin(0.235)),
    "zyx_30_20_10": _zyx(30, 20, 10),
}
GEOMETRIES = {
    "box": lambda rotation: mw.Box(position=(0.1, -0.05, 0.2), size=(0.6, 0.3, 0.45), rotation=rotation),
    "cylinder_z": lambda rotation: mw.Cylinder(position=(0.1, -0.05, 0.2), radius=0.2, height=0.7,
                                               axis="z", rotation=rotation),
    "cylinder_x": lambda rotation: mw.Cylinder(position=(-0.1, 0.05, 0.0), radius=0.15, height=0.5,
                                               axis="x", rotation=rotation),
}


@pytest.mark.parametrize("rotation_name", sorted(ROTATIONS))
@pytest.mark.parametrize("geometry_name", sorted(GEOMETRIES))
def test_rotated_primitive_matches_witwin_inside_on_a_point_grid(geometry_name, rotation_name):
    geometry = GEOMETRIES[geometry_name](ROTATIONS[rotation_name])
    exported = _convert_geometry(geometry, td, S)
    axis = np.linspace(-0.6, 0.6, 41)
    x, y, z = (value.ravel() for value in np.meshgrid(axis, axis, axis, indexing="ij"))
    distance = geometry.signed_distance(*(torch.as_tensor(v, dtype=torch.float64) for v in (x, y, z)))
    distance = distance.detach().cpu().numpy().ravel()
    keep = np.abs(distance) > 1e-6  # exact surface ties are undefined in both conventions
    expected = distance[keep] < 0
    actual = np.asarray(exported.inside(x[keep], y[keep], z[keep]))
    assert expected.sum() > 100 and (~expected).sum() > 100
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("geometry_name", sorted(GEOMETRIES))
def test_unrotated_primitive_export_is_unchanged(geometry_name):
    geometry = GEOMETRIES[geometry_name](None)
    exported = _convert_geometry(geometry, td, S)
    assert type(exported).__name__ in {"Box", "Cylinder"}
