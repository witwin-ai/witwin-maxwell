"""Boolean geometry export through the Tidy3D adapter (requires the real package)."""

from __future__ import annotations

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")

import witwin.maxwell as mw  # noqa: E402
from witwin.maxwell.adapters.tidy3d import _convert_geometry  # noqa: E402

S = 1.0e6  # metre -> micrometre


def _bounds(geometry):
    return np.asarray(geometry.bounds, dtype=np.float64)


def test_sharp_union_maps_to_geometry_group_of_analytic_children():
    union = mw.Union(mw.Box(size=(2e-6, 2e-6, 2e-6)), mw.Sphere(position=(2e-6, 0, 0), radius=1e-6))
    exported = _convert_geometry(union, td, S)
    assert isinstance(exported, td.GeometryGroup)
    assert [type(child) for child in exported.geometries] == [td.Box, td.Sphere]
    np.testing.assert_allclose(_bounds(exported), [[-1, -1, -1], [3, 1, 1]], atol=1e-9)


def test_difference_and_intersection_map_to_clip_operations():
    base = mw.Box(size=(2e-6, 2e-6, 2e-6))
    tools = (mw.Sphere(radius=0.5e-6), mw.Cylinder(radius=0.2e-6, height=4e-6))
    difference = _convert_geometry(mw.Difference(base, *tools), td, S)
    assert isinstance(difference, td.ClipOperation)
    assert difference.operation == "difference"
    assert isinstance(difference.geometry_a, td.Box)
    assert isinstance(difference.geometry_b, td.GeometryGroup)
    np.testing.assert_allclose(_bounds(difference), [[-1, -1, -1], [1, 1, 1]], atol=1e-9)

    lens = _convert_geometry(
        mw.Intersection(
            mw.Sphere(position=(-0.5e-6, 0, 0), radius=1e-6),
            mw.Sphere(position=(0.5e-6, 0, 0), radius=1e-6),
            mw.Box(size=(4e-6, 4e-6, 1e-6)),
        ),
        td,
        S,
    )
    assert isinstance(lens, td.ClipOperation) and lens.operation == "intersection"
    assert isinstance(lens.geometry_a, td.ClipOperation)
    assert lens.inside(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]
    assert not lens.inside(np.array([0.0]), np.array([0.0]), np.array([0.8]))[0]


def test_node_frame_maps_to_transformed_geometry():
    node = mw.Difference(
        mw.Box(size=(2e-6, 1e-6, 1e-6)),
        mw.Sphere(radius=0.3e-6),
        position=(1e-6, 0, 0),
        rotation=(0.0, 0.0, np.pi / 2),
    )
    exported = _convert_geometry(node, td, S)
    assert isinstance(exported, td.Transformed)
    assert isinstance(exported.geometry, td.ClipOperation)
    np.testing.assert_allclose(_bounds(exported), [[0.5, -1, -0.5], [1.5, 1, 0.5]], atol=1e-6)
    inside = exported.inside(np.array([1.0, 1.4, 1.0]), np.array([0.0, 0.0, 0.8]), np.array([0.0, 0.0, 0.0]))
    assert list(inside) == [False, True, True]


def test_smooth_boolean_falls_back_to_closed_triangle_mesh():
    smooth = mw.Union(
        mw.Sphere(position=(-0.6e-6, 0, 0), radius=0.5e-6),
        mw.Sphere(position=(0.6e-6, 0, 0), radius=0.5e-6),
        smoothness=0.3e-6,
    )
    exported = _convert_geometry(smooth, td, S)
    assert isinstance(exported, td.TriangleMesh)
    lower, upper = _bounds(exported)
    assert lower[0] == pytest.approx(-1.1, abs=0.05) and upper[0] == pytest.approx(1.1, abs=0.05)


def test_scene_with_difference_structure_exports_simulation():
    scene = (
        mw.Scene(
            domain=mw.Domain(bounds=((-2e-6, 2e-6), (-2e-6, 2e-6), (-2e-6, 2e-6))),
            grid=mw.GridSpec.uniform(0.1e-6),
            boundary=mw.BoundarySpec.pml(num_layers=8),
            device="cpu",
        )
        .add_structure(
            mw.Structure(
                mw.Difference(mw.Box(size=(1.5e-6, 1.5e-6, 1.5e-6)), mw.Sphere(radius=0.5e-6)),
                mw.Material(eps_r=4.0),
            )
        )
        .add_source(
            mw.PointDipole(
                position=(1.8e-6, 0, 0),
                polarization="Ez",
                source_time=mw.GaussianPulse(frequency=2e14, fwidth=2e13),
            )
        )
    )
    simulation = scene.to_tidy3d(frequencies=[2e14])
    geometry = simulation.structures[0].geometry
    assert isinstance(geometry, td.ClipOperation) and geometry.operation == "difference"
