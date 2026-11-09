"""Opt-in corner / layer-bounds / gap refinement for GridSpec.auto."""

from __future__ import annotations

import math

import numpy as np
import pytest

import witwin.maxwell as mw
from witwin.maxwell.fdtd.meshing import corners as cg
from witwin.maxwell.fdtd.meshing import mesh_axis, resolve_auto_grid

_RATIO = 1.4
_SLACK = 1.0 + 1e-9
# Structure faces come from float32 mesh vertices; corner snaps merge onto them.
_NODE_TOL = 1e-8


def _corners(poly, angle=math.pi / 4, distance=None):
    loop = cg.orient_loops([np.asarray(poly, dtype=np.float64)])[0]
    return cg.find_corners(loop, angle, distance)


def _as_set(points, digits=9):
    return {tuple(np.round(p, digits)) for p in points}


L_SHAPE = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
U_SHAPE = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]


@pytest.mark.parametrize("reverse", [False, True])
def test_l_shape_has_five_convex_and_one_concave_corner(reverse):
    poly = L_SHAPE[::-1] if reverse else L_SHAPE
    points, convex = _corners(poly)
    assert _as_set(points) == _as_set(L_SHAPE)
    concave = _as_set(points[~convex])
    assert concave == {(1.0, 1.0)}
    assert int(convex.sum()) == 5


def test_u_shape_has_two_concave_corners():
    points, convex = _corners(U_SHAPE)
    assert len(points) == 8
    assert _as_set(points[~convex]) == {(2.0, 1.0), (1.0, 1.0)}


def test_rotated_square_corners_are_all_convex():
    c = math.cos(math.pi / 4)
    rotation = np.array([[c, -c], [c, c]])
    square = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) - 0.5) @ rotation.T
    points, convex = _corners(square)
    assert _as_set(points) == _as_set(square)
    assert convex.all()


def test_collinear_and_duplicate_vertices_are_not_corners():
    poly = [(0, 0), (0.5, 0), (0.5, 0), (1, 0), (1, 0.25), (1, 0.5), (1, 1), (0, 1), (0, 0.5)]
    points, convex = _corners(poly)
    assert _as_set(points) == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    assert convex.all()


def test_angle_threshold_on_regular_octagon():
    angles = np.arange(8) * math.pi / 4
    octagon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert len(_corners(octagon, angle=math.pi / 4)[0]) == 8
    assert len(_corners(octagon, angle=math.pi / 4 + 0.01)[0]) == 0
    assert len(_corners(octagon, angle=0.0)[0]) == 8


def test_distance_threshold_merges_a_small_chamfer():
    chamfered = [(0, 0), (1, 0), (1, 0.99), (0.99, 1), (0, 1)]
    points, _ = _corners(chamfered)
    assert len(points) == 5
    merged, convex = _corners(chamfered, distance=0.05)
    assert len(merged) == 4 and convex.all()
    assert np.min(np.linalg.norm(merged - np.array([1.0, 1.0]), axis=1)) < 0.02


def test_hole_corners_are_concave_for_the_material():
    outer = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    hole = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    loops = cg.orient_loops([np.asarray(outer, float), np.asarray(hole, float)])
    _, outer_convex = cg.find_corners(loops[0], math.pi / 4)
    _, hole_convex = cg.find_corners(loops[1], math.pi / 4)
    assert outer_convex.all() and not hole_convex.any()


def test_cross_sections_of_supported_geometries():
    slab = mw.PolySlab(vertices=L_SHAPE, bounds=(-0.5, 0.5), axis="y", position=(10.0, 3.0, -4.0))
    loops = cg.cross_section_loops(slab, 1, 3.2, 1e-9)
    assert len(loops) == 1
    # axis y -> in-plane (x, z); the position shifts the drawn polygon.
    assert _as_set(loops[0]) == _as_set(np.asarray(L_SHAPE, float) + [10.0, -4.0])
    assert cg.cross_section_loops(slab, 1, 4.0, 1e-9) == []

    pec = mw.PlanarPEC(L_SHAPE, holes=[[(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]], coordinate=0.5)
    assert len(cg.cross_section_loops(pec, 2, 0.5, 1e-9)) == 2
    assert cg.cross_section_loops(pec, 2, 0.6, 1e-9) == []

    rotated = mw.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), rotation=(0.0, 0.0, math.pi / 4))
    (loop,) = cg.cross_section_loops(rotated, 2, 0.1, 1e-9)
    points, convex = cg.find_corners(cg.orient_loops([loop])[0], math.pi / 4)
    assert len(points) == 4 and convex.all()
    assert np.allclose(np.sort(np.abs(points).max(axis=1)), math.sqrt(0.5), atol=1e-6)


def _scene(structures, grid, bounds=((-0.03, 0.05), (-0.03, 0.05), (-0.02, 0.02))):
    return mw.Scene(domain=mw.Domain(bounds=bounds), grid=grid, device="cpu", structures=structures)


def _ratio(nodes):
    d = np.diff(nodes)
    return float(max((d[1:] / d[:-1]).max(), (d[:-1] / d[1:]).max()))


def _l_trace():
    return mw.Structure(
        mw.PolySlab(vertices=np.asarray(L_SHAPE, float) * 0.01, bounds=(0.0, 0.001), axis="z"),
        mw.Material.pec(),
    )


def _corner_layer(**kwargs):
    defaults = dict(
        min_cells=None, axis="z", bounds=(0.0, 0.001),
        corner_finder=mw.CornerFinderSpec(),
        corner_refinement=mw.GridRefinement(dl=0.001, num_cells=4),
    )
    defaults.update(kwargs)
    return mw.LayerRefinementSpec(**defaults)


def test_corner_refinement_snaps_and_refines_with_bounded_grading():
    plain = resolve_auto_grid(_scene([_l_trace()], mw.GridSpec.auto(wavelength=0.1)))
    refined = resolve_auto_grid(
        _scene([_l_trace()], mw.GridSpec.auto(wavelength=0.1, layer_refinement=_corner_layer()))
    )
    for axis in (0, 1):
        nodes = refined[axis]
        assert _ratio(nodes) <= _RATIO * _SLACK
        for corner in (0.0, 0.01, 0.02):
            assert np.min(np.abs(nodes - corner)) < _NODE_TOL
            near = (nodes[:-1] >= corner - 0.002 - _NODE_TOL) & (nodes[1:] <= corner + 0.002 + _NODE_TOL)
            assert np.count_nonzero(near) >= 4
            assert np.diff(nodes)[near].max() <= 0.001 * _SLACK
        assert np.diff(nodes).max() > 0.004  # far field stays coarse
        assert nodes.size > plain[axis].size
    # No refinement along the layer normal: z is untouched.
    assert np.array_equal(refined[2], plain[2])


def test_layer_spec_without_features_leaves_the_grid_unchanged():
    base = mw.GridSpec.auto(wavelength=0.1, layer_refinement=mw.LayerRefinementSpec(min_cells=2))
    located = mw.GridSpec.auto(
        wavelength=0.1, layer_refinement=mw.LayerRefinementSpec(min_cells=2, axis="z", bounds=(0.0, 0.001))
    )
    for a, b in zip(resolve_auto_grid(_scene([_l_trace()], base)), resolve_auto_grid(_scene([_l_trace()], located))):
        assert np.array_equal(a, b)


def test_corner_refinement_coexists_with_mesh_override_and_explicit_nodes():
    override = mw.MeshOverrideStructure(geometry=mw.Box(position=(0.03, 0.03, 0.0), size=(0.01, 0.01, 0.01)), dl=0.0005)
    grid = mw.GridSpec.auto(wavelength=0.1, override_structures=(override,), layer_refinement=_corner_layer())
    scene = _scene([_l_trace()], grid)
    x, y, z = resolve_auto_grid(scene)
    for nodes in (x, y):
        inside = (nodes[:-1] >= 0.025 - 1e-12) & (nodes[1:] <= 0.035 + 1e-12)
        assert np.diff(nodes)[inside].max() <= 0.0005 * _SLACK
        assert np.min(np.abs(nodes - 0.02)) < _NODE_TOL
        assert _ratio(nodes) <= _RATIO * _SLACK
    explicit = np.union1d(x, [0.0153, -0.0042])
    prepared = mw.scene.prepare_scene(scene.clone(grid=mw.GridSpec.custom(explicit, y, z)), lazy_meshgrid=True)
    assert np.all(np.diff(prepared.x_nodes64) > 0)
    assert np.isin([0.0153, -0.0042], prepared.x_nodes64).all()
    assert all(np.min(np.abs(prepared.x_nodes64 - c)) < _NODE_TOL for c in (0.0, 0.01, 0.02))


def test_refinement_inside_sim_only_drops_outside_corners():
    # Domain cuts the L at x = 0.015: corners at x = 0.02 lie outside.
    bounds = ((-0.03, 0.015), (-0.03, 0.05), (-0.02, 0.02))
    inside = resolve_auto_grid(_scene([_l_trace()], mw.GridSpec.auto(wavelength=0.1, layer_refinement=_corner_layer()), bounds))
    outside = resolve_auto_grid(
        _scene([_l_trace()], mw.GridSpec.auto(wavelength=0.1, layer_refinement=_corner_layer(refinement_inside_sim_only=False)), bounds)
    )
    # The corner (0.02, 0.01) only refines y when outside corners are honored.
    assert np.min(np.abs(inside[1] - 0.01)) < _NODE_TOL  # (0.01, 0.01) is inside either way
    near = lambda nodes: np.count_nonzero(np.abs(nodes - 0.0) < 0.002)  # noqa: E731
    assert near(outside[1]) >= near(inside[1])
    assert outside[0][-1] == 0.015 and _ratio(outside[0]) <= _RATIO * _SLACK


def test_bounds_refinement_min_steps_and_snapping_along_layer_axis():
    slab = mw.Structure(mw.Box(position=(0.0, 0.0, 0.0002), size=(0.2, 0.2, 0.0004)), mw.Material(eps_r=4.0))
    layer = mw.LayerRefinementSpec(
        min_cells=None, axis="z", bounds=(0.0, 0.0004), min_steps_along_axis=4,
        bounds_refinement=mw.GridRefinement(dl=0.00005, num_cells=4), bounds_snapping="center",
    )
    z = resolve_auto_grid(_scene([slab], mw.GridSpec.auto(wavelength=0.1, layer_refinement=layer)))[2]
    assert np.min(np.abs(z - 0.0002)) < _NODE_TOL
    inside = (z[:-1] >= -1e-12) & (z[1:] <= 0.0004 + 1e-12)
    assert np.count_nonzero(inside) >= 8
    for plane in (0.0, 0.0004):
        near = (z[:-1] >= plane - 0.0001 - 1e-12) & (z[1:] <= plane + 0.0001 + 1e-12)
        assert np.diff(z)[near].max() <= 0.00005 * _SLACK and np.count_nonzero(near) >= 4
    assert _ratio(z) <= _RATIO * _SLACK
    # Bounds refinement coarser than min_steps_along_axis is dropped.
    coarse = mw.LayerRefinementSpec(
        min_cells=None, axis="z", bounds=(0.0, 0.0004), min_steps_along_axis=4,
        bounds_refinement=mw.GridRefinement(dl=0.0002),
    )
    steps_only = mw.LayerRefinementSpec(min_cells=None, axis="z", bounds=(0.0, 0.0004), min_steps_along_axis=4)
    za = resolve_auto_grid(_scene([slab], mw.GridSpec.auto(wavelength=0.1, layer_refinement=coarse)))[2]
    zb = resolve_auto_grid(_scene([slab], mw.GridSpec.auto(wavelength=0.1, layer_refinement=steps_only)))[2]
    assert np.array_equal(za, zb)


def test_gap_width_refinement_resolves_narrow_metal_gaps():
    strips = [
        mw.Structure(mw.PlanarPEC.rectangle(((-0.04, -0.0003), (-0.005, 0.005)), coordinate=0.0), mw.Material.pec()),
        mw.Structure(mw.PlanarPEC.rectangle(((0.0003, 0.04), (-0.005, 0.005)), coordinate=0.0), mw.Material.pec()),
    ]
    bounds = ((-0.03, 0.03), (-0.03, 0.03), (-0.02, 0.02))
    layer = mw.LayerRefinementSpec(min_cells=None, axis="z", bounds=(0.0, 0.0), dl_min_from_gap_width=True)
    x = resolve_auto_grid(_scene(strips, mw.GridSpec.auto(wavelength=0.1, layer_refinement=layer), bounds))[0]
    gap = (x[:-1] >= -0.0003 - 1e-12) & (x[1:] <= 0.0003 + 1e-12)
    assert np.count_nonzero(gap) >= 2
    assert np.min(np.abs(x + 0.0003)) < _NODE_TOL and np.min(np.abs(x - 0.0003)) < _NODE_TOL
    assert _ratio(x) <= _RATIO * _SLACK
    assert cg.scanline_gaps([[np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)]], 0, 1.0, (0, 1)) == []


def test_convex_resolution_resolves_strip_width():
    strip = mw.Structure(mw.PlanarPEC.rectangle(((-0.04, 0.04), (-0.0015, 0.0015)), coordinate=0.0), mw.Material.pec())
    bounds = ((-0.03, 0.03), (-0.03, 0.03), (-0.02, 0.02))
    layer = mw.LayerRefinementSpec(
        min_cells=None, axis="z", bounds=(0.0, 0.0),
        corner_finder=mw.CornerFinderSpec(convex_resolution=6),
    )
    y = resolve_auto_grid(_scene([strip], mw.GridSpec.auto(wavelength=0.1, layer_refinement=layer), bounds))[1]
    across = (y[:-1] >= -0.0015 - 1e-12) & (y[1:] <= 0.0015 + 1e-12)
    assert np.count_nonzero(across) >= 6
    assert _ratio(y) <= _RATIO * _SLACK


def test_multiple_layer_specs_and_validation():
    specs = (mw.LayerRefinementSpec(min_cells=2, axes=("x",)), mw.LayerRefinementSpec(min_cells=4, axes=("x", "y")), _corner_layer())
    grid = mw.GridSpec.auto(wavelength=0.1, layer_refinement=list(specs))
    assert grid.layer_refinement == specs
    resolve_auto_grid(_scene([_l_trace()], grid))
    with pytest.raises(TypeError, match="LayerRefinementSpec"):
        mw.GridSpec.auto(layer_refinement=[mw.LayerRefinementSpec(), object()])
    with pytest.raises(TypeError, match="LayerRefinementSpec"):
        mw.GridSpec.auto(layer_refinement=[])
    with pytest.raises(ValueError, match="axis= and bounds="):
        mw.LayerRefinementSpec(corner_finder=mw.CornerFinderSpec())
    with pytest.raises(ValueError, match="together"):
        mw.LayerRefinementSpec(axis="z")
    with pytest.raises(ValueError, match="requires corner_finder"):
        mw.LayerRefinementSpec(axis="z", bounds=(0, 1), corner_refinement=mw.GridRefinement())
    with pytest.raises(ValueError, match="lower <= upper"):
        mw.LayerRefinementSpec(axis="z", bounds=(1, 0))
    with pytest.raises(ValueError, match="angle_threshold"):
        mw.CornerFinderSpec(angle_threshold=math.pi)
    with pytest.raises(ValueError, match="medium"):
        mw.CornerFinderSpec(medium="copper")
    with pytest.raises(ValueError, match="num_cells"):
        mw.GridRefinement(num_cells=0)
    with pytest.raises(ValueError, match="dl"):
        mw.GridRefinement(dl=-1.0)
    assert mw.LayerRefinementSpec(axis=2, bounds=(0, 1)).axis == "z"
    assert not mw.LayerRefinementSpec(min_cells=None).covers("x")


def test_grid_refinement_step_semantics():
    assert mw.GridRefinement().grid_size(1.0) == 0.5
    assert mw.GridRefinement(refinement_factor=4).grid_size(1.0) == 0.25
    assert mw.GridRefinement(dl=0.1).grid_size(1.0) == 0.1
    assert mw.GridRefinement(refinement_factor=4, dl=0.3).grid_size(1.0) == 0.25


def test_layer_spec_from_structures():
    spec = mw.LayerRefinementSpec.from_structures([_l_trace()], axis="z", corner_finder=mw.CornerFinderSpec())
    assert spec.axis == "z" and spec.bounds == pytest.approx((0.0, 0.001), abs=1e-9)


def test_soft_window_edges_never_create_slivers():
    # A window edge 1% of a step from a structure face snaps onto that face.
    nodes = mesh_axis(
        0.0, 1.0, wavelength=1.0, min_steps_per_wavelength=10, max_ratio=_RATIO,
        index_regions=[(0.4, 0.6, 1.0)],
        refine_regions=[(0.6005, 0.7, 0.01), (0.2, 0.3995, 0.01)],
        snap_points=[(0.6002, 0.002), (0.75, 0.002)],
    )
    d = np.diff(nodes)
    assert d.min() >= 0.01 / _RATIO
    assert _ratio(nodes) <= _RATIO * _SLACK
    assert np.min(np.abs(nodes - 0.75)) < _NODE_TOL and np.min(np.abs(nodes - 0.6002)) > 1e-4
    window = (nodes[:-1] >= 0.6 - 1e-12) & (nodes[1:] <= 0.7 + 1e-12)
    assert d[window].max() <= 0.01 * _SLACK
