"""Explicit sheet geometry: preservation, topology, ownership and placement."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.planar_pec import compile_planar_pec
from witwin.maxwell.fdtd.runtime.materials import _pec_edge_open_fractions
from witwin.maxwell.scene import prepare_scene


def _scene(structures, *, spacing=.6e-3, inserts=(), pec="staircase"):
    nodes = np.unique(np.r_[np.arange(-4, 5, dtype=np.float64) * spacing, inserts])
    return prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((nodes[0], nodes[-1]),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.none(), device="cpu", structures=structures,
        subpixel_samples=mw.SubpixelSpec(pec=pec, averaging="arithmetic", samples=1)))


def _structure(bounds=((-1.2e-3, 1.2e-3),) * 2, **kwargs):
    return mw.Structure(mw.PlanarPEC.rectangle(bounds, **kwargs), mw.Material.pec())


def test_rotated_planar_sheet_rejects_without_axis_aligned_projection():
    sheet = _structure()
    assert compile_planar_pec(_scene([sheet]))['audit']['retention_fraction'] == 1
    sheet.geometry.rotation = torch.tensor([np.cos(np.pi/8), np.sin(np.pi/8), 0., 0.])
    with pytest.raises(NotImplementedError, match='rotated sheets need a cut-surface'):
        compile_planar_pec(_scene([sheet]))


@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('side', (-1, 1))
@pytest.mark.parametrize('boundary', (mw.BoundarySpec.periodic(), mw.BoundarySpec.bloch((.2, .3, .4))))
def test_planar_sheet_rejects_unowned_periodic_normal_seam(axis, side, boundary):
    interior = prepare_scene(_scene([_structure(normal_axis=axis)]).clone(boundary=boundary))
    assert compile_planar_pec(interior)['audit']['retention_fraction'] == 1
    endpoint = float(getattr(interior, axis + '_nodes64')[0 if side < 0 else -1])
    seam = prepare_scene(interior.clone(structures=[_structure(normal_axis=axis, coordinate=endpoint)]))
    with pytest.raises(NotImplementedError, match='periodic normal seam requires duplicate-plane ownership'):
        compile_planar_pec(seam)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("spacing", [.1e-3, .3e-3, .6e-3, 1e-3])
@pytest.mark.parametrize("thickness", [0, 18e-6, 35e-6, 70e-6])
@pytest.mark.parametrize("offset", [0, .1, .25, .5, .75])
def test_sheet_matrix_retains_tangential_edges_after_explicit_plane_insertion(axis, spacing, thickness, offset):
    coordinate = spacing * offset
    scene = _scene([_structure(bounds=((-3 * spacing, 3 * spacing),) * 2,
                              normal_axis=axis, coordinate=coordinate,
                              physical_thickness=thickness)], spacing=spacing, inserts=[coordinate])
    layout = compile_planar_pec(scene)
    assert layout["audit"]["retention_fraction"] == 1
    assert layout["audit"]["sheets"][0]["physical_thickness_m"] == thickness
    for component, mask in layout["edges"].items():
        assert not bool(((mask > 0) & (mask < 1)).any())
        assert bool(mask.any()) == (component[-1] != axis)
        if component[-1] != axis:
            reduced = mask.sum(dim=tuple(d for d in range(3) if d != "xyz".index(axis)))
            assert int((reduced > 0).sum()) == 1


@pytest.mark.parametrize("mode", ["staircase", "conformal"])
def test_native_sheet_has_no_normal_short_or_fractional_loss(mode):
    model = _scene([_structure()], pec=mode).compile_materials()
    opened = _pec_edge_open_fractions(SimpleNamespace(_compiled_material_model=model))
    assert torch.all(opened["Ez"] == 1)
    for component in ("Ex", "Ey"):
        assert bool((opened[component] == 0).any())
        assert bool(((opened[component] == 0) | (opened[component] == 1)).all())
    assert model["pec_volume_occupancy"] is None


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_offgrid_sheet_rejects_instead_of_snapping(axis):
    with pytest.raises(ValueError, match="off-grid.*Insert.*GridSpec.custom"):
        _scene([_structure(normal_axis=axis, coordinate=.3e-3)]).compile_materials()


@pytest.mark.parametrize("mode", ["staircase", "conformal"])
def test_legacy_unresolved_35um_box_rejects_instead_of_disappearing(mode):
    geometry = mw.Box(position=(.3e-3, 0, 0), size=(35e-6, 2.4e-3, 2.4e-3))
    with pytest.raises(ValueError, match="would disappear"):
        _scene([mw.Structure(geometry, mw.Material.pec(), name="foil")], pec=mode).compile_materials()


def test_union_t_contact_overlap_and_gap_have_deterministic_owners():
    structures = [
        _structure(bounds=((-1.8e-3, 0), (-.6e-3, .6e-3))),
        _structure(bounds=((0, 1.8e-3), (-.6e-3, .6e-3))),
        _structure(bounds=((-1.2e-3, 1.2e-3), (.6e-3, 1.8e-3))),
        _structure(bounds=((-1.2e-3, 1.2e-3), (-1.8e-3, -1.2e-3))),
    ]
    layout = compile_planar_pec(_scene(structures))
    assert layout["audit"]["contacts"] == [[0, 1], [0, 2], [1, 2]]
    assert torch.all(layout["edges"]["Ey"][:, 2, :] == 0)
    overlap = compile_planar_pec(_scene([structures[0], structures[0]]))
    assert torch.all(overlap["owners"]["Ex"][overlap["edges"]["Ex"].bool()] == 1)


def test_subcell_hole_opens_crossing_edge_even_when_endpoints_and_midpoint_miss_it():
    outer = ((-1.8e-3, -1.8e-3), (1.8e-3, -1.8e-3), (1.8e-3, 1.8e-3), (-1.8e-3, 1.8e-3))
    hole = ((.09e-3, -.2e-3), (.15e-3, -.2e-3), (.15e-3, .2e-3), (.09e-3, .2e-3))
    layout = compile_planar_pec(_scene([mw.Structure(mw.PlanarPEC(outer, holes=(hole,)), mw.Material.pec())]))
    assert layout["edges"]["Ex"][4, 4, 4] == 0
    assert layout["edges"]["Ex"][3, 4, 4] == 1


def test_nonconvex_polygon_and_plane_intersection_preserve_contacts():
    polygon = ((-1.2e-3, -1.2e-3), (1.2e-3, -1.2e-3), (1.2e-3, 0), (0, 0), (0, 1.2e-3), (-1.2e-3, 1.2e-3))
    a = mw.Structure(mw.PlanarPEC(polygon), mw.Material.pec())
    layout = compile_planar_pec(_scene([a, _structure(normal_axis="x")]))
    assert layout["audit"]["contacts"] == [[0, 1]]
    assert layout["edges"]["Ex"][5, 5, 4] == 0


def test_invalid_material_unresolved_strip_and_gradient_fail_explicitly():
    wrong = mw.Structure(mw.PlanarPEC.rectangle(((-1e-3, 1e-3),) * 2), mw.Material(eps_r=4))
    with pytest.raises(ValueError, match="requires Material.pec"):
        _scene([wrong]).compile_materials()
    with pytest.raises(ValueError, match="unresolved lateral support"):
        compile_planar_pec(_scene([_structure(bounds=((-1.2e-3, 1.2e-3), (.1e-3, .2e-3)))]))
    with pytest.raises(NotImplementedError, match="shape gradients"):
        mw.PlanarPEC(torch.randn(4, 2, requires_grad=True))


def test_holes_outside_or_crossing_outer_boundary_are_rejected():
    outer = ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))
    with pytest.raises(ValueError, match="strictly inside"):
        mw.PlanarPEC(outer, holes=(((2., 2.), (3., 2.), (3., 3.), (2., 3.)),))
    with pytest.raises(ValueError, match="disjoint"):
        mw.PlanarPEC(outer, holes=(((.5, .5), (2., .5), (2., 2.), (.5, 2.)),))


def test_auto_grid_inserts_exact_float64_plane_without_copper_cfl_interval():
    coordinate = .123456789e-3
    geometry = mw.PlanarPEC.rectangle(((-.9e-3, .9e-3),)*2,
                                      coordinate=coordinate, physical_thickness=18e-6)
    public = mw.Scene(domain=mw.Domain(bounds=((-2e-3, 2e-3),)*3),
                      grid=mw.GridSpec.auto(wavelength=.01, min_steps_per_wavelength=10),
                      device="cpu", structures=[mw.Structure(geometry, mw.Material.pec())])
    scene = prepare_scene(public)
    layout = compile_planar_pec(scene)
    assert coordinate in scene.z_nodes64
    assert np.diff(scene.z_nodes64).min() > 18e-6
    assert layout["audit"]["retention_fraction"] == 1


def test_inherited_geometry_translation_and_trainable_position_are_not_ignored():
    structure = _structure()
    structure.geometry.position[0] = .2e-3
    with pytest.raises(ValueError, match="derived from its plane"):
        compile_planar_pec(_scene([structure]))
    structure = _structure()
    structure.geometry.position.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="shape gradients"):
        compile_planar_pec(_scene([structure]))
