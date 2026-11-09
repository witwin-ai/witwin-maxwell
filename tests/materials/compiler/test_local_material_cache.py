"""Density patch reuse matches full material compilation with global Yee halos."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.materials import LocalMaterialCache, compile_material_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def scene(*, samples, averaging, periodic=False, graded=False):
    coordinates = np.linspace(-.3, .3, 49)
    if graded:
        coordinates = coordinates + .1 * coordinates**3
    grid = mw.GridSpec.custom(coordinates, coordinates, coordinates) if graded else mw.GridSpec.uniform(.0125)
    domain = mw.Domain(bounds=((-float(coordinates[-1]), float(coordinates[-1])),) * 3)
    public = mw.Scene(domain=domain, grid=grid, device="cuda:0",
        boundary=mw.BoundarySpec.periodic() if periodic else mw.BoundarySpec.pml(num_layers=3),
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging))
    public.add_structure(mw.Structure(name="background",
        geometry=mw.Sphere(radius=.12, position=(.02, 0., 0.), device="cuda:0"),
        material=mw.Material(eps_r=2.7, sigma_e=.01)))
    public.add_material_region(mw.MaterialRegion(name="design",
        geometry=mw.Box(size=(.04, .05, .03), device="cuda:0"),
        density=torch.full((4, 5, 3), .2, device="cuda:0"), eps_bounds=(1., 4.),
        mu_bounds=(1., 1.2), filter_radius=.013, projection_beta=2.))
    return public


def assert_model(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_model(actual[key], expected[key])


@pytest.mark.parametrize("samples,averaging,periodic,graded", [
    (1, "arithmetic", False, False), (1, "arithmetic", False, True),
    (2, "polarized", False, False), (2, "polarized", True, True),
])
def test_local_material_bank_matches_full_recompilation(samples, averaging, periodic, graded):
    """Local patch reuse equals full recompilation.

    The former polarized samples=1 graded case is now arithmetic samples=1
    graded (2026-10-29), because single-sample polarized averaging raises.
    """
    public = scene(samples=samples, averaging=averaging, periodic=periodic, graded=graded)
    base = prepare_scene(public)
    cache = LocalMaterialCache(base, ("design",))
    if not periodic:
        assert all(s.stop - s.start < n for s, n in zip(cache.node_selection, (base.Nx, base.Ny, base.Nz)))
    for density in (.8, .0, .2):
        changed = public.clone()
        changed.material_regions = [replace(changed.material_regions[0],
            density=torch.full((4, 5, 3), density, device="cuda:0"))]
        prepared = prepare_scene(changed)
        actual = cache.compile(prepared)
        expected = compile_material_model(prepared, subpixel=prepared.subpixel)
        assert_model(actual, expected)
    assert cache.full_compilations == 1
    assert cache.patch_compilations == 3
    changed.structures = [mw.Structure(name="background", geometry=changed.structures[0].geometry,
                                      material=mw.Material(eps_r=3.))]
    with pytest.raises(ValueError, match="Fixed material definition"):
        cache.compile(prepare_scene(changed))


@pytest.mark.parametrize("periodic,graded", [(False, False), (False, True), (True, True)])
def test_structure_influence_envelopes_match_full_materials(periodic, graded):
    public = scene(samples=2, averaging="polarized", periodic=periodic, graded=graded)
    envelope = mw.Box(size=(.32, .32, .32), device="cuda:0")
    public.add_structure(mw.Structure(name="moving", priority=2,
        geometry=mw.Box(size=(.04, .05, .03), device="cpu"), material=mw.Material(eps_r=3.)))
    cache = LocalMaterialCache(prepare_scene(public), structure_envelopes={"moving": envelope})
    for position, radius, epsilon in ((.05, .025, 4.), (-.06, .035, 1.2), (0., .02, 3.)):
        changed = public.clone()
        changed.structures[-1] = mw.Structure(name="moving", priority=2,
            geometry=mw.Sphere(position=(position, 0., 0.), radius=radius, device="cuda:0"),
            material=mw.Material(eps_r=epsilon, sigma_e=.02, mu_r=1.1))
        compiled = prepare_scene(changed)
        assert_model(cache.compile(compiled), compile_material_model(compiled, subpixel=compiled.subpixel))
    changed.structures[-1] = mw.Structure(name="moving", priority=2,
        geometry=mw.Sphere(position=(.2, 0., 0.), radius=.02, device="cuda:0"),
        material=mw.Material(eps_r=3.))
    with pytest.raises(ValueError, match="envelope"):
        cache.compile(prepare_scene(changed))
    envelope.size.mul_(2)
    with pytest.raises(ValueError, match="envelopes must remain unchanged"):
        cache.compile(prepare_scene(public))


# --- PlanarPEC sheets and diff-driven local recompiles -----------------------

MM = 1e-3


def assert_identical(actual, expected, path="model"):
    """Exact equality, including key order, dtypes and every compiled tensor bit."""
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), path
        for key in expected:
            assert_identical(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_identical(a, b, f"{path}[{index}]")
    else:
        assert actual == expected, path


def _nodes(lo, hi, fine_lo, fine_hi):
    return np.unique(np.round(np.r_[np.arange(lo, fine_lo, .2), np.arange(fine_lo, fine_hi, .1),
                                    np.arange(fine_hi, hi + 1e-9, .2), hi], 6)) * MM


def _sheet(name, x0, x1, y0, y1, z=.2, priority=0):
    return mw.Structure(mw.PlanarPEC.rectangle(((x0 * MM, x1 * MM), (y0 * MM, y1 * MM)), normal_axis="z",
                                               coordinate=z * MM), mw.Material.pec(), name=name, priority=priority)


def _box(name, center, size, material, priority=0):
    return mw.Structure(mw.Box(position=tuple(c * MM for c in center), size=tuple(s * MM for s in size),
                               device="cpu"), material, name=name, priority=priority)


def _patch(top=.6, *, width=.7, extra=(), drop=(), load=None, pec=None):
    items = [_box("substrate", (0, 0, .1), (3., 3., .2), mw.Material(eps_r=2.56)),
             _sheet("ground", -1.5, 1.5, -1.5, 1.5, z=0.),
             _sheet("patch", -width, width, -.6, top),
             _sheet("feed", -.1, .1, -1.2, -.6)]
    if load is not None:
        items.append(_box("load", *load, mw.Material(eps_r=4., sigma_e=.02), priority=1))
    if pec is not None:
        items.append(_box("pin", *pec, mw.Material.pec()))
    return [s for s in items if s.name not in drop] + list(extra)


def _patch_scene(items, *, subpixel=None):
    x, y, z = _nodes(-2., 2., -1., 1.), _nodes(-2., 2., -1.3, 1.), _nodes(-1., 1.2, -.2, .4)
    return prepare_scene(mw.Scene(domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
        grid=mw.GridSpec.custom(x, y, z), boundary=mw.BoundarySpec.pml(num_layers=3), device="cuda:0",
        structures=items, subpixel_samples=subpixel or mw.SubpixelSpec(averaging="arithmetic", samples=1)))


_LOAD = ((1.1, 0., .1), (.4, .4, .2))
_PIN = ((.5, .3, .1), (.2, .2, .2))
PATCH_EDITS = {
    "top_across_cells": (_patch(), _patch(.25)),
    "top_within_cell": (_patch(), _patch(.63)),
    "top_on_grid_line": (_patch(.65), _patch(.7)),
    "shrink_width": (_patch(), _patch(width=.42)),
    "grow_width": (_patch(), _patch(width=.93)),
    "sheet_added_overlapping": (_patch(), _patch(extra=[_sheet("tab", .5, 1.2, .1, .5, priority=1)])),
    "sheet_added_before": (_patch(), [_sheet("front", -1.3, -.9, -.4, .4)] + _patch()),
    "sheet_removed": (_patch(), _patch(drop=("feed",))),
    "touching_pec_box": (_patch(.25, pec=_PIN), _patch(.6, pec=_PIN)),
    "dielectric_resized_next_to_sheet": (_patch(load=_LOAD), _patch(load=((1.2, .1, .1), (.6, .8, .2)))),
    "dielectric_added_and_sheet_moved": (_patch(), _patch(.25, load=_LOAD)),
}


@pytest.mark.parametrize("edit", sorted(PATCH_EDITS))
def test_material_cache_local_recompile_is_identical_to_full_compile(edit):
    base, changed = PATCH_EDITS[edit]
    cache = mw.FDTDMaterialCache()
    assert cache._bind(_patch_scene(base), frequency=60e9)["outcome"] == "miss"
    scene = _patch_scene(changed)
    info = cache._bind(scene, frequency=60e9)
    assert info["outcome"] == "local" and info["local_recompiles"] == 1
    expected = compile_material_model(_patch_scene(changed), subpixel=scene.subpixel)
    assert_identical(scene._material_model_override, expected)
    # The solver copy is owned: the retained snapshot and its base survive mutation.
    scene._material_model_override["eps_components"]["x"].zero_()
    again = _patch_scene(changed)
    assert cache._bind(again, frequency=60e9)["outcome"] == "hit"
    assert_identical(again._material_model_override, expected)


def test_local_recompile_reuses_unchanged_sheets_and_follows_successive_edits():
    from witwin.maxwell.compiler import planar_pec
    base = _patch_scene(_patch())
    local = LocalMaterialCache(base, base=base.compile_materials())
    calls = []
    original = planar_pec._compile_planar_sheet
    def counting(scene, structure, *args):
        calls.append(structure.name)
        return original(scene, structure, *args)
    tops = (.25, .63, .7, .6)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(planar_pec, "_compile_planar_sheet", counting)
        actual = [local.recompile(_patch_scene(_patch(top)), {0: 0, 1: 1, 3: 3}) for top in tops]
    # The first edit fills the unchanged ground/feed layouts; later edits resample the patch only.
    assert calls == ["ground", "patch", "feed", "patch", "patch", "patch"]
    for top, model in zip(tops, actual):
        expected = _patch_scene(_patch(top))
        assert_identical(model, compile_material_model(expected, subpixel=expected.subpixel))
    assert local.full_compilations == 0 and local.patch_compilations == 4


@pytest.mark.parametrize("reason,base,changed,subpixel", [
    ("conformal sheet", _patch(), _patch(.25), mw.SubpixelSpec(pec="conformal", averaging="arithmetic", samples=1)),
    ("volume PEC change", _patch(pec=_PIN), _patch(pec=((.4, .3, .1), (.2, .2, .2))), None),
])
def test_unproven_local_changes_fall_back_to_a_full_compile(reason, base, changed, subpixel):
    cache = mw.FDTDMaterialCache()
    cache._bind(_patch_scene(base, subpixel=subpixel), frequency=60e9)
    scene = _patch_scene(changed, subpixel=subpixel)
    assert cache._bind(scene, frequency=60e9)["outcome"] == "miss", reason
    assert cache.stats["local_recompiles"] == 0
    assert_identical(scene._material_model_override,
                     compile_material_model(_patch_scene(changed, subpixel=subpixel), subpixel=scene.subpixel))


@pytest.mark.parametrize("material", [
    mw.Material.pec(), mw.CopperFoil(thickness=2e-6, frequency_range=(50e9, 70e9)),
], ids=["pec", "copper"])
def test_surface_mesh_material_flip_requires_full_compile(material):
    mesh = mw.SurfaceMesh(np.array([[-.7, -.6, .2], [.7, -.6, .2], [.7, .6, .2], [-.7, .6, .2]]) * MM,
                          [(0, 1, 2), (0, 2, 3)], device="cuda:0")
    base = _patch()
    base[2] = replace(base[2], geometry=mesh, material=material)
    cache = mw.FDTDMaterialCache()
    # Future admission of SurfaceMesh must handle thin-sheet ownership flips (owns_thin_sheet) as a full compile.
    with pytest.raises(NotImplementedError, match="FDTDMaterialCache supports static Mesh/Box/PlanarPEC"):
        cache._bind(_patch_scene(base), frequency=60e9)
    assert cache.stats["local_recompiles"] == 0


@pytest.mark.parametrize("subpixel", [mw.SubpixelSpec(samples=2, averaging="arithmetic"), mw.SubpixelSpec(samples=2, averaging="polarized")])
def test_multi_sample_local_recompile_is_identical(subpixel):
    base, changed = PATCH_EDITS["dielectric_added_and_sheet_moved"]
    cache = mw.FDTDMaterialCache()
    cache._bind(_patch_scene(base, subpixel=subpixel), frequency=60e9)
    scene = _patch_scene(changed, subpixel=subpixel)
    assert cache._bind(scene, frequency=60e9)["outcome"] == "local"
    assert_identical(scene._material_model_override,
                     compile_material_model(_patch_scene(changed, subpixel=subpixel), subpixel=scene.subpixel))


def test_reordered_structures_are_not_local():
    base = _patch_scene(_patch())
    local = LocalMaterialCache(base, base=base.compile_materials())
    assert local.recompile(_patch_scene(_patch()), {0: 1, 1: 0}) is None


@pytest.mark.parametrize("edit", ["top_across_cells", "shrink_width", "touching_pec_box"])
def test_batch_planar_pec_envelopes_match_full_materials(edit):
    base, changed = PATCH_EDITS[edit]
    envelope = mw.Box(position=(0., 0., .2 * MM), size=(2.4 * MM,) * 3, device="cuda:0")
    cache = LocalMaterialCache(_patch_scene(base), structure_envelopes={"patch": envelope})
    items = [new if new.name == "patch" else old for old, new in zip(base, changed)]
    scene = _patch_scene(items)
    assert_identical(cache.compile(scene), compile_material_model(_patch_scene(items), subpixel=scene.subpixel))
    with pytest.raises(ValueError, match="envelope"):
        cache.compile(_patch_scene([_sheet("patch", -.7, 1.8, -.6, .6) if s.name == "patch" else s for s in base]))
