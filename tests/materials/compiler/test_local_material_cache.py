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
    (1, "arithmetic", False, False), (1, "polarized", False, True),
    (2, "polarized", False, False), (2, "polarized", True, True),
])
def test_local_material_bank_matches_full_recompilation(samples, averaging, periodic, graded):
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
        geometry=mw.Box(size=(.04, .05, .03), device="cuda:0"), material=mw.Material(eps_r=3.)))
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
