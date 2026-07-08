import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _slab_scene(spec):
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.1)
    scene = mw.Scene(domain=domain, grid=grid, device="cpu", subpixel_samples=spec)
    scene.add_structure(
        mw.Structure(
            name="slab",
            geometry=mw.Box(position=(0.35, 0.5, 0.5), size=(0.5, 1.0, 1.0)),
            material=mw.Material(eps_r=6.0),
        )
    )
    return scene


def test_subpixel_spec_rejects_unknown_vocabulary():
    with pytest.raises(ValueError):
        mw.SubpixelSpec(averaging="bilinear")
    with pytest.raises(ValueError):
        mw.SubpixelSpec(pec="dey_mittra")
    with pytest.raises(ValueError):
        mw.SubpixelSpec(samples=0)


def test_subpixel_spec_normalizes_samples():
    assert mw.SubpixelSpec(samples=3).samples == (3, 3, 3)
    assert mw.SubpixelSpec(samples=(2, 3, 4)).samples == (2, 3, 4)


def test_subpixel_spec_is_frozen_and_hashable():
    spec = mw.SubpixelSpec(averaging="polarized", pec="conformal")
    with pytest.raises(Exception):
        spec.averaging = "arithmetic"
    cache = {spec: 1, mw.SubpixelSpec(): 2}
    assert cache[mw.SubpixelSpec(averaging="polarized", pec="conformal")] == 1


def test_default_spec_matches_int_subpixel_samples_bitwise():
    eps_int, mu_int = prepare_scene(_slab_scene(1)).compile_material_components()
    eps_spec, mu_spec = prepare_scene(_slab_scene(mw.SubpixelSpec())).compile_material_components()
    for axis in ("x", "y", "z"):
        assert torch.equal(eps_int[axis], eps_spec[axis])
        assert torch.equal(mu_int[axis], mu_spec[axis])


def test_explicit_arithmetic_spec_matches_default_bitwise():
    eps_default, mu_default = prepare_scene(_slab_scene(mw.SubpixelSpec())).compile_material_components()
    eps_arith, mu_arith = prepare_scene(
        _slab_scene(mw.SubpixelSpec(averaging="arithmetic", pec="staircase"))
    ).compile_material_components()
    for axis in ("x", "y", "z"):
        assert torch.equal(eps_default[axis], eps_arith[axis])
        assert torch.equal(mu_default[axis], mu_arith[axis])


def test_scene_clone_round_trips_averaging_and_pec():
    scene = _slab_scene(mw.SubpixelSpec(averaging="polarized", pec="conformal"))
    clone = scene.clone()
    assert clone.subpixel.averaging == "polarized"
    assert clone.subpixel.pec == "conformal"
    assert clone.subpixel_samples == (1, 1, 1)
