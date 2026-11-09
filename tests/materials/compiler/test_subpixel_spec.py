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
        mw.SubpixelSpec(averaging="bilinear", samples=1)
    with pytest.raises(ValueError):
        mw.SubpixelSpec(pec="dey_mittra", averaging="arithmetic", samples=1)
    with pytest.raises(ValueError):
        mw.SubpixelSpec(samples=0, averaging="arithmetic")


def test_subpixel_spec_normalizes_samples():
    assert mw.SubpixelSpec(samples=3, averaging="arithmetic").samples == (3, 3, 3)
    assert mw.SubpixelSpec(samples=(2, 3, 4), averaging="arithmetic").samples == (2, 3, 4)


def test_conformal_pec_warns_about_systematic_bias():
    with pytest.warns(
        UserWarning,
        match=r'known systematic bias: non-converging scattering error of about 30-40% '
        r'on curved PEC\. Use pec="staircase" instead\. See docs/memory/conformal-pec-bias\.md\.',
    ):
        spec = mw.SubpixelSpec(pec="conformal", averaging="arithmetic", samples=1)
    assert spec.pec == "conformal"


def test_subpixel_spec_is_frozen_and_hashable():
    """Uses samples=2 since single-sample polarized averaging raises (2026-10-29)."""
    spec = mw.SubpixelSpec(samples=2, averaging="polarized", pec="conformal")
    with pytest.raises(Exception):
        spec.averaging = "arithmetic"
    cache = {spec: 1, mw.SubpixelSpec(averaging="arithmetic", samples=1): 2}
    assert cache[mw.SubpixelSpec(samples=2, averaging="polarized", pec="conformal")] == 1


@pytest.mark.parametrize("averaging", ["polarized", "polarized_tensor"])
def test_polarized_spec_requires_more_than_one_sample(averaging):
    """A single point sample has a binary fill, so the normal projection would
    silently reproduce the staircase; polarized averaging requires >= 2 samples."""
    with pytest.raises(ValueError, match="more than one subpixel sample"):
        mw.SubpixelSpec(averaging=averaging, samples=1)
    with pytest.raises(ValueError, match="more than one subpixel sample"):
        mw.SubpixelSpec(samples=1, averaging=averaging, pec="conformal")
    assert mw.SubpixelSpec(samples=(1, 1, 2), averaging=averaging).samples == (1, 1, 2)


def test_explicit_single_sample_spec_matches_int_bitwise():
    eps_int, mu_int = prepare_scene(_slab_scene(1)).compile_material_components()
    eps_spec, mu_spec = prepare_scene(_slab_scene(mw.SubpixelSpec(averaging="arithmetic", samples=1))).compile_material_components()
    for axis in ("x", "y", "z"):
        assert torch.equal(eps_int[axis], eps_spec[axis])
        assert torch.equal(mu_int[axis], mu_spec[axis])


def test_default_spec_matches_explicit_edge_policy_bitwise():
    assert mw.SubpixelSpec().samples == (4, 4, 4)
    assert mw.SubpixelSpec().averaging == "kottke_edge"
    assert mw.SubpixelSpec().pec == "staircase"
    eps_default, mu_default = prepare_scene(_slab_scene(mw.SubpixelSpec())).compile_material_components()
    eps_arith, mu_arith = prepare_scene(
        _slab_scene(mw.SubpixelSpec(samples=4, averaging="kottke_edge", pec="staircase"))
    ).compile_material_components()
    for axis in ("x", "y", "z"):
        assert torch.equal(eps_default[axis], eps_arith[axis])
        assert torch.equal(mu_default[axis], mu_arith[axis])


def test_scene_clone_round_trips_averaging_and_pec():
    """Uses samples=2 since single-sample polarized averaging raises (2026-10-29)."""
    scene = _slab_scene(mw.SubpixelSpec(samples=2, averaging="polarized", pec="conformal"))
    clone = scene.clone()
    assert clone.subpixel.averaging == "polarized"
    assert clone.subpixel.pec == "conformal"
    assert clone.subpixel_samples == (2, 2, 2)
