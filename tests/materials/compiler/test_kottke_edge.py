import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _PCBConductivityReplay
from witwin.maxwell.scene import prepare_scene


def test_oblique_binary_window_has_physical_electric_and_magnetic_diagonals():
    # At (-.25, 0, 0), three of the eight samples lie inside x+y < -.20.
    # Both moment components are equal: n_x^2 = n_y^2 = 1/2.
    shift = (.05 / math.sqrt(2) - 5) / math.sqrt(2)
    epsilon = torch.tensor(4., requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.5), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=2, averaging="kottke_edge"),
        structures=[mw.Structure(
            mw.Box(position=(-.25 + shift, shift, 0), size=(10, 10, 10),
                   rotation=(math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8))),
            mw.Material(eps_r=epsilon, mu_r=9, sigma_e=.2, sigma_m=.03),
        )],
    )
    edges = prepare_scene(scene).compile_materials()["edge_components"]
    expected = .5 / (.25 + .75 / 4) + .5 * (.25 + .75 * 4)
    actual = edges["eps"]["Ex"][0, 1, 1]
    assert float(actual.detach()) == pytest.approx(expected, rel=1e-6)
    gradient, = torch.autograd.grad(actual, epsilon)
    expected_gradient = .5 * .75 / (4**2 * (.25 + .75/4)**2) + .5 * .75
    assert float(gradient) == pytest.approx(expected_gradient, rel=1e-6)
    # Hx at (0, -.25, -.25) has the same x+y and the same binary window.
    expected_mu = .5 / (.25 + .75 / 9) + .5 * (.25 + .75 * 9)
    assert float(edges["mu"]["Hx"][1, 0, 0]) == pytest.approx(expected_mu, rel=1e-6)


def test_kottke_edge_rejects_trainable_geometry_and_conductivity_replay():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.5), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_edge"),
        structures=[mw.Structure(
            mw.Box(size=torch.tensor([.4, .4, .4], requires_grad=True)),
            mw.Material(eps_r=4),
        )],
    )
    with pytest.raises(ValueError, match="kottke_edge.*not differentiable in geometry"):
        prepare_scene(scene).compile_materials()
    with pytest.raises(NotImplementedError, match="replay.*kottke_edge"):
        _PCBConductivityReplay(scene)


def _edge_sigma(averaging, material, *, region=False):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.125), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging=averaging),
        structures=[mw.Structure(mw.Sphere(radius=.3), material)],
    )
    if region:
        scene.add_material_region(mw.MaterialRegion(
            name="design", geometry=mw.Box(position=(.25, 0, 0), size=(.3, .3, .3)),
            density=torch.full((4, 4, 4), .5), eps_bounds=(1., 3.)))
    edges = prepare_scene(scene).compile_materials()["edge_components"]
    return edges["sigma_e"], edges["sigma_m"]


def test_kottke_edge_conductivity_matches_polarized_and_lossless_shortcut_is_exactly_zero():
    lossy = mw.Material(eps_r=4, mu_r=2, sigma_e=.2, sigma_m=.03)
    for region in (False, True):
        sigma_e, sigma_m = _edge_sigma("kottke_edge", lossy, region=region)
        reference_e, reference_m = _edge_sigma("polarized", lossy, region=region)
        for name in sigma_e:
            assert bool((sigma_e[name] > 0).any()) and torch.equal(sigma_e[name], reference_e[name])
        for name in sigma_m:
            assert bool((sigma_m[name] > 0).any()) and torch.equal(sigma_m[name], reference_m[name])
        # Lossless structures (with or without a region, which has no sigma) take the shortcut.
        sigma_e, sigma_m = _edge_sigma("kottke_edge", mw.Material(eps_r=4, mu_r=2), region=region)
        assert all(not bool(value.any()) for value in (*sigma_e.values(), *sigma_m.values()))
