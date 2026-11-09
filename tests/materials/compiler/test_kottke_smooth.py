import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _PCBConductivityReplay
from witwin.maxwell.scene import prepare_scene


def test_fractional_oblique_window_has_physical_electric_and_magnetic_diagonals():
    shift = (.05 / math.sqrt(2) - 5) / math.sqrt(2)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.5), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=2, averaging="kottke_smooth"),
        structures=[mw.Structure(
            mw.Box(position=(-.25 + shift, shift, 0), size=(10, 10, 10),
                   rotation=(math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8))),
            mw.Material(eps_r=4, mu_r=9, sigma_e=.2, sigma_m=.03),
        )],
    )
    edges = prepare_scene(scene).compile_materials()["edge_components"]
    # x+y < -.20: fills on the four xy offsets are 1, .6, .6, .1,
    # repeated on both z offsets. Symmetry gives n_x^2 = n_y^2 = 1/2.
    def expected(value):
        samples = 1 + torch.tensor([1., .6, .6, .1]) * (value - 1)
        return .5 / samples.reciprocal().mean() + .5 * samples.mean()

    assert float(edges["eps"]["Ex"][0, 1, 1]) == pytest.approx(float(expected(4)), rel=2e-5)
    assert float(edges["mu"]["Hx"][1, 0, 0]) == pytest.approx(float(expected(9)), rel=2e-5)


def test_trainable_geometry_and_material_are_accepted_and_replay_is_rejected():
    size = torch.tensor([.43, .37, .41], requires_grad=True)
    epsilon = torch.tensor(4., requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.125), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_smooth"),
        structures=[mw.Structure(
            mw.Box(size=size, rotation=(math.cos(.235), 0, 0, math.sin(.235))),
            mw.Material(eps_r=epsilon),
        )],
    )
    edges = prepare_scene(scene).compile_materials()["edge_components"]["eps"]
    gradients = torch.autograd.grad(sum(value.sum() for value in edges.values()), (size, epsilon))
    assert all(bool(torch.isfinite(value).all() & (value.abs().sum() > 0)) for value in gradients)
    with pytest.raises(NotImplementedError, match="replay.*kottke_smooth"):
        _PCBConductivityReplay(scene)
