import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _slab_components(averaging, eps_r=12.0):
    # Domain [0,1]^3, grid 0.25 -> nodes at 0, 0.25, 0.5, 0.75, 1.0.
    # A slab thick enough that its low face lands on the interior node x=0.25 while
    # the neighbouring node x=0.5 is strictly interior, so the node-based
    # finite-difference normal at x=0.25 resolves cleanly along x.
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)
    scene = mw.Scene(
        domain=domain,
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(averaging=averaging),
    )
    scene.add_structure(
        mw.Structure(
            name="slab",
            geometry=mw.Box(position=(0.625, 0.5, 0.5), size=(0.75, 1.0, 1.0)),
            material=mw.Material(eps_r=eps_r),
        )
    )
    eps, _ = prepare_scene(scene).compile_material_components()
    return eps


def test_polarized_normal_axis_uses_harmonic_mean():
    eps_r = 12.0
    eps = _slab_components("polarized", eps_r=eps_r)
    node = (1, 1, 1)  # x = 0.25 interface node, normal along x.

    arithmetic_mean = 0.5 * 1.0 + 0.5 * eps_r
    harmonic_mean = 1.0 / (0.5 / 1.0 + 0.5 / eps_r)

    eps_x = eps["x"][node].item()
    eps_y = eps["y"][node].item()
    eps_z = eps["z"][node].item()

    # Normal (x) component closer to harmonic; tangential (y, z) equal to arithmetic.
    assert abs(eps_x - harmonic_mean) < abs(eps_x - arithmetic_mean)
    assert abs(eps_x - harmonic_mean) < 1e-3
    assert abs(eps_y - arithmetic_mean) < 1e-3
    assert abs(eps_z - arithmetic_mean) < 1e-3

    # The polarized normal component beats arithmetic at approaching the harmonic mean.
    eps_x_arith = _slab_components("arithmetic", eps_r=eps_r)["x"][node].item()
    assert abs(eps_x - harmonic_mean) < abs(eps_x_arith - harmonic_mean)


def test_arithmetic_mode_keeps_isotropic_components():
    eps = _slab_components("arithmetic")
    node = (1, 1, 1)
    assert abs(eps["x"][node].item() - eps["y"][node].item()) < 1e-6
    assert abs(eps["x"][node].item() - eps["z"][node].item()) < 1e-6
