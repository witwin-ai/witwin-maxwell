import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def test_polarized_eps_is_differentiable_wrt_box_size():
    # Grid 0.25 with the box centered at (0.5, 0.5, 0.5): 0.25/0.5/0.75 are
    # exactly representable in float32, so the SDF is bitwise-symmetric and the
    # center node has |grad(sdf)| == 0. This exercises the degenerate-node case
    # where an unregularized sqrt backward would produce NaN (guards the
    # in-sqrt normalization floor). A 0.1 grid would pass by float accident.
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)
    size = torch.tensor([0.4, 0.4, 0.4], requires_grad=True)
    scene = mw.Scene(
        domain=domain,
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(averaging="polarized"),
    )
    scene.add_structure(
        mw.Structure(
            name="slab",
            geometry=mw.Box(position=(0.5, 0.5, 0.5), size=size),
            material=mw.Material(eps_r=12.0),
        )
    )
    eps, _ = prepare_scene(scene).compile_material_components()
    loss = sum(eps[axis].sum() for axis in ("x", "y", "z"))
    (grad,) = torch.autograd.grad(loss, size)
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_conformal_pec_occupancy_is_differentiable_wrt_box_size():
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.1)
    size = torch.tensor([0.35, 0.35, 0.35], requires_grad=True)
    scene = mw.Scene(
        domain=domain,
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(pec="conformal"),
    )
    scene.add_structure(
        mw.Structure(
            name="pec",
            geometry=mw.Box(position=(0.5, 0.5, 0.5), size=size),
            material=mw.Material.pec(),
        )
    )
    occupancy = prepare_scene(scene).compile_materials()["pec_occupancy"]
    (grad,) = torch.autograd.grad(occupancy.sum(), size)
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
