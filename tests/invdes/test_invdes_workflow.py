import json
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell import invdes
from witwin.maxwell.scene import prepare_scene


def test_density_parameterization_pipeline_and_symmetric_gradient():
    torch.manual_seed(0)
    design = invdes.DensityParameterization(
        (10, 10, 1), (0.5, 0.5, 0.1), radius=0.1, beta=8.0, diagonal=(0, 1), init=torch.rand(10, 10, 1)
    )
    assert design.spacing == pytest.approx((0.05, 0.05, 0.1))
    density = design()
    assert density.shape == (10, 10, 1)
    torch.testing.assert_close(density, density.transpose(0, 1))
    expected = invdes.tanh_projection(
        invdes.conic_filter(invdes.diagonal_symmetrize(design.params.detach()), 0.1, design.spacing), 8.0
    )
    torch.testing.assert_close(density.detach(), expected)
    (density * torch.arange(100.0).reshape(10, 10, 1)).sum().backward()
    torch.testing.assert_close(design.params.grad, design.params.grad.transpose(0, 1))

    eroded, nominal, dilated = design.robust(0.1)
    assert (eroded <= nominal + 1e-6).all() and (nominal <= dilated + 1e-6).all()
    design.smoothed = True
    assert torch.isfinite(design(beta=math.inf)).all()


def test_density_parameterization_feeds_material_region_gradients():
    design = invdes.DensityParameterization((8, 8, 1), (0.4, 0.4, 0.1), radius=0.1, beta=4.0, init=torch.rand(8, 8, 1))
    scene = mw.Scene(
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
        domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.1)),
            density=design(),
            eps_bounds=(1.0, 4.0),
        )
    )
    eps, _ = prepare_scene(scene).compile_material_tensors()
    assert 1.0 <= float(eps.detach().min()) and float(eps.detach().max()) <= 4.0
    eps.sum().backward()
    assert design.params.grad is not None and design.params.grad.abs().sum() > 0


def test_design_loop_steps_clips_and_records_history():
    target = torch.tensor([0.2, 1.5, -0.4], dtype=torch.float64)
    params = torch.nn.Parameter(torch.full((3,), 0.5, dtype=torch.float64))
    optimizer = torch.optim.Adam([params], lr=0.1)
    schedule = invdes.BetaSchedule(initial=1.0, factor=2.0, interval=10, maximum=8.0)
    loop = invdes.DesignLoop(optimizer, bounds=(0.0, 1.0), beta_schedule=schedule)

    with pytest.raises(RuntimeError, match="backward"):
        loop.step(torch.tensor(0.0))
    for _ in range(60):
        loop.zero_grad()
        loss = ((params - target) ** 2).sum()
        loss.backward()
        loop.step(loss, distance=(params - target).abs().max())
        assert float(params.detach().min()) >= 0.0 and float(params.detach().max()) <= 1.0

    torch.testing.assert_close(params.detach(), target.clamp(0.0, 1.0), atol=0.05, rtol=0)
    history = loop.history
    assert len(history) == 60 and loop.iteration == 60
    assert history.column("beta") == schedule.values(60)
    assert history.losses[-1] < history.losses[0]
    assert history.best()["loss"] == min(history.losses)
    table = history.to_dict()
    assert set(table) == {"iteration", "loss", "beta", "grad_norm", "distance"}
    json.dumps(table)


def test_clip_to_bounds_accepts_tensor_bounds():
    p = torch.nn.Parameter(torch.tensor([-1.0, 0.5, 2.0]))
    invdes.clip_to_bounds_([p], torch.tensor([0.0, 0.6, 0.0]), 1.0)
    torch.testing.assert_close(p.detach(), torch.tensor([0.0, 0.6, 1.0]))
