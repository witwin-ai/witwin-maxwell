import pytest
import torch

import witwin.maxwell as mw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9


class _DensityScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.CW(frequency=FREQ, amplitude=50.0),
            )
        )
        return scene


class _BoxPositionScene(mw.SceneModule):
    def __init__(self, init_x=0.10):
        super().__init__()
        self.box_x = torch.nn.Parameter(torch.tensor(float(init_x), device="cuda"))

    def to_scene(self):
        position = torch.stack(
            (self.box_x, self.box_x.new_tensor(0.0), self.box_x.new_tensor(0.06))
        )
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
            subpixel_samples=5,
        )
        scene.add_structure(
            mw.Structure(
                name="design_box",
                geometry=mw.Box(position=position, size=(0.18, 0.18, 0.18)),
                material=mw.Material(eps_r=20.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.CW(frequency=FREQ, amplitude=50.0),
            )
        )
        return scene


def _loss(model):
    result = mw.Simulation.fdfd(
        model,
        frequency=FREQ,
        solver=mw.GMRES(solver_type="direct"),
    ).run()
    ez = result.fields["EZ"]
    probe = ez[ez.shape[0] // 2, ez.shape[1] // 2, ez.shape[2] - 2]
    return probe.abs() ** 2


@requires_cuda
def test_fdfd_gradient_matches_central_difference_for_density():
    model = _DensityScene()
    loss = _loss(model)
    assert loss.requires_grad
    loss.backward()
    backward_grad = model.logits.grad.detach().reshape(())

    delta = 1.0e-2
    loss_plus = _loss(_DensityScene(delta)).detach()
    loss_minus = _loss(_DensityScene(-delta)).detach()
    finite_diff = (loss_plus - loss_minus) / (2.0 * delta)

    assert torch.allclose(backward_grad, finite_diff, rtol=5e-2, atol=1e-10), (
        backward_grad.item(),
        finite_diff.item(),
    )


@requires_cuda
def test_fdfd_gradient_matches_central_difference_for_geometry_position():
    model = _BoxPositionScene()
    loss = _loss(model)
    loss.backward()
    backward_grad = model.box_x.grad.detach()

    delta = 1.0e-2
    loss_plus = _loss(_BoxPositionScene(0.10 + delta)).detach()
    loss_minus = _loss(_BoxPositionScene(0.10 - delta)).detach()
    finite_diff = (loss_plus - loss_minus) / (2.0 * delta)

    assert torch.allclose(backward_grad, finite_diff, rtol=6e-1, atol=1e-10), (
        backward_grad.item(),
        finite_diff.item(),
    )


@requires_cuda
def test_fdfd_bridge_reports_solver_stats_and_converges():
    model = _DensityScene()
    result = mw.Simulation.fdfd(
        model,
        frequency=FREQ,
        solver=mw.GMRES(solver_type="direct"),
    ).run()
    stats = result.stats()
    assert stats["solver"]["type"] == "direct"
    assert stats["converged"] is True
    assert result.fields["EZ"].requires_grad
