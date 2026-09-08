"""Mur history, complex phase covariance, and long-horizon material gradients."""
import pytest
import torch

import witwin.maxwell as mw
from tests.gradients.test_fdtd_mixed_boundary_consistency import _scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _run(value, axis, complex_fields, *, grad=False):
    density = torch.full((1, 1, 1), value, device="cuda", dtype=torch.float64, requires_grad=grad)
    scene = _scene(density, axis)
    boundary = (mw.BoundarySpec.faces(default="mur", bloch_wavevector=(0.7, 0.9, 1.1),
        **{a: "mur" if a == axis else "bloch" for a in "xyz"}) if complex_fields else mw.BoundarySpec.mur())
    simulation = mw.Simulation.fdtd(scene.clone(boundary=boundary), frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=400), spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    simulation.config.adjoint_checkpoint_stride = 37 if complex_fields else 61
    result = simulation.run()
    return result, result.monitor("probe")["data"], density


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("complex_fields", [False, True])
def test_mur_forward_and_gradient_match_fixed_material(axis, complex_fields):
    fixed, expected, _ = _run(0.5, axis, complex_fields)
    trained, got, density = _run(0.5, axis, complex_fields, grad=True)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    assert trained.solver._mur_state
    got.abs().square().sum().backward()
    derivative = float(density.grad.sum())
    h = 0.003
    _, plus, _ = _run(0.5 + h, axis, complex_fields)
    _, minus, _ = _run(0.5 - h, axis, complex_fields)
    finite_difference = float((plus.abs().square().sum() - minus.abs().square().sum()) / (2 * h))
    assert abs(derivative - finite_difference) <= 2e-3 * max(abs(derivative), abs(finite_difference), 1e-12)


@pytest.mark.parametrize("complex_fields", [False, True])
def test_mur_histories_survive_disk_resume(tmp_path, complex_fields):
    expected, _, _ = _run(0.5, "x", complex_fields)
    def simulation():
        return mw.Simulation.fdtd(expected.solver.scene.clone(monitors=()), frequencies=[1e9],
            run_time=mw.TimeConfig(time_steps=400),
            spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    checkpoint = simulation().prepare().run_until(173)
    path = tmp_path / "mur.pt"
    checkpoint.save(path)
    restored = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    actual = simulation().prepare()
    actual.simulation._execute_fdtd_solve(actual.solver, resume_from=restored)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(getattr(actual.solver, name), getattr(expected.solver, name), rtol=0, atol=0)
    for got, want in zip(actual.solver._mur_state, expected.solver._mur_state):
        for key in ("prev_boundary", "prev_adjacent"):
            torch.testing.assert_close(got[key], want[key], rtol=0, atol=0)
