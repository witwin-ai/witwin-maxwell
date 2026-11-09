"""Material gradients through mode-matched faces and internal absorber planes.

Both reuse the one-step Mur entry (and its discrete transpose), so the adjoint
gradient of a design-region permittivity must match central finite differences.
"""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_PLANE = (0.0, math.inf, math.inf)


def _scene(density, variant):
    if variant == "internal":
        boundary = mw.BoundarySpec.pml(num_layers=2)
        absorbers = (mw.InternalAbsorber((0.3, 0.0, 0.0), _PLANE, "+", boundary=mw.MurABC(permittivity=1.2)),)
    else:
        boundary = mw.BoundarySpec.faces(default="pml", num_layers=2, x_high="mode_abc",
                                         mode_abc=mw.ModeABC(frequency=5e8))
        absorbers = ()
    scene = mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"), domain=mw.Domain(bounds=((-0.75, 0.75), (-0.6, 0.6), (-0.6, 0.6))),
                     grid=mw.GridSpec.uniform(0.15), boundary=boundary, device="cuda",
                     internal_absorbers=absorbers)
    scene.add_material_region(mw.MaterialRegion(name="design", geometry=mw.Box(position=(0., 0., 0.), size=(0.15,) * 3),
                                                density=density, eps_bounds=(1., 6.), mu_bounds=(1., 1.)))
    scene.add_source(mw.PointDipole(position=(-0.45, 0.0, 0.0), polarization="Ey", width=0.12,
                                    source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=25.)))
    scene.add_monitor(mw.PointMonitor("probe", (0.15, 0., 0.), fields=("Ey",)))
    return scene


def _objective(value, variant, *, grad=False):
    density = torch.full((1, 1, 1), value, device="cuda", dtype=torch.float64, requires_grad=grad)
    simulation = mw.Simulation.fdtd(_scene(density, variant), frequencies=[1e9],
                                    run_time=mw.TimeConfig(time_steps=400),
                                    spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    simulation.config.adjoint_checkpoint_stride = 61
    result = simulation.run()
    return result, result.monitor("probe")["data"].abs().square().sum(), density


@pytest.mark.parametrize("variant", ["internal", "mode_abc"])
def test_one_way_absorber_gradient_matches_finite_difference(variant):
    result, objective, density = _objective(0.5, variant, grad=True)
    assert result.solver._mur_state
    assert any(entry.get("source") for entry in result.solver._mur_state)
    objective.backward()
    derivative = float(density.grad.sum())
    h = 0.003
    _, plus, _ = _objective(0.5 + h, variant)
    _, minus, _ = _objective(0.5 - h, variant)
    finite_difference = float((plus - minus) / (2 * h))
    assert abs(derivative) > 0.0
    assert abs(derivative - finite_difference) <= 2e-3 * max(abs(derivative), abs(finite_difference), 1e-12)
