"""Long-horizon public forward and adjoint checks with active boundary memory."""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _scene(density, axis):
    normal = "xyz".index(axis)
    transverse = (normal + 1) % 3
    bounds = [(-0.6, 0.6)] * 3
    bounds[normal] = (-0.75, 0.75)
    wavevector = [math.pi / 2.4] * 3
    wavevector[normal] = 0.0
    boundary = {a: "pml" if a == axis else "bloch" for a in "xyz"}
    scene = mw.Scene(domain=mw.Domain(bounds=tuple(bounds)), grid=mw.GridSpec.uniform(0.15),
                     boundary=mw.BoundarySpec.faces(default="pml", num_layers=2,
                         bloch_wavevector=tuple(wavevector), **boundary), device="cuda")
    scene.add_material_region(mw.MaterialRegion(name="design",
        geometry=mw.Box(position=(0., 0., 0.), size=(0.15,) * 3),
        density=density, eps_bounds=(1., 6.), mu_bounds=(1., 1.)))
    position = [0.] * 3
    position[normal] = -0.55
    component = "E" + "xyz"[transverse]
    scene.add_source(mw.PointDipole(position=tuple(position), polarization=component, width=0.12,
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=25.)))
    scene.add_monitor(mw.PointMonitor("probe", (0., 0., 0.), fields=(component,)))
    return scene


def _run(value, *, axis, memory, stride, grad=False):
    density = torch.full((1, 1, 1), value, device="cuda", dtype=torch.float64, requires_grad=grad)
    simulation = mw.Simulation.fdtd(_scene(density, axis), frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=400), spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=False, cpml_config={"memory_mode": memory})
    simulation.config.adjoint_checkpoint_stride = stride
    result = simulation.run()
    spectrum = result.monitor("probe")["data"]
    return result, spectrum, density


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("memory", ["dense", "slab"])
def test_trainable_bloch_cpml_forward_and_gradient(axis, memory):
    kwargs = dict(axis=axis, memory=memory, stride=37 if memory == "dense" else 61)
    fixed, expected, _ = _run(0.5, **kwargs)
    trained, got, density = _run(0.5, grad=True, **kwargs)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    assert any(float(value.abs().max()) > 1e-12 for name, value in vars(fixed.solver).items()
               if name.startswith("psi_h") and name.endswith("_imag") and torch.is_tensor(value) and value.numel())
    loss = got.abs().square().sum()
    loss.backward()
    derivative = float(density.grad.sum())
    h = 0.003
    _, plus, _ = _run(0.5 + h, **kwargs)
    _, minus, _ = _run(0.5 - h, **kwargs)
    finite_difference = float((plus.abs().square().sum() - minus.abs().square().sum()) / (2 * h))
    assert abs(derivative - finite_difference) <= 2e-3 * max(abs(derivative), abs(finite_difference), 1e-12)
