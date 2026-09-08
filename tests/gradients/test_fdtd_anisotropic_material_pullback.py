"""End-to-end tensor-interface density gradients at a fixed discrete time step."""
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _TensorInterfaceDesign(mw.SceneModule):
    def __init__(self, interface):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.1, device="cuda"))
        self.interface = interface
        self.register_buffer("direction", torch.tensor([1., -.4, .2, .5, -.3, .6, .1, -.2], device="cuda").reshape(2, 2, 2))

    def to_scene(self):
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.6, .6),) * 3),
            grid=mw.GridSpec.uniform(.12), boundary=mw.BoundarySpec.pml(num_layers=2), device="cuda")
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0., 0., 0.), size=(.36, .36, .12)),
            material=mw.Material(epsilon_tensor=mw.Tensor3x3(((5., 1.5, .8), (1.5, 4., .7), (.8, .7, 3.))))))
        # The interface variant places the design boundary through the tensor's
        # central node, making its off-diagonal inverse depend on the density.
        scene.add_material_region(mw.MaterialRegion(name="design",
            geometry=mw.Box(position=(.06, .06, .12 if self.interface else .30), size=(.24, .24, .24)),
            density=.5 + self.theta * self.direction, eps_bounds=(1., 6.)))
        scene.add_source(mw.PointDipole(position=(0., 0., -.24), polarization="Ez", width=.04,
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=.25e9, amplitude=50.)))
        scene.add_monitor(mw.PointMonitor("probe", (0., 0., .24), fields=("Ez",)))
        return scene


@pytest.mark.parametrize("interface", [False, True])
@pytest.mark.parametrize("stride,delta", [(1, .01), (7, .005), (31, .02)])
def test_anisotropic_density_gradient_matches_fixed_step_finite_difference(interface, stride, delta):
    model = _TensorInterfaceDesign(interface)
    simulation = mw.Simulation.fdtd(model, frequency=1e9,
        run_time=mw.TimeConfig(time_steps=200, dt=1.8e-11),
        spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=False, cuda_graph=False)
    simulation.config.adjoint_checkpoint_stride = stride

    def loss():
        field = simulation.run().monitor("probe")["data"]
        return field.abs().square().sum()

    baseline = model.theta.detach().clone()
    loss().backward()
    analytic = model.theta.grad.detach().clone()
    with torch.no_grad():
        model.theta.copy_(baseline + delta)
    upper = loss().detach()
    with torch.no_grad():
        model.theta.copy_(baseline - delta)
    lower = loss().detach()
    with torch.no_grad():
        model.theta.copy_(baseline)
    numerical = (upper - lower) / (2 * delta)
    assert float(numerical.abs()) > 1e-4
    torch.testing.assert_close(analytic, numerical, rtol=.015, atol=1e-3)
