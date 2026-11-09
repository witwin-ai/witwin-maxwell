"""Full fixed-horizon foil-to-Huygens radiation derivatives on a fixed mesh."""
import math

import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


class RadiatingFoil(mw.SceneModule):
    def __init__(self, sample_spacing=.006):
        super().__init__()
        self.sample_spacing = sample_spacing
        self.log_thickness = torch.nn.Parameter(torch.tensor(math.log(1e-6), device='cuda', dtype=torch.float64))
        self.log_conductivity = torch.nn.Parameter(torch.tensor(math.log(1e6), device='cuda', dtype=torch.float64))

    def to_scene(self):
        return mw.Scene(domain=mw.Domain(bounds=((-0.05, .05),)*3), grid=mw.GridSpec.uniform(.005),
            boundary=mw.BoundarySpec.pml(num_layers=4), device='cuda',
            structures=[mw.Structure(mw.Box(size=(0., .02, .02)), mw.CopperFoil(
                thickness=self.log_thickness.exp(), conductivity=self.log_conductivity.exp(),
                frequency_range=(2e9, 4e9), mode_count=16), name='foil')],
            ports=[mw.LumpedPort('feed', positive=(-.01, 0., .005), negative=(-.01, 0., -.005),
                voltage_path=mw.AxisPath('z'), current_surface=mw.Box(position=(-.01, 0., -.0025), size=(.015, .015, 0.)))],
            monitors=[mw.ClosedSurfaceMonitor.box('radiation', size=(.045, .045, .045), sample_spacing=self.sample_spacing)])


def objective(model, graph=False, stride=7):
    simulation = mw.Simulation.fdtd(model, frequency=3e9, cuda_graph=graph, full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=96, transient_cycles=0, steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window='none', normalize_source=False),
        excitations=mw.PortExcitation('feed', source_time=mw.CW(3e9)))
    simulation.config.adjoint_checkpoint_stride = stride
    result = simulation.run()
    antenna = result.antenna(surface='radiation', driven_port='feed', theta_points=31, phi_points=25, radius=10.)
    return antenna.p_rad.sum(), result


@pytest.mark.parametrize('sample_spacing', (None, .006))
def test_foil_to_radiated_power_gradient_matches_normal_forward_step_scan(sample_spacing, record_property):
    model = RadiatingFoil(sample_spacing)
    value, _ = objective(model)
    value.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        saved = parameter.detach().clone()
        errors = []
        for step in (.08, .04, .02, .01):
            with torch.no_grad():
                parameter.copy_(saved+step)
                plus = float(objective(model)[0])
                parameter.copy_(saved-step)
                minus = float(objective(model)[0])
                parameter.copy_(saved)
            finite = (plus-minus)/(2*step)
            errors.append(abs(finite-float(gradients[name]))/abs(float(gradients[name])))
        record_property(name+'_radiation_relative_errors', repr(errors))
        assert gradients[name].isfinite() and abs(float(gradients[name])) > 1e-30
        assert sum(error < .01 for error in errors) >= 2, (name, gradients[name], errors)
    normal, _ = objective(model)
    torch.testing.assert_close(normal, value.detach(), rtol=2e-4, atol=1e-30)
    model.requires_grad_(True)
    for stride in (7, 13):
        model.zero_grad(set_to_none=True)
        captured, result = objective(model, True, stride)
        captured.backward()
        assert result.stats()['cuda_graph_active']
        torch.testing.assert_close(captured.detach(), value.detach(), rtol=2e-4, atol=1e-30)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=.001, atol=1e-30)
