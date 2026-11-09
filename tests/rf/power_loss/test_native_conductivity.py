"""Independent staggered-interface work controls for automatic bulk loss."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.postprocess.power_loss import compute_power_loss_data
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('normal', range(3))
def test_native_thin_slab_loss_and_sigma_derivative(normal):
    sigma = torch.tensor(.3, requires_grad=True)
    size = [.04]*3
    size[normal] = .002
    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=((-0.004, .004),)*3),
        grid=mw.GridSpec.uniform(.004), device='cpu', boundary=mw.BoundarySpec.none(),
        structures=[mw.Structure(mw.Box(size=size), mw.Material(eps_r=2., sigma_e=sigma))]))
    monitor = mw.PowerLossMonitor('loss', position=(0.,)*3, size=(.008,)*3, frequencies=(1e9,))
    compiled = compile_power_loss_monitor(scene, monitor)
    material = scene.compile_materials()
    fields = {}
    expected = sigma*0
    for axis, component in enumerate(('Ex', 'Ey', 'Ez')):
        # Integrate the conductivity used by the field update. A thin slab's
        # subpixel average differs from its profile at the native edge centre.
        mask = compiled.component_masks[component]
        native_sigma = material['edge_components']['sigma_e'][component]
        torch.testing.assert_close(compiled.conductivity[component], native_sigma[mask])
        amplitude = axis+1
        fields[component] = torch.full((1, *mask.shape), complex(amplitude))
        expected = expected + .5*(native_sigma[mask]*amplitude**2*compiled.component_volumes[component]).sum()
    actual = compute_power_loss_data(compiled, torch.tensor([1e9]), electric_fields=fields).conduction[0]
    torch.testing.assert_close(actual, expected)
    (gradient,) = torch.autograd.grad(actual, sigma, retain_graph=True)
    # Differentiate the independent compiled-coefficient integral as well:
    # dividing rounded FP32 forward sums by sigma is not their exact pullback.
    (expected_gradient,) = torch.autograd.grad(expected, sigma)
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_loss_conductivity_matches_native_field_decay():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.012, .012),)*3),
        grid=mw.GridSpec.uniform(.004), device='cuda', boundary=mw.BoundarySpec.none(),
        structures=[mw.Structure(mw.Box(size=(.002, .04, .04)), mw.Material(eps_r=2., sigma_e=.3))])
    solver = mw.Simulation.fdtd(scene, frequency=1e9, full_field_dft=False).prepare().solver
    monitor = mw.PowerLossMonitor('loss', position=(0.,)*3, size=(.016,)*3, frequencies=(1e9,))
    compiled = compile_power_loss_monitor(solver.scene, monitor)
    for component in ('Ex', 'Ey', 'Ez'):
        decay = getattr(solver, 'c'+component.lower()+'_decay')
        curl = getattr(solver, 'c'+component.lower()+'_curl')
        effective_sigma = (1-decay)/curl
        torch.testing.assert_close(compiled.conductivity[component],
            effective_sigma[compiled.component_masks[component]], rtol=2e-5, atol=1e-6)
