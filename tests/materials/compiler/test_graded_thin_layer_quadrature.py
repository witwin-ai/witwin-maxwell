"""Independent physical-volume controls for thin layers on graded Yee grids."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _axis_sample_offsets, _edge_axis_offsets
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('helper', (_axis_sample_offsets, _edge_axis_offsets))
def test_thin_layer_integral_converges_to_physical_thickness(axis, helper):
    thickness = 50e-6
    nodes = np.array([-.0024, 0., thickness, .00085])
    primal = np.diff(nodes)
    dual = np.r_[primal[0], (primal[:-1]+primal[1:])/2, primal[-1]]
    scene = SimpleNamespace(grid=SimpleNamespace(is_custom=True, is_auto=False), device='cpu')
    setattr(scene, axis, torch.tensor(nodes, dtype=torch.float32))
    setattr(scene, 'd'+axis+'_primal64', primal)
    setattr(scene, 'd'+axis+'_dual64', dual)
    offsets = helper(scene, axis, 4096, *(() if helper is _axis_sample_offsets else (False,)))
    coordinates = getattr(scene, axis)
    points = torch.stack([coordinates+offset for offset in offsets])
    fill = ((points >= 0) & (points <= thickness)).double().mean(0)
    measured = float((fill*torch.tensor(dual)).sum())
    assert measured == pytest.approx(thickness, rel=.01)
    # The old symmetric windows count the entire film at both interface nodes.
    centered = torch.tensor((np.arange(4096)+.5)/4096-.5)[:, None]*torch.tensor(dual)
    old_fill = ((coordinates+centered >= 0) & (coordinates+centered <= thickness)).double().mean(0)
    assert float((old_fill*torch.tensor(dual)).sum()) == pytest.approx(2*thickness, rel=.01)
    assert helper(scene, axis, 1, *(() if helper is _axis_sample_offsets else (False,))) == [0.]


@pytest.mark.parametrize('axis', 'xyz')
def test_compiled_tangential_conductivity_preserves_graded_layer_integral(axis):
    thickness = 50e-6
    sigma = torch.tensor(.1, requires_grad=True)
    axes = [np.linspace(-.002, .002, 5)]*3
    normal = 'xyz'.index(axis)
    axes[normal] = np.array([-.0024, 0., thickness, .00085])
    position, size = [0.]*3, [.1]*3
    position[normal], size[normal] = thickness/2, thickness
    samples = [1]*3
    samples[normal] = 256
    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=[(float(a[0]), float(a[-1])) for a in axes]),
        grid=mw.GridSpec.custom(*axes), device='cpu', subpixel_samples=mw.SubpixelSpec(samples=tuple(samples), averaging="arithmetic"),
        structures=[mw.Structure(mw.Box(position=torch.tensor(position, dtype=torch.float64),
                                       size=torch.tensor(size, dtype=torch.float64)), mw.Material(eps_r=3., sigma_e=sigma))]))
    model = scene.compile_materials()
    component = 'E'+'xyz'[(normal+1)%3]
    values = model['edge_components']['sigma_e'][component]
    selection = [1]*3
    selection[normal] = slice(None)
    profile = values[tuple(selection)].double()
    widths = torch.tensor(getattr(scene, 'd'+axis+'_dual64'))
    integral = (profile*widths).sum()
    assert float(integral.detach()) == pytest.approx(float(sigma.detach())*thickness, rel=.03)
    (derivative,) = torch.autograd.grad(integral, sigma)
    assert float(derivative) == pytest.approx(thickness, rel=.03)
