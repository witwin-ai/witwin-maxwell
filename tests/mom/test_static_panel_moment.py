"""Independent quadrature and derivative controls for Coulomb affine moments."""
import math
import pytest
import torch
from witwin.maxwell.mom.fields import (_static_panel_gradient, _static_panel_potential,
                                       _static_panel_vector_moment)
from witwin.maxwell.mom.quadrature import triangle_rule, duffy_fan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('side', [-1, 0, 1])
@pytest.mark.parametrize('reverse', [False, True])
def test_affine_static_moment_matches_independent_surface_cubature(side, reverse):
    triangle = torch.tensor([[[.1,.2,.3],[.8,.25,.4],[.2,.9,.5]]],device='cuda',dtype=torch.float64)
    if reverse:
        triangle = triangle[:, [0,2,1]]
    on_panel = side == 0
    normal = torch.linalg.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
    normal = normal/normal.norm(dim=-1,keepdim=True)
    observer = triangle.mean(1) if on_panel else triangle.mean(1)+side*.6*normal
    observer = observer.detach().requires_grad_()
    gradient = _static_panel_gradient(observer,triangle,
        principal_value=torch.tensor([True],device='cuda') if on_panel else None)
    potential = _static_panel_potential(observer,triangle,normal,gradient)
    moment = _static_panel_vector_moment(observer,triangle,normal,potential)
    nodes,weights = duffy_fan(observer,triangle,64) if on_panel else triangle_rule(triangle,64)
    delta = nodes-observer[:,None]
    scalar = weights/(4*math.pi*delta.norm(dim=-1))
    torch.testing.assert_close(potential,scalar.sum(-1),rtol=2e-12,atol=1e-14)
    torch.testing.assert_close(moment,(delta*scalar[...,None]).sum(1),rtol=2e-12,atol=1e-14)
    if not on_panel:
        derivative, = torch.autograd.grad(potential.sum(),observer,retain_graph=True)
        torch.testing.assert_close(derivative,gradient,rtol=2e-12,atol=1e-14)
        actual, = torch.autograd.grad(moment.square().sum(),observer,retain_graph=True)
        expected, = torch.autograd.grad((delta*scalar[...,None]).sum(1).square().sum(),observer)
        torch.testing.assert_close(actual,expected,rtol=2e-11,atol=1e-14)
