"""Analytic P1 gradient controls for weak assembly without inner backward calls."""
import pytest
import torch
from test_static_tensor_green import coefficient
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_linear_analytic_gradient_matches_autograd_and_higher_sensitivities():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2
    kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]]).expand(3,-1,-1)
    points=a.new_tensor([[.24,-.12,.31],[-.21,.23,-.17],[100.,-200.,300.]],requires_grad=True)
    values,actual=kernel.panel_linear_integrals(points,tri)
    expected=torch.stack([torch.autograd.grad(values[:,i].sum(),points,create_graph=True,retain_graph=True)[0] for i in range(3)],1)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-12)
    weights=expected.detach().norm(dim=(1,2))[:,None,None]
    own=torch.autograd.grad((actual/weights).square().sum(),(points,parameter),retain_graph=True)
    oracle=torch.autograd.grad((expected/weights).square().sum(),(points,parameter))
    for x,y in zip(own,oracle):torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-9)


def test_linear_gradient_pv_no_backward_and_inference_mode(monkeypatch):
    a=coefficient();kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]])
    center=tri.mean(1).requires_grad_();mask=torch.ones(1,device=a.device,dtype=torch.bool)
    values,actual=kernel.panel_linear_integrals(center,tri,principal_value=mask)
    expected=torch.stack([torch.autograd.grad(values[:,i].sum(),center,retain_graph=True)[0] for i in range(3)],1)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)
    def forbidden(*args,**kwargs):raise AssertionError('Internal backward is forbidden')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    with torch.inference_mode():
        value,gradient=kernel.panel_linear_integrals(center.detach(),tri,principal_value=mask)
        quadratic=kernel.panel_quadratic_double_layer(center.detach(),tri,principal_value=mask)
    torch.testing.assert_close(gradient,expected,rtol=1e-10,atol=1e-12)
    assert not value.requires_grad and not gradient.requires_grad and not quadratic.requires_grad
