"""Independent static PDE/flux and physical-panel quadrature controls."""
import math
import pytest
import torch
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.fields import _static_panel_gradient,_static_panel_potential
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def coefficient():
    return torch.tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]],device='cuda',dtype=torch.float64)


def test_point_pde_flux_rotation_and_scalar_limit():
    a=coefficient();kernel=StaticTensorGreen(a)
    x=a.new_tensor([[.7,-.4,1.2],[-.3,.9,.5]])
    for point in x:
        hessian=torch.autograd.functional.hessian(lambda p:kernel.evaluate(p[None])[0][0],point)
        assert abs(float((a*hessian).sum()))<2e-15
    # Integrate physical conormal flux on a unit sphere, independent of the
    # coordinate-transformed ellipsoid used in the construction derivation.
    n=16384;i=torch.arange(n,device=a.device,dtype=a.dtype)
    z=1-2*(i+.5)/n;phi=i*(math.pi*(3-math.sqrt(5)))
    normals=torch.stack(((1-z*z).sqrt()*phi.cos(),(1-z*z).sqrt()*phi.sin(),z),1)
    _,gradient=kernel.evaluate(normals)
    flux=(normals*(gradient@a.T)).sum(1).mean()*4*math.pi
    assert abs(float(flux)+1)<2e-6
    q,_=torch.linalg.qr(a.new_tensor([[1.,2.,3.],[0.,1.,2.],[2.,1.,1.]]))
    rotated=q@a@q.T;rotated=(rotated+rotated.T)/2
    value,gradient=kernel.evaluate(x)
    other=StaticTensorGreen(rotated).evaluate(x@q.T)
    torch.testing.assert_close(other[0],value,rtol=2e-14,atol=1e-15)
    torch.testing.assert_close(other[1],gradient@q.T,rtol=2e-14,atol=1e-15)
    scalar=StaticTensorGreen(3*torch.eye(3,device=a.device,dtype=a.dtype))
    value,gradient=scalar.evaluate(x);radius=x.norm(dim=-1)
    torch.testing.assert_close(value,1/(12*math.pi*radius),rtol=2e-15,atol=0)
    torch.testing.assert_close(gradient,-x/(12*math.pi*radius[:,None]**3),rtol=2e-15,atol=0)


def test_physical_triangle_quadrature_scalar_limit_and_conormal_jump():
    a=coefficient();kernel=StaticTensorGreen(a)
    panels=a.new_tensor([[[0.,0.,0.],[1.,0.,0.],[.1,.8,0.]]])
    x=a.new_tensor([[.2,.3,1.1]])
    potential,gradient=kernel.panel_integrals(x,panels)
    nodes,weights=triangle_rule(panels,24)
    delta=x[:,None]-nodes
    inverse=torch.linalg.inv(a);metric=delta@inverse
    rho2=(metric*delta).sum(-1)
    value=1/(4*math.pi*torch.linalg.det(a).sqrt()*rho2.sqrt())
    # Direct physical-coordinate quadrature, without transforming the panel.
    expected=(value*weights).sum(-1)
    expected_gradient=(-metric*(value/rho2)[...,None]*weights[...,None]).sum(1)
    torch.testing.assert_close(potential,expected,rtol=2e-12,atol=1e-14)
    torch.testing.assert_close(gradient,expected_gradient,rtol=2e-12,atol=1e-14)
    normal=a.new_tensor([[0.,0.,1.]])
    g=_static_panel_gradient(x,panels)
    v=_static_panel_potential(x,panels,normal,g)
    sv,sg=StaticTensorGreen(3*torch.eye(3,device=a.device,dtype=a.dtype)).panel_integrals(x,panels)
    torch.testing.assert_close(sv,v/3,rtol=1e-13,atol=1e-15)
    torch.testing.assert_close(sg,g/3,rtol=1e-13,atol=1e-15)
    center=panels.mean(1)
    pv=kernel.panel_integrals(center,panels,principal_value=torch.ones(1,device=a.device,dtype=torch.bool))[1]
    plus=kernel.panel_integrals(center+1e-7*normal,panels)[1]
    minus=kernel.panel_integrals(center-1e-7*normal,panels)[1]
    jump=((plus-minus)@a.T*normal).sum()
    assert abs(float(jump)+1)<2e-6
    torch.testing.assert_close((plus+minus)/2,pv,rtol=1e-6,atol=1e-8)


def test_native_material_query_gradients_and_admission():
    a=coefficient();x=a.new_tensor([[.2,.3,1.1]])
    panels=a.new_tensor([[[0.,0.,0.],[1.,0.,0.],[.1,.8,0.]]])
    def function(raw,points):
        kernel=StaticTensorGreen((raw+raw.T)/2)
        value,gradient=kernel.evaluate(points)
        panel=kernel.panel_integrals(points,panels)
        return torch.cat((value,gradient.flatten(),panel[0],panel[1].flatten()))
    assert torch.autograd.gradcheck(function,(a.clone().requires_grad_(),x.clone().requires_grad_()),eps=1e-6,atol=1e-7,rtol=1e-5)
    for bad in (a.to(torch.complex128),a.cpu(),-a,a+torch.triu(torch.ones_like(a),1)):
        with pytest.raises(ValueError):StaticTensorGreen(bad)
    kernel=StaticTensorGreen(a)
    with pytest.raises(ValueError,match='singularity'):kernel.evaluate(torch.zeros_like(x))
    with pytest.raises(ValueError,match='principal'):kernel.panel_integrals(panels.mean(1),panels)
    with pytest.raises(ValueError,match='principal'):kernel.panel_integrals(x,panels,principal_value=torch.ones(1,device=a.device,dtype=torch.bool))
    with pytest.raises(ValueError,match='edges'):kernel.panel_integrals(panels[:,0],panels,principal_value=torch.ones(1,device=a.device,dtype=torch.bool))
    kernel.coefficient.add_(torch.eye(3,device=a.device,dtype=a.dtype))
    with pytest.raises(ValueError,match='changed'):kernel.evaluate(x)


def test_finite_extreme_coordinates_do_not_silently_become_zero_fields():
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    kernel=StaticTensorGreen(eye)
    x=eye.new_tensor([[1e200,0.,0.],[1e154,0.,0.]])
    value,gradient=kernel.evaluate(x)
    expected=(1/(4*math.pi))/x[:,0]
    assert bool((value>0).all())
    torch.testing.assert_close(value,expected,rtol=2e-15,atol=0.)
    expected_gradient=-expected/x[:,0]
    assert gradient[1,0]!=0
    torch.testing.assert_close(gradient[:,0],expected_gradient,rtol=1e-14,atol=1e-322)
    small=StaticTensorGreen(eye*1e-200)
    panels=eye.new_tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    with pytest.raises(ValueError,match='representable'):
        small.panel_integrals(eye.new_tensor([[.2,.2,1.]]),panels)
