"""Packed-real tangent products match a dense reference without constructing it."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_jacobian_action import HarmonicJacobianAction

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_native_tangent_transpose_and_product_gradients(monkeypatch):
    model=runpy.run_path('tests/mom/test_harmonic_balance.py')['model'](harmonics=2,chi2=.2,chi3=.3)
    field=model.background.clone();field[0]=.1;field[1]+=.2j;field[2]=.05-.1j
    action=HarmonicJacobianAction(model,field,pump_scale=.7)
    point=model._pack(field)
    equation=lambda v:model._pack(model.residual(model._unpack(v),pump_scale=.7))
    reference=torch.autograd.functional.jacobian(equation,point)
    v=torch.randn(2,3,len(point),device='cuda',dtype=point.dtype,requires_grad=True)
    def forbidden(*args,**kwargs):raise AssertionError('Tangent products must not construct a global Jacobian')
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd.functional,'jacobian',forbidden)
        a=action.apply(v);b=action.adjoint(v)
        torch.testing.assert_close(a,v@reference.mT,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(b,v@reference,rtol=1e-12,atol=1e-12)
        gradient,=torch.autograd.grad(a.square().sum(),v,create_graph=True)
        expected=2*(v@reference.mT)@reference
        torch.testing.assert_close(gradient,expected,rtol=1e-12,atol=1e-12)
        second,=torch.autograd.grad(gradient.sum(),v)
        torch.testing.assert_close(second,2*torch.ones_like(v)@reference.mT@reference,rtol=1e-12,atol=1e-12)
    assert action.apply(v[:0]).shape==v[:0].shape
    with pytest.raises(ValueError,match='packed-real'):action.apply(v.to(torch.complex128))
    delayed=action.apply(v)
    with torch.no_grad():action.vector[0]+=1
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(delayed.sum(),v)


@pytest.mark.parametrize('kind,backend',[('current','hmatrix'),('mixed','h2'),('dual','h2')])
def test_compressed_tangent_finite_difference_and_transpose(kind,backend):
    scene,config=runpy.run_path('tests/mom/test_compressed_hb_public.py')['setup'](kind,backend)
    prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    model=prepared.solver.model
    check_compressed_tangent(model,model.background)
    check_inverse=runpy.run_path('tests/mom/test_harmonic_tangent_inverse.py')['check_compressed_inverse']
    check_inverse(model,model.background)


def check_compressed_tangent(model,field):
    """Also usable with restored stage evidence, without rerunning roots."""
    tangent=HarmonicJacobianAction(model,field)
    point=model._pack(field)
    generator=torch.Generator(device=point.device).manual_seed(41)
    v=torch.randn(point.shape,device=point.device,dtype=point.dtype,generator=generator)
    w=torch.randn(point.shape,device=point.device,dtype=point.dtype,generator=generator)
    jv=tangent.apply(v);jtw=tangent.adjoint(w)
    torch.testing.assert_close(w@jv,v@jtw,rtol=1e-9,atol=1e-10)
    step=1e-5
    equation=lambda x:model._pack(model.residual(model._unpack(x)))
    difference=(equation(point+step*v)-equation(point-step*v))/(2*step)
    torch.testing.assert_close(jv,difference,rtol=2e-6,atol=1e-8)
    vector=v.detach().requires_grad_(True)
    gradient,=torch.autograd.grad((tangent.apply(vector)*w).sum(),vector)
    torch.testing.assert_close(gradient,jtw,rtol=1e-12,atol=1e-12)
