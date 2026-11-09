"""Explicit tensor source sampling and ordered native coefficient expansion."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_material import compile_tensor_harmonic_material,compile_harmonic_material,cell_nonlinear_coefficients

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_tensor_material_opt_in_order_zero_fill_and_gradient():
    q=torch.randn(3,3,3,device='cuda',dtype=torch.float64,requires_grad=True)
    declaration=mw.Material(eps_r=2.,nonlinearity=mw.TensorNonlinearSusceptibility(chi2=q))
    with pytest.raises(NotImplementedError,match='not integrated'):
        compile_tensor_harmonic_material(declaration,3e8,1,device='cuda')
    tensor=compile_tensor_harmonic_material(declaration,3e8,1,device='cuda',tensor_sources=True)
    scalar=compile_harmonic_material(mw.Material(eps_r=3.),3e8,1)
    assert tensor.is_nonlinear and not scalar.is_nonlinear
    # Only ordered region/cell routing is tested; these are not physical meshes.
    volumes=[SimpleNamespace(region_id=2,cells=[0,1]),SimpleNamespace(region_id=1,cells=[0])]
    sample=torch.zeros(2,3,3,device='cuda',dtype=torch.complex128)
    coefficients=cell_nonlinear_coefficients((tensor,scalar),volumes,sample)
    values=coefficients['chi2_tensor']
    assert values.shape==(3,3,3,3) and torch.count_nonzero(values[:2])==0
    torch.testing.assert_close(values[2],q,rtol=0,atol=0)
    gradient,=torch.autograd.grad(values.sum(),q)
    torch.testing.assert_close(gradient,torch.ones_like(q),rtol=0,atol=0)
    assert 'chi3_tensor' not in coefficients
    with pytest.raises(ValueError,match='real precision'):
        cell_nonlinear_coefficients((tensor,scalar),volumes,sample.to(torch.complex64))


def test_scalar_background_tensor_sources_preserve_channels_and_native_gradient():
    from witwin.maxwell.mom.physics import Medium
    from witwin.maxwell.mom.harmonic_material import StaticChannel
    q=torch.arange(27,device='cuda',dtype=torch.float64).reshape(3,3,3).requires_grad_()
    angle=torch.tensor(.23,device='cuda',dtype=torch.float64,requires_grad=True)
    zero=angle*0;one=zero+1
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin(),zero)),
        torch.stack((angle.sin(),angle.cos(),zero)),torch.stack((zero,zero,one))))
    declaration=mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.004,chi3=.002),
        mw.TensorNonlinearSusceptibility(chi2=q,rotation=rotation),
        mw.TensorNonlinearSusceptibility(chi2=q*.5)))
    with pytest.raises(NotImplementedError,match='not integrated'):
        compile_harmonic_material(declaration,3e8,2)
    sampled=compile_harmonic_material(declaration,3e8,2,tensor_sources=True,device='cuda')
    linear=compile_harmonic_material(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001),3e8,2)
    assert all(isinstance(m,Medium) for m in sampled.linear_samples)
    assert sampled.linear_samples==linear.linear_samples
    assert isinstance(sampled.electric_dc,StaticChannel)
    assert sampled.electric_dc==linear.electric_dc and sampled.magnetic_dc==linear.magnetic_dc
    assert sampled.chi2==.004 and sampled.chi3==.002 and sampled.chi3_tensor is None
    expected=torch.einsum('ia,jb,kc,abc->ijk',rotation,rotation,rotation,q)+q*.5
    torch.testing.assert_close(sampled.chi2_tensor,expected,rtol=0,atol=0)
    actual_grad=torch.autograd.grad(sampled.chi2_tensor.square().sum(),(q,angle),retain_graph=True)
    expected_grad=torch.autograd.grad(expected.square().sum(),(q,angle))
    for actual,reference in zip(actual_grad,expected_grad):
        torch.testing.assert_close(actual,reference,rtol=1e-13,atol=1e-12)
    single=compile_harmonic_material(declaration,3e8,2,tensor_sources=True,device='cuda',dtype=torch.complex64)
    assert single.chi2_tensor.dtype==torch.float32
    with pytest.raises(TypeError,match='explicit boolean'):
        compile_harmonic_material(declaration,3e8,2,tensor_sources=1)
    for kwargs in ({},{'device':'cpu'},{'device':'cuda','dtype':torch.float64}):
        with pytest.raises(ValueError,match='explicit CUDA'):
            compile_harmonic_material(declaration,3e8,2,tensor_sources=True,**kwargs)


def test_scalar_background_tensor_cubic_coefficients_and_gradient():
    c=torch.arange(81,device='cuda',dtype=torch.float64).reshape(3,3,3,3).requires_grad_()
    rotation=c.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    material=mw.Material(nonlinearity=(mw.TensorNonlinearSusceptibility(chi3=c,rotation=rotation),
        mw.TensorNonlinearSusceptibility(chi3=c*.25)))
    sampled=compile_harmonic_material(material,3e8,2,tensor_sources=True,device='cuda')
    expected=torch.einsum('ia,jb,kc,ld,abcd->ijkl',rotation,rotation,rotation,rotation,c)+c*.25
    assert sampled.chi2_tensor is None and sampled.is_nonlinear
    torch.testing.assert_close(sampled.chi3_tensor,expected,rtol=0,atol=0)
    actual,=torch.autograd.grad(sampled.chi3_tensor.square().sum(),c,retain_graph=True)
    reference,=torch.autograd.grad(expected.square().sum(),c)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
