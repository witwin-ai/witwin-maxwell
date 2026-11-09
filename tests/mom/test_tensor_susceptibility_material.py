"""Explicit tensor susceptibility declarations and guarded compiler boundaries."""
import pytest
import torch
import witwin.maxwell as mw


def test_tensor_declaration_identity_and_no_silent_scalar_lowering():
    q=[[[float(i==j==k) for k in range(3)] for j in range(3)] for i in range(3)]
    spec=mw.TensorNonlinearSusceptibility(chi2=q)
    q[0][0][0]=9
    assert spec.chi2[0][0][0]==1
    material=mw.Material(nonlinearity=(spec,mw.NonlinearSusceptibility(chi3=.1)))
    assert material.is_nonlinear and material.has_tensor_nonlinearity
    assert material.nonlinear_chi2==0 and material.nonlinear_chi3==.1
    from witwin.maxwell.compiler.materials import _structure_nonlinearity
    from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
    with pytest.raises(NotImplementedError,match='tensor constitutive runtime'):_structure_nonlinearity(material)
    with pytest.raises(NotImplementedError,match='not integrated'):compile_harmonic_material(material,3e7,1)
    from witwin.maxwell.mom.physics import compile_medium
    with pytest.raises((ValueError,NotImplementedError)):compile_medium(material,3e7)
    with pytest.raises(ValueError):mw.TensorNonlinearSusceptibility()
    with pytest.raises(ValueError):mw.TensorNonlinearSusceptibility(chi2=[1,2,3])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_tensor_material_compilation_preserves_native_gradients_and_frame():
    q=torch.randn(3,3,3,device='cuda',dtype=torch.float64,requires_grad=True)
    angle=torch.tensor(.31,device='cuda',dtype=torch.float64,requires_grad=True)
    z=angle*0;o=z+1
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin(),z)),
        torch.stack((angle.sin(),angle.cos(),z)),torch.stack((z,z,o))))
    spec=mw.TensorNonlinearSusceptibility(chi2=q,rotation=rotation)
    compiled=spec.compile_lab_tensors(device='cuda')
    expected=torch.einsum('ia,jb,kc,abc->ijk',rotation,rotation,rotation,q)
    torch.testing.assert_close(compiled['chi2_tensor'],expected,rtol=0,atol=0)
    gradients=torch.autograd.grad(compiled['chi2_tensor'][0,1,2],(q,angle))
    assert all(bool(torch.isfinite(g).all()) and bool(g.abs().sum()>0) for g in gradients)
    bad=mw.TensorNonlinearSusceptibility(chi2=q,rotation=torch.eye(3,device='cuda')*2)
    with pytest.raises(ValueError,match='proper orthogonal'):bad.compile_lab_tensors(device='cuda')
    from witwin.maxwell.mom.harmonic_material import compile_tensor_harmonic_material
    material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,3.,4.),nonlinearity=spec)
    with pytest.raises(NotImplementedError,match='not integrated'):
        compile_tensor_harmonic_material(material,3e7,1,device='cuda')
    mutated=torch.zeros(3,3,3,device='cuda',dtype=torch.float64)
    spec=mw.TensorNonlinearSusceptibility(chi2=mutated)
    mutated.resize_(1)
    with pytest.raises(ValueError,match='changed shape'):spec.compile_lab_tensors(device='cuda')


def test_tensor_declaration_rejects_nested_tensors_and_lossy_adapter_lowering():
    q=torch.ones(3,3,3,dtype=torch.float64,requires_grad=True)
    with pytest.raises(ValueError,match='one complete native tensor'):
        mw.TensorNonlinearSusceptibility(chi2=tuple(q))
    material=mw.Material(nonlinearity=(mw.TensorNonlinearSusceptibility(chi2=q),
        mw.NonlinearSusceptibility(chi3=.1)))
    from witwin.maxwell.adapters.tidy3d import _nonlinear_spec
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('Unsupported tensor reached external scalar construction')
    with pytest.raises(NotImplementedError,match='Tensor nonlinear susceptibility export'):
        _nonlinear_spec(material,Forbidden(),1e6)
