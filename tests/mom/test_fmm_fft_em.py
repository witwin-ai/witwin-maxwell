"""Independent original PMCHWT and weak RWG controls for FFT composition."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_fft import FFTElectromagneticM2L
from witwin.maxwell.mom.fmm_expansion import multipole_to_local,multipole_moments
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.mesh import RWGBasis,sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('material',(mw.Material(),mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,sigma_m=.002)))
def test_fft_em_all_channels_dense_material_curl_adjoint_and_current_derivative(material,tmp_path):
    medium=compile_medium(material,3e8)
    d=torch.tensor([1.,.2,-.1],device='cuda',dtype=torch.float64)
    order=6;op=FFTElectromagneticM2L(d,medium,order,rhs_batch=3)
    green,gradient=multipole_to_local(d,medium.k,order)
    gradient=gradient.movedim(0,-1).contiguous()
    generator=torch.Generator(device='cuda').manual_seed(831)
    points=torch.randn((7,3),device='cuda',dtype=torch.float64,generator=generator)
    points=points/points.norm(dim=-1,keepdim=True)*.02
    charges=torch.randn((2,3,2,4,7),device='cuda',dtype=op.dtype,generator=generator)
    moments=multipole_moments(points,charges,order).movedim(-1,-2).detach().requires_grad_(True)
    actual=op.apply(moments);reference=scalar_moment_action(green,gradient,medium,moments)
    degrees=op.scalar.indices.sum(-1);weight=(.02**degrees)[:,None]
    error=float(((actual-reference)*weight).norm()/(reference*weight).norm())
    dual=torch.randn(actual.shape,device='cuda',dtype=op.dtype,generator=generator)*weight
    adjoint=op.adjoint(dual);expected_adjoint=scalar_moment_action(green,gradient,medium,dual,adjoint=True)
    adjoint_error=float(((adjoint-expected_adjoint)*weight).norm()/(expected_adjoint*weight).norm())
    derivative,=torch.autograd.grad(actual,moments,dual)
    derivative_error=float(((derivative-adjoint)*weight).norm()/(adjoint*weight).norm())
    lhs=(actual.conj()*dual).sum();rhs=(moments.conj()*adjoint).sum()
    dot=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-300))
    record=dict(forward_relative_error=error,adjoint_relative_error=adjoint_error,
        current_gradient_relative_error=derivative_error,dot_identity_relative_error=dot,receipt=op.receipt(),
        physical_qualification=False)
    (tmp_path/'em.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert error<2e-11 and adjoint_error<2e-11 and dot<1e-11 and derivative_error<1e-13


def test_fft_em_actual_rwg_all_column_original_weak_action_and_fixed_backward(tmp_path):
    def projection(center):
        basis=RWGBasis(sphere_mesh(radius=.01,center=(center,0.,0.),subdivisions=0,device='cuda',dtype=torch.float64))
        triangles=basis.mesh.vertices[basis.mesh.faces]
        support=torch.cat((triangles[basis.plus_faces],triangles[basis.minus_faces]),dim=1)
        return NestedRWGMultipoles(basis,build_support_octree(support.amin(1),support.amax(1),leaf_size=30),order=6)
    test,source=projection(-.5),projection(.5)
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,sigma_m=.002),3e8)
    op=FFTElectromagneticM2L(test.centers[0]-source.centers[0],medium,6)
    def action(value,adjoint=False):
        origin,target=(test,source) if adjoint else (source,test)
        n=origin.tree.size
        moments=origin.upward(torch.stack((value[..., :n],value[..., n:]),dim=-2))[0]
        translated=op.adjoint(moments) if adjoint else op.apply(moments)
        result=target.downward((translated,))
        return torch.cat((result[...,0,:],result[...,1,:]),dim=-1)
    n=test.tree.size;eye=torch.eye(2*n,device='cuda',dtype=op.dtype)
    actual=action(eye).mT
    reference,_=scalar_regional_block(test.basis,source.basis,torch.arange(n,device='cuda'),
        torch.arange(n,device='cuda'),(medium,),IntegrationConfig(order=6,singular_order=3))
    eta=abs(medium.eta);left=torch.tensor([eta]*n+[1.]*n,device='cuda');right=torch.tensor([1.]*n+[eta]*n,device='cuda')
    error=float((left[:,None]*(actual-reference)*right).norm()/(left[:,None]*reference*right).norm())
    x=torch.randn((3,2*n),device='cuda',dtype=op.dtype,requires_grad=True);dual=torch.randn_like(x)
    applied=action(x);adjoint=action(dual,True)
    dense_adjoint=dual@actual.conj()
    adjoint_error=float((adjoint-dense_adjoint).norm()/dense_adjoint.norm())
    derivative,=torch.autograd.grad(applied,x,dual)
    derivative_error=float((derivative-adjoint).norm()/adjoint.norm())
    lhs=(applied.conj()*dual).sum();rhs=(x.conj()*adjoint).sum()
    dot=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-300))
    record=dict(balanced_original_weak_all_column_error=error,adjoint_relative_error=adjoint_error,
        current_gradient_relative_error=derivative_error,dot_identity_relative_error=dot,
        edges=n,order=6,physical_qualification=False,full_system_qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert error<1e-8 and adjoint_error<1e-11 and derivative_error<1e-11 and dot<1e-11
    op.scalar.coefficients.add_(1)
    with pytest.raises(ValueError,match='changed'):action(x)
