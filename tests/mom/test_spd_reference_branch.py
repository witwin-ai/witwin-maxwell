"""Reference node placement preserves the actual nonexact rotated material."""
import math
import pytest
import torch
import witwin.maxwell.mom.dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.matched_spd_green import spd_reference_transverse_rule
from witwin.maxwell.mom.quadrature import gauss_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_rotated_nonexact_material_point_covariance(monkeypatch):
    metric=torch.diag(torch.tensor([1.25,1.,.75],device='cuda',dtype=torch.float64))
    rotation=metric.new_tensor([[math.cos(.37),-math.sin(.37),0.],[math.sin(.37),math.cos(.37),0.],[0.,0.,1.]])
    eps=rotation@(2*metric)@rotation.T;mu=rotation@(1.3*metric)@rotation.T
    assert not torch.equal(.65*eps,mu)
    medium=TensorMedium(3e7,eps.to(torch.complex128),mu.to(torch.complex128))
    before=(medium.epsilon_r.clone(),medium.mu_r.clone())
    config=dg.DyadicIntegrationConfig(radial_map='spd_reference_branch_v1',relative_tolerance=1e-5,
        absolute_tolerance=1e-10,max_radial_order=256,max_azimuthal_order=128)
    points=metric.new_tensor([[.15,.1,.2],[.07,-.04,.11]])
    # The oracle is the ideal rotated proportional material, subject to the
    # declared floating-point rotation discrepancy. No tensor is projected.
    transform=torch.linalg.cholesky(metric)/torch.linalg.det(metric).sqrt()
    inverse=torch.linalg.inv(transform);dual=torch.block_diag(inverse,inverse).to(torch.complex128)
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    scalar=TensorMedium(3e7,2*eye,1.3*eye)
    reference,_=dg.isotropic_dyadic_green(points@inverse.T,scalar,regular_part=True)
    frame=torch.block_diag(rotation,rotation).to(torch.complex128)
    expected=frame@(dual.T@reference@dual)@frame.T
    # Inspect every actual local material entering the modal solver: the
    # physical tensors must retain the rotation discrepancy, not a proxy.
    original=dg.compile_plane_modes;seen=[]
    def inspect(q,local,*args,**kwargs):
        seen.append(not torch.equal(.65*local.epsilon_r,local.mu_r))
        return original(q,local,*args,**kwargs)
    monkeypatch.setattr(dg,'compile_plane_modes',inspect)
    actual,receipt=dg.DyadicGreenKernel(medium,config).evaluate_regular(points@rotation.T)
    assert seen and all(seen) and receipt['converged']
    assert receipt['exact_modal_branch_claimed'] is False
    for value,old in zip((medium.epsilon_r,medium.mu_r),before):torch.testing.assert_close(value,old,rtol=0,atol=0)
    errors=[]
    for i in (0,3):
        for j in (0,3):
            e=(actual[:,i:i+3,j:j+3]-expected[:,i:i+3,j:j+3]).norm(dim=(-2,-1))/expected[:,i:i+3,j:j+3].norm(dim=(-2,-1))
            assert bool(torch.isfinite(e).all()) and bool((e<.005).all());errors.extend(e.tolist())
    print({'block_errors':errors,'constitutive_projection':False,'physical_qualification':False},flush=True)


def test_general_spd_reference_measure_and_guards():
    eps=torch.diag(torch.tensor([2.,3.,4.],device='cuda',dtype=torch.complex128))
    mu=eps.new_tensor([[1.5,.1,0.],[.1,2.,.2],[0.,.2,1.1]])
    medium=TensorMedium(3e7,eps,mu)
    nodes,weights=gauss_rule(96,device='cuda',dtype=torch.float64)
    phi=torch.arange(64,device='cuda',dtype=torch.float64)*(2*math.pi/64)
    circle=torch.stack((phi.cos(),phi.sin()),-1)
    q,w=spd_reference_transverse_rule(nodes,weights,2.,.5,medium,circle)
    value=(torch.exp(-q.square().sum(-1))*w).sum()/4
    torch.testing.assert_close(value,value.new_tensor(1/(4*math.pi)),rtol=2e-12,atol=1e-14)
    for other in (TensorMedium(3e7,eps*(1+.01j),mu),TensorMedium(3e7,eps.clone().requires_grad_(),mu)):
        with pytest.raises(ValueError,match='fixed real symmetric'):
            spd_reference_transverse_rule(nodes,weights,2.,.5,other,circle)
    config=dg.DyadicIntegrationConfig(radial_map='spd_reference_branch_v1')
    assert dg.DyadicIntegrationConfig(**config.payload())==config
