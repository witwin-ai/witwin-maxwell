"""Checked analytic modes for exact proportional lossy tensor coefficients."""
import pytest
import torch
import witwin.maxwell.mom.dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.spectral_linalg import batched_eig4
from witwin.maxwell.mom.matched_spd_green import proportional_spd_plane_modes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_lossy_distinct_scales_use_checked_modes_and_match_generic(monkeypatch):
    metric=torch.tensor([[1.25,.125,.0625],[.125,1.,.0625],[.0625,.0625,1.]],device='cuda',dtype=torch.float64)
    medium=TensorMedium(3e7,(2+.5j)*metric,(1.5+.25j)*metric)
    q=metric.new_tensor([[0.,0.],[.2,.3],[1.8,.7],[4.,-3.],[20.,10.],[100.,-30.]])
    actual=dg.compile_plane_modes(q,medium)
    assert actual.eigensolver['version']=='analytic_proportional_spd_transformation_modes_v1'
    assert actual.eigensolver['analytic_fallback_count']==0
    with monkeypatch.context() as patch:
        patch.setattr(dg,'_plane_eigensystem',lambda g,t,m:batched_eig4(g))
        expected=dg.compile_plane_modes(q,medium)
    a,b=actual.spectrum(.01),expected.spectrum(.01)
    eta=(dg.MU_0/dg.EPSILON_0)**.5;balance=a.new_tensor([1.]*3+[eta]*3)
    difference=(a-b)*balance[:,None]*balance[None,:]
    target=b*balance[:,None]*balance[None,:]
    errors=difference.norm(dim=(-2,-1))/target.norm(dim=(-2,-1))
    assert bool(torch.isfinite(errors).all()) and bool((errors<2e-6).all())
    # A representable nonproportional perturbation must not be projected away.
    perturbed=medium.mu_r.clone();perturbed[0,0]+=1e-12
    other=TensorMedium(medium.frequency,medium.epsilon_r,perturbed)
    assert proportional_spd_plane_modes(None,q,other) is None
    trainable=TensorMedium(medium.frequency,medium.epsilon_r.clone().requires_grad_(),medium.mu_r)
    assert proportional_spd_plane_modes(None,q,trainable) is None


def test_proportional_lossy_point_kernel_matches_coordinate_reference():
    metric=torch.tensor([[1.25,.125,.0625],[.125,1.,.0625],[.0625,.0625,1.]],device='cuda',dtype=torch.float64)
    electric,magnetic=2+.5j,1.5+.25j
    medium=TensorMedium(3e7,electric*metric,magnetic*metric)
    eye=torch.eye(3,device=metric.device,dtype=torch.complex128)
    scalar=TensorMedium(medium.frequency,electric*eye,magnetic*eye)
    transform=torch.linalg.cholesky(metric)/torch.linalg.det(metric).sqrt()
    inverse=torch.linalg.inv(transform);dual=torch.block_diag(inverse,inverse).to(eye.dtype)
    points=metric.new_tensor([[.15,.1,.2],[.07,-.04,.11]])
    kernel=dg.DyadicGreenKernel(medium,dg.DyadicIntegrationConfig(
        relative_tolerance=2e-4,absolute_tolerance=1e-9,max_radial_order=256,max_azimuthal_order=128))
    actual,receipt=kernel.evaluate_regular(points)
    reference,_=dg.isotropic_dyadic_green(points@inverse.T,scalar,regular_part=True)
    expected=dual.T@reference@dual
    errors=[]
    for i in (0,3):
        for j in (0,3):
            a,b=actual[:,i:i+3,j:j+3],expected[:,i:i+3,j:j+3]
            error=(a-b).norm(dim=(-2,-1))/b.norm(dim=(-2,-1))
            assert bool(torch.isfinite(error).all()) and bool((error<.005).all())
            errors.extend(error.tolist())
    assert receipt['converged']
    assert receipt['spectral_cache']['native_eigen_fallbacks']==0
    print({'point_block_errors':errors,'spectral':receipt},flush=True)
