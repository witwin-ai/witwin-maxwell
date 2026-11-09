"""Positive proportional SPD branch measure and independent point reference."""
import math
import pytest
import torch
import witwin.maxwell.mom.dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.matched_spd_green import proportional_spd_transverse_rule
from witwin.maxwell.mom.quadrature import gauss_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_proportional_branch_measure_reference_and_refusal():
    metric=torch.diag(torch.tensor([1.25,1.,.75],device='cuda',dtype=torch.float64))
    medium=TensorMedium(3e7,(2*metric).to(torch.complex128),(1.3*metric).to(torch.complex128))
    nodes,weights=gauss_rule(96,device='cuda',dtype=torch.float64)
    phi=torch.arange(64,device='cuda',dtype=torch.float64)*(2*math.pi/64)
    circle=torch.stack((phi.cos(),phi.sin()),-1)
    q,w=proportional_spd_transverse_rule(nodes,weights,2.,.5,medium,circle)
    integral=(torch.exp(-q.square().sum(-1))*w).sum()/4
    torch.testing.assert_close(integral,integral.new_tensor(1/(4*math.pi)),rtol=2e-12,atol=1e-14)
    altered=medium.mu_r.clone();altered[0,0]+=1e-12
    for other in (TensorMedium(medium.frequency,medium.epsilon_r,altered),
                  TensorMedium(medium.frequency,medium.epsilon_r*(1+.001j),medium.mu_r),
                  TensorMedium(medium.frequency,medium.epsilon_r.clone().requires_grad_(),medium.mu_r)):
        with pytest.raises(ValueError,match='Proportional-SPD'):
            proportional_spd_transverse_rule(nodes,weights,2.,.5,other,circle)
    transform=torch.linalg.cholesky(metric)/torch.linalg.det(metric).sqrt()
    inverse=torch.linalg.inv(transform);dual=torch.block_diag(inverse,inverse).to(torch.complex128)
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    scalar=TensorMedium(medium.frequency,2*eye,1.3*eye)
    points=metric.new_tensor([[.15,.1,.2],[.07,-.04,.11]])
    config=dg.DyadicIntegrationConfig(radial_map='proportional_spd_branch_v1',
        relative_tolerance=1e-5,max_radial_order=256,max_azimuthal_order=128)
    actual,receipt=dg.DyadicGreenKernel(medium,config).evaluate_regular(points)
    reference,_=dg.isotropic_dyadic_green(points@inverse.T,scalar,regular_part=True)
    expected=dual.T@reference@dual
    errors=[]
    for i in (0,3):
        for j in (0,3):
            error=(actual[:,i:i+3,j:j+3]-expected[:,i:i+3,j:j+3]).norm(dim=(-2,-1))/expected[:,i:i+3,j:j+3].norm(dim=(-2,-1))
            assert bool(torch.isfinite(error).all()) and bool((error<.005).all())
            errors.extend(error.tolist())
    assert receipt['converged'] and receipt['kernel_version']=='dyadic_weyl_proportional_spd_branch_v1'
    assert dg.DyadicIntegrationConfig(**config.payload())==config
    print({'point_block_errors':errors,'physical_qualification':False},flush=True)


@pytest.mark.parametrize('policy',['proportional_spd_branch_v1','spd_reference_branch_v1'])
def test_proportional_policy_cache_and_periodic_payload(policy):
    from dataclasses import replace
    from test_periodic_dyadic_persistence import problem
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicRegularKernel
    metric=torch.diag(torch.tensor([1.25,1.,.75],device='cuda',dtype=torch.complex128))
    medium=TensorMedium(3e7,2*metric,1.3*metric)
    points=metric.real.new_tensor([[.01,.02,.2]])
    cache=dg.SpectralModeCache(medium)
    ordinary=dg.integrate_spatial_green(points,medium,radial_order=4,azimuthal_order=4,mode_cache=cache)
    misses=cache.misses
    branch=dg.integrate_spatial_green(points,medium,radial_order=4,azimuthal_order=4,mode_cache=cache,radial_map=policy)
    assert cache.misses>misses
    misses=cache.misses
    repeated=dg.integrate_spatial_green(points,medium,radial_order=4,azimuthal_order=4,mode_cache=cache,radial_map=policy)
    assert cache.misses==misses and not torch.equal(branch,ordinary)
    torch.testing.assert_close(repeated,branch,rtol=0,atol=0)
    original=problem(2)
    original=replace(original,kernel=replace(original.kernel,medium=medium),
        integration=replace(original.integration,radial_map=policy))
    restored=QuasiperiodicDyadicRegularKernel.from_payload(original.payload(),medium=medium,lattice=original.lattice)
    assert restored.payload()==original.payload()
