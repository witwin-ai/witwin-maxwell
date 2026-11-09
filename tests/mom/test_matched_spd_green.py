"""Exact matched SPD spectral modes, branch measure and refusal controls."""
import math
import pytest
import torch
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.matched_spd_green import matched_spd_plane_modes,matched_spd_transverse_rule
from witwin.maxwell.mom.quadrature import gauss_rule
from witwin.maxwell.mom.spectral_linalg import batched_eig4
import witwin.maxwell.mom.dyadic_green as dg

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def medium():
    a=torch.tensor([[1.2,.12,.05],[.12,1.1,.07],[.05,.07,1.05]],device='cuda',dtype=torch.complex128)
    return TensorMedium(3e7,a,a)


def test_modes_match_generic_spectrum_and_preserve_grazing_rejection(monkeypatch):
    m=medium();q=m.epsilon_r.real.new_tensor([[0.,0.],[.2,.3],[1.8,.7],[4.,-3.],[20.,10.],[100.,-30.],[1000.,200.]])
    actual=dg.compile_plane_modes(q,m)
    assert actual.eigensolver['version']=='analytic_matched_spd_transformation_modes_v1'
    assert actual.eigensolver['analytic_fallback_count']==0
    with monkeypatch.context() as patch:
        patch.setattr(dg,'_plane_eigensystem',lambda g,t,m:batched_eig4(g))
        expected=dg.compile_plane_modes(q,m)
    a,b=actual.spectrum(1e-6),expected.spectrum(1e-6)
    eta=(dg.MU_0/dg.EPSILON_0)**.5;balance=a.new_tensor([1.]*3+[eta]*3)
    error=((a-b)*balance[:,None]*balance[None,:]).norm(dim=(-2,-1))
    magnitude=(b*balance[:,None]*balance[None,:]).norm(dim=(-2,-1))
    assert bool((error<2e-6*magnitude).all())
    coefficient=torch.diag(m.epsilon_r.new_tensor([4.,9.,16.]))
    grazing=TensorMedium(m.frequency,coefficient,coefficient)
    with pytest.raises(ValueError,match='grazing|sheets'):
        dg.compile_plane_modes(q.new_tensor([[12.,0.]]),grazing,pole_relative_tolerance=1e-10)


def test_branch_measure_policy_and_unsupported_mode_guards():
    m=medium();eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    nodes,weights=gauss_rule(64,device='cuda',dtype=torch.float64)
    phi=torch.arange(64,device='cuda',dtype=torch.float64)*(2*math.pi/64)
    circle=torch.stack((phi.cos(),phi.sin()),-1)
    k0=2.;points,measure=matched_spd_transverse_rule(nodes,weights,k0,.5,m,circle)
    integral=(torch.exp(-points.square().sum(-1))*measure).sum()/k0**2
    torch.testing.assert_close(integral,integral.new_tensor(1/(4*math.pi)),rtol=2e-12,atol=1e-14)
    # Unsupported samples must return before reading a dummy generator.
    q=nodes.new_zeros(1,2);generator=eye.new_zeros(1,4,4)
    for sample in (TensorMedium(m.frequency,m.epsilon_r,eye),
                   TensorMedium(m.frequency,m.epsilon_r.clone().requires_grad_(),m.mu_r),
                   TensorMedium(m.frequency,1e110*eye,1e110*eye)):
        assert matched_spd_plane_modes(generator,q,sample) is None
        with pytest.raises(ValueError,match='Matched-SPD'):
            matched_spd_transverse_rule(nodes,weights,k0,.5,sample,circle)
    config=dg.DyadicIntegrationConfig(radial_map='matched_spd_branch_v1')
    assert config.payload()['radial_map']=='matched_spd_branch_v1'
    assert dg.DyadicIntegrationConfig(**config.payload())==config


def test_branch_public_refusal_cache_isolation_and_periodic_roundtrip():
    from dataclasses import replace
    from test_periodic_dyadic_persistence import problem
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicRegularKernel
    m=medium();points=m.epsilon_r.real.new_tensor([[.01,.02,.2]])
    cache=dg.SpectralModeCache(m)
    ordinary=dg.integrate_spatial_green(points,m,radial_order=4,azimuthal_order=4,mode_cache=cache)
    misses=cache.misses
    branch=dg.integrate_spatial_green(points,m,radial_order=4,azimuthal_order=4,mode_cache=cache,
        radial_map='matched_spd_branch_v1')
    assert cache.misses>misses
    misses=cache.misses
    repeated=dg.integrate_spatial_green(points,m,radial_order=4,azimuthal_order=4,mode_cache=cache,
        radial_map='matched_spd_branch_v1')
    assert cache.misses==misses
    torch.testing.assert_close(repeated,branch,rtol=0,atol=0)
    assert not torch.equal(ordinary,branch)
    config=dg.DyadicIntegrationConfig(radial_order=2,azimuthal_order=2,
        max_radial_order=4,max_azimuthal_order=4,radial_map='matched_spd_branch_v1')
    with pytest.raises(dg.DyadicConvergenceError,match='single regional'):
        dg.evaluate_spatial_green(points,m,config,_exterior=m)
    trainable=TensorMedium(m.frequency,m.epsilon_r.clone().requires_grad_(),m.mu_r)
    with pytest.raises(dg.DyadicConvergenceError,match='fixed exact matched'):
        dg.evaluate_spatial_green(points,trainable,config)
    original=problem(2)
    original=replace(original,kernel=replace(original.kernel,medium=m),
        integration=replace(original.integration,radial_map='matched_spd_branch_v1'))
    payload=original.payload()
    restored=QuasiperiodicDyadicRegularKernel.from_payload(payload,medium=m,lattice=original.lattice)
    assert restored.payload()==payload
    assert restored.integration.radial_map=='matched_spd_branch_v1'
