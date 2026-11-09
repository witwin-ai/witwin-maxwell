"""Linear complex material admission preserves passive constitutive parameters."""
import json
import math

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.physics import compile_medium, validate_scene


@pytest.mark.parametrize('epsilon,mu',[(2.+.05j,1.3+.02j),(-2.+.05j,1.3+.02j),(2.+0j,1.3+0j)])
def test_no_absorption_never_samples_a_real_host_index(epsilon,mu,monkeypatch):
    import witwin.maxwell.media as media
    material=mw.Material(eps_r=epsilon,mu_r=mu)
    def forbidden(*args,**kwargs):
        raise AssertionError('Linear material sampled a nonlinear host index')
    monkeypatch.setattr(media.np,'sqrt',forbidden)
    assert material.tpa_sigma_scale==0. and not material.is_nonlinear
    medium=compile_medium(material,3e8)
    assert medium.epsilon_r==epsilon and medium.mu_r==mu
    assert medium.k.imag>=0 and medium.eta.real>=0
    assert medium.k**2==pytest.approx(medium.omega**2*medium.epsilon*medium.mu)


def test_complex_loss_and_both_conductivities_are_counted_once():
    epsilon,mu=2.+.05j,1.3+.02j
    material=mw.Material(eps_r=epsilon,mu_r=mu,sigma_e=.002,sigma_m=.003)
    frequency=3e8;medium=compile_medium(material,frequency)
    assert medium.epsilon_r==pytest.approx(epsilon+1j*.002/(2*math.pi*frequency*EPSILON_0))
    assert medium.mu_r==pytest.approx(mu+1j*.003/(2*math.pi*frequency*MU_0))
    assert medium.metadata['constitutive_parameters']['eps_base']==epsilon
    assert medium.metadata['constitutive_parameters']['mu_base']==mu
    assert medium.eta==pytest.approx(medium.omega*medium.mu/medium.k)


@pytest.mark.parametrize('epsilon,mu,background',[(2.-.05j,1.3+.02j,False),
                                                (2.+.05j,1.3-.02j,False),
                                                (2.+.05j,1.3+.02j,True)])
def test_complex_gain_and_lossy_background_still_reject(epsilon,mu,background):
    with pytest.raises(ValueError,match='passive|lossless'):
        compile_medium(mw.Material(eps_r=epsilon,mu_r=mu),3e8,background=background)


def test_active_complex_absorption_host_index_is_not_silently_reinterpreted():
    material=mw.Material(eps_r=2.+.05j,nonlinearity=mw.TwoPhotonAbsorption(beta=.001))
    with pytest.raises(TypeError):
        _=material.tpa_sigma_scale
    with pytest.raises(TypeError):
        _=material.is_nonlinear


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA public preparation required')
def test_public_complex_scalar_prepare_matches_complete_original_weak_matrix(tmp_path):
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.linalg import StoppingRule
    epsilon,mu=2.+.05j,1.3+.02j
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01),mw.Material(eps_r=epsilon,mu_r=mu))],
                   sources=[mw.PlaneWave()])
    config=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=mw.FMMConfig(order=6,far_action_batch=32,
            table_storage='compact_shared_displacement_v1',shift_storage='separable_v1',
            near_storage='scalar_reciprocal_dk_v1',leaf_size=8,separation_ratio=.25),
        riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='rwg_diagonal',fallback='none'))
    prepared=mw.Simulation.mom(scene,frequency=9e7,config=config).prepare()
    exterior,interior=validate_scene(scene,9e7)
    assert interior.epsilon_r==epsilon and interior.mu_r==mu and exterior.epsilon_r==1.
    op=prepared.solver.handle.actions
    edges=torch.arange(op.shape[0]//2,device='cuda')
    reference=op.provider(edges,edges)
    identity=torch.eye(op.shape[0],device='cuda',dtype=op.dtype)
    actual=op.apply(identity).mT
    error=float((actual-reference).norm()/reference.norm())
    assert error<1e-8
    generator=torch.Generator(device='cuda').manual_seed(619)
    x=torch.randn((3,op.shape[0]),device='cuda',dtype=op.dtype,generator=generator).requires_grad_()
    y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
    derivative=torch.autograd.grad(op.apply(x),x,y)[0]
    adjoint=op.adjoint(y)
    gradient_error=float((derivative-adjoint).norm()/adjoint.norm())
    original_H_error=float((adjoint-y@reference.conj()).norm()/(y@reference.conj()).norm())
    assert gradient_error<1e-11 and original_H_error<1e-8
    (tmp_path/'public.json').write_text(json.dumps(dict(rows=op.shape[0],original_matrix_relative_error=error,
        original_H_relative_error=original_H_error,current_gradient_relative_error=gradient_error,
        compiled_epsilon=[interior.epsilon_r.real,interior.epsilon_r.imag],
        compiled_mu=[interior.mu_r.real,interior.mu_r.imag],native_CUDA=True,
        solve_or_archive_replayed=False,physical_qualification=False,production_qualification=False),indent=2)+'\n')
