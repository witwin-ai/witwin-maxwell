"""Numerical and ownership contracts for optimized CUDA local updates."""
import pytest
import torch
from test_local_response import setup_response

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_sparse_absolute_updates_match_dense_and_own_inputs():
    with setup_response() as r:
        indices=torch.tensor([0,3,5],device=r.device)
        delta=torch.tensor([3+2j,7.,0.],device=r.device,dtype=r.dtype)
        pec=torch.tensor([False,False,True],device=r.device)
        trial=r.trial_updates(indices,delta=delta,pec=pec)
        dense=torch.zeros_like(r._u0); dense[indices]=delta
        mask=torch.zeros_like(dense,dtype=torch.bool); mask[indices]=pec
        expected=r.run(delta=dense,pec=mask).raw_output['projections']
        delta.zero_(); pec.zero_(); indices.zero_()
        torch.testing.assert_close(trial.projections,expected,rtol=1e-8,atol=1e-8)
        result=r.accept(trial)
        torch.testing.assert_close(result.raw_output['projections'],expected,rtol=1e-8,atol=1e-8)
        with pytest.raises(ValueError,match='stale'):
            r.accept(trial)
        with pytest.raises(ValueError,match='unique'):
            r.trial_updates(indices,delta=delta)


def test_snapshot_comparison_detects_unversioned_tensor_mutation():
    from witwin.maxwell.fdfd.response import _snapshot,_matches
    x=torch.arange(4,device='cuda')
    snapshot=_snapshot({'data':x})
    assert _matches(snapshot,{'data':x})
    x.data[0]=10
    assert not _matches(snapshot,{'data':x})
    assert not _matches(_snapshot(1),{'data':x})


@pytest.mark.parametrize('rank',[1,4,12])
def test_update_certificate_well_conditioned_and_singular(rank,monkeypatch):
    from witwin.maxwell.fdfd.update_solve import solve_update
    generator=torch.Generator(device='cuda').manual_seed(4)
    product=.01*torch.randn(rank,rank,device='cuda',dtype=torch.complex128,generator=generator)
    matrix=torch.eye(rank,device='cuda',dtype=product.dtype)+product
    rhs=torch.randn(rank,device='cuda',dtype=product.dtype,generator=generator)
    expected=torch.linalg.solve(matrix,rhs)
    original=torch.linalg.svdvals
    def unexpected(*args,**kwargs):
        raise AssertionError('A well conditioned candidate should not invoke SVD.')
    monkeypatch.setattr(torch.linalg,'svdvals',unexpected)
    actual,residual,inverse=solve_update(matrix,product,rhs,1e-10)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(matrix@inverse,torch.eye(rank,device='cuda',dtype=product.dtype),rtol=1e-12,atol=1e-12)
    assert residual<1e-10
    zero,_,_=solve_update(matrix,product,torch.zeros_like(rhs),1e-10)
    assert torch.count_nonzero(zero)==0
    monkeypatch.setattr(torch.linalg,'svdvals',original)
    matrix[-1]=0
    with pytest.raises(RuntimeError,match='singular'):
        solve_update(matrix,matrix-torch.eye(rank,device='cuda',dtype=product.dtype),rhs,1e-10)


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_fused_solver_uses_current_stream_and_all_rank_buckets(dtype):
    from witwin.maxwell.fdfd.update_solve import solve_update
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for rank in (3,7,15,31,33):
            p=.005*torch.randn(rank,rank,device='cuda',dtype=dtype)
            a=torch.eye(rank,device='cuda',dtype=dtype)+p
            b=torch.randn(rank,device='cuda',dtype=dtype)
            actual,_,_=solve_update(a,p,b,1e-5 if dtype==torch.complex64 else 1e-10)
            torch.testing.assert_close(actual,torch.linalg.solve(a,b),rtol=2e-5 if dtype==torch.complex64 else 1e-11,atol=1e-6 if dtype==torch.complex64 else 1e-12)


def test_long_mixed_sequence_and_zero_forcing_rebuild_bound():
    with setup_response() as r:
        torch.manual_seed(71)
        delta=torch.zeros_like(r._u0);pec=torch.zeros_like(delta,dtype=torch.bool)
        for i in range(280):
            index=torch.tensor([i%delta.numel()],device=r.device)
            new_delta=delta.clone();new_pec=pec.clone()
            new_delta[index]=complex((i%7)*2,.01*(i%3))
            new_pec[index]=(i%11==0)
            trial=r.trial_updates(index,delta=new_delta[index],pec=new_pec[index])
            if i%4!=3:
                r._accept(trial)
                delta,pec=new_delta,new_pec
            if i%23==0:
                expected=r.run(delta=new_delta,pec=new_pec).raw_output['projections']
                torch.testing.assert_close(trial.projections,expected,rtol=2e-8,atol=1e-8)
        assert r._rebuild_count>=1
        reset=r.trial(delta=torch.zeros_like(delta),pec=torch.zeros_like(pec))
        r.accept(reset)
        assert torch.count_nonzero(r._accepted_current)==0
        torch.testing.assert_close(r._accepted_projection,r._y0,rtol=0,atol=0)
        torch.testing.assert_close(r._projected_inverse,r._observation_green@r._inverse,rtol=1e-7,atol=1e-8)


def test_batch_proposals_own_state_and_match_individual_updates():
    with setup_response() as r:
        indices=torch.tensor([[0,2,4],[1,3,5],[2,4,6]],device=r.device)
        delta=torch.tensor([[2.,4.,6.],[3.,2.,1.],[4.,5.,2.]],device=r.device,dtype=r.dtype)
        pec=torch.tensor([[True,False,False],[False,False,True],[False,True,False]],device=r.device)
        trials=r.trial_updates_batch(indices,delta=delta,pec=pec)
        expected=[r.trial_updates(i,delta=d,pec=p).projections for i,d,p in zip(indices,delta,pec)]
        for trial,value in zip(trials,expected):
            torch.testing.assert_close(trial.projections,value,rtol=1e-8,atol=1e-8)
        # Batched and individual updates agree to round-off only; ownership is
        # checked bitwise against the trial's own values before workspace reuse.
        snapshot=trials[0].projections.clone()
        r.trial_updates_batch(indices,delta=delta*2,pec=pec)
        delta.zero_();pec.zero_();indices.zero_()
        torch.testing.assert_close(trials[0].projections,snapshot,rtol=0,atol=0)
        r.accept(trials[1])
        with pytest.raises(ValueError,match='stale'):
            r.accept(trials[0])


def test_batched_search_preserves_sequential_first_improvement(tmp_path):
    with setup_response() as original:
        simulation=original.simulation
        original.save_cache(tmp_path/'response.pt')
    outcomes=[]
    for batch in (1,4):
        with simulation.prepare(response_cache=tmp_path/'response.pt') as r:
            groups=[torch.tensor([i,i+1],device=r.device) for i in range(6)]
            outcome=r.optimize_binary(groups,lambda x:x.abs().square().sum(),mode='dielectric',
                                      epsilon_contrast=.5,max_passes=3,seed=71,batch_size=batch)
            outcomes.append(outcome)
    assert outcomes[0].attempts==outcomes[1].attempts
    assert outcomes[0].accepted==outcomes[1].accepted
    assert outcomes[0].converged==outcomes[1].converged
    torch.testing.assert_close(outcomes[0].tiles,outcomes[1].tiles)
    torch.testing.assert_close(outcomes[0].objective_history,outcomes[1].objective_history,rtol=1e-8,atol=1e-8)


def test_batch_singular_failure_is_transactional():
    with setup_response() as r:
        indices=torch.tensor([[0],[1]],device=r.device)
        delta=torch.zeros((2,1),device=r.device,dtype=r.dtype)
        delta[1,0]=-1/r._green[1,1]
        before=r._inverse.clone()
        with pytest.raises(RuntimeError,match='singular'):
            r.trial_updates_batch(indices,delta=delta)
        assert r._generation==0
        torch.testing.assert_close(r._inverse,before,rtol=0,atol=0)


def test_fused_trial_accepts_strided_material_and_pec_views():
    with setup_response() as r:
        storage=torch.zeros((r._u0.numel(),2),device=r.device,dtype=r.dtype)
        flags=torch.zeros(storage.shape,device=r.device,dtype=torch.bool)
        delta=storage[:,0];pec=flags[:,1]
        delta[0]=3+2j;delta[3]=4;pec[2]=True
        assert not delta.is_contiguous() and not pec.is_contiguous()
        actual=r.trial(delta=delta,pec=pec).projections
        expected=r.run(delta=delta,pec=pec).raw_output['projections']
        torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)


def test_auto_grid_rechecks_geometry_dependent_resolved_coordinates():
    import witwin.maxwell as mw
    with setup_response() as base:
        scene=base.simulation.scene.clone();config=base.config
    scene.grid=mw.GridSpec.auto(wavelength=.3,min_steps_per_wavelength=6)
    with mw.Simulation.fdfd(scene,frequency=1e9,solver=mw.GMRES(solver_type='direct',precision='double'),local_response=config).prepare() as r:
        zero=torch.zeros_like(r._u0)
        r.trial(delta=zero)
        scene.add_structure(mw.Structure(name='mesh-change',geometry=mw.Box(size=(.04,)*3),material=mw.Material(eps_r=2.)))
        with pytest.raises(ValueError,match='Background'):
            r.trial(delta=zero)


def test_reconstruction_eliminates_identity_rows_exactly():
    with setup_response() as r:
        for count in (0,4,r._u0.numel()):
            delta=torch.zeros_like(r._u0);delta[:count]=4+1j
            pec=torch.zeros_like(delta,dtype=torch.bool);pec[:count:2]=True
            a,b=r._rows(delta,pec)
            matrix=r._identity*a[:,None]+r._green*b[:,None]
            inverse,current,residual=r._rebuild_state(matrix,b*r._u0,b)
            torch.testing.assert_close(inverse,torch.linalg.solve(matrix,r._identity),rtol=1e-8,atol=1e-7)
            torch.testing.assert_close(current,torch.linalg.solve(matrix,b*r._u0),rtol=1e-8,atol=1e-8)
            assert residual<=r.config.tolerance
