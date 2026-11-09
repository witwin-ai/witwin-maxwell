"""Bounded original compact FAR batches preserve physical and current contracts."""
import copy
import hashlib
import json
import math
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_compact import compact_moment_actions, compact_batch_workspace_bytes
from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors, storage_bytes
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_operator import MultipoleOperator
from witwin.maxwell.mom.runtime import SystemOperator, SurfaceData
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def relative(actual,expected):
    return float(((actual-expected).norm()/expected.norm()).detach())


@pytest.fixture(scope='module')
def physical(tmp_path_factory):
    out=tmp_path_factory.mktemp('bounded-compact-physical')
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01,position=(x,0.,0.)),
        mw.Material(eps_r=2.+i,mu_r=1.2+.1*i)) for i,x in enumerate((-.2,.2))],
        sources=[mw.PlaneWave()])
    fmm=mw.FMMConfig(order=6,kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        kernel_bound_rule='direct_first_wave_min_v1',table_storage='compact_shared_displacement_v1',
        shift_storage='separable_v1',near_storage='scalar_reciprocal_dk_v1',far_action_batch=32,
        leaf_size=8,separation_ratio=.25,leaf_factor_kernel='primary',leaf_factor_max_edges=16)
    config=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
                             rtol=1e-12,atol=0.,maxiter=240,restart=120))
    prepared=mw.Simulation.mom(scene,frequency=9e7,config=config).prepare()
    op=prepared.solver.handle.actions
    assert op.exterior.far and op.far_action_batch==32
    # One complete original weak reference is shared by every new physical
    # control; no second original matrix or duplicate near construction.
    rows=torch.arange(op.shape[0]//2,device='cuda')
    reference=op.provider(rows,rows)
    return scene,prepared,op,reference,out


def test_all_original_columns_H_noncontiguous_15_RHS_current_higher_and_owned(physical,monkeypatch):
    scene,prepared,op,reference,out=physical
    import witwin.maxwell.mom.fmm_compact as compact
    calls=[];original=compact.compact_moment_actions
    def observed(operators,value,**kwargs):
        calls.append(dict(pairs=len(operators),rhs=len(value),reverse=sum(hasattr(o,'_reverse_owner') for o in operators)))
        assert len(operators)<=32 and len(value)<=4
        return original(operators,value,**kwargs)
    monkeypatch.setattr(compact,'compact_moment_actions',observed)
    identity=torch.eye(op.shape[0],device='cuda',dtype=op.dtype)
    actual=op.apply(identity).mT
    assert relative(actual,reference)<1e-8
    generator=torch.Generator(device='cuda').manual_seed(1917)
    x=torch.randn((5,3,op.shape[0]),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
    y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
    assert not x.is_contiguous()
    applied=op.apply(x);adjoint=op.adjoint(y)
    assert relative(applied,x@reference.mT)<1e-8 and relative(adjoint,y@reference.conj())<1e-8
    lhs=(applied.conj()*y).sum();rhs=(x.conj()*adjoint).sum()
    native_error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()))
    assert native_error<1e-11
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        value=op.apply(x)
    assert not saved
    derivative=torch.autograd.grad(value,x,y,retain_graph=True)[0]
    assert relative(derivative,adjoint)<1e-11
    loss=op.apply(x).abs().square().sum()
    first=torch.autograd.grad(loss,x,create_graph=True)[0]
    direction=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
    second=torch.autograd.grad(first,x,direction)[0]
    assert relative(second,2*op.adjoint(op.apply(direction)))<1e-11
    owned=storage_bytes(multipole_action_tensors(op))
    assert owned==op.capacity_plan['planned_bytes']
    assert calls and any(c['pairs']>1 for c in calls) and any(c['rhs']==4 for c in calls)
    assert any(c['reverse'] for c in calls)
    (out/'native.json').write_text(json.dumps(dict(original_columns=op.shape[0],
        original_relative_error=relative(actual,reference),native_H_relative_error=native_error,
        saved_forward_tensors=0,leading_RHS=15,actual_owned_bytes=owned,capacity=op.capacity_plan,
        batch_calls=calls,whole_peak_qualified=False,production_qualification=False),indent=2)+'\n')


def test_actual_compact_batch_matches_single_translations_reverse_and_mutations(physical):
    scene,prepared,op,reference,out=physical
    available=[c._compact for _,_,c in op.exterior.far[:32]]
    operators=[available[n%len(available)] for n in range(32)]
    count=operators[0].count
    generator=torch.Generator(device='cuda').manual_seed(449)
    value=torch.randn((4,len(operators),2,count,4),device='cuda',dtype=op.dtype,generator=generator)
    for adjoint in (False,True):
        actual=compact_moment_actions(operators,value,adjoint=adjoint)
        expected=torch.stack([o.adjoint(v) if adjoint else o.apply(v)
            for o,v in zip(operators,value.unbind(1),strict=True)],dim=1)
        assert relative(actual,expected)<3e-12
    original=operators[0]
    owner=type(original).__new__(type(original))
    owner._bind_coefficients(original.coefficients.clone(),original.medium,original.order,original.row_batch,
                             layout=(original.indices,original.binomial))
    operators[0]=owner
    with torch.no_grad():
        owner.indices.add_(0)
    # Restore the exact bytes, but native version mutation must stay detectable.
    with pytest.raises(ValueError,match='changed'):compact_moment_actions(operators,value)
    # The independent mutated owner leaves every shared physical consumer intact.


@pytest.mark.parametrize('kind', ['pairs','rhs','precision','order','trainable','workspace'])
def test_batch_preflight_rejects_before_native_stacks(kind,monkeypatch):
    import witwin.maxwell.mom.fmm_compact as compact
    from witwin.maxwell.mom.physics import compile_medium
    medium=compile_medium(mw.Material(),3e8)
    delta=torch.tensor([1.,.2,-.1],device='cuda',dtype=torch.float64)
    operator=compact.CompactElectromagneticM2L(delta,medium,0)
    value=torch.ones((1,1,2,operator.count,4),device='cuda',dtype=torch.complex128)
    operators=[operator]
    if kind=='pairs':operators=[operator]*33
    elif kind=='rhs':value=value.expand(5,*value.shape[1:])
    elif kind=='precision':value=value.to(torch.complex64)
    elif kind=='order':operator.order=13
    elif kind=='trainable':operator.coefficients.requires_grad_(True)
    elif kind=='workspace':monkeypatch.setattr(compact,'compact_batch_workspace_bytes',lambda *args:128*1024**2+1)
    def forbidden(*args,**kwargs):raise AssertionError('invalid batch reached tensor stacks')
    monkeypatch.setattr(torch,'stack',forbidden)
    monkeypatch.setattr(torch,'isfinite',forbidden)
    with pytest.raises((ValueError,MemoryError)):compact_moment_actions(operators,value)


def test_public_new_policy_solve_exact_restore_and_complete_original_residual(physical,monkeypatch):
    scene,prepared,op,reference,out=physical
    result=prepared.run()
    assert result.stats()['converged'] and result.stats()['error_budget']['checked']
    current=result.surface_currents.scattered_coefficients
    original_residual=current@reference.mT-prepared.solver.rhs
    error=float(original_residual.norm()/prepared.solver.rhs.norm())
    assert error<=1e-10
    residual=dict(version='shared_original_weak_matrix_vector_residual_v1',
        all_rhs_passed=True,all_original_rows_measured=True,rows=op.shape[0],
        relative_tolerance=1e-10,absolute_tolerance=0.,relative_error=error,
        original_matrix_constructed_once=True,original_quadrature_replayed=False,
        physical_qualification=False)
    artifact=out/'bounded-compact-result.pt';result.save(artifact)
    def forbidden(*args,**kwargs):raise AssertionError('archive restore constructed or solved an operator')
    with monkeypatch.context() as patch:
        for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(artifact,scene=scene)
        assert restored.surface_currents.config.fmm.far_action_batch==32
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        for field,value in [('far_action_batch',1),('far_action_batch',16),('table_storage','compact')]:
            bad=copy.deepcopy(result.surface_currents.snapshot());bad['fmm_config']['parameters'][field]=value
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,9e7)
    (out/'public.json').write_text(json.dumps(dict(artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        fmm_config=result.surface_currents.config.fmm.payload(),complete_original_residual=residual,
        production_qualification=False),indent=2)+'\n')


def test_new_config_requires_exact_batch_version_and_rejects_unsupported_cases():
    old=json.loads((Path(__file__).parent/'fixtures/bounded_compact_original_payloads.json').read_text())
    for case in old['cases']:
        current=mw.FMMConfig(**case['options'])
        assert current.payload()==case['payload']
        assert mw.FMMConfig.restore(case['payload'])==current
        assert 'far_action_batch' not in current.construction()
    config=mw.FMMConfig(table_storage='compact_shared_displacement_v1',far_action_batch=32)
    payload=config.payload();assert payload['version']=='cartesian_fmm_batched_compact_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==config
    for value in (0,1,33,True,2.,None):
        bad=copy.deepcopy(payload);bad['parameters']['far_action_batch']=value
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    bad=copy.deepcopy(payload);bad['version']='cartesian_fmm_shared_compact_config_v1'
    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for kwargs in (dict(far_action_batch=32),dict(table_storage='fft_v1',far_action_batch=32),
        dict(table_storage='compact',far_action_batch=32,order=13),dict(table_storage='compact',far_action_batch=True)):
        with pytest.raises(ValueError):mw.FMMConfig(**kwargs)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=config,riesz_storage='local_rwg_v1',
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)),stopping=StoppingRule(preconditioner='none'))
    assert compact_batch_workspace_bytes(12,32,4,16)<=128*1024**2


def test_maximum_order_pairs_rhs_boundary_matches_original_actions(tmp_path):
    from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
    from witwin.maxwell.mom.physics import compile_medium
    medium=compile_medium(mw.Material(eps_r=2.+.05j,mu_r=1.3+.02j),3e8)
    delta=torch.tensor([[1.,.2,-.1],[.8,-.3,.15]],device='cuda',dtype=torch.float64)
    owners=CompactElectromagneticM2L.create_many(delta,medium,12)
    unique=[owners[0],owners[0].reversed_translation(),owners[1],owners[1].reversed_translation()]
    operators=[unique[n%4] for n in range(32)]
    generator=torch.Generator(device='cuda').manual_seed(1931)
    x=torch.randn((4,32,2,owners[0].count,4),device='cuda',dtype=torch.complex128,generator=generator)
    y=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
    errors={}
    with torch.no_grad():
        forward=compact_moment_actions(operators,x)
        hermitian=compact_moment_actions(operators,y,adjoint=True)
        # Each original owner/reversal is evaluated once with all of its RHS;
        # aliases exercise the pair limit without duplicate polynomial builds.
        for adjoint,value,actual in [(False,x,forward),(True,y,hermitian)]:
            expected=torch.empty_like(actual)
            for index,operator in enumerate(unique):
                moments=value[:,index::4].reshape(-1,2,operator.count,4)
                action=operator.adjoint(moments) if adjoint else operator.apply(moments)
                expected[:,index::4]=action.reshape_as(expected[:,index::4])
            error=relative(actual,expected)
            assert error<1e-11
            errors['adjoint' if adjoint else 'forward']=error
        lhs=(forward.conj()*y).sum();rhs=(x.conj()*hermitian).sum()
        error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()))
        assert error<1e-11
    (tmp_path/'boundary.json').write_text(json.dumps(dict(order=12,pairs=32,rhs=4,
        rows=16,count=owners[0].count,independent_coefficient_owners=2,
        original_action_errors=errors,native_H_relative_error=error,
        visible_workspace_bound_bytes=compact_batch_workspace_bytes(12,32,4,16),
        whole_peak_qualified=False,production_qualification=False),indent=2)+'\n')
