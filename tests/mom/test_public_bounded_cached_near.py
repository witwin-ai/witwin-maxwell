"""New public cache integration; previous original references are reused."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_capacity import storage_bytes,multipole_action_tensors

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return dict(finite=False,native_value=repr(value))
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json_safe(v) for v in value]
    return value


def _write(path,value):
    path.write_text(json.dumps(_json_safe(value),indent=2,allow_nan=False)+'\n')


def _fixture():
    # Reuse only the unchanged declarative fixture helpers. No old test runs.
    path=Path(__file__).with_name('test_public_signed_axis.py')
    spec=importlib.util.spec_from_file_location('_cached_original_public_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scene=module.scene()
    config=module.config(near_storage='scalar_reciprocal_cached_dk_v1',leaf_factor_kernel='original',
        # Declare the original admitted p7 directly. Original p6..12 order
        # search is covered and is not replayed; actual p7 M2L gates still run.
        order=7,max_order=None)
    return scene,config


def test_new_cached_schema_and_affected_existing_payloads(tmp_path):
    path=os.environ.get('GSP_CACHE_ORIGINAL_FMM_CONFIG')
    if path is None:pytest.skip('Pinned original configuration required')
    path=Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GSP_CACHE_ORIGINAL_FMM_CONFIG_SHA256']
    spec=importlib.util.spec_from_file_location('witwin.maxwell.mom._cached_original_fmm_config',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    choices=({},dict(table_storage='compact_shared_signed_axis_v1'),
        dict(near_storage='scalar_reciprocal_streamed_dk_v1'),
        dict(table_storage='compact',expansion_center_policy='binary_lattice_v1'))
    for options in choices:
        original=old.FMMConfig(**options);current=mw.FMMConfig(**options)
        assert current.payload()==original.payload() and current.construction()==original.construction()
        assert mw.FMMConfig.restore(original.payload())==current
    _,cfg=_fixture();payload=cfg.payload()
    assert payload['version']=='cartesian_fmm_cached_reciprocal_dk_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==cfg
    for kind in ('version','missing','extra','bool','negative','policy'):
        bad=copy.deepcopy(payload)
        if kind=='version':bad['version']='cartesian_fmm_signed_axis_config_v1'
        elif kind=='missing':bad['parameters'].pop('near_coefficient_cache_budget_bytes')
        elif kind=='extra':bad['parameters']['global_cache_budget']=True
        elif kind=='bool':bad['parameters']['near_coefficient_cache_budget_bytes']=True
        elif kind=='negative':bad['parameters']['near_coefficient_cache_budget_bytes']=-1
        else:bad['parameters']['near_storage']='scalar_reciprocal_streamed_dk_v1'
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    with pytest.raises(ValueError,match='requires cached'):
        mw.FMMConfig(near_coefficient_cache_budget_bytes=1)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(preconditioner='none',fallback='none'),
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    with pytest.raises(ValueError,match='complex128'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(preconditioner='none',fallback='none'),dtype=torch.complex64)
    _write(tmp_path/'cached-configuration.json',dict(completed=True,affected_old_payloads_exact=True,
        new_version_exact=True,old_planner_not_replayed=True,production_qualification=False))


def test_new_cached_capacity_matches_native_owner_and_aggregate_before_coefficients(monkeypatch,tmp_path):
    from witwin.maxwell.mom.cluster import build_support_octree
    import witwin.maxwell.mom.fmm_capacity as capacity
    from witwin.maxwell.mom.packed_near import BoundedCachedReciprocalScalarNearChannels
    from witwin.maxwell.mom.physics import compile_medium
    coords=torch.tensor([[.01,.02,.03],[.07,.09,.02],[.09,.03,.04],[-.13,.04,.08],
        [-.17,.08,.09],[.23,-.14,.05],[.28,-.12,.07],[.31,-.13,.06],
        [.48,.18,-.04],[.49,.24,-.06],[.52,.26,-.08]],device='cuda',dtype=torch.float64)
    tree=build_support_octree(coords-.001,coords+.001,leaf_size=3)
    observed=[];original=capacity.sphere_block_partition
    def collect(*a,**kw):
        for aa,bb,far in original(*a,**kw):
            observed.extend(zip(aa.tolist(),bb.tolist(),far.tolist()))
            yield aa,bb,far
    monkeypatch.setattr(capacity,'sphere_block_partition',collect)
    plan=capacity.plan_multipole_storage((tree,tree),order=3,separation_ratio=.31,budget_bytes=2**30,
        table_storage='compact_shared_signed_axis_v1',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_cached_dk_v1',near_coefficient_cache_budget_bytes=379)
    midpoint=len(observed)//2
    assert observed[:midpoint]==observed[midpoint:]
    layout=[]
    for a,b,far in observed[:midpoint]:
        if not far:
            rows=tree.permutation[tree.starts[a]:tree.stops[a]]
            cols=tree.permutation[tree.starts[b]:tree.stops[b]]
            layout.append((a,b+tree.node_count,torch.cat((rows,rows+tree.size)),
                torch.cat((cols,cols+tree.size))+2*tree.size))
    def forbidden(*a,**kw):pytest.fail('Structural cache planning integrated coefficients')
    owner=BoundedCachedReciprocalScalarNearChannels.from_layout(layout,2*tree.size,2*tree.size,
        forbidden,source_node_offset=tree.node_count,
        medium=compile_medium(mw.Material(eps_r=2.4,mu_r=1.2),167e6),integration=IntegrationConfig(),
        device='cuda',dtype=torch.complex128,physical_guard=tree.assert_current,
        physical_inputs=tree.tensors(),coefficient_cache_budget_bytes=379)
    receipt=owner.receipt();regional=plan['regional_coefficient_cache']
    assert len(regional)==2 and regional[0]==regional[1]
    assert regional[0]['retained_kernel_bytes']==receipt['retained_kernel_bytes']<=379
    assert regional[0]['cache_segments']==[list(s) for s in owner._cache_plan]
    assert plan['categories']['near']==2*receipt['retained_bytes']
    assert plan['retained_near_kernel_bytes']==2*receipt['retained_kernel_bytes']
    assert receipt['cache_ready_blocks']==0
    _write(tmp_path/'cached-capacity.json',dict(completed=True,plan=plan,native_receipt=receipt,
        per_owner_and_aggregate_exact=True,no_original_integrations=True,production_qualification=False))


@pytest.mark.parametrize('component',('action','near'))
def test_new_cached_payload_refuses_before_moments_when_maps_alone_would_fit(component,monkeypatch):
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    declared,cfg=_fixture()
    options=dict(cfg.__dict__)
    # The exact same physical trees with p7 and streamed maps previously owned
    #1326624B total and12480B near. Cached payload must enter both gates.
    options['action_retained_budget_bytes' if component=='action' else 'near_budget_bytes']=(
        1326624 if component=='action' else 12480)
    def forbidden(*a,**kw):pytest.fail('Refused cached capacity reached moment setup')
    monkeypatch.setattr(NestedRWGMultipoles,'__init__',forbidden)
    config=mw.MoMConfig(integration=IntegrationConfig(order=4,singular_order=3),operator_backend='fmm',
        fmm=mw.FMMConfig(**options),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(preconditioner='rwg_diagonal',fallback='none'))
    with pytest.raises(MemoryError):mw.Simulation.mom(declared,frequency=1.43e8,config=config).prepare()


def test_new_public_cached_native_actions_solve_and_exact_restore(monkeypatch,tmp_path):
    from witwin.maxwell.mom.runtime import SystemOperator,SurfaceData
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    import witwin.maxwell.mom.fmm_regional as regional
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    location=os.environ.get('GSP_CACHE_PUBLIC_ORIGINAL_REFERENCE')
    if location is None:pytest.skip('Pinned original public weak reference required; never regenerated')
    reference_path=Path(location)
    assert reference_path.is_file(), 'Pinned original public weak reference is unavailable'
    declared,cfg=_fixture()
    config=mw.MoMConfig(integration=IntegrationConfig(order=4,singular_order=3),operator_backend='fmm',
        fmm=cfg,riesz_storage='local_rwg_v1',stopping=StoppingRule(method='gmres',
            preconditioner='rwg_diagonal',fallback='none',rtol=1e-12,atol=0.,maxiter=240,restart=120))
    # Physical Scene and numerical configuration are independently declared
    # before opening the reference, and its identity is verified before setup.
    digest=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert digest=='837bcf6cf2f3870c11fee5926d840f0744aec3403a4c06ce17ded28a21aa2e2e'
    reference=torch.load(reference_path,weights_only=False,map_location='cuda')['reference']
    assert reference.shape==(72,72) and reference.dtype==torch.complex128
    raw=dict(version='public_bounded_cached_original_near_v1',completed=False,
        production_qualification=False,physical_qualification=False,
        declared_fixed_order_reuses_original_admitted_p7=True,automatic_order_search_not_replayed=True)
    path=tmp_path/'cached-public.json';_write(path,raw)
    calls=[]
    exterior=ScalarSystemBlocks.exterior_channels;interior=regional.scalar_regional_channels
    def counted_exterior(self,*a,**kw):calls.append('exterior');return exterior(self,*a,**kw)
    def counted_interior(*a,**kw):calls.append('interior');return interior(*a,**kw)
    monkeypatch.setattr(ScalarSystemBlocks,'exterior_channels',counted_exterior)
    monkeypatch.setattr(regional,'scalar_regional_channels',counted_interior)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    try:
        prepared=mw.Simulation.mom(declared,frequency=1.43e8,config=config).prepare()
        op=prepared.solver.handle.actions;regions=(op.exterior,*op.interiors)
        constructor_calls=len(calls)
        assert constructor_calls==0 and op.shape==(72,72)
        assert op.order_plan is None and all(r.test.order==7 for r in regions)
        raw.update(capacity_plan=op.capacity_plan,fmm_config=cfg.payload(),
            original_reference_path=str(reference_path),original_reference_sha256=digest,
            constructor_factory_calls=constructor_calls)
        _write(path,raw)
        columns=op.apply(torch.eye(72,device='cuda',dtype=op.dtype)).mT
        cold_calls=len(calls)
        generator=torch.Generator(device='cuda').manual_seed(487039)
        x=torch.randn((3,5,72),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
        dual=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):forward=op.apply(x)
        adjoint=op.adjoint(dual)
        gradient=torch.autograd.grad(forward,x,dual)[0]
        first=torch.autograd.grad(op.apply(x).abs().square().sum(),x,create_graph=True)[0]
        second=torch.autograd.grad(first,x,dual)[0]
        lhs,rhs=(forward.conj()*dual).sum(),(x.conj()*adjoint).sum()
        errors=dict(columns=relative(columns,reference),forward=relative(forward,x@reference.mT),
            original_H=relative(adjoint,dual@reference.conj()),
            native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()),
            current_gradient=relative(gradient,adjoint),
            higher_current_gradient=relative(second,2*(dual@columns.mT)@columns.conj()))
        actual=tmp_path/'cached-public-native-state.pt'
        torch.save(dict(columns=columns,x=x.detach(),dual=dual,forward=forward.detach(),
            adjoint=adjoint.detach(),gradient=gradient.detach(),higher=second.detach()),actual)
        raw.update(errors=errors,native_errors_finite=all(math.isfinite(v) for v in errors.values()),cold_factory_calls=cold_calls,warm_factory_calls=len(calls)-cold_calls,
            saved_forward_tensors=len(saved),actual_native_state=str(actual),
            actual_native_state_sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
        _write(path,raw)
        assert raw['native_errors_finite'],errors
        assert max(errors[k] for k in ('columns','forward','original_H'))<1e-8
        assert max(errors[k] for k in ('native_H','current_gradient','higher_current_gradient'))<1e-11
        assert cold_calls>0 and len(calls)==cold_calls and saved==[]
        receipts=[r.near.receipt() for r in regions]
        planned=op.capacity_plan['regional_coefficient_cache']
        for receipt,plan in zip(receipts,planned,strict=True):
            assert receipt['retained_kernel_bytes']==plan['retained_kernel_bytes']
            assert receipt['cache_ready_blocks']==receipt['cache_planned_blocks']==plan['cache_planned_blocks']
        owned=storage_bytes(multipole_action_tensors(op))
        assert owned==op.capacity_plan['planned_bytes']
        result=prepared.run()
        current=result.surface_currents.scattered_coefficients
        stats=result.stats();residual=relative(current@reference.mT,prepared.solver.rhs)
        state=tmp_path/'cached-public-solver-state.pt'
        torch.save(dict(stats=stats,current=current.detach(),rhs=prepared.solver.rhs.detach()),state)
        raw.update(actual_solver_stats=stats,actual_solver_state=str(state),
            actual_solver_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
            complete_original_residual=residual,complete_original_residual_finite=math.isfinite(residual),
            residual_relative_gate=1e-10,actual_owned_bytes=owned,
            warm_factory_calls=len(calls)-cold_calls,near_receipts=receipts)
        _write(path,raw)
        assert stats['converged'] and stats['error_budget']['checked']
        assert raw['complete_original_residual_finite'] and residual<=1e-10
        assert len(calls)==cold_calls
        artifact=tmp_path/'cached-public-result.pt';result.save(artifact)
        def forbidden(*a,**kw):pytest.fail('Cache restore assembled or solved')
        with monkeypatch.context() as patch:
            for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(artifact,scene=declared)
            assert restored.surface_currents.config.fmm==cfg
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            bad=copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters']['near_coefficient_cache_budget_bytes']-=1
            with pytest.raises(ValueError):SurfaceData.restore(bad,declared,1.43e8)
        raw.update(completed=True,artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            exact_restore=True,whole_peak_qualified=False,independent_EM_reference=False)
        _write(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None))
        _write(path,raw)
        raise
