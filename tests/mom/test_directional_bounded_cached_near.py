"""New Directional cache wiring; one saved original weak reference is reused."""
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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
REFERENCE_SHA = 'bc999b41dbca5779ad484e00b457cbe3c9eb42562218502a16649305d8afe646'


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(finite=False, native_value=repr(value))
    if isinstance(value, dict):
        return {str(k): _safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_safe(v) for v in value]
    return value


def _write(path, value):
    path.write_text(json.dumps(_safe(value), indent=2, allow_nan=False)+'\n')


def cached_config(**changes):
    options = dict(near_storage='scalar_reciprocal_cached_dk_v1',
        near_coefficient_workspace_budget_bytes=4096, leaf_size=3, separation_ratio=.15,
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        regional_near_budget_bytes=1024**3, tensor_workspace_budget_bytes=512*1024**2)
    options.update(changes)
    return mw.DirectionalFMMConfig(**options)


def declare_scene():
    # Reuse only the accepted declarative physical fixture, never its tests.
    path = Path(__file__).with_name('test_directional_streamed_near.py')
    spec = importlib.util.spec_from_file_location('_cached_directional_physical_fixture',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.declare_scene()


def test_new_directional_cached_schema_and_affected_existing_payloads(tmp_path):
    location = os.environ.get('GSP_DIRECTIONAL_CACHE_ORIGINAL_CONFIG')
    if location is None:
        pytest.skip('Pinned original configuration required')
    path = Path(location)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GSP_DIRECTIONAL_CACHE_ORIGINAL_CONFIG_SHA256']
    spec = importlib.util.spec_from_file_location('witwin.maxwell.mom._cached_directional_original_config',path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    for policy in ('scalar_dk_v1','scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1'):
        original = old.DirectionalFMMConfig(near_storage=policy)
        current = mw.DirectionalFMMConfig(near_storage=policy)
        assert current.payload()==original.payload()
        assert 'near_coefficient_cache_budget_bytes' not in current.payload()['parameters']
        assert mw.FMMConfig.restore(original.payload())==current
    cfg = cached_config()
    payload = cfg.payload()
    assert payload['version']=='automatic_directional_cached_reciprocal_dag_fmm_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==cfg
    assert mw.FMMConfig.restore(cached_config(near_coefficient_cache_budget_bytes=0).payload()).near_coefficient_cache_budget_bytes==0
    for field in payload['parameters']:
        bad = copy.deepcopy(payload)
        bad['parameters'].pop(field)
        with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
    for value in (False,-1,4096.):
        bad = copy.deepcopy(payload)
        bad['parameters']['near_coefficient_cache_budget_bytes'] = value
        with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
    for version in ('automatic_directional_dag_fmm_config_v1',
                    'automatic_directional_streamed_reciprocal_dag_fmm_config_v1',
                    'cartesian_fmm_cached_reciprocal_dk_config_v1'):
        bad = copy.deepcopy(payload)
        bad['version'] = version
        with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
    bad = copy.deepcopy(payload)
    bad['parameters']['near_storage'] = 'scalar_reciprocal_streamed_dk_v1'
    with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
    with pytest.raises(ValueError, match='requires cached'):
        mw.DirectionalFMMConfig(near_coefficient_cache_budget_bytes=0)
    options = dict(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(preconditioner='none'))
    with pytest.raises(ValueError, match='complex128'): mw.MoMConfig(**options,dtype=torch.complex64)
    with pytest.raises(NotImplementedError, match='free space'):
        mw.MoMConfig(**options,periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    _write(tmp_path/'directional-cached-config.json',dict(completed=True,payload=payload,
        old_three_payloads_exact=True,production_qualification=False))


def test_new_directional_cached_nonuniform_capacity_matches_native_owner(tmp_path):
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.fmm_capacity import plan_directional_reciprocal_near_storage, storage_bytes
    from witwin.maxwell.mom.packed_near import BoundedCachedReciprocalScalarNearChannels
    from witwin.maxwell.mom.physics import compile_medium
    coords = torch.tensor([[.013,.024,.037],[.073,.091,.025],[.096,.036,.043],[-.131,.047,.081],
        [-.174,.086,.092],[.239,-.143,.059],[.283,-.126,.074],[.317,-.131,.063],
        [.483,.187,-.044],[.497,.243,-.064],[.527,.267,-.087]],device='cuda',dtype=torch.float64)
    tree = build_support_octree(coords-.0013,coords+.0013,leaf_size=3)
    children = tree.children.cpu().tolist()
    sizes = (tree.stops-tree.starts).cpu().tolist()
    leaves = [i for i,c in enumerate(children) if all(v<0 for v in c)]
    assert len({sizes[i] for i in leaves})>1
    pairs = [(a,b) for a in leaves for b in leaves]
    canonical = [(a,b) for a,b in pairs if a<=b]
    maps = 8*(2*len({a for a,_ in canonical})+1+3*len(canonical))+sum(16*(sizes[a]+sizes[b]) for a,b in canonical)
    layout = []
    for a,b in pairs:
        rows = tree.permutation[tree.starts[a]:tree.stops[a]]
        cols = tree.permutation[tree.starts[b]:tree.stops[b]]
        layout.append((a,b+tree.node_count,torch.cat((rows,rows+tree.size)),
            torch.cat((cols,cols+tree.size))+2*tree.size))
    def forbidden(*a,**kw): pytest.fail('Structural cache admission integrated original coefficients')
    medium = compile_medium(mw.Material(eps_r=2.41,mu_r=1.23),169e6)
    records = []
    for budget in (0,379,128*1024**2):
        options = dict(near_storage='scalar_reciprocal_cached_dk_v1',
            near_coefficient_cache_budget_bytes=budget,near_coefficient_workspace_budget_bytes=4096)
        plan = plan_directional_reciprocal_near_storage(tree,pairs,budget_bytes=2**20,**options)
        owner = BoundedCachedReciprocalScalarNearChannels.from_layout(layout,2*tree.size,2*tree.size,
            forbidden,source_node_offset=tree.node_count,medium=medium,integration=IntegrationConfig(),
            device='cuda',dtype=torch.complex128,physical_guard=tree.assert_current,physical_inputs=tree.tensors(),
            coefficient_cache_budget_bytes=budget,coefficient_workspace_budget_bytes=4096)
        receipt = owner.receipt()
        assert plan['retained_near_kernel_bytes']==receipt['retained_kernel_bytes']<=budget
        assert plan['retained_bytes']==maps+receipt['retained_kernel_bytes']==storage_bytes(owner.tensors())
        assert plan['cache_segments']==[list(s) for s in owner._cache_plan]
        assert plan['cache_planned_blocks']==receipt['cache_planned_blocks']
        assert receipt['cache_ready_blocks']==0
        if budget:
            assert receipt['retained_kernel_bytes']>0
            with pytest.raises(MemoryError, match='before maps'):
                plan_directional_reciprocal_near_storage(tree,pairs,budget_bytes=plan['retained_bytes']-1,**options)
        records.append(dict(plan=plan,native_receipt=receipt))
    _write(tmp_path/'directional-cached-capacity.json',dict(completed=True,records=records,
        independent_integer_map_bytes=maps,manual_complete_leaf_pairs=True,
        sphere_partition_not_run=True,no_original_integrations=True,production_qualification=False))


@pytest.fixture(scope='module')
def prepared_case(tmp_path_factory):
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    import witwin.maxwell.mom.fmm_regional as regional
    location = os.environ.get('GSP_DIRECTIONAL_CACHE_ORIGINAL_REFERENCE')
    if location is None: pytest.skip('Pinned original weak reference required; never regenerated')
    reference_path = Path(location)
    assert reference_path.is_file()
    scene, fmm = declare_scene(), cached_config()
    config = mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3,
        pair_batch=4,far_pair_batch=16),operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==REFERENCE_SHA
    reference = torch.load(reference_path,weights_only=False,map_location='cuda')['reference']
    assert reference.shape==(36,36) and reference.dtype==torch.complex128
    folder = tmp_path_factory.mktemp('new-directional-cached-public')
    path = folder/'directional-cached-public.json'
    raw = dict(completed=False,production_qualification=False,physical_qualification=False,
        new_cached_policy_public_construction=True,
        automatic_plans_rebuilt_once_as_necessary_new_policy_dependency=True,
        old_planner_tests_not_replayed=True,original_reference_regenerated=False,
        original_reference_path=str(reference_path),original_reference_sha256=REFERENCE_SHA)
    _write(path,raw)
    calls = []
    exterior, interior = ScalarSystemBlocks.exterior_channels, regional.scalar_regional_channels
    def counted_exterior(self,*a,**kw): calls.append('exterior'); return exterior(self,*a,**kw)
    def counted_interior(*a,**kw): calls.append('interior'); return interior(*a,**kw)
    patch = pytest.MonkeyPatch()
    patch.setattr(ScalarSystemBlocks,'exterior_channels',counted_exterior)
    patch.setattr(regional,'scalar_regional_channels',counted_interior)
    try:
        prepared = mw.Simulation.mom(scene,frequency=1.73e8,config=config).prepare()
        op = prepared.solver.handle.actions
        raw.update(fmm_config=fmm.payload(),constructor_factory_calls=len(calls),physical_currents=op.shape[0],
            FAR_counts=[len(r.far.rectangles) for r in (op.exterior,*op.interiors)],
            exterior_positive_changing_links=op.exterior.test.plan_guard.selected_positive_links)
        _write(path,raw)
        assert not calls and op.shape==(36,36) and op.exterior.far.rectangles
        yield scene,fmm,prepared,op,reference,calls,folder,path,raw
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None)); _write(path,raw)
        raise
    finally:
        patch.undo()


@pytest.mark.parametrize('builder',('dag','sparse'))
@pytest.mark.parametrize('component',('cached_payload','coefficient_workspace'))
def test_new_directional_cached_component_refuses_before_maps(prepared_case,builder,component,monkeypatch):
    from witwin.maxwell.mom.fmm_capacity import plan_directional_reciprocal_near_storage
    from witwin.maxwell.mom.fmm_order import AutomaticDirectionalDAGPlan
    from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
    _,_,_,op,_,_,_,_,_ = prepared_case
    plan = op.exterior.test.plan_guard
    options = dict(integration=op.provider.config,near_storage='scalar_reciprocal_cached_dk_v1',
        near_coefficient_workspace_budget_bytes=4096)
    if component=='cached_payload':
        maps_only = plan_directional_reciprocal_near_storage(plan.region_plan.tree,plan.region_plan.near_nodes,
            budget_bytes=2**30,near_storage='scalar_reciprocal_cached_dk_v1',near_coefficient_cache_budget_bytes=0,
            near_coefficient_workspace_budget_bytes=4096)
        options['near_retained_budget_bytes'] = maps_only['retained_bytes']
    else:
        options['near_coefficient_workspace_budget_bytes'] = 1
    def forbidden(*a,**kw): pytest.fail('Cached component refusal reached map construction')
    if builder=='dag':
        monkeypatch.setattr(AutomaticDirectionalDAGPlan,'build_far',forbidden)
        build = plan.build_region
    else:
        monkeypatch.setattr(SparseDirectionalRWGProjection,'__init__',forbidden)
        options['quadrature_order'] = 12
        build = plan.region_plan.build_region
    with pytest.raises(MemoryError,match='before maps'): build(**options)


def test_new_public_directional_cached_native_solve_sparse_and_exact_restore(prepared_case,monkeypatch):
    from witwin.maxwell.mom.fmm_capacity import directional_retained_inventory,storage_bytes,plan_directional_reciprocal_near_storage
    from witwin.maxwell.mom.fmm_order import AutomaticDirectionalDAGPlan
    from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator,MultipoleOperator
    from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
    from witwin.maxwell.mom.runtime import SurfaceData,SystemOperator
    from witwin.maxwell.mom.packed_near import BoundedCachedReciprocalScalarNearChannels
    scene,fmm,prepared,op,reference,calls,folder,path,raw = prepared_case
    relative = lambda a,b: float(((a-b).norm()/b.norm()).detach())
    try:
        regions = (op.exterior,*op.interiors)
        assert all(isinstance(r.near,BoundedCachedReciprocalScalarNearChannels) for r in regions)
        columns = op.apply(torch.eye(36,device='cuda',dtype=op.dtype)).mT
        cold_calls = len(calls)
        generator = torch.Generator(device='cuda').manual_seed(482903)
        x = torch.randn((3,5,36),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
        dual = torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            forward = op.apply(x)
        adjoint = op.adjoint(dual)
        gradient = torch.autograd.grad(forward,x,dual)[0]
        first = torch.autograd.grad(op.apply(x).abs().square().sum(),x,create_graph=True)[0]
        second = torch.autograd.grad(first,x,dual)[0]
        lhs,rhs = (forward.conj()*dual).sum(),(x.conj()*adjoint).sum()
        errors = dict(columns=relative(columns,reference),forward=relative(forward,x@reference.mT),
            original_H=relative(adjoint,dual@reference.conj()),
            native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()),
            current_gradient=relative(gradient,adjoint),
            higher_current_gradient=relative(second,2*(dual@columns.mT)@columns.conj()))
        state = folder/'directional-cached-native-state.pt'
        torch.save(dict(columns=columns,x=x.detach(),dual=dual,forward=forward.detach(),adjoint=adjoint.detach(),
            gradient=gradient.detach(),higher=second.detach()),state)
        raw.update(errors=errors,native_errors_finite=all(math.isfinite(v) for v in errors.values()),
            cold_factory_calls=cold_calls,warm_factory_calls=len(calls)-cold_calls,saved_forward_tensors=len(saved),
            actual_native_state=str(state),actual_native_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest())
        _write(path,raw)
        assert raw['native_errors_finite']
        assert max(errors[k] for k in ('columns','forward','original_H'))<1e-8
        assert max(errors[k] for k in ('native_H','current_gradient','higher_current_gradient'))<1e-11
        assert cold_calls>0 and len(calls)==cold_calls and not saved
        receipts = []
        for region in regions:
            receipt = region.near.receipt()
            plan = plan_directional_reciprocal_near_storage(region.test.tree,region.near_nodes,
                budget_bytes=fmm.regional_near_budget_bytes,near_storage=fmm.near_storage,
                near_coefficient_cache_budget_bytes=fmm.near_coefficient_cache_budget_bytes,
                near_coefficient_workspace_budget_bytes=fmm.near_coefficient_workspace_budget_bytes)
            assert plan['retained_bytes']==region.retained_bytes==storage_bytes(region.tensors())
            assert receipt['retained_kernel_bytes']==plan['retained_near_kernel_bytes']
            assert receipt['cache_ready_blocks']==receipt['cache_planned_blocks']==plan['cache_planned_blocks']
            with monkeypatch.context() as patch:
                patch.setattr(region,'near_coefficient_cache_budget_bytes',fmm.near_coefficient_cache_budget_bytes-1)
                with pytest.raises(ValueError,match='budgets changed'): region.assert_current()
            receipts.append(receipt)
        result = prepared.run()
        current, stats = result.surface_currents.scattered_coefficients, result.stats()
        residual = relative(current@reference.mT,prepared.solver.rhs)
        state = folder/'directional-cached-solver-state.pt'
        torch.save(dict(current=current.detach(),rhs=prepared.solver.rhs.detach(),stats=stats),state)
        raw.update(actual_solver_stats=stats,complete_original_residual=residual,
            complete_original_residual_finite=math.isfinite(residual),residual_relative_gate=1e-10,
            near_receipts=receipts,retained_inventory=directional_retained_inventory(op),
            actual_solver_state=str(state),actual_solver_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
            warm_factory_calls=len(calls)-cold_calls)
        _write(path,raw)
        assert stats['converged'] and stats['error_budget']['checked']
        assert raw['complete_original_residual_finite'] and residual<=1e-10 and len(calls)==cold_calls
        artifact = folder/'directional-cached-result.pt'
        result.save(artifact)
        raw.update(artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        _write(path,raw)
        def forbidden(*a,**kw): pytest.fail('Directional cached restore constructed or solved')
        with monkeypatch.context() as patch:
            for cls in (AutomaticDirectionalDAGPlan,DirectionalSystemActions,DirectionalMultipoleOperator,MultipoleOperator,SystemOperator):
                patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored = mw.Result.load(artifact,scene=scene)
            assert restored.surface_currents.config.fmm==fmm
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            bad = copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters']['near_coefficient_cache_budget_bytes']-=1
            with pytest.raises(ValueError): SurfaceData.restore(bad,scene,1.73e8)
        # Reuse the new cached public root plan; build sparse maps once with zero
        # retained cache. No second automatic selection, reference or solve.
        sparse = op.exterior.test.plan_guard.region_plan.build_region(integration=op.provider.config,
            quadrature_order=fmm.quadrature_order,face_batch=fmm.face_batch,rhs_batch=fmm.rhs_batch,
            near_storage=fmm.near_storage,near_coefficient_workspace_budget_bytes=4096,
            near_coefficient_cache_budget_bytes=0)
        start = len(calls)
        sparse_forward, sparse_H = sparse.apply(x), sparse.adjoint(dual)
        sparse_error = dict(forward=relative(sparse_forward,op.exterior.apply(x)),
            H=relative(sparse_H,op.exterior.adjoint(dual)))
        sparse_state = folder/'directional-cached-sparse-state.pt'
        torch.save(dict(forward=sparse_forward.detach(),H=sparse_H.detach()),sparse_state)
        raw.update(sparse_errors=sparse_error,sparse_errors_finite=all(math.isfinite(v) for v in sparse_error.values()),
            sparse_factory_calls=len(calls)-start,sparse_near_receipt=sparse.near.receipt(),
            sparse_native_state=str(sparse_state),sparse_native_state_sha256=hashlib.sha256(sparse_state.read_bytes()).hexdigest())
        _write(path,raw)
        assert raw['sparse_errors_finite'] and max(sparse_error.values())<1e-8
        assert sparse.near.receipt()['retained_kernel_bytes']==0 and len(calls)>start
        raw.update(completed=True,exact_restore=True,whole_peak_qualified=False,
            independent_EM_reference=False,high_frequency_or_positive_changing_paths_qualified=False)
        _write(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None)); _write(path,raw)
        raise
