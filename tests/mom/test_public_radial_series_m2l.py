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
    spec=importlib.util.spec_from_file_location('_radial_original_public_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scene=module.scene()
    config=module.config(table_storage='radial_series_shared_radius_v1',
        near_storage='scalar_reciprocal_cached_dk_v1',leaf_factor_kernel='original',
        # Declare the original admitted p7 directly. Original p6..12 order
        # search is covered and is not replayed; actual p7 M2L gates still run.
        order=7,max_order=None)
    return scene,config



def test_new_radial_public_configuration_and_affected_payload_identity(tmp_path):
    location=os.environ.get('GSP_RADIAL_PUBLIC_ORIGINAL_CONFIG')
    if location is None:pytest.skip('Pinned original configuration required')
    path=Path(location)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GSP_RADIAL_PUBLIC_ORIGINAL_CONFIG_SHA256']
    spec=importlib.util.spec_from_file_location('witwin.maxwell.mom._radial_original_config',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    choices=({},dict(table_storage='compact_shared_signed_axis_v1'),
        dict(near_storage='scalar_reciprocal_cached_dk_v1'),
        dict(table_storage='compact',near_storage='scalar_reciprocal_streamed_dk_v1'))
    for options in choices:
        old=module.FMMConfig(**options);new=mw.FMMConfig(**options)
        assert old.payload()==new.payload() and old.construction()==new.construction()
        assert mw.FMMConfig.restore(old.payload())==new
    for near in ('full_matrix','scalar_dk_v1','scalar_reciprocal_dk_v1',
                 'scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
        config=mw.FMMConfig(table_storage='radial_series_shared_radius_v1',near_storage=near)
        payload=config.payload()
        assert payload['version']=='cartesian_fmm_radial_series_config_v1'
        assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==config
        for kind in ('version','policy','missing','extra','order','rows','bool_budget'):
            bad=copy.deepcopy(payload)
            if kind=='version':bad['version']='cartesian_fmm_cached_reciprocal_dk_config_v1'
            elif kind=='policy':bad['parameters']['table_storage']='compact_shared_signed_axis_v1'
            elif kind=='missing':bad['parameters'].pop('near_coefficient_cache_budget_bytes')
            elif kind=='extra':bad['parameters']['rounded_radius']=True
            elif kind=='order':bad['parameters']['order']=13
            elif kind=='rows':bad['parameters']['m2l_row_batch']=17
            else:bad['parameters']['table_budget_bytes']=True
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for options in (dict(periodic=mw.MoMPeriodicConfig(((1.,0.,0.),))),dict(dtype=torch.complex64)):
        with pytest.raises((ValueError,NotImplementedError)):
            mw.MoMConfig(operator_backend='fmm',fmm=mw.FMMConfig(table_storage='radial_series_shared_radius_v1'),
                riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none',fallback='none'),**options)
    _write(tmp_path/'radial-configuration.json',dict(completed=True,new_version_exact=True,
        all_five_near_identities_roundtrip=True,affected_old_payloads_exact=True,
        old_numerical_tests_not_replayed=True,production_qualification=False))


def test_new_radial_physical_pair_catalog_crosses1024_and_retains_nonorbit_vectors(tmp_path):
    from witwin.maxwell.mom.fmm_storage import radial_series_pair_catalog
    centers=torch.tensor([[0.,0.,0.],[.1875,.25,0.],[.3125,0.,0.],
        [0.,-.3125,0.],[-.1875,0.,.25]],device='cuda',dtype=torch.float64)
    changed=centers[2].clone();changed[0]=torch.nextafter(changed[0],changed.new_tensor(math.inf))
    centers=torch.cat((centers,changed[None]))
    pairs=tuple((i,0) for _ in range(264) for i in (1,2,3,4))+((5,0),)
    representatives,bindings=radial_series_pair_catalog(centers,pairs)
    assert representatives==(0,1056) and bindings==(0,)*1056+(1,)
    _write(tmp_path/'radial-physical-catalog.json',dict(completed=True,physical_pairs=1057,
        original_radius_owners=2,nonorbit_vectors_distinct=True,one_ulp_radius_distinct=True,
        catalog_chunk_limit=1024,production_qualification=False))


@pytest.mark.parametrize('order',(0,3,12))
def test_new_radial_capacity_native_owner_and_all_physical_vector_bytes(order,monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_capacity as capacity
    from witwin.maxwell.mom.cluster import build_support_octree,support_spheres
    from witwin.maxwell.mom.fmm_compact import RadialSeriesElectromagneticM2L
    from witwin.maxwell.mom.fmm_storage import radial_series_m2l_owner_bytes,radial_series_catalog_rule
    from witwin.maxwell.mom.physics import compile_medium
    axes=torch.arange(4,device='cuda',dtype=torch.float64)*.125-.1875
    coords=torch.cartesian_prod(axes,axes,axes)
    tree=build_support_octree(coords-.0005,coords+.0005,leaf_size=2,expansion_center_policy='binary_lattice_v1')
    observed=[];partition=capacity.sphere_block_partition
    def collect(*a,**kw):
        for aa,bb,far in partition(*a,**kw):
            observed.extend(zip(aa.tolist(),bb.tolist(),far.tolist()));yield aa,bb,far
    monkeypatch.setattr(capacity,'sphere_block_partition',collect)
    plan=capacity.plan_multipole_storage((tree,),order=order,separation_ratio=.23,budget_bytes=2**30,
        table_storage='radial_series_shared_radius_v1',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_streamed_dk_v1',near_coefficient_workspace_budget_bytes=4096)
    centers=support_spheres(tree)[0];pairs=[(a,b) for a,b,far in observed if far]
    assert pairs
    vectors=torch.stack([centers[a]-centers[b] for a,b in pairs])
    # Independent CUDA radius-bit oracle; no production catalog is called here.
    radius=torch.linalg.vector_norm(vectors,dim=-1)
    keys=set(radius.contiguous().view(torch.int64).cpu().tolist())
    medium=compile_medium(mw.Material(eps_r=2.1+.013j,mu_r=1.2),163e6)
    views=RadialSeriesElectromagneticM2L.create_many(vectors,medium,order)
    owned=storage_bytes(t for view in views for t in view.tensors())
    expected=len(keys)*radial_series_m2l_owner_bytes(order)+24*len(pairs)
    assert owned==expected==plan['categories']['m2l'] and len(keys)<len(pairs)
    assert plan['coefficient_sharing_rule']==radial_series_catalog_rule
    covered=set()
    for a,b,_ in observed:
        rows=tree.permutation[tree.starts[a]:tree.stops[a]].cpu().tolist()
        cols=tree.permutation[tree.starts[b]:tree.stops[b]].cpu().tolist()
        for pair in ((i,j) for i in rows for j in cols):
            assert pair not in covered;covered.add(pair)
    assert len(covered)==tree.size**2
    _write(tmp_path/'radial-capacity.json',dict(completed=True,order=order,plan=plan,
        actual_far_pairs=len(pairs),actual_original_radius_owners=len(keys),native_M2L_bytes=owned,
        every_physical_partition_entry_retained=True,independent_original_radius_bit_oracle=True,
        no_numerical_actions_or_old_planner_replay=True,production_qualification=False))


@pytest.mark.parametrize('rwg',(False,True))
def test_new_radial_order_accounting_both_paths_include_physical_vectors(rwg,monkeypatch,tmp_path):
    from witwin.maxwell.mom.cluster import build_support_octree,sphere_block_partition,support_spheres
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.fmm_storage import radial_series_m2l_owner_bytes
    import witwin.maxwell.mom.fmm_order as order_module
    axes=torch.arange(4,device='cuda',dtype=torch.float64)*.125-.1875
    coords=torch.cartesian_prod(axes,axes,axes)
    tree=build_support_octree(coords-.0005,coords+.0005,leaf_size=1,expansion_center_policy='binary_lattice_v1')
    native_fixture=tmp_path/'radial-order-native-fixture.pt'
    torch.save(dict(coords=coords,tree_tensors=tuple(tree.tensors())),native_fixture)
    medium=compile_medium(mw.Material(eps_r=1.9+.011j,mu_r=1.1),157e6)
    def bounds(delta,r1,r2,wave,orders,**kw):
        shape=(len(delta),len(orders)) if not isinstance(orders,int) else (len(delta),)
        zero=delta.new_zeros(shape)
        return dict(available=torch.ones(shape,device='cuda',dtype=torch.bool),green_absolute=zero,
            gradient_component_absolute=zero)
    monkeypatch.setattr(order_module,'multipole_truncation_bounds_many',bounds)
    monkeypatch.setattr(order_module,'multipole_truncation_bounds',bounds)
    monkeypatch.setattr(order_module,'multipole_block_error_log',lambda *a,**kw:coords.new_tensor(-30.))
    options=dict(rwg_statistics=(coords.new_zeros((tree.node_count,3)),),rwg_absolute_budget=1.,
        balancing_impedance=1.) if rwg else {}
    receipt=order_module.plan_multipole_order(((tree,medium),),mw.KernelErrorBudget(1e-6,1e-5),
        min_order=3,max_order=3,separation_ratio=.23,table_storage='radial_series_shared_radius_v1',**options)
    centers=support_spheres(tree)[0]
    pairs=[(a,b) for aa,bb,flags in sphere_block_partition(tree,separation_ratio=.23,pair_batch=32)
        for a,b,far in zip(aa.tolist(),bb.tolist(),flags.tolist()) if far]
    assert pairs,'The new structural ledger control requires actual original FAR pairs'
    vectors=torch.stack([centers[a]-centers[b] for a,b in pairs])
    keys=set(torch.linalg.vector_norm(vectors,dim=-1).contiguous().view(torch.int64).cpu().tolist())
    expected=len(keys)*radial_series_m2l_owner_bytes(3)+24*len(pairs)
    assert receipt['selected_order']==3 and receipt['table_bytes']==expected
    assert receipt['attempts'][0]['table_bytes']==expected and receipt['attempts'][0]['far_pairs']==len(pairs)
    _write(tmp_path/'radial-order-accounting.json',dict(completed=True,RWG_path=rwg,receipt=receipt,
        native_fixture=str(native_fixture),native_fixture_sha256=hashlib.sha256(native_fixture.read_bytes()).hexdigest(),
        frozen_original_partition_and_native_radius_bytes=True,original_kernel_contours_not_replayed=True,
        structural_mock_bounds_only=True,automatic_order_physics_qualified=False,production_qualification=False))


@pytest.mark.parametrize('component',('action','table','near','projection','coefficient_workspace'))
def test_new_radial_public_component_budgets_refuse_before_moments(component,monkeypatch):
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    declared,cfg=_fixture();options=dict(cfg.__dict__)
    key=dict(action='action_retained_budget_bytes',table='table_budget_bytes',near='near_budget_bytes',
        projection='projection_budget_bytes',coefficient_workspace='near_coefficient_workspace_budget_bytes')[component]
    options[key]=1
    def forbidden(*a,**kw):pytest.fail('Refused radial public capacity reached moments')
    monkeypatch.setattr(NestedRWGMultipoles,'__init__',forbidden)
    config=mw.MoMConfig(integration=IntegrationConfig(order=4,singular_order=3),operator_backend='fmm',
        fmm=mw.FMMConfig(**options),riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none',fallback='none'))
    with pytest.raises(MemoryError):mw.Simulation.mom(declared,frequency=1.43e8,config=config).prepare()


def test_new_public_radial_native_actions_solve_and_exact_restore(monkeypatch,tmp_path):
    from witwin.maxwell.mom.runtime import SystemOperator,SurfaceData
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    import witwin.maxwell.mom.fmm_regional as regional
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    location=os.environ.get('GSP_RADIAL_PUBLIC_ORIGINAL_REFERENCE')
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
    raw=dict(version='public_original_radial_series_m2l_v1',completed=False,
        production_qualification=False,physical_qualification=False,
        declared_fixed_order_reuses_original_admitted_p7=True,automatic_order_search_not_replayed=True)
    path=tmp_path/'radial-public.json';_write(path,raw)
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
        from witwin.maxwell.mom.fmm_compact import RadialSeriesElectromagneticM2L
        from witwin.maxwell.mom.fmm_storage import radial_series_m2l_owner_bytes
        views=[coupling._compact for region in regions for _,_,coupling in region.far]
        assert views and all(type(v) is RadialSeriesElectromagneticM2L for v in views)
        owners={id(v.owner):v.owner for v in views}
        native_M2L=storage_bytes(t for v in views for t in v.tensors())
        expected_M2L=sum(radial_series_m2l_owner_bytes(v.order) for v in owners.values())+24*len(views)
        assert native_M2L==expected_M2L==op.capacity_plan['categories']['m2l']
        assert len(owners)<len(views)
        raw.update(actual_physical_far_pairs=len(views),actual_radial_owners=len(owners),
            native_M2L_bytes=native_M2L,all_original_physical_vectors_retained=True,
            full_cartesian_jets_retained=False)
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
        actual=tmp_path/'radial-public-native-state.pt'
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
        state=tmp_path/'radial-public-solver-state.pt'
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
        # The new geometry-bound standalone entry must not allocate a finite
        # mask over arbitrary leading RHS before its accepted primitive chunks.
        coupling=next(c for region in regions for _,_,c in region.far)
        count=coupling.test.count
        moments=torch.ones((17,2,count,4),device='cuda',dtype=op.dtype)
        observed=[];finite=torch.isfinite
        def bounded_finite(tensor):
            observed.append(tensor.numel())
            assert tensor.numel()<=4*2*count*4,'Unadmitted standalone coupling finite mask'
            return finite(tensor)
        with monkeypatch.context() as patch:
            patch.setattr(torch,'isfinite',bounded_finite)
            standalone=coupling.action(moments)
        assert standalone.shape==moments.shape and observed
        standalone_state=tmp_path/'radial-standalone-large-rhs.pt'
        torch.save(dict(moments=moments,actual=standalone),standalone_state)
        raw.update(standalone_RHS=17,max_standalone_finite_mask_elements=max(observed),
            standalone_native_state=str(standalone_state),
            standalone_native_state_sha256=hashlib.sha256(standalone_state.read_bytes()).hexdigest())
        _write(path,raw)
        artifact=tmp_path/'radial-public-result.pt';result.save(artifact)
        def forbidden(*a,**kw):pytest.fail('Cache restore assembled or solved')
        with monkeypatch.context() as patch:
            for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(artifact,scene=declared)
            assert restored.surface_currents.config.fmm==cfg
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            bad=copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters']['table_storage']='compact_shared_signed_axis_v1'
            with pytest.raises(ValueError):SurfaceData.restore(bad,declared,1.43e8)
        raw.update(completed=True,artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            exact_restore=True,whole_peak_qualified=False,independent_EM_reference=False)
        _write(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None))
        _write(path,raw)
        raise
