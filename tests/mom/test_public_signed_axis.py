"""New public signed-axis sharing with exact capacity and native weak evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_capacity import storage_bytes, multipole_action_tensors

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def raw_write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def config(**options):
    values=dict(order=6,max_order=12,table_storage='compact_shared_signed_axis_v1',
        shift_storage='separable_v1',near_storage='scalar_reciprocal_streamed_dk_v1',
        expansion_center_policy='binary_lattice_v1',far_action_batch=16,m2l_row_batch=8,
        leaf_size=2,separation_ratio=.2,near_coefficient_workspace_budget_bytes=4096,
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),kernel_bound_rule='direct_first_wave_min_v1')
    values.update(options)
    return mw.FMMConfig(**values)


def scene():
    vertices=torch.tensor([[0.,0.,0.],[.009,.001,.0005],[.002,.011,.001],[.001,.002,.013]],
        device='cuda',dtype=torch.float64)
    # Split one original edge and both adjacent faces; the closed physical
    # tetrahedral boundary is preserved, while RWG support centers vary.
    vertices=torch.cat((vertices,(vertices[0]+vertices[1])[None,:]*.5))
    faces=torch.tensor([[0,2,4],[4,2,1],[0,4,3],[4,1,3],[0,3,2],[1,2,3]],device='cuda')
    positions=((-0.3125,-0.3125,-0.15625),(0.3125,-0.3125,0.15625),
        (-0.3125,0.3125,0.15625),(0.3125,0.3125,-0.15625))
    materials=((2.3+.019j,1.21,.006),(2.9+.023j,1.34,.008),
        (3.2+.029j,1.46,.010),(3.7+.031j,1.57,.012))
    return mw.Scene(structures=[mw.Structure(name='signed-axis-'+str(index),geometry=mw.Mesh(
        vertices+torch.tensor(position,device='cuda',dtype=torch.float64),faces.clone(),recenter=False,fill_mode='solid'),
        material=mw.Material(eps_r=epsilon,mu_r=mu,sigma_m=loss))
        for index,(position,(epsilon,mu,loss)) in enumerate(zip(positions,materials,strict=True))],sources=[mw.PlaneWave()])


def test_new_signed_axis_version_and_affected_payload_branches(tmp_path):
    names=('GSP_SIGNED_AXIS_ORIGINAL_CONFIG','GSP_SIGNED_AXIS_ORIGINAL_CONFIG_SHA256')
    if all(os.environ.get(n) is None for n in names):pytest.skip('Pinned pre-integration configuration required')
    assert all(os.environ.get(n) for n in names)
    path=Path(os.environ[names[0]])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ[names[1]]
    spec=importlib.util.spec_from_file_location('witwin.maxwell.mom._gsp_signed_axis_original_config',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    # Only payload branches affected by the new highest-precedence schema are
    # compared. There is no old planner/reference/producer/archive replay.
    choices=({},dict(table_storage='compact_shared_displacement_v1'),
        dict(table_storage='compact',far_action_batch=2),
        dict(table_storage='compact',expansion_center_policy='binary_lattice_v1'),
        dict(table_storage='compact',near_storage='scalar_reciprocal_streamed_dk_v1'))
    for kwargs in choices:
        expected=old.FMMConfig(**kwargs)
        actual=mw.FMMConfig(**kwargs)
        assert actual.payload()==expected.payload() and actual.construction()==expected.construction()
    cfg=config();payload=cfg.payload()
    assert payload['version']=='cartesian_fmm_signed_axis_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==cfg
    for kind in ('policy','version','missing','extra','bool_budget','high_order','high_rows'):
        bad=copy.deepcopy(payload)
        if kind=='policy':bad['parameters']['table_storage']='compact_shared_displacement_v1'
        elif kind=='version':bad['version']='cartesian_fmm_streamed_reciprocal_dk_config_v1'
        elif kind=='missing':bad['parameters'].pop('near_coefficient_workspace_budget_bytes')
        elif kind=='extra':bad['parameters']['rounded_symmetry']=True
        elif kind=='bool_budget':bad['parameters']['near_coefficient_workspace_budget_bytes']=True
        elif kind=='high_order':bad['parameters']['max_order']=13
        else:bad['parameters']['m2l_row_batch']=17
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for version in ('cartesian_fmm_config_v7','cartesian_fmm_shared_compact_config_v1',
        'cartesian_fmm_binary_lattice_config_v1','cartesian_fmm_batched_compact_config_v1'):
        bad=copy.deepcopy(payload);bad['version']=version
        with pytest.raises(ValueError,match='explicit configuration version'):mw.FMMConfig.restore(bad)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none',fallback='none'),
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    with pytest.raises(ValueError,match='complex128'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none',fallback='none'),dtype=torch.complex64)
    raw_write(tmp_path/'signed-axis-configuration.json',dict(new_schema_exact=True,
        affected_old_payloads_exact=True,no_old_numeric_replay=True,production_qualification=False))


def test_new_signed_axis_capacity_counts_actual_physical_catalog_once(tmp_path,monkeypatch):
    import witwin.maxwell.mom.fmm_capacity as capacity
    from witwin.maxwell.mom.cluster import build_support_octree,support_spheres
    from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes,signed_axis_catalog_rule
    axes=torch.arange(4,device='cuda',dtype=torch.float64)*.125-.1875
    coords=torch.cartesian_prod(axes,axes,axes)
    tree=build_support_octree(coords-.0005,coords+.0005,leaf_size=2,expansion_center_policy='binary_lattice_v1')
    observed=[];original=capacity.sphere_block_partition
    def collect(*args,**kwargs):
        for aa,bb,far in original(*args,**kwargs):
            observed.extend(zip(aa.tolist(),bb.tolist(),far.tolist()))
            yield aa,bb,far
    monkeypatch.setattr(capacity,'sphere_block_partition',collect)
    plan=capacity.plan_multipole_storage((tree,),order=3,separation_ratio=.23,budget_bytes=2**30,
        table_storage='compact_shared_signed_axis_v1',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_streamed_dk_v1',near_coefficient_workspace_budget_bytes=4096)
    centers=support_spheres(tree)[0];keys=set();covered=set()
    far_pairs=[(a,b) for a,b,far in observed if far]
    # Independent host IEEE keys check exact native displacement values. Host
    # use is oracle metadata, not a production numerical fallback.
    for a,b in far_pairs:
        delta=centers[a]-centers[b]
        keys.add(tuple(sorted(abs(float(v)) for v in delta.cpu())))
        test=tree.permutation[tree.starts[a]:tree.stops[a]].cpu().tolist()
        source=tree.permutation[tree.starts[b]:tree.stops[b]].cpu().tolist()
        for pair in ((i,j) for i in test for j in source):
            assert pair not in covered;covered.add(pair)
    expected=len(keys)*m2l_storage_bytes(3,8,'compact')
    assert far_pairs and len(keys)<len(far_pairs)
    assert plan['categories']['m2l']==expected
    assert plan['coefficient_sharing_rule']==signed_axis_catalog_rule
    assert plan['retained_near_kernel_bytes']==0 and plan['physical_pairs_retained']
    raw_write(tmp_path/'signed-axis-capacity.json',dict(completed=True,plan=plan,
        original_far_pairs=len(far_pairs),exact_signed_axis_owners=len(keys),
        independent_actual_displacement_keys=True,physical_far_entry_count=len(covered),
        actual_partition_observed_once=True,no_old_planner_replay=True,
        physical_qualification=False,production_qualification=False))


@pytest.mark.parametrize('component',('action','table','near','projection','coefficient_workspace'))
def test_new_signed_axis_budgets_refuse_before_numerical_setup(component,monkeypatch):
    from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    bases=[RWGBasis(sphere_mesh(radius=.018,center=(offset,-.013,.007),subdivisions=0,device='cuda',dtype=torch.float64))
        for offset in (-.4,.4)]
    exterior=compile_medium(mw.Material(),149e6,background=True)
    interior=compile_medium(mw.Material(eps_r=1.79+.011j,mu_r=1.16),149e6)
    provider=ScalarSystemBlocks(SurfaceSystem.create([(index,'new-signed-axis-capacity-'+str(index),basis,interior)
        for index,basis in enumerate(bases)]),
        exterior,IntegrationConfig(order=4,singular_order=3))
    def forbidden(*args,**kwargs):raise AssertionError('Refused signed-axis capacity reached numerical setup')
    monkeypatch.setattr(NestedRWGMultipoles,'__init__',forbidden)
    options=dict(order=2,leaf_size=2,separation_ratio=.4,table_storage='compact_shared_signed_axis_v1',
        expansion_center_policy='binary_lattice_v1',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_streamed_dk_v1')
    key=dict(action='action_retained_budget_bytes',table='table_budget_bytes',near='near_budget_bytes',
        projection='projection_budget_bytes',coefficient_workspace='near_coefficient_workspace_budget_bytes')[component]
    options[key]=1
    with pytest.raises(MemoryError):MultipoleSystemActions(provider,**options)


def test_new_binary_zero_partition_span_preserves_complete_physical_leaf(tmp_path):
    from witwin.maxwell.mom.cluster import build_support_octree,support_spheres
    widths=torch.tensor([[.002,.003,.004],[.001,.002,.003],[.0015,.0025,.0035]],device='cuda',dtype=torch.float64)
    center=torch.tensor([.3125,-.3125,.15625],device='cuda',dtype=torch.float64)
    lower=center-widths;upper=center+widths
    tree=build_support_octree(lower,upper,leaf_size=1,expansion_center_policy='binary_lattice_v1')
    assert tree.node_count==1 and tree.size==3 and tree.receipt()['oversized_leaves']==1
    torch.testing.assert_close(tree.lower[0],lower.amin(0),rtol=0,atol=0)
    torch.testing.assert_close(tree.upper[0],upper.amax(0),rtol=0,atol=0)
    actual,radii=support_spheres(tree)
    corners=torch.tensor([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],device='cuda',dtype=torch.bool)
    boxes=torch.where(corners,tree.upper[0],tree.lower[0])
    assert bool((torch.linalg.vector_norm(boxes-actual[0],dim=-1)<=radii[0]).all())
    scaled=torch.ldexp(actual,actual.new_tensor(-tree.binary_lattice_exponent,dtype=torch.int64))
    assert bool((scaled==scaled.round()).all()) and bool((scaled.abs()<=2**51).all())
    with pytest.raises(ValueError,match='nonzero partition span'):
        build_support_octree(center[None,:],center[None,:],expansion_center_policy='binary_lattice_v1')
    raw_write(tmp_path/'binary-zero-span.json',dict(completed=True,
        one_original_complete_leaf=True,no_partition_point_jitter=True,corner_enclosure=True,
        exact_native_binary_center=True,collapsed_support_refused=True,production_qualification=False))


def test_new_public_signed_axis_original_columns_native_H_gradients_solve_and_restore(tmp_path,monkeypatch):
    from witwin.maxwell.mom.runtime import SystemOperator,SurfaceData
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    from witwin.maxwell.mom.fmm_compact import SignedAxisCompactElectromagneticM2L
    raw=dict(version='new_public_signed_axis_streamed_binary_v1',completed=False,
        production_qualification=False,physical_qualification=False)
    path=tmp_path/'signed-axis-public.json';raw_write(path,raw)
    declared_scene=scene()
    cfg=mw.MoMConfig(integration=IntegrationConfig(order=4,singular_order=3),operator_backend='fmm',
        fmm=config(leaf_factor_kernel='original'),riesz_storage='local_rwg_v1',stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',
            fallback='none',rtol=1e-12,atol=0.,maxiter=240,restart=120))
    try:
        prepared=mw.Simulation.mom(declared_scene,frequency=1.43e8,config=cfg).prepare()
        op=prepared.solver.handle.actions
        regions=(op.exterior,*op.interiors)
        views=[coupling._compact for region in regions for _,_,coupling in region.far]
        assert views and all(type(view) is SignedAxisCompactElectromagneticM2L for view in views)
        owners={id(view._signed_axis_owner) for view in views}
        assert len(owners)<len(views)
        assert op.capacity_plan['retained_near_kernel_bytes']==0
        raw.update(capacity_plan=op.capacity_plan,order_plan=op.order_plan,fmm_config=cfg.fmm.payload(),
            actual_far_pairs=len(views),actual_original_owners=len(owners),
            additional_view_owned_tensor_bytes=sum(view.receipt()['additional_owned_tensor_bytes'] for view in views))
        raw_write(path,raw)
        edges=torch.arange(op.shape[0]//2,device='cuda')
        reference=op.provider(edges,edges)
        reference_artifact=tmp_path/'signed-axis-public-native-reference.pt'
        saved=dict(reference=reference.detach(),frequency_Hz=1.43e8)
        torch.save(saved,reference_artifact)
        relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
        columns=op.apply(torch.eye(op.shape[0],device='cuda',dtype=op.dtype)).mT
        generator=torch.Generator(device='cuda').manual_seed(23879)
        x=torch.randn((3,5,op.shape[0]),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
        y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        forward=op.apply(x);hermitian=op.adjoint(y)
        lhs=(forward.conj()*y).sum();rhs=(x.conj()*hermitian).sum()
        errors=dict(all_columns=relative(columns,reference),forward=relative(forward,x@reference.mT),
            adjoint=relative(hermitian,y@reference.conj()),
            native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
        saved.update(columns=columns.detach(),x=x.detach(),y=y.detach(),forward=forward.detach(),adjoint=hermitian.detach())
        torch.save(saved,reference_artifact)
        raw.update(errors=errors,reference_constructed_once=True,reference_artifact=str(reference_artifact))
        raw_write(path,raw)
        assert max(errors[k] for k in ('all_columns','forward','adjoint'))<1e-8 and errors['native_H']<1e-11
        captured=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:captured.append(t) or t,lambda t:t):value=op.apply(x)
        assert not captured
        gradient=torch.autograd.grad(value,x,y)[0]
        first=torch.autograd.grad(op.apply(x).abs().square().sum(),x,create_graph=True)[0]
        direction=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        second=torch.autograd.grad(first,x,direction)[0]
        raw.update(current_gradient_error=relative(gradient,hermitian),
            higher_current_gradient_error=relative(second,2*(direction@columns.mT)@columns.conj()),saved_forward_tensors=0)
        raw_write(path,raw)
        assert raw['current_gradient_error']<1e-11 and raw['higher_current_gradient_error']<1e-11
        # NEW signed-axis B1 routing must still chunk RHS at4 and use the
        # guarded signed-axis batch action; it reuses the original moments.
        region=op.exterior
        expected_B16=region.apply(x.detach())
        old_batch=region.far_action_batch;old_binding=region._binding
        region.far_action_batch=1;region._binding=region._state()
        try:
            raw['signed_axis_B1_routing_error']=relative(region.apply(x.detach()),expected_B16)
            assert raw['signed_axis_B1_routing_error']<1e-11
        finally:
            region.far_action_batch=old_batch;region._binding=old_binding
        owned=storage_bytes(multipole_action_tensors(op))
        assert owned==op.capacity_plan['planned_bytes']
        result=prepared.run()
        assert result.stats()['converged'] and result.stats()['error_budget']['checked']
        current=result.surface_currents.scattered_coefficients
        residual=relative(current@reference.mT,prepared.solver.rhs)
        raw.update(complete_original_residual=residual,residual_relative_gate=1e-10,
            residual_absolute_gate=0.,actual_owned_bytes=owned)
        raw_write(path,raw);assert residual<=1e-10
        artifact=tmp_path/'signed-axis-public-result.pt';result.save(artifact)
        def forbidden(*args,**kwargs):raise AssertionError('Signed-axis restore assembled or solved')
        with monkeypatch.context() as patch:
            for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(artifact,scene=declared_scene)
            assert restored.surface_currents.config.fmm==cfg.fmm
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            bad=copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters']['table_storage']='compact_shared_displacement_v1'
            with pytest.raises(ValueError):SurfaceData.restore(bad,declared_scene,1.43e8)
        raw.update(completed=True,artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(reference_artifact.read_bytes()).hexdigest(),exact_restore=True,
            whole_peak_qualified=False,original_scalar_reference_is_not_independent_EM_reference=True)
        raw_write(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None))
        raw_write(path,raw)
        raise
