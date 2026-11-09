"""New public near recomputation identity, planned storage and original weak solve."""
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
from witwin.maxwell.mom.fmm_capacity import plan_multipole_storage,storage_bytes,multipole_action_tensors

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def write_raw(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def config(**options):
    values=dict(order=6,max_order=12,table_storage='compact_shared_displacement_v1',
        shift_storage='separable_v1',near_storage='scalar_reciprocal_streamed_dk_v1',far_action_batch=32,
        leaf_size=8,separation_ratio=.25,near_coefficient_workspace_budget_bytes=8192,
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),kernel_bound_rule='direct_first_wave_min_v1')
    values.update(options)
    return mw.FMMConfig(**values)


def test_new_streamed_identity_and_affected_old_workspace_omission():
    names=('GSP_STREAMED_ORIGINAL_CONFIG','GSP_STREAMED_ORIGINAL_CONFIG_SHA256')
    if all(os.environ.get(n) is None for n in names):pytest.skip('Pinned whole pre-integration config required')
    assert all(os.environ.get(n) for n in names)
    path=Path(os.environ[names[0]])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ[names[1]]
    spec=importlib.util.spec_from_file_location('witwin.maxwell.mom._gsp_streamed_original_config',path)
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    # This targets only the newly added field's old-schema omission/parser
    # effect. No old solver, numerical reference, archive or planner is run.
    choices=[{},dict(near_storage='scalar_dk_v1'),dict(near_storage='scalar_reciprocal_dk_v1'),
        dict(table_storage='compact_shared_displacement_v1'),
        dict(table_storage='compact',far_action_batch=2),
        dict(table_storage='compact',expansion_center_policy='dyadic_cell_v1')]
    for kwargs in choices:
        old=legacy.FMMConfig(**kwargs);new=mw.FMMConfig(**kwargs)
        assert old.payload()==new.payload() and old.construction()==new.construction()
        assert mw.FMMConfig.restore(old.payload()).payload()==old.payload()
        assert 'near_coefficient_workspace_budget_bytes' not in new.payload()['parameters']
    cfg=config();payload=cfg.payload()
    assert payload['version']=='cartesian_fmm_streamed_reciprocal_dk_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==cfg
    assert cfg.construction()['near_coefficient_workspace_budget_bytes']==8192
    for kind in ('policy','version','missing','extra','bool_budget'):
        bad=copy.deepcopy(payload)
        if kind=='policy':bad['parameters']['near_storage']='scalar_reciprocal_dk_v1'
        elif kind=='version':bad['version']='cartesian_fmm_shared_compact_config_v1'
        elif kind=='missing':bad['parameters'].pop('near_coefficient_workspace_budget_bytes')
        elif kind=='extra':bad['parameters']['approximate_near']=True
        else:bad['parameters']['near_coefficient_workspace_budget_bytes']=True
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    with pytest.raises(ValueError):mw.FMMConfig(near_coefficient_workspace_budget_bytes=8192)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none'),
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    with pytest.raises(ValueError,match='complex128'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none'),dtype=torch.complex64)


def test_new_streamed_structural_maps_count_without_old_capacity_replay(monkeypatch,tmp_path):
    from witwin.maxwell.mom.cluster import build_support_octree
    import witwin.maxwell.mom.fmm_capacity as capacity
    axes=[torch.arange(n,device='cuda',dtype=torch.float64) for n in (4,4,3)]
    coords=torch.cartesian_prod(*axes)*.113+torch.tensor([.043,-.107,.029],device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(11843)
    coords+=torch.rand(coords.shape,device='cuda',dtype=torch.float64,generator=generator)*.003
    widths=.001+torch.rand(coords.shape,device='cuda',dtype=torch.float64,generator=generator)*.002
    tree=build_support_octree(coords-widths,coords+widths,leaf_size=2)
    sizes=(tree.stops-tree.starts).cpu().tolist()
    observed=[]
    original=capacity.sphere_block_partition
    def collect(*args,**kwargs):
        for aa,bb,far in original(*args,**kwargs):
            observed.extend(zip(aa.tolist(),bb.tolist(),far.tolist()))
            yield aa,bb,far
    monkeypatch.setattr(capacity,'sphere_block_partition',collect)
    plan=plan_multipole_storage((tree,),order=2,separation_ratio=.27,budget_bytes=2**30,
        table_storage='compact_shared_displacement_v1',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_streamed_dk_v1',near_coefficient_workspace_budget_bytes=4096)
    canonical=[(a,b) for a,b,far in observed if not far and a<=b]
    expected=sum(16*(sizes[a]+sizes[b]) for a,b in canonical)
    if canonical:expected+=8*(2*len({a for a,b in canonical})+1+3*len(canonical))
    omitted=sum(32*sizes[a]*sizes[b] for a,b in canonical)
    assert plan['categories']['near']==expected and plan['retained_near_kernel_bytes']==0 and omitted>0
    write_raw(tmp_path/'streamed-capacity.json',dict(completed=True,plan=plan,
        expected_map_CSR_bytes=expected,omitted_original_coefficient_bytes=omitted,
        canonical_near_pairs=len(canonical),partition_observed_once=True,
        old_capacity_planner_not_replayed=True,physical_qualification=False,production_qualification=False))


@pytest.mark.parametrize('component',('action','near','projection','coefficient_workspace'))
def test_new_streamed_capacity_refuses_before_moments_or_original_near(component,monkeypatch):
    from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    basis=RWGBasis(sphere_mesh(radius=.017,center=(.031,.014,-.011),subdivisions=0,device='cuda',dtype=torch.float64))
    exterior=compile_medium(mw.Material(),113e6,background=True)
    interior=compile_medium(mw.Material(eps_r=1.63,mu_r=1.12,sigma_e=.003),113e6)
    provider=ScalarSystemBlocks(SurfaceSystem.create([(0,'new-streamed-capacity',basis,interior)]),
        exterior,IntegrationConfig(order=4,singular_order=3))
    def forbidden(*args,**kwargs):raise AssertionError('Refused streamed capacity reached numerical setup')
    monkeypatch.setattr(NestedRWGMultipoles,'__init__',forbidden)
    options=dict(order=2,leaf_size=8,separation_ratio=.25,table_storage='compact',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_streamed_dk_v1')
    keys=dict(action='action_retained_budget_bytes',near='near_budget_bytes',projection='projection_budget_bytes',
        coefficient_workspace='near_coefficient_workspace_budget_bytes')
    options[keys[component]]=1
    with pytest.raises(MemoryError):MultipoleSystemActions(provider,**options)


def test_new_public_streamed_near_all_columns_native_H_gradients_and_solve(tmp_path,monkeypatch):
    from witwin.maxwell.mom.runtime import SystemOperator,SurfaceData
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    raw=dict(version='new_public_streamed_reciprocal_near_v1',completed=False,
        production_qualification=False,physical_qualification=False)
    path=tmp_path/'streamed-public.json';write_raw(path,raw)
    scene=mw.Scene(structures=[
        mw.Structure(mw.Sphere(radius=.008,position=(-.253,.027,.013)),mw.Material(eps_r=1.9+.012j,mu_r=1.18)),
        mw.Structure(mw.Sphere(radius=.011,position=(.247,-.023,-.009)),mw.Material(eps_r=2.8+.017j,mu_r=1.29))],
        sources=[mw.PlaneWave()])
    cfg=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=config(),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    try:
        prepared=mw.Simulation.mom(scene,frequency=1.17e8,config=cfg).prepare()
        op=prepared.solver.handle.actions
        raw.update(capacity_plan=op.capacity_plan,order_plan=op.order_plan,fmm_config=cfg.fmm.payload())
        write_raw(path,raw)
        from witwin.maxwell.mom.packed_near import StreamedReciprocalScalarNearChannels
        assert op.exterior.far and op.near_storage=='scalar_reciprocal_streamed_dk_v1'
        for region in (op.exterior,*op.interiors):
            assert isinstance(region.near,StreamedReciprocalScalarNearChannels)
            assert region.near.receipt()['retained_kernel_bytes']==0
        assert op.capacity_plan['retained_near_kernel_bytes']==0
        edges=torch.arange(op.shape[0]//2,device='cuda')
        reference=op.provider(edges,edges)
        reference_artifact=tmp_path/'streamed-native-reference.pt'
        saved=dict(reference=reference.detach(),frequency_Hz=1.17e8)
        torch.save(saved,reference_artifact)
        relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
        columns=op.apply(torch.eye(op.shape[0],device='cuda',dtype=op.dtype)).mT
        generator=torch.Generator(device='cuda').manual_seed(10537)
        x=torch.randn((3,5,op.shape[0]),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
        y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        forward=op.apply(x);hermitian=op.adjoint(y)
        lhs=(forward.conj()*y).sum();rhs=(x.conj()*hermitian).sum()
        native_error=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
        errors=dict(all_columns=relative(columns,reference),forward=relative(forward,x@reference.mT),
            adjoint=relative(hermitian,y@reference.conj()),native_H=native_error)
        saved.update(columns=columns.detach(),x=x.detach(),y=y.detach(),forward=forward.detach(),adjoint=hermitian.detach())
        torch.save(saved,reference_artifact)
        raw.update(errors=errors,reference_constructed_once=True,reference_artifact=str(reference_artifact))
        write_raw(path,raw)
        assert max(errors[k] for k in ('all_columns','forward','adjoint'))<1e-8 and native_error<1e-11
        captured=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:captured.append(t) or t,lambda t:t):value=op.apply(x)
        assert not captured
        gradient=torch.autograd.grad(value,x,y)[0]
        first=torch.autograd.grad(op.apply(x).abs().square().sum(),x,create_graph=True)[0]
        direction=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        second=torch.autograd.grad(first,x,direction)[0]
        gradient_error=relative(gradient,hermitian)
        second_error=relative(second,2*(direction@columns.mT)@columns.conj())
        raw.update(current_gradient_error=gradient_error,higher_current_gradient_error=second_error,saved_forward_tensors=0)
        write_raw(path,raw)
        assert gradient_error<1e-11 and second_error<1e-11
        owned=storage_bytes(multipole_action_tensors(op))
        assert owned==op.capacity_plan['planned_bytes']
        result=prepared.run()
        assert result.stats()['converged'] and result.stats()['error_budget']['checked']
        current=result.surface_currents.scattered_coefficients
        residual=relative(current@reference.mT,prepared.solver.rhs)
        raw.update(complete_original_residual=residual,residual_relative_gate=1e-10,
            residual_absolute_gate=0.,actual_owned_bytes=owned)
        write_raw(path,raw);assert residual<=1e-10
        artifact=tmp_path/'streamed-near-result.pt';result.save(artifact)
        def forbidden(*args,**kwargs):raise AssertionError('Streamed near restore assembled or solved')
        with monkeypatch.context() as patch:
            for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(artifact,scene=scene)
            assert restored.surface_currents.config.fmm==cfg.fmm
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            bad=copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters']['near_coefficient_workspace_budget_bytes']=4096
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,1.17e8)
        raw.update(completed=True,artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            exact_restore=True,whole_peak_qualified=False,original_scalar_reference_is_not_independent_EM_reference=True)
        write_raw(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None))
        write_raw(path,raw)
        raise
