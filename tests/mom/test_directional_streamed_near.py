"""New directional near recomputation with one public original weak reference."""
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

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def raw_write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def directional_config():
    return mw.DirectionalFMMConfig(near_storage='scalar_reciprocal_streamed_dk_v1',
        near_coefficient_workspace_budget_bytes=4096,leaf_size=3,separation_ratio=.15,
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        regional_near_budget_bytes=1024**3,tensor_workspace_budget_bytes=512*1024**2)


def declare_scene():
    vertices=torch.tensor([[0.,0.,0.],[.013,.001,.0005],[.002,.015,.001],[.001,.003,.017]],
        device='cuda',dtype=torch.float64)
    faces=torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
    entries=[((-.613,.017,.023),1.,2.1+.013j,1.24,.007),
        ((.031,-.019,.011),1.1,2.7+.017j,1.37,.011),
        ((.729,.029,-.017),.9,3.4+.023j,1.53,.009)]
    structures=[]
    for index,(offset,scale,epsilon,mu,sigma_m) in enumerate(entries):
        geometry=mw.Mesh(scale*vertices+vertices.new_tensor(offset),faces.clone(),recenter=False,fill_mode='solid')
        structures.append(mw.Structure(name='directional-streamed-'+str(index),geometry=geometry,
            material=mw.Material(eps_r=epsilon,mu_r=mu,sigma_m=sigma_m)))
    return mw.Scene(structures=structures,sources=[mw.PlaneWave()])


def test_new_directional_streamed_version_and_affected_old_field_omissions(tmp_path):
    keys=('GSP_DIRECTIONAL_STREAMED_ORIGINAL_CONFIG','GSP_DIRECTIONAL_STREAMED_ORIGINAL_CONFIG_SHA256')
    if all(os.environ.get(n) is None for n in keys):pytest.skip('Pinned whole original config required')
    assert all(os.environ.get(n) for n in keys)
    path=Path(os.environ[keys[0]]);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ[keys[1]]
    spec=importlib.util.spec_from_file_location('witwin.maxwell.mom._gsp_directional_streamed_original_config',path)
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    # Only the new field's omission and restoration impact is compared. No
    # previous numerical, planner, successful producer or archive is rerun.
    for policy in ('scalar_dk_v1','scalar_reciprocal_dk_v1'):
        old=original.DirectionalFMMConfig(near_storage=policy)
        new=mw.DirectionalFMMConfig(near_storage=policy)
        assert old.payload()==new.payload()
        assert 'near_coefficient_workspace_budget_bytes' not in new.payload()['parameters']
        assert mw.FMMConfig.restore(old.payload()).payload()==old.payload()
    cfg=directional_config();payload=cfg.payload()
    assert payload['version']=='automatic_directional_streamed_reciprocal_dag_fmm_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==cfg
    for field in payload['parameters']:
        bad=copy.deepcopy(payload);bad['parameters'].pop(field)
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for value in (False,0,-1,4096.):
        bad=copy.deepcopy(payload);bad['parameters']['near_coefficient_workspace_budget_bytes']=value
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for version in ('automatic_directional_dag_fmm_config_v1','automatic_directional_reciprocal_dag_fmm_config_v1'):
        bad=copy.deepcopy(payload);bad['version']=version
        bad['parameters'].pop('near_coefficient_workspace_budget_bytes')
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    bad=copy.deepcopy(payload);bad['parameters']['near_storage']='scalar_reciprocal_dk_v1'
    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    bad=copy.deepcopy(payload);bad['parameters']['unbounded_near']=True
    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    with pytest.raises(ValueError):mw.DirectionalFMMConfig(near_coefficient_workspace_budget_bytes=4096)
    options=dict(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='none'))
    with pytest.raises(ValueError,match='complex128'):mw.MoMConfig(**options,dtype=torch.complex64)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(**options,periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    raw_write(tmp_path/'streamed-directional-config.json',dict(completed=True,payload=payload,
        old_two_payloads_exact=True,new_field_impact_only=True,production_qualification=False))


def test_new_directional_streamed_manual_complete_leaf_inventory(tmp_path):
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.fmm_capacity import plan_directional_reciprocal_near_storage
    grid=torch.cartesian_prod(*[torch.arange(n,device='cuda',dtype=torch.float64) for n in (4,3,4)])
    generator=torch.Generator(device='cuda').manual_seed(9163)
    centers=grid*.071+grid.new_tensor([.039,-.053,.017])
    centers+=torch.rand(centers.shape,device='cuda',dtype=torch.float64,generator=generator)*.002
    width=.001+torch.rand(centers.shape,device='cuda',dtype=torch.float64,generator=generator)*.001
    tree=build_support_octree(centers-width,centers+width,leaf_size=2)
    children=tree.children.cpu().tolist();sizes=(tree.stops-tree.starts).cpu().tolist()
    leaves=[i for i,c in enumerate(children) if all(v<0 for v in c)]
    pairs=[(a,b) for a in leaves for b in leaves]
    canonical=[(a,b) for a,b in pairs if a<=b]
    expected=8*(2*len({a for a,b in canonical})+1+3*len(canonical))+sum(16*(sizes[a]+sizes[b]) for a,b in canonical)
    omitted=sum(32*sizes[a]*sizes[b] for a,b in canonical)
    plan=plan_directional_reciprocal_near_storage(tree,pairs,budget_bytes=2**20,
        near_storage='scalar_reciprocal_streamed_dk_v1',near_coefficient_workspace_budget_bytes=4096)
    assert plan['retained_bytes']==expected and plan['retained_near_kernel_bytes']==0 and omitted>0
    raw_write(tmp_path/'streamed-directional-capacity.json',dict(completed=True,plan=plan,
        independent_integer_map_bytes=expected,omitted_theoretical_original_coefficient_bytes=omitted,
        logical_pairs=len(pairs),canonical_pairs=len(canonical),manual_complete_leaf_pairs=True,
        sphere_partition_not_replayed=True,production_qualification=False))


@pytest.fixture(scope='module')
def prepared_case(tmp_path_factory):
    folder=tmp_path_factory.mktemp('new-directional-streamed-public')
    path=folder/'streamed-directional-public.json'
    raw=dict(version='new_directional_streamed_public_v1',completed=False,
        production_qualification=False,physical_qualification=False)
    raw_write(path,raw)
    try:
        scene=declare_scene();fmm=directional_config()
        config=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3,
            pair_batch=4,far_pair_batch=16),operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
                rtol=1e-12,atol=0.,maxiter=240,restart=120))
        prepared=mw.Simulation.mom(scene,frequency=1.73e8,config=config).prepare()
        actions=prepared.solver.handle.actions
        raw.update(configuration=fmm.payload(),physical_currents=actions.shape[0],
            FAR_counts=[len(r.far.rectangles) for r in (actions.exterior,*actions.interiors)],
            exterior_positive_changing_links=actions.exterior.test.plan_guard.selected_positive_links)
        raw_write(path,raw)
        return scene,fmm,prepared,actions,folder,path,raw
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None));raw_write(path,raw)
        raise


@pytest.mark.parametrize('builder',('dag','sparse'))
@pytest.mark.parametrize('component',('near','coefficient'))
def test_new_directional_streamed_component_refuses_before_maps(prepared_case,builder,component,monkeypatch):
    from witwin.maxwell.mom.fmm_order import AutomaticDirectionalDAGPlan
    from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
    _,_,_,actions,_,_,_=prepared_case
    plan=actions.exterior.test.plan_guard
    options=dict(integration=actions.provider.config,near_storage='scalar_reciprocal_streamed_dk_v1',
        near_coefficient_workspace_budget_bytes=4096)
    options['near_retained_budget_bytes' if component=='near' else 'near_coefficient_workspace_budget_bytes']=1
    def forbidden(*args,**kwargs):raise AssertionError('Streamed directional refusal reached map construction')
    if builder=='dag':
        monkeypatch.setattr(AutomaticDirectionalDAGPlan,'build_far',forbidden)
        build=plan.build_region
    else:
        monkeypatch.setattr(SparseDirectionalRWGProjection,'__init__',forbidden)
        options['quadrature_order']=12
        build=plan.region_plan.build_region
    with pytest.raises(MemoryError,match='before maps'):build(**options)


def test_new_public_directional_streamed_all_columns_gradients_true_weak_and_exact_restore(prepared_case,monkeypatch):
    from witwin.maxwell.mom.fmm_order import AutomaticDirectionalDAGPlan
    from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator,MultipoleOperator
    from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
    from witwin.maxwell.mom.runtime import SurfaceData,SystemOperator
    from witwin.maxwell.mom.packed_near import StreamedReciprocalScalarNearChannels
    from witwin.maxwell.mom.fmm_capacity import directional_retained_inventory,storage_bytes
    scene,fmm,prepared,actions,folder,path,raw=prepared_case
    try:
        regions=(actions.exterior,*actions.interiors)
        assert actions.shape==(36,36) and actions.exterior.far.rectangles
        for region in regions:
            assert isinstance(region.near,StreamedReciprocalScalarNearChannels)
            assert region.near.receipt()['retained_kernel_bytes']==0
            sizes=(region.test.tree.stops-region.test.tree.starts).cpu().tolist()
            canonical=[(a,b) for a,b in region.near_nodes if a<=b]
            expected=8*(2*len({a for a,b in canonical})+1+3*len(canonical))+sum(16*(sizes[a]+sizes[b]) for a,b in canonical)
            assert storage_bytes(region.tensors())==region.retained_bytes==region.planned_retained_bytes==expected
        ids=torch.arange(18,device='cuda')
        reference=actions.provider(ids,ids)
        reference_file=folder/'streamed-directional-native-reference.pt'
        saved=dict(reference=reference.detach(),frequency_Hz=1.73e8)
        torch.save(saved,reference_file)
        columns=actions.apply(torch.eye(36,device='cuda',dtype=torch.complex128)).mT
        generator=torch.Generator(device='cuda').manual_seed(85127)
        x=torch.randn((3,5,36),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
        z=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
        forward=actions.apply(x);hermitian=actions.adjoint(z)
        relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
        lhs=(forward.conj()*z).sum();rhs=(x.conj()*hermitian).sum()
        errors=dict(all_columns=relative(columns,reference),forward=relative(forward,x@reference.mT),
            original_H=relative(hermitian,z@reference.conj()),
            native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
        saved.update(columns=columns.detach(),x=x.detach(),z=z.detach(),forward=forward.detach(),H=hermitian.detach())
        torch.save(saved,reference_file)
        raw.update(errors=errors,reference_constructed_once=True,reference_artifact=str(reference_file))
        raw_write(path,raw)
        assert max(errors[k] for k in ('all_columns','forward','original_H'))<1e-8 and errors['native_H']<1e-11
        captured=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:captured.append(t) or t,lambda t:t):value=actions.apply(x)
        assert not captured
        gradient=torch.autograd.grad(value,x,z)[0]
        first=torch.autograd.grad(actions.apply(x).abs().square().sum(),x,create_graph=True)[0]
        tangent=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
        higher=torch.autograd.grad(first,x,tangent)[0]
        current_error=relative(gradient,hermitian);higher_error=relative(higher,2*(tangent@columns.mT)@columns.conj())
        raw.update(current_gradient_error=current_error,higher_current_gradient_error=higher_error,
            saved_forward_tensors=0,retained_inventory=directional_retained_inventory(actions),
            regional_receipts=[r.receipt() for r in regions])
        raw_write(path,raw);assert current_error<1e-11 and higher_error<1e-11
        result=prepared.run();stats=result.stats();assert stats['converged'] and stats['error_budget']['checked']
        current=result.surface_currents.scattered_coefficients
        residual=relative(current@reference.mT,prepared.solver.rhs)
        raw.update(complete_original_residual=residual,residual_relative_gate=1e-10,residual_absolute_gate=0.)
        raw_write(path,raw);assert residual<=1e-10
        artifact=folder/'streamed-directional-result.pt';result.save(artifact)
        def forbidden(*args,**kwargs):raise AssertionError('Directional streamed restore repeated maps or solving')
        with monkeypatch.context() as patch:
            for cls in (AutomaticDirectionalDAGPlan,DirectionalSystemActions,DirectionalMultipoleOperator,MultipoleOperator,SystemOperator):
                patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(artifact,scene=scene)
            assert restored.surface_currents.config.fmm==fmm
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            for key,value in [('near_coefficient_workspace_budget_bytes',2048),('near_storage','scalar_reciprocal_dk_v1')]:
                bad=copy.deepcopy(result.surface_currents.snapshot());bad['fmm_config']['parameters'][key]=value
                with pytest.raises(ValueError):SurfaceData.restore(bad,scene,1.73e8)
        raw.update(completed=True,artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            exact_restore=True,whole_peak_qualified=False,original_weak_reference_is_not_independent_EM=True,
            high_frequency_or_positive_changing_paths_not_qualified=True)
        raw_write(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None));raw_write(path,raw)
        raise
