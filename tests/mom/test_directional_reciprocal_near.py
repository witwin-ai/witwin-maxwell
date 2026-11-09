"""Changed reciprocal near storage with shared original directional FAR owners."""
import copy
import hashlib
import json
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_capacity import plan_directional_reciprocal_near_storage, directional_retained_inventory
from witwin.maxwell.mom.fmm_regional import DirectionalRegionalOperator
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalDAGPlan
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator, MultipoleOperator
from witwin.maxwell.mom.runtime import SurfaceData, SystemOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_directional_reciprocal_configuration_all_fields_and_old_default_identity(tmp_path):
    config=mw.DirectionalFMMConfig(near_storage='scalar_reciprocal_dk_v1')
    payload=json.loads(json.dumps(config.payload()))
    assert payload['version']=='automatic_directional_reciprocal_dag_fmm_config_v1'
    assert mw.FMMConfig.restore(payload)==mw.DirectionalFMMConfig.restore(payload)==config
    for field in payload['parameters']:
        bad=copy.deepcopy(payload);del bad['parameters'][field]
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for key,value in (('version','automatic_directional_dag_fmm_config_v1'),('version','cartesian_fmm_reciprocal_dk_config_v1')):
        bad=copy.deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    bad=copy.deepcopy(payload);bad['parameters']['near_storage']='scalar_dk_v1'
    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    default=replace(config,near_storage='scalar_dk_v1').payload()
    assert default['version']=='automatic_directional_dag_fmm_config_v1' and 'near_storage' not in default['parameters']
    assert mw.FMMConfig.restore(default).near_storage=='scalar_dk_v1'
    for mode in ('full_matrix','unknown',None):
        with pytest.raises(ValueError):replace(config,near_storage=mode)
    options=dict(operator_backend='fmm',fmm=config,riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='rwg_diagonal'))
    assert mw.MoMConfig(**options).fmm is config
    with pytest.raises(NotImplementedError):mw.MoMConfig(**options,periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    with pytest.raises(ValueError):mw.MoMConfig(**options,dtype=torch.complex64)
    (tmp_path/'configuration.json').write_text(json.dumps(dict(payload=payload,all_fields_bound=True,old_payload_omits_default_near=True,qualification=False),indent=2)+'\n')


@pytest.fixture(scope='module')
def physical():
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.02,position=(i,1e-6*(i%2),0.)),
        mw.Material(eps_r=2.+i,mu_r=1.3+.1*i,sigma_e=.01,sigma_m=.02)) for i in range(3)],sources=[mw.PlaneWave()])
    fmm=mw.DirectionalFMMConfig(leaf_size=4,separation_ratio=.15,cone_subdivisions=(2**20,),
        max_contexts=4096,tensor_workspace_budget_bytes=512*1024**2,
        leaf_factor_kernel='original',leaf_factor_max_edges=16,near_storage='scalar_reciprocal_dk_v1')
    config=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=6,singular_order=3,pair_batch=4,
        far_pair_batch=16,near_max_order=32),operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',rtol=1e-12,atol=0.,maxiter=240,restart=120))
    prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    actions=prepared.solver.handle.actions
    assert actions.exterior.test.plan_guard.selected_positive_links>0
    regions=(actions.exterior,*actions.interiors)
    assert all(r.near_storage=='scalar_reciprocal_dk_v1' for r in regions)
    assert all(r.near.receipt()['stored_blocks']<r.near.receipt()['logical_blocks'] for r in regions)
    return scene,fmm,prepared,actions


def test_directional_reciprocal_actual_reverse_and_capacity_preflight_before_maps(physical,monkeypatch,tmp_path):
    _,_,_,actions=physical
    region=actions.exterior;plan=region.test.plan_guard;root=plan.region_plan
    receipt=plan_directional_reciprocal_near_storage(root.tree,root.near_nodes,budget_bytes=1024**3)
    assert receipt['retained_bytes']==region.retained_bytes and receipt['stored_blocks']<receipt['logical_blocks']
    pairs=list(root.near_nodes);pair=next((a,b) for a,b in pairs if a!=b)
    bad=pairs.copy();bad.remove((pair[1],pair[0]))
    for nodes in (bad,pairs+[pair]):
        with pytest.raises(ValueError,match='reverse pairs'):
            plan_directional_reciprocal_near_storage(root.tree,nodes,budget_bytes=1024**3)
    def forbidden(*args,**kwargs):raise AssertionError('Rejected reciprocal capacity reached directional maps')
    with monkeypatch.context() as patch:
        patch.setattr(AutomaticDirectionalDAGPlan,'build_far',forbidden)
        with pytest.raises(MemoryError,match='before maps'):
            plan.build_region(integration=region.integration,near_storage='scalar_reciprocal_dk_v1',
                near_retained_budget_bytes=receipt['retained_bytes']-1)
    (tmp_path/'capacity.json').write_text(json.dumps(dict(actual=receipt,missing_reverse_and_duplicate_reject=True,
        one_byte_below_rejected_before_maps=True,qualification=False),indent=2)+'\n')


def test_directional_reciprocal_complete_columns_H_current_higher_and_inventory(physical,tmp_path):
    _,_,_,actions=physical
    regions=(actions.exterior,*actions.interiors)
    # The existing FAR maps/couplings are the identical owners in both actions.
    # Only original ordered D/K near blocks are independently integrated here;
    # previously covered FAR quadrature and original full-matrix assembly are
    # not repeated or relabelled as new physical convergence evidence.
    ordered_regions=[DirectionalRegionalOperator(r.far,r.near_nodes,integration=r.integration,
        tensor_workspace_budget_bytes=r.tensor_workspace_budget_bytes) for r in regions]
    ordered=DirectionalSystemActions(actions.provider,ordered_regions[0],ordered_regions[1:],
        ideal_rwg_error_budget=actions.ideal_rwg_error_budget,balancing_impedance=actions.balancing_impedance,
        tensor_workspace_budget_bytes=actions.tensor_workspace_budget_bytes)
    assert all(a.far is b.far for a,b in zip(regions,ordered_regions,strict=True))
    n=actions.shape[0]//2;eye=torch.eye(2*n,device='cuda',dtype=torch.complex128)
    original=ordered.apply(eye).mT;actual=actions.apply(eye).mT
    eta=actions.balancing_impedance;l=actual.real.new_tensor([eta]*n+[1.]*n);r=actual.real.new_tensor([1.]*n+[eta]*n)
    weak=float((l[:,None]*(actual-original)*r).norm()/(l[:,None]*original*r).norm())
    assert weak<1e-11
    generator=torch.Generator(device='cuda').manual_seed(74011)
    x=torch.randn((3,5,2*n),device='cuda',dtype=actual.dtype,generator=generator).transpose(0,1).requires_grad_()
    z=torch.randn(x.shape,device='cuda',dtype=actual.dtype,generator=generator).requires_grad_()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):y=actions.apply(x)
    H=actions.adjoint(z);current,=torch.autograd.grad(y,x,z,retain_graph=True)
    yH=actions.adjoint(x);Hcurrent,=torch.autograd.grad(yH,x,z,retain_graph=True)
    first,=torch.autograd.grad(y.abs().square().sum(),x,create_graph=True)
    tangent=torch.randn_like(x);higher,=torch.autograd.grad(first,x,tangent)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    lhs=(y.conj()*z).sum();rhs=(x.conj()*H).sum()
    errors=dict(forward=relative(y,x@actual.mT),H=relative(H,z@actual.conj()),
        independently_integrated_ordered_near_H=relative(H,z@original.conj()),current=relative(current,H),
        adjoint_current=relative(Hcurrent,actions.apply(z)),higher=relative(higher,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    assert max(errors.values())<1e-11 and not saved
    inventories=[directional_retained_inventory(a) for a in (actions,ordered)]
    saved_bytes=sum(b.retained_bytes-a.retained_bytes for a,b in zip(regions,ordered_regions,strict=True))
    assert saved_bytes>0 and inventories[1]['retained_native_bytes']-inventories[0]['retained_native_bytes']==saved_bytes
    guarded=actions.apply(x);old=regions[0].near_storage;regions[0].near_storage='scalar_dk_v1'
    try:
        with pytest.raises(ValueError):actions.receipt()
        with pytest.raises(ValueError):torch.autograd.grad(guarded,x,z)
    finally:regions[0].near_storage=old
    actions.assert_current()
    (tmp_path/'system.json').write_text(json.dumps(dict(physical_edges=n,full_columns=2*n,
        independently_integrated_ordered_near_balanced_relative=weak,native_errors=errors,
        saved_internal_tensors=len(saved),same_FAR_owners=True,near_saved_bytes=saved_bytes,
        inventories=inventories,near_receipts=[r.near.receipt() for r in regions],saved_policy_guard=True,
        original_FAR_and_full_matrix_not_reintegrated=True,original_full_matrix_or_quadrature_not_certified=True,
        positive_changing_exterior_links=regions[0].test.plan_guard.selected_positive_links,
        interior_FAR_counts=[len(r.far.rectangles) for r in regions[1:]],qualification=False),indent=2)+'\n')


def test_public_directional_reciprocal_solve_exact_restore_and_policy_binding(physical,tmp_path,monkeypatch):
    scene,fmm,prepared,actions=physical
    result=prepared.run();stats=result.stats()
    assert stats['converged'] and stats['error_budget']['checked'] and not stats['fallback_used']
    assert not stats['original_global_residual_certified']
    path=tmp_path/'directional-reciprocal-result.pt';result.save(path)
    current=result.surface_currents.scattered_coefficients
    def forbidden(*args,**kwargs):raise AssertionError('Reciprocal directional restore repeated construction or solving')
    with monkeypatch.context() as patch:
        for cls in (DirectionalSystemActions,DirectionalMultipoleOperator,MultipoleOperator,AutomaticDirectionalDAGPlan,SystemOperator):
            patch.setattr(cls,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        assert restored.surface_currents.config.fmm==fmm
        for key,value in (('near_storage','scalar_dk_v1'),('max_candidates',256),('phase_relative_budget',1e-11),('cone_subdivisions',[2])):
            bad=copy.deepcopy(result.surface_currents.snapshot());bad['fmm_config']['parameters'][key]=value
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,3e8)
    (tmp_path/'public.json').write_text(json.dumps(dict(artifact=str(path),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        configuration=fmm.payload(),stats=stats,near_receipts=[r.near.receipt() for r in (actions.exterior,*actions.interiors)],
        exact_current_restore=True,constructor_and_solve_refused=True,original_residual_sample_not_global=True,
        qualification=False),indent=2,default=str)+'\n')
