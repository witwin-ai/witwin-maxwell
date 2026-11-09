"""New public center identity, complete original action, and exact archive checks."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors, storage_bytes
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_operator import MultipoleOperator
from witwin.maxwell.mom.runtime import SurfaceData, SystemOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def config(**options):
    return mw.FMMConfig(order=6, table_storage='compact_shared_displacement_v1',
        shift_storage='separable_v1', near_storage='scalar_reciprocal_dk_v1',
        far_action_batch=32, leaf_size=8, separation_ratio=.25,
        expansion_center_policy='dyadic_cell_v1',
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        kernel_bound_rule='direct_first_wave_min_v1', **options)


def test_new_center_version_exact_fields_and_affected_default_payloads():
    # Configuration identity changed; this is the necessary default-schema
    # control. No previously passed numerical construction is repeated.
    old=json.loads((Path(__file__).parent/'fixtures/bounded_compact_original_payloads.json').read_text())
    for case in old['cases']:
        current=mw.FMMConfig(**case['options'])
        assert current.payload()==case['payload']
        assert mw.FMMConfig.restore(case['payload'])==current
        assert 'expansion_center_policy' not in current.construction()
    for batch in (1,32):
        values=config().payload()['parameters'];values['far_action_batch']=batch
        candidate=mw.FMMConfig(**{**config().construction(),'far_action_batch':batch})
        payload=candidate.payload()
        assert payload['version']=='cartesian_fmm_dyadic_centers_config_v1'
        assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==candidate
        assert candidate.construction()['expansion_center_policy']=='dyadic_cell_v1'
        for kind in ('missing','extra','policy','version','budget','storage'):
            bad=copy.deepcopy(payload)
            if kind=='missing':bad['parameters'].pop('expansion_center_policy')
            elif kind=='extra':bad['parameters']['approximation']=True
            elif kind=='policy':bad['parameters']['expansion_center_policy']='support_box_midpoint_v1'
            elif kind=='version':bad['version']='cartesian_fmm_batched_compact_config_v1'
            elif kind=='budget':bad['parameters']['kernel_error_budget']['extra']=1
            else:bad['parameters']['table_storage']='dense'
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)


@pytest.mark.parametrize('kind',['policy','dense','high_order','periodic','precision'])
def test_public_unsupported_geometry_rejects_before_prepare(kind):
    values=config().payload()['parameters'];values['kernel_error_budget']=mw.KernelErrorBudget(1e-6,1e-5)
    if kind=='policy':values['expansion_center_policy']='approximate_bins'
    elif kind=='dense':values['table_storage']='dense'
    elif kind=='high_order':values['max_order']=13
    with pytest.raises((ValueError,NotImplementedError)):
        candidate=mw.FMMConfig(**values)
        mw.MoMConfig(operator_backend='fmm',fmm=candidate,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(preconditioner='none'),
            dtype=torch.complex64 if kind=='precision' else torch.complex128,
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)) if kind=='periodic' else None)


def test_new_public_geometry_all_columns_native_gradients_solve_and_exact_restore(tmp_path,monkeypatch):
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01,position=(x,0.,0.)),
        mw.Material(eps_r=2.+i,mu_r=1.2+.1*i)) for i,x in enumerate((-.2,.2))],sources=[mw.PlaneWave()])
    cfg=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=config(max_order=12),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    prepared=mw.Simulation.mom(scene,frequency=9e7,config=cfg).prepare();op=prepared.solver.handle.actions
    assert op.exterior.far and op.order_plan['selected_order']>=6
    assert op.expansion_center_policy=='dyadic_cell_v1'
    assert op.receipt()['expansion_center_policy']=='dyadic_cell_v1'
    for region in (op.exterior,*op.interiors):
        assert region.test.tree.expansion_center_policy=='dyadic_cell_v1'
        torch.testing.assert_close(region.test.centers,region.test.tree.expansion_centers,rtol=0,atol=0)
        assert region.test.centers.data_ptr()!=region.test.tree.expansion_centers.data_ptr()
    # One complete original reference for this new center geometry; reuse it
    # for columns, all RHS and the actual solved original-vector residual.
    edges=torch.arange(op.shape[0]//2,device='cuda');reference=op.provider(edges,edges)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    original_error=relative(op.apply(torch.eye(op.shape[0],device='cuda',dtype=op.dtype)).mT,reference)
    generator=torch.Generator(device='cuda').manual_seed(818)
    x=torch.randn((3,5,op.shape[0]),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
    y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
    forward=op.apply(x);hermitian=op.adjoint(y)
    assert original_error<1e-8 and relative(forward,x@reference.mT)<1e-8
    assert relative(hermitian,y@reference.conj())<1e-8
    lhs=(forward.conj()*y).sum();rhs=(x.conj()*hermitian).sum()
    native_error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()))
    assert native_error<1e-11
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):value=op.apply(x)
    assert not saved
    gradient=torch.autograd.grad(value,x,y)[0];assert relative(gradient,hermitian)<1e-11
    first=torch.autograd.grad(op.apply(x).abs().square().sum(),x,create_graph=True)[0]
    direction=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
    second=torch.autograd.grad(first,x,direction)[0]
    assert relative(second,2*op.adjoint(op.apply(direction)))<1e-11
    owned=storage_bytes(multipole_action_tensors(op));assert owned==op.capacity_plan['planned_bytes']
    result=prepared.run();assert result.stats()['converged'] and result.stats()['error_budget']['checked']
    current=result.surface_currents.scattered_coefficients
    residual=relative(current@reference.mT,prepared.solver.rhs);assert residual<=1e-10
    artifact=tmp_path/'dyadic-centers-result.pt';result.save(artifact)
    def forbidden(*args,**kwargs):raise AssertionError('restore reconstructed or solved the FMM operator')
    with monkeypatch.context() as patch:
        for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(artifact,scene=scene)
        assert restored.surface_currents.config.fmm==cfg.fmm
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        for key,value in [('expansion_center_policy','support_box_midpoint_v1'),('leaf_size',30),
                ('far_action_batch',1),('table_storage','compact')]:
            bad=copy.deepcopy(result.surface_currents.snapshot());bad['fmm_config']['parameters'][key]=value
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,9e7)
    record=dict(version='public_dyadic_centers_geometry_check_v1',original_columns=op.shape[0],
        original_relative_error=original_error,native_H_relative_error=native_error,
        current_gradient_relative_error=relative(gradient,hermitian),complete_original_residual=residual,
        residual_relative_gate=1e-10,residual_absolute_gate=0.,saved_forward_tensors=0,
        actual_owned_bytes=owned,capacity_plan=op.capacity_plan,order_plan=op.order_plan,
        artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        fmm_config=cfg.fmm.payload(),physical_qualification=False,production_qualification=False,
        whole_peak_qualified=False,original_reference_constructed_once=True)
    (tmp_path/'public.json').write_text(json.dumps(record,indent=2)+'\n')
    # A same-byte native mutation must invalidate the actual public action.
    op.exterior.test.tree.expansion_centers.add_(0)
    with pytest.raises(ValueError,match='changed'):op.apply(x)
