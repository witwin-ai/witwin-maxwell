"""Explicit reciprocal FMM policy, physical capacity/actions and exact public state."""
import copy
import hashlib
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.operator import IntegrationConfig,assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_capacity import plan_multipole_storage,multipole_action_tensors,storage_bytes,MultipoleCapacityError
from witwin.maxwell.mom.fmm_operator import MultipoleOperator
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.runtime import SurfaceData,SystemOperator
from test_directional_complete_system import _basis

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _relative(actual,reference):return float(((actual-reference).norm()/reference.norm()).detach())


def _provider():
    exterior = compile_medium(mw.Material(),3e8)
    bodies = [(i,str(i),_basis(x),compile_medium(mw.Material(eps_r=2.+i,mu_r=1.4+.2*i,
        sigma_e=.01,sigma_m=.02),3e8)) for i,x in enumerate((-.3,.3))]
    return ScalarSystemBlocks(SurfaceSystem.create(bodies),exterior,
        IntegrationConfig(order=3,singular_order=3,pair_batch=4,far_pair_batch=8))


def _construction():
    return dict(order=8,quadrature_order=12,leaf_size=2,separation_ratio=.25,
        table_storage='fft_v1',shift_storage='separable_v1',near_storage='scalar_reciprocal_dk_v1',
        kernel_bound_rule='direct_first_wave_min_v1',kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        near_action_batch=2)


def test_reciprocal_policy_exact_version_all_fields_and_early_public_rejection(tmp_path):
    for table in ('dense','compact','fft_v1'):
        for shift in ('dense','separable_v1'):
            config = mw.FMMConfig(table_storage=table,shift_storage=shift,
                near_storage='scalar_reciprocal_dk_v1',kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5))
            payload = json.loads(json.dumps(config.payload()))
            assert payload['version']=='cartesian_fmm_reciprocal_dk_config_v1'
            assert mw.FMMConfig.restore(payload)==config
            assert config.construction()['near_storage']=='scalar_reciprocal_dk_v1'
            for field in payload['parameters']:
                bad = copy.deepcopy(payload);del bad['parameters'][field]
                with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
            for policy in ('full_matrix','scalar_dk_v1'):
                bad = copy.deepcopy(payload);bad['parameters']['near_storage']=policy
                with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
            bad = copy.deepcopy(payload);bad['version']='cartesian_fmm_scalar_dk_config_v1'
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.FMMConfig(near_storage='scalar_reciprocal_dk_v1',table_storage='fft_ewald_v1')
    options = dict(operator_backend='fmm',riesz_storage='local_rwg_v1',stopping=StoppingRule(preconditioner='rwg_diagonal'))
    with pytest.raises(ValueError,match='complex128'):
        mw.MoMConfig(fmm=config,dtype=torch.complex64,**options)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(fmm=config,periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)),**options)
    (tmp_path/'configuration.json').write_text(json.dumps(dict(explicit_version=payload['version'],
        all_fields_required=True,old_policy_version_mismatch_rejected=True,periodic_and_complex64_rejected=True,
        qualification=False),indent=2)+'\n')


@pytest.fixture(scope='module')
def physical():
    provider = _provider()
    original,_ = assemble_system(provider.system,provider.exterior,provider.config)
    op = MultipoleSystemActions(provider,**_construction())
    return provider,op,original


@pytest.fixture(scope='module')
def compressed_physical():
    from test_fmm_scalar_near_channels import _provider as sphere_provider
    provider = sphere_provider()
    original,_ = assemble_system(provider.system,provider.exterior,provider.config)
    op = MultipoleSystemActions(provider,**_construction())
    return provider,op,original


def test_actual_full_cartesian_reciprocal_operator_original_columns_H_current_capacity(compressed_physical,monkeypatch,tmp_path):
    provider,op,original = compressed_physical
    regions = (op.exterior,*op.interiors)
    receipts = [r.near.receipt() for r in regions]
    assert all(p['version']=='grouped_reciprocal_scalar_original_dk_csr_v1' for p in receipts)
    assert any(p['stored_blocks']<p['logical_blocks'] for p in receipts)
    assert op.exterior.far
    actual = op.apply(torch.eye(op.shape[1],device='cuda',dtype=op.dtype)).mT
    n = provider.system.total_edges;eta = abs(provider.exterior.eta)
    left = actual.real.new_tensor([eta]*n+[1.]*n);right = actual.real.new_tensor([1.]*n+[eta]*n)
    weak = _relative(left[:,None]*actual*right,left[:,None]*original*right)
    assert weak<1e-8
    generator = torch.Generator(device='cuda').manual_seed(630213)
    current = torch.randn((3,5,op.shape[1]),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=op.dtype,generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):forward = op.apply(current)
    assert not saved and not current.is_contiguous()
    H = op.adjoint(dual)
    gradient, = torch.autograd.grad(forward,current,dual,retain_graph=True)
    H_gradient, = torch.autograd.grad(H,dual,current,retain_graph=True)
    first, = torch.autograd.grad(forward.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn_like(current);second, = torch.autograd.grad(first,current,tangent)
    lhs,rhs = (forward.conj()*dual).sum(),(current.conj()*H).sum()
    errors = dict(forward=_relative(forward,current@actual.mT),H=_relative(H,dual@actual.conj()),
        current=_relative(gradient,H),adjoint_current=_relative(H_gradient,forward),
        higher=_relative(second,2*op.adjoint(op.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    original_H = _relative(H,dual@original.conj())
    assert original_H<1e-8 and max(errors.values())<1e-11
    owned = storage_bytes(multipole_action_tensors(op))
    assert owned==op.capacity_plan['planned_bytes']
    trees = tuple(r.test.tree for r in regions)
    ordered = plan_multipole_storage(trees,order=8,separation_ratio=.25,budget_bytes=1024**3,
        table_storage='fft_v1',shift_storage='separable_v1',near_storage='scalar_dk_v1')
    assert ordered['categories']['near']>op.capacity_plan['categories']['near']
    assert ordered['planned_bytes']-owned==ordered['categories']['near']-op.capacity_plan['categories']['near']
    for key,value in ordered['categories'].items():
        if key!='near':assert op.capacity_plan['categories'][key]==value
    assert op.near_bytes==sum(p['retained_bytes'] for p in receipts)
    def forbidden(*args,**kwargs):raise AssertionError('Rejected compressed-case capacity reached moment setup')
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_system.NestedRWGMultipoles',forbidden)
        with pytest.raises(MultipoleCapacityError):
            MultipoleSystemActions(provider,**_construction(),action_retained_budget_bytes=owned-1)
    output = op.apply(current)
    field = regions[0].near.groups[0][2];field.requires_grad_(True)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,dual)
    field.requires_grad_(False);op.assert_current()
    (tmp_path/'system.json').write_text(json.dumps(dict(actual_bodies=2,physical_edges=n,
        full_original_columns_balanced_relative=weak,original_H_relative=original_H,native_errors=errors,
        saved_internal_tensors=len(saved),near_receipts=receipts,owned_action_retained_bytes=owned,
        capacity=op.capacity_plan,ordered_DK_capacity=ordered,saved_whole_system_near_mode_guard=True,
        actual_compressed_case_one_byte_below_capacity_rejected_before_maps=True,
        original_near_quadrature_shared_not_convergence=True,whole_peak_or_crossover_qualified=False,
        qualification=False),indent=2)+'\n')


def test_reciprocal_whole_action_capacity_before_maps(physical,monkeypatch,tmp_path):
    provider,op,_ = physical
    def forbidden(*args,**kwargs):raise AssertionError('Rejected reciprocal capacity reached moment setup')
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_system.NestedRWGMultipoles',forbidden)
        with pytest.raises(MultipoleCapacityError):
            MultipoleSystemActions(provider,**_construction(),action_retained_budget_bytes=op.capacity_plan['planned_bytes']-1)
    (tmp_path/'capacity.json').write_text(json.dumps(dict(exact_one_byte_below_action_capacity_rejected=True,
        maps_and_original_integration_not_reached=True,qualification=False),indent=2)+'\n')


def test_public_reciprocal_FMM_solve_and_constructor_free_exact_result_restore(tmp_path,monkeypatch):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01,position=(x,0.,0.)),
        mw.Material(eps_r=2.+i,mu_r=1.2+.1*i)) for i,x in enumerate((-.2,.2))],sources=[mw.PlaneWave()])
    fmm = mw.FMMConfig(order=6,max_order=10,kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        kernel_bound_rule='direct_first_wave_min_v1',table_storage='fft_v1',shift_storage='separable_v1',
        near_storage='scalar_reciprocal_dk_v1',leaf_size=8,separation_ratio=.25,
        leaf_factor_kernel='primary',leaf_factor_max_edges=16)
    config = mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    result = mw.Simulation.mom(scene,frequency=9e7,config=config).run()
    stats = result.stats();assert stats['converged'] and stats['error_budget']['checked']
    actions = result.surface_currents.handle.actions
    near = [r.near.receipt() for r in (actions.exterior,*actions.interiors)]
    assert actions.near_storage=='scalar_reciprocal_dk_v1'
    assert all(p['version']=='grouped_reciprocal_scalar_original_dk_csr_v1' for p in near)
    assert any(p['stored_blocks']<p['logical_blocks'] for p in near)
    assert actions.exterior.far and actions.capacity_plan['near_storage']==fmm.near_storage
    current = result.surface_currents.scattered_coefficients
    path = tmp_path/'reciprocal-fmm-result.pt';result.save(path)
    def forbidden(*args,**kwargs):raise AssertionError('Exact reciprocal result restore constructed or solved an operator')
    with monkeypatch.context() as patch:
        for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored = mw.Result.load(path,scene=scene)
        assert restored.surface_currents.config.fmm==fmm
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        assert restored.surface_currents.snapshot()['fmm_config']==fmm.payload()
        for field,value in (('near_storage','scalar_dk_v1'),('table_storage','compact'),
                ('kernel_bound_rule','bivariate_cauchy_v1'),('shift_storage','dense')):
            bad = copy.deepcopy(result.surface_currents.snapshot());bad['fmm_config']['parameters'][field]=value
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,9e7)
    (tmp_path/'public.json').write_text(json.dumps(dict(artifact=str(path),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fmm_config=fmm.payload(),stats=stats,
        selected_order=actions.order_plan['selected_order'],near_receipts=near,
        exact_current_restore=True,original_solver_residual_is_sample_not_global_certificate=True,
        physics_and_resource_qualification=False),indent=2,default=str)+'\n')
