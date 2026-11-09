"""Complete prescribed directional solves through the shared native protocol."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig,assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalNestedFarActions
from witwin.maxwell.mom.fmm_regional import DirectionalRegionalOperator
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import StoppingRule,residual_loss
from witwin.maxwell.mom.runtime import SystemDecodeHandle,MoMConfig
from witwin.maxwell.mom.lifting import build_lifted_rhs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(scope='module')
def case():
    frequency = 4e8
    exterior = compile_medium(mw.Material(),frequency)
    entries = []
    for i,(x,eps,mu,sigma) in enumerate(((-.4,2.2,1.15,.002),(.4,3.1,1.45,.003))):
        vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],device='cuda',dtype=torch.float64)*.015
        vertices[:,0]+=x
        faces = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
        basis = RWGBasis(SurfaceMesh(vertices,faces))
        interior = compile_medium(mw.Material(eps_r=eps,mu_r=mu,sigma_e=sigma,sigma_m=.01+.005*i),frequency)
        entries.append((i,str(i),basis,interior))
    system = SurfaceSystem.create(entries)
    config = IntegrationConfig(order=3,singular_order=3,pair_batch=4,far_pair_batch=8)
    provider = ScalarSystemBlocks(system,exterior,config)
    lower,upper = provider.support_boxes();tree = build_support_octree(lower,upper,leaf_size=6)
    minus = lower.new_tensor([-1.,0.,0.]);eta = abs(exterior.eta)
    def owner(b,t,d,m,**kw):
        return NestedDirectionalRWGProjection(b,t,d,m.k,8,quadrature_order=12,face_batch=4,rhs_batch=4,**kw)
    left = owner(provider,tree,minus,exterior);right = owner(provider,tree,-minus,exterior,layout=left.layout)
    a,b = left.children[0]
    coupling = tuple(DirectionalFFTElectromagneticM2L.from_nested_nodes(p,t,p,s,exterior,
        lower.new_tensor(.01),kernel_error_budget=KernelErrorBudget(1e-6,1e-4),
        ideal_rwg_error_budget=1e-4,balancing_impedance=eta) for p,t,s in ((left,a,b),(right,b,a)))
    far = DirectionalNestedFarActions(left,left,exterior,coupling,ideal_rwg_error_budget=1e-4)
    regional = DirectionalRegionalOperator(far,((a,a),(b,b)),integration=config)
    interiors = []
    for i,body in enumerate(system.bodies):
        begin,end = body.edge_offset,body.edge_offset+body.basis.n_edges
        local_tree = build_support_octree(lower[begin:end],upper[begin:end],leaf_size=6)
        local = owner(body.basis,local_tree,minus,body.interior,geometry=provider.geometry[i])
        empty = DirectionalNestedFarActions(local,local,body.interior,(),
            ideal_rwg_error_budget=1e-4,balancing_impedance=eta)
        interiors.append(DirectionalRegionalOperator(empty,((0,0),),integration=config))
    actions = DirectionalSystemActions(provider,regional,interiors,ideal_rwg_error_budget=1e-4)
    dense,_ = assemble_system(system,exterior,config)
    rhs_values = []
    for polarization in ('Ex','Ey'):
        decoder = SystemDecodeHandle(system,exterior,[mw.PlaneWave(polarization=polarization)],
            MoMConfig(integration=config),{})
        rhs = torch.empty(2*system.total_edges,device='cuda',dtype=torch.complex128)
        for i,handle in enumerate(decoder.body_handles):
            local,_ = build_lifted_rhs(handle);count = handle.basis.n_edges
            rhs[system.J_slices[i]],rhs[system.M_slices[i]] = local[:count],local[count:]
        rhs_values.append(rhs)
    rhs = torch.stack(rhs_values)
    reference = torch.linalg.solve(dense,rhs.mT).mT
    return provider,actions,dense,rhs,reference


def _forbid(*args,**kwargs):raise AssertionError('Directional adapter invoked an unrelated construction or dense fallback')


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def test_directional_shared_multi_rhs_native_Riesz_Krylov_original_residual(case,monkeypatch,tmp_path):
    provider,actions,dense,rhs,reference = case
    with monkeypatch.context() as patch:
        for path in ('witwin.maxwell.mom.hmatrix.HierarchicalOperator.__init__',
                'witwin.maxwell.mom.direct_system.DirectSystemActions.__init__',
                'witwin.maxwell.mom.fmm_system.MultipoleSystemActions.__init__',
                'witwin.maxwell.mom.linalg.DenseOperator.__init__',
                'witwin.maxwell.mom.linalg._one_shot_direct'):
            patch.setattr(path,_forbid)
        op = DirectionalMultipoleOperator(provider,actions)
        assert isinstance(op,SurfaceOperatorProtocol) and op.actions is actions
        assert not op.supports_direct and not hasattr(op,'matrix')
        left,right,_ = op.preconditioner_scales('rwg_diagonal');n = provider.system.total_edges
        assert _relative(left[:n]/left[n:],torch.full_like(left[:n],abs(provider.exterior.eta)))<1e-11
        assert _relative(right[:n],left[n:])<1e-11
        generator = torch.Generator(device='cuda').manual_seed(2573)
        x = torch.randn(rhs.shape,device='cuda',dtype=torch.complex128,generator=generator,requires_grad=True)
        gradient, = torch.autograd.grad(residual_loss(op,x,rhs),x)
        expected = op.adjoint(op.riesz_inverse(op.apply(x)-rhs))
        current_error = _relative(gradient,expected);assert current_error<1e-11
        def dual(value):return (value.conj()*op.riesz_inverse(value)).sum(-1).real.clamp_min(0).sqrt()
        scale = torch.cat((torch.ones(n,device='cuda'),torch.full((n,),abs(provider.exterior.eta),device='cuda')))
        original_edges = torch.arange(n,device='cuda')
        diagnostics = []
        for method in ('gmres','block_gmres'):
            solved,log = op.solve(rhs,StoppingRule(method=method,preconditioner='rwg_diagonal',
                rtol=1e-12,atol=0.,maxiter=96,restart=48),original_edges=original_edges,source_batch=4)
            original_dual = float((dual(solved@dense.mT-rhs)/dual(rhs)).max())
            error = float((torch.linalg.vector_norm((solved-reference)/scale,dim=-1)/
                torch.linalg.vector_norm(reference/scale,dim=-1)).max())
            record = dict(method=method,original_full_dual_relative=original_dual,
                independent_dense_current_relative=error,log=log)
            diagnostics.append(record);print(record,flush=True)
            (tmp_path/'solve.json').write_text(json.dumps(dict(methods=diagnostics,
                current_Riesz_gradient_relative=current_error,physical_rhs=2,
                reference_only_dense=True,positive_interior_far_covered=False,
                production_qualification=False),indent=2)+'\n')
            assert log['converged'] and log['inner_converged'] and log['error_budget']['checked'],log
            assert log['error_budget']['compressed_dual_pass'] and log['error_budget']['original_sampled_pass']
            assert not log['fallback_used'] and not log['original_global_residual_certified']
            assert original_dual<1e-8 and error<1e-7
        assert op.operator_stats()['apply_calls']>0 and op.operator_stats()['adjoint_calls']>0
        assert not op.operator_stats()['production_qualification']


def test_directional_failed_iteration_refused_direct_and_residual_configuration(case,tmp_path):
    provider,actions,_,rhs,_ = case;op = DirectionalMultipoleOperator(provider,actions)
    _,failed = op.solve(rhs,StoppingRule(method='gmres',preconditioner='rwg_diagonal',
        rtol=1e-12,atol=0.,maxiter=1,restart=1))
    assert not failed['converged'] and not failed['inner_converged']
    assert not failed['error_budget']['checked'] and not failed['fallback_used']
    for rule in (StoppingRule(method='direct'),StoppingRule(fallback='direct')):
        with pytest.raises(ValueError,match='requires a dense operator'):op.solve(rhs,rule)
    report = op.residual_report(torch.zeros_like(rhs),rhs,
        original_edges=torch.arange(provider.system.total_edges,device='cuda'),source_batch=4)
    assert max(abs(x-1.) for x in report['compressed_dual']['relative'])<1e-11
    assert max(abs(x-1.) for x in report['original_sampled']['relative'])<1e-11
    receipt = op.persistence_receipt()
    assert not receipt['executable_snapshot'] and not receipt['compressed_tensor_snapshot_included']
    assert receipt['construction']['complete_coefficient_pair_coverage']
    assert receipt['solve_configuration']==dict(solve_dual_tolerance=1e-8,original_sample_tolerance=1e-8)
    assert not op.metadata['ideal_truncation_is_residual_certificate']
    (tmp_path/'failure_and_receipt.json').write_text(json.dumps(dict(one_step_failure=failed,
        wrong_current_residual_report=report,persistence=receipt,
        direct_method_and_fallback_refused=True,production_qualification=False),indent=2)+'\n')


def test_directional_pre_Riesz_identity_and_saved_current_metric_action_guards(case,monkeypatch,tmp_path):
    provider,actions,_,rhs,_ = case
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.direct_operator.LocalRWGRiesz',_forbid)
        with pytest.raises(TypeError,match='complete prescribed'):DirectionalMultipoleOperator(provider,object())
        with pytest.raises(ValueError,match='actual admitted provider'):DirectionalMultipoleOperator(object(),actions)
        old = actions.rhs_batch;actions.rhs_batch+=1
        with pytest.raises(ValueError,match='changed'):DirectionalMultipoleOperator(provider,actions)
        actions.rhs_batch=old;actions.assert_current()
    op = DirectionalMultipoleOperator(provider,actions)
    for name,new in (('actions',object()),('provider',object()),('tree',object())):
        old = getattr(op,name);setattr(op,name,new)
        with pytest.raises(ValueError,match='binding changed'):op.apply(rhs)
        setattr(op,name,old);op.refresh()
    x = torch.randn_like(rhs,requires_grad=True)
    loss = residual_loss(op,x,rhs);old = actions.ideal_rwg_error_budget;actions.ideal_rwg_error_budget=.9
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(loss,x)
    actions.ideal_rwg_error_budget=old;op.refresh()
    loss = residual_loss(op,x,rhs);op.metric.diagonal.add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(loss,x)
    with pytest.raises(ValueError,match='changed'):op.persistence_receipt()
    (tmp_path/'solver_guards.json').write_text(json.dumps(dict(actual_action_provider_identity_pre_Riesz=True,
        no_action_reconstruction=True,solver_binding_forward=True,
        saved_residual_current_actual_action_and_Riesz_numeric_guards=True,
        production_qualification=False),indent=2)+'\n')
