"""Real-support bound selection reaches actual free-space regional couplings."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_order import MultipoleOrderError, plan_multipole_order
from witwin.maxwell.mom.cluster import build_support_octree

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RULE = 'real_support_min_v1'


@pytest.mark.parametrize('storage', ('dense','compact'))
def test_order_selection_actual_far_action_gradient_and_policy_mutation(storage, monkeypatch):
    exterior = compile_medium(mw.Material(),3e8)
    bodies = [(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0,0),subdivisions=0,device='cuda')),
               compile_medium(mw.Material(eps_r=2.+i,mu_r=1.2+.1*i),3e8)) for i,x in enumerate((-.5,.5))]
    system = SurfaceSystem.create(bodies)
    provider = ScalarSystemBlocks(system,exterior,IntegrationConfig(order=3,singular_order=3))
    budget = mw.KernelErrorBudget(1e-6,1e-5)
    config = mw.FMMConfig(order=1,max_order=8,leaf_size=30,table_storage=storage,
        shift_storage='separable_v1',kernel_error_budget=budget,kernel_bound_rule=RULE)
    op = MultipoleSystemActions(provider,**config.construction())
    plan = op.order_plan
    assert plan['kernel_bound_rule']==RULE and plan['table_bytes']==op.table_bytes
    assert plan['attempts'][-1]['kernel_pass']
    assert all(not a['kernel_pass'] for a in plan['attempts'][:-1])
    couplings = [c for region in (op.exterior,*op.interiors) for _,_,c in region.far]
    assert couplings
    for coupling in couplings:
        assert coupling.kernel_bound_rule==RULE
        bound = coupling.truncation_bound()
        assert bound['kernel_bound_rule']==RULE
        assert bound['green_absolute'] <= budget.green_absolute
        assert bound['gradient_component_absolute'] <= budget.gradient_component_absolute
        assert coupling.receipt()['kernel_bound_rule']==RULE
    matrix,_ = assemble_system(system,exterior,provider.config)
    actual = op.apply(torch.eye(op.shape[0],device=op.device,dtype=op.dtype)).mT
    n=system.total_edges; eta=abs(exterior.eta)
    left=actual.real.new_tensor([eta]*n+[1.]*n); right=actual.real.new_tensor([1.]*n+[eta]*n)
    error=(left[:,None]*(actual-matrix)*right).norm()/(left[:,None]*matrix*right).norm()
    assert float(error)<1e-8
    x=torch.randn((2,op.shape[0]),device=op.device,dtype=op.dtype,requires_grad=True)
    y=torch.randn_like(x)
    gradient,=torch.autograd.grad(op.apply(x),x,y)
    torch.testing.assert_close(gradient,op.adjoint(y),rtol=2e-11,atol=1e-14)
    # Every policy-bearing owner rejects mutation before reuse.
    coupling=couplings[0]; coupling.kernel_bound_rule='bivariate_cauchy_v1'
    with pytest.raises(ValueError,match='changed'):op.apply(x)
    coupling.kernel_bound_rule=RULE
    op.exterior.kernel_bound_rule='bivariate_cauchy_v1'
    with pytest.raises(ValueError,match='changed'):op.assert_current()
    op.exterior.kernel_bound_rule=RULE
    op.kernel_bound_rule='bivariate_cauchy_v1'
    with pytest.raises(ValueError,match='changed'):op.assert_current()
    op.kernel_bound_rule=RULE; op.assert_current()
    import witwin.maxwell.mom.fmm_system as construction
    monkeypatch.setattr(construction,'NestedRWGMultipoles',lambda *a,**k:pytest.fail('Admission must precede moments'))
    with pytest.raises(MultipoleOrderError) as failure:
        MultipoleSystemActions(provider,**{**config.construction(),'table_budget_bytes':1})
    assert failure.value.receipt['reason']=='table_budget'
    assert failure.value.receipt['kernel_bound_rule']==RULE
    print(dict(storage=storage,selected_order=plan['selected_order'],scaled_action_error=float(error)),flush=True)


def test_real_support_rwg_ordered_aggregation_and_unavailable_guard():
    medium = compile_medium(mw.Material(),3e8)
    lower = torch.tensor([[0.,0,0],[1.,0,0]],device='cuda',dtype=torch.float64)
    upper=lower+.1
    tree=build_support_octree(lower,upper,leaf_size=1)
    # Real nonzero log-integral channels exercise ordered aggregation.
    statistics=torch.full((tree.node_count,3),-3.,device='cuda',dtype=torch.float64)
    budget=mw.KernelErrorBudget(1e-5,1e-4)
    plan=plan_multipole_order(((tree,medium),),budget,min_order=1,max_order=12,
        rwg_statistics=(statistics,),rwg_absolute_budget=1.,balancing_impedance=abs(medium.eta),
        table_storage='compact',kernel_bound_rule=RULE)
    assert plan['kernel_bound_rule']==RULE and plan['rwg_bound']['absolute_bound']<=1.
    assert plan['attempts'][-1]['rwg_pass']
    with pytest.raises(ValueError,match='kernel_bound_rule'):
        plan_multipole_order(((tree,medium),),budget,kernel_bound_rule='invalid')
    overlap=build_support_octree(lower,lower+.7,leaf_size=1)
    # Ratio .5 leaves these as near blocks, with no unjustified far bound.
    near_only=plan_multipole_order(((overlap,medium),),budget,max_order=1,kernel_bound_rule=RULE)
    assert near_only['attempts'][0]['far_pairs']==0
