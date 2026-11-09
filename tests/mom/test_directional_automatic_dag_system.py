"""Complete actual scalar automatic DAG composition without qualification claims."""
import json
import pytest
import torch
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan,AutomaticDirectionalDAGPlan
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator
from witwin.maxwell.mom.linalg import residual_loss
from test_directional_automatic_dag import _physical_plan

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_complete_automatic_DAG_actual_near_interiors_original_weak_H_current_and_Riesz(tmp_path):
    original=_physical_plan()
    config=IntegrationConfig(order=6,singular_order=3,pair_batch=4,far_pair_batch=16,near_max_order=32)
    provider=ScalarSystemBlocks(original.basis.system,original.medium,config)
    common=dict(kernel_error_budget=KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,
        balancing_impedance=abs(provider.exterior.eta),leaf_size=4,separation_ratio=.15,
        cone_subdivisions=(2**20,))
    roots=[AutomaticDirectionalRegionPlan(provider,provider.exterior,**common)]
    roots.extend(AutomaticDirectionalRegionPlan(b.basis,b.interior,geometry=g,**common)
        for b,g in zip(provider.system.bodies,provider.geometry,strict=True))
    plans=[AutomaticDirectionalDAGPlan(p,quadrature_order=12,face_batch=4,max_contexts=4096,
        tensor_workspace_budget_bytes=512*1024**2) for p in roots]
    regions=[]
    for i,p in enumerate(plans):
        print(dict(stage='complete_auto_DAG_region_build',region=i,plan=p.receipt()),flush=True)
        regions.append(p.build_region(integration=config))
        print(dict(stage='complete_auto_DAG_region_ready',region=i,receipt=regions[-1].receipt()),flush=True)
    actions=DirectionalSystemActions(provider,regions[0],regions[1:],ideal_rwg_error_budget=1e-4,
        tensor_workspace_budget_bytes=512*1024**2)
    n=provider.system.total_edges
    # This new fixture has no existing complete original weak matrix. Assemble
    # it once in bounded original blocks; no matrix archive is saved.
    dense=torch.empty((2*n,2*n),device='cuda',dtype=torch.complex128)
    for row in range(0,n,16):
        a=torch.arange(row,min(row+16,n),device='cuda');rr=torch.cat((a,a+n))
        for col in range(0,n,16):
            b=torch.arange(col,min(col+16,n),device='cuda');cc=torch.cat((b,b+n))
            dense[rr[:,None],cc]=provider(a,b)
    actual=actions.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)).mT
    eta=abs(provider.exterior.eta)
    l,r=dense.real.new_tensor([eta]*n+[1.]*n),dense.real.new_tensor([1.]*n+[eta]*n)
    weak=float((l[:,None]*(actual-dense)*r).norm()/(l[:,None]*dense*r).norm())
    record=dict(phase='complete_original_weak',full_edges=n,full_currents=2*n,weak_relative=weak,
        root_plans=[p.receipt() for p in plans],system=actions.receipt(),
        full_near_and_far_original_discrete_composition=True,original_near_quadrature_shared=True,
        positive_exterior_changing_links=plans[0].selected_positive_links,
        positive_magnetic_interior_far=[len(p.region_plan.far_nodes) for p in plans[1:]],
        complete_physics_qualification=False,qualification=False)
    (tmp_path/'complete.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert weak<1e-8 and plans[0].selected_positive_links>0
    g=torch.Generator(device='cuda').manual_seed(329170)
    x=torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=g).transpose(0,1).requires_grad_()
    z=torch.randn(x.shape,device='cuda',dtype=torch.complex128,generator=g).requires_grad_()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):y=actions.apply(x)
    H=actions.adjoint(z);current,=torch.autograd.grad(y,x,z,retain_graph=True)
    first,=torch.autograd.grad(y.abs().square().sum(),x,create_graph=True)
    tangent=torch.randn_like(x);higher,=torch.autograd.grad(first,x,tangent)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    errors=dict(original_H=relative(H,z@dense.conj()),native_H=relative(H,z@actual.conj()),
        current=relative(current,H),higher=relative(higher,2*actions.adjoint(actions.apply(tangent))))
    op=DirectionalMultipoleOperator(provider,actions)
    rhs=torch.randn_like(x);grad,=torch.autograd.grad(residual_loss(op,x,rhs),x)
    errors['Riesz_current']=relative(grad,op.adjoint(op.riesz_inverse(op.apply(x)-rhs)))
    value=actions.adjoint(x);old=plans[0].links;plans[0].links=old[::-1]
    try:
        with pytest.raises(ValueError):actions.receipt()
        with pytest.raises(ValueError):torch.autograd.grad(value,x,z)
    finally:plans[0].links=old
    actions.assert_current()
    record.update(phase='complete',errors=errors,saved_internal_tensors=len(saved),
        noncontiguous_rhs=15,whole_system_saved_planner_guard=True,solver_acceptance_not_run=True)
    (tmp_path/'complete.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert not saved and errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11
