"""Finite automatic child-context sharing on actual admitted physical supports."""
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan,AutomaticDirectionalDAGPlan
from test_directional_far_actions import _original_global_far_block
from test_directional_dag_coupling import _mp_products
from test_directional_sparse_coupling import _mp_integrals

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _physical_plan():
    exterior=compile_medium(mw.Material(),3e8)
    bodies=[]
    for i in range(3):
        basis=RWGBasis(sphere_mesh(radius=.02,center=(i,1e-6*(i%2),0.),
            subdivisions=0,device='cuda',dtype=torch.float64))
        medium=compile_medium(mw.Material(eps_r=2.+i,mu_r=1.3+.1*i,sigma_e=.01,sigma_m=.02),3e8)
        bodies.append((i,str(i),basis,medium))
    provider=ScalarSystemBlocks(SurfaceSystem.create(bodies),exterior,IntegrationConfig(order=3,singular_order=3))
    return AutomaticDirectionalRegionPlan(provider,exterior,kernel_error_budget=KernelErrorBudget(1e-6,1e-4),
        ideal_rwg_error_budget=1e-4,balancing_impedance=abs(exterior.eta),leaf_size=4,
        separation_ratio=.15,cone_subdivisions=(2**20,))


@pytest.fixture(scope='module')
def root_plan():return _physical_plan()


def _select(root_plan,**kw):
    return AutomaticDirectionalDAGPlan(root_plan,quadrature_order=12,face_batch=4,
        max_contexts=4096,max_candidates=10000,tensor_workspace_budget_bytes=512*1024**2,**kw)


def test_automatic_positive_shared_child_physical_partition_all_ancestors_and_cost(root_plan,tmp_path):
    selected=_select(root_plan)
    receipt=selected.receipt();print(dict(root=root_plan.receipt(),DAG=receipt),flush=True)
    record=dict(root=root_plan.receipt(),selected=receipt,qualification=False)
    (tmp_path/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    assert root_plan.far_nodes and receipt['shared_children']>0
    assert receipt['selected_positive_links']>0
    assert receipt['selected_component_cost']<receipt['initial_component_cost']
    coverage=torch.zeros((root_plan.tree.size,root_plan.tree.size),device='cuda',dtype=torch.int64)
    for a,b in (*root_plan.far_nodes,*root_plan.near_nodes):
        rows=root_plan.tree.permutation[int(root_plan.tree.starts[a]):int(root_plan.tree.stops[a])]
        cols=root_plan.tree.permutation[int(root_plan.tree.starts[b]):int(root_plan.tree.stops[b])]
        coverage[rows[:,None],cols]+=1
    assert bool((coverage==1).all())
    # Independent MP100 products enumerate all complete paths, including every
    # ancestor whose closure changed. Parent indegree is not a path multiplier.
    with mp.workdps(100):
        lookup={key:i for i,key in enumerate(selected.context_keys)}
        products=[mp.mpf(1)]*len(lookup);errors=[]
        for parent in reversed(selected.context_keys):
            values=[]
            for a,c,L in selected.links:
                if a!=parent:continue
                source=selected.carriers[a[1]].tolist();target=selected.carriers[c[1]].tolist()
                difference=mp.sqrt(sum((mp.mpf(x)-mp.mpf(y))**2 for x,y in zip(source,target,strict=True)))
                x=difference*abs(mp.mpc(root_plan.medium.k))*mp.mpf(float(root_plan.radii[c[0]]))
                delta=mp.mpf(0) if source==target else mp.exp(2*x)*x**(L+1)/mp.factorial(L+1)
                values.append((1+delta)*products[lookup[c]])
            if values:products[lookup[parent]]=max(values)
            defect=products[lookup[parent]]-1
            actual=mp.mpf(float(torch.expm1(selected.path_products[lookup[parent]])))
            if defect:errors.append(float(abs(actual/defect-1)))
            else:assert actual==0
        assert errors and max(errors)<1e-11
    again=_select(root_plan)
    assert selected.orders==again.orders and selected.requests==again.requests and selected.links==again.links
    assert torch.equal(selected.carriers,again.carriers) and torch.equal(selected.path_products,again.path_products)
    record.update(all_physical_pairs_once=True,all_ancestor_mp100_relative=max(errors),mp_absolute_tolerance=0,
        deterministic_selection=True,whole_closure_and_unique_layout_cost=True)
    (tmp_path/'selection.json').write_text(json.dumps(record,indent=2)+'\n')


def test_automatic_dag_capacity_and_configuration_binding_before_factories(root_plan,monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_order as module
    def forbid(*args,**kwargs):raise AssertionError('numeric factory reached before capacity admission')
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_directional_moments.DirectionalRWGDAGProjection.__init__',forbid)
        for options in (dict(max_candidates=True),dict(phase_relative_budget=math.nan),dict(tensor_workspace_budget_bytes=1)):
            with pytest.raises((ValueError,MemoryError)):
                module.AutomaticDirectionalDAGPlan(root_plan,quadrature_order=12,**options)
        selected=_select(root_plan)
        with pytest.raises(MemoryError):selected.build_region(integration=root_plan.basis.config,table_retained_budget_bytes=1)
        old=selected.links;selected.links=old[::-1]
        with pytest.raises(ValueError):selected.receipt()
        with pytest.raises(ValueError):selected.build_region(integration=root_plan.basis.config)
        selected.links=old;selected.assert_current()
        selected.carriers.requires_grad_(True)
        with pytest.raises(ValueError):selected.receipt()
        selected.carriers.requires_grad_(False);selected.assert_current()
    (tmp_path/'guards.json').write_text(json.dumps(dict(capacities_before_factories=True,
        links_and_native_modes_bound=True,qualification=False),indent=2)+'\n')


@pytest.fixture(scope='module')
def selected_far(root_plan):
    selected=_select(root_plan)
    far=selected.build_far()
    print(dict(stage='selected_automatic_actual_far_built',planner=selected.receipt(),maps=far.test.receipt()),flush=True)
    return selected,far


def test_selected_automatic_actual_far_original_q16_H_current_higher_all_root_MP100(selected_far,tmp_path):
    selected,far=selected_far;owner=far.test;n=owner.tree.size;medium=far.medium;eta=far.balancing_impedance
    assert selected.selected_positive_links>0 and owner.plan_guard is selected
    reference=torch.zeros(far.shape,device='cuda',dtype=torch.complex128)
    columns=[];mp_errors=[];positive=0
    with mp.workdps(100):
        products=_mp_products(owner)
        for coupling,rectangle in zip(far.couplings,far.rectangles,strict=True):
            start,stop,low,high,a,b=rectangle
            rows,cols=owner.tree.permutation[start:stop],owner.tree.permutation[low:high]
            rr,cc=torch.cat((rows,rows+n)),torch.cat((cols,cols+n))
            reference[rr[:,None],cc]=_original_global_far_block(owner.basis,rows,cols,medium)
            columns.append(cc);bound=coupling.ideal_rwg_truncation_bound()
            pure=coupling.scalar.receipt()['ideal_truncation_admission'];delta=products[a]*products[b]-1
            if delta:
                positive+=1;mp_errors.append(float(abs(mp.mpf(bound['combined_phase_relative_bound'])/delta-1)))
            else:assert bound['combined_phase_relative_bound']==0
            D=mp.sqrt(sum(mp.mpf(v)**2 for v in coupling.scalar.displacement.tolist()))
            d=D-sum(mp.mpf(v) for v in coupling.scalar.support_radii.tolist())
            G=mp.exp(-mp.mpf(medium.k.imag)*d)/(4*mp.pi*d);H=G*(abs(mp.mpc(medium.k))+1/d)
            for name,M in (('green_absolute',G),('gradient_component_absolute',H)):
                E=mp.mpf(pure[name]);expected=E+delta*(M+E)
                mp_errors.append(float(abs(mp.mpf(bound['effective_'+name])/expected-1)))
            left=_mp_integrals(owner,bound['test_physical_node']);right=_mp_integrals(owner,bound['source_physical_node'])
            k=abs(mp.mpc(medium.k));z=mp.mpf(eta)
            e,m=abs(mp.mpc(1j*medium.omega*medium.epsilon)),abs(mp.mpc(-1j*medium.omega*medium.mu))
            dsum=sum((u*v+q*r/k**2)**2 for u,q in left for v,r in right)
            usum=sum((u*v)**2 for u,_ in left for v,_ in right)
            expected=mp.sqrt((z**4*e**2+m**2)*dsum*mp.mpf(bound['effective_green_absolute'])**2
                +6*z**2*usum*mp.mpf(bound['effective_gradient_component_absolute'])**2)
            mp_errors.append(float(abs(mp.mpf(bound['absolute_bound'])/expected-1)))
            assert bound['absolute_bound']<=selected.region_plan.per_pair_budget
        expected=mp.sqrt(sum(mp.mpf(c.ideal_rwg_truncation_bound()['absolute_bound'])**2 for c in far.couplings))
        mp_errors.append(float(abs(mp.mpf(far.ideal_rwg_truncation_bound()['absolute_bound'])/expected-1)))
        assert expected<=selected.region_plan.ideal_rwg_error_budget
    assert positive>0 and max(mp_errors)<1e-11
    columns=torch.cat(columns).unique();actual=torch.zeros_like(reference)
    actual[:,columns]=far.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)[columns]).mT
    l,r=reference.real.new_tensor([eta]*n+[1.]*n),reference.real.new_tensor([1.]*n+[eta]*n)
    absolute=(l[:,None]*(actual-reference)*r).norm()
    weak=float(absolute/(l[:,None]*reference*r).norm())
    record=dict(phase='all_selected_FAR_original',weak_relative=weak,weak_absolute=float(absolute),
        all_root_MP100_relative_errors=mp_errors,mp_absolute_tolerance=0,positive_affected_root_pairs=positive,
        full_physical_current_coordinates=2*n,complete_near_plus_far_system=False,
        original_q16_far_pairs=len(far.couplings),planner=selected.receipt(),qualification=False)
    (tmp_path/'actual.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert weak<1e-8 and float(absolute)<=far.ideal_rwg_truncation_bound()['absolute_bound']
    gen=torch.Generator(device='cuda').manual_seed(772391)
    x=torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    z=torch.randn(x.shape,device='cuda',dtype=torch.complex128,generator=gen).requires_grad_()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):y=far.apply(x)
    H=far.adjoint(z);current,=torch.autograd.grad(y,x,z,retain_graph=True)
    adjoint_current,=torch.autograd.grad(H,z,x,retain_graph=True)
    first,=torch.autograd.grad(y.abs().square().sum(),x,create_graph=True)
    tangent=torch.randn_like(x);second,=torch.autograd.grad(first,x,tangent)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    lhs,rhs=(y.conj()*z).sum(),(x.conj()*H).sum()
    errors=dict(original_H=relative(H,z@reference.conj()),native_H=relative(H,z@actual.conj()),
        forward=relative(y,x@actual.mT),current=relative(current,H),
        adjoint_current=relative(adjoint_current,far.apply(x)),higher=relative(second,2*far.adjoint(far.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    record.update(phase='selected_FAR_complete',errors=errors,noncontiguous_rhs=15,saved_internal_tensors=len(saved))
    (tmp_path/'actual.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert not saved and errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11


def test_bound_automatic_plan_guards_actions_and_saved_H_after_build(selected_far,tmp_path):
    selected,far=selected_far;trips=[]
    for obj,name,new in ((selected,'links',selected.links[::-1]),
            (selected,'phase_relative_budget',.5),
            (selected.region_plan,'cone_subdivisions',(1,)),(far.test,'plan_guard',None)):
        current=torch.ones((2,far.shape[0]),device='cuda',dtype=torch.complex128,requires_grad=True)
        value=far.adjoint(current);old=getattr(obj,name);setattr(obj,name,new)
        try:
            with pytest.raises(ValueError):far.receipt()
            with pytest.raises(ValueError):far.apply(current)
            with pytest.raises(ValueError):far.adjoint(current)
            with pytest.raises(ValueError):torch.autograd.grad(value,current,torch.ones_like(value))
        finally:setattr(obj,name,old)
        far.assert_current();trips.append(name)
    (tmp_path/'bound_guards.json').write_text(json.dumps(dict(planner_reassignment_saved_H_guards=trips,
        guarded_receipt_forward_H_backward=True,qualification=False),indent=2)+'\n')


def test_exact_same_carrier_mixed_order_automatic_shared_closures(root_plan,tmp_path):
    entries=[]
    for i,body in enumerate(root_plan.basis.system.bodies):
        basis=RWGBasis(sphere_mesh(radius=.02,center=(i,0.,0.),subdivisions=0,
            device='cuda',dtype=torch.float64))
        entries.append((i,str(i),basis,body.interior))
    provider=ScalarSystemBlocks(SurfaceSystem.create(entries),root_plan.medium,root_plan.basis.config)
    aligned=AutomaticDirectionalRegionPlan(provider,provider.exterior,
        kernel_error_budget=root_plan.kernel_error_budget,ideal_rwg_error_budget=root_plan.ideal_rwg_error_budget,
        balancing_impedance=root_plan.balancing_impedance,leaf_size=4,separation_ratio=.15,
        cone_subdivisions=(2**20,))
    selected=_select(aligned);incoming={}
    for parent,child,L in selected.links:
        incoming.setdefault(child,[]).append(parent)
        assert L==0 and torch.equal(selected.carriers[parent[1]],selected.carriers[child[1]])
    mixed=[child for child,parents in incoming.items() if len({selected.orders[p[1]] for p in parents})>1]
    record=dict(phase='exact_mixed_order_sharing',root=aligned.receipt(),DAG=selected.receipt(),
        mixed_order_shared_children=len(mixed),all_phase_paths_exact_zero=bool((selected.path_products==0).all()),qualification=False)
    (tmp_path/'exact_mixed.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert mixed and selected.selected_positive_links==0 and bool((selected.path_products==0).all())
    assert selected.selected_component_cost<selected.initial_component_cost


def test_unmergeable_padded_order_and_finite_search_keep_physical_partition(root_plan,tmp_path):
    bounded=AutomaticDirectionalRegionPlan(root_plan.basis,root_plan.medium,
        kernel_error_budget=root_plan.kernel_error_budget,ideal_rwg_error_budget=root_plan.ideal_rwg_error_budget,
        balancing_impedance=root_plan.balancing_impedance,leaf_size=4,separation_ratio=.15,
        cone_subdivisions=(2**20,),min_order=12,max_order=12)
    selected=_select(bounded)
    assert selected.selected_positive_links==0 and selected.selected_merges==0
    assert selected.selected_component_cost==selected.initial_component_cost
    assert selected.region_plan.far_nodes==root_plan.far_nodes and selected.region_plan.near_nodes==root_plan.near_nodes
    finite=AutomaticDirectionalDAGPlan(root_plan,quadrature_order=12,face_batch=4,max_contexts=4096,
        max_candidates=1,tensor_workspace_budget_bytes=512*1024**2)
    assert finite.search_limit_reached and finite.candidate_attempts==1
    assert finite.region_plan is root_plan
    assert finite.receipt()['ideal_bound']['absolute_bound']<=root_plan.ideal_rwg_error_budget
    (tmp_path/'unmergeable.json').write_text(json.dumps(dict(padded_order12=selected.receipt(),
        finite_search=finite.receipt(),original_near_far_partition_preserved=True,
        no_resource_win_claim_for_rejected_merges=True,qualification=False),indent=2)+'\n')
