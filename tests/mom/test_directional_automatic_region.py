"""Automatic complete partitions with independent original and MP admission."""
import json
import math
from dataclasses import replace
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig,assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan,DirectionalConeOrderError
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.direct_capacity import storage_bytes
from test_directional_far_actions import _original_global_far_block
from test_directional_sparse_coupling import _mp_integrals
from test_directional_cone_order import _cube_oracle,_mp_bounds

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _provider():
    faces=torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
    bodies=[]
    for i,(x,eps,mu) in enumerate(((-.5,2.,1.3),(.5,3.,1.5))):
        vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],device='cuda',dtype=torch.float64)*.01
        vertices[:,0]+=x
        basis=RWGBasis(SurfaceMesh(vertices,faces.clone()))
        medium=compile_medium(mw.Material(eps_r=eps,mu_r=mu,sigma_e=.01,sigma_m=.02),3e8)
        bodies.append((i,str(i),basis,medium))
    return ScalarSystemBlocks(SurfaceSystem.create(bodies),compile_medium(mw.Material(),3e8),
        IntegrationConfig(order=4,singular_order=3,pair_batch=4,far_pair_batch=8))


def _plan(basis,medium,eta,geometry=None,**kwargs):
    options=dict(kernel_error_budget=KernelErrorBudget(1e-10,1e-8),ideal_rwg_error_budget=1e-6,
        balancing_impedance=eta,leaf_size=6,cone_subdivisions=(1,2,4,8),geometry=geometry)
    options.update(kwargs)
    return AutomaticDirectionalRegionPlan(basis,medium,**options)


def _relative(a,b):return float(((a-b).norm()/b.norm()).detach())


@pytest.fixture(scope='module')
def case():
    provider=_provider();eta=abs(provider.exterior.eta)
    plans=(_plan(provider,provider.exterior,eta),*(_plan(b.basis,b.interior,eta,g)
        for b,g in zip(provider.system.bodies,provider.geometry,strict=True)))
    regions=tuple(p.build_region(integration=provider.config,quadrature_order=12,face_batch=4) for p in plans)
    actions=DirectionalSystemActions(provider,regions[0],regions[1:],ideal_rwg_error_budget=2e-6,
        balancing_impedance=eta)
    return provider,plans,regions,actions


def test_automatic_complete_actual_system_original_far_full_h_current_and_higher(case,tmp_path):
    provider,plans,regions,actions=case;n=provider.system.total_edges
    assert len(plans[0].far_nodes)==2 and len(plans[0].near_nodes)==2
    assert all(not p.far_nodes and len(p.near_nodes)==1 for p in plans[1:])
    assert all(r.far.test.retained_bytes==0 and not r.far.test.context_keys for r in regions[1:])
    for p in plans:
        tree=p.tree
        seen=torch.zeros((tree.size,tree.size),device='cuda',dtype=torch.int64)
        for a,b in (*p.far_nodes,*p.near_nodes):
            rows=tree.permutation[int(tree.starts[a]):int(tree.stops[a])]
            cols=tree.permutation[int(tree.starts[b]):int(tree.stops[b])]
            seen[rows[:,None],cols]+=1
        assert bool((seen==1).all())
        assert storage_bytes(p.tensors())==p.retained_bytes
    reference=assemble_system(provider.system,provider.exterior,provider.config)[0]
    eye=torch.eye(2*n,device='cuda',dtype=torch.complex128)
    actual=actions.apply(eye).mT
    eta=actions.balancing_impedance
    left=actual.real.new_tensor([eta]*n+[1.]*n);right=actual.real.new_tensor([1.]*n+[eta]*n)
    weak=_relative(left[:,None]*actual*right,left[:,None]*reference*right)
    assert weak<1e-8
    independent=[];far=regions[0].far
    far_actual=far.apply(eye).mT
    for a,b in plans[0].far_nodes:
        tree=plans[0].tree
        rows=tree.permutation[int(tree.starts[a]):int(tree.stops[a])]
        cols=tree.permutation[int(tree.starts[b]):int(tree.stops[b])]
        rr,cc=torch.cat((rows,rows+n)),torch.cat((cols,cols+n))
        original=_original_global_far_block(provider,rows,cols,provider.exterior)
        l=original.real.new_tensor([eta]*len(rows)+[1.]*len(rows))
        r=original.real.new_tensor([1.]*len(cols)+[eta]*len(cols))
        independent.append(_relative(l[:,None]*far_actual[rr[:,None],cc]*r,l[:,None]*original*r))
    assert max(independent)<1e-8
    gen=torch.Generator(device='cuda').manual_seed(6742)
    q=torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    z=torch.randn(q.shape,device='cuda',dtype=torch.complex128,generator=gen).requires_grad_()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        value=actions.apply(q)
    assert not saved
    h=actions.adjoint(z)
    current,=torch.autograd.grad(value,q,z,create_graph=True)
    tangent=torch.randn(q.shape,device='cuda',dtype=torch.complex128,generator=gen)
    higher,=torch.autograd.grad(current,z,tangent)
    errors=dict(original_H=_relative(h,z@reference.conj()),native_current=_relative(current,h),
        native_higher=_relative(higher,actions.apply(tangent)))
    assert errors['original_H']<1e-8
    assert errors['native_current']<1e-11 and errors['native_higher']<1e-11
    lhs=(value.conj()*z).sum();rhs=(q.conj()*h).sum()
    dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach());assert dot<1e-11
    record=dict(original_complete_weak=weak,independent_q16_far=independent,errors=errors,dot=dot,
        plans=[p.receipt() for p in plans],noncontiguous_rhs=15,saved_tensors=0,qualification=False)
    torch.save(dict(actual=actual.detach(),reference=reference.detach()),tmp_path/'weak.pt')
    (tmp_path/'actual.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_independent_mp_material_weighted_catalog_selection_and_rss(case,tmp_path):
    provider,plans,regions,_=case;plan=plans[0];owner=regions[0].far.test
    errors=[];expected_orders=[]
    with mp.workdps(100):
        eta=mp.mpf(plan.balancing_impedance);medium=provider.exterior
        k=abs(mp.mpc(medium.k));e=abs(mp.mpc(1j*medium.omega*medium.epsilon));m=abs(mp.mpc(-1j*medium.omega*medium.mu))
        expected_bounds=[]
        for (a,b),p in zip(plan.far_nodes,plan.plans,strict=True):
            test=_mp_integrals(owner,a);source=_mp_integrals(owner,b)
            ds=sum((u*v+q*r/k**2)**2 for u,q in test for v,r in source)
            us=sum((u*v)**2 for u,_ in test for v,_ in source)
            A=mp.sqrt((eta**4*e**2+m**2)*ds);B=mp.sqrt(6*eta**2*us)
            G=min(mp.mpf(plan.kernel_error_budget.green_absolute),mp.mpf(plan.per_pair_budget)/(mp.sqrt(2)*A))
            H=min(mp.mpf(plan.kernel_error_budget.gradient_component_absolute),mp.mpf(plan.per_pair_budget)/(mp.sqrt(2)*B))
            for actual,wanted in ((p.kernel_error_budget.green_absolute,G),(p.kernel_error_budget.gradient_component_absolute,H)):
                error=float(abs(mp.mpf(actual)/wanted-1));errors.append(error);assert error<1e-11
            choices=[]
            R=p.displacements[0].tolist();rt,rs=float(p.test_radii[0]),float(p.source_radii[0])
            for grid in plan.cone_subdivisions:
                carrier,key=_cube_oracle(R,grid)
                for order in range(plan.min_order,plan.max_order+1):
                    g,h=_mp_bounds(R,carrier,rt,rs,medium.k,order)
                    if mp.mpf(g)<=G and mp.mpf(h)<=H:choices.append((order,grid,key,g,h))
            best=min(choices,key=lambda v:v[:2]);expected_orders.append(best[0])
            assert int(p.orders[0])==best[0] and p.cone_keys[0].tolist()==best[2]
            bound=mp.sqrt(A*A*mp.mpf(best[3])**2+B*B*mp.mpf(best[4])**2)
            expected_bounds.append(bound)
        rss=mp.sqrt(sum(v*v for v in expected_bounds));actual=plan.receipt()['ideal_bound']['absolute_bound']
        error=float(abs(mp.mpf(actual)/rss-1));errors.append(error);assert error<1e-11
    (tmp_path/'mp.json').write_text(json.dumps(dict(relative_errors=errors,orders=expected_orders,
        precision_digits=100,absolute_tolerance=0,qualification=False),indent=2)+'\n')


def test_automatic_preflight_order_family_capacity_and_fixed_plan_saved_guards(case,monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    import witwin.maxwell.mom.fmm_directional_fft as fft
    provider,plans,regions,actions=case;eta=abs(provider.exterior.eta)
    def forbidden(*a,**k):raise AssertionError('Rejected automatic admission reached numerical factory')
    with monkeypatch.context() as patch:
        patch.setattr(rwg.DirectionalRWGProjection,'__init__',forbidden)
        patch.setattr(fft.CartesianDirectionalFFTTranslation,'__init__',forbidden)
        for option in (dict(max_pairs=1),dict(max_families=1),dict(retained_budget_bytes=1),dict(tensor_workspace_budget_bytes=1)):
            with pytest.raises(MemoryError):_plan(provider,provider.exterior,eta,**option)
        with pytest.raises(DirectionalConeOrderError):_plan(provider,provider.exterior,eta,max_order=0)
        with pytest.raises(MemoryError):plans[0].build_region(integration=provider.config,quadrature_order=12,table_retained_budget_bytes=1)
        with pytest.raises(ValueError,match='quadrature'):plans[0].build_region(integration=provider.config,quadrature_order=1)
        with pytest.raises(ValueError,match='actual exterior'):_plan(provider,provider.system.bodies[0].interior,eta)
    for mutate in ('orders','near_nodes','carriers','tree_mode'):
        p=_plan(provider,provider.exterior,eta)
        if mutate=='orders':p.orders=(12,)*len(p.orders)
        elif mutate=='near_nodes':p.near_nodes=()
        elif mutate=='carriers':p.carriers[0,0]+=1e-4
        else:p.tree.lower.requires_grad_()
        with pytest.raises(ValueError):p.assert_current()
    q=torch.ones((2,actions.shape[1]),device='cuda',dtype=torch.complex128,requires_grad=True)
    out=actions.apply(q);regions[0].far.test.carriers.requires_grad_()
    with pytest.raises(ValueError):torch.autograd.grad(out.real.sum(),q)
    regions[0].far.test.carriers.requires_grad_(False)
    actions.assert_current()
    (tmp_path/'guards.json').write_text(json.dumps(dict(capacity=True,unavailable_order=True,
        actual_medium=True,plan_and_saved_mode=True,qualification=False),indent=2)+'\n')


def test_empty_sparse_forest_complete_near_no_moments_and_native_zero(case,monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
    provider,plans,regions,_=case;p=plans[1];owner=regions[1].far.test
    def forbidden(*a,**k):raise AssertionError('Empty sparse forest allocated a numerical map')
    with monkeypatch.context() as patch:
        patch.setattr(rwg.DirectionalRWGProjection,'__init__',forbidden)
        empty=SparseDirectionalRWGProjection(p.basis,p.tree,p.carriers,(),(),p.medium.k,
            geometry=p.geometry,quadrature_order=1)
    assert not empty.context_keys and empty.retained_bytes==0
    assert storage_bytes(empty.tensors())==0 and not empty.layouts and not empty.transfers and not empty.leaves
    x=torch.ones((3,5,p.tree.size),device='cuda',dtype=torch.complex128).transpose(0,1)
    assert empty.source_upward(x)==() and empty.receiver_upward(x)==()
    for kind in ('source','receiver'):
        zero=empty._downward_dispatch((),kind,x.shape[:-1]);assert bool((zero==0).all())
    value=torch.ones((3,5,regions[1].shape[1]),device='cuda',dtype=torch.complex128).transpose(0,1).requires_grad_()
    f=regions[1].far.apply(value);assert bool((f==0).all())
    grad,=torch.autograd.grad(f.real.sum(),value,create_graph=True);assert bool((grad==0).all())
    assert regions[1].far.ideal_rwg_truncation_bound()['absolute_bound']==0
    (tmp_path/'empty.json').write_text(json.dumps(dict(owned_bytes=0,no_numeric_maps=True,near_complete=True,
        exact_zero_far=True,qualification=False),indent=2)+'\n')


def test_local_medium_values_and_tree_metadata_rejected_before_region_factory(monkeypatch,tmp_path):
    from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
    provider=_provider();body=provider.system.bodies[0]
    p=_plan(body.basis,body.interior,abs(provider.exterior.eta),provider.geometry[0])
    def forbidden(*a,**k):raise AssertionError('Stale local plan reached numerical factory')
    checked=[]
    with monkeypatch.context() as patch:
        patch.setattr(SparseDirectionalRWGProjection,'__init__',forbidden)
        for name in ('frequency','epsilon_r','mu_r','k','eta'):
            original=getattr(body.interior,name)
            object.__setattr__(body.interior,name,original*1.01)
            try:
                with pytest.raises(ValueError,match='partition/catalog'):p.receipt()
                with pytest.raises(ValueError,match='partition/catalog'):
                    p.build_region(integration=provider.config,quadrature_order=12)
                checked.append(name)
            finally:object.__setattr__(body.interior,name,original)
            p.assert_current()
        for name in ('version','leaf_size','morton_bits'):
            original=getattr(p.tree,name)
            object.__setattr__(p.tree,name,original+'_changed' if isinstance(original,str) else original+1)
            try:
                with pytest.raises(ValueError,match='partition/catalog'):p.receipt()
                with pytest.raises(ValueError,match='partition/catalog'):
                    p.build_region(integration=provider.config,quadrature_order=12)
                checked.append('tree.'+name)
            finally:object.__setattr__(p.tree,name,original)
            p.assert_current()
    (tmp_path/'local-binding.json').write_text(json.dumps(dict(checked=checked,numerical_factory_never_reached=True,
        original5PASS_not_repeated=True,qualification=False),indent=2)+'\n')


def test_active_material_tightening_independent_mp_selection_without_numeric_maps(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    provider=_provider();eta=abs(provider.exterior.eta)
    def forbidden(*a,**k):raise AssertionError('Planner-only admission allocated moments')
    with monkeypatch.context() as patch:
        patch.setattr(rwg.DirectionalRWGProjection,'__init__',forbidden)
        plan=_plan(provider,provider.exterior,eta,kernel_error_budget=KernelErrorBudget(1.,1.),
            ideal_rwg_error_budget=1e-10)
    proxy=SimpleNamespace(basis=provider,tree=plan.tree)
    limits=[];bounds=[];orders=[];errors=[]
    with mp.workdps(100):
        medium=provider.exterior;k=abs(mp.mpc(medium.k))
        e,m=abs(mp.mpc(1j*medium.omega*medium.epsilon)),abs(mp.mpc(-1j*medium.omega*medium.mu))
        pair=mp.mpf(plan.ideal_rwg_error_budget)/mp.sqrt(len(plan.far_nodes))
        for (a,b),p in zip(plan.far_nodes,plan.plans,strict=True):
            left,right=_mp_integrals(proxy,a),_mp_integrals(proxy,b)
            ds=sum((u*v+q*r/k**2)**2 for u,q in left for v,r in right)
            us=sum((u*v)**2 for u,_ in left for v,_ in right)
            A,B=mp.sqrt((mp.mpf(eta)**4*e**2+m**2)*ds),mp.sqrt(6*mp.mpf(eta)**2*us)
            G,H=pair/(mp.sqrt(2)*A),pair/(mp.sqrt(2)*B)
            assert G<1 and H<1
            for actual,wanted in ((p.kernel_error_budget.green_absolute,G),(p.kernel_error_budget.gradient_component_absolute,H)):
                errors.append(float(abs(mp.mpf(actual)/wanted-1)));assert errors[-1]<1e-11
            candidates=[];R=p.displacements[0].tolist()
            for grid in plan.cone_subdivisions:
                carrier,key=_cube_oracle(R,grid)
                for order in range(13):
                    g,h=_mp_bounds(R,carrier,float(p.test_radii[0]),float(p.source_radii[0]),medium.k,order)
                    if mp.mpf(g)<=G and mp.mpf(h)<=H:candidates.append((order,grid,key,g,h))
            best=min(candidates,key=lambda v:v[:2]);orders.append(best[0])
            assert int(p.orders[0])==best[0] and p.cone_keys[0].tolist()==best[2]
            limits.append([float(G),float(H)])
            bounds.append(mp.sqrt(A*A*mp.mpf(best[3])**2+B*B*mp.mpf(best[4])**2))
        rss=mp.sqrt(sum(v*v for v in bounds))
        errors.append(float(abs(mp.mpf(plan.receipt()['ideal_bound']['absolute_bound'])/rss-1)))
        assert errors[-1]<1e-11 and rss<=mp.mpf(plan.ideal_rwg_error_budget)
    (tmp_path/'tightening.json').write_text(json.dumps(dict(material_limits=limits,orders=orders,
        relative_errors=errors,precision_digits=100,absolute_tolerance=0,no_numeric_maps=True,
        old6PASS_not_repeated=True,qualification=False),indent=2)+'\n')
