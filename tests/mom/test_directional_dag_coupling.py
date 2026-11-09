"""Actual scalar material admission and shared changing-carrier DAG FAR maps."""
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.direct_capacity import storage_bytes
from witwin.maxwell.mom.fmm_directional_moments import DirectionalRWGDAGProjection
from witwin.maxwell.mom.fmm_directional_rwg import (DirectionalFFTElectromagneticM2L,
    DirectionalDAGFarActions, DirectionalSparseFarActions, directional_sparse_admission_workspace_bytes)
from test_directional_far_actions import _original_global_far_block
from test_directional_sparse_coupling import _mp_integrals

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _fixture(lossy=False, order=8):
    medium = compile_medium(mw.Material(eps_r=2., mu_r=1.3,
        sigma_e=.01 if lossy else 0., sigma_m=.02 if lossy else 0.), 3e8)
    exterior = compile_medium(mw.Material(), 3e8)
    bases = [RWGBasis(sphere_mesh(radius=.02, center=(x,0.,0.), subdivisions=0,
        device='cuda', dtype=torch.float64)) for x in (-.5,.5)]
    system = SurfaceSystem.create([(i,str(i),basis,medium) for i,basis in enumerate(bases)])
    provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(order=3, singular_order=3))
    lower, upper = provider.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=2)
    children, starts, stops = tree.children.tolist(), tree.starts.tolist(), tree.stops.tolist()
    permutation = tree.permutation.tolist()
    nodes = [next(n for n,row in enumerate(children) if n>0 and sum(c>=0 for c in row)>=2
        and 3<=stops[n]-starts[n]<=6 and all(body.edge_offset<=edge<body.edge_offset+body.basis.n_edges
            for edge in permutation[starts[n]:stops[n]])) for body in system.bodies]
    a,b = nodes
    step = 1e-4 if order>=6 else .003
    carriers = lower.new_tensor([[-1.,step,0.],[-1.,-step,step],[-1.,0.,0.],
        [1.,step,0.],[1.,-step,step],[1.,0.,0.]])
    carriers = carriers/carriers.norm(dim=-1,keepdim=True)
    requests = tuple((node,family) for node in (a,b) for family in (0,1,3,4))
    phase = 4 if order>=6 else 0
    links, seen, pending = [],set(),list(requests)
    while pending:
        key = pending.pop()
        if key in seen:continue
        seen.add(key)
        for child in children[key[0]]:
            if child>=0:
                target = (child,2 if key[1]<3 else 5)
                links.append((key,target,phase if key[1] not in (2,5) else 0))
                pending.append(target)
    owner = DirectionalRWGDAGProjection(provider,tree,carriers,(order,order,order+phase,order,order,order+phase),requests,tuple(links),medium.k,
        quadrature_order=12 if order>=6 else 4,face_batch=4,phase_relative_budget=1e-12 if order>=6 else .5)
    index = {key:i for i,key in enumerate(owner.context_keys)}
    pairs = ((index[a,0],index[b,0]),(index[b,3],index[a,3]))
    return owner,medium,abs(exterior.eta),pairs


def _coupling(fixture,pair,**kwargs):
    owner,medium,eta,_ = fixture
    a,b = pair;order = owner.node_orders[a]
    options = dict(kernel_error_budget=KernelErrorBudget(1e-6,1e-4) if order>=6 else KernelErrorBudget(1.,1.),
        ideal_rwg_error_budget=1e-4 if order>=6 else 10.,balancing_impedance=eta)
    options.update(kwargs)
    scale = torch.minimum((owner.centers[a]-owner.centers[b]).norm()/(2*order+2),torch.maximum(owner.radii[a],owner.radii[b]))
    return DirectionalFFTElectromagneticM2L.from_dag_contexts(owner,a,owner,b,medium,scale,**options)


def _build(fixture):
    owner,medium,eta,pairs = fixture
    couplings = tuple(_coupling(fixture,pair) for pair in pairs)
    return DirectionalDAGFarActions(owner,owner,medium,couplings,ideal_rwg_error_budget=1e-4 if owner.node_orders[0]>=6 else 20.,
        balancing_impedance=eta,tensor_workspace_budget_bytes=512*1024**2)


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def _mp_products(owner):
    products = [mp.mpf(1)]*len(owner.node_counts)
    for parent in reversed(range(len(products))):
        paths = []
        for child in owner.children[parent]:
            shift = owner.transfers[parent,child]
            if shift.exact_phase:delta = mp.mpf(0)
            else:
                u,v = [list(map(mp.mpf,t.tolist())) for t in (shift.direction,shift.target_direction)]
                x = abs(mp.mpc(shift.wave_number))*mp.mpf(float(shift.child_radius))*mp.sqrt(sum((a-b)**2 for a,b in zip(u,v)))
                delta = mp.exp(2*x)*x**(shift.phase_order+1)/mp.factorial(shift.phase_order+1)
            paths.append((1+delta)*products[child])
        if paths:products[parent] = max(paths)
    return products


@pytest.mark.parametrize('lossy',(False,True))
def test_actual_dag_material_original_weak_H_current_higher_and_mp_phase_weights(lossy,tmp_path):
    fixture = _fixture(lossy);actions = _build(fixture);owner,medium,eta,_ = fixture
    n = owner.tree.size
    assert owner.receipt()['shared_child_contexts']>0
    reference = torch.zeros(actions.shape,device='cuda',dtype=torch.complex128)
    columns,mp_errors,phase_values = [],[],[]
    with mp.workdps(100):
        products = _mp_products(owner)
        for coupling,rectangle in zip(actions.couplings,actions.rectangles,strict=True):
            start,stop,low,high,a,b = rectangle
            rows,cols = owner.tree.permutation[start:stop],owner.tree.permutation[low:high]
            rr,cc = torch.cat((rows,rows+n)),torch.cat((cols,cols+n))
            reference[rr[:,None],cc] = _original_global_far_block(owner.basis,rows,cols,medium)
            columns.append(cc)
            bound = coupling.ideal_rwg_truncation_bound()
            pure = coupling.scalar.receipt()['ideal_truncation_admission']
            delta = products[a]*products[b]-1
            phase_values.append(float(delta));assert delta>0
            mp_errors.append(float(abs(mp.mpf(bound['combined_phase_relative_bound'])/delta-1)))
            D = mp.sqrt(sum(mp.mpf(v)**2 for v in coupling.scalar.displacement.tolist()))
            d = D-sum(mp.mpf(v) for v in coupling.scalar.support_radii.tolist())
            G = mp.exp(-mp.mpf(medium.k.imag)*d)/(4*mp.pi*d)
            H = G*(abs(mp.mpc(medium.k))+1/d)
            for name,M in (('green_absolute',G),('gradient_component_absolute',H)):
                E = mp.mpf(pure[name]);expected = E+delta*(M+E)
                mp_errors.append(float(abs(mp.mpf(bound['effective_'+name])/expected-1)))
            left = _mp_integrals(owner,bound['test_physical_node']);right = _mp_integrals(owner,bound['source_physical_node'])
            k = abs(mp.mpc(medium.k));z = mp.mpf(eta)
            e,m = abs(mp.mpc(1j*medium.omega*medium.epsilon)),abs(mp.mpc(-1j*medium.omega*medium.mu))
            dsum = sum((u*v+q*r/k**2)**2 for u,q in left for v,r in right)
            usum = sum((u*v)**2 for u,_ in left for v,_ in right)
            expected = mp.sqrt((z**4*e**2+m**2)*dsum*mp.mpf(bound['effective_green_absolute'])**2
                +6*z**2*usum*mp.mpf(bound['effective_gradient_component_absolute'])**2)
            mp_errors.append(float(abs(mp.mpf(bound['absolute_bound'])/expected-1)))
            assert bound['changing_carrier_DAG'] and bound['shared_child_indegree_not_multiplied']
        expected = mp.sqrt(sum(mp.mpf(c.ideal_rwg_truncation_bound()['absolute_bound'])**2 for c in actions.couplings))
        mp_errors.append(float(abs(mp.mpf(actions.ideal_rwg_truncation_bound()['absolute_bound'])/expected-1)))
    assert max(mp_errors)<1e-11
    columns = torch.cat(columns).unique()
    actual = torch.zeros_like(reference)
    actual[:,columns] = actions.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)[columns]).mT
    left,right = actual.real.new_tensor([eta]*n+[1.]*n),actual.real.new_tensor([1.]*n+[eta]*n)
    absolute = (left[:,None]*(actual-reference)*right).norm()
    weak = float(absolute/(left[:,None]*reference*right).norm())
    record = dict(lossy=lossy,phase='original_weak',weak_relative=weak,weak_absolute=float(absolute),
        mp100_relative_errors=mp_errors,mp_absolute_tolerance=0,positive_combined_phase=phase_values,
        original_q16_rectangles=actions.rectangles,projection_receipt=owner.receipt(),receipt=actions.receipt(),qualification=False)
    (tmp_path/'dag_weak.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert weak<1e-8 and float(absolute)<=actions.ideal_rwg_truncation_bound()['absolute_bound']
    gen = torch.Generator(device='cuda').manual_seed(245813)
    current = torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=gen).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):forward = actions.apply(current)
    H = actions.adjoint(dual)
    derivative, = torch.autograd.grad(forward,current,dual,retain_graph=True)
    inverse, = torch.autograd.grad(H,dual,current,retain_graph=True)
    first, = torch.autograd.grad(forward.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn_like(current);second, = torch.autograd.grad(first,current,tangent)
    lhs,rhs = (forward.conj()*dual).sum(),(current.conj()*H).sum()
    errors = dict(original_H=_relative(H,dual@reference.conj()),native_H=_relative(H,dual@actual.conj()),
        native_forward=_relative(forward,current@actual.mT),current=_relative(derivative,H),
        adjoint_current=_relative(inverse,actions.apply(current)),higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    record.update(phase='complete',errors=errors,saved_internal_tensors=len(saved),noncontiguous_physical_RHS=15,
        projection_owned_bytes=owner.retained_bytes,coupling_owned_bytes=sum(storage_bytes(c.tensors()) for c in actions.couplings),
        caller_projection_and_coupling_excluded=True,complete_system_certified=False)
    (tmp_path/'dag_far.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert not saved and not current.is_contiguous()
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11


def test_dag_positive_phase_budget_before_coefficients_protocol_capacity_and_saved_backward(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as module
    fixture = _fixture(True,order=1);owner,medium,eta,pairs = fixture
    coupling = _coupling(fixture,pairs[0]);bound = coupling.ideal_rwg_truncation_bound()
    pure = coupling.scalar.receipt()['ideal_truncation_admission']
    assert bound['combined_phase_relative_bound']>0
    assert bound['effective_green_absolute']>pure['green_absolute']
    middle = (bound['effective_green_absolute']+pure['green_absolute'])/2
    def forbidden(*a,**k):raise AssertionError('Rejected DAG budget reached coefficient allocation')
    with monkeypatch.context() as patch:
        patch.setattr(module,'CartesianDirectionalFFTTranslation',forbidden)
        for option,exc in ((dict(kernel_error_budget=KernelErrorBudget(middle,1.)),ValueError),
                (dict(ideal_rwg_error_budget=bound['absolute_bound']/2),ValueError),
                (dict(tensor_workspace_budget_bytes=directional_sparse_admission_workspace_bytes(owner,owner,dag=True)-1),MemoryError)):
            with pytest.raises(exc):_coupling(fixture,pairs[0],**option)
        for pair in ((-1,pairs[0][1]),(True,pairs[0][1]),(len(owner.context_keys),pairs[0][1]),(pairs[1][1],pairs[0][1])):
            with pytest.raises(ValueError):DirectionalFFTElectromagneticM2L.from_dag_contexts(owner,pair[0],owner,pair[1],medium,
                owner.centers.new_tensor(.02),kernel_error_budget=KernelErrorBudget(1.,1.),ideal_rwg_error_budget=10.,balancing_impedance=eta)
        with pytest.raises(TypeError):DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,pairs[0][0],owner,pairs[0][1],medium,
            owner.centers.new_tensor(.02),kernel_error_budget=KernelErrorBudget(1.,1.),ideal_rwg_error_budget=10.,balancing_impedance=eta)
    with pytest.raises(TypeError):DirectionalSparseFarActions(owner,owner,medium,(coupling,),ideal_rwg_error_budget=10.)
    with pytest.raises(ValueError,match='overlap'):DirectionalDAGFarActions(owner,owner,medium,(coupling,coupling),
        ideal_rwg_error_budget=20.,balancing_impedance=eta)
    actions = _build(fixture)
    with pytest.raises(MemoryError):DirectionalDAGFarActions(owner,owner,medium,actions.couplings,
        ideal_rwg_error_budget=20.,balancing_impedance=eta,tensor_workspace_budget_bytes=actions.planned_tensor_workspace_bytes-1)
    x = torch.ones((2,2*owner.tree.size),device='cuda',dtype=torch.complex128,requires_grad=True)
    out = actions.apply(x)
    shift = next(iter(owner.transfers.values()));shift.source_phase.requires_grad_()
    with pytest.raises(ValueError):torch.autograd.grad(out,x,torch.ones_like(out))
    shift.source_phase.requires_grad_(False);actions.assert_current()
    out = actions.adjoint(x);actions.couplings[0].scalar.coefficients.requires_grad_()
    with pytest.raises(ValueError):torch.autograd.grad(out,x,torch.ones_like(out))
    actions.couplings[0].scalar.coefficients.requires_grad_(False);actions.assert_current()
    (tmp_path/'dag_guard.json').write_text(json.dumps(dict(positive_phase_only_budget_rejected=True,
        coefficient_factory_forbidden=True,duplicate_physical_rectangles_rejected=True,
        saved_forward_and_H_native_configuration_guarded=True,receipt=actions.receipt(),qualification=False),indent=2)+'\n')
