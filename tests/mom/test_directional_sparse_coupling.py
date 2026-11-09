"""Sparse material-bound PMCHWT controls in full physical coordinates."""
import json
import math
import weakref
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
from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import (DirectionalFFTElectromagneticM2L,
    DirectionalSparseFarActions, DirectionalNestedFarActions, directional_sparse_admission_workspace_bytes)
from test_directional_far_actions import _original_global_far_block

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
    children = tree.children.tolist(); starts = tree.starts.tolist(); stops = tree.stops.tolist()
    permutation = tree.permutation.tolist()
    nodes = [next(n for n,row in enumerate(children) if n>0 and sum(c>=0 for c in row)>=2
        and 3<=stops[n]-starts[n]<=6 and all(body.edge_offset<=edge<body.edge_offset+body.basis.n_edges
            for edge in permutation[starts[n]:stops[n]])) for body in system.bodies]
    a,b = nodes; sources = tuple(c for c in children[b] if c>=0)[:2]
    carriers = lower.new_tensor([[-1.,0.,0.],[1.,0.,0.]])
    # One owner with both directions and overlapping ancestor/descendant
    # requests. Context IDs deliberately differ from physical node IDs.
    requests = ((a,0),(b,0),(sources[0],0),(a,1),(b,1))
    owner = SparseDirectionalRWGProjection(provider,tree,carriers,(order,order),requests,medium.k,
        quadrature_order=12 if order>=6 else 4, face_batch=4, rhs_batch=4)
    index = {key:i for i,key in enumerate(owner.context_keys)}
    pairs = [(index[a,0],index[c,0]) for c in sources]+[(index[b,1],index[a,1])]
    return owner, medium, abs(exterior.eta), pairs


def _coupling(owner, medium, eta, pair, **kwargs):
    a,b = pair; displacement = owner.centers[a]-owner.centers[b]
    order = owner.node_orders[a]
    options = dict(kernel_error_budget=KernelErrorBudget(1e-6,1e-4) if order>=6 else KernelErrorBudget(1.,1.),
        ideal_rwg_error_budget=1e-4 if order>=6 else 10., balancing_impedance=eta)
    options.update(kwargs)
    scale = torch.minimum(displacement.norm()/(2*order+2), torch.maximum(owner.radii[a],owner.radii[b]))
    return DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,a,owner,b,medium,scale,**options)


def _build(fixture, **kwargs):
    owner, medium, eta, pairs = fixture
    couplings = tuple(_coupling(owner,medium,eta,pair) for pair in pairs)
    options = dict(ideal_rwg_error_budget=1e-4 if owner.node_orders[0]>=6 else 10.,
        balancing_impedance=eta, rhs_batch=4, tensor_workspace_budget_bytes=512*1024**2)
    options.update(kwargs)
    return DirectionalSparseFarActions(owner,owner,medium,couplings,**options)


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def _mp_integrals(owner, node):
    edges = owner.tree.permutation[int(owner.tree.starts[node]):int(owner.tree.stops[node])].tolist()
    values = []
    for edge in edges:
        bi,body = next((i,b) for i,b in enumerate(owner.basis.system.bodies) if b.edge_offset<=edge<b.edge_offset+b.basis.n_edges)
        basis = body.basis; local = edge-body.edge_offset
        geometry = owner.basis.geometry[bi]
        U,Q = mp.mpf(0),mp.mpf(0)
        for face in (int(basis.plus_faces[local]),int(basis.minus_faces[local])):
            position = int((basis.face_edges[face]==local).nonzero()[0])
            triangle = [list(map(mp.mpf,v)) for v in geometry.triangles[face].tolist()]
            origin = triangle[position]; other = [v for j,v in enumerate(triangle) if j!=position]
            a,b = [[x-y for x,y in zip(v,origin)] for v in other]
            cross = [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
            area = mp.sqrt(sum(v*v for v in cross))/2
            mean = (sum(v*v for v in a)+sum(v*v for v in b)+sum(x*y for x,y in zip(a,b)))/6
            slope = abs(mp.mpf(float(geometry.slopes[face,position])))
            U+=area*slope*mp.sqrt(mean);Q+=2*area*slope
        values.append((U,Q))
    return values


@pytest.mark.parametrize('lossy',(False,True))
def test_actual_sparse_multi_carrier_original_weak_H_current_higher_and_mp_weights(lossy,tmp_path):
    fixture = _fixture(lossy); actions = _build(fixture); owner = fixture[0]
    n = owner.tree.size; reference = torch.zeros(actions.shape,device='cuda',dtype=torch.complex128)
    columns, rows, mp_errors = [],[],[]
    for coupling,rectangle in zip(actions.couplings,actions.rectangles,strict=True):
        start,stop,low,high,a,b = rectangle
        selected_rows, selected_cols = owner.tree.permutation[start:stop],owner.tree.permutation[low:high]
        rr,cc = torch.cat((selected_rows,selected_rows+n)),torch.cat((selected_cols,selected_cols+n))
        reference[rr[:,None],cc] = _original_global_far_block(owner.basis,selected_rows,selected_cols,actions.medium)
        rows.append(rr);columns.append(cc)
        bound = coupling.ideal_rwg_truncation_bound()
        assert bound['test_context']==a and bound['source_context']==b
        assert bound['test_physical_node']==owner.context_keys[a][0]
        assert bound['source_physical_node']==owner.context_keys[b][0]
        assert bound['combined_phase_relative_bound']==0.
        assert bound['phase_path_included'] and not bound['quadrature_included']
        with mp.workdps(100):
            left = _mp_integrals(owner,bound['test_physical_node']);right = _mp_integrals(owner,bound['source_physical_node'])
            eta = mp.mpf(actions.balancing_impedance);k = abs(mp.mpc(actions.medium.k))
            e,m = abs(mp.mpc(1j*actions.medium.omega*actions.medium.epsilon)),abs(mp.mpc(-1j*actions.medium.omega*actions.medium.mu))
            dsum = sum((u*v+q*r/k**2)**2 for u,q in left for v,r in right)
            usum = sum((u*v)**2 for u,_ in left for v,_ in right)
            G,H = mp.mpf(bound['effective_green_absolute']),mp.mpf(bound['effective_gradient_component_absolute'])
            expected = mp.sqrt((eta**4*e**2+m**2)*dsum*G**2+6*eta**2*usum*H**2)
            mp_errors.append(float(abs(mp.mpf(bound['absolute_bound'])/expected-1)))
            assert mp_errors[-1]<1e-11
        assert storage_bytes(coupling.tensors())==coupling.scalar.retained_bytes
    columns = torch.cat(columns).unique(); rows = torch.cat(rows).unique()
    actual = torch.zeros_like(reference)
    actual[:,columns] = actions.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)[columns]).mT
    eta = actions.balancing_impedance
    left,right = actual.real.new_tensor([eta]*n+[1.]*n),actual.real.new_tensor([1.]*n+[eta]*n)
    absolute = (left[:,None]*(actual-reference)*right).norm()
    weak = float(absolute/(left[:,None]*reference*right).norm())
    diagnostic = dict(lossy=lossy,phase='original_weak',weak_relative=weak,weak_absolute=float(absolute),
        mp100_weighted_relative_errors=mp_errors,rectangles=actions.rectangles,physical_tree_nodes=owner.tree.node_count,
        sparse_contexts=len(owner.context_keys),hypothetical_two_complete_owners=2*owner.tree.node_count,
        coupling_receipts=[c.receipt() for c in actions.couplings],receipt=actions.receipt(),qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    torch.save(dict(actual=actual.detach(),reference=reference.detach()),tmp_path/'weak.pt');print(diagnostic,flush=True)
    assert weak<1e-8 and float(absolute)<=actions.ideal_rwg_truncation_bound()['absolute_bound']
    with mp.workdps(100):
        expected = mp.sqrt(sum(mp.mpf(c.ideal_rwg_truncation_bound()['absolute_bound'])**2 for c in actions.couplings))
        assert actions.ideal_rwg_truncation_bound()['absolute_bound']==pytest.approx(float(expected),rel=1e-11,abs=0)
    generator = torch.Generator(device='cuda').manual_seed(41131)
    current = torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):forward = actions.apply(current)
    assert not saved and not current.is_contiguous()
    H = actions.adjoint(dual)
    derivative, = torch.autograd.grad(forward,current,dual,retain_graph=True)
    adjoint_derivative, = torch.autograd.grad(H,dual,current,retain_graph=True)
    first, = torch.autograd.grad(forward.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn_like(current);second, = torch.autograd.grad(first,current,tangent)
    lhs,rhs = (forward.conj()*dual).sum(),(current.conj()*H).sum()
    errors = dict(original_H=_relative(H,dual@reference.conj()),native_H=_relative(H,dual@actual.conj()),
        native_forward=_relative(forward,current@actual.mT),current=_relative(derivative,H),
        adjoint_current=_relative(adjoint_derivative,actions.apply(current)),
        higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    absent_rows = torch.ones(2*n,device='cuda',dtype=torch.bool);absent_rows[rows]=False
    omitted = current.detach().clone();omitted[...,columns]=0
    assert bool((actions.apply(omitted)==0).all()) and bool((forward[...,absent_rows]==0).all())
    assert actions.tensors()==() and actions.retained_bytes==0
    assert len(owner.context_keys)<2*owner.tree.node_count
    assert any(a!=owner.context_keys[a][0] or b!=owner.context_keys[b][0] for a,b in fixture[-1])
    diagnostic.update(phase='complete',errors=errors,owned_projection_bytes=owner.retained_bytes,
        owned_coupling_bytes=sum(c.scalar.retained_bytes for c in actions.couplings),new_far_owned_bytes=0,
        component_map_workspace_bytes=owner.planned_tensor_workspace_bytes,saved_internal_tensors=len(saved),
        fifteen_noncontiguous_physical_RHS=True,omitted_physical_edges_exact_zero=True)
    (tmp_path/'sparse_far.json').write_text(json.dumps(diagnostic,indent=2)+'\n');print(diagnostic,flush=True)
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11


def test_sparse_id_medium_carrier_order_budget_capacity_and_saved_backward_guards(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_fft as module
    fixture = _fixture(True,order=1);owner,medium,eta,pairs = fixture
    coupling = _coupling(owner,medium,eta,pairs[0]);bound = coupling.ideal_rwg_truncation_bound()
    def forbidden(*args,**kwargs):raise AssertionError('Rejected sparse admission reached scalar coefficients')
    rejects = []
    with monkeypatch.context() as patch:
        patch.setattr(module,'directional_helmholtz_taylor',forbidden)
        for context in (-1,True,len(owner.context_keys)):
            with pytest.raises(ValueError):
                DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,context,owner,pairs[0][1],medium,
                    owner.centers.new_tensor(.02),kernel_error_budget=KernelErrorBudget(1.,1.),
                    ideal_rwg_error_budget=10.,balancing_impedance=eta)
        for label,kw in [('ideal',dict(ideal_rwg_error_budget=bound['absolute_bound']*.99)),
                         ('kernel',dict(kernel_error_budget=KernelErrorBudget(1e-300,1e-300))),
                         ('workspace',dict(tensor_workspace_budget_bytes=directional_sparse_admission_workspace_bytes(owner,owner)-1)),
                         ('retained',dict(retained_budget_bytes=coupling.scalar.retained_bytes-1)),
                         ('radius',dict(test_radius=owner.radii[pairs[0][0]]))]:
            with pytest.raises((ValueError,MemoryError)):_coupling(owner,medium,eta,pairs[0],**kw)
            rejects.append(label)
        a,b = pairs[0];other = next(i for i,key in enumerate(owner.context_keys) if key[1]!=owner.context_keys[a][1])
        with pytest.raises(ValueError):_coupling(owner,medium,eta,(a,other))
        other_medium = compile_medium(mw.Material(eps_r=3.),3e8)
        with pytest.raises(ValueError):_coupling(owner,other_medium,eta,pairs[0])
    actions = _build(fixture)
    with pytest.raises(ValueError):
        DirectionalSparseFarActions(owner,owner,medium,actions.couplings+actions.couplings[:1],
            ideal_rwg_error_budget=10.,balancing_impedance=eta,tensor_workspace_budget_bytes=512*1024**2)
    with pytest.raises(TypeError):DirectionalNestedFarActions(owner,owner,medium,(),ideal_rwg_error_budget=10.)
    with pytest.raises(MemoryError):_build(fixture,tensor_workspace_budget_bytes=actions.planned_tensor_workspace_bytes-1)
    with pytest.raises(ValueError):_build(fixture,balancing_impedance=abs(medium.eta))
    modes = []
    for kind in ('context','carrier','geometry','material','coupling_budget','action_budget','rectangle'):
        f = _fixture(True,order=1);action = _build(f);o,m,_,_ = f
        current = torch.ones(action.shape[1],device='cuda',dtype=torch.complex128,requires_grad=True)
        output = action.apply(current)
        if kind=='context':o.context_node_ids.add_(1)
        elif kind=='carrier':o.carriers.mul_(.99)
        elif kind=='geometry':o.basis.geometry[0].triangles.add_(.001)
        elif kind=='material':object.__setattr__(m,'k',m.k*1.01)
        elif kind=='coupling_budget':action.couplings[0].ideal_rwg_error_budget*=2
        elif kind=='action_budget':action.ideal_rwg_error_budget*=2
        else:action.rectangles=action.rectangles[::-1]
        with pytest.raises(ValueError):action.receipt()
        with pytest.raises(ValueError):torch.autograd.grad(output,current,torch.ones_like(output))
        modes.append(kind)
    # Numerical FAR action and protocol identities are shared, including exact
    # sparse slot addressing and bounded outer physical current chunks.
    for name in ('_action','apply','adjoint','_state','assert_current','ideal_rwg_truncation_bound'):
        assert getattr(DirectionalSparseFarActions,name) is getattr(DirectionalNestedFarActions,name)
    record = dict(before_coefficients_rejections=rejects,saved_output_tripwires=modes,
        physical_overlap_rejected=True,common_exterior_impedance_required=True,shared_native_protocol=True,qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_sparse_physical_chunks_cache_lifetime_and_empty_far(monkeypatch,tmp_path):
    f = _fixture(True,order=1);actions = _build(f);owner,medium,eta,_ = f
    refs = [];calls = [];max_rhs = 0
    for adjoint in (False,True):
        up,down = ('receiver_upward','source_downward') if adjoint else ('source_upward','receiver_downward')
        native_up,native_down = getattr(owner,up),getattr(owner,down)
        def upward(current):
            nonlocal max_rhs
            assert all(r() is None for r in refs);refs.clear()
            assert current.shape[0]<=4;max_rhs=max(max_rhs,current.shape[0]);calls.append(('up',adjoint))
            values = native_up(current);refs.extend(weakref.ref(v) for v in values);return values
        def downward(values):
            calls.append(('down',adjoint));refs.extend(weakref.ref(v) for v in values);return native_down(values)
        with monkeypatch.context() as patch:
            patch.setattr(owner,up,upward);patch.setattr(owner,down,downward)
            q = torch.ones((3,5,actions.shape[1]),device='cuda',dtype=torch.complex128).transpose(0,1)
            (actions.adjoint if adjoint else actions.apply)(q)
        assert all(r() is None for r in refs);refs.clear()
    assert len(calls)==16 and max_rhs==4
    empty = DirectionalSparseFarActions(owner,owner,medium,(),ideal_rwg_error_budget=10.,balancing_impedance=eta)
    assert empty.ideal_rwg_truncation_bound()['absolute_bound']==0
    for method in (empty.apply,empty.adjoint):
        q = torch.empty((0,actions.shape[1]),device='cuda',dtype=torch.complex128,requires_grad=True)
        out = method(q);grad, = torch.autograd.grad(out.real.sum(),q);assert grad.shape==q.shape
        assert bool((method(torch.ones(actions.shape[1],device='cuda',dtype=torch.complex128))==0).all())
    record = dict(max_physical_RHS=max_rhs,shared_owner_maps_once_per_chunk=True,cache_released_between_chunks=True,
        empty_far_current_backward=True,qualification=False)
    (tmp_path/'chunks.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_sparse_unequal_order_native_modes_and_saved_adjoint_configurations(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_fft as module
    owner,medium,eta,pairs = _fixture(True,order=1)
    other = SparseDirectionalRWGProjection(owner.basis,owner.tree,owner.carriers,(2,2),
        tuple(owner.context_keys[i] for i in owner.request_indices),medium.k,
        quadrature_order=4,face_batch=4)
    a,b = pairs[0];key = owner.context_keys[b];other_b = other.context_keys.index(key)
    def forbidden(*args,**kwargs):raise AssertionError('Unequal sparse orders reached coefficients')
    with monkeypatch.context() as patch:
        patch.setattr(module,'directional_helmholtz_taylor',forbidden)
        with pytest.raises(ValueError,match='order'):
            DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,a,other,other_b,medium,
                owner.centers.new_tensor(.02),kernel_error_budget=KernelErrorBudget(1.,1.),
                ideal_rwg_error_budget=10.,balancing_impedance=eta)
    tripwires = []
    # Owned fixed tensor mode flips leave _version unchanged. Test both the
    # coefficient and full physical wrappers, including saved adjoint outputs.
    for kind in ('scalar_coefficients','scalar_scale','scalar_direction','full_carrier_mode','full_tree_mode'):
        f = _fixture(True,order=1);actions = _build(f);o,_,_,_ = f
        coupling = actions.couplings[0]
        q = torch.ones(actions.shape[0],device='cuda',dtype=torch.complex128,requires_grad=True)
        saved = actions.adjoint(q)
        z = torch.ones((2,coupling.count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
        scalar_saved = coupling.adjoint(z)
        if kind=='scalar_coefficients':tensor = coupling.scalar.coefficients
        elif kind=='scalar_scale':tensor = coupling.scalar.scale
        elif kind=='scalar_direction':tensor = coupling.scalar.direction
        elif kind=='full_carrier_mode':tensor = o.carriers
        else:tensor = o.tree.lower
        version = tensor._version;tensor.requires_grad_(True);assert tensor._version==version
        for receipt in (coupling.receipt,actions.receipt):
            with pytest.raises(ValueError):receipt()
        with pytest.raises(ValueError):torch.autograd.grad(saved,q,torch.ones_like(saved))
        with pytest.raises(ValueError):torch.autograd.grad(scalar_saved,z,torch.ones_like(scalar_saved))
        tripwires.append(kind)
    record = dict(unequal_order_rejected_before_coefficients=True,
        native_configuration_mode_saved_adjoint_tripwires=tripwires,
        original6PASS_not_repeated=True,qualification=False)
    (tmp_path/'modes.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
