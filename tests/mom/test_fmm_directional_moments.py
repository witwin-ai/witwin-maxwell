"""Independent all-node carrier quadrature, four Hermitian maps and actual EM."""
from dataclasses import replace
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_block
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L
from witwin.maxwell.mom.fmm_expansion import multi_indices
from witwin.maxwell.mom.direct_system import DirectSystemActions
from test_fmm_directional_rwg import _independent_edge_moments

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _basis(center=(0., 0., 0.), radius=.04):
    return RWGBasis(sphere_mesh(radius=radius, center=center, subdivisions=0, device='cuda', dtype=torch.float64))


def _space(kind):
    if kind == 'single':
        basis = _basis()
        triangles = basis.mesh.triangles
        supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), 1)
        lower, upper = supports.amin(1), supports.amax(1)
    else:
        medium = compile_medium(mw.Material(), 3e8)
        system = SurfaceSystem.create([(i, str(i), _basis((x,0.,0.)), medium) for i,x in enumerate((-.3,.3))])
        basis = ScalarSystemBlocks(system, medium, IntegrationConfig(order=3, singular_order=3))
        lower, upper = basis.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=len(lower) if kind == 'mixed_leaf' else 2)
    direction = lower.new_tensor([.7, -.2, .3]);direction = direction/direction.norm()
    return basis, tree, direction


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def _flatten(values):return torch.cat([v.flatten(-2) for v in values], -1)


def _unflatten(value, owner):
    return tuple(v.reshape(*value.shape[:-1], owner.count, 4) for v in value.split(owner.count*4, -1))


def _independent_all_node_maps(owner, q):
    maps = [torch.zeros((owner.tree.node_count, owner.count, 4, owner.tree.size), device='cuda', dtype=torch.complex128) for _ in range(2)]
    for node in range(owner.tree.node_count):
        edges = owner.tree.permutation[owner.starts[node]:owner.stops[node]]
        groups = owner.basis._body_selection(edges) if isinstance(owner.basis, ScalarSystemBlocks) else ((None,None),)
        for bi, selected in groups:
            if bi is None:basis, ids, local_edges = owner.basis, edges, edges
            else:
                body = owner.basis.system.bodies[bi];basis,ids = body.basis,edges[selected];local_edges = ids-body.edge_offset
            for index, sign in enumerate((-1,1)):
                fields = _independent_edge_moments(basis,local_edges,owner.centers[node],owner.direction,
                                                  sign*owner.wave_number,owner.order,q)
                maps[index][node, :, :, ids] = fields.permute(1,2,0)
    return tuple(v.flatten(0,2) for v in maps)


@pytest.mark.parametrize('kind', ('single','global','mixed_leaf'))
def test_complete_all_node_direct_quadrature_four_maps_and_current_backward(kind,tmp_path):
    basis,tree,direction = _space(kind)
    owner = NestedDirectionalRWGProjection(basis,tree,direction,5.+.8j,3,quadrature_order=8,face_batch=4)
    if kind == 'mixed_leaf':assert tree.node_count == 1 and owner.receipt()['integrated_leaf_body_groups'] == 2
    else:assert int(tree.depth.max()) >= 2
    source,receiver = _independent_all_node_maps(owner,12)
    generator = torch.Generator(device='cuda').manual_seed(2271)
    currents = torch.randn((3,5,tree.size),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    degrees = torch.tensor([sum(a) for a in multi_indices(owner.order)],device='cuda')
    scale = torch.cat([owner.radii[node].pow(degrees).repeat_interleave(4) for node in range(tree.node_count)])
    coefficients = (torch.randn((3,5,len(scale)),device='cuda',dtype=torch.complex128,generator=generator)/scale).transpose(0,1).requires_grad_()
    expectations = dict(source_upward=currents@source.mT, receiver_upward=currents@receiver.conj().mT,
                        source_downward=coefficients@source.conj(), receiver_downward=coefficients@receiver)
    errors, gradients, dots = {},{},{}
    for name in ('source_upward','receiver_upward','source_downward','receiver_downward'):
        upward = name.endswith('upward');value = currents if upward else coefficients
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
            raw = getattr(owner,name)(value if upward else _unflatten(value,owner))
            result = _flatten(raw) if upward else raw
        assert not saved
        expected = expectations[name]
        errors[name] = _relative(result/scale,expected/scale) if upward else _relative(result,expected)
        dual = coefficients.detach() if upward else currents.detach()
        derivative, = torch.autograd.grad(result,value,dual,retain_graph=True)
        opposite = {'source_upward':'source_downward','receiver_upward':'receiver_downward',
                    'source_downward':'source_upward','receiver_downward':'receiver_upward'}[name]
        reverse = getattr(owner,opposite)(_unflatten(dual,owner) if upward else dual)
        reverse = reverse if upward else _flatten(reverse)
        gradients[name] = _relative(derivative,reverse)
        lhs,rhs = (result.conj()*dual).sum(),(value.conj()*reverse).sum()
        dots[name] = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    # A root-only output induces None cotangents at every unused node.
    root = owner.source_upward(currents)[0]
    root_dual = torch.randn_like(root);partial, = torch.autograd.grad(root,currents,root_dual)
    explicit = root_dual.flatten(-2)@source[:owner.count*4].conj()
    partial_error = _relative(partial,explicit)
    energy = sum(v.abs().square().sum() for v in owner.source_upward(currents))
    first, = torch.autograd.grad(energy,currents,create_graph=True)
    tangent = torch.randn_like(currents)
    second, = torch.autograd.grad(first,currents,tangent)
    second_expected = 2*(tangent@source.mT)@source.conj()
    second_error = _relative(second,second_expected)
    separation = _relative(receiver,source.conj());assert separation > 1e-3
    assert max(*errors.values(),*gradients.values(),*dots.values(),partial_error,second_error) < 1e-11
    record = dict(kind=kind,depth=int(tree.depth.max()),four_map_errors=errors,current_gradient_errors=gradients,
                  complex_dot_errors=dots,unused_node_cotangent_error=partial_error,higher_current_derivative_error=second_error,
                  receiver_vs_conjugate_source=separation,
                  independently_integrated_all_node_columns=True,oracle_quadrature=12,receipt=owner.receipt(),qualification=False)
    (tmp_path/'tree.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_owned_shared_inventory_preflight_mutation_and_bounded_input_finite_checks(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_moments as module
    basis,tree,direction = _space('single')
    owned = NestedDirectionalRWGProjection(basis,tree,direction,5.+.8j,2,quadrature_order=6,face_batch=4)
    shared = NestedDirectionalRWGProjection(basis,tree,direction,5.+.8j,2,quadrature_order=6,face_batch=4,
                                            geometry=owned.geometry,layout=owned.layout)
    for owner in (owned,shared):
        tensors = owner.tensors()
        assert len({(t.device,t.untyped_storage().data_ptr()) for t in tensors}) == len(tensors)
        assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
        assert sum(t.untyped_storage().nbytes() for t in tensors) == owner.retained_bytes == owner.planned_retained_bytes
    assert owned.retained_bytes-shared.retained_bytes == owned.owned_geometry_bytes+owned.owned_layout_bytes
    def forbidden(*args,**kwargs):raise AssertionError('Rejected budget reached quadrature/layout/geometry construction')
    with monkeypatch.context() as patch:
        for name in ('PreparedRWGGeometry','DirectionalRWGProjection','SeparableMultipoleShift'):patch.setattr(module,name,forbidden)
        with pytest.raises(MemoryError,match='before geometry'):
            NestedDirectionalRWGProjection(basis,tree,direction,5.+.8j,2,quadrature_order=6,face_batch=4,
                retained_budget_bytes=owned.retained_bytes-1)
        with pytest.raises(MemoryError,match='before geometry'):
            NestedDirectionalRWGProjection(basis,tree,direction,5.+.8j,2,quadrature_order=6,face_batch=4,
                tensor_workspace_budget_bytes=owned.planned_tensor_workspace_bytes-1)
    x = torch.randn((3,5,tree.size),device='cuda',dtype=torch.complex128).transpose(0,1).requires_grad_()
    finite_shapes = [];native = torch.isfinite
    def bounded(value):
        if value.dtype == torch.complex128 and value.ndim >= 2:
            finite_shapes.append(tuple(value.shape));assert value.shape[0] <= owned.rhs_batch
        return native(value)
    with monkeypatch.context() as patch:
        patch.setattr(torch,'isfinite',bounded)
        values = shared.source_upward(x)
        shared.receiver_downward(values)
        torch.autograd.grad(values[0],x,torch.ones_like(values[0]))
    assert finite_shapes
    assert all(v.shape==(0,3,shared.count,4) for v in shared.source_upward(x[:0]))
    assert shared.receiver_downward(tuple(v[:0] for v in values)).shape==(0,3,tree.size)
    with monkeypatch.context() as patch:
        patch.setattr(module,'DirectionalRWGProjection',forbidden)
        shifted = replace(tree,lower=tree.lower+10,upper=tree.upper+10)
        with pytest.raises(ValueError,match='complete RWG supports'):
            NestedDirectionalRWGProjection(basis,shifted,direction,5.+.8j,2,quadrature_order=6,face_batch=4)
    for name in ('source_upward','receiver_upward'):
        with pytest.raises(ValueError,match='finite'):getattr(shared,name)(x.detach()*float('nan'))
    for name in ('source_downward','receiver_downward'):
        with pytest.raises(ValueError,match='finite'):getattr(shared,name)(tuple(v.detach()*float('inf') for v in values))
    partial = shared.source_upward(x)[0]
    shared.layout.layouts[0][0].add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(partial,x,torch.ones_like(partial))
    record = dict(owned_retained_bytes=owned.retained_bytes,shared_retained_bytes=shared.retained_bytes,
                  owned_geometry_bytes=owned.owned_geometry_bytes,owned_layout_bytes=owned.owned_layout_bytes,
                  maximum_finite_check_RHS=max(s[0] for s in finite_shapes),rhs_batch=owned.rhs_batch,
                  actual_unique_storages=True,qualification=False)
    (tmp_path/'ownership.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


class _SelectedTreeEM:
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint
    def __init__(self,target,origin,a,b,coupling):
        self.target,self.origin,self.a,self.b,self.coupling = target,origin,a,b,coupling
        self.rows = target.tree.permutation[target.starts[a]:target.stops[a]]
        self.cols = origin.tree.permutation[origin.starts[b]:origin.stops[b]]
    def assert_current(self):
        self.target.assert_current();self.origin.assert_current();self.coupling.assert_current()
    def _action(self,value,*,adjoint):
        self.assert_current()
        owner,edges = (self.target,self.rows) if adjoint else (self.origin,self.cols)
        n = len(edges);current = value.new_zeros((*value.shape[:-1],2,owner.tree.size))
        current[...,edges] = torch.stack((value[...,:n],value[...,n:]),-2)
        target = self.origin if adjoint else self.target
        node = self.b if adjoint else self.a
        coefficients = [value.new_zeros((*value.shape[:-1],2,count,4)) for count in target.node_counts]
        if adjoint:
            coefficients[node] = self.coupling.adjoint(self.target.receiver_upward(current)[self.a])
            result = self.origin.source_downward(coefficients)[...,self.cols]
        else:
            coefficients[node] = self.coupling.apply(self.origin.source_upward(current)[self.b])
            result = self.target.receiver_downward(coefficients)[...,self.rows]
        return torch.cat((result[...,0,:],result[...,1,:]),-1)


@pytest.mark.parametrize('lossy',(False,True))
def test_nonroot_tree_directional_em_original_high_ka_weak_columns_and_H(lossy,tmp_path):
    test,source = _basis((-.5,.1,0.),.02),_basis((.5,-.1,.05),.02)
    def tree(basis):
        triangles = basis.mesh.triangles
        supports = torch.cat((triangles[basis.plus_faces],triangles[basis.minus_faces]),1)
        return build_support_octree(supports.amin(1),supports.amax(1),leaf_size=2)
    tt,st = tree(test),tree(source)
    a = next(i for i,row in enumerate(tt.children.tolist()) if i>0 and any(c>=0 for c in row))
    b = next(i for i,row in enumerate(st.children.tolist()) if i>0 and any(c>=0 for c in row))
    displacement = (tt.lower[a]+tt.upper[a]-st.lower[b]-st.upper[b])*.5
    direction = displacement/displacement.norm()
    medium = compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01 if lossy else 0.,sigma_m=.02 if lossy else 0.),2.4e9)
    target = NestedDirectionalRWGProjection(test,tt,direction,medium.k,8,quadrature_order=12,face_batch=4)
    origin = NestedDirectionalRWGProjection(source,st,direction,medium.k,8,quadrature_order=12,face_batch=4)
    coupling = DirectionalFFTElectromagneticM2L(displacement,direction,medium,displacement.new_tensor(.02),8)
    op = _SelectedTreeEM(target,origin,a,b,coupling)
    actual = op.apply(torch.eye(2*len(op.cols),device='cuda',dtype=torch.complex128)).mT
    original,_ = scalar_regional_block(test,source,op.rows,op.cols,(medium,),IntegrationConfig(order=12,singular_order=3))
    nr,nc = len(op.rows),len(op.cols);eta = abs(medium.eta)
    left = actual.real.new_tensor([eta]*nr+[1.]*nr);right = actual.real.new_tensor([1.]*nc+[eta]*nc)
    weak = _relative(left[:,None]*actual*right,left[:,None]*original*right)
    generator = torch.Generator(device='cuda').manual_seed(2272)
    x = torch.randn((3,5,2*nc),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    y = torch.randn((*x.shape[:-1],2*nr),device='cuda',dtype=torch.complex128,generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):applied = op.apply(x)
    assert not saved
    adjoint = op.adjoint(y);gradient, = torch.autograd.grad(applied,x,y)
    errors = dict(actual_matrix_adjoint=_relative(adjoint,y@actual.conj()),original_adjoint=_relative(adjoint,y@original.conj()),
                  current_gradient=_relative(gradient,adjoint),forward=_relative(applied,x@actual.mT))
    lhs,rhs = (applied.conj()*y).sum(),(x.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    record = dict(lossy=lossy,target_node=a,source_node=b,target_depth=int(tt.depth[a]),source_depth=int(st.depth[b]),
                  selected_rows=op.rows.tolist(),selected_cols=op.cols.tolist(),actual_ka=float(abs(medium.k)*displacement.norm()),
                  original_balanced_all_column_error=weak,errors=errors,complex_dot_error=dot,
                  quadrature_convergence_certified=False,full_directional_system_qualified=False,qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert weak < 1e-8 and errors['original_adjoint'] < 1e-8
    assert max(errors['actual_matrix_adjoint'],errors['current_gradient'],errors['forward'],dot) < 1e-11
