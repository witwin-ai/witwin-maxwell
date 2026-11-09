"""Actual full-support variable-carrier trees and independent weak/current maps."""
from dataclasses import replace
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L
from witwin.maxwell.mom.fmm_expansion import multi_indices
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks,scalar_regional_block
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.cluster import build_support_octree
from test_fmm_directional_moments import _space,_basis,_flatten,_relative,_SelectedTreeEM
from test_fmm_directional_rwg import _independent_edge_moments

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _schedule(tree,direction,root_order=2,phase_order=1,step=1e-7,anchor_depth=0):
    orders=tuple(root_order+phase_order*int(d) for d in tree.depth.tolist())
    phases=(0,)+(phase_order,)*(tree.node_count-1)
    offset=direction.new_tensor([.2,-.4,.7])
    dirs=direction[None]+(tree.depth-anchor_depth)[:,None]*step*offset
    dirs=dirs/dirs.norm(dim=-1,keepdim=True)
    return dirs,orders,phases


def _owner(kind):
    basis,tree,direction=_space(kind)
    dirs,orders,phases=_schedule(tree,direction)
    owner=NestedDirectionalRWGProjection(basis,tree,dirs[0],5.+.8j,orders[0],
        quadrature_order=12,face_batch=4,node_directions=dirs,node_orders=orders,phase_orders=phases)
    return owner


def _unflatten(value,owner):
    return tuple(v.reshape(*value.shape[:-1],count,4) for v,count in
        zip(value.split(tuple(c*4 for c in owner.node_counts),-1),owner.node_counts))


def _direct(owner,q):
    result=[[],[]]
    for node in range(owner.tree.node_count):
        fields=[torch.zeros((owner.node_counts[node],4,owner.tree.size),device='cuda',dtype=torch.complex128) for _ in range(2)]
        edges=owner.tree.permutation[owner.starts[node]:owner.stops[node]]
        groups=owner.basis._body_selection(edges) if isinstance(owner.basis,ScalarSystemBlocks) else ((None,None),)
        for bi,selected in groups:
            if bi is None:basis,ids,local=owner.basis,edges,edges
            else:
                body=owner.basis.system.bodies[bi];basis,ids=body.basis,edges[selected];local=ids-body.edge_offset
            for index,sign in enumerate((-1,1)):
                independent=_independent_edge_moments(basis,local,owner.centers[node],owner.node_directions[node],
                    sign*owner.wave_number,owner.node_orders[node],q)
                fields[index][:,:,ids]=independent.permute(1,2,0)
        for i in range(2):result[i].append(fields[i].flatten(0,1))
    return tuple(torch.cat(rows,0) for rows in result)


@pytest.mark.parametrize('kind',('single','global','mixed_leaf'))
def test_variable_carrier_orders_independent_all_node_four_maps_H_higher_and_support(kind,tmp_path):
    owner=_owner(kind);tree=owner.tree
    if kind=='mixed_leaf':assert tree.node_count==1 and owner.receipt()['integrated_leaf_body_groups']==2
    else:
        assert int(tree.depth.max())>=2 and len(set(owner.node_counts))>1
        assert any(not torch.equal(shift.direction,shift.target_direction) for shift in owner.transfers.values())
    source,receiver=_direct(owner,16)
    gen=torch.Generator(device='cuda').manual_seed(2291)
    current=torch.randn((3,5,tree.size),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    scale=torch.cat([owner.radii[n].pow(torch.tensor([sum(a) for a in multi_indices(p)],device='cuda')).repeat_interleave(4)
                     for n,p in enumerate(owner.node_orders)])
    coefficients=(torch.randn((3,5,len(scale)),device='cuda',dtype=torch.complex128,generator=gen)/scale).transpose(0,1).requires_grad_()
    expected=dict(source_upward=current@source.mT,receiver_upward=current@receiver.conj().mT,
        source_downward=coefficients@source.conj(),receiver_downward=coefficients@receiver)
    paired=dict(source_upward='source_downward',source_downward='source_upward',
        receiver_upward='receiver_downward',receiver_downward='receiver_upward')
    errors={};gradients={};dots={}
    for name in expected:
        up=name.endswith('upward');value=current if up else coefficients;saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
            raw=getattr(owner,name)(value if up else _unflatten(value,owner))
            actual=_flatten(raw) if up else raw
        assert not saved
        errors[name]=_relative(actual/scale,expected[name]/scale) if up else _relative(actual,expected[name])
        dual=coefficients.detach() if up else current.detach()
        derivative,=torch.autograd.grad(actual,value,dual)
        reverse=getattr(owner,paired[name])(_unflatten(dual,owner) if up else dual)
        reverse=reverse if up else _flatten(reverse)
        gradients[name]=_relative(derivative,reverse)
        lhs,rhs=(actual.conj()*dual).sum(),(value.conj()*reverse).sum()
        dots[name]=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    root=owner.source_upward(current)[0];dual=torch.randn_like(root)
    partial,=torch.autograd.grad(root,current,dual)
    partial_error=_relative(partial,dual.flatten(-2)@source[:owner.node_counts[0]*4].conj())
    energy=sum(v.abs().square().sum() for v in owner.source_upward(current))
    first,=torch.autograd.grad(energy,current,create_graph=True);tangent=torch.randn_like(current)
    second,=torch.autograd.grad(first,current,tangent)
    higher_error=_relative(second,2*(tangent@source.mT)@source.conj())
    # Complete actual plus/minus support vertices, including global body routes.
    for n in range(tree.node_count):
        edges=tree.permutation[owner.starts[n]:owner.stops[n]]
        groups=owner.basis._body_selection(edges) if isinstance(owner.basis,ScalarSystemBlocks) else ((None,None),)
        for bi,selected in groups:
            if bi is None:basis,local=owner.basis,edges
            else:
                body=owner.basis.system.bodies[bi];basis,local=body.basis,edges[selected]-body.edge_offset
            support=torch.cat((basis.mesh.triangles[basis.plus_faces[local]],basis.mesh.triangles[basis.minus_faces[local]]),1)
            assert bool(((support-owner.centers[n]).norm(dim=-1)<=owner.radii[n]).all())
    record=dict(kind=kind,depth=int(tree.depth.max()),four_map_errors=errors,gradients=gradients,dots=dots,
        unused_node_cotangent_error=partial_error,higher_current_derivative_error=higher_error,
        independent_all_node_quadrature=16,all_actual_support_vertices_enclosed=True,receipt=owner.receipt(),qualification=False)
    (tmp_path/'tree.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert max(*errors.values(),*gradients.values(),*dots.values(),partial_error,higher_error)<1e-11


def test_variable_carrier_unique_inventory_preflight_support_phase_node_mutations_and_chunk_bounds(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_moments as module
    owner=_owner('single');basis,tree=owner.basis,owner.tree
    args=dict(basis=basis,tree=tree,direction=owner.node_directions[0],wave_number=owner.wave_number,order=owner.order,
        quadrature_order=12,face_batch=4,node_directions=owner.node_directions,node_orders=owner.node_orders,phase_orders=owner.phase_orders)
    tensors=owner.tensors()
    assert len({t.untyped_storage().data_ptr() for t in tensors})==len(tensors)
    assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
    assert sum(t.untyped_storage().nbytes() for t in tensors)==owner.retained_bytes==owner.planned_retained_bytes
    def forbidden(*a,**kw):raise AssertionError('Rejected configuration reached leaf geometry or quadrature')
    with monkeypatch.context() as patch:
        patch.setattr(module,'PreparedRWGGeometry',forbidden);patch.setattr(module,'DirectionalRWGProjection',forbidden)
        for kw,exc in [({'retained_budget_bytes':owner.retained_bytes-1},MemoryError),
                ({'tensor_workspace_budget_bytes':owner.planned_tensor_workspace_bytes-1},MemoryError),
                ({'node_orders':(owner.order,)*tree.node_count},ValueError),({'quadrature_order':1},ValueError),
                ({'phase_orders':None},ValueError),({'node_directions':owner.node_directions*2},ValueError)]:
            altered=dict(args);altered.update(kw)
            with pytest.raises(exc):NestedDirectionalRWGProjection(**altered)
    with monkeypatch.context() as patch:
        patch.setattr(module,'DirectionalRWGProjection',forbidden)
        altered=dict(args);altered['tree']=replace(tree,lower=tree.lower+10,upper=tree.upper+10)
        with pytest.raises(ValueError,match='complete RWG supports'):NestedDirectionalRWGProjection(**altered)
        altered=dict(args);altered['phase_relative_budget']=1e-30
        with pytest.raises(ValueError,match='phase bound'):NestedDirectionalRWGProjection(**altered)
    value=torch.randn((3,5,tree.size),device='cuda',dtype=torch.complex128).transpose(0,1).requires_grad_()
    native=torch.isfinite;observed=[]
    def bounded(t):
        if t.is_complex() and t.ndim>=3:observed.append(t.shape[0]);assert t.shape[0]<=4
        return native(t)
    with monkeypatch.context() as patch:
        patch.setattr(torch,'isfinite',bounded)
        up=owner.source_upward(value);owner.receiver_downward(up)
        torch.autograd.grad(up[0],value,torch.ones_like(up[0]))
    assert observed and max(observed)==4
    empty=owner.receiver_upward(value[:0])
    assert all(v.shape==(0,3,c,4) for v,c in zip(empty,owner.node_counts))
    assert owner.source_downward(empty).shape==(0,3,tree.size)
    for field,new in [('node_counts',(1,)*tree.node_count),('node_orders',(0,)*tree.node_count),
            ('phase_orders',(0,)*tree.node_count),('phase_relative_budget',1e-10)]:
        previous=getattr(owner,field);setattr(owner,field,new)
        with pytest.raises(ValueError,match='changed'):owner.receipt()
        setattr(owner,field,previous);owner.assert_current()
    root=owner.source_upward(value)[0];owner.node_directions.add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(root,value,torch.ones_like(root))
    record=dict(exact_owned_bytes=owner.retained_bytes,all_unique_storages=True,maximum_finite_check_RHS=max(observed),
        node_configuration_saved_backward_guard=True,support_radius_checked=True,qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


@pytest.mark.parametrize('lossy',(False,True))
def test_nonroot_variable_carrier_tree_original_high_ka_weak_and_full_H(lossy,tmp_path):
    test,source=_basis((-.5,.1,0.),.02),_basis((.5,-.1,.05),.02)
    def make_tree(basis):
        tris=basis.mesh.triangles;support=torch.cat((tris[basis.plus_faces],tris[basis.minus_faces]),1)
        return build_support_octree(support.amin(1),support.amax(1),leaf_size=2)
    tt,st=make_tree(test),make_tree(source)
    a=next(i for i,row in enumerate(tt.children.tolist()) if i>0 and any(c>=0 for c in row))
    b=next(i for i,row in enumerate(st.children.tolist()) if i>0 and any(c>=0 for c in row))
    displacement=(tt.lower[a]+tt.upper[a]-st.lower[b]-st.upper[b])*.5
    direction=displacement/displacement.norm()
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01 if lossy else 0.,sigma_m=.02 if lossy else 0.),2.4e9)
    owners=[]
    for basis,tree,node in [(test,tt,a),(source,st,b)]:
        dirs,orders,phases=_schedule(tree,direction,root_order=6,phase_order=2,step=1e-5,anchor_depth=int(tree.depth[node]))
        dirs[int(tree.depth[node])==tree.depth]=direction
        owner=NestedDirectionalRWGProjection(basis,tree,dirs[0],medium.k,orders[0],quadrature_order=12,
            face_batch=4,node_directions=dirs,node_orders=orders,phase_orders=phases)
        assert owner.node_orders[node]==8 and torch.equal(owner.node_directions[node],direction)
        assert any(not torch.equal(t.direction,t.target_direction) for t in owner.transfers.values())
        owners.append(owner)
    target,origin=owners
    coupling=DirectionalFFTElectromagneticM2L(displacement,direction,medium,displacement.new_tensor(.02),8)
    op=_SelectedTreeEM(target,origin,a,b,coupling)
    actual=op.apply(torch.eye(2*len(op.cols),device='cuda',dtype=torch.complex128)).mT
    original,_=scalar_regional_block(test,source,op.rows,op.cols,(medium,),IntegrationConfig(order=12,singular_order=3))
    nr,nc=len(op.rows),len(op.cols);eta=abs(medium.eta)
    left=actual.real.new_tensor([eta]*nr+[1.]*nr);right=actual.real.new_tensor([1.]*nc+[eta]*nc)
    weak=_relative(left[:,None]*actual*right,left[:,None]*original*right)
    gen=torch.Generator(device='cuda').manual_seed(2292)
    x=torch.randn((3,5,2*nc),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    y=torch.randn((*x.shape[:-1],2*nr),device='cuda',dtype=torch.complex128,generator=gen)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):out=op.apply(x)
    assert not saved
    H=op.adjoint(y);gradient,=torch.autograd.grad(out,x,y)
    lhs,rhs=(out.conj()*y).sum(),(x.conj()*H).sum()
    dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    errors=dict(actual_H=_relative(H,y@actual.conj()),original_H=_relative(H,y@original.conj()),
        gradient=_relative(gradient,H),forward=_relative(out,x@actual.mT),dot=dot)
    record=dict(lossy=lossy,target_node=a,source_node=b,actual_ka=float(abs(medium.k)*displacement.norm()),
        selected_rows=op.rows.tolist(),selected_cols=op.cols.tolist(),original_balanced_weak_error=weak,
        errors=errors,target_receipt=target.receipt(),source_receipt=origin.receipt(),
        descendants_use_different_carriers=True,full_directional_system_qualified=False,qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert weak<1e-8 and errors['original_H']<1e-8
    assert max(errors[k] for k in ('actual_H','gradient','forward','dot'))<1e-11
