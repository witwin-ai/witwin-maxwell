"""Shared changing-carrier RWG DAG versus independent full-support maps."""
import json
import math
import mpmath as mp
import pytest
import torch
from witwin.maxwell.mom.fmm_directional_moments import DirectionalRWGDAGProjection,SparseDirectionalRWGProjection,NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L
from witwin.maxwell.mom.cluster import SupportOctree
from witwin.maxwell.mom.fmm_expansion import multi_indices
from witwin.maxwell.mom.direct_capacity import storage_bytes
from test_directional_sparse_contexts import _fixture as _physical_fixture,_independent_maps,_flatten,_unflatten,_relative

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _fixture(kind='single'):
    basis,tree,_,geometry,_=_physical_fixture('global' if kind=='mixed_leaf' else kind)
    if kind=='mixed_leaf':
        # One complete mixed-body physical child shared by two root contexts.
        n=tree.size;lower=tree.lower[0].repeat(2,1);upper=tree.upper[0].repeat(2,1)
        children=torch.full((2,8),-1,device='cuda',dtype=torch.int64);children[0,0]=1
        tree=SupportOctree(tree.permutation.clone(),torch.tensor([0,0],device='cuda'),torch.tensor([n,n],device='cuda'),
            lower,upper,children,torch.tensor([0,1],device='cuda'),n,21)
    carriers=tree.lower.new_tensor([[1.,.003,0.],[1.,-.004,.002],[1.,0.,0.]])
    carriers=carriers/carriers.norm(dim=-1,keepdim=True)
    links=[]
    for child in tree.children[0].tolist():
        if child>=0:links.extend((((0,0),(child,2),6),((0,1),(child,2),6)))
    for node,row in enumerate(tree.children.tolist()):
        if node:
            links.extend(((node,2),(child,2),0) for child in row if child>=0)
    return basis,tree,carriers,geometry,tuple(links)


def _build(fixture,**kwargs):
    basis,tree,carriers,geometry,links=fixture
    options=dict(quadrature_order=12,geometry=geometry,face_batch=4)
    options.update(kwargs)
    return DirectionalRWGDAGProjection(basis,tree,carriers,(3,3,9),((0,0),(0,1)),links,5.+.8j,**options)


@pytest.mark.parametrize('kind',('single','global','mixed_leaf'))
def test_shared_children_independent_four_maps_H_current_higher_and_mp_paths(kind,tmp_path):
    fixture=_fixture(kind);owner=_build(fixture)
    assert owner.receipt()['shared_child_contexts']>0
    assert len(owner.context_keys)==owner.tree.node_count+1<2*owner.tree.node_count
    assert len(owner.leaves)==int((owner.tree.children<0).all(-1).sum())
    assert storage_bytes(owner.tensors())==owner.retained_bytes
    assert len({id(t) for t in owner.transfers.values()})==len(owner.transfers)
    for name in ('_upward','_downward','_upward_dispatch','_downward_dispatch','source_upward','source_downward','receiver_upward','receiver_downward'):
        assert getattr(DirectionalRWGDAGProjection,name) is getattr(SparseDirectionalRWGProjection,name)
    source,receiver=_independent_maps(owner)
    scale=torch.cat([owner.radii[i].pow(torch.tensor([sum(a) for a in multi_indices(p)],device='cuda')).repeat_interleave(4)
        for i,p in enumerate(owner.node_orders)])
    gen=torch.Generator(device='cuda').manual_seed(659387)
    q=torch.randn((3,5,owner.tree.size),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    z=(torch.randn((3,5,len(scale)),device='cuda',dtype=torch.complex128,generator=gen)/scale).transpose(0,1).requires_grad_()
    reference=dict(source_upward=q@source.mT,receiver_upward=q@receiver.conj().mT,
        source_downward=z@source.conj(),receiver_downward=z@receiver)
    opposite=dict(source_upward='source_downward',receiver_upward='receiver_downward',
        source_downward='source_upward',receiver_downward='receiver_upward')
    errors,current,higher={},{},{}
    for name,wanted in reference.items():
        up=name.endswith('upward');x=q if up else z;dual=z if up else q
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
            raw=getattr(owner,name)(x if up else _unflatten(x,owner))
        value=_flatten(raw) if up else raw
        assert not saved
        errors[name]=_relative(value/scale,wanted/scale) if up else _relative(value,wanted)
        assert errors[name]<1e-11
        backward,=torch.autograd.grad(value,x,dual,create_graph=True)
        expected=getattr(owner,opposite[name])(_unflatten(dual,owner) if up else dual)
        expected=expected if up else _flatten(expected)
        current[name]=_relative(backward,expected);assert current[name]<1e-11
        tangent,=torch.autograd.grad(backward,dual,x)
        higher[name]=_relative(tangent,value);assert higher[name]<1e-11
    # Two independently supplied parent locals sum into the one shared child.
    roots=owner.request_indices
    slots=[None]*len(owner.node_counts)
    for root in roots:slots[root]=torch.ones((*q.shape[:-1],owner.node_counts[root],4),device='cuda',dtype=torch.complex128)
    summed=owner._downward_dispatch(slots,'source',q.shape[:-1],allow_missing=True)
    separate=[]
    for root in roots:
        alone=[None]*len(slots);alone[root]=slots[root]
        separate.append(owner._downward_dispatch(alone,'source',q.shape[:-1],allow_missing=True))
    assert _relative(summed,sum(separate))<1e-11
    derivative,=torch.autograd.grad(owner.source_upward(q)[roots[0]],q,slots[roots[0]])
    assert _relative(derivative,separate[0])<1e-11
    mp_errors=[]
    with mp.workdps(100):
        products=[mp.mpf(1)]*len(owner.node_counts)
        for parent in reversed(range(len(products))):
            values=[]
            for child in owner.children[parent]:
                shift=owner.transfers[parent,child]
                if shift.exact_phase:delta=mp.mpf(0)
                else:
                    a=[mp.mpf(v) for v in shift.direction.tolist()];b=[mp.mpf(v) for v in shift.target_direction.tolist()]
                    x=abs(mp.mpc(shift.wave_number))*mp.mpf(float(shift.child_radius))*mp.sqrt(sum((u-v)**2 for u,v in zip(a,b)))
                    delta=mp.exp(2*x)*x**(shift.phase_order+1)/mp.factorial(shift.phase_order+1)
                values.append((1+delta)*products[child])
            if values:products[parent]=max(values)
        for context,p in enumerate(products):
            receipt=owner.ideal_phase_path_bound(context);expected=p-1
            if expected==0:assert receipt['relative_bound']==0 and receipt['exact_phase_path']
            else:
                error=float(abs(mp.mpf(receipt['relative_bound'])/expected-1));mp_errors.append(error);assert error<1e-11
    record=dict(kind=kind,four_map_errors=errors,native_current=current,native_higher=higher,
        mp_relative_errors=mp_errors,mp_digits=100,mp_absolute_tolerance=0,noncontiguous_rhs=15,
        summed_shared_parent_locals=True,missing_cotangents=True,saved_tensors=0,receipt=owner.receipt(),qualification=False)
    (tmp_path/'dag.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_dag_preflight_topology_phase_native_modes_and_old_admission_rejection(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_moments as module
    fixture=_fixture();owner=_build(fixture)
    def forbidden(*a,**k):raise AssertionError('Rejected DAG admission reached layout/maps')
    with monkeypatch.context() as patch:
        patch.setattr(module,'SeparableMultipoleShift',forbidden)
        for option in (dict(retained_budget_bytes=owner.retained_bytes-1),dict(tensor_workspace_budget_bytes=1)):
            with pytest.raises(MemoryError):_build(fixture,**option)
        with pytest.raises(ValueError):_build((*fixture[:-1],fixture[-1][1:]))
        with pytest.raises(ValueError):_build((*fixture[:-1],fixture[-1]+(fixture[-1][0],)))
        bad=list(fixture[-1]);parent,child,L=bad[0];bad[0]=(parent,child,7)
        with pytest.raises(ValueError):_build((*fixture[:-1],tuple(bad)))
        with pytest.raises(ValueError,match='phase bound'):_build(fixture,phase_relative_budget=1e-30)
    x=torch.ones((2,owner.tree.size),device='cuda',dtype=torch.complex128,requires_grad=True)
    out=owner.source_upward(x)[owner.request_indices[0]]
    owner.transfers[next(iter(owner.transfers))].source_phase.requires_grad_()
    with pytest.raises(ValueError):torch.autograd.grad(out.real.sum(),x)
    owner.transfers[next(iter(owner.transfers))].source_phase.requires_grad_(False)
    owner.assert_current()
    assert not isinstance(owner,(SparseDirectionalRWGProjection,NestedDirectionalRWGProjection))
    with pytest.raises(TypeError,match='sparse physical'):
        DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,0,owner,1,None,owner.centers.new_tensor(.01),
            kernel_error_budget=__import__('witwin.maxwell.mom.interpolation_error',fromlist=['KernelErrorBudget']).KernelErrorBudget(1.,1.),
            ideal_rwg_error_budget=1.,balancing_impedance=1.)
    (tmp_path/'guards.json').write_text(json.dumps(dict(capacity=True,physical_child_coverage=True,
        duplicate_link=True,child_order=True,phase_preflight=True,saved_native_mode=True,
        distinct_phase_protocol=True,qualification=False),indent=2)+'\n')
