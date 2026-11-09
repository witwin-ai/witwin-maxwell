"""Exact nested-basis/action storage without algebraic reconversion."""
import copy
from dataclasses import asdict
import torch
from .h2_basis import NestedClusterBasis,NestedBasisConfig
from .h2 import NestedH2Actions,H2Config
from .packed_near import PackedNearBlocks
from .volume_h_snapshot import _coverage


def _cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    if isinstance(value,list):return [_cpu(v) for v in value]
    return copy.deepcopy(value)


def _keys(value,expected):
    if not isinstance(value,dict) or set(value)!=set(expected):raise ValueError('Invalid nested snapshot fields')


def _array(value,shape,dtype,device):
    if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(shape) or value.dtype!=dtype
            or not bool(torch.isfinite(value).all())):raise ValueError('Invalid nested snapshot array')
    return value.to(device).clone()


def _indices(tree,node,components):
    ids=tree.permutation[int(tree.starts[node]):int(tree.stops[node])]
    return torch.cat(tuple(ids+k*tree.size for k in range(components)))


def _basis_encode(basis):
    basis.assert_current()
    return _cpu({'configuration':asdict(basis.config),'ranks':basis.ranks,
        'leaf_bases':basis.leaf_bases,'leaf_indices':basis.leaf_indices,'transfers':basis.transfers,
        'seed_errors':basis.seed_errors})


def _basis_restore(payload,tree,components,dtype,config):
    _keys(payload,('configuration','ranks','leaf_bases','leaf_indices','transfers','seed_errors'))
    if payload['configuration']!=asdict(config):raise ValueError('Nested basis configuration mismatch')
    count=tree.node_count
    for name in ('ranks','leaf_bases','leaf_indices','transfers'):
        if not isinstance(payload[name],(tuple,list)) or len(payload[name])!=count:
            raise ValueError('Nested basis node count mismatch')
    ranks=tuple(payload['ranks'])
    starts,stops=tuple(tree.starts.tolist()),tuple(tree.stops.tolist())
    children=tuple(tuple(c for c in row if c>=0) for row in tree.children.tolist())
    if any(type(r) is not int or not 0<=r<=min(config.max_rank,components*(b-a)) for r,a,b in zip(ranks,starts,stops)):
        raise ValueError('Invalid stored nested rank')
    parents={child:parent for parent,cs in enumerate(children) for child in cs}
    leaves=[];indices=[];transfers=[]
    for node in range(count):
        q,ids,e=(payload[k][node] for k in ('leaf_bases','leaf_indices','transfers'))
        if children[node]:
            if q is not None or ids is not None or ranks[node]>sum(ranks[c] for c in children[node]):
                raise ValueError('Invalid internal nested basis layout')
            leaves.append(None);indices.append(None)
        else:
            leaves.append(_array(q,(components*(stops[node]-starts[node]),ranks[node]),dtype,tree.device))
            expected=_indices(tree,node,components)
            actual=_array(ids,expected.shape,torch.int64,tree.device)
            if not torch.equal(actual,expected):raise ValueError('Nested leaf index mismatch')
            indices.append(actual)
        if node==0:
            if e is not None:raise ValueError('Nested root cannot have a transfer')
            transfers.append(None)
        else:transfers.append(_array(e,(ranks[node],ranks[parents[node]]),dtype,tree.device))
    basis=object.__new__(NestedClusterBasis)
    basis.tree,basis.config,basis.components=tree,config,components
    basis.dtype,basis.device=dtype,tree.device
    basis.starts,basis.stops,basis.children,basis.ranks=starts,stops,children,ranks
    basis.leaf_bases,basis.leaf_indices,basis.transfers=tuple(leaves),tuple(indices),tuple(transfers)
    basis.seed_errors=copy.deepcopy(payload['seed_errors'])
    basis._binding=(id(tree),config,components,dtype,tree.device,starts,stops,children,ranks)
    basis._versions=tuple((id(t),t._version) for t in basis.tensors())
    basis.assert_current()
    return basis


def _encode_near(packed):
    near=None
    if packed is not None:
        records=[]
        for a,b,gi,bi in sorted(packed.entries,key=lambda r:r[2:]):
            rows,columns,values=packed.groups[gi]
            records.append((a,b,rows[bi],columns[bi],values[bi]))
        near={'action_batch':packed.action_batch,'records':records}
    return _cpu(near)


def encode_nested(action):
    if type(action) is not NestedH2Actions:raise TypeError('Expected exact native nested H2 actions')
    action.refresh()
    near=_encode_near(action.near)
    return _cpu({'version':'exact_nested_h2_actions_v1','configuration':asdict(action.config),
        'components':action.components,'dtype':str(action.dtype),'row_scale':action.row_scale,
        'column_scale':action.column_scale,'test_basis':_basis_encode(action.test_basis),
        'trial_basis':_basis_encode(action.trial_basis),'couplings':action.couplings,
        'conversion_errors':action.conversion_errors,'capacity_plan':action.capacity_plan,'near':near,
        'receipt':action.receipt()})


def _restore_near(raw,tree,components,dtype,rectangles):
    near=None
    if raw is not None:
        _keys(raw,('action_batch','records'))
        records=[]
        for record in raw['records']:
            if not isinstance(record,(tuple,list)) or len(record)!=5:raise ValueError('Invalid nested near record')
            a,b,rows,cols,value=record
            if any(type(n) is not int or not 0<=n<tree.node_count for n in (a,b)):
                raise ValueError('Nested near node outside tree')
            rectangles.append((int(tree.starts[a]),int(tree.stops[a]),int(tree.starts[b]),int(tree.stops[b])))
            rr,cc=_indices(tree,a,components),_indices(tree,b,components)
            for actual,expected in ((rows,rr),(cols,cc)):
                if not torch.equal(_array(actual,expected.shape,torch.int64,tree.device),expected):
                    raise ValueError('Nested near index mismatch')
            records.append((a,b,rr,cc,_array(value,(len(rr),len(cc)),dtype,tree.device)))
        near=PackedNearBlocks(records,components*tree.size,action_batch=raw['action_batch'])
    return near


def restore_nested(payload,tree,row_scale,column_scale,*,expected_config,components,dtype):
    """Restore exact arrays; caller must verify the enclosing snapshot digest.

    This validates structure and retained array budgets, not compression error.
    Geometry, temporary near repacking and allocator peaks are separate costs.
    """
    _keys(payload,('version','configuration','components','dtype','row_scale','column_scale','test_basis',
        'trial_basis','couplings','conversion_errors','capacity_plan','near','receipt'))
    if (payload['version']!='exact_nested_h2_actions_v1' or payload['configuration']!=asdict(expected_config)
            or type(payload['components']) is not int or payload['components']!=components
            or payload['dtype']!=str(dtype)):raise ValueError('Nested snapshot configuration mismatch')
    tree.assert_current();config=H2Config(**payload['configuration'])
    def bytes_in(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(bytes_in(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(bytes_in(v) for v in value)
        return 0
    # Near storage is shared and counted by the enclosing physical snapshot;
    # this budget covers the nested object's owned bases/scales/couplings.
    owned={k:v for k,v in payload.items() if k not in ('near','receipt')}
    if bytes_in(owned)>config.retained_budget_bytes:raise MemoryError('Nested snapshot exceeds owned retained budget')
    shape=(components*tree.size,)*2
    scales=[]
    for name,expected in (('row_scale',row_scale),('column_scale',column_scale)):
        value=_array(payload[name],shape[:1],tree.lower.dtype,tree.device)
        if not torch.equal(value,expected) or not bool((value>0).all()):raise ValueError('Nested snapshot scaling mismatch')
        scales.append(value)
    basis_config=NestedBasisConfig(config.max_rank,config.tolerance/8)
    test=_basis_restore(payload['test_basis'],tree,components,dtype,basis_config)
    trial=_basis_restore(payload['trial_basis'],tree,components,dtype,basis_config)
    rectangles=[]
    def pair(a,b):
        if any(type(n) is not int or not 0<=n<tree.node_count for n in (a,b)):
            raise ValueError('Nested snapshot node outside tree')
        rectangles.append((int(tree.starts[a]),int(tree.stops[a]),int(tree.starts[b]),int(tree.stops[b])))
    couplings=[]
    for record in payload['couplings']:
        if not isinstance(record,(tuple,list)) or len(record)!=3:raise ValueError('Invalid nested coupling record')
        a,b,value=record;pair(a,b)
        couplings.append((a,b,_array(value,(test.ranks[a],trial.ranks[b]),dtype,tree.device)))
    near=_restore_near(payload['near'],tree,components,dtype,rectangles)
    _coverage(rectangles,tree.size,tree.size)
    action=object.__new__(NestedH2Actions)
    action.tree,action.config,action.near=tree,config,near
    action.components,action.device,action.dtype,action.shape=components,tree.device,dtype,shape
    action.row_scale,action.column_scale=scales
    action.test_basis,action.trial_basis=test,trial
    action.couplings=tuple(couplings)
    action.capacity_plan=copy.deepcopy(payload['capacity_plan'])
    action.conversion_errors=copy.deepcopy(payload['conversion_errors'])
    action._binding=(id(tree),id(near),id(test),id(trial),config,components,tree.device,dtype,shape)
    action._layout=tuple((a,b,id(c)) for a,b,c in action.couplings)
    action._versions=tuple((id(t),t._version) for t in action.tensors())
    if action.receipt()!=payload['receipt']:raise ValueError('Nested snapshot receipt mismatch')
    return action
