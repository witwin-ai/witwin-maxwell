"""Exact H cross/volume block snapshots; restoration never samples kernels."""
import copy
import hashlib
import json
from dataclasses import asdict
import torch
from .cluster import build_support_octree
from .hierarchical_blocks import HierarchicalBlockAction
from .lowrank import LowRankBlock
from .fmm_capacity import storage_bytes
from .volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from .surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from .interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig

_TYPES={
    'volume':(VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig),
    'source':(SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig),
    'interior':(InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig)}


def _digest(value):
    digest=hashlib.sha256()
    def visit(v):
        digest.update(type(v).__name__.encode()+b':')
        if isinstance(v,torch.Tensor):
            digest.update(str((str(v.dtype),tuple(v.shape))).encode())
            digest.update(v.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(v,dict):
            digest.update(str(len(v)).encode()+b':')
            for key in sorted(v):visit(key);visit(v[key])
        elif isinstance(v,(list,tuple)):
            digest.update(str(len(v)).encode())
            for item in v:visit(item)
        elif isinstance(v,complex):visit((v.real,v.imag))
        else:
            encoded=json.dumps(v,sort_keys=True,allow_nan=False).encode()
            digest.update(str(len(encoded)).encode()+b':'+encoded)
    visit(value)
    return digest.hexdigest()


def _physics(provider):
    if type(provider) is VolumeCellBlocks:
        if provider.tensor_kernel is not None:
            from .tensor_medium import sampled_medium_payload
            return {'medium':sampled_medium_payload(provider.medium),'field':provider.field,
                'order':provider.order,'region':provider.mesh.region_id,
                'radial_order':provider.radial_order,'face_rule':provider.face_rule,
                'kernel_batch':provider.kernel_batch,
                'dyadic_integration':provider.tensor_kernel.integration.payload(),
                'cache_max_bytes':provider.tensor_kernel.cache_max_bytes}
        return {'medium':asdict(provider.medium),'field':provider.field,'order':provider.order,
                'region':provider.mesh.region_id}
    binding=provider.binding if type(provider) is SurfaceVolumeBlocks else provider.geometry.binding
    result={'region':binding.volume.region_id,'order':provider.order,
            'interiors':[]}
    from .tensor_medium import TensorMedium,sampled_medium_payload
    if any(isinstance(b.interior,TensorMedium) for b in binding.surface_system.bodies):
        result['interiors']=[sampled_medium_payload(b.interior) for b in binding.surface_system.bodies]
    else:result['interiors']=[asdict(b.interior) for b in binding.surface_system.bodies]
    geometry=provider if type(provider) is SurfaceVolumeBlocks else provider.geometry
    if geometry.tensor_kernel is not None:result['tensor_integration']=geometry.tensor_policy()
    result.update({'inner_order':provider.inner_order} if type(provider) is SurfaceVolumeBlocks else {'field':provider.field})
    return result


def _geometry(provider,config):
    if type(provider) is VolumeCellBlocks:
        test=build_support_octree(provider.lower,provider.upper,leaf_size=config.leaf_size,partition_points=provider.points)
        return test,test,3,3,provider.mesh.volumes.sqrt().repeat(3),provider.mesh.volumes.rsqrt().repeat(3)
    source_kind=type(provider) is SurfaceVolumeBlocks
    test=build_support_octree(provider.row_lower,provider.row_upper,leaf_size=config.leaf_size,
        partition_points=None if source_kind else provider.points)
    source=build_support_octree(provider.column_lower,provider.column_upper,leaf_size=config.leaf_size,
        partition_points=provider.column_points if source_kind else None)
    return test,source,2 if source_kind else 3,3 if source_kind else 2,provider.row_scale,provider.column_scale


def _coverage(rectangles,height,width):
    """Sweep integer rectangles with range-add min/max; no dense pair grid."""
    events={0:[],height:[]}
    for a,b,c,d in rectangles:
        events.setdefault(a,[]).append((c,d,1));events.setdefault(b,[]).append((c,d,-1))
    lo=[0]*(4*width);hi=lo.copy();lazy=lo.copy()
    def add(node,left,right,start,stop,value):
        if start<=left and right<=stop:
            lo[node]+=value;hi[node]+=value;lazy[node]+=value;return
        mid=(left+right)//2
        if start<mid:add(node*2,left,mid,start,stop,value)
        if stop>mid:add(node*2+1,mid,right,start,stop,value)
        lo[node]=lazy[node]+min(lo[node*2],lo[node*2+1])
        hi[node]=lazy[node]+max(hi[node*2],hi[node*2+1])
    for point in sorted(events):
        for start,stop,value in events[point]:add(1,0,width,start,stop,value)
        expected=0 if point==height else 1
        if lo[1]!=expected or hi[1]!=expected:raise ValueError('Snapshot blocks do not cover each entry exactly once')


def encode(action):
    kinds=[k for k,(_,cls,_) in _TYPES.items() if type(action) is cls]
    if not kinds:raise TypeError('Snapshot requires an exact native volume or cross H action')
    action.assert_current()
    blocks=[]
    for (rows,columns,value),nodes in zip(action.blocks,action.block_nodes):
        low=isinstance(value,LowRankBlock)
        blocks.append({'nodes':nodes,'rows':rows.detach().cpu().clone(),'columns':columns.detach().cpu().clone(),
            'factors':tuple(t.detach().cpu().clone() for t in ((value.left,value.right) if low else (value,))),
            'receipt':copy.deepcopy(value.receipt) if low else None})
    payload={'version':'exact_volume_cross_h_blocks_v1','kind':kinds[0],
        'configuration':asdict(action.config),'dtype':str(action.dtype),
        'provider_digest':_digest((_physics(action.provider),tuple(action.provider.tensors()))),
        'tree_digest':_digest((action.test_tree.tensors(),action.source_tree.tensors())),
        'scaling_digest':_digest((action.row_scale,action.column_scale)),
        'blocks':blocks,'receipt':copy.deepcopy(action.receipt)}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,provider):
    """Bind exact stored factors to independently constructed physical geometry.

    Checksums detect accidental corruption, not authenticity. This does not
    requalify the stored compression/physics claim or re-evaluate kernels.
    Geometry trees are rebuilt deterministically and matched before factors load.
    Retained limits are checked before factor loading; geometry construction,
    index temporaries and allocator peak memory are not preflight bounded here.
    """
    keys={'version','kind','configuration','dtype','provider_digest','tree_digest','scaling_digest','blocks','receipt','sha256'}
    if (not isinstance(payload,dict) or set(payload)!=keys or payload['version']!='exact_volume_cross_h_blocks_v1'
            or payload['kind'] not in _TYPES):raise ValueError('Invalid H block snapshot schema')
    if payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'}):
        raise ValueError('H block snapshot checksum mismatch')
    provider_type,cls,config_type=_TYPES[payload['kind']]
    if type(provider) is not provider_type:raise ValueError('H block snapshot provider kind mismatch')
    provider.assert_current();config=config_type(**payload['configuration'])
    if (payload['dtype']!=str(provider.dtype) or
            payload['provider_digest']!=_digest((_physics(provider),tuple(provider.tensors())))):
        raise ValueError('H block snapshot physical binding mismatch')
    test,source,tc,sc,rs,cs=_geometry(provider,config)
    if (payload['tree_digest']!=_digest((test.tensors(),source.tensors())) or
            payload['scaling_digest']!=_digest((rs,cs))):raise ValueError('H block snapshot geometry/scaling mismatch')
    action=object.__new__(cls)
    action.provider,action.config=provider,config
    action.test_tree,action.source_tree=test,source
    action.test_components,action.source_components=tc,sc
    action.device,action.dtype=provider.device,provider.dtype
    action.shape=(tc*test.size,sc*source.size)
    action.row_scale,action.column_scale=rs.clone(),cs.clone()
    retained=storage_bytes((*provider.tensors(),*test.tensors(),*source.tensors(),action.row_scale,action.column_scale))
    blocks=[];nodes=[];rectangles=[]
    starts=[t.starts.tolist() for t in (test,source)];stops=[t.stops.tolist() for t in (test,source)]
    def array(value,shape,dtype):
        if (not isinstance(value,torch.Tensor) or value.dtype!=dtype or tuple(value.shape)!=tuple(shape)
                or not bool(torch.isfinite(value).all())):raise ValueError('Invalid stored H block array')
        return value.to(provider.device).clone()
    if not isinstance(payload['blocks'],list) or not payload['blocks']:raise ValueError('Missing H snapshot blocks')
    for record in payload['blocks']:
        if not isinstance(record,dict) or set(record)!={'nodes','rows','columns','factors','receipt'}:
            raise ValueError('Invalid H block record')
        pair=record['nodes']
        if (not isinstance(pair,(tuple,list)) or len(pair)!=2 or any(type(n) is not int or not 0<=n<t.node_count for n,t in zip(pair,(test,source)))):
            raise ValueError('Invalid H block nodes')
        a,b=pair;indices=[]
        for side,(tree,components,node) in enumerate(((test,tc,a),(source,sc,b))):
            ids=tree.permutation[starts[side][node]:stops[side][node]]
            indices.append(torch.cat(tuple(ids+k*tree.size for k in range(components))))
        rows,columns=indices
        for name,ids in zip(('rows','columns'),indices):
            stored=array(record[name],ids.shape,torch.int64)
            if not torch.equal(stored,ids):raise ValueError('H block index layout mismatch')
        factors=record['factors'];m,n=len(rows),len(columns)
        if not isinstance(factors,(tuple,list)) or len(factors) not in (1,2):raise ValueError('Invalid H block factors')
        if len(factors)==1:
            if record['receipt'] is not None or m>tc*config.leaf_size or n>sc*config.leaf_size:
                raise ValueError('Invalid exact H leaf')
            shapes=((m,n),)
        else:
            left=factors[0]
            if not isinstance(left,torch.Tensor) or left.ndim!=2:raise ValueError('Invalid H left factor')
            rank=left.shape[1]
            if not 0<=rank<=min(m,n,config.max_rank) or not isinstance(record['receipt'],dict):
                raise ValueError('Invalid H compression rank or receipt')
            shapes=((m,rank),(rank,n))
        retained+=sum(t.numel()*t.element_size() for t in indices)+sum(torch.empty((),dtype=provider.dtype).element_size()*s[0]*s[1] for s in shapes)
        if retained>config.retained_budget_bytes:raise MemoryError('H snapshot exceeds retained budget')
        values=[array(v,s,provider.dtype) for v,s in zip(factors,shapes)]
        value=values[0] if len(values)==1 else LowRankBlock(*values,copy.deepcopy(record['receipt']))
        blocks.append((rows,columns,value));nodes.append((a,b))
        rectangles.append((starts[0][a],stops[0][a],starts[1][b],stops[1][b]))
    _coverage(rectangles,test.size,source.size)
    action.blocks,action.block_nodes=tuple(blocks),tuple(nodes)
    action._binding=(id(provider),id(config),id(test),id(source),tc,sc,action.device,action.dtype,action.shape,action.block_nodes)
    action._layout=tuple((id(r),id(c),id(v)) for r,c,v in action.blocks)
    action._versions=tuple((id(t),t._version) for t in action.tensors())
    action.receipt=copy.deepcopy(payload['receipt'])
    measured=storage_bytes(action.tensors())
    if measured>config.retained_budget_bytes:raise MemoryError('H snapshot retained measurement exceeds budget')
    # Geometry aliasing may differ after independent reconstruction. The saved
    # receipt describes historical storage; actual capacity is enforced above.
    action.restored_retained_bytes=measured
    historical=action.receipt.get('retained_bytes') if isinstance(action.receipt,dict) else None
    if (not isinstance(action.receipt,dict) or action.receipt.get('physical_qualification') is not False
            or action.receipt.get('configuration')!=asdict(config)
            or type(historical) is not int or not 0<=historical<=config.retained_budget_bytes
            or action.receipt.get('far_blocks')!=sum(isinstance(v,LowRankBlock) for _,_,v in blocks)
            or action.receipt.get('exact_blocks')!=sum(not isinstance(v,LowRankBlock) for _,_,v in blocks)):
        raise ValueError('H snapshot receipt does not match stored blocks')
    action.assert_current()
    return action
