"""Exact shared E/H induced actions, without rebuilding electromagnetic blocks."""
import copy
from .coupled_volume_action import CoupledVolumeAction
from .surface_h_snapshot import encode_inverse,restore_inverse
from . import volume_h_snapshot as h_snapshot
from . import volume_h2_snapshot as h2_snapshot
from .volume_h2 import VolumeNestedH2Action
from .surface_volume_hmatrix import SurfaceVolumeBlocks
from .interior_volume_hmatrix import InteriorVolumeBlocks
from .volume_hmatrix import VolumeCellBlocks
from .volume_mesh import bind_volume_region
from .nested_h2_snapshot import _keys
from .runtime import _same_metadata
from .volume_h_snapshot import _digest
from .tensor_medium import TensorMedium


def _pair(electric,magnetic):
    if any(type(a) is not CoupledVolumeAction for a in (electric,magnetic)):
        raise TypeError('Snapshot requires exact coupled E/H actions')
    for action in (electric,magnetic):action.assert_current()
    if (electric.field!='E' or magnetic.field!='H' or electric.inverse is not magnetic.inverse
            or len(electric.sources)!=len(magnetic.sources)
            or any(a is not b for a,b in zip(electric.sources,magnetic.sources))):
        raise ValueError('Coupled snapshot requires shared inverse and ordered source actions')


def encode(electric,magnetic):
    _pair(electric,magnetic)
    tensor=any(s.provider.tensor_kernel is not None for s in electric.sources)
    regions=[]
    for source in electric.sources:
        p=source.provider;v=p.binding.volume
        regions.append({'region':v.region_id,'mesh_digest':_digest((v.vertices,v.cells)),
            'order':p.order,'inner_order':p.inner_order,'source':h_snapshot.encode(source)})
        if tensor:regions[-1]['tensor_policy']=p.tensor_policy()
    channels=[]
    for action in (electric,magnetic):
        channels.append({'field':action.field,'receipt':copy.deepcopy(action.receipt),
            'regions':[{'order':f.provider.order,'surface':h_snapshot.encode(f),
                'direct':(h2_snapshot if type(d) is VolumeNestedH2Action else h_snapshot).encode(d)}
                for f,d in zip(action.fields,action.direct)]})
    if tensor:
        for channel,action in zip(channels,(electric,magnetic)):
            for record,direct in zip(channel['regions'],action.direct):
                p=direct.provider
                record['tensor_policy']=(None if p.tensor_kernel is None else dict(
                    radial_order=p.radial_order,face_rule=p.face_rule,kernel_batch=p.kernel_batch))
    payload={'version':('exact_tensor_coupled_eh_response_v2' if tensor else 'exact_coupled_eh_response_v1'),
        'inverse':encode_inverse(electric.inverse),
        'regions':regions,'channels':channels}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,provider,volumes,*,sources=()):
    """Use supplied ordered meshes and physical sources; saved receipts are historical.

    Component retained budgets apply independently. No aggregate capacity or
    renewed quadrature, residual or physical qualification is implied.
    """
    _keys(payload,('version','inverse','regions','channels','sha256'))
    if (payload['version'] not in ('exact_coupled_eh_response_v1','exact_tensor_coupled_eh_response_v2')
            or payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'})):
        raise ValueError('Coupled snapshot checksum/version mismatch')
    volumes=tuple(volumes)
    if not volumes or len(volumes)!=len(payload['regions']) or len(payload['channels'])!=2:
        raise ValueError('Coupled snapshot region/channel count mismatch')
    tensor=payload['version']=='exact_tensor_coupled_eh_response_v2'
    extra=('tensor_policy',) if tensor else ()
    bindings=[]
    for volume,record in zip(volumes,payload['regions']):
        _keys(record,('region','mesh_digest','order','inner_order','source')+extra)
        if volume.region_id!=record['region'] or _digest((volume.vertices,volume.cells))!=record['mesh_digest']:
            raise ValueError('Coupled snapshot ordered volume binding mismatch')
        bindings.append(bind_volume_region(volume,provider.system))
    inverse=restore_inverse(payload['inverse'],provider,sources=sources)
    kernels=[]
    for binding in bindings:
        region=binding.volume.region_id
        if isinstance(provider.system.bodies[region-1].interior,TensorMedium):
            owned=getattr(provider,'interior_kernels',())
            if len(owned)!=len(provider.system.bodies) or owned[region-1] is None:
                raise ValueError('Coupled tensor snapshot requires the surface regional kernel')
            kernels.append(owned[region-1])
        else:kernels.append(None)
    if tensor!=any(k is not None for k in kernels):
        raise ValueError('Coupled snapshot tensor region/version mismatch')
    source_actions=[]
    for b,r,kernel in zip(bindings,payload['regions'],kernels):
        policy=r.get('tensor_policy');options={}
        if kernel is not None:
            _keys(policy,('face_rule','kernel_batch','dyadic_integration','cache_max_bytes'))
            options=dict(tensor_kernel=kernel,tensor_face_rule=policy['face_rule'],
                         tensor_kernel_batch=policy['kernel_batch'])
        elif policy is not None:raise ValueError('Scalar coupled source cannot own tensor policy')
        p=SurfaceVolumeBlocks(b,order=r['order'],inner_order=r['inner_order'],**options)
        if not _same_metadata(p.tensor_policy(),policy):
            raise ValueError('Coupled snapshot source tensor policy mismatch')
        source_actions.append(h_snapshot.restore(r['source'],p))
    actions=[]
    for field,channel in zip(('E','H'),payload['channels']):
        _keys(channel,('field','receipt','regions'))
        if channel['field']!=field or len(channel['regions'])!=len(bindings):
            raise ValueError('Coupled snapshot channel layout mismatch')
        decoders=[];direct=[]
        for binding,record,kernel in zip(bindings,channel['regions'],kernels):
            _keys(record,('order','surface','direct')+extra)
            decoders.append(h_snapshot.restore(record['surface'],InteriorVolumeBlocks(binding,field=field,order=record['order'],tensor_kernel=kernel)))
            body=provider.system.bodies[binding.volume.region_id-1]
            policy=record.get('tensor_policy');options={}
            if kernel is not None:
                _keys(policy,('radial_order','face_rule','kernel_batch'))
                options=dict(tensor_kernel=kernel,**policy)
            elif policy is not None:raise ValueError('Scalar coupled volume cannot own tensor policy')
            p=VolumeCellBlocks(binding.volume,body.interior,field=field,order=record['order'],**options)
            codec=h2_snapshot if record['direct'].get('version')=='exact_volume_h2_v1' else h_snapshot
            direct.append(codec.restore(record['direct'],p))
        action=CoupledVolumeAction(inverse,source_actions,decoders,direct)
        saved=channel['receipt']
        key='component_referenced_bytes'
        if (not isinstance(saved,dict) or set(saved)!=set(action.receipt)
                or type(saved.get(key)) is not int or saved[key]<0
                or not _same_metadata({k:v for k,v in action.receipt.items() if k!=key},
                    {k:v for k,v in saved.items() if k!=key})):
            raise ValueError('Coupled snapshot receipt differs from reconstructed components')
        action.restored_component_referenced_bytes=action.receipt[key]
        action.receipt=copy.deepcopy(saved)
        actions.append(action)
    _pair(*actions)
    return tuple(actions)
