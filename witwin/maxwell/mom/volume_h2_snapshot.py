"""Exact regional volume H2 persistence without source H reconstruction."""
import copy
from dataclasses import asdict,replace
import torch
from .volume_h2 import VolumeNestedH2Action
from .volume_hmatrix import VolumeCellBlocks
from .h2 import H2Config
from .cluster import build_support_octree
from .fmm_capacity import storage_bytes
from .volume_h_snapshot import _digest,_physics
from .nested_h2_snapshot import encode_nested,restore_nested,_keys


def encode(action):
    if type(action) is not VolumeNestedH2Action:raise TypeError('Expected exact native volume H2 action')
    action.assert_current();tree=action.nested.tree
    payload={'version':'exact_volume_h2_v1','configuration':asdict(action.config),
        'provider_digest':_digest((_physics(action.provider),tuple(action.provider.tensors()))),
        'tree':{'leaf_size':tree.leaf_size,'morton_bits':tree.morton_bits,'digest':_digest(tree.tensors())},
        'nested':encode_nested(action.nested),'receipt':copy.deepcopy(action.receipt)}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,provider):
    """Restore saved bases/transfers/couplings/near arrays on matching geometry.

    Checksums are integrity checks, not authenticity or physical qualification.
    Retained arrays are preflight counted after geometric tree reconstruction;
    transient uploads/near repacking and allocator peaks are excluded.
    """
    _keys(payload,('version','configuration','provider_digest','tree','nested','receipt','sha256'))
    if payload['version']!='exact_volume_h2_v1':raise ValueError('Unknown volume H2 snapshot version')
    if payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'}):
        raise ValueError('Volume H2 snapshot checksum mismatch')
    if type(provider) is not VolumeCellBlocks:raise TypeError('Volume H2 snapshot requires native volume blocks')
    provider.assert_current();config=H2Config(**payload['configuration'])
    if payload['provider_digest']!=_digest((_physics(provider),tuple(provider.tensors()))):
        raise ValueError('Volume H2 snapshot physical binding mismatch')
    raw_tree=payload['tree'];_keys(raw_tree,('leaf_size','morton_bits','digest'))
    tree=build_support_octree(provider.lower,provider.upper,leaf_size=raw_tree['leaf_size'],
        morton_bits=raw_tree['morton_bits'],partition_points=provider.points)
    if _digest(tree.tensors())!=raw_tree['digest']:raise ValueError('Volume H2 snapshot tree mismatch')
    fixed=storage_bytes((*provider.tensors(),*tree.tensors()))
    def array_bytes(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(array_bytes(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(array_bytes(v) for v in value)
        return 0
    raw=payload['nested'];near=raw['near'];near_bytes=0
    if near is not None:
        records=near['records']
        near_bytes=array_bytes(records)+8*(2*len({r[0] for r in records})+1+3*len(records))
    owned=array_bytes({k:v for k,v in raw.items() if k not in ('near','receipt')})
    if fixed+near_bytes+owned>config.retained_budget_bytes:
        raise MemoryError('Volume H2 snapshot exceeds retained budget before factor loading')
    nested_config=replace(config,retained_budget_bytes=config.retained_budget_bytes-fixed-near_bytes)
    nested=restore_nested(raw,tree,provider.mesh.volumes.sqrt().repeat(3),
        provider.mesh.volumes.rsqrt().repeat(3),expected_config=nested_config,components=3,dtype=provider.dtype)
    action=object.__new__(VolumeNestedH2Action)
    action.provider,action.nested,action.config=provider,nested,config
    action.device,action.dtype,action.shape=provider.device,provider.dtype,nested.shape
    action._binding=(id(provider),id(nested),id(config),action.device,action.dtype,action.shape)
    action.receipt=copy.deepcopy(payload['receipt'])
    measured=storage_bytes(action.tensors())
    if measured>config.retained_budget_bytes:raise MemoryError('Volume H2 snapshot retained measurement exceeds budget')
    receipt=action.receipt
    if (not isinstance(receipt,dict) or receipt.get('version')!='regional_volume_nested_h2_v1'
            or receipt.get('retained_bytes')!=measured or receipt.get('retained_budget_bytes')!=config.retained_budget_bytes
            or receipt.get('fixed_geometry_tree_bytes')!=fixed or receipt.get('packed_near_bytes')!=near_bytes
            or receipt.get('nested')!=nested.receipt()
            or any(receipt.get(k) is not False for k in ('physical_qualification','global_operator_error_verified',
                'setup_complexity_qualified','source_h_object_retained','source_h_far_factors_retained'))):
        raise ValueError('Volume H2 snapshot receipt mismatch')
    action.assert_current()
    return action
