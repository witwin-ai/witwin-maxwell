"""Nested regional volume H2 actions converted from verified H blocks."""
from copy import deepcopy
from dataclasses import replace
from .volume_hmatrix import VolumeHierarchicalAction
from .lowrank import LowRankBlock
from .h2 import NestedH2Actions,H2Config
from .h2_capacity import plan_h2_storage,H2CapacityError
from .packed_near import PackedNearBlocks
from .fmm_capacity import storage_bytes


class VolumeNestedH2Action:
    """Cell-volume balanced nested bases with copied packed original near blocks.

    Conversion verifies each coupled far block against the source H factors.
    Source H errors remain additional; this is not a global error certificate.
    The source H object/factors are not retained. Setup still inherits its
    original-entry scans and conversion workspaces are outside retained limits.
    """
    def __init__(self, source, config=H2Config()):
        if not isinstance(source,VolumeHierarchicalAction) or not isinstance(config,H2Config):
            raise TypeError('Volume H2 requires a verified volume H action and H2Config')
        source.assert_current()
        provider,tree=source.provider,source.tree
        far=[];near=[]
        for (a,b),(rows,cols,block) in zip(source.block_nodes,source.blocks):
            if isinstance(block,LowRankBlock):far.append((a,b,block))
            else:near.append((a,b,rows,cols,block))
        fixed=storage_bytes((*provider.tensors(),*tree.tensors()))
        remaining=config.retained_budget_bytes-fixed
        if remaining<1:raise MemoryError('Volume H2 geometry exceeds retained budget')
        plan=plan_h2_storage(tree,far,components=3,dtype=source.dtype,
            max_rank=config.max_rank,budget_bytes=remaining)
        if not plan['admitted']:raise H2CapacityError(plan)
        near_budget=remaining-plan['retained_bytes_bound']
        if near and near_budget<1:raise MemoryError('Volume H2 near storage exceeds retained budget')
        packed=PackedNearBlocks(near,source.shape[0],retained_budget_bytes=near_budget) if near else None
        near_bytes=0 if packed is None else storage_bytes(packed.tensors())
        nested_config=replace(config,retained_budget_bytes=remaining-near_bytes)
        row_scale=provider.mesh.volumes.sqrt().repeat(3)
        column_scale=provider.mesh.volumes.rsqrt().repeat(3)
        nested=NestedH2Actions(tree,far,near=packed,components=3,dtype=source.dtype,
            row_scale=row_scale,column_scale=column_scale,config=nested_config)
        source.assert_current()
        self.provider,self.nested,self.config=provider,nested,config
        self.device,self.dtype,self.shape=source.device,source.dtype,source.shape
        self._binding=(id(provider),id(nested),id(config),self.device,self.dtype,self.shape)
        measured=storage_bytes(self.tensors())
        if measured>config.retained_budget_bytes:raise MemoryError('Volume H2 measured retained storage exceeds budget')
        self.receipt={'version':'regional_volume_nested_h2_v1','retained_bytes':measured,
            'retained_budget_bytes':config.retained_budget_bytes,'fixed_geometry_tree_bytes':fixed,
            'packed_near_bytes':near_bytes,'source_h':deepcopy(source.receipt),'nested':nested.receipt(),
            'source_h_object_retained':False,'source_h_far_factors_retained':False,
            'physical_qualification':False,'global_operator_error_verified':False,
            'surface_coupling_included':False,'setup_complexity_qualified':False,
            'coefficient_layout':'component_major_cell','block_error_metric':'cell_volume_weighted_L2_Frobenius'}

    def tensors(self):
        yield from self.provider.tensors();yield from self.nested.tree.tensors()
        yield from self.nested.tensors()
        yield from self.nested.test_basis.tensors();yield from self.nested.trial_basis.tensors()
        if self.nested.near is not None:yield from self.nested.near.tensors()

    def assert_current(self):
        if (id(self.provider),id(self.nested),id(self.config),self.device,self.dtype,self.shape)!=self._binding:
            raise ValueError('Volume H2 binding changed')
        self.provider.assert_current();self.nested.refresh()

    def apply(self,value):
        self.assert_current()
        return self.nested.apply(value)

    def adjoint(self,value):
        self.assert_current()
        return self.nested.adjoint(value)
