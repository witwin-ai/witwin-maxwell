"""Shared rectangular H block storage, verification and native actions."""
from dataclasses import dataclass,asdict
import math
import torch
from .cluster import support_block_partition
from .lowrank import streamed_cross_approximation,CompressionFailure,LowRankBlock
from .fmm_capacity import storage_bytes


@dataclass(frozen=True)
class HierarchicalBlockConfig:
    leaf_size: int=16
    max_rank: int=32
    tolerance: float=1e-8
    admissibility: float=1.
    row_batch: int=8
    retained_budget_bytes: int=1024**3

    def __post_init__(self):
        for name in ('leaf_size','max_rank','row_batch','retained_budget_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(name+' must be a positive integer')
        for name in ('tolerance','admissibility'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError(name+' must be positive and finite')


class HierarchicalBlockAction:
    """One fixed H object for batched forward, adjoint and current gradients.

    Far blocks use provider-supplied positive row/column scaling. Verified block
    Frobenius error is not a global operator or physical discretization bound.
    Retained capacity includes referenced mesh, tree, index maps and factors;
    setup workspaces, allocator overhead and backward graphs are excluded.
    """
    def __init__(self,provider,test_tree,source_tree,*,test_components,source_components,
                 row_scale,column_scale,config=HierarchicalBlockConfig()):
        if not isinstance(config,HierarchicalBlockConfig):
            raise TypeError('Hierarchical blocks require HierarchicalBlockConfig')
        if any(type(v) is not int or v<1 for v in (test_components,source_components)):
            raise ValueError('Block component counts must be positive integers')
        provider.assert_current();test_tree.assert_current();source_tree.assert_current()
        self.provider,self.config=provider,config
        self.device,self.dtype=provider.device,provider.dtype
        if any(t.device!=self.device or t.lower.dtype!=test_tree.lower.dtype for t in (test_tree,source_tree)):
            raise ValueError('Block trees must share provider device and real precision')
        self.test_tree,self.source_tree=test_tree,source_tree
        self.test_components,self.source_components=test_components,source_components
        self.shape=(test_components*test_tree.size,source_components*source_tree.size)
        for scale,size in zip((row_scale,column_scale),self.shape):
            if (not isinstance(scale,torch.Tensor) or scale.device!=self.device or scale.dtype!=test_tree.lower.dtype
                    or scale.shape!=(size,) or scale.requires_grad or not bool(torch.isfinite(scale).all() & (scale>0).all())):
                raise ValueError('Block scaling must be fixed matching finite positive CUDA vectors')
        self.row_scale,self.column_scale=row_scale.clone(),column_scale.clone()
        if any(t.receipt()['oversized_leaves'] for t in (test_tree,source_tree)):
            raise ValueError('Coincident supports exceed the leaf bound')
        retained=storage_bytes((*provider.tensors(),*test_tree.tensors(),*source_tree.tensors(),self.row_scale,self.column_scale))
        if retained>config.retained_budget_bytes:
            raise MemoryError('Hierarchical geometry exceeds retained budget')
        blocks=[];nodes=[];failures=[]
        starts=[t.starts.tolist() for t in (test_tree,source_tree)]
        stops=[t.stops.tolist() for t in (test_tree,source_tree)]
        children=[t.children.tolist() for t in (test_tree,source_tree)]
        def cells(node,side):
            tree=(test_tree,source_tree)[side]
            return tree.permutation[starts[side][node]:stops[side][node]]
        def dofs(ids,side):
            tree=(test_tree,source_tree)[side];components=(test_components,source_components)[side]
            return torch.cat(tuple(ids+k*tree.size for k in range(components)))
        for aa,bb,ff in support_block_partition(test_tree,source_tree,admissibility=config.admissibility):
            pending=list(zip(aa.tolist(),bb.tolist(),ff.tolist()))
            while pending:
                a,b,far=pending.pop();rows,cols=cells(a,0),cells(b,1)
                nr,nc=len(rows),len(cols);value=None
                index_bytes=(test_components*nr+source_components*nc)*8
                if far:
                    # Bound the final factor allocation before starting cross setup.
                    rank=min(config.max_rank,test_components*nr,source_components*nc)
                    estimate=(test_components*nr+source_components*nc)*rank*torch.empty((),dtype=self.dtype).element_size()
                    if retained+index_bytes+estimate>config.retained_budget_bytes:
                        raise MemoryError('Hierarchical factor ceiling exceeds retained budget')
                    def sample_rows(ids):
                        local=ids%nr;component=ids//nr
                        block=provider(rows[local],cols)
                        return block[component*len(ids)+torch.arange(len(ids),device=self.device)]
                    def sample_column(index):
                        block=provider(rows,cols[index%nc:index%nc+1])
                        return block[:,index//nc]
                    rs=self.row_scale[dofs(rows,0)]
                    cs=self.column_scale[dofs(cols,1)]
                    try:
                        value=streamed_cross_approximation(sample_rows,sample_column,(test_components*nr,source_components*nc),
                            device=self.device,dtype=self.dtype,max_rank=config.max_rank,rtol=config.tolerance,
                            row_batch=config.row_batch,row_scale=rs,column_scale=cs,pivoting='adaptive')
                    except CompressionFailure as failure:
                        failures.append({'test_node':a,'source_node':b,**failure.receipt})
                        ac=[c for c in children[0][a] if c>=0];bc=[c for c in children[1][b] if c>=0]
                        if ac or bc:
                            pending.extend((c,b,True) for c in ac) if ac and (not bc or nr>=nc) else pending.extend((a,c,True) for c in bc)
                            continue
                if value is None:
                    if nr>config.leaf_size or nc>config.leaf_size:
                        raise RuntimeError('Exact block exceeds leaf bound')
                    estimate=test_components*source_components*nr*nc*torch.empty((),dtype=self.dtype).element_size()
                    if retained+index_bytes+estimate>config.retained_budget_bytes:
                        raise MemoryError('Hierarchical near block exceeds retained budget')
                    value=provider(rows,cols)
                ri,ci=dofs(rows,0),dofs(cols,1)
                arrays=(value.left,value.right) if isinstance(value,LowRankBlock) else (value,)
                retained+=storage_bytes((ri,ci,*arrays))
                if retained>config.retained_budget_bytes:
                    raise MemoryError('Hierarchical factors exceed retained budget')
                blocks.append((ri,ci,value));nodes.append((a,b))
        self.blocks=tuple(blocks);self.block_nodes=tuple(nodes)
        self._binding=(id(provider),id(config),id(self.test_tree),id(self.source_tree),self.test_components,self.source_components,self.device,self.dtype,self.shape,self.block_nodes)
        self._layout=tuple((id(r),id(c),id(v)) for r,c,v in self.blocks)
        self._versions=tuple((id(t),t._version) for t in self.tensors())
        measured=storage_bytes(self.tensors())
        if measured>config.retained_budget_bytes:raise MemoryError('Hierarchical retained measurement exceeds budget')
        self.receipt={'version':'rectangular_hierarchical_blocks_v1',
            'configuration':asdict(config),'retained_bytes':measured,
            'far_blocks':sum(isinstance(v,LowRankBlock) for _,_,v in self.blocks),
            'exact_blocks':sum(not isinstance(v,LowRankBlock) for _,_,v in self.blocks),
            'compression_rejections':failures,'physical_qualification':False,
            'global_operator_error_verified':False,'setup_complexity_qualified':False,
            'coefficient_layout':'component_major','block_error_metric':'supplied_diagonal_scaling_Frobenius'}

    def tensors(self):
        yield from self.provider.tensors();yield from self.test_tree.tensors();yield from self.source_tree.tensors()
        yield self.row_scale;yield self.column_scale
        for rows,cols,value in self.blocks:
            yield rows;yield cols
            if isinstance(value,LowRankBlock):yield value.left;yield value.right
            else:yield value

    def assert_current(self):
        if (id(self.provider),id(self.config),id(self.test_tree),id(self.source_tree),self.test_components,self.source_components,self.device,self.dtype,self.shape,self.block_nodes)!=self._binding:
            raise ValueError('Hierarchical binding changed')
        self.provider.assert_current();self.test_tree.assert_current();self.source_tree.assert_current()
        if (tuple((id(r),id(c),id(v)) for r,c,v in self.blocks)!=self._layout
                or tuple((id(t),t._version) for t in self.tensors())!=self._versions):
            raise ValueError('Hierarchical blocks changed')
        for _,_,value in self.blocks:
            if isinstance(value,LowRankBlock):value.assert_current()

    def _action(self, value, adjoint):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0 if adjoint else 1]
                or value.device!=self.device or value.dtype!=self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Hierarchical action requires finite matching complex CUDA vectors')
        result=value.new_zeros((*value.shape[:-1],self.shape[1 if adjoint else 0]))
        for rows,cols,block in self.blocks:
            if adjoint:
                contribution=block.adjoint(value[...,rows]) if isinstance(block,LowRankBlock) else value[...,rows]@block.conj()
                result.index_add_(-1,cols,contribution)
            else:
                contribution=block.apply(value[...,cols]) if isinstance(block,LowRankBlock) else value[...,cols]@block.mT
                result.index_add_(-1,rows,contribution)
        return result

    def apply(self,value):return self._action(value,False)
    def adjoint(self,value):return self._action(value,True)
