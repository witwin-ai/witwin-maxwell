"""Exact scalar surface H/H2 numerical states for guarded inverse recovery."""
import copy
from dataclasses import asdict
import torch
from .hmatrix import HierarchicalOperator,HMatrixConfig
from .h2_operator import HierarchicalH2Operator
from .h2 import H2Config
from .nearfield import ScalarSystemBlocks
from .tensor_nearfield import GeneralizedSystemBlocks
from .ewald_nearfield import EwaldSystemBlocks, GeneralizedEwaldSystemBlocks
from .tensor_medium import sampled_medium_payload
from .riesz import LocalRWGRiesz
from .cluster import build_support_octree
from .operator_protocol import IterativeState
from .lowrank import LowRankBlock
from .volume_h_snapshot import _digest,_coverage
from .nested_h2_snapshot import encode_nested,restore_nested,_encode_near,_restore_near,_cpu,_keys,_array,_indices


def _array_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(_array_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(_array_bytes(v) for v in value)
    return 0


def _near_bytes(raw):
    records=[] if raw is None else raw['records']
    return _array_bytes(records)+(8*(2*len({r[0] for r in records})+1+3*len(records)) if records else 0)


def _provider_digest(provider):
    allowed = (ScalarSystemBlocks, GeneralizedSystemBlocks, EwaldSystemBlocks,
               GeneralizedEwaldSystemBlocks)
    if type(provider) not in allowed:
        raise TypeError('Surface snapshot requires an exact native regional block provider')
    provider.assert_current()
    if type(provider) is ScalarSystemBlocks:
        physics={'exterior':asdict(provider.exterior),'integration':asdict(provider.config),
            'bodies':[(b.structure_index,b.name,b.region_id,asdict(b.interior)) for b in provider.system.bodies]}
    else:
        physics = dict(provider=type(provider).__name__, exterior=sampled_medium_payload(provider.exterior),
            integration=asdict(provider.config), bodies=[(b.structure_index,b.name,b.region_id,
            sampled_medium_payload(b.interior)) for b in provider.system.bodies])
        if isinstance(provider, GeneralizedSystemBlocks):
            def policy(kernel):
                return None if kernel is None else dict(integration=kernel.integration.payload(),
                                                        cache_max_bytes=kernel.cache_max_bytes)
            physics['regional_policies'] = (policy(provider.exterior_kernel),
                                           tuple(map(policy,provider.interior_kernels)))
        if isinstance(provider, (EwaldSystemBlocks, GeneralizedEwaldSystemBlocks)):
            physics['periodic_kernel'] = provider.kernel.receipt()
    return _digest((physics,provider.input_tensors(),provider.geometry_tensors()))


def _numerical_version(provider):
    return ('exact_scalar_surface_h_v1' if type(provider) is ScalarSystemBlocks
            else 'exact_regional_surface_h_v2')


def _metric_encode(metric):
    metric.assert_current()
    if type(metric) is not LocalRWGRiesz:raise TypeError('Surface snapshot requires native local RWG metric')
    return _cpu({'rtol':metric.rtol,'atol':metric.atol,'maxiter':metric.maxiter,
        'impedance_squared':metric.impedance_squared,'records':metric.records,'diagonal':metric.diagonal})


def _metric_restore(payload,provider):
    _keys(payload,('rtol','atol','maxiter','impedance_squared','records','diagonal'))
    # Construct only local geometric Gram data, never an electromagnetic matrix
    # or inverse. This independently validates metric parameters and layout.
    metric=LocalRWGRiesz(provider.system,provider.exterior.eta,rtol=payload['rtol'],
        atol=payload['atol'],maxiter=payload['maxiter'])
    if payload['impedance_squared']!=metric.impedance_squared or len(payload['records'])!=len(metric.records):
        raise ValueError('Stored surface metric binding mismatch')
    records=[]
    for stored,reference in zip(payload['records'],metric.records):
        if not isinstance(stored,(tuple,list)) or len(stored)!=4 or tuple(stored[:2])!=reference[:2]:
            raise ValueError('Stored surface metric layout mismatch')
        ids=_array(stored[2],reference[2].shape,torch.int64,metric.device)
        blocks=_array(stored[3],reference[3].shape,metric.dtype,metric.device)
        if not torch.equal(ids,reference[2]) or not torch.equal(blocks,reference[3]):
            raise ValueError('Stored surface metric Gram mismatch')
        records.append((*reference[:2],ids,blocks))
    diagonal=_array(payload['diagonal'],metric.diagonal.shape,metric.diagonal.dtype,metric.device)
    if not bool((diagonal>0).all()) or not torch.equal(diagonal,metric.diagonal):
        raise ValueError('Stored surface metric diagonal mismatch')
    metric.records,metric.diagonal=tuple(records),diagonal
    metric._versions=tuple((id(t),t._version) for t in metric.tensors())
    metric.assert_current()
    return metric


def encode(operator):
    from .hierarchical_runtime import HierarchicalSystemOperator,HierarchicalH2SystemOperator
    from .periodic_hierarchical_runtime import PeriodicHierarchicalSystemOperator, PeriodicHierarchicalH2SystemOperator
    if type(operator) in (HierarchicalSystemOperator,HierarchicalH2SystemOperator,
                          PeriodicHierarchicalSystemOperator,PeriodicHierarchicalH2SystemOperator):
        from .physical_surface_snapshot import encode as encode_physical
        return encode_physical(operator)
    if type(operator) not in (HierarchicalOperator,HierarchicalH2Operator):
        raise TypeError('Snapshot requires an exact native scalar H or H2 operator')
    operator.refresh();nested=type(operator) is HierarchicalH2Operator
    data=encode_nested(operator.actions) if nested else {
        'near':_encode_near(operator.packed_near),'far':_cpu(tuple(
            (a,b,block.left,block.right,block.receipt)
            for (a,b),(_,_,block) in zip(operator.block_nodes,operator.blocks) if isinstance(block,LowRankBlock)))}
    payload={'version':_numerical_version(operator.provider),'kind':'h2' if nested else 'hmatrix',
        'provider_digest':_provider_digest(operator.provider),'configuration':operator.config.payload(),
        'source_configuration':operator.source_configuration.payload() if nested else None,
        'tree_digest':_digest(operator.tree.tensors()),'metric':_metric_encode(operator.metric),
        'data':data,'metadata':copy.deepcopy(operator.metadata)}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,provider,*,sources=()):
    """Load fixed numerical state on independently constructed scalar geometry.

    Geometry and local Gram preparation are allowed; kernel sampling, H setup,
    H2 conversion and solves are not. The original owned-retained budget scopes
    apply; full peak allocations and original compression revalidation do not.
    """
    if isinstance(payload,dict) and payload.get('version') in ('exact_physical_scalar_surface_h_v1','exact_physical_regional_surface_h_v2'):
        from .physical_surface_snapshot import restore as restore_physical
        return restore_physical(payload,provider,sources=sources)
    _keys(payload,('version','kind','provider_digest','configuration','source_configuration','tree_digest','metric','data','metadata','sha256'))
    if payload['version']!=_numerical_version(provider) or payload['kind'] not in ('hmatrix','h2'):
        raise ValueError('Unsupported surface H snapshot schema')
    if payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'}):
        raise ValueError('Surface H snapshot checksum mismatch')
    if payload['provider_digest']!=_provider_digest(provider):raise ValueError('Surface H snapshot physical binding mismatch')
    nested=payload['kind']=='h2'
    config=H2Config.restore(payload['configuration']) if nested else HMatrixConfig.restore(payload['configuration'])
    source_config=HMatrixConfig.restore(payload['source_configuration']) if nested else config
    if not nested and payload['source_configuration'] is not None:raise ValueError('Unexpected H source configuration')
    tree=build_support_octree(*provider.support_boxes(),leaf_size=source_config.leaf_size)
    if _digest(tree.tensors())!=payload['tree_digest']:raise ValueError('Surface H snapshot tree mismatch')
    metric=_metric_restore(payload['metric'],provider)
    dtype=metric.dtype;device=provider.system.device;shape=metric.shape
    metadata=copy.deepcopy(payload['metadata'])
    if (not isinstance(metadata,dict) or metadata.get('physical_qualification') is not False
            or metadata.get('global_riesz_error_verified') is not False
            or metadata.get('setup_complexity_qualified') is not False):
        raise ValueError('Invalid surface snapshot qualification metadata')
    if nested:
        fixed=(tree.receipt()['retained_bytes']+metric.receipt()['retained_bytes']
            +sum(t.numel()*t.element_size() for t in provider.geometry_tensors()))
        if fixed+_near_bytes(payload['data']['near'])>source_config.retained_budget_bytes:
            raise MemoryError('Surface H2 shared geometry/near exceeds source retained budget')
        total=provider.system.total_edges
        row=torch.cat((torch.full((total,),abs(provider.exterior.eta),device=device,dtype=tree.lower.dtype),
            torch.ones(total,device=device,dtype=tree.lower.dtype)))
        actions=restore_nested(payload['data'],tree,row,row.flip(0),expected_config=config,components=2,dtype=dtype)
        operator=object.__new__(HierarchicalH2Operator)
        operator.actions,operator.source_configuration=actions,source_config
        operator.provider,operator.metric,operator.tree=provider,metric,tree
        operator.config,operator.device,operator.dtype,operator.shape=config,device,dtype,shape
        operator._bindings=(id(actions),id(provider),id(metric),id(tree),config,source_config,shape,device,dtype)
        if (metadata.get('configuration')!=config.payload() or metadata.get('source_h_configuration')!=source_config.payload()
                or metadata.get('quadrature')!=asdict(provider.config)):
            raise ValueError('Surface H2 snapshot configuration metadata mismatch')
    else:
        raw=payload['data'];_keys(raw,('near','far'))
        geometry_bytes=sum(t.numel()*t.element_size() for t in provider.geometry_tensors())
        fixed=tree.receipt()['retained_bytes']+metric.receipt()['retained_bytes']+geometry_bytes
        records=[] if raw['near'] is None else raw['near']['records']
        near_bytes=_near_bytes(raw['near'])
        far_bytes=_array_bytes(raw['far'])
        for a,b,*_ in raw['far']:
            if any(type(n) is not int or not 0<=n<tree.node_count for n in (a,b)):raise ValueError('Invalid stored surface far node')
            far_bytes+=16*(int(tree.stops[a]-tree.starts[a])+int(tree.stops[b]-tree.starts[b]))
        if fixed+near_bytes+far_bytes>config.retained_budget_bytes:
            raise MemoryError('Surface H snapshot exceeds owned retained budget')
        rectangles=[];near=_restore_near(raw['near'],tree,2,dtype,rectangles)
        blocks=[];nodes=[]
        if near is not None:
            for a,b,gi,bi in near.entries:
                rows,columns,values=near.groups[gi]
                blocks.append((rows[bi],columns[bi],values[bi]));nodes.append((a,b))
        for record in raw['far']:
            if not isinstance(record,(tuple,list)) or len(record)!=5:raise ValueError('Invalid stored surface far record')
            a,b,left,right,receipt=record
            rows,columns=_indices(tree,a,2),_indices(tree,b,2)
            if not isinstance(left,torch.Tensor) or left.ndim!=2:raise ValueError('Invalid stored surface far factor')
            rank=left.shape[1]
            if not 0<=rank<=min(config.max_rank,len(rows),len(columns)):raise ValueError('Invalid stored surface far rank')
            block=LowRankBlock(_array(left,(len(rows),rank),dtype,device),
                _array(right,(rank,len(columns)),dtype,device),copy.deepcopy(receipt))
            blocks.append((rows,columns,block));nodes.append((a,b))
            rectangles.append((int(tree.starts[a]),int(tree.stops[a]),int(tree.starts[b]),int(tree.stops[b])))
        _coverage(rectangles,tree.size,tree.size)
        operator=object.__new__(HierarchicalOperator)
        operator.provider,operator.config,operator._bindings=provider,config,(provider,config)
        operator.device,operator.dtype,operator.shape=device,dtype,shape
        operator.tree,operator.metric=tree,metric
        operator.blocks,operator.block_nodes=tuple(blocks),tuple(nodes)
        operator.packed_near=operator._packed_near=near
        operator._block_nodes=operator.block_nodes
        operator._tree,operator._metric=tree,metric
        operator._versions=tuple((id(t),t._version) for t in operator._block_tensors())
        operator._layout=tuple((id(r),id(c),id(v)) for r,c,v in blocks)
        if (metadata.get('configuration')!=asdict(config) or metadata.get('retained_bytes')!=fixed+near_bytes+far_bytes
                or metadata.get('prepared_geometry_bytes')!=geometry_bytes
                or metadata.get('exact_blocks')!=len(records) or metadata.get('far_blocks')!=len(raw['far'])
                or metadata.get('exact_storage')!=(None if near is None else near.receipt())):
            raise ValueError('Surface H snapshot storage metadata mismatch')
    operator.iterative_state=IterativeState()
    operator._apply_calls=operator._adjoint_calls=0
    operator.metadata=metadata
    operator.refresh()
    return operator


def encode_inverse(inverse):
    """Store the guarded operator plus the exact configured Krylov policy."""
    from .implicit_surface_solve import FixedSurfaceInverse
    if type(inverse) is not FixedSurfaceInverse:raise TypeError('Expected exact fixed surface inverse')
    inverse.assert_current()
    payload={'version':'exact_fixed_surface_inverse_v1','operator':encode(inverse.operator),
        'stopping':asdict(inverse.rule),'refinement_steps':inverse.refinement_steps,
        'diagnostics':_cpu(inverse.receipt)}
    payload['sha256']=_digest(payload)
    return payload


def restore_inverse(payload,provider,*,sources=()):
    """Restore numerical factors and solve policy, without executing a solve."""
    from .implicit_surface_solve import FixedSurfaceInverse
    from .linalg import StoppingRule
    _keys(payload,('version','operator','stopping','refinement_steps','diagnostics','sha256'))
    if payload['version']!='exact_fixed_surface_inverse_v1' or payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'}):
        raise ValueError('Invalid fixed inverse snapshot identity/checksum')
    inverse=FixedSurfaceInverse(restore(payload['operator'],provider,sources=sources),StoppingRule(**payload['stopping']),
        refinement_steps=payload['refinement_steps'])
    log=payload['diagnostics']
    if (not isinstance(log,dict) or set(log)!=set(inverse.receipt)
            or any(log[k]!=v for k,v in inverse.receipt.items() if k not in ('last_forward','last_adjoint'))
            or any(log[k] is not None and not isinstance(log[k],dict) for k in ('last_forward','last_adjoint'))):
        raise ValueError('Invalid fixed inverse snapshot diagnostics')
    inverse.receipt=copy.deepcopy(log)
    return inverse
