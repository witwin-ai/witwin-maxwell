"""Bounded original-column and quadrature checks for compressed weak sources."""
import math
import torch
from .surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from .riesz import LocalRWGRiesz
from .volume_response_matrix import VolumeMatrixConvergenceError


def _validate(action,metric,relative,absolute,cell_batch):
    if type(action) is not SurfaceVolumeHierarchicalAction or type(metric) is not LocalRWGRiesz:
        raise TypeError('Source checks require a regional source H action and local RWG metric')
    action.assert_current();metric.assert_current()
    if metric.system is not action.provider.binding.surface_system:
        raise ValueError('Source check metric must belong to the exact surface system')
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (relative,absolute))
            or relative+absolute==0 or type(cell_batch) is not int or cell_batch<1):
        raise ValueError('Source checks require finite positive tolerance and cell batch settings')


def _norms(value,action,metric):
    body=metric.system.bodies[action.provider.binding.volume.region_id-1]
    ids=torch.arange(body.edge_offset,body.edge_offset+body.basis.n_edges,device=action.device)
    indices=torch.cat((ids,ids+metric.system.total_edges))
    global_value=value.new_zeros(value.shape[0],metric.shape[0]).index_add(-1,indices,value)
    return checked_dual_norms(global_value,metric)


def checked_dual_norms(global_value,metric):
    """Global column dual norms with an independently checked metric inverse."""
    inverse=metric.inverse(global_value)
    # A caller-configured loose metric solve must not erase a nonzero norm.
    residual=torch.linalg.vector_norm(metric.apply(inverse)-global_value,dim=-1)
    magnitude=torch.linalg.vector_norm(global_value,dim=-1)
    tolerance=1e-12 if global_value.dtype==torch.complex128 else 1e-5
    if not bool(torch.isfinite(residual).all() and torch.isfinite(magnitude).all()
                and (residual<=tolerance*magnitude).all()):
        raise ValueError('Source check metric inverse failed its independent true residual gate')
    return (global_value.conj()*inverse).real.sum(-1).clamp_min(0).sqrt()


@torch.no_grad()
def check_source_columns(action,metric,*,previous=None,relative_tolerance=1e-4,
                         absolute_tolerance=1e-12,cell_batch=8):
    """Check compressed versus original columns, or a one-axis order increase.

    Every unit cell/Cartesian column is checked in the global exact RWG L2 dual
    proxy. Quadrature axes each receive half the requested tolerance, matching
    dense preparation. Original-column compression uses the full supplied
    tolerance. Neither is a certified trace-space or arbitrary-source bound.
    """
    _validate(action,metric,relative_tolerance,absolute_tolerance,cell_batch)
    p=action.provider;axis='compression';fraction=1.
    if previous is not None:
        _validate(previous,metric,relative_tolerance,absolute_tolerance,cell_batch)
        q=previous.provider
        if q.binding is not p.binding:
            raise ValueError('Source order checks require identical regional bindings')
        if p.tensor_kernel is not q.tensor_kernel or p.tensor_policy()!=q.tensor_policy():
            raise ValueError('Source order checks require identical tensor kernel and integration policy')
        if p.order==q.order and p.inner_order>q.inner_order:axis='inner'
        elif p.inner_order==q.inner_order and p.order>q.order:axis='outer'
        else:raise ValueError('Source quadrature checks must increase exactly one axis')
        fraction=.5
    receipt={'version':'streamed_source_column_dual_check_v1','axis':axis,
        'region_id':p.binding.volume.region_id,'outer_order':p.order,'inner_order':p.inner_order,
        'relative_tolerance':relative_tolerance,'absolute_tolerance_dual_proxy':absolute_tolerance,
        'axis_fraction':fraction,'cell_batch':cell_batch,'columns':3*p.column_size,
        'metric_inverse_relative_tolerance':1e-12 if action.dtype==torch.complex128 else 1e-5,
        'passed':True,'checks':[],'physical_qualification':False,'certified_error_bound':False}
    rows=torch.arange(p.row_size,device=action.device)
    for start in range(0,p.column_size,cell_batch):
        cells=torch.arange(start,min(start+cell_batch,p.column_size),device=action.device)
        cols=torch.cat(tuple(cells+k*p.column_size for k in range(3)))
        unit=torch.zeros(len(cols),action.shape[1],device=action.device,dtype=action.dtype)
        unit[torch.arange(len(cols),device=action.device),cols]=1
        value=action.apply(unit)
        reference=p(rows,cells).mT if previous is None else previous.apply(unit)
        error=_norms(value-reference,action,metric)
        magnitude=_norms(reference if previous is None else value,action,metric)
        threshold=fraction*(absolute_tolerance+relative_tolerance*magnitude)
        finite=torch.isfinite(error)&torch.isfinite(magnitude)&torch.isfinite(threshold)
        accepted=finite&(error<=threshold)
        passed=bool(accepted.all());receipt['passed']=receipt['passed'] and passed
        ratio=torch.where(threshold>0,error/threshold,torch.where(error==0,0.,torch.inf))
        receipt['checks'].append({'cell_start':start,'cell_stop':start+len(cells),'passed':passed,
            'failed_columns':int((~accepted).sum()),'finite':bool(finite.all()),
            'maximum_column_difference_dual_proxy':float(error.max()),
            'maximum_threshold_ratio':float(ratio.max())})
    action.assert_current();metric.assert_current()
    if previous is not None:previous.assert_current()
    return receipt


def prepare_source_action(binding,metric,*,config=SurfaceVolumeHConfig(),order=2,inner_order=2,
                          max_order=32,max_inner_order=64,relative_tolerance=1e-4,
                          absolute_tolerance=1e-12,cell_batch=8,tensor_kernel=None,
                          tensor_face_rule='triangle',tensor_kernel_batch=128):
    """Refine original weak-source integration without a full source matrix.

    Final outer comparison holds the accepted inner order fixed. Its previous
    outer row is rebuilt at that inner order when necessary, so different
    inner errors cannot be mistaken for a pure outer-order estimate.
    Compression admission is a separate per-column check, not an additive
    total error proof. Setup can still scan all original far entries.
    """
    if (any(type(v) is not int for v in (order,inner_order,max_order,max_inner_order))
            or not 2<=order<max_order or not 2<=inner_order<max_inner_order):
        raise ValueError('Source refinement requires increasing valid integration budgets')
    receipt={'version':'compressed_source_independent_axes_v1','converged':False,'checks':[],
        'evaluations':[],'relative_tolerance':relative_tolerance,
        'absolute_tolerance_dual_proxy':absolute_tolerance,'physical_qualification':False,
        'certified_error_bound':False,'max_outer_order':max_order,'max_inner_order':max_inner_order}
    def evaluate(outer,inner):
        action=SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=outer,inner_order=inner,
            tensor_kernel=tensor_kernel,tensor_face_rule=tensor_face_rule,tensor_kernel_batch=tensor_kernel_batch),config)
        compression=check_source_columns(action,metric,relative_tolerance=config.tolerance,
            absolute_tolerance=0.,cell_batch=cell_batch)
        if tensor_kernel is not None:receipt['tensor_source_policy']=action.provider.tensor_policy()
        receipt['evaluations'].append({'outer':outer,'inner':inner,'compression':compression})
        if not compression['passed']:
            receipt['failure_reason']='original_column_compression_check_failed'
            raise VolumeMatrixConvergenceError(receipt)
        return action
    def check(value,old):
        result=check_source_columns(value,metric,previous=old,relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,cell_batch=cell_batch)
        receipt['checks'].append(result)
        return result['passed']
    def inner_converged(outer,start):
        previous=evaluate(outer,start)
        while start<max_inner_order:
            start=min(2*start,max_inner_order);value=evaluate(outer,start)
            if check(value,previous):return value,start
            previous=value
        receipt['failure_reason']='inner_order_budget_exhausted'
        raise VolumeMatrixConvergenceError(receipt)
    previous,inner=inner_converged(order,inner_order)
    while order<max_order:
        old_order=order;order=min(2*order,max_order)
        value,new_inner=inner_converged(order,max(2,inner//2))
        if new_inner!=inner:previous=evaluate(old_order,new_inner)
        inner=new_inner
        if check(value,previous):
            receipt.update(converged=True,final_outer_order=order,final_inner_order=inner,failure_reason=None)
            return value,receipt
        previous=value
    receipt['failure_reason']='outer_order_budget_exhausted'
    raise VolumeMatrixConvergenceError(receipt)
