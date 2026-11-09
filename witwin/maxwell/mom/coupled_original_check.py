"""Bounded original-equation and field checks for complete compressed responses."""
import math
import torch
from ..constants import ETA_0
from .tensor_medium import TensorMedium
from .coupled_volume_action import CoupledVolumeAction
from .volume_source_refinement import checked_dual_norms
from .fields import FieldQueryConfig,decode_interior_columns
from .runtime import DecodeHandle
from .operator import IntegrationConfig


def _indices(cells,count,components):
    return torch.cat(tuple(cells+k*count for k in range(components)))


def _original_source(action,local):
    rhs=local[0].new_zeros(local[0].shape[0],action.inverse.shape[0])
    for source,x,ids in zip(action.sources,local,action.indices):
        p=source.provider
        cells=x.reshape(len(x),3,p.column_size).abs().sum((0,1)).nonzero().flatten()
        if not len(cells):continue
        selected=_indices(cells,p.column_size,3)
        block=p(torch.arange(p.row_size,device=action.device),cells)
        rhs.index_add_(-1,ids,x[:,selected]@block.mT)
    return rhs


def _original_surface(action,surface,edge_batch):
    provider=action.inverse.operator.provider
    count=provider.system.total_edges
    result=torch.zeros_like(surface)
    for start in range(0,count,edge_batch):
        rows=torch.arange(start,min(start+edge_batch,count),device=action.device)
        row_ids=_indices(rows,count,2)
        result[:,row_ids]=provider.sampled_action(rows,surface,source_batch=edge_batch)
    return result


def _original_fields(action,local,surface,query_batch):
    system=action.inverse.operator.provider.system
    exterior=action.inverse.operator.provider.exterior
    output=[]
    for x,decoder,direct,ids in zip(local,action.fields,action.direct,action.indices):
        p=decoder.provider;v=direct.provider
        body=system.bodies[p.geometry.binding.volume.region_id-1]
        config=IntegrationConfig(order=p.order,singular_order=p.order)
        handle=DecodeHandle(body.basis,exterior,body.interior,(),config,{})
        if p.geometry.tensor_kernel is not None:handle.tensor_kernel=p.geometry.tensor_kernel
        cells=x.reshape(len(x),3,v.size).abs().sum((0,1)).nonzero().flatten()
        current=x[:,_indices(cells,v.size,3)]
        chunks=[]
        for start in range(0,v.size,query_batch):
            rows=torch.arange(start,min(start+query_batch,v.size),device=action.device)
            # Actual current columns avoid assembling a surface decoder matrix.
            field=decode_interior_columns(handle,surface[:,ids].mT,p.points[rows])[action.field=='H']
            value=field.permute(2,1,0).reshape(len(x),3*len(rows))
            if len(cells):value=value+current@v(rows,cells).mT
            chunks.append(action._cell_major(value))
        output.append(torch.cat(chunks,dim=-1))
    return torch.cat(output,dim=-1)


@torch.no_grad()
def _settings(action,*,relative_tolerance=1e-8,absolute_tolerance=1e-12,
                                   field_config=FieldQueryConfig(),column_batch=8,edge_batch=32,query_batch=8):
    """Check all original A/B rows and complete fixed-order field columns.

    This validates the represented quadrature, not its physical accuracy.
    Original blocks are streamed; all original rows/columns are visited, so
    cost can be quadratic even though a full matrix is never retained.
    """
    if type(action) is not CoupledVolumeAction:
        raise TypeError('Original checks require a complete coupled action')
    if (not isinstance(field_config,FieldQueryConfig)
            or any(type(v) is not int or v<1 for v in (column_batch,edge_batch,query_batch))
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (relative_tolerance,absolute_tolerance))
            or relative_tolerance+absolute_tolerance==0):
        raise ValueError('Invalid original-column check configuration')
    action.assert_current()
    n=action.shape[0];metric=action.inverse.operator.metric
    absolute=torch.cat([s.provider.binding.volume.volumes.new_full((size//3,),field_config.absolute_tolerance)
        /((ETA_0 if isinstance(s.provider.binding.surface_system.bodies[s.provider.binding.volume.region_id-1].interior,TensorMedium) else abs(s.provider.binding.surface_system.bodies[s.provider.binding.volume.region_id-1].interior.eta))
          if action.field=='H' else 1.) for s,size in zip(action.sources,action.sizes)])
    receipt={'version':'complete_original_coupled_columns_v1','field':action.field,
        'columns':n,'surface_rows':action.inverse.shape[0],'queries':n//3,'checks':[],
        'passed':True,'physical_qualification':False,'quadrature_error_estimated':False,
        'arbitrary_source_error_certified':False,'column_batch':column_batch,'edge_batch':edge_batch,
        'query_batch':query_batch,'relative_dual_tolerance':relative_tolerance,
        'absolute_dual_tolerance':absolute_tolerance,'field_relative_tolerance':field_config.relative_tolerance,
        'field_absolute_electric_tolerance':field_config.absolute_tolerance}
    return receipt,absolute


@torch.no_grad()
def check_coupled_original_columns_many(actions,*,relative_tolerance=1e-8,absolute_tolerance=1e-12,
                                        field_config=FieldQueryConfig(),column_batch=8,edge_batch=32,query_batch=8):
    """Check field channels with one shared original equation evaluation."""
    actions=tuple(actions)
    if not actions:raise ValueError('Original checks require at least one field channel')
    settings=[_settings(a,relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance,
        field_config=field_config,column_batch=column_batch,edge_batch=edge_batch,query_batch=query_batch) for a in actions]
    first=actions[0]
    if len({a.field for a in actions})!=len(actions):raise ValueError('Original field channels must be unique')
    for action in actions:
        if (action.inverse is not first.inverse or len(action.sources)!=len(first.sources)
                or any(a is not b for a,b in zip(action.sources,first.sources))):
            raise ValueError('Original field channels must share exact source actions and inverse')
    n=first.shape[0];metric=first.inverse.operator.metric
    for start in range(0,n,column_batch):
        for action in actions:action.assert_current()
        stop=min(n,start+column_batch)
        unit=torch.zeros(stop-start,n,device=first.device,dtype=first.dtype)
        unit[torch.arange(stop-start,device=first.device),torch.arange(start,stop,device=first.device)]=1
        local=first._split(unit);surface=first._surface(local,False)
        rhs=_original_source(first,local)
        error=checked_dual_norms(_original_surface(first,surface,edge_batch)-rhs,metric)
        magnitude=checked_dual_norms(rhs,metric)
        threshold=absolute_tolerance+relative_tolerance*magnitude
        equation_pass=bool(torch.isfinite(error).all() and torch.isfinite(threshold).all() and (error<=threshold).all())
        for action,(receipt,absolute) in zip(actions,settings):
            original=_original_fields(action,local,surface,query_batch).reshape(stop-start,-1,3)
            actual=action._decode(local,surface,False).reshape_as(original)
            field_error=torch.linalg.vector_norm(actual-original,dim=-1)
            field_threshold=absolute+field_config.relative_tolerance*torch.linalg.vector_norm(original,dim=-1)
            field_pass=bool(torch.isfinite(field_error).all() and torch.isfinite(field_threshold).all()
                and (field_error<=field_threshold).all())
            receipt['passed']=receipt['passed'] and equation_pass and field_pass
            receipt['checks'].append({'column_start':start,'column_stop':stop,
                'original_equation_pass':equation_pass,'original_field_pass':field_pass,
                'maximum_dual_residual':float(error.max()),'maximum_field_difference':float(field_error.max())})
    for action in actions:action.assert_current()
    return tuple(receipt for receipt,absolute in settings)


def check_coupled_original_columns(action,**settings):
    """Check all original rows and columns for one complete field channel."""
    return check_coupled_original_columns_many((action,),**settings)[0]
