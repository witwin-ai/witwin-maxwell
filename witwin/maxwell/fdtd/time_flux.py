"""Prepared native time-flux reduction with actual collapsed copper side traces."""
from __future__ import annotations
import numpy as np
import torch
from .quadrature import _AXIS_CODES, _exact_cell_center_widths, _plane_coord_names


def prepare_time_flux(solver, record, num_samples):
    from .observers import (_FLUX_KERNEL_COMPONENTS, _PLANE_ALIGNMENT_RULES,
        _flux_yee_plane_position, _resolve_coord_source, _physical_plane_masks,
        _resolve_time_plane_component, resolve_plane_observer)
    if getattr(solver.scene, 'coordinates', None) is not None:
        raise NotImplementedError('Native time flux on mapped coordinates requires a qualified metric and sided trace reduction.')
    axis=record['axis'];normal=_AXIS_CODES[axis]
    position=_flux_yee_plane_position(solver.scene,axis,record['position'])
    record['yee_plane_position']=position
    components={}
    for component in record['fields']:
        meta=_resolve_time_plane_component(solver,axis,position,component)
        samples,_=resolve_plane_observer(solver.scene,dict(axis=axis,position=position,component=component))
        meta['plane_samples']=tuple((int(s['plane_index']),float(s['weight'])) for s in samples if s['weight']!=0)
        field=getattr(solver,meta['field_name'])
        meta['plane_views']=tuple((field.select(meta['axis_code'],i),w)for i,w in meta['plane_samples'])
        meta['plane_buffer']=torch.empty(meta['plane_shape'],device=solver.device,dtype=field.dtype)
        components[component]=meta
    rules=_PLANE_ALIGNMENT_RULES[axis];names=_plane_coord_names(axis)
    payloads={c:dict(coords=m['plane_coords']) for c,m in components.items()}
    coords=[_resolve_coord_source(payloads,*rules['coord_sources'][name]) for name in names]
    widths=[_exact_cell_center_widths(solver.scene,name,coord) for name,coord in zip(names,coords)]
    masks=_physical_plane_masks(solver.scene,axis,*coords)
    weights=widths[0][:,None]*widths[1][None,:]*masks[0][:,None]*masks[1][None,:]
    record.update(flux_components=components,
        flux_weights=torch.as_tensor(weights,device=solver.device,dtype=torch.float32).contiguous(),
        buffer=torch.zeros(num_samples,device=solver.device,dtype=torch.float32),
        flux_kernel_components=_FLUX_KERNEL_COMPONENTS[axis],
        flux_average_axis=rules['average_axis'][_FLUX_KERNEL_COMPONENTS[axis][0]],
        flux_scale=1. if record['normal_direction']=='+' else -1.,
        flux_copper_heads={})
    tangents=[i for i in range(3) if i!=normal]
    magnetic_validated=False
    for entry in getattr(solver,'_copper_foils',()):
        if entry.get('imaginary',False):continue
        component=entry['electric_name'];meta=components.get(component)
        if meta is None:continue
        face=entry['face'];samples=meta['plane_samples'];active=[i for i,w in samples]
        indices=entry['indices']
        selected=(indices[:,normal]>=min(active)) & (indices[:,normal]<=max(active))
        if not bool(selected.any()):continue
        if face.axis==normal:
            raise NotImplementedError('Time flux touching a normal copper stencil requires an explicitly sided surface.')
        if len(samples)!=1 or samples[0][1]!=1.:
            raise NotImplementedError('Copper time flux requires an exact native electric face plane.')
        if not magnetic_validated:
            for magnetic in ('Hx','Hy','Hz'):
                mu=getattr(solver,'mu_'+magnetic);sigma=getattr(solver,'sigma_m_'+magnetic)
                if not bool(torch.all(mu==float(solver.mu0))) or bool((sigma!=0).any()):
                    raise NotImplementedError('Copper time flux requires a uniform nonmagnetic background.')
            magnetic_validated=True
        rows=torch.nonzero(selected,as_tuple=False).reshape(-1)
        local=indices[rows][:,tangents]
        average=rules['average_axis'][component]
        shape=meta['plane_shape']
        # Endpoint/seam ownership requires its own periodic qualification.
        if bool(((local[:,average]<=0)|(local[:,average]>=shape[average]-1)).any()):
            raise NotImplementedError('Copper time flux requires native cells on both sides of every trace.')
        record['flux_copper_heads'].setdefault(component,[]).append(dict(
            indices=(local[:,0],local[:,1]),linear_indices=local[:,0]*shape[1]+local[:,1],
            rows=rows,right_e=entry['right_e'],
            gather_buffer=torch.empty(rows.numel(),device=solver.device,dtype=entry['right_e'].dtype)))
    record['flux_positive_planes']={}
    record['flux_positive_flat']={}
    for component in record['flux_kernel_components'][:2]:
        plane=components[component]['plane_buffer']
        positive=torch.empty_like(plane)if record['flux_copper_heads'].get(component)else plane
        record['flux_positive_planes'][component]=positive
        record['flux_positive_flat'][component]=positive.view(-1)


def _plane(solver,meta):
    # Views and contiguous interpolation storage are prepared once. Field values
    # remain live, including when native stepping replays a CUDA graph.
    target=meta['plane_buffer'];views=meta['plane_views']
    value,weight=views[0];target.copy_(value)
    if weight!=1.:target.mul_(weight)
    for value,weight in views[1:]:target.add_(value,alpha=weight)
    return target


def accumulate_time_flux(solver,record,k):
    planes={name:_plane(solver,meta) for name,meta in record['flux_components'].items()}
    ea,eb,ha,hb=record['flux_kernel_components'];positive=record['flux_positive_planes']
    for component in (ea,eb):
        heads=record['flux_copper_heads'].get(component,())
        if not heads:continue
        positive[component].copy_(planes[component])
        for head in heads:
            torch.index_select(head['right_e'],0,head['rows'],out=head['gather_buffer'])
            record['flux_positive_flat'][component].index_copy_(0,head['linear_indices'],head['gather_buffer'])
    solver.fdtd_module.planeFluxReduce(ea=planes[ea],eb=planes[eb],ha=planes[ha],hb=planes[hb],
        eaPositive=positive[ea],ebPositive=positive[eb],weights=record['flux_weights'],
        out=record['buffer'],outIndex=k,scale=record['flux_scale'],
        averageAxis=record['flux_average_axis']).launchRaw()
