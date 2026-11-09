"""Read-only native product-first Poynting diagnostic, not a production flux API."""
from __future__ import annotations
import numpy as np
import torch
from witwin.maxwell.result import _resample_plane_samples

RULES={'x':(('Ey','Hz',1,1.),('Ez','Hy',0,-1.)),
       'y':(('Ez','Hx',0,1.),('Ex','Hz',1,-1.)),
       'z':(('Ex','Hy',1,1.),('Ey','Hx',0,-1.))}
COORDS={'x':('y','z'),'y':('x','z'),'z':('x','y')}


def adjacent(value,axis):
    left=[slice(None)]*value.ndim;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
    return (value[tuple(left)]+value[tuple(right)])*.5


def native_product_flux(raw,selected,*,side_traces=(),native_axes=None,require_sided=False,single_normal_H_DOF_contract=False):
    """Retain normal interpolation, exact crop/cell widths/sign; change product order."""
    components=raw['components'];axis=selected['axis'];names=COORDS[axis]
    if components and all(value.get('collocated',False) for value in components.values()):
        from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
        density,area=plane_normal_poynting(selected)
        value=float((density*area).sum())
        return dict(old=value,product_first=value,covariance_correction=0.,fallback='already_collocated')
    if side_traces and not single_normal_H_DOF_contract:
        raise ValueError('Cu product-first flux requires the current collapsed-planar single normal-H DOF contract.')
    if require_sided and (not side_traces or native_axes is None):
        raise ValueError('Copper product-first flux requires explicit actual side traces and complete native axes.')
    for trace in side_traces:
        if trace['normal_axis']!=axis:continue
        entry=components.get(trace['component'])
        if entry is None:continue
        normal_index=torch.unique(trace['indices'][:,'xyz'.index(axis)])
        if len(normal_index)!=1:raise ValueError('Side trace must retain one declared normal grid plane.')
        active=[index for index,weight in zip(entry['plane_indices'],entry['plane_weights']) if weight!=0]
        if active and min(active)<=int(normal_index[0])<=max(active):
            raise ValueError('A flux face normal stencil touching copper requires an explicitly sided surface definition.')
    native_coords=tuple(raw[name] for name in names);target=tuple(selected[name] for name in names)
    old_density=0.;new_density=0.;correction=0.;side_density=0.;side_counts=0;left_errors=[]
    for e,h,averaging_axis,sign in RULES[axis]:
        if e not in components or h not in components:raise ValueError('Product-first flux requires all native tangential fields.')
        a,b=components[e],components[h]
        if a.get('collocated') or b.get('collocated'):raise ValueError('Mixed collocated/native flux support is unsupported.')
        if any(not np.array_equal(x,y) for x,y in zip(a['coords'],b['coords'])):
            raise ValueError('Native electric/magnetic product coordinates do not coincide.')
        E,H=[torch.as_tensor(value['data'],dtype=torch.complex128) for value in (a,b)]
        if E.ndim!=2 or E.shape!=H.shape:raise ValueError('Diagnostic requires one complete 2D frequency slice.')
        aligned_coords=list(a['coords']);aligned_coords[averaging_axis]=adjacent(np.asarray(aligned_coords[averaging_axis]),0)
        if any(not np.array_equal(x,y) for x,y in zip(aligned_coords,native_coords)):
            raise ValueError('Native product alignment does not match the retained exact cell grid.')
        A=adjacent(E,averaging_axis);B=adjacent(H,averaging_axis)
        first=adjacent(E*H.conj(),averaging_axis)
        diffE=torch.diff(E,dim=averaging_axis);diffH=torch.diff(H,dim=averaging_axis)
        covariance=.25*diffE*diffH.conj()
        error=abs(first-A*B.conj()-covariance)
        scale=adjacent(abs(E*H.conj()),averaging_axis)+abs(A*B.conj())+abs(covariance)
        if not bool((error<=32*torch.finfo(torch.float64).eps*scale+1e-300).all()):
            raise ValueError('Native covariance identity failed its input-conditioned float64 roundoff bound.')
        side_adjustment=torch.zeros_like(first)
        for trace in side_traces:
            if trace['component']!=e or trace['normal_axis']!=names[averaging_axis]:continue
            if native_axes is None:raise ValueError('Side traces require retained global native coordinate support.')
            face_axis='xyz'.index(axis);tangents=[i for i in range(3) if i!=face_axis]
            if len(a['plane_indices'])!=1 or tuple(a['plane_weights'])!=(1.0,):
                raise ValueError('Copper Et normal interpolation is unsupported in this bounded product diagnostic.')
            if any(not np.array_equal(x,native_axes[e][i]) for x,i in zip(a['coords'],tangents)):
                raise ValueError('Side-trace mapping requires complete matching global native tangential axes.')
            indices=trace['indices'].long(); chosen=indices[:,face_axis]==a['plane_indices'][0]
            indices=indices[chosen];values=trace['electric_fields'][:,chosen].to(torch.complex128)
            if not len(indices):continue
            ii=indices[:,tangents[0]];jj=indices[:,tangents[1]]
            coordinate=torch.as_tensor(a['coords'][averaging_axis])[indices[:,tangents[averaging_axis]]]
            if not bool((coordinate==trace['normal_coordinate']).all()):
                raise ValueError('Copper side trace does not lie on its declared native plane.')
            before=[ii.clone(),jj.clone()];after=[ii.clone(),jj.clone()]
            before[averaging_axis]-=1
            if bool((before[averaging_axis]<0).any()) or bool((after[averaging_axis]>=first.shape[averaging_axis]).any()):
                raise ValueError('Copper side trace requires physical cells on both sides.')
            original=E[ii,jj];normal_H=H[ii,jj]
            mismatch=float(abs(values[0]-original).max());left_errors.append(mismatch)
            budget=64*torch.finfo(torch.float32).eps*float(abs(E).max())+1e-300
            if mismatch>budget:raise ValueError('Copper side/native left phasors do not share the retained normalization/time convention.')
            side_adjustment[tuple(before)]+=.5*(values[0]-original)*normal_H.conj()
            side_adjustment[tuple(after)]+=.5*(values[1]-original)*normal_H.conj()
            side_counts+=len(indices)
        side_density=side_density+sign*.5*side_adjustment.real
        first=first+side_adjustment
        old_density=old_density+sign*.5*(A*B.conj()).real
        new_density=new_density+sign*.5*first.real
        correction=correction+sign*.5*covariance.real
    old_density=_resample_plane_samples(old_density,native_coords,target)
    new_density=_resample_plane_samples(new_density,native_coords,target)
    correction=_resample_plane_samples(correction,native_coords,target)
    side_density=_resample_plane_samples(side_density,native_coords,target)
    direction=1. if selected['normal_direction']=='+' else -1.
    widths=selected['cell_widths'];area=torch.as_tensor(widths[names[0]])[:,None]*torch.as_tensor(widths[names[1]])[None,:]
    values=[float((value*area).sum())*direction for value in (old_density,new_density,correction)]
    return dict(old=values[0],product_first=values[1],covariance_correction=values[2],side_trace_correction=float((side_density*area).sum())*direction,side_trace_edge_count=side_counts,max_native_left_trace_error=max(left_errors,default=0.),fallback=None)
