"""Native Yee spectral flux with physical cell projection and live side heads."""
from __future__ import annotations
import numpy as np
import torch

_PAIRS={'x':(('Ey','Hz',1,1.),('Ez','Hy',0,-1.)),
        'y':(('Ez','Hx',0,1.),('Ex','Hz',1,-1.)),
        'z':(('Ex','Hy',1,1.),('Ey','Hx',0,-1.))}
_COORDS={'x':('y','z'),'y':('x','z'),'z':('x','y')}


def _geometry(value,device):
    return torch.as_tensor(value,device=device,dtype=torch.float64)


def _interpolate_axis(values,source,target,axis):
    if len(source)==1:
        return values.index_select(axis,torch.zeros(len(target),device=values.device,dtype=torch.long))
    high=torch.searchsorted(source.contiguous(),target.contiguous()).clamp(1,len(source)-1)
    low=high-1
    weight=(target-source[low])/(source[high]-source[low])
    shape=[1]*values.ndim;shape[axis]=len(target)
    weight=weight.to(values.real.dtype).reshape(shape)
    return values.index_select(axis,low)*(1-weight)+values.index_select(axis,high)*weight


def native_plane_poynting(payload):
    """Return native product flux when retained staggered cell support is complete.

    Coordinate-only, already collocated, or incomplete field presentations retain
    the point-field Poynting functional. Geometry planning uses float64; field and
    head values retain their PyTorch graph, including right-side copper outputs.
    """
    if payload.get('native_product_disabled',False):return None
    axis=payload['axis'];names=_COORDS[axis];pairs=_PAIRS[axis]
    components=payload.get('native_components',payload.get('components',{}))
    required={name for e,h,_,_ in pairs for name in (e,h)}
    if not required.issubset(components) or 'cell_widths' not in payload:return None
    entries=[components[name] for name in required]
    if any(not isinstance(entry,dict) or 'coords' not in entry for entry in entries):return None
    if all(entry.get('collocated',False) for entry in entries):return None
    first=entries[0]
    # Older external payloads can omit collocated=True while supplying four
    # common-point arrays. They contain no staggered covariance to reconstruct.
    def same_coordinates(a,b):
        return all(torch.equal(torch.as_tensor(x).detach().cpu(),torch.as_tensor(y).detach().cpu()) for x,y in zip(a,b))
    if all(same_coordinates(entry['coords'],first['coords']) for entry in entries):return None
    if payload.get('requires_native_flux_heads',False) and not payload.get('native_flux_heads'):
        raise ValueError('Native flux crossing copper requires its actual two-sided spectral heads.')
    tensors=[entry['data'] for entry in entries if isinstance(entry['data'],torch.Tensor)]
    device=tensors[0].device if tensors else torch.device('cpu')
    use_torch=bool(tensors) or any(isinstance(payload[name],torch.Tensor) for name in names)
    wide=any(torch.as_tensor(entry['data']).dtype in (torch.float64,torch.complex128) for entry in entries) or bool(payload.get('native_flux_heads'))
    dtype=torch.complex128 if wide else torch.complex64
    targets=[_geometry(payload[name],device) for name in names]
    widths=[_geometry(payload['cell_widths'][name],device) for name in names]
    if any(w.shape!=x.shape or not bool(torch.all(w>0)) for w,x in zip(widths,targets)):
        raise ValueError('Native flux requires positive exact widths matching its physical cells.')
    density=None
    for e,h,averaging,sign in pairs:
        a,b=components[e],components[h]
        if a.get('collocated',False) or b.get('collocated',False) or not same_coordinates(a['coords'],b['coords']):
            raise ValueError('Native product flux requires coincident native E/H pair support.')
        E=torch.as_tensor(a['data'],device=device).to(dtype);H=torch.as_tensor(b['data'],device=device).to(dtype)
        if E.shape!=H.shape or E.ndim not in (2,3):raise ValueError('Native flux component shapes differ.')
        geometric=[_geometry(value,device) for value in a['coords']]
        dimension=E.ndim-2+averaging;other=1-averaging
        nodes=geometric[averaging]
        if len(nodes)<2:raise ValueError('Native flux averaging requires two primal endpoints.')
        Q=E*H.conj();left=[slice(None)]*Q.ndim;right=left.copy()
        left[dimension]=slice(None,-1);right[dimension]=slice(1,None)
        lo=Q[tuple(left)].clone();hi=Q[tuple(right)].clone()
        for head in payload.get('native_flux_heads',()):
            if head['component']!=e:continue
            if head['averaging_axis']!=averaging:raise ValueError('Copper flux head has the wrong tangential normal axis.')
            indices=head['indices'].to(device=device,dtype=torch.long);ii,jj=indices[:,0],indices[:,1]
            before=[ii.clone(),jj.clone()];before[averaging]-=1
            if bool((before[averaging]<0).any()) or bool((indices[:,averaging]>=lo.shape[dimension]).any()):
                raise ValueError('Copper flux heads need native cells on both sides.')
            native=E[...,ii,jj];normalH=H[...,ii,jj]
            negative=head['negative'].to(device=device,dtype=dtype);positive=head['positive'].to(device=device,dtype=dtype)
            if negative.shape!=native.shape or positive.shape!=native.shape:raise ValueError('Copper flux head frequency/support shapes differ.')
            hi[...,before[0],before[1]]=hi[...,before[0],before[1]]+(negative-native)*normalH.conj()
            lo[...,ii,jj]=lo[...,ii,jj]+(positive-native)*normalH.conj()
        center=targets[averaging];width=widths[averaging]
        cell=(torch.searchsorted(nodes.contiguous(),center.contiguous(),right=True)-1).clamp(0,len(nodes)-2)
        lower,upper=center-width/2,center+width/2
        tolerance=4*torch.finfo(torch.float64).eps*torch.maximum(center.abs(),nodes[cell].abs()).clamp_min(torch.finfo(torch.float64).tiny)
        tolerance=torch.minimum(tolerance,(nodes[cell+1]-nodes[cell])/32)
        if not bool(((lower>=nodes[cell]-tolerance)&(upper<=nodes[cell+1]+tolerance)).all()):
            raise ValueError('Native flux cell projection cannot span multiple primal cells.')
        fraction=(center-nodes[cell])/(nodes[cell+1]-nodes[cell])
        shape=[1]*E.ndim;shape[dimension]=len(cell)
        fraction=fraction.to(E.real.dtype).reshape(shape)
        projected=lo.index_select(dimension,cell)*(1-fraction)+hi.index_select(dimension,cell)*fraction
        projected=_interpolate_axis(projected,geometric[other],targets[other],E.ndim-2+other)
        value=.5*sign*projected.real
        density=value if density is None else density+value
    density=density*(1. if payload.get('normal_direction','+')=='+' else -1.)
    area=widths[0][:,None]*widths[1][None,:]
    return (density,area) if use_torch else (density.detach().numpy(),area.numpy())


def _require_uniform_nonmagnetic(scene):
    materials=[structure.material for structure in scene.structures
               if bool(getattr(structure,'enabled',True)) and not bool(getattr(structure.material,'is_copper_foil',False))]
    for attribute in ('background','background_material'):
        material=getattr(scene,attribute,None)
        if material is not None:materials.append(material)
    from ..media import GyromagneticFerrite
    for material in materials:
        try:
            declaration=getattr(material,'mu_tensor',None)
            if declaration is None:declaration=material.mu_r
            elif hasattr(declaration,'as_tuple'):declaration=declaration.as_tuple()
            elif hasattr(declaration,'rows'):declaration=declaration.rows
            mu=torch.as_tensor(declaration)
        except (AttributeError,TypeError,ValueError,RuntimeError):
            raise NotImplementedError('Native copper flux requires explicit fixed scalar unit permeability.') from None
        if mu.requires_grad or mu.numel()!=1 or mu.is_complex() or float(mu)!=1.:
            raise NotImplementedError('Native copper flux currently requires a fixed uniform nonmagnetic background.')
        sigma_m=torch.as_tensor(getattr(material,'sigma_m',0.))
        if sigma_m.requires_grad or bool((sigma_m!=0).any()) or getattr(material,'sigma_m_tensor',None) is not None or any(getattr(material,key,()) for key in ('mu_debye_poles','mu_drude_poles','mu_lorentz_poles')) or isinstance(material,GyromagneticFerrite) or bool(getattr(material,'is_ferrite',False)):
            raise NotImplementedError('Native copper flux does not yet support magnetic loss, dispersion or gyrotropy.')
    for region in scene.material_regions:
        bounds=torch.as_tensor(region.mu_bounds)
        if bounds.requires_grad or not bool((bounds==1).all()):
            raise NotImplementedError('Native copper flux requires fixed unit permeability in all material regions.')


def enrich_plane_flux(result,payload):
    """Late bind current collapsed-Cu Et heads; never infer a phase/scale fit."""
    if payload.get('kind')!='plane' or payload.get('axis') not in _PAIRS:return payload
    if payload.get('native_copper_flux_context')=='collapsed_planar_v4':return payload
    selected=dict(payload)
    if getattr(result.scene,'coordinates',None) is not None:
        selected['native_product_disabled']=True
    entries=selected.get('native_components',selected.get('components',{}))
    required={name for e,h,_,_ in _PAIRS[selected['axis']] for name in (e,h)}
    if not required.issubset(entries) or 'cell_widths' not in selected:return selected
    if not selected.get('native_product_disabled') and any(bool(getattr(s.material,'is_copper_foil',False)) for s in result.scene.structures):
        from ..compiler.copper_foil import compile_copper_faces
        from .coords import component_coords
        # Surface-only results deliberately need no full-volume DFT arrays.
        from ..scene import prepare_scene
        prepared=result._prepared_scene
        if prepared is None:
            dtype=torch.float64 if result.stats().get('field_dtype')=='torch.float64' else torch.float32
            prepared=prepare_scene(result.scene,material_dtype=dtype)
            result._prepared_scene=prepared
        faces=compile_copper_faces(prepared,device='cpu')
        records=result._metadata.get('conductor_losses',{})
        normal='xyz'.index(selected['axis']);tangents=[i for i in range(3) if i!=normal]
        names=_COORDS[selected['axis']];heads=[];normal_corrections=[]
        def array(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        for face in faces:
            component='E'+'xyz'[face.component]
            if component not in required:continue
            entry=entries[component]
            if entry.get('collocated',False):continue
            rule=next(rule for rule in _PAIRS[selected['axis']] if rule[0]==component)
            averaging=rule[2]
            active_pairs=[(index,float(weight)) for index,weight in zip(entry['plane_indices'],entry['plane_weights']) if weight!=0]
            active=[index for index,_ in active_pairs]
            if not active:continue
            if face.axis==normal:
                if not min(active)<=face.normal_index<=max(active):continue
                candidates=face.indices.cpu().numpy()
            else:
                candidates=face.indices.cpu().numpy()
                candidates=candidates[np.isin(candidates[:,normal],active)]
            if not len(candidates):continue
            axes=component_coords(prepared,component);local=[];valid=np.ones(len(candidates),dtype=bool)
            for dim,global_axis in enumerate(tangents):
                coordinates=array(entry['coords'][dim]).astype(np.float64,copy=False)
                points=np.asarray(axes[global_axis])[candidates[:,global_axis]]
                indices=np.searchsorted(coordinates,points).clip(0,len(coordinates)-1)
                valid &= coordinates[indices]==points
                targets=array(selected[names[dim]]).astype(np.float64,copy=False)
                if dim==averaging:
                    cell=(np.searchsorted(coordinates,targets,side='right')-1).clip(0,len(coordinates)-2)
                    allowed=np.unique(np.r_[cell,cell+1])
                else:
                    hi=np.searchsorted(coordinates,targets).clip(1,max(1,len(coordinates)-1)) if len(coordinates)>1 else np.zeros(len(targets),dtype=int)
                    allowed=np.unique(np.r_[np.maximum(hi-1,0),hi])
                valid &= np.isin(indices,allowed);local.append(indices)
            candidates=candidates[valid]
            if not len(candidates):continue

            if any(not np.isfinite(weight) or weight<0. or weight>1. for _,weight in active_pairs) or abs(sum(weight for _,weight in active_pairs)-1.)>32*np.finfo(np.float64).eps:
                raise ValueError('Native copper flux requires a convex declared normal interpolation stencil.')
            local_indices=np.stack(local,axis=1)[valid]
            record=records.get(face.name,{})
            traces=[trace for trace in record.get('bulk_electric_traces',()) if trace['component']==component]
            if len(traces)!=1:raise ValueError('Copper-crossing native flux requires one actual retained Et trace record.')
            trace=traces[0];global_indices=trace['indices'].detach().cpu().numpy()
            shape=[len(axis) for axis in axes]
            flatten=lambda idx:(idx[:,0]*shape[1]+idx[:,1])*shape[2]+idx[:,2]
            flat=flatten(global_indices);order=np.argsort(flat);wanted=flatten(candidates)
            positions=np.searchsorted(flat[order],wanted)
            if np.any(positions>=len(order)) or not np.array_equal(flat[order][positions],wanted):
                raise NotImplementedError('Native copper flux has unresolved PEC-contact or missing side-trace support.')
            values=trace['electric_fields']
            frequencies=tuple(float(f) for f in record['frequencies'])
            requested=tuple(float(f) for f in selected.get('frequencies') or (selected['frequency'],))
            frequency_indices=torch.tensor([frequencies.index(f) for f in requested],device=values.device,dtype=torch.long)
            row_indices=torch.tensor(order[positions],device=values.device,dtype=torch.long)
            values=values.index_select(0,frequency_indices).index_select(-1,row_indices)
            spectrum=selected.get('source_normalization_spectra',{}).get(component)
            if result.stats().get('normalize_source',False):
                if spectrum is None:
                    raise NotImplementedError('Normalized copper native flux requires the exact applied source normalization spectrum.')
                source_frequencies=tuple(selected.get('source_normalization_frequencies',requested))
                factor=torch.as_tensor(spectrum,device=values.device)
                if factor.ndim:
                    indices=torch.tensor([source_frequencies.index(f) for f in requested],device=values.device)
                    factor=factor.index_select(0,indices)
                values=values/factor.reshape(-1,1,1)
            _require_uniform_nonmagnetic(result.scene)
            # Ordinary reports can reside on CPU while native observers remain
            # on CUDA; .to preserves live adjoint outputs in either placement.
            if not isinstance(entry['data'],torch.Tensor):
                raise NotImplementedError('Copper-sided flux currently requires native Torch observer spectra.')
            values=values.to(device=entry['data'].device)
            if entry['data'].ndim==2:values=values[0]
            negative,positive=(values[:,0],values[:,1]) if values.ndim==3 else (values[0],values[1])
            if face.axis==normal:
                # Choose the physical side at the actual sampling coordinate,
                # never the requested coordinate if the observer snapped it.
                position=float(selected.get('yee_plane_position',selected['position']))
                sheet_position=float(axes[normal][face.normal_index])
                if position==sheet_position:
                    raise NotImplementedError('A spectral flux plane exactly on CopperFoil requires an explicit physical side.')
                side=1 if position>sheet_position else -1
                Hentry=entries[rule[1]]
                Haxis=component_coords(prepared,rule[1])[normal]
                Hpositions=[float(Haxis[index]) for index,weight in zip(Hentry['plane_indices'],Hentry['plane_weights']) if weight!=0]
                if not Hpositions or any(side*(coordinate-sheet_position)<=0 for coordinate in Hpositions):
                    raise NotImplementedError('Copper-normal spectral flux requires every active tangential-H sample in the same exterior halfspace.')
                correction=entry['data'].to(device=values.device,dtype=values.dtype)*0
                normal_weight=dict(active_pairs)[face.normal_index]
                if side>0:
                    ii,jj=(torch.as_tensor(local_indices[:,i],device=values.device,dtype=torch.long) for i in range(2))
                    correction=correction.clone()
                    correction[...,ii,jj]=normal_weight*(positive-negative)
                    entry=dict(entry,data=entry['data'].to(device=values.device,dtype=values.dtype)+correction)
                    entries=dict(entries);entries[component]=entry
                    selected['native_components']=entries
                    # Preserve the exact existing presentation linear operator:
                    # native endpoint averaging, then finite-cell resampling.
                    from ..result import _resample_plane_samples
                    dimension=correction.ndim-2+averaging
                    low=[slice(None)]*correction.ndim;high=low.copy()
                    low[dimension]=slice(None,-1);high[dimension]=slice(1,None)
                    aligned=.5*(correction[tuple(low)]+correction[tuple(high)])
                    aligned_coords=list(entry['coords'])
                    points=array(aligned_coords[averaging])
                    aligned_coords[averaging]=.5*(points[:-1]+points[1:])
                    target=tuple(selected[name] for name in names)
                    presentation_delta=_resample_plane_samples(aligned,tuple(aligned_coords),target)
                    if component in selected:selected[component]=selected[component]+presentation_delta
                    if selected.get('components',{}).get(component,{}).get('collocated',False):
                        components=dict(selected['components'])
                        components[component]=dict(components[component],data=components[component]['data']+presentation_delta)
                        selected['components']=components
                    elif 'native_components' not in payload:
                        components=dict(selected['components']);components[component]=entry;selected['components']=components
                normal_corrections.append(dict(component=component,sheet_coordinate=sheet_position,
                    sampling_position=position,side=side,normal_weight=normal_weight,
                    active_H_positions=tuple(Hpositions),edge_count=len(candidates)))
                continue
            if len(active_pairs)>1:
                # The ordinary observer has already gathered the exact convex
                # normal interpolation of native left Et (and exterior E where
                # the finite foil ends). Change only actual Cu rows to right Et:
                # Eright_projected=Eleft_gathered+sum_j w_j*(right_j-left_j).
                # This also retains the original exterior contribution if only
                # one stencil node belongs to the finite foil footprint.
                unique,inverse=np.unique(local_indices,axis=0,return_inverse=True)
                weights=dict(active_pairs)
                w=torch.tensor([weights[int(row[normal])] for row in candidates],device=values.device,dtype=values.real.dtype)
                jump=(positive-negative)*w
                inverse=torch.as_tensor(inverse,device=values.device,dtype=torch.long)
                total=torch.zeros((*jump.shape[:-1],len(unique)),device=values.device,dtype=values.dtype).index_add(-1,inverse,jump)
                local_indices=unique
                ii,jj=(torch.as_tensor(local_indices[:,i],device=entry['data'].device,dtype=torch.long) for i in range(2))
                negative=entry['data'][...,ii,jj].to(device=values.device,dtype=values.dtype)
                positive=negative+total
            heads.append(dict(component=component,averaging_axis=averaging,
                indices=torch.as_tensor(local_indices,dtype=torch.long),negative=negative,positive=positive,
                normal_interpolation=tuple(active_pairs)))
        if normal_corrections:selected['native_normal_copper_corrections']=normal_corrections
        selected['native_copper_flux_context']='collapsed_planar_v4'
        if heads:
            selected['native_flux_heads']=heads;selected['requires_native_flux_heads']=True
            selected['native_flux_contract']='collapsed_axis_aligned_CopperFoil_single_normal_H_DOF_uniform_nonmagnetic'
    if selected.get('compute_flux') or 'flux' in selected:
        from .quadrature import plane_normal_poynting
        density,area=plane_normal_poynting(selected)
        value=(density*area).sum(axis=(-2,-1))
        selected['flux']=selected['power']=value
    if selected.get('native_product_disabled'):
        provenance='collocated_mapped_presentation_native_product_not_applied'
    elif native_plane_poynting(selected) is None:
        provenance='point_field_product_incomplete_or_already_collocated_support'
    else:
        provenance='native_Yee_product_then_cell_projection_v1'
    selected['flux_reduction_provenance']=provenance
    return selected
