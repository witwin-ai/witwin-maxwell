"""Sparse native electric observations for the shared physical-window loss API.

Field propagation and spectral sampling remain CUDA operations. Explicit CPU
postprocess uses the existing loss compiler, split-trace correction and native
copper-work function; no conductivity or constitutive formula is duplicated.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
from witwin.maxwell.postprocess.fdtd_loss import conductor_region_power
from witwin.maxwell.scene import prepare_scene


def _index_digest(indices):
    return hashlib.sha256(memoryview(indices.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


def _regional_support(prepared,compiled):
    selections={component:torch.zeros_like(mask) for component,mask in compiled.component_masks.items()}
    for component,mask in compiled.component_masks.items():
        selections[component][mask]=compiled.conductivity[component]!=0
    # A split trace can have zero effective native sigma but a lossy material on
    # one side. Its real native left sample binds the retained two-sided spectra
    # to the observation's exact phase and normalization.
    for face in compile_copper_faces(prepared,device='cpu'):
        component='E'+'xyz'[face.component];indices=face.indices
        valid=compiled.component_masks[component][tuple(indices.T)]
        selected=indices[valid]
        selections[component][tuple(selected.T)]=True
    return selections


def regional_monitors(cpu_scene, bounds, frequencies, *, observation="points", include_flux=True):
    """Record every nonzero bulk coefficient and selected foil left trace."""
    if str(cpu_scene.device) != 'cpu':
        raise ValueError('Regional observation geometry must be compiled explicitly on CPU.')
    if observation not in ('points','planes'):
        raise ValueError('Regional observation is points or planes.')
    center=tuple((lo+hi)/2 for lo,hi in bounds);size=tuple(hi-lo for lo,hi in bounds)
    loss=mw.PowerLossMonitor('line_roi_bulk',center,size,frequencies=None if observation=='planes' else frequencies,
        quadrature='physical_intersection')
    surface=mw.ClosedSurfaceMonitor.box('line_roi_flux',position=center,size=size,frequencies=frequencies)
    prepared=prepare_scene(cpu_scene);compiled=compile_power_loss_monitor(prepared,loss)
    selections=_regional_support(prepared,compiled)
    monitors=[loss,*([surface] if include_flux else [])];schema={}
    if observation=='planes':
        from witwin.maxwell.fdtd.coords import component_coords
        for component,selection in selections.items():
            axes=component_coords(prepared,component)
            # Reduce masks directly, avoiding a full selected-edge coordinate
            # list or millions of PointMonitor Python objects.
            normal_indices=torch.nonzero(selection.any(0).any(0)).flatten().tolist()
            planes=[]
            for index in normal_indices:
                name=f'line_roi_{component}_z{index}'
                position=float(axes[2][index])
                monitors.append(mw.PlaneMonitor(name,axis='z',position=position,
                    fields=(component,),frequencies=frequencies,compute_flux=False))
                planes.append({'name':name,'normal_index':index,'normal_position_m':position})
            schema[component]={'observation':'native_z_planes','planes':planes,
                'frequencies_hz':list(frequencies),'include_flux':include_flux,
                'full_shape':list(compiled.full_component_shapes[component]),
                'tangential_coords_m':[np.asarray(axis).tolist() for axis in axes[:2]],
                'loss_global_ids_sha256':_index_digest(compiled.global_ids[component]),
                'loss_global_id_count':int(compiled.global_ids[component].numel()),
                'selected_native_edge_count':int(selection.sum()),
                'recorded_native_plane_edge_count':len(normal_indices)*len(axes[0])*len(axes[1]),
                'volume_m3':float(compiled.component_volumes[component].sum())}
        return monitors,schema
    for component,selection in selections.items():
        indices=torch.nonzero(selection);names=[]
        component_axis='xyz'.index(component[-1].lower())
        axes=[getattr(prepared,axis+('_half64' if number==component_axis else '_nodes64'))
            for number,axis in enumerate('xyz')]
        for number,index in enumerate(indices.tolist()):
            name=f'line_roi_{component}_{number}';names.append(name)
            monitors.append(mw.PointMonitor(name,tuple(float(axis[i]) for axis,i in zip(axes,index)),fields=(component,)))
        schema[component]={'names':names,'indices':indices.tolist(),
            'full_shape':list(compiled.full_component_shapes[component]),
            'loss_global_ids':compiled.global_ids[component].tolist(),
            'volume_m3':float(compiled.component_volumes[component].sum())}
    return monitors,schema


def _native_plane_field(result,component,selection,frequency):
    """Decode one frequency without interpolation or a frequency-volume array."""
    shape=tuple(selection['full_shape'])
    dense=torch.zeros((1,*shape),dtype=torch.complex128)
    seen=set()
    for plane in selection['planes']:
        index=plane['normal_index']
        if not isinstance(index,int) or not 0<=index<shape[2] or index in seen:
            raise ValueError('Regional native plane indices must be unique valid z indices.')
        seen.add(index)
        payload=result.monitor(plane['name'],frequency=frequency)
        if tuple(payload.get('frequencies',()))!=(frequency,):
            raise ValueError('Regional native plane frequency selection changed.')
        entry=payload['components'][component]
        if (entry.get('collocated',False) or entry.get('plane_index')!=index
            or tuple(entry.get('plane_indices',()))!=(index,)
            or tuple(entry.get('plane_weights',()))!=(1.,)
            or tuple(entry.get('plane_positions',()))!=(plane['normal_position_m'],)):
            raise ValueError('Regional observation must retain one exact native Yee plane without interpolation.')
        coords=entry['coords']
        if len(coords)!=2 or any(not np.array_equal(np.asarray(actual),np.asarray(expected))
                for actual,expected in zip(coords,selection['tangential_coords_m'])):
            raise ValueError('Regional observation changed its native tangential coordinates.')
        values=entry['data']
        if tuple(values.shape)!=shape[:2] or not bool(torch.isfinite(values).all()):
            raise ValueError('Regional observation requires a finite complete native electric plane.')
        dense[0,:,:,index]=values.detach().cpu().to(torch.complex128)
    return dense


def _diagnostic_cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {key:_diagnostic_cpu_tree(item) for key,item in value.items()}
    if isinstance(value,tuple):return tuple(_diagnostic_cpu_tree(item) for item in value)
    if isinstance(value,list):return [_diagnostic_cpu_tree(item) for item in value]
    return value


def regional_plane_power(result,cpu_scene,schema,bounds):
    """Use exact native planes and single-frequency CPU inference loss inputs.

    This detached benchmark diagnostic does not advance fields on CPU or expose
    placeholder values as physical volume fields. It reuses shared bulk, split
    trace and film work, including their physical-window quadrature.
    """
    if result.stats().get('normalize_source',False):
        raise ValueError('This physical benchmark requires raw common spectral normalization.')
    if str(cpu_scene.device)!='cpu':
        raise ValueError('Regional loss reduction requires an explicit CPU diagnostic scene.')
    if set(schema)!={'Ex','Ey','Ez'} or any(
        selection.get('observation')!='native_z_planes'
        or tuple(selection['frequencies_hz'])!=tuple(result.frequencies)
        for selection in schema.values()):
        raise ValueError('Regional native-plane schema must cover every component and declared frequency in order.')
    loss=[monitor for monitor in cpu_scene.monitors if monitor.name=='line_roi_bulk']
    if len(loss)!=1 or loss[0].frequencies is not None:
        raise ValueError('Single-frequency regional inference requires an unspecified loss-monitor frequency axis.')
    if tuple(tuple(interval) for interval in loss[0].bounds)!=tuple(tuple(interval) for interval in bounds):
        raise ValueError('Regional loss bounds differ from the declared diagnostic monitor.')
    include_flux={selection['include_flux'] for selection in schema.values()}
    if len(include_flux)!=1:raise ValueError('Regional native-plane flux contracts disagree.')
    prepared=prepare_scene(cpu_scene)
    compiled=compile_power_loss_monitor(prepared,loss[0])
    supports=_regional_support(prepared,compiled)
    from witwin.maxwell.fdtd.coords import component_coords
    for component,selection in schema.items():
        axes=component_coords(prepared,component)
        required=torch.nonzero(supports[component].any(0).any(0)).flatten().tolist()
        if ([plane['normal_index'] for plane in selection['planes']]!=required
            or tuple(selection['full_shape'])!=tuple(compiled.full_component_shapes[component])
            or any(not np.array_equal(np.asarray(actual),np.asarray(expected))
                for actual,expected in zip(selection['tangential_coords_m'],axes[:2]))
            or any(plane['normal_position_m']!=float(axes[2][plane['normal_index']])
                for plane in selection['planes'])):
            raise ValueError('Regional plane geometry does not cover every compiled conductivity and split trace support.')
    del supports,compiled
    records=_diagnostic_cpu_tree(result.conductor_losses)
    copper_full=sum((conductor_region_power(record,bounds) for record in records.values()),
        torch.zeros(len(result.frequencies),dtype=torch.float64))
    for record in records.values():
        if tuple(float(f) for f in record['frequencies'])!=tuple(result.frequencies):
            raise ValueError('Regional conductor records changed their declared frequency order.')
    conduction=[];faces=[]
    for frequency in result.frequencies:
        fields={component:_native_plane_field(result,component,selection,frequency)
            for component,selection in schema.items()}
        diagnostics=mw.Result(method='fdtd',scene=cpu_scene,prepared_scene=prepared,
            frequencies=(frequency,),metadata={'conductor_losses':records},solver_stats=result.stats())
        data=diagnostics.power_loss('line_roi_bulk',electric_fields=fields)
        for component,selection in schema.items():
            ids=data.global_ids[component]
            if ids.numel()!=selection['loss_global_id_count'] or _index_digest(ids)!=selection['loss_global_ids_sha256']:
                raise ValueError('Regional loss support changed from its recorded native-plane contract.')
        conduction.append(float(data.conduction[0]))
        if True in include_flux:
            payload=result.monitor('line_roi_flux',frequency=frequency)['faces']
            faces.append({name:float((density*area).sum().detach().cpu())
                for name,(density,area) in ((name,plane_normal_poynting(face)) for name,face in payload.items())})
        del data,fields,diagnostics
    output={'frequencies_hz':result.frequencies,'bounds_m':bounds,
        'bulk_conduction_w':conduction,'copper_film_w':copper_full,
        'dense_temporary_frequency_count':1,
        'scope':'Detached diagnostic: exact CUDA native planes, single-frequency CPU inference through shared physical-intersection bulk/split-trace/film work. Compiled homogenized density; no material subcell-fill or new gradient claim.'}
    if faces:
        net=np.array([sum(record.values()) for record in faces])
        entering=np.array([sum(max(-power,0.) for power in record.values()) for record in faces])
        heat=np.array(conduction)+copper_full.numpy();residual=net+heat
        output.update(native_signed_face_flux_w=faces,net_outward_flux_w=net,
            entering_net_face_power_w=entering,power_balance_residual_w=residual,
            balance_relative_to_entering=[float(abs(r)/p) if p>0 else None for r,p in zip(residual,entering)],
            balance_relative_to_heat=[float(abs(r)/p) if p>0 else None for r,p in zip(residual,heat)])
    return output


def regional_power(result,cpu_scene,schema,bounds):
    """Reduce recorded spectra through an explicit CPU inference Result only.

    Dense tensors are temporary inputs to this bounded monitor, containing
    placeholders outside its nonzero conductivity and foil supports. They are
    never exposed as global physical fields or used by another field query.
    """
    if result.stats().get('normalize_source',False):
        raise ValueError('This physical benchmark requires raw common spectral normalization.')
    frequency_count=len(result.frequencies);fields={};samples={}
    for component,selection in schema.items():
        names=selection['names'];indices=torch.tensor(selection['indices'],dtype=torch.long).reshape(-1,3)
        # One packed transfer per component avoids thousands of scalar device
        # synchronizations; no field solve takes place on CPU.
        native=[]
        for name,index in zip(names,selection['indices']):
            payload=result.monitor(name)
            if tuple(payload['field_indices'][component])!=tuple(index):
                raise ValueError('Regional electric observation changed its declared native Yee address.')
            native.append(payload['components'][component].reshape(frequency_count))
        values=(torch.stack(native).detach().cpu().to(torch.complex128)
            if names else torch.empty((0,frequency_count),dtype=torch.complex128))
        dense=torch.zeros((frequency_count,*selection['full_shape']),dtype=torch.complex128)
        if len(indices):dense[(slice(None),*tuple(indices.T))]=values.T
        fields[component]=dense;samples[component]=values
    diagnostics=mw.Result(method='fdtd',scene=cpu_scene,frequencies=result.frequencies,
        metadata={'conductor_losses':result.conductor_losses},solver_stats=result.stats())
    data=diagnostics.power_loss('line_roi_bulk',electric_fields=fields)
    for component,selection in schema.items():
        if not torch.equal(data.global_ids[component],torch.tensor(selection['loss_global_ids'],dtype=torch.long)):
            raise ValueError('Regional compiled native support differs from its recorded observation contract.')
    copper=sum((conductor_region_power(record,bounds) for record in diagnostics.conductor_losses.values()),
        torch.zeros(frequency_count,dtype=torch.float64))
    faces=[]
    for frequency in result.frequencies:
        payload=result.at(frequency=frequency).monitor('line_roi_flux')['faces']
        faces.append({name:float((density*area).sum().detach().cpu())
            for name,(density,area) in ((name,plane_normal_poynting(face)) for name,face in payload.items())})
    net=np.array([sum(record.values()) for record in faces])
    entering=np.array([sum(max(-power,0.) for power in record.values()) for record in faces])
    heat=data.conduction.detach().cpu().numpy()+copper.numpy()
    residual=net+heat
    return {'frequencies_hz':result.frequencies,'bounds_m':bounds,
        'bulk_conduction_w':data.conduction,'copper_film_w':copper,'native_signed_face_flux_w':faces,
        'net_outward_flux_w':net,'entering_net_face_power_w':entering,
        'power_balance_residual_w':residual,
        'balance_relative_to_entering':[float(abs(r)/p) if p>0 else None for r,p in zip(residual,entering)],
        'balance_relative_to_heat':[float(abs(r)/p) if p>0 else None for r,p in zip(residual,heat)],
        'native_electric_samples':samples,'scope':'Raw spectral CUDA fields; explicit CPU postprocess through shared physical-intersection and split-trace loss. Compiled homogenized density, no joint material/subcell-fill claim.'}
