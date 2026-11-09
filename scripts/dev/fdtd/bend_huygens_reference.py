"""Independent Hertzian fields on a physical curved chart control surface.

These analytic samples validate postprocessing, not native field evolution.
The current moment and physical source position are explicit; no field, phase,
geometry or vacuum normalization is fitted to a numerical result.
"""
from __future__ import annotations
import math
import torch
import witwin.maxwell as mw


def dipole_fields(points, *, frequency, position, direction, magnetic=False):
    omega=2*math.pi*frequency;k=omega/mw.constants.C_0;eta=mw.constants.ETA_0
    displacement=points-position
    radius=torch.linalg.vector_norm(displacement,dim=-1)
    unit=displacement/radius[:,None]
    projection=(unit*direction).sum(-1)
    transverse=direction-unit*projection[:,None]
    phase=torch.exp(1j*k*radius)
    radial=eta*projection/(2*math.pi*radius.square())*(1+1j/(k*radius))*phase
    tangent=eta*k/(4*math.pi*radius)*(1j-1/(k*radius)-1j/(k*radius).square())*phase
    electric=radial[:,None]*unit+tangent[:,None]*transverse
    magnetic_field=(k/(4*math.pi*radius)*(1/(k*radius)-1j)*phase)[:,None]*torch.cross(direction.expand_as(unit),unit,dim=-1)
    return (-eta*magnetic_field,electric/eta) if magnetic else (electric,magnetic_field)


def dipole_far(directions, *, frequency, position, direction, radius, magnetic=False):
    k=2*math.pi*frequency/mw.constants.C_0;eta=mw.constants.ETA_0
    transverse=direction-directions*(directions*direction).sum(-1)[:,None]
    phase=torch.exp(1j*k*(radius-directions@position))
    electric=(1j*eta*k/(4*math.pi*radius)*phase)[:,None]*transverse
    magnetic_field=torch.cross(directions.to(electric.dtype),electric,dim=-1)/eta
    return (-eta*magnetic_field,electric/eta) if magnetic else (electric,magnetic_field)


def analytic_result(*, curvature=8., count=24, frame=None, magnetic=False, device='cpu',frequency=1e9):
    chart=mw.BendCoordinates(curvature=curvature,origin=(.02,-.01,.03),frame=frame,device=device)
    surface=mw.ClosedSurfaceMonitor.box('huygens',size=(.12,.07,.10),sample_spacing=.01,frequencies=(frequency,))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.09,.09),(-.06,.06),(-.08,.08))),grid=mw.GridSpec.uniform(.01),
        coordinates=chart,monitors=(surface,),device=device)
    chart=scene.coordinates
    # Emulate the solver payload: closed faces lie on their snapped Yee planes
    # and span the watertight grid-resolved rectangle.
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.fdtd.observers import finite_plane_integration_bounds,nearest_primal_node
    prepared=prepare_scene(scene)
    direction=chart.frame@torch.tensor((.3,-.4,.8),dtype=torch.float64,device=device)
    direction=direction/torch.linalg.vector_norm(direction)
    samples={};physical_flux=torch.zeros((),device=device,dtype=torch.float64)
    for face in surface.faces:
        normal_axis='xyz'.index(face.axis);tangential=[axis for axis in range(3) if axis!=normal_axis]
        q=torch.empty((count,count,3),device=device,dtype=torch.float64)
        vectors=[];widths=[]
        bounds=finite_plane_integration_bounds(prepared,face)
        plane=nearest_primal_node(getattr(prepared,face.axis+'_nodes64'),face.plane_position)
        for axis in tangential:
            name='xyz'[axis];lower,upper=bounds[name]
            width=(upper-lower)/count
            vectors.append(lower+(torch.arange(count,device=device,dtype=torch.float64)+.5)*width)
            widths.append(torch.full((count,),width,device=device,dtype=torch.float64))
        aa,bb=torch.meshgrid(*vectors,indexing='ij')
        q[...,normal_axis]=plane;q[...,tangential[0]]=aa;q[...,tangential[1]]=bb
        flat=q.reshape(-1,3);world=chart.position(flat)
        e,h=dipole_fields(world,frequency=frequency,position=chart.origin,direction=direction,magnetic=magnetic)
        normal=torch.zeros_like(flat);normal[:,normal_axis]=1 if face.normal_direction=='+' else -1
        area_vector=chart.physical_area_vector(flat,normal)
        physical_flux+=(.5*torch.cross(e,h.conj(),dim=-1).real*area_vector).sum()*float(widths[0][0]*widths[1][0])
        ce=chart.pullback_field(flat,e).reshape(count,count,3);ch=chart.pullback_field(flat,h).reshape(count,count,3)
        payload=dict(kind='plane',monitor_type='plane',fields=face.fields,axis=face.axis,position=face.plane_position,yee_plane_position=plane,
            frequency=frequency,frequencies=(frequency,),compute_flux=True,normal_direction=face.normal_direction,
            coords=tuple(vectors),cell_widths=dict(zip(('xyz'[axis] for axis in tangential),widths)),components={})
        payload.update(zip(('xyz'[axis] for axis in tangential),vectors))
        for component in face.fields:
            array=(ce if component[0]=='E' else ch)[...,'xyz'.index(component[1].lower())]
            payload[component]=array;payload['components'][component]=dict(data=array,coords=tuple(vectors))
        samples[face.name]=payload
    result=mw.Result(method='fdtd',scene=scene,frequency=frequency,monitors=samples)
    return result,direction,physical_flux
