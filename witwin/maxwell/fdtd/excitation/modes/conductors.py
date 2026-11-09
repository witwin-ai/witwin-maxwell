"""Explicit good-conductor perturbation around a longitudinal PEC-limit mode."""
from __future__ import annotations

import cmath
from copy import copy
import math

import torch

from ....constants import C_0, ETA_0
from ....compiler.copper_foil import compile_copper_faces, copper_edge_admittance_scale
from ....compiler.materials import evaluate_material_components
from .common import _discrete_mode_profile_power, _normalize_mode_profiles_to_unit_power


def _reject_curved_modal_aperture(scene, source):
    from ....compiler.cut_surface import clip_triangles
    for structure in scene.structures:
        if (not getattr(structure,'enabled',True)
                or not getattr(structure.material,'is_copper_foil',False)
                or getattr(structure.geometry,'kind',None)!='surface_mesh'):
            continue
        vertices,faces=structure.geometry.to_mesh()
        triangles=vertices[faces].to(dtype=torch.float64)
        center=triangles.new_tensor(source['position'])
        half=triangles.new_tensor(source['size'])/2
        # Clipping a zero-normal aperture must retain true plane/edge contact
        # despite only arithmetic roundoff; this bound is tied to source scale.
        scale=triangles.abs().max()+center.abs().max()+half.abs().max()
        tolerance=16*torch.finfo(torch.float64).eps*scale
        lows=(center-half-tolerance).expand(len(triangles),3)
        highs=(center+half+tolerance).expand(len(triangles),3)
        _polygon,counts=clip_triangles(triangles,lows,highs)
        if bool((counts>0).any()):
            raise NotImplementedError('SurfaceMesh CopperFoil modal apertures require a curved finite-Et mode solve; good_conductor supports longitudinal planar foils only.')


def prepare_good_conductor_context(solver, source):
    """Return a private modal boundary model; never alter the physical foil model."""
    if source.get('wave_family') != 'tem':
        raise NotImplementedError("good_conductor is supported only for TEM/quasi-TEM WavePort modes.")
    if getattr(solver,'_mode_source_rebuild_from_fields',False):
        raise NotImplementedError("Good-conductor mode-shape gradients are unsupported.")
    scene=solver.scene
    _reject_curved_modal_aperture(scene,source)
    propagation='xyz'.index(source['normal_axis'])
    nodes=getattr(scene,source['normal_axis']+'_nodes64')
    plane=min(range(len(nodes)),key=lambda index:abs(nodes[index]-source['position'][propagation]))
    faces=compile_copper_faces(scene,device=solver.Ex.device)
    model=dict(solver._compiled_material_model)
    edges=dict(model.get('pec_sheet_edges',{}))
    key='E'+source['normal_axis']
    if model.get('pec_occupancy') is None:
        model['pec_occupancy']=torch.zeros((scene.Nx,scene.Ny,scene.Nz),device=solver.Ex.device)
    shape=list(model['pec_occupancy'].shape)
    shape[propagation]-=1
    mask=torch.zeros(shape,device=solver.Ex.device,dtype=torch.float32)
    accepted=[]
    for face in faces:
        if face.axis==propagation:
            if abs(face.normal_index-plane)<=1:
                intersects=torch.ones(len(face.indices),device=face.indices.device,dtype=torch.bool)
                for axis in range(3):
                    if axis==propagation:
                        continue
                    coordinate_key='xyz'[axis]+('_half64' if axis==face.component else '_nodes64')
                    values=torch.as_tensor(getattr(scene,coordinate_key),device=face.indices.device)[face.indices[:,axis]]
                    center,span=source['position'][axis],source['size'][axis]
                    intersects &= (values>=center-span/2)&(values<=center+span/2)
                if bool(intersects.any()):
                    raise NotImplementedError('A good-conductor mode requires longitudinal foils throughout its source/probe stencil.')
            continue
        if face.component!=propagation:
            continue
        transverse=[axis for axis in range(3) if axis!=propagation]
        selected=torch.ones(len(face.indices),device=face.indices.device,dtype=torch.bool)
        for axis in transverse:
            values=torch.as_tensor(getattr(scene,'xyz'[axis]+'_nodes64'),device=face.indices.device)[face.indices[:,axis]]
            center,span=source['position'][axis],source['size'][axis]
            selected &= (values>=center-span/2-1e-12)&(values<=center+span/2+1e-12)
        low=face.indices[selected&(face.indices[:,propagation]==plane-1)]
        high=face.indices[selected&(face.indices[:,propagation]==plane)]
        if not len(low) and not len(high):
            continue
        if low.shape!=high.shape or not torch.equal(low[:,transverse],high[:,transverse]):
            raise NotImplementedError('A good-conductor mode cannot cross a foil end or axial discontinuity.')
        if any(torch.is_tensor(value) and value.requires_grad for value in (face.material.thickness,face.material.conductivity)):
            raise NotImplementedError('Good-conductor mode profiles require fixed foil material parameters.')
        mask[tuple(face.indices.T)]=1
        accepted.append((face,high[:,transverse]))
    if not accepted:
        raise ValueError("conductor_model='good_conductor' requires a longitudinal CopperFoil in the aperture.")
    edges[key]=torch.maximum(edges.get(key,torch.zeros_like(mask)),mask)
    model['pec_sheet_edges']=edges
    proxy=copy(solver)
    proxy._compiled_material_model=model
    proxy._good_conductor_faces=accepted
    proxy._good_conductor_all_faces=faces
    # Free-edge contacts must see the physical conductors, not the private
    # PEC-limit modal boundary that turns the foil traces into sheets.
    proxy._good_conductor_physical_model=solver._compiled_material_model
    proxy._good_conductor_context=True
    return proxy


def apply_good_conductor_perturbation(solver, source, data):
    """Apply the exact finite-film impedance to the PEC-limit surface currents.

    The launch and measurement transverse fields retain their PEC-limit boundary
    shape. The complex propagation and V/I receive the first-order series-
    impedance correction. This is an explicit asymptotic model, not a finite-Et
    modal boundary solve.
    """
    from . import sample_mode_source_component
    frequency=float(source['source_time']['frequency'])
    dt=getattr(solver,'dt',None)
    if dt is None:
        dt=solver.resolve_mode_time_step(frequency)
    axis='xyz'.index(data['normal_axis'])
    transverse=[index for index in range(3) if index!=axis]
    scene=solver.scene
    nodes=getattr(scene,data['normal_axis']+'_nodes64')
    plane=data['plane_index']
    spacing=float(nodes[plane+1]-nodes[plane])
    if not math.isclose(spacing,float(nodes[plane]-nodes[plane-1]),rel_tol=1e-7):
        raise NotImplementedError('Good-conductor mode source/probe planes require locally uniform axial spacing.')
    beta0=data.get('beta_complex')
    if beta0 is None:
        k0=2*math.pi*frequency/C_0
        beta0=2/spacing*cmath.asin(spacing/(C_0*dt)*math.sin(math.pi*frequency*dt)*data['beta']/k0)
    power_args=dict(coords_u=data['coords_u'],coords_v=data['coords_v'],normal_axis=data['normal_axis'],staggered=True)
    power=abs(float(_discrete_mode_profile_power(data['component_profiles'],**power_args)))
    work=torch.zeros((),device=solver.Ex.device,dtype=torch.complex128)
    records=[]
    eps, mu = evaluate_material_components(solver._compiled_material_model,frequency)
    medium_impedance = min(float(torch.sqrt(torch.abs(mu[key]/eps[key])).min())*ETA_0 for key in 'xyz')
    for face,indices in solver._good_conductor_faces:
        material=face.material
        minimum_frequency=material.frequency_range[0]
        if not material.frequency_range[0]<=frequency<=material.frequency_range[1]:
            raise ValueError('Good-conductor modal frequency lies outside the declared CopperFoil band.')
        delta=float(material.skin_depth((minimum_frequency,))[0])
        thickness_ratio=float(material.thickness)/delta
        if thickness_ratio<3.:
            raise ValueError('good_conductor requires thickness/skin_depth >= 3 throughout the foil band; resistive ultrathin sheets need a finite-Et mode solve.')
        minimum_spacing=min(float(torch.diff(data[key]).min()) for key in ('coords_u','coords_v'))
        skin_ratio=delta/minimum_spacing
        if skin_ratio>.03:
            raise ValueError('good_conductor requires skin_depth/min_transverse_grid_spacing <= 0.03.')
        impedance=material.surface_impedance((frequency,))[0].to(device=solver.Ex.device)
        # Also constrain the local medium contrast; the series correction below
        # independently bounds the geometry-dependent perturbation.
        impedance_ratio=float(torch.linalg.matrix_norm(impedance,ord=2))/medium_impedance
        if impedance_ratio>1e-3:
            raise ValueError('good_conductor requires two-face surface impedance/local medium impedance <= 1e-3.')
        tangent=next(index for index in range(3) if index not in (axis,face.axis))
        local=transverse.index(tangent)
        tangent_indices=torch.unique(indices[:,local],sorted=True)
        coordinates=torch.as_tensor(getattr(scene,'xyz'[tangent]+'_nodes64'),device=solver.Ex.device)[tangent_indices]
        if len(coordinates)<2:
            raise ValueError('Good-conductor foil cross-section requires at least two tangential nodes.')
        low_coordinate=float(getattr(scene,'xyz'[face.axis]+'_half64')[face.normal_index-1])
        high_coordinate=float(getattr(scene,'xyz'[face.axis]+'_half64')[face.normal_index])
        points=torch.zeros((len(coordinates),3),device=solver.Ex.device,dtype=torch.float64)
        points[:,axis]=source['position'][axis]
        points[:,tangent]=coordinates
        points[:,face.axis]=low_coordinate
        left=sample_mode_source_component(data,points,'H'+'xyz'[tangent]).to(torch.complex128)
        points[:,face.axis]=high_coordinate
        right=sample_mode_source_component(data,points,'H'+'xyz'[tangent]).to(torch.complex128)
        # The split Ampere dual face includes the normal-H curl over each
        # adjacent half cell. These native samples extrapolate the exterior
        # tangential-H values to the foil trace, including edge-row currents.
        # They are the same PEC-limit forcing used by advance_copper_electric.
        normal_nodes=getattr(scene,'xyz'[face.axis]+'_nodes64')
        tangent_half=torch.as_tensor(getattr(scene,'xyz'[tangent]+'_half64'),device=solver.Ex.device)
        points[:,face.axis]=float(normal_nodes[face.normal_index])
        lower=(tangent_indices-1).clamp(0,len(tangent_half)-1)
        upper=tangent_indices.clamp(0,len(tangent_half)-1)
        points[:,tangent]=tangent_half[lower]
        normal_low=sample_mode_source_component(data,points,'H'+'xyz'[face.axis]).to(torch.complex128)
        normal_low=torch.where(tangent_indices>0,normal_low,0.)
        points[:,tangent]=tangent_half[upper]
        normal_high=sample_mode_source_component(data,points,'H'+'xyz'[face.axis]).to(torch.complex128)
        normal_high=torch.where(tangent_indices<len(tangent_half),normal_high,0.)
        tangent_dual=torch.as_tensor(getattr(scene,'d'+'xyz'[tangent]+'_dual64'),device=solver.Ex.device)[tangent_indices]
        normal_derivative=(normal_high-normal_low)/tangent_dual
        normal_primal=getattr(scene,'d'+'xyz'[face.axis]+'_primal64')
        left_width=float(normal_primal[face.normal_index-1])/2
        right_width=float(normal_primal[face.normal_index])/2
        current=torch.stack((left+left_width*normal_derivative,
                             -right+right_width*normal_derivative),dim=-1)
        density=.5*torch.einsum('mi,ij,mj->m',current.conj(),impedance,current)
        # Match the passive forward operator's full Yee dual face at interior
        # foil-edge nodes. Trapezoidal half-end weights would undercount its
        # strongest edge currents on a coarse transverse mesh.
        widths=torch.as_tensor(getattr(scene,'d'+'xyz'[tangent]+'_dual64'),device=solver.Ex.device)[tangent_indices]
        endpoint=(tangent_indices==0)|(tangent_indices==len(getattr(scene,'xyz'[tangent]+'_nodes64'))-1)
        widths=widths*torch.where(endpoint,.5,1.)
        # The same free-edge impedance factor as the forward foil update, on
        # the cross-section traces of the high modal plane in tangential order.
        traces=torch.zeros((len(indices),3),dtype=indices.dtype,device=indices.device)
        traces[:,axis]=data['plane_index']
        traces[:,transverse]=indices
        order=torch.argsort(traces[:,tangent])
        if not torch.equal(traces[order,tangent],tangent_indices):
            raise RuntimeError('Good-conductor cross-section traces must have unique tangential nodes.')
        contact=None
        if all(getattr(solver,'c'+name.lower()+'_curl',None) is not None for name in ('Ex','Ey','Ez')):
            from ...copper_foil import copper_contact_masks
            physical=copy(solver)
            physical._compiled_material_model=getattr(solver,'_good_conductor_physical_model',solver._compiled_material_model)
            contact=copper_contact_masks(physical,frequency=material.frequency_range[1],attachments=False)
        scale,_geometry=copper_edge_admittance_scale(scene,face,traces[order],
            getattr(solver,'_good_conductor_all_faces',(face,)),contact=contact)
        density=density/scale.to(density.device)
        contribution=torch.sum(density*widths)
        work=work+contribution
        records.append(dict(name=face.name,frequency_range_hz=material.frequency_range,
            thickness_skin_depth_ratio=thickness_ratio,
            skin_depth_grid_ratio=skin_ratio,surface_impedance_ratio=impedance_ratio,
            quadrature="native Yee dual surface metric",
            edge_impedance_factor=(1/scale).cpu().tolist(),
            complex_surface_work_per_m=complex(contribution)))
    correction=.5j*complex(work)/power
    relative=abs(correction/beta0)
    if relative>.01 or correction.imag<0:
        raise ValueError('Good-conductor first-order relative propagation correction must be passive and <= 0.01.')
    beta=beta0+correction
    impedance_factor=beta/beta0
    profiles={name:value.to(torch.complex128)/(impedance_factor if name.startswith('H') else 1.)
              for name,value in data['component_profiles'].items()}
    profiles=_normalize_mode_profiles_to_unit_power(profiles,**power_args)
    data=dict(data,component_profiles=profiles,profile=profiles[data['profile_component']],
        beta=beta.real,beta_complex=beta,beta_tensor=torch.tensor(beta,device=solver.Ex.device,dtype=torch.complex128),
        effective_index=beta.real*C_0/(2*math.pi*frequency),effective_index_complex=beta*C_0/(2*math.pi*frequency))
    data['conductor_approximation']=dict(model='good_conductor',launch='PEC-limit tangential Et=0',
        projection='PEC-limit transverse shape with first-order series impedance',
        frequency_hz=frequency,minimum_local_medium_impedance_ohm=medium_impedance,
        base_beta=beta0,propagation_correction=correction,
        relative_propagation_correction=relative,estimated_second_order_indicator=relative**2,
        launch_field_relative_indicator=max(relative,max(record['surface_impedance_ratio'] for record in records)),
        applicability='t/delta>=3; delta/grid<=0.03; ||Ztwoface||/eta_min<=1e-3; |delta_beta/beta|<=0.01',
        total_grid_error_bounded=False,foils=records)
    return data


def copper_intersects_mode(scene, source):
    """Detect physical foil crossings without assigning any ideal ownership."""
    _reject_curved_modal_aperture(scene,source)
    propagation='xyz'.index(source['normal_axis'])
    for structure in scene.structures:
        if not getattr(structure,'enabled',True) or not getattr(structure.material,'is_copper_foil',False):
            continue
        geometry=structure.geometry
        if not hasattr(geometry,'size'):
            continue
        center=geometry.position.detach().cpu().double().tolist()
        size=geometry.size.detach().cpu().double().tolist()
        if not center[propagation]-size[propagation]/2<=source['position'][propagation]<=center[propagation]+size[propagation]/2:
            continue
        if all(center[axis]+size[axis]/2>=source['position'][axis]-source['size'][axis]/2
               and center[axis]-size[axis]/2<=source['position'][axis]+source['size'][axis]/2
               for axis in range(3) if axis!=propagation):
            return True
    return False
