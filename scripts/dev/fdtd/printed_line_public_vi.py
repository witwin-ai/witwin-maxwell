"""Fixed physical line V/I using only public selected Cu queries at runtime.

Ordinary native Points and signed SI path coefficients are retained. This
benchmark adapter is not a solver or a second copper reconstruction backend.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.postprocess.yee_interpolation import interpolation_stencil
from scripts.dev.fdtd.printed_line_fixed_vi import fixed_physical_probe_monitors


@dataclass(frozen=True)
class PublicCuTerm:
    component: str
    owner: str
    normal_axis: str
    normal_index: int
    normal_coordinate: float
    points_m: tuple
    region_id: str


@dataclass(frozen=True)
class PublicVIPlan:
    monitors: tuple
    rows: tuple
    expected_points: dict
    queries: dict
    plane_names: dict
    native_rows: dict
    frequencies: tuple
    contract: dict


def _region(scene, points):
    q=torch.tensor(points,dtype=torch.float64)
    inside=torch.zeros(len(q),dtype=torch.bool)
    for structure in scene.structures:
        if structure.enabled:
            inside |= structure.geometry.signed_distance(*q.unbind(-1)) < 0
    if not bool((inside==inside[0]).all()):
        raise ValueError('A selected request cannot mix interior and exterior regions.')
    return 'interior' if bool(inside[0]) else 'exterior'


def _owned_row(prepared, faces, component, point):
    if component not in ('Ex','Ey','Ez'):
        return None
    axes=[torch.as_tensor(a,dtype=torch.float64)for a in component_coords(prepared,component)]
    candidates=[f for f in faces if component=='E'+'xyz'[f.component]
        and point[f.axis]==float(axes[f.axis][f.normal_index])]
    if not candidates:
        return None
    # Pure CPU geometry only: no face fields/clock exist before execution.
    # Exact tangential points use the shared ordinary native corner planner;
    # membership is then checked against the actual compiled owner support.
    # The runtime still calls public query_fields with real faces/heads/clock.
    stencil=interpolation_stencil(SimpleNamespace(prepared_scene=prepared),
        torch.tensor([point],dtype=torch.float64),component,axes,faces=[])
    entries=tuple(sorted((tuple(int(i)for i in indices[0]),float(weights[0]))
        for indices,weights,_ in stencil if float(weights[0])!=0))
    matches=[]
    for candidate in candidates:
        support=set(map(tuple,candidate.indices.tolist()))
        membership=[index in support for index,_ in entries]
        if not any(membership):
            continue
        if not all(membership):
            raise ValueError('A fixed Cu term crosses an owner footprint, gap, or contact.')
        matches.append(candidate)
    if not matches:
        return None
    if len(matches)!=1:
        raise ValueError('A physical Cu term has ambiguous compiled ownership.')
    face=matches[0]
    bounds=np.asarray(prepared._public_scene.domain.bounds)
    for index,_ in entries:
        native=np.asarray([float(a[i])for a,i in zip(axes,index)])
        if not np.all((native>bounds[:,0])&(native<bounds[:,1])):
            raise ValueError('A fixed Cu support touches PML or its physical interface.')
    return face,tuple(sorted(entries))


def select_frequencies(plan, frequencies=None):
    selected=plan.frequencies if frequencies is None else tuple(frequencies)
    if not selected or len(set(selected))!=len(selected) or any(isinstance(f,bool)or not isinstance(f,(int,float))or not math.isfinite(f)or f not in plan.frequencies for f in selected):
        raise ValueError('Select unique recorded physical frequencies in output order.')
    return selected


def query_kwargs(term, frequencies, *, side='negative', representation='published_native'):
    if side not in ('negative','positive')or representation not in ('published_native','constitutive'):
        raise ValueError('Use an explicit physical Cu side and representation.')
    if representation=='published_native'and side!='negative':
        raise ValueError('The published primitive has only the negative side.')
    return dict(region_id=term.region_id,field_kind='total',components=(term.component,),
        frequencies=tuple(frequencies),owner=term.owner,trace_side=side,representation=representation)


def build_public_vi(scene,reference_scene,reference_monitors,reference_rows,frequencies):
    """Plan the same physical functional through shared native Yee stencils."""
    if scene.coordinates is not None or any(scene.boundary.axis_kind(a)in('bloch','periodic')for a in 'xyz'):
        raise ValueError('This bounded line adapter requires Cartesian nonperiodic geometry.')
    prepared=prepare_scene(scene);faces=compile_copper_faces(prepared,device='cpu')
    byname={m.name:m for m in reference_monitors}
    requested={(name,c)for pair in reference_rows for row in pair for name,c,_ in row}
    typed={};queries={};native_rows={};plane_names={};planes={}
    for name,component in sorted(requested):
        point=tuple(byname[name].position);owned=_owned_row(prepared,faces,component,point)
        if owned is None:
            continue
        face,row=owned;virtual='negative_Cu_'+str(len(queries))
        normal='xyz'[face.axis];coordinate=float(getattr(prepared,normal+'_nodes64')[face.normal_index])
        term=PublicCuTerm(component,face.name,normal,int(face.normal_index),coordinate,(point,),_region(scene,(point,)))
        key=(normal,coordinate,component)
        if key not in planes:
            planes[key]=mw.PlaneMonitor('published_Cu_plane_'+str(len(planes)),axis=normal,
                position=coordinate,fields=(component,),frequencies=tuple(frequencies))
        typed[name,component]=virtual;queries[virtual]=term;native_rows[virtual]=row;plane_names[virtual]=planes[key].name
    ordinary=sorted(requested-set(typed));ordinary_rows=[([(n,c,1.)],[])for n,c in ordinary]
    used={name for name,_ in ordinary}
    monitors,expanded,expected,ordinary_contract=fixed_physical_probe_monitors(scene,reference_scene,
        [m for m in reference_monitors if m.name in used],ordinary_rows)
    ordinary_terms={key:pair[0]for key,pair in zip(ordinary,expanded)}
    rows=[];physical=[]
    for pair in reference_rows:
        station=[]
        for number,row in enumerate(pair):
            terms=[]
            for name,c,weight in row:
                if not math.isfinite(weight):
                    raise ValueError('Physical integral coefficients must be finite.')
                if (name,c)in typed:
                    terms.append((typed[name,c],c,float(weight)))
                else:
                    terms.extend((n,cc,float(weight)*w)for n,cc,w in ordinary_terms[name,c])
                physical.append(dict(kind='voltage'if number==0 else'current',component=c,
                    point_m=byname[name].position,weight_SI=float(weight),
                    primitive='public_Cu_negative_published_native'if(name,c)in typed else'original_strict_Point'))
            station.append(tuple(terms))
        rows.append(tuple(station))
    contract=dict(physical_quadrature=physical,ordinary=ordinary_contract,primary_Cu_side='negative',
        primary_Cu_representation='published_native',secondary_Cu=('constitutive_negative','constitutive_positive'),
        physical_curve_or_source_modified=False,typed_Cu_primitive_count=len(queries),
        public_clock_policy='Actual complete current producer only; partial64ns prefix and old740 records reject.',
        scope='Fixed physical signed VI; ordinary Points/H contours/source apertures/materials/feeds unchanged.')
    plan=PublicVIPlan(tuple([*monitors,*planes.values()]),tuple(rows),expected,queries,plane_names,native_rows,tuple(frequencies),contract)
    select_frequencies(plan)
    return plan


def assemble_state(plan, values, frequencies):
    """Linear complex SI integration only; no power normalization or fit."""
    needed={n for pair in plan.rows for row in pair for n,_,_ in row}
    if set(values)!=needed:
        raise ValueError('The exact complete physical sample set is required.')
    prototype=next(iter(values.values()));zero=torch.zeros(len(frequencies),dtype=torch.complex128,device=prototype.device)
    for value in values.values():
        if not isinstance(value,torch.Tensor)or not value.is_complex()or value.shape!=(len(frequencies),)or not bool(torch.isfinite(value).all()):
            raise ValueError('Every physical sample must be a finite complex ordered spectrum.')
    return torch.stack([torch.stack([sum((values[n].to(torch.complex128)*w for n,_,w in row),zero)for row in pair])for pair in plan.rows])


def evaluate_public_vi(plan, *, result, frequencies=None, device='cuda'):
    """Query six Cu terms publicly; retain both true constitutive side heads."""
    selected=select_frequencies(plan,frequencies)
    if torch.device(device).type!='cuda':
        raise ValueError('Executable public field queries require CUDA; CPU planning is separate.')
    values={};evidence={}
    for name,term in plan.queries.items():
        points=torch.tensor(term.points_m,dtype=torch.float64,device=device)
        primary=result.query_fields(points,**query_kwargs(term,selected))[term.component]
        negative=result.query_fields(points,**query_kwargs(term,selected,representation='constitutive'))[term.component]
        positive=result.query_fields(points,**query_kwargs(term,selected,side='positive',representation='constitutive'))[term.component]
        if primary.shape!=(len(selected),1)or negative.shape!=primary.shape or positive.shape!=primary.shape:
            raise ValueError('Selected Cu output must retain frequency and point axes.')
        if negative.dtype!=torch.complex128 or positive.dtype!=torch.complex128:
            raise ValueError('Constitutive Cu side heads must retain complex128 precision.')
        values[name]=primary[:,0]
        evidence[name]=dict(primary=primary,constitutive_negative=negative,constitutive_positive=positive,
            kwargs=query_kwargs(term,selected),points=points,native_support=plan.native_rows[name])
    for name,(component,index)in plan.expected_points.items():
        payload=result.monitor(name)
        if tuple(payload['field_indices'][component])!=tuple(index):
            raise ValueError('An ordinary Point changed its actual native index.')
        axis=tuple(payload['frequencies']);indices=[axis.index(f)for f in selected]
        value=payload['components'][component].reshape(-1)
        values[name]=value[indices].to(device=device,dtype=torch.complex128)
    return assemble_state(plan,values,selected),evidence


def validate_plan_support(plan, scene):
    """CPU admission rechecks owner/region/support, never inventing spectra."""
    if len(plan.queries)!=6 or plan.contract['typed_Cu_primitive_count']!=6:
        raise ValueError('The bounded CPW/pair functional requires six selected Cu terms.')
    prepared=prepare_scene(scene);faces=compile_copper_faces(prepared,device='cpu')
    for name,term in plan.queries.items():
        owned=_owned_row(prepared,faces,term.component,term.points_m[0])
        if owned is None:
            raise ValueError('The declared physical Cu term has no actual native owner.')
        face,row=owned
        if (face.name!=term.owner or 'xyz'[face.axis]!=term.normal_axis or face.normal_index!=term.normal_index
            or float(getattr(prepared,term.normal_axis+'_nodes64')[face.normal_index])!=term.normal_coordinate
            or row!=tuple(sorted(plan.native_rows[name])) or _region(scene,term.points_m)!=term.region_id):
            raise ValueError('Actual Cu owner, native support, normal or strict region changed.')
    return True
