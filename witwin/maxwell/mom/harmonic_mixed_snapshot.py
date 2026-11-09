"""Explicit mixed-DC numerical records and decode-only operator restoration."""
import torch
from .harmonic_current_snapshot import encode as encode_current,restore as restore_current
from .harmonic_mixed import MixedHarmonicClosure
from .static_mixed import MixedStaticOperator
from .quadrature import triangle_rule
from ..constants import EPSILON_0

OPERATOR_ARRAYS=('exterior','interiors','indicator','current_faces','contrast','charge_test',
                 'lengths','matrix','lu','pivots','nodes','weights')
MAP_ARRAYS=('electric_response','surface_response','offset_response')


def encode(static):
    closure=static['current_closure']
    if not isinstance(closure,MixedHarmonicClosure):
        raise TypeError('Mixed records require a mixed harmonic closure')
    closure.assert_current()
    result=encode_current({**{k:static[k] for k in ('material_channels','magnetic_permeabilities','points','receipt')},
                           'current_closure':closure.current,'floating_operator':static['floating_operator']})
    result['mixed_operator']={**{name:getattr(closure.operator,name) for name in OPERATOR_ARRAYS},
                              'current_regions':closure.operator.current_regions}
    result['mixed_closure']={'regions':tuple(b.volume.region_id for b in closure.bindings),
        'counts':closure.counts,'current_indices':closure.current_indices,
        'dielectric_indices':closure.dielectric_indices,'maps':closure.maps}
    return result


def restore(static,bindings,materials,testing_order):
    """Restore validated arrays without matrix assembly, factorization or solve."""
    bindings=tuple(bindings)
    flags=tuple(m.tpa_scale!=0 or m.electric_dc.kind=='conductive_limit' for m in materials)
    current_regions=tuple(i+1 for i,flag in enumerate(flags) if flag)
    if not current_regions or all(flags):
        raise ValueError('Mixed snapshot requires dielectric and current bodies')
    selected=tuple(b for b in bindings if b.volume.region_id in current_regions)
    dielectric=tuple(b for b in bindings if b.volume.region_id not in current_regions)
    if set(b.volume.region_id for b in selected)!=set(current_regions) or not dielectric:
        raise ValueError('Mixed snapshot current/dielectric volume mapping mismatch')
    raw=static.get('mixed_operator')
    layout=static.get('mixed_closure')
    if not isinstance(raw,dict) or set(raw)!={*OPERATOR_ARRAYS,'current_regions'} or raw['current_regions']!=current_regions:
        raise ValueError('Invalid mixed operator snapshot fields or regions')
    if not isinstance(layout,dict) or set(layout)!={'regions','counts','current_indices','dielectric_indices','maps'}:
        raise ValueError('Invalid mixed closure snapshot fields')
    counts=tuple(len(b.volume.cells) for b in bindings)
    if layout['regions']!=tuple(b.volume.region_id for b in bindings) or layout['counts']!=counts:
        raise ValueError('Mixed snapshot volume order mismatch')
    restored=dict(static)
    restore_current(restored,selected,materials,testing_order)
    current,laplace=restored['current_closure'],restored['floating_operator']
    sample=laplace.triangles
    def array(value,shape,dtype=None):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=sample.device
                or value.dtype!=(sample.dtype if dtype is None else dtype) or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid mixed snapshot numerical array')
    def close(actual,expected):
        if actual.shape!=expected.shape or not bool(torch.all(
                (actual-expected).abs()<=512*torch.finfo(sample.dtype).eps*expected.abs().max())):
            raise ValueError('Inconsistent mixed snapshot operator or mapping')
    n,c=len(sample),len(current_regions)
    selected_ids=torch.tensor([r-1 for r in current_regions],device=sample.device)
    nodes,weights=triangle_rule(sample,testing_order)
    weights=weights/weights.sum(-1,keepdim=True)
    shapes={'exterior':(),'indicator':(n,c),'current_faces':(n,),'contrast':(n,),
        'charge_test':(c,n),'lengths':(c,),'matrix':(n+c,n+c),'lu':(n+c,n+c),
        'pivots':(n+c,),'nodes':nodes.shape,'weights':weights.shape}
    for name,shape in shapes.items():
        array(raw[name],shape,torch.int32 if name=='pivots' else torch.bool if name=='current_faces' else None)
    if not isinstance(raw['interiors'],(tuple,list)) or len(raw['interiors'])!=len(materials):
        raise ValueError('Invalid mixed snapshot interior samples')
    close(raw['exterior'],sample.new_tensor(EPSILON_0))
    for value,material,flag in zip(raw['interiors'],materials,flags):
        array(value,())
        expected=EPSILON_0 if flag else material.electric_dc.dielectric_sample(EPSILON_0)
        close(value,sample.new_tensor(expected))
    indicator=laplace.indicator[:,selected_ids]
    close(raw['indicator'],indicator)
    if not torch.equal(raw['current_faces'],indicator.sum(-1).bool()):
        raise ValueError('Mixed snapshot current face mask mismatch')
    face_epsilon=torch.cat([value.expand(size) for value,size in zip(raw['interiors'],laplace.body_faces)])
    contrast=2*(face_epsilon-raw['exterior'])/(face_epsilon+raw['exterior'])
    close(raw['contrast'],contrast)
    close(raw['charge_test'],laplace.charge_test[selected_ids])
    close(raw['lengths'],laplace.lengths[selected_ids])
    close(raw['nodes'],nodes)
    close(raw['weights'],weights)
    normal=torch.einsum('qti,qi->qt',laplace.mean_gradient,laplace.normals)
    dielectric_matrix=torch.eye(n,device=sample.device,dtype=sample.dtype)+contrast[:,None]*normal
    surface=torch.where(raw['current_faces'][:,None],laplace.single_layer,dielectric_matrix)
    matrix=torch.cat((torch.cat((surface,-indicator),1),torch.cat((raw['charge_test']/raw['lengths'][:,None],sample.new_zeros(c,c)),1)))
    close(raw['matrix'],matrix)
    if bool(((raw['pivots']<1)|(raw['pivots']>n+c)).any()):
        raise ValueError('Invalid mixed snapshot LU pivots')
    permutation,lower,upper=torch.lu_unpack(raw['lu'],raw['pivots'])
    close(permutation@lower@upper,matrix)
    operator=MixedStaticOperator.__new__(MixedStaticOperator)
    operator.laplace,operator.current_regions=laplace,current_regions
    for name in OPERATOR_ARRAYS:
        setattr(operator,name,tuple(raw[name]) if name=='interiors' else raw[name])
    operator._versions=operator._state()
    groups=[[],[]]
    offset=0
    for binding,count in zip(bindings,counts):
        groups[0 if binding.volume.region_id in current_regions else 1].append(torch.arange(offset,offset+count,device=sample.device))
        offset+=count
    for name,parts in zip(('current_indices','dielectric_indices'),groups):
        expected=torch.cat(parts)
        array(layout[name],expected.shape,torch.int64)
        if not torch.equal(layout[name],expected):
            raise ValueError('Mixed snapshot cell index order mismatch')
    maps=layout['maps']
    if not isinstance(maps,dict) or set(maps)!={*MAP_ARRAYS,'polarization_columns','current_face_counts','source_regions','current_regions','receipt'}:
        raise ValueError('Invalid mixed snapshot response map fields')
    columns=3*sum(len(b.volume.cells) for b in dielectric)
    face_counts=tuple(laplace.body_faces[r-1] for r in current_regions)
    if (maps['polarization_columns']!=columns or maps['current_face_counts']!=face_counts
            or maps['source_regions']!=tuple(b.volume.region_id for b in dielectric) or maps['current_regions']!=current_regions):
        raise ValueError('Mixed snapshot response column order mismatch')
    for name,rows in zip(MAP_ARRAYS,(columns,n,c)):
        array(maps[name],(rows,columns+sum(face_counts)))
    if (not isinstance(maps['receipt'],dict) or maps['receipt'].get('physical_qualification') is not False
            or maps['receipt'].get('neutral_current_bodies') is not True):
        raise ValueError('Invalid mixed response qualification receipt')
    roundoff=512*torch.finfo(sample.dtype).eps
    surface_map=maps['surface_response']
    if not bool(((operator.charge_test@surface_map).abs()<=roundoff*(operator.charge_test.abs()@surface_map.abs())).all()):
        raise ValueError('Mixed snapshot response violates neutral current charge')
    boundary_map=torch.cat((surface_map[:,columns:],maps['offset_response'][:,columns:]))
    rhs=torch.cat((torch.eye(n,device=sample.device,dtype=sample.dtype)[:,operator.current_faces],
                   sample.new_zeros(c,sum(face_counts))))
    if not bool(((matrix@boundary_map-rhs).abs()<=roundoff*(matrix.abs()@boundary_map.abs()+rhs.abs())).all()):
        raise ValueError('Mixed snapshot boundary response violates its operator')
    # Check every input direction, including those orthogonal to the saved state.
    # Decode the stored density maps; never solve or rebuild an interface matrix.
    from .static_coupling import StaticVolumeFieldResponse
    counts_dielectric=tuple(len(b.volume.cells) for b in dielectric)
    def decode_maps(value):
        polarization=value[:columns].reshape(-1,3).split(counts_dielectric)
        decoder=StaticVolumeFieldResponse(laplace.system,operator.exterior,operator.interiors,
            dielectric,polarization,surface_map@value,testing_order=testing_order)
        means=[]
        for points,normal,weight in zip(operator.nodes,laplace.normals,operator.weights):
            potential,electric,jump=decoder._particular(points,principal_value=True,
                normals=normal.expand_as(points))
            means.append(torch.stack(((potential*weight).sum(),
                ((electric*normal).sum(-1)*weight).sum()+(jump*weight).sum()/2)))
        means=torch.stack(means)
        boundary=sample.new_zeros(n).masked_scatter(operator.current_faces,value[columns:])
        right=torch.cat((torch.where(operator.current_faces,boundary-means[:,0],
            operator.contrast*means[:,1]),sample.new_zeros(c)))
        electric=torch.cat([decoder.fields(b.volume.vertices[b.volume.cells].mean(1),
            b.volume.region_id)[1].flatten() for b in dielectric])
        return torch.cat((right,electric))
    decoded=torch.autograd.functional.jacobian(decode_maps,sample.new_zeros(columns+sum(face_counts)),
        vectorize=False)
    all_maps=torch.cat((surface_map,maps['offset_response']))
    expected_rhs=decoded[:n+c]
    if not bool(((matrix@all_maps-expected_rhs).abs()<=roundoff*(
            matrix.abs()@all_maps.abs()+expected_rhs.abs())).all()):
        raise ValueError('Mixed snapshot polarization response violates its operator')
    close(maps['electric_response'],decoded[n+c:])
    closure=MixedHarmonicClosure.__new__(MixedHarmonicClosure)
    closure.operator,closure.bindings=operator,bindings
    closure.current_bindings,closure.dielectric_bindings=selected,dielectric
    closure.volumes,closure.counts=tuple(b.volume for b in bindings),counts
    closure.current,closure.maps=current,maps
    closure.current_indices,closure.dielectric_indices=layout['current_indices'],layout['dielectric_indices']
    closure._versions=closure._state()
    static.pop('mixed_closure')
    static.update(current_closure=closure,floating_operator=laplace,mixed_operator=operator)
    return static


def validate_state(static,state,bindings,threshold):
    """Validate actual mixed electric/current/magnetic equations without solves."""
    import math
    from .harmonic_current import surface_face_potentials
    from .static_coupling import StaticVolumeFieldResponse
    from .harmonic_current_snapshot import validate_magnetic
    closure=static['current_closure']
    operator=closure.operator
    laplace=operator.laplace
    regions=tuple(b.volume.region_id for b in closure.current_bindings)
    if state.dc_current_regions!=regions:
        raise ValueError('Stored mixed current region order mismatch')
    sample=laplace.triangles
    roundoff=512*torch.finfo(sample.dtype).eps
    boundary={}
    electric=[]
    records=state.receipt.get('dc_current',{}).get('current',[])
    currents=state.total_volume_current[0,closure.current_indices].real.split(closure.current.counts)
    for index,(binding,block,potential,current) in enumerate(zip(closure.current_bindings,
            closure.current.blocks,state.dc_face_potentials,currents)):
        gauge=(potential*block.nodal_volume).sum()
        if not bool(gauge.abs()<=roundoff*(potential.abs()*block.nodal_volume).sum()):
            raise ValueError('Stored mixed current potential violates its gauge')
        allowed=records[index].get('threshold_A_l2') if index<len(records) else None
        closure.current.validate_receipt(records[index] if index<len(records) else None,block)
        if type(allowed) not in (int,float) or not math.isfinite(allowed) or allowed<0 or not bool(block.weak_current(current).norm()<=allowed):
            raise ValueError('Stored mixed current violates conservation')
        electric.append(block.electric(potential))
        boundary[binding.volume.region_id]=surface_face_potentials(block,binding,potential)
    dc_p=state.polarization[0,closure.dielectric_indices].real
    packed=torch.cat((dc_p.flatten(),*(boundary[r] for r in operator.current_regions)))
    for actual,matrix in ((state.dc_surface_charge,closure.maps['surface_response']),
            (torch.stack([state.dc_potential_offsets[regions.index(r)] for r in operator.current_regions]),closure.maps['offset_response'])):
        expected=matrix@packed
        if not bool(((actual-expected).abs()<=roundoff*(matrix.abs()@packed.abs()+actual.abs())).all()):
            raise ValueError('Stored mixed layer or offset violates its response map')
    counts=tuple(len(b.volume.cells) for b in closure.dielectric_bindings)
    decoder=StaticVolumeFieldResponse(laplace.system,operator.exterior,operator.interiors,
        closure.dielectric_bindings,dc_p.split(counts),state.dc_surface_charge,testing_order=laplace.testing_order)
    dielectric=torch.cat([decoder.fields(b.volume.vertices[b.volume.cells].mean(1),b.volume.region_id)[1]
                          for b in closure.dielectric_bindings])
    predicted=torch.zeros_like(state.cell_electric[0].real).index_copy(0,closure.current_indices,torch.cat(electric))
    predicted=predicted.index_copy(0,closure.dielectric_indices,dielectric)
    if not bool((predicted-state.cell_electric[0].real).norm()<=threshold):
        raise ValueError('Stored mixed DC field violates its physical decoder')
    mapped=(closure.maps['electric_response']@packed).reshape_as(dielectric)
    if not bool((mapped-dielectric).norm()<=threshold):
        raise ValueError('Stored mixed electric response violates its decoded state')
    phi,normal=[],[]
    for points,n,weights in zip(operator.nodes,laplace.normals,operator.weights):
        potential,field,jump=decoder._particular(points,principal_value=True,normals=n.expand_as(points))
        phi.append((potential*weights).sum())
        normal.append(((field*n).sum(-1)*weights).sum()+(jump*weights).sum()/2)
    phi,normal=torch.stack(phi),torch.stack(normal)
    boundary_vector=torch.cat([boundary.get(i+1,sample.new_zeros(size)) for i,size in enumerate(laplace.body_faces)])
    rhs=torch.cat((torch.where(operator.current_faces,boundary_vector-phi,operator.contrast*normal),sample.new_zeros(len(regions))))
    offsets=torch.stack([state.dc_potential_offsets[regions.index(r)] for r in operator.current_regions])
    unknown=torch.cat((state.dc_surface_charge,offsets))
    bound=roundoff*(operator.matrix.abs()@unknown.abs()+rhs.abs())
    if not bool(((operator.matrix@unknown-rhs).abs()<=bound).all()):
        raise ValueError('Stored mixed state violates its interface or neutral charge equations')
    validate_magnetic(static,state,bindings)
    return predicted.flatten()
