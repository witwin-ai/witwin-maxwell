"""Exact induced static response state bound to validated regional preparation.

This component restores without assembly or solve. The owning mixed Result
must separately restore the preparation and retain the expected binding.
"""
import copy
import math
import torch
from .volume_h_snapshot import _digest
from .static_tensor_interface import StaticTensorInterfaceOperator,StaticTensorInterfaceResponse
from .static_tensor_polarization import StaticTensorPolarization
from .static_tensor_floating import StaticTensorFloatingOperator,StaticTensorFloatingResponse

VERSION='regional-spd-static-response-v2'
FLOATING_VERSION='regional-floating-static-response-v1'


def _preparation(operator):
    if type(operator) not in (StaticTensorInterfaceOperator,StaticTensorFloatingOperator):
        raise TypeError('Static snapshot requires native regional preparation')
    operator.assert_current()
    if type(operator) is StaticTensorInterfaceOperator and operator.potential_basis!='p0':
        raise ValueError('Static snapshots currently require P0 potential')
    record=dict(pair_batch=operator.pair_batch,face_counts=operator.face_counts,
        bodies=[dict(region=b.region_id,vertices=b.mesh.vertices,faces=b.mesh.faces) for b in operator.system.bodies],
        kernels=[None if k is None else (k.coefficient,k.factor,k.root_determinant) for k in (operator.exterior,*operator.interiors)],
        triangles=operator.triangles,normals=operator.normals,centers=operator.centers,
        flux_scale=operator.flux_scale,matrix=operator.matrix,metadata=operator.metadata)
    if type(operator) is StaticTensorFloatingOperator:
        record.update(current_regions=operator.current_regions,areas=operator.areas)
    return _digest(record)


def _volume(binding):
    binding.assert_current()
    return _digest(dict(region=binding.volume.region_id,vertices=binding.volume.vertices,
        cells=binding.volume.cells,mapping=binding.boundary_to_surface_vertices))


def snapshot_static_tensor_response(response):
    if type(response) not in (StaticTensorInterfaceResponse,StaticTensorFloatingResponse):
        raise TypeError('Static snapshot requires a native response')
    response.assert_current()
    floating=type(response) is StaticTensorFloatingResponse
    if floating!=(type(response.operator) is StaticTensorFloatingOperator):
        raise ValueError('Static response kind differs from its preparation')
    version=FLOATING_VERSION if floating else VERSION
    names=('field','potential','conormal','normal_p')+(('offsets',) if floating else ())
    for index,source in enumerate(response.sources):
        if source is not None and any(not torch.equal(getattr(source.kernel,name),getattr(response.operator.interiors[index],name))
                                      for name in ('coefficient','factor','root_determinant')):
            raise ValueError('Static source kernel differs from its regional preparation')
    payload=dict(version=version,preparation=_preparation(response.operator),offset=None if floating else response.offset,
        state={name:getattr(response,name).detach().cpu().clone()
               for name in names},
        sources=[dict(region=source.binding.volume.region_id,volume=_volume(source.binding),
                      polarization=source.polarization.detach().cpu().clone())
                 for source in response.sources if source is not None],
        receipt=copy.deepcopy(response.receipt))
    binding=dict(version=version,sha256=_digest(payload))
    # Repeat structural admission, never assembly or a numerical solve.
    restore_static_tensor_response(payload,response.operator,
        tuple(s.binding for s in response.sources if s is not None),expected_binding=binding)
    return payload,binding


def restore_static_tensor_response(payload,operator,bindings,*,expected_binding):
    """Decode exact state against independent preparation, volumes and owner hash."""
    floating=type(operator) is StaticTensorFloatingOperator
    version=FLOATING_VERSION if floating else VERSION
    if (type(payload) is not dict or set(payload)!={'version','preparation','offset','state','sources','receipt'}
            or payload['version']!=version or type(expected_binding) is not dict
            or expected_binding!={'version':version,'sha256':_digest(payload)}):
        raise ValueError('Static response differs from its owning state binding')
    if payload['preparation']!=_preparation(operator):
        raise ValueError('Static response preparation differs')
    bindings=tuple(bindings);by_region={}
    for binding in bindings:
        if getattr(binding,'surface_system',None) is not operator.system:
            raise ValueError('Static response volumes must bind the exact prepared surface system')
        binding.assert_current();region=binding.volume.region_id
        if floating and region in operator.current_regions:
            raise ValueError('Static polarization cannot bind a conductive region')
        if region in by_region:raise ValueError('Duplicate static response volume region')
        by_region[region]=binding
    records=payload['sources']
    if type(records) is not list or len(records)!=len(bindings):
        raise ValueError('Static response source inventory differs')
    device=operator.system.device;dtype=operator.system.dtype
    def array(value,shape):
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=dtype
                or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid exact static response array')
        return value.to(device=device).clone()
    sources=[None]*len(operator.system.bodies);previous=0
    for row in records:
        if type(row) is not dict or set(row)!={'region','volume','polarization'}:
            raise ValueError('Invalid static response source record')
        region=row['region']
        if type(region) is not int or region<=previous or region not in by_region:
            raise ValueError('Static response sources require exact increasing region order')
        previous=region;volume_binding=by_region[region]
        if row['volume']!=_volume(volume_binding):
            raise ValueError('Static response volume geometry differs')
        polarization=array(row['polarization'],(len(volume_binding.volume.cells),3))
        sources[region-1]=StaticTensorPolarization(volume_binding,operator.interiors[region-1].coefficient,
                                                   polarization,pair_batch=operator.pair_batch)
        # Replay the owned factors rather than a fresh decomposition's rounding.
        kernel=sources[region-1].kernel;regional=operator.interiors[region-1]
        kernel.factor=regional.factor.clone();kernel.root_determinant=regional.root_determinant.clone()
        kernel._binding=kernel._state()
    state=payload['state'];offset=payload['offset'];receipt=payload['receipt']
    names={'field','potential','conormal','normal_p'}|({'offsets'} if floating else set())
    if (type(state) is not dict or set(state)!=names
            or (offset is not None if floating else type(offset) not in (int,float) or not math.isfinite(offset))):
        raise ValueError('Invalid static response state')
    metadata=operator.metadata
    extra={'algebraic_residual_norm','relative_algebraic_residual'}| (set() if floating else {'induced_regions'})
    if (type(receipt) is not dict or set(receipt)!=(set(metadata)|extra)
            or any(receipt[k]!=v for k,v in metadata.items()) or receipt['physical_qualification'] is not False
            or (not floating and receipt['induced_regions']!=tuple(sorted(by_region)))
            or any(type(receipt[k]) not in (int,float) or not math.isfinite(receipt[k]) or receipt[k]<0
                   for k in ('algebraic_residual_norm','relative_algebraic_residual'))):
        raise ValueError('Invalid stored static response diagnostics')
    response=object.__new__(StaticTensorFloatingResponse if floating else StaticTensorInterfaceResponse)
    response.operator=operator;response.sources=tuple(sources)
    if not floating:response.offset=offset
    count=len(operator.centers)
    for name,value in state.items():
        shape=(3,) if name=='field' else (len(operator.current_regions),) if name=='offsets' else (count,)
        setattr(response,name,array(value,shape))
    normal_p=[];start=0
    for size,source in zip(operator.face_counts,sources):
        normal=operator.normals[start:start+size]
        normal_p.append(normal.new_zeros(size) if source is None else
                        (source.polarization[source.boundary_cells]*normal).sum(-1))
        start+=size
    if not torch.equal(response.normal_p,torch.cat(normal_p)):
        raise ValueError('Stored static response boundary polarization differs from source cells')
    response.receipt=copy.deepcopy(receipt)
    response._binding=response._state()
    response.assert_current()
    return response


PREPARATION_VERSION='regional-spd-static-preparation-v1'
FLOATING_PREPARATION_VERSION='regional-floating-static-preparation-v1'


def _geometry(system):
    system.assert_current()
    return _digest([dict(region=b.region_id,vertices=b.mesh.vertices,faces=b.mesh.faces)
                    for b in system.bodies])


def snapshot_static_tensor_preparation(operator):
    """Store the actual matrix and kernel factors, not a recipe to reassemble."""
    _preparation(operator)
    def host(value):return value.detach().cpu().clone()
    version=FLOATING_PREPARATION_VERSION if type(operator) is StaticTensorFloatingOperator else PREPARATION_VERSION
    payload=dict(version=version,geometry=_geometry(operator.system),
        pair_batch=operator.pair_batch,matrix=host(operator.matrix),flux_scale=host(operator.flux_scale),
        kernels=[None if k is None else dict(coefficient=host(k.coefficient),factor=host(k.factor),
                      root_determinant=host(k.root_determinant)) for k in (operator.exterior,*operator.interiors)],
        metadata=copy.deepcopy(operator.metadata))
    binding=dict(version=version,sha256=_digest(payload))
    restore_static_tensor_preparation(payload,operator.system,expected_binding=binding,
                                     max_matrix_bytes=operator.matrix.numel()*operator.matrix.element_size())
    return payload,binding


def restore_static_tensor_preparation(payload,system,*,expected_binding,max_matrix_bytes=256*1024**2):
    """Restore against validated surface geometry, with no panel assembly or solve.

    Stored Cholesky factors are validated against their coefficient and retained
    exactly. The owner's expected binding is independent of this payload.
    """
    from .static_tensor_green import StaticTensorGreen
    floating=type(payload) is dict and payload.get('version')==FLOATING_PREPARATION_VERSION
    version=FLOATING_PREPARATION_VERSION if floating else PREPARATION_VERSION
    if (type(payload) is not dict or set(payload)!={'version','geometry','pair_batch','matrix','flux_scale','kernels','metadata'}
            or payload['version']!=version or type(expected_binding) is not dict
            or expected_binding!={'version':version,'sha256':_digest(payload)}):
        raise ValueError('Static preparation differs from its owning state binding')
    if payload['geometry']!=_geometry(system):raise ValueError('Static preparation surface geometry differs')
    if (type(payload['pair_batch']) is not int or payload['pair_batch']<1
            or type(max_matrix_bytes) is not int or max_matrix_bytes<1):
        raise ValueError('Static preparation budgets must be positive integers')
    counts=tuple(len(b.mesh.faces) for b in system.bodies);count=sum(counts)
    records=payload['kernels']
    if (type(records) is not list or len(records)!=len(counts)+1 or records[0] is None
            or (any(v is None for v in records)!=floating)):
        raise ValueError('Static preparation kernel inventory differs')
    regions=tuple(i for i,k in enumerate(records) if k is None)
    size=2*count+len(regions)
    required=size**2*torch.empty((),dtype=system.dtype).element_size()
    if required>max_matrix_bytes:raise ValueError('Stored static matrix exceeds restore budget')
    def array(value,shape):
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=system.dtype
                or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid exact static preparation array')
        return value.to(device=system.device).clone()
    records=payload['kernels']
    if type(records) is not list or len(records)!=len(counts)+1:
        raise ValueError('Static preparation kernel inventory differs')
    kernels=[]
    for record in records:
        if record is None:
            kernels.append(None);continue
        if type(record) is not dict or set(record)!={'coefficient','factor','root_determinant'}:
            raise ValueError('Invalid stored static kernel record')
        coefficient=array(record['coefficient'],(3,3));factor=array(record['factor'],(3,3))
        determinant=array(record['root_determinant'],())
        if (not torch.equal(coefficient,coefficient.T) or not torch.equal(factor,factor.tril())
                or not bool((factor.diagonal()>0).all()) or not bool(determinant>0)):
            raise ValueError('Invalid stored SPD factor structure')
        _,info=torch.linalg.cholesky_ex(coefficient)
        if bool(info!=0):
            raise ValueError('Stored static coefficient must independently be positive definite')
        product=factor@factor.T
        tolerance=64*torch.finfo(system.dtype).eps*coefficient.abs().amax()
        if (not bool(torch.isfinite(product).all()) or not bool((product-coefficient).abs().amax()<=tolerance)
                or not torch.equal(determinant,factor.diagonal().prod())):
            raise ValueError('Stored static kernel factors do not match coefficients')
        kernel=object.__new__(StaticTensorGreen)
        kernel.coefficient=coefficient;kernel.factor=factor;kernel.root_determinant=determinant
        kernel._binding=kernel._state();kernels.append(kernel)
    operator=object.__new__(StaticTensorFloatingOperator if floating else StaticTensorInterfaceOperator)
    operator.system=system;operator.pair_batch=payload['pair_batch'];operator.face_counts=counts
    operator.exterior=kernels[0];operator.interiors=tuple(kernels[1:])
    if not floating:
        operator.potential_basis='p0';operator.potential_counts=counts
        operator.potential_count=sum(counts)
        operator.potential_indices=torch.arange(sum(counts),device=system.device)[:,None]
    if floating:
        operator.current_regions=regions
        operator.areas=torch.cat([b.mesh.areas for b in system.bodies])
    operator.triangles=torch.cat([b.mesh.triangles for b in system.bodies])
    operator.normals=torch.cat([b.mesh.normals for b in system.bodies]);operator.centers=operator.triangles.mean(1)
    operator.flux_scale=array(payload['flux_scale'],())
    extent=(operator.triangles.amax((0,1))-operator.triangles.amin((0,1))).norm()
    expected_scale=operator.exterior.coefficient.diagonal().mean()/extent
    if not bool(operator.flux_scale>0) or not torch.equal(operator.flux_scale,expected_scale):
        raise ValueError('Stored static flux scale differs from geometry and coefficient')
    expected=dict(formulation='regional_static_potential_conormal_p0_collocation_v1',physical_qualification=False,
        retained_matrix_bytes=required,pair_batch=operator.pair_batch,surface_discretization_error_estimated=False,
        units='caller_supplied_constitutive_coefficient')
    if floating:
        expected=dict(formulation='regional_tensor_floating_neutral_p0_v1',physical_qualification=False,
            retained_matrix_bytes=required,current_regions=regions,surface_discretization_error_estimated=False)
    if payload['metadata']!=expected or payload['metadata'].get('physical_qualification') is not False:
        raise ValueError('Invalid stored static preparation metadata')
    operator.matrix=array(payload['matrix'],(size,size));operator.metadata=copy.deepcopy(expected)
    operator._binding=operator._state();operator.assert_current()
    return operator
