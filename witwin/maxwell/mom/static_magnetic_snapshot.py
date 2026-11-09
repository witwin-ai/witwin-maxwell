"""Exact magnetic contrast response storage against independent bound geometry."""
import copy
import torch
from .static_magnetic_volume import MagneticVolumeContrastResponse
from .static_tensor_green import StaticTensorGreen
from .static_tensor_polarization import StaticTensorPolarization
from .static_tensor_snapshot import _geometry,_volume
from .volume_h_snapshot import _digest
from .harmonic_snapshot import tensors
from .magnetostatic import cell_current_continuity

VERSION='regional-magnetic-volume-contrast-response-v1'
ARRAYS=('background','centers','contrast','matrix','cell_fields','source_field')


def snapshot_magnetic_volume_response(response):
    if type(response) is not MagneticVolumeContrastResponse:
        raise TypeError('Magnetic snapshot requires a native volume response')
    response.assert_current()
    kernels=(response.exterior,*response.interiors)
    names=('coefficient','factor','root_determinant')
    if any(not torch.equal(getattr(s.kernel,n),getattr(response.exterior,n)) for s in response.sources for n in names):
        raise ValueError('Magnetic polarization kernels differ from exterior')
    payload=tensors(dict(version=VERSION,geometry=_geometry(response.system),
        volumes=[_volume(b) for b in response.bindings],pair_batch=response.pair_batch,
        kernels=[{n:getattr(k,n) for n in names} for k in kernels],
        currents=list(response.currents),polarizations=[s.polarization for s in response.sources],
        state={n:getattr(response,n) for n in ARRAYS},receipt=copy.deepcopy(response.receipt)),'cpu')
    owner=dict(version=VERSION,sha256=_digest(payload))
    restore_magnetic_volume_response(payload,response.system,response.bindings,expected_binding=owner,
        max_matrix_bytes=response.matrix.numel()*response.matrix.element_size())
    return payload,owner


def restore_magnetic_volume_response(payload,system,bindings,*,expected_binding,max_matrix_bytes=256*1024**2):
    """Restore retained matrix/state without panel integration or magnetic solve.

    Matrix/source provenance is the independent owner digest; the stored source
    field is not recomputed. Residual and constitutive partition are checked.
    """
    keys={'version','geometry','volumes','pair_batch','kernels','currents','polarizations','state','receipt'}
    if (type(payload) is not dict or set(payload)!=keys or payload['version']!=VERSION
            or type(expected_binding) is not dict or expected_binding!={'version':VERSION,'sha256':_digest(payload)}):
        raise ValueError('Magnetic response differs from its owning binding')
    system.assert_current();bindings=tuple(bindings)
    if len(bindings)!=len(system.bodies):raise ValueError('Magnetic volume inventory differs')
    for region,b in enumerate(bindings,1):
        b.assert_current()
        if b.surface_system is not system or b.volume.region_id!=region:
            raise ValueError('Magnetic volumes must bind the exact system in region order')
    if payload['geometry']!=_geometry(system) or payload['volumes']!=[_volume(b) for b in bindings]:
        raise ValueError('Magnetic response geometry differs')
    if type(max_matrix_bytes) is not int or max_matrix_bytes<1 or type(payload['pair_batch']) is not int or payload['pair_batch']<1:
        raise ValueError('Magnetic restore budgets must be positive integers')
    counts=tuple(len(b.volume.cells) for b in bindings);count=sum(counts)
    required=(3*count)**2*torch.empty((),dtype=system.dtype).element_size()
    if required>max_matrix_bytes:raise ValueError('Magnetic matrix exceeds restore budget')
    def array(value,shape):
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=system.dtype
                or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid exact magnetic array')
        return value.to(device=system.device).clone()
    records=payload['kernels']
    if type(records) is not list or len(records)!=len(counts)+1:raise ValueError('Magnetic kernel inventory differs')
    kernels=[]
    for record in records:
        if type(record) is not dict or set(record)!={'coefficient','factor','root_determinant'}:
            raise ValueError('Invalid magnetic kernel record')
        k=StaticTensorGreen(array(record['coefficient'],(3,3)))
        factor=array(record['factor'],(3,3));det=array(record['root_determinant'],())
        product=factor@factor.T;tolerance=64*torch.finfo(system.dtype).eps*k.coefficient.abs().max()
        if (not torch.equal(factor,factor.tril()) or not bool((factor.diagonal()>0).all())
                or not bool(torch.isfinite(product).all()) or not bool((product-k.coefficient).abs().max()<=tolerance)
                or not bool(det>0) or not torch.equal(det,factor.diagonal().prod())):
            raise ValueError('Magnetic kernel factors differ from coefficients')
        k.factor=factor;k.root_determinant=det;k._binding=k._state();kernels.append(k)
    r=object.__new__(MagneticVolumeContrastResponse)
    r.system=system;r.bindings=bindings;r.counts=counts;r.pair_batch=payload['pair_batch']
    r.exterior=kernels[0];r.interiors=tuple(kernels[1:])
    state=payload['state']
    if type(state) is not dict or set(state)!=set(ARRAYS):raise ValueError('Magnetic state inventory differs')
    for name,shape in zip(ARRAYS,((3,),(count,3),(count,3,3),(3*count,3*count),(count,3),(count,3))):
        setattr(r,name,array(state[name],shape))
    centers=torch.cat([b.volume.vertices[b.volume.cells].mean(1) for b in bindings])
    contrast=torch.cat([(k.coefficient-r.exterior.coefficient).expand(n,3,3) for k,n in zip(r.interiors,counts)])
    if not torch.equal(r.centers,centers) or not torch.equal(r.contrast,contrast):
        raise ValueError('Magnetic collocation geometry or contrast differs')
    for name in ('currents','polarizations'):
        if type(payload[name]) is not list or len(payload[name])!=len(counts):raise ValueError('Magnetic source inventory differs')
    r.currents=tuple(array(j,(n,3)) for j,n in zip(payload['currents'],counts))
    polarizations=tuple(array(p,(n,3)) for p,n in zip(payload['polarizations'],counts))
    expected=torch.einsum('tij,tj->ti',r.contrast,r.cell_fields)
    if not torch.allclose(torch.cat(polarizations),expected,rtol=64*torch.finfo(system.dtype).eps,atol=0):
        raise ValueError('Magnetic constitutive partition differs')
    r.sources=tuple(StaticTensorPolarization(b,r.exterior.coefficient,p,pair_batch=r.pair_batch)
                    for b,p in zip(bindings,polarizations))
    for source in r.sources:
        source.kernel.factor=r.exterior.factor.clone();source.kernel.root_determinant=r.exterior.root_determinant.clone()
        source.kernel._binding=source.kernel._state()
    def norm(v):
        scale=v.abs().max()
        return scale*(v/scale).norm() if bool(scale>0) else scale
    residual=r.matrix@r.cell_fields.flatten()-r.source_field.flatten()
    error=norm(residual)
    threshold=512*torch.finfo(system.dtype).eps*(norm(r.matrix)*norm(r.cell_fields)+norm(r.source_field))
    jumps=torch.stack([cell_current_continuity(b.volume,j)['area_weighted_jump_rms_A_per_m2'] for b,j in zip(bindings,r.currents)])
    if not bool(torch.isfinite(error)&torch.isfinite(threshold)&torch.isfinite(jumps).all()) or not bool(error<=threshold):
        raise ValueError('Stored magnetic response fails its residual check')
    receipt=copy.deepcopy(payload['receipt'])
    structural=dict(formulation='regional_magnetic_p0_contrast_collocation_v1',retained_matrix_bytes=required,
        pair_batch=r.pair_batch,physical_qualification=False,pointwise_constitutive_accuracy_verified=False,
        interface_continuity_verified=False,source_current_jump_rms_A_per_m2=jumps.detach().cpu().tolist())
    if type(receipt) is not dict or set(receipt)!=set(structural)|{'algebraic_residual_norm','threshold'}:
        raise ValueError('Invalid magnetic response receipt')
    if any(receipt[k]!=v for k,v in structural.items()):raise ValueError('Magnetic response receipt differs')
    for name,actual in (('algebraic_residual_norm',error),('threshold',threshold)):
        value=receipt[name]
        if type(value) not in (int,float) or not torch.isfinite(actual.new_tensor(value)) or value<0:
            raise ValueError('Invalid magnetic residual receipt')
        # Saved reductions can differ by a few ulps on a subsequent CUDA pass.
        allowance=64*torch.finfo(system.dtype).eps*threshold
        if not bool((actual-actual.new_tensor(value)).abs()<=allowance):raise ValueError('Magnetic residual receipt differs')
    r.receipt=receipt;r._binding=r._state();r.assert_current()
    return r
