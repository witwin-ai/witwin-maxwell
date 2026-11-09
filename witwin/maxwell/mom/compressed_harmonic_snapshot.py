"""Exact admitted positive-harmonic preparation with shared native action state."""
import copy
import torch
from . import coupled_response_snapshot as coupled
from . import background_bank_snapshot as background
from .nested_h2_snapshot import _keys
from .volume_h_snapshot import _digest
from .runtime import _same_metadata


def _validate(part):
    _keys(part,('electric_action','magnetic_action','bindings','receipt','background_bank',
        'background_surface','background_electric','background_magnetic','cell_volumes','points'))
    e,h=part['electric_action'],part['magnetic_action'];coupled._pair(e,h)
    bank=part['background_bank'];bank.assert_current()
    bindings=tuple(s.provider.binding for s in e.sources)
    if (bank.inverse is not e.inverse or len(bank.bindings)!=len(bindings)
            or len(part['bindings'])!=len(bindings)
            or any(a is not b for a,b in zip(bank.bindings,bindings))
            or any(a is not b for a,b in zip(part['bindings'],bindings))):
        raise ValueError('Compressed harmonic snapshot requires shared exact bindings')
    expected=bank.combine(torch.ones(len(bank.carriers),device=e.device,dtype=e.dtype))
    expected.update(cell_volumes=torch.cat([b.volume.volumes for b in bindings]),
        points=torch.cat([b.volume.vertices[b.volume.cells].mean(1) for b in bindings]))
    for name,value in expected.items():
        actual=part[name]
        if (not isinstance(actual,torch.Tensor) or actual.dtype!=value.dtype
                or actual.device!=value.device or actual.shape!=value.shape or not torch.equal(actual,value)):
            raise ValueError('Compressed harmonic snapshot derived arrays mismatch')
    receipt=part['receipt']
    if (receipt.get('version')!='adaptive_compressed_induced_response_v1'
            or receipt.get('converged') is not True or receipt.get('failure_reason') is not None
            or receipt.get('physical_qualification') is not False
            or receipt.get('total_error_certified') is not False
            or receipt.get('aggregate_capacity_enforced') is not False
            or receipt.get('incident_background_included') is not True
            or not _same_metadata(receipt.get('background'),bank.receipt)):
        raise ValueError('Compressed harmonic snapshot admission receipt mismatch')


def encode(part,*,sources):
    _validate(part)
    payload={'version':'exact_compressed_harmonic_preparation_v1',
        'actions':coupled.encode(part['electric_action'],part['magnetic_action']),
        'background':background.encode(part['background_bank'],sources=sources),
        'receipt':copy.deepcopy(part['receipt'])}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,provider,volumes,*,sources):
    """Restore historical admission and exact factors, without re-admitting physics."""
    _keys(payload,('version','actions','background','receipt','sha256'))
    if (payload['version']!='exact_compressed_harmonic_preparation_v1'
            or payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'})):
        raise ValueError('Compressed harmonic snapshot checksum/version mismatch')
    sources=tuple(sources)
    e,h=coupled.restore(payload['actions'],provider,volumes,sources=sources)
    bindings=tuple(s.provider.binding for s in e.sources)
    bank=background.restore(payload['background'],e.inverse,bindings,sources=sources)
    result={'electric_action':e,'magnetic_action':h,'bindings':bindings,
        'background_bank':bank,'receipt':copy.deepcopy(payload['receipt']),
        'cell_volumes':torch.cat([b.volume.volumes for b in bindings]),
        'points':torch.cat([b.volume.vertices[b.volume.cells].mean(1) for b in bindings])}
    result.update(bank.combine(torch.ones(len(bank.carriers),device=e.device,dtype=e.dtype)))
    _validate(result)
    return result


@torch.no_grad()
def validate_surface_state(part,current,induced_surface):
    """Check stored induced coefficients and decode E/H without applying an inverse.

    This checks the saved compressed surface equation, not the original kernel
    equation or a physical error bound. Supplied currents exclude background;
    callers add the stored pump-scaled background once. No response Jacobian or
    implicit inverse derivative is defined by this detached validation path.
    """
    _validate(part)
    e,h=part['electric_action'],part['magnetic_action']
    local=e._split(current)
    inverse=e.inverse
    if (not isinstance(induced_surface,torch.Tensor)
            or induced_surface.shape!=(*current.shape[:-1],inverse.shape[0])
            or induced_surface.dtype!=inverse.dtype or induced_surface.device!=inverse.device
            or not bool(torch.isfinite(induced_surface).all())):
        raise ValueError('Invalid stored induced surface coefficients')
    rhs=current.new_zeros(induced_surface.shape)
    for value,source,ids in zip(local,e.sources,e.indices):
        rhs=rhs.index_add(-1,ids,source.apply(value))
    residual=inverse.operator.apply(induced_surface)-rhs
    error=torch.linalg.vector_norm(residual,dim=-1)
    threshold=torch.maximum(torch.linalg.vector_norm(rhs,dim=-1)*inverse.rule.rtol,
        rhs.real.new_tensor(inverse.rule.atol))
    if not bool(torch.isfinite(error).all() and torch.isfinite(threshold).all() and (error<=threshold).all()):
        raise ValueError('Stored induced surface violates compressed equation tolerance')
    if not inverse._dual_gate(residual,rhs)[0]:
        raise ValueError('Stored induced surface violates physical dual tolerance')
    fields=tuple(a._decode(local,induced_surface,False) for a in (e,h))
    if not all(bool(torch.isfinite(v).all()) for v in fields):
        raise ValueError('Nonfinite stored induced field response')
    return fields
