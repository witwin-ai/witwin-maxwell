"""Exact declared-source arrays bound to an existing restored surface inverse."""
import copy
import torch
from .compressed_background import CompressedBackgroundBank
from .carrier import IncidentCarrier
from .surface_h_snapshot import encode_inverse
from .volume_h_snapshot import _digest
from .nested_h2_snapshot import _keys,_array


def _inverse_identity(inverse):
    payload=encode_inverse(inverse)
    return _digest({k:v for k,v in payload.items() if k not in ('sha256','diagnostics')})


def _bindings(bindings):
    return [(b.volume.region_id,_digest((b.volume.vertices,b.volume.cells))) for b in bindings]


def _validate(bank,sources):
    bank.assert_current();sources=tuple(sources)
    if len(bank.carriers)!=len(sources):raise ValueError('Background snapshot source count mismatch')
    for carrier,source in zip(bank.carriers,sources):
        IncidentCarrier.restore(carrier.payload(),expected_identity=carrier.identity,
            sources=(source,),exterior=bank.inverse.operator.provider.exterior)
    count=sum(len(b.volume.cells) for b in bank.bindings);n=len(sources)
    for value,shape in zip(bank.tensors(),((n,bank.inverse.shape[0]),)*2+((n,count,3),)*2):
        if (value.shape!=shape or value.dtype!=bank.dtype or value.device!=bank.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Background snapshot array shape/precision mismatch')
    system=bank.inverse.operator.provider.system
    if (not bank.bindings or any(b.surface_system is not system for b in bank.bindings)
            or len({b.volume.region_id for b in bank.bindings})!=len(bank.bindings)):
        raise ValueError('Background snapshot ordered bindings mismatch')
    if (bank.receipt.get('source_count')!=n
            or bank.receipt.get('regions')!=[b.volume.region_id for b in bank.bindings]
            or bank.receipt.get('source_carrier_identities')!=[c.identity for c in bank.carriers]
            or bank.receipt.get('physical_qualification') is not False):
        raise ValueError('Background snapshot receipt binding mismatch')


def encode(bank,*,sources):
    if type(bank) is not CompressedBackgroundBank:raise TypeError('Expected exact background bank')
    _validate(bank,sources)
    payload={'version':'exact_compressed_background_bank_v1',
        'inverse_sha256':_inverse_identity(bank.inverse),'bindings':_bindings(bank.bindings),
        'carriers':[c.payload() for c in bank.carriers],'identities':[c.identity for c in bank.carriers],
        'arrays':[v.detach().cpu().clone() for v in bank.tensors()],
        'receipt':copy.deepcopy(bank.receipt)}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,inverse,bindings,*,sources):
    """Restore stored responses; historical convergence is not re-evaluated here."""
    _keys(payload,('version','inverse_sha256','bindings','carriers','identities','arrays','receipt','sha256'))
    if (payload['version']!='exact_compressed_background_bank_v1'
            or payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'})):
        raise ValueError('Background snapshot checksum/version mismatch')
    inverse.assert_current();bindings=tuple(bindings);sources=tuple(sources)
    if (payload['inverse_sha256']!=_inverse_identity(inverse)
            or payload['bindings']!=_bindings(bindings)):
        raise ValueError('Background snapshot inverse/volume binding mismatch')
    n=len(sources);count=sum(len(b.volume.cells) for b in bindings)
    if len(payload['carriers'])!=n or len(payload['identities'])!=n or len(payload['arrays'])!=4:
        raise ValueError('Background snapshot source/array count mismatch')
    carriers=[IncidentCarrier.restore(p,expected_identity=i,sources=(s,),exterior=inverse.operator.provider.exterior)
        for p,i,s in zip(payload['carriers'],payload['identities'],sources)]
    shapes=((n,inverse.shape[0]),)*2+((n,count,3),)*2
    arrays=[_array(v,shape,inverse.dtype,inverse.device) for v,shape in zip(payload['arrays'],shapes)]
    bank=CompressedBackgroundBank(inverse,bindings,carriers,*arrays,copy.deepcopy(payload['receipt']))
    _validate(bank,sources)
    return bank
