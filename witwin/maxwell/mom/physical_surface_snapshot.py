"""Exact physical H/H2 carrier bindings around saved scalar numerical factors."""
import copy
import hashlib
from .runtime import MoMConfig,_numerical_identity,_same_metadata
from .hierarchical_runtime import HierarchicalSystemOperator,HierarchicalH2SystemOperator,hierarchical_operator_identity
from .hmatrix import HierarchicalOperator,HMatrixConfig
from .h2_operator import HierarchicalH2Operator
from .h2 import H2Config
from .operator import IntegrationConfig
from .fields import FieldQueryConfig
from .linalg import StoppingRule
from .carrier import IncidentCarrier
from .nearfield import ScalarSystemBlocks
from .tensor_nearfield import GeneralizedSystemBlocks
from .ewald_nearfield import EwaldSystemBlocks, GeneralizedEwaldSystemBlocks
from .periodic_hierarchical_runtime import PeriodicHierarchicalSystemOperator, PeriodicHierarchicalH2SystemOperator
from .periodic_runtime import periodic_operator_identity
from .periodic_config import MoMPeriodicConfig
from .dyadic_green import DyadicIntegrationConfig
from .volume_h_snapshot import _digest
from .nested_h2_snapshot import _keys
import torch


_TYPES={HierarchicalSystemOperator:HierarchicalOperator,HierarchicalH2SystemOperator:HierarchicalH2Operator,
        PeriodicHierarchicalSystemOperator:HierarchicalOperator,
        PeriodicHierarchicalH2SystemOperator:HierarchicalH2Operator}


def _physical_version(config):
    return ('exact_physical_scalar_surface_h_v1' if config.periodic is None
            and config.dyadic_integration is None else 'exact_physical_regional_surface_h_v2')


def _operator_identity(provider, config):
    if config.periodic is not None:
        return periodic_operator_identity(provider.system, provider.exterior, config, provider.kernel)
    return hierarchical_operator_identity(provider.system, provider.exterior, config)


def _validate(config,provider,numerical):
    if (not isinstance(config,MoMConfig) or config.operator_backend not in ('hmatrix','h2')
            or config.harmonic_balance is not None or config.integration!=provider.config
            or config.dtype!=(torch.complex128 if provider.system.dtype==torch.float64 else torch.complex64)
            or not _same_metadata(_numerical_identity(config),numerical)):
        raise ValueError('Physical H snapshot requires matching regional configuration')
    expected = ((GeneralizedEwaldSystemBlocks if config.dyadic_integration is not None else EwaldSystemBlocks)
                if config.periodic is not None else
                (GeneralizedSystemBlocks if config.dyadic_integration is not None else ScalarSystemBlocks))
    if type(provider) is not expected:
        raise ValueError('Physical H snapshot regional provider kind mismatch')
    provider.assert_current()
    if config.periodic is not None:
        kernel=config.periodic.compile(device=provider.system.device,dtype=config.dtype)
        if not _same_metadata(kernel.receipt(),provider.kernel.receipt()):
            raise ValueError('Physical H snapshot periodic kernel mismatch')
    if config.dyadic_integration is not None:
        for kernel in (provider.exterior_kernel,*provider.interior_kernels):
            if kernel is not None and (kernel.integration!=config.dyadic_integration
                    or kernel.cache_max_bytes!=config.dyadic_cache_max_bytes//2):
                raise ValueError('Physical H snapshot regional spectral policy mismatch')


def _restore_config(value):
    keys={'subdivisions','dtype','riesz_storage','integration','stopping','query_config','operator_backend','hierarchical'}
    if isinstance(value,dict) and value.get('operator_backend')=='h2':keys.add('h2')
    if isinstance(value,dict) and 'periodic' in value:keys.add('periodic')
    if isinstance(value,dict) and 'dyadic_integration' in value:
        keys.update(('dyadic_integration','dyadic_cache_max_bytes'))
    _keys(value,keys)
    dtype={'torch.complex64':torch.complex64,'torch.complex128':torch.complex128}.get(value['dtype'])
    return MoMConfig(subdivisions=value['subdivisions'],dtype=dtype,riesz_storage=value['riesz_storage'],
        integration=IntegrationConfig(**value['integration']),stopping=StoppingRule(**value['stopping']),
        query_config=FieldQueryConfig(**value['query_config']),operator_backend=value['operator_backend'],
        hierarchical=HMatrixConfig.restore(value['hierarchical']),
        h2=H2Config.restore(value['h2']) if 'h2' in value else None,
        periodic=MoMPeriodicConfig.restore(value['periodic']) if 'periodic' in value else None,
        dyadic_integration=DyadicIntegrationConfig(**value['dyadic_integration']) if 'dyadic_integration' in value else None,
        dyadic_cache_max_bytes=value.get('dyadic_cache_max_bytes',256*1024**2))


def _validate_identities(operator,carrier,config):
    identity,payload=_operator_identity(operator.provider,config)
    metadata=operator.metadata
    if (metadata.get('operator_identity')!=identity
            or not _same_metadata(metadata.get('operator_identity_payload'),payload)):
        raise ValueError('Physical H snapshot operator identity mismatch')
    expected={'incident_carrier_identity':carrier.identity,
        'rhs_identity':hashlib.sha256((identity+':'+carrier.identity).encode()).hexdigest()}
    for name,value in expected.items():
        if name in metadata and metadata[name]!=value:
            raise ValueError('Physical H snapshot RHS/carrier identity mismatch')


def encode(operator):
    if type(operator) not in _TYPES:raise TypeError('Physical snapshot requires an exact H/H2 runtime type')
    operator.refresh();config=operator._public_config;numerical=_numerical_identity(config)
    _validate(config,operator.provider,numerical)
    if (not _same_metadata(numerical,operator.metadata.get('numerical_identity'))
            or (config.hierarchical!=(operator.source_configuration if config.operator_backend=='h2' else operator.config))
            or (config.operator_backend=='h2' and config.h2!=operator.config)):
        raise ValueError('Physical H snapshot numerical binding changed')
    carrier=IncidentCarrier.restore(operator.incident_carrier.payload(),expected_identity=operator.incident_carrier.identity,
        sources=operator.sources,exterior=operator.exterior)
    _validate_identities(operator,carrier,config)
    # A read-only numerical view shares immutable factor state. This avoids
    # either rebuilding the factors or widening the bare codec's accepted types.
    bare=object.__new__(_TYPES[type(operator)])
    bare.__dict__.update(vars(operator))
    from .surface_h_snapshot import encode as encode_numerical
    payload={'version':_physical_version(config),'numerical':numerical,
        'operator':encode_numerical(bare),'carrier':operator.incident_carrier.payload(),
        'carrier_identity':operator.incident_carrier.identity}
    payload['sha256']=_digest(payload)
    return payload


def restore(payload,provider,*,sources=()):
    """Validate supplied sources and rebind stored factors without any solve."""
    _keys(payload,('version','numerical','operator','carrier','carrier_identity','sha256'))
    if (payload['version'] not in ('exact_physical_scalar_surface_h_v1','exact_physical_regional_surface_h_v2')
            or payload['sha256']!=_digest({k:v for k,v in payload.items() if k!='sha256'})):
        raise ValueError('Physical H snapshot identity/checksum mismatch')
    config=_restore_config(payload['numerical'])
    if payload['version']!=_physical_version(config):
        raise ValueError('Physical H snapshot version does not match regional configuration')
    _validate(config,provider,payload['numerical']);sources=tuple(sources)
    carrier=IncidentCarrier.restore(payload['carrier'],expected_identity=payload['carrier_identity'],
        sources=sources,exterior=provider.exterior)
    from .surface_h_snapshot import restore as restore_numerical
    # Only a bare numerical payload may be nested: physical nesting is invalid.
    if not isinstance(payload['operator'],dict) or payload['operator'].get('version')!=('exact_scalar_surface_h_v1' if type(provider) is ScalarSystemBlocks
                                            else 'exact_regional_surface_h_v2'):
        raise ValueError('Physical H snapshot requires a bare numerical payload')
    bare=restore_numerical(payload['operator'],provider)
    if (payload['operator']['kind']!=config.operator_backend
            or (config.hierarchical!=(bare.source_configuration if config.operator_backend=='h2' else bare.config))
            or (config.operator_backend=='h2' and config.h2!=bare.config)):
        raise ValueError('Physical H snapshot factor configuration mismatch')
    cls = ((PeriodicHierarchicalH2SystemOperator if config.operator_backend=='h2' else PeriodicHierarchicalSystemOperator)
           if config.periodic is not None else
           (HierarchicalH2SystemOperator if config.operator_backend=='h2' else HierarchicalSystemOperator))
    result=object.__new__(cls);result.__dict__.update(vars(bare))
    metadata=copy.deepcopy(bare.metadata)
    if config.periodic is None:
        result._bind_physical(provider.system,provider.exterior,sources,config)
    else:
        identity,identity_payload=_operator_identity(provider,config)
        result._bind_hierarchy(provider.system,provider.exterior,sources,config,provider.kernel,identity,identity_payload)
    if (result.incident_carrier.identity!=carrier.identity or not _same_metadata(result.metadata,metadata)):
        raise ValueError('Physical H snapshot metadata does not match reconstructed bindings')
    _validate_identities(result,carrier,config)
    result.refresh()
    return result
