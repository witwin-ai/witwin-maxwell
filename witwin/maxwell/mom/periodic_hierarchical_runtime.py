"""Periodic H/H2 operators with local scalar or tensor interior blocks."""
from dataclasses import asdict
from .hmatrix import HierarchicalOperator
from .h2_operator import HierarchicalH2Operator
from .runtime import SystemDecodeHandle, _numerical_identity
from .periodic_runtime import (PeriodicSystemOperator, PeriodicMultipoleSystemOperator,
                               periodic_operator_identity)
from .periodic_operator import validate_periodic_sources
from .ewald_nearfield import EwaldSystemBlocks, GeneralizedEwaldSystemBlocks
from .tensor_medium import TensorMedium
from .dyadic_green import DyadicGreenKernel


def _provider(system, exterior, sources, config, kernel):
    identity, payload = periodic_operator_identity(system, exterior, config, kernel)
    validate_periodic_sources(sources, kernel.lattice, exterior)
    expected = config.periodic.compile(device=system.device, dtype=config.dtype)
    if expected.receipt() != kernel.receipt():
        raise ValueError('Periodic kernel does not match its declared numerical configuration')
    if config.dyadic_integration is None:
        provider = EwaldSystemBlocks(system, exterior, config.integration, kernel)
    else:
        kernels = {body.region_id: DyadicGreenKernel(body.interior, config.dyadic_integration,
            config.dyadic_cache_max_bytes//2) for body in system.bodies
            if isinstance(body.interior, TensorMedium)}
        provider = GeneralizedEwaldSystemBlocks(system, exterior, config.integration,
                                                kernel, interior_kernels=kernels)
    return provider, identity, payload


class _PeriodicHierarchicalBinding:
    _bind_periodic = PeriodicSystemOperator._bind_periodic
    _validate_periodic_binding = PeriodicSystemOperator._validate_periodic_binding
    query_fields = PeriodicSystemOperator.query_fields
    far_field = PeriodicSystemOperator.far_field
    build_rhs = PeriodicMultipoleSystemOperator.build_rhs

    def _bind_hierarchy(self, system, exterior, sources, config, kernel, identity, payload):
        kernels = ({body.region_id: local for body, local in
                    zip(system.bodies, self.provider.interior_kernels)
                    if isinstance(body.interior, TensorMedium)}
                   if isinstance(self.provider, GeneralizedEwaldSystemBlocks) else None)
        self._bind_periodic(system, exterior, sources, config, kernel, tensor_kernels=kernels)
        self.metadata.update(body_count=len(system.bodies), quadrature=asdict(config.integration),
            numerical_identity=_numerical_identity(config), riesz_storage='local_rwg_v1',
            kernel=kernel.receipt(), operator_identity=identity, operator_identity_payload=payload)

    def _refresh_periodic_hierarchy(self):
        self._validate_periodic_binding()
        if (self.provider.system is not self.system or self.provider.exterior is not self.exterior
                or self.provider.kernel is not self.periodic_kernel
                or self.integration != self.provider.config
                or any(local.integration != self.provider.config for local in self.body_handles)):
            raise ValueError('Periodic hierarchical provider or local integration binding changed')


class PeriodicHierarchicalSystemOperator(_PeriodicHierarchicalBinding,
                                         HierarchicalOperator, SystemDecodeHandle):
    def __init__(self, system, exterior, sources, config, *, kernel):
        if config.operator_backend != 'hmatrix' or config.periodic is None or config.harmonic_balance is not None:
            raise ValueError('Periodic H runtime requires explicit periodic H configuration')
        sources = tuple(sources)
        provider, identity, payload = _provider(system, exterior, sources, config, kernel)
        HierarchicalOperator.__init__(self, provider, config.hierarchical)
        self._bind_hierarchy(system, exterior, sources, config, kernel, identity, payload)

    def refresh(self):
        HierarchicalOperator.refresh(self)
        self._refresh_periodic_hierarchy()


class PeriodicHierarchicalH2SystemOperator(_PeriodicHierarchicalBinding,
                                           HierarchicalH2Operator, SystemDecodeHandle):
    def __init__(self, system, exterior, sources, config, *, kernel):
        if config.operator_backend != 'h2' or config.periodic is None or config.harmonic_balance is not None:
            raise ValueError('Periodic H2 runtime requires explicit periodic H2 configuration')
        sources = tuple(sources)
        provider, identity, payload = _provider(system, exterior, sources, config, kernel)
        source = HierarchicalOperator(provider, config.hierarchical)
        HierarchicalH2Operator.__init__(self, source, config.h2)
        self._bind_hierarchy(system, exterior, sources, config, kernel, identity, payload)

    def refresh(self):
        HierarchicalH2Operator.refresh(self)
        self._refresh_periodic_hierarchy()
