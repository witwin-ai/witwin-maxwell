"""Experimental scalar hierarchical integration in the physical lifted runtime."""
from dataclasses import asdict
import hashlib
import json
from .hmatrix import HierarchicalOperator
from .h2_operator import HierarchicalH2Operator
from .direct_operator import DirectHierarchicalOperator
from .fmm_operator import MultipoleOperator, DirectionalMultipoleOperator
from .nearfield import ScalarSystemBlocks
from .tensor_nearfield import GeneralizedSystemBlocks
from .tensor_medium import TensorMedium, sampled_medium_payload
from .dyadic_green import DyadicGreenKernel


def _physical_provider(system, exterior, config):
    if config.dyadic_integration is None:
        return ScalarSystemBlocks(system, exterior, config.integration)
    kernels = {b.region_id: DyadicGreenKernel(b.interior, config.dyadic_integration,
        config.dyadic_cache_max_bytes//2) for b in system.bodies if isinstance(b.interior, TensorMedium)}
    return GeneralizedSystemBlocks(system, exterior, config.integration, interior_kernels=kernels)

from .runtime import SystemDecodeHandle, SystemOperator, _numerical_identity, _representation_identity
from .kernels import FREE_SPACE


def hierarchical_operator_identity(system, exterior, config):
    """Sampled operator identity, separately from incident-carrier amplitudes."""
    system.assert_current()
    def tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()}
    def medium(value):
        if isinstance(value, TensorMedium):
            return sampled_medium_payload(value)
        return {'frequency': float(value.frequency), **{
            key: [complex(getattr(value, key)).real, complex(getattr(value, key)).imag]
            for key in ('epsilon_r', 'mu_r', 'k', 'eta')}}
    payload = {'version': 'hierarchical_sampled_operator_identity_v1',
        **_representation_identity(config), 'numerical': _numerical_identity(config),
        'kernel': FREE_SPACE.receipt(), 'exterior': medium(exterior), 'bodies': [
            {'region_id': b.region_id, 'structure_index': b.structure_index,
             'vertices': tensor(b.mesh.vertices), 'faces': tensor(b.mesh.faces),
             'interior': medium(b.interior)} for b in system.bodies]}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


class HierarchicalSystemOperator(HierarchicalOperator, SystemDecodeHandle):
    def __init__(self, system, exterior, sources, config):
        if config.operator_backend != 'hmatrix' or config.periodic is not None or config.harmonic_balance is not None:
            raise ValueError('Hierarchical scalar runtime requires its explicit free-space configuration')
        HierarchicalOperator.__init__(self, _physical_provider(system, exterior, config), config.hierarchical)
        self._bind_physical(system, exterior, sources, config)

    def _bind_physical(self, system, exterior, sources, config):
        identity, payload = hierarchical_operator_identity(system, exterior, config)
        SystemDecodeHandle.__init__(self, system, exterior, sources, config, self.metadata, kernel=FREE_SPACE,
            tensor_kernels={b.region_id: k for b, k in zip(system.bodies, self.provider.interior_kernels)}
                if isinstance(self.provider, GeneralizedSystemBlocks) else None)
        self._source_binding = self.sources
        self._carrier_binding = self.incident_carrier
        self._representation = _representation_identity(config)
        self.metadata.update(self._representation)
        self.metadata.update(body_count=len(system.bodies), quadrature=asdict(config.integration),
            numerical_identity=_numerical_identity(config), kernel_version=FREE_SPACE.version,
            riesz_storage='local_rwg_v1', operator_identity=identity, operator_identity_payload=payload,
            residual_metric='impedance_balanced_L2_dual_proxy', trace_norm_certified=False)

    def refresh(self):
        HierarchicalOperator.refresh(self)
        self._refresh_physical()

    def _refresh_physical(self):
        self.validate_tensor_state()
        if (self.system is not self.provider.system or self.exterior is not self.provider.exterior
                or self.sources != self._source_binding or self.incident_carrier is not self._carrier_binding
                or self.integration != self.provider.config
                or self.kernel is not FREE_SPACE or len(self.body_handles) != len(self.system.bodies)):
            raise ValueError('Hierarchical physical region or source binding changed')
        for body, local in zip(self.system.bodies, self.body_handles):
            if (local.basis is not body.basis or local.interior is not body.interior
                    or local.exterior is not self.exterior or local.kernel is not FREE_SPACE
                    or local.integration != self.provider.config
                    or local.incident_carrier is not self._carrier_binding):
                raise ValueError('Hierarchical local decoder binding changed')

    def build_rhs(self):
        self.refresh()
        rhs = SystemOperator.build_rhs(self)
        self.metadata.update(self._representation)
        self.metadata['rhs_identity'] = hashlib.sha256((self.metadata['operator_identity']+
            ':'+self.incident_carrier.identity).encode()).hexdigest()
        return rhs


class HierarchicalH2SystemOperator(HierarchicalH2Operator, SystemDecodeHandle):
    """Physical H2 runtime preserving both construction and conversion identities."""
    _bind_physical = HierarchicalSystemOperator._bind_physical
    _refresh_physical = HierarchicalSystemOperator._refresh_physical
    build_rhs = HierarchicalSystemOperator.build_rhs

    def __init__(self, system, exterior, sources, config):
        if config.operator_backend != 'h2' or config.periodic is not None or config.harmonic_balance is not None:
            raise ValueError('Hierarchical H2 runtime requires its explicit free-space configuration')
        source = HierarchicalOperator(_physical_provider(system, exterior, config), config.hierarchical)
        HierarchicalH2Operator.__init__(self, source, config.h2)
        self._bind_physical(system, exterior, sources, config)

    def refresh(self):
        HierarchicalH2Operator.refresh(self)
        self._refresh_physical()



class DirectH2SystemOperator(DirectHierarchicalOperator, SystemDecodeHandle):
    """Physical scalar runtime with direct nested interpolation construction."""
    _bind_physical = HierarchicalSystemOperator._bind_physical
    _refresh_physical = HierarchicalSystemOperator._refresh_physical
    build_rhs = HierarchicalSystemOperator.build_rhs

    def __init__(self, system, exterior, sources, config):
        if config.operator_backend != 'direct_h2' or config.periodic is not None or config.harmonic_balance is not None:
            raise ValueError('Direct H2 runtime requires its explicit free-space configuration')
        DirectHierarchicalOperator.__init__(self, ScalarSystemBlocks(system, exterior, config.integration),
            solve_config=config.direct_h2.solve_config(), **config.direct_h2.construction())
        self._bind_physical(system, exterior, sources, config)

    def refresh(self):
        DirectHierarchicalOperator.refresh(self)
        self._refresh_physical()


class MultipoleSystemOperator(MultipoleOperator, SystemDecodeHandle):
    """Experimental physical runtime using true Cartesian multipole construction."""
    _bind_physical = HierarchicalSystemOperator._bind_physical
    _refresh_physical = HierarchicalSystemOperator._refresh_physical
    build_rhs = HierarchicalSystemOperator.build_rhs

    def __init__(self, system, exterior, sources, config):
        if config.operator_backend != 'fmm' or config.periodic is not None or config.harmonic_balance is not None:
            raise ValueError('FMM runtime requires its explicit free-space scalar configuration')
        MultipoleOperator.__init__(self, ScalarSystemBlocks(system, exterior, config.integration),
            solve_config=config.fmm.solve_config(), **config.fmm.construction())
        self._bind_physical(system, exterior, sources, config)

    def refresh(self):
        MultipoleOperator.refresh(self)
        self._refresh_physical()


class DirectionalSystemOperator(DirectionalMultipoleOperator, SystemDecodeHandle):
    """Automatic physical scalar directional DAG construction and decode."""
    _bind_physical = HierarchicalSystemOperator._bind_physical
    _refresh_physical = HierarchicalSystemOperator._refresh_physical
    build_rhs = HierarchicalSystemOperator.build_rhs

    def __init__(self, system, exterior, sources, config):
        from .fmm_config import DirectionalFMMConfig
        from .fmm_system import DirectionalSystemActions
        if (config.operator_backend != 'fmm' or type(config.fmm) is not DirectionalFMMConfig
                or config.periodic is not None or config.harmonic_balance is not None
                or config.dyadic_integration is not None):
            raise ValueError('Directional runtime requires explicit free-space scalar configuration')
        provider = ScalarSystemBlocks(system, exterior, config.integration)
        actions = DirectionalSystemActions.from_automatic(provider, config.fmm)
        DirectionalMultipoleOperator.__init__(self, provider, actions,
            solve_config=config.fmm.solve_config())
        self._bind_physical(system, exterior, sources, config)

    def refresh(self):
        DirectionalMultipoleOperator.refresh(self)
        self._refresh_physical()
