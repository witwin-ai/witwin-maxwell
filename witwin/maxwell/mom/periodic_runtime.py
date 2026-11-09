"""Periodic scalar operator integration with the shared CUDA solve protocol."""
from dataclasses import asdict
import hashlib
import json
import torch
from .kernels import FREE_SPACE
from .linalg import DenseOperator, CompactRWGRiesz, BlockCompactRWGRiesz
from .runtime import MoMConfig, SystemOperator, SystemDecodeHandle, _preflight_dense_capacity, _numerical_identity
from .periodic import EwaldHelmholtzKernel
from .physics import Medium
from .tensor_medium import TensorMedium, sampled_medium_payload
from .dyadic_green import DyadicGreenKernel
from .periodic_operator import assemble_periodic_system, validate_periodic_cell, validate_periodic_sources
from .periodic_fields import decode_periodic
from .lifting import FORMULATION_ID, CURRENT_KIND, KERNEL_DIFFERENCE_VERSION


def periodic_operator_identity(system, exterior, config, kernel):
    """Content identity of the sampled operator, independent of incident amplitude.

    Scene declaration identity belongs to the public compiler. This identity
    includes actual geometry, sampled media, lattice/Bloch and all Ewald and
    numerical settings. Metadata hashing transfers data explicitly to the host;
    operator assembly and solve remain native CUDA.
    """
    if not isinstance(kernel, EwaldHelmholtzKernel):
        raise TypeError('Periodic scalar operator requires EwaldHelmholtzKernel')
    if not isinstance(config, MoMConfig):
        raise TypeError('Periodic scalar operator requires MoMConfig')
    if config.harmonic_balance is not None:
        raise NotImplementedError('Periodic harmonic balance requires its coupled surface-volume runtime')
    if not isinstance(exterior, Medium) or any(not isinstance(body.interior, (Medium, TensorMedium)) for body in system.bodies):
        raise TypeError('Periodic Ewald operators require a scalar exterior and scalar/tensor interiors')
    if config.dyadic_integration is None and any(isinstance(body.interior, TensorMedium) for body in system.bodies):
        raise ValueError('Periodic tensor interiors require explicit dyadic integration')
    validate_periodic_cell(system, kernel.lattice)
    def tensor(value):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'sha256': hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()}
    def medium(value):
        if isinstance(value, TensorMedium):
            return sampled_medium_payload(value)
        return {'frequency_Hz': float(value.frequency), **{
            name: [complex(getattr(value, name)).real, complex(getattr(value, name)).imag]
            for name in ('epsilon_r', 'mu_r', 'k', 'eta')}}
    from .runtime import _representation_identity
    payload = {'version': 'periodic_sampled_operator_identity_v1', 'dtype': str(config.dtype),
        'formulation_id': _representation_identity(config)['formulation_id'], 'current_kind': CURRENT_KIND, 'backend': config.operator_backend,
        'kernel': kernel.receipt(), 'numerical': _numerical_identity(config),
        'exterior': medium(exterior), 'bodies': [
            {'structure_index': body.structure_index, 'region_id': body.region_id,
             'vertices': tensor(body.mesh.vertices), 'faces': tensor(body.mesh.faces),
             'interior': medium(body.interior)} for body in system.bodies]}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


class PeriodicSystemOperator(SystemOperator):
    """Scalar periodic dense reference under the common operator/Riesz protocol.

    Local analytic carriers and interior decoders retain free-space handles.
    The periodic kernel is bound separately and may not be swapped after matrix
    preparation. Physical and cutoff qualification are separate from this solve
    adapter and do not confer production qualification.
    """
    def __init__(self, system, exterior, sources, config, *, kernel, assembly_check=None):
        if not isinstance(kernel, EwaldHelmholtzKernel):
            raise TypeError('Periodic scalar operator requires EwaldHelmholtzKernel')
        sources = tuple(sources)
        validate_periodic_cell(system, kernel.lattice)
        validate_periodic_sources(sources, kernel.lattice, exterior)
        if any(body.interior.frequency != exterior.frequency for body in system.bodies):
            raise ValueError('Every periodic region must share one frequency')
        identity, payload = periodic_operator_identity(system, exterior, config, kernel)
        if config.periodic is not None:
            expected = config.periodic.compile(device=system.device, dtype=config.dtype)
            if expected.receipt() != kernel.receipt():
                raise ValueError('Periodic kernel does not match its declared numerical configuration')
        capacity = _preflight_dense_capacity(system, config)
        kernels = {body.region_id: DyadicGreenKernel(body.interior, config.dyadic_integration,
            config.dyadic_cache_max_bytes//2) for body in system.bodies if isinstance(body.interior, TensorMedium)}
        matrix, metadata = assemble_periodic_system(system, exterior, config.integration,
                                                    kernel=kernel, check=assembly_check, interior_kernels=kernels)
        if matrix.dtype != config.dtype:
            raise ValueError('Periodic matrix precision must match its numerical configuration')
        grams = tuple(body.basis.mass_matrix() for body in system.bodies)
        if config.riesz_storage == 'compact_rwg_v1':
            riesz = BlockCompactRWGRiesz(tuple(CompactRWGRiesz(g, abs(exterior.eta)**2) for g in grams))
        else:
            gram = torch.block_diag(*grams).to(matrix.dtype)
            riesz = torch.block_diag(gram/abs(exterior.eta)**2, gram)
        metadata.update(body_count=len(system.bodies), quadrature=asdict(config.integration), residual_metric='impedance_balanced_L2_dual_proxy', trace_norm_certified=False,
                        memory_preflight=capacity, numerical_identity=_numerical_identity(config),
                        operator_identity=identity, operator_identity_payload=payload)
        if config.riesz_storage == 'compact_rwg_v1':
            metadata['riesz_storage'] = 'block_compact_rwg_v1'
        DenseOperator.__init__(self, matrix, riesz, metadata)
        self._bind_periodic(system, exterior, sources, config, kernel, tensor_kernels=kernels)

    def _bind_periodic(self, system, exterior, sources, config, kernel, *, tensor_kernels=None):
        SystemDecodeHandle.__init__(self, system, exterior, sources, config, self.metadata,
                                    kernel=FREE_SPACE, tensor_kernels=tensor_kernels)
        self.periodic_kernel = kernel
        self._prepared_query_config = config.query_config
        self._bound_periodic_kernel = kernel
        self._bound_system, self._bound_exterior = system, exterior
        self._bound_sources, self._bound_carrier = self.sources, self.incident_carrier
        from .runtime import _representation_identity
        self._periodic_metadata = {**_representation_identity(config),
            'kernel_version': kernel.version, 'physical_qualification': False}
        # Internal adapters may receive the kernel directly without a public
        # periodic declaration; their representation must still be periodic.
        tensor = config.dyadic_integration is not None
        self._periodic_metadata.update(
            kernel_id=f'quasiperiodic_ewald_{kernel.lattice.dimension}d'+('_tensor_interiors_v1' if tensor else ''),
            decoder_version='periodic_tensor_analytic_lifted_fields_v1' if tensor else 'periodic_analytic_lifted_fields_v1')
        self.metadata.update(self._periodic_metadata)

    def refresh(self):
        self._validate_periodic_binding()
        super().refresh()

    def _validate_periodic_binding(self):
        self.validate_tensor_state()
        if self.system is not self._bound_system or self.exterior is not self._bound_exterior:
            raise ValueError('Periodic region binding changed after matrix preparation')
        if self.sources != self._bound_sources or self.incident_carrier is not self._bound_carrier:
            raise ValueError('Periodic source binding changed after preparation')
        if len(self.body_handles) != len(self.system.bodies) or any(
                handle.incident_carrier is not self._bound_carrier or handle.basis is not body.basis
                or handle.interior is not body.interior or handle.exterior is not self.exterior
                for handle, body in zip(self.body_handles, self.system.bodies)):
            raise ValueError('Periodic local region or source binding changed after preparation')
        if self.periodic_kernel is not self._bound_periodic_kernel:
            raise ValueError('Periodic kernel changed after matrix preparation; rebuild the operator')
        self.periodic_kernel.lattice.assert_current()
        if self.kernel is not FREE_SPACE or any(body.kernel is not FREE_SPACE for body in self.body_handles):
            raise ValueError('Periodic analytic lifting requires free-space local handles')

    def build_rhs(self):
        self._validate_periodic_binding()
        validate_periodic_sources(self.sources, self.periodic_kernel.lattice, self.exterior)
        rhs = super().build_rhs()
        # The shared scalar RHS builder records its free-space identity; replace
        # it immediately with the identity of the actual assembled operator.
        self.metadata.update(self._periodic_metadata)
        self.metadata['rhs_identity'] = hashlib.sha256((self.metadata['operator_identity']+
            ':'+self.incident_carrier.identity).encode()).hexdigest()
        return rhs

    def query_fields(self, coefficients, points, region_id, field_kind, trace_side=None, *, query_config=None):
        self._validate_periodic_binding()
        self.last_periodic_field_log = None
        try:
            return decode_periodic(self, coefficients, points, region_id, field_kind, trace_side,
                kernel=self.periodic_kernel,
                query_config=self._prepared_query_config if query_config is None else query_config)
        finally:
            self.last_field_query_log = self.last_periodic_field_log
            self.metadata['last_field_query'] = self.last_field_query_log

    def far_field(self, *args, **kwargs):
        raise ValueError('Periodic results require diffraction or Bloch observables, not isolated-scatterer far fields')


from .fmm_periodic_operator import PeriodicMultipoleOperator


class PeriodicMultipoleSystemOperator(PeriodicMultipoleOperator, SystemDecodeHandle):
    """Experimental public Ewald FMM with the original periodic reference."""
    _bind_periodic = PeriodicSystemOperator._bind_periodic
    _validate_periodic_binding = PeriodicSystemOperator._validate_periodic_binding
    query_fields = PeriodicSystemOperator.query_fields
    far_field = PeriodicSystemOperator.far_field

    def __init__(self, system, exterior, sources, config, *, kernel):
        from .ewald_nearfield import EwaldSystemBlocks
        if config.operator_backend != 'fmm' or config.periodic is None:
            raise ValueError('Periodic FMM requires its explicit periodic configuration')
        sources = tuple(sources)
        identity, payload = periodic_operator_identity(system, exterior, config, kernel)
        validate_periodic_sources(sources, kernel.lattice, exterior)
        expected = config.periodic.compile(device=system.device, dtype=config.dtype)
        if expected.receipt() != kernel.receipt():
            raise ValueError('Periodic kernel does not match its declared numerical configuration')
        construction = config.fmm.construction()
        if construction.pop('kernel_bound_rule') != 'bivariate_cauchy_v1':
            raise NotImplementedError('Real-support FMM kernel bounds require free space')
        for name in ('max_order', 'rwg_absolute_budget', 'kernel_error_budget'):
            if construction.pop(name) is not None:
                raise NotImplementedError('Periodic Ewald FMM requires fixed order without error-budget admission')
        PeriodicMultipoleOperator.__init__(self,
            EwaldSystemBlocks(system, exterior, config.integration, kernel=kernel),
            solve_config=config.fmm.solve_config(), **construction)
        self._bind_periodic(system, exterior, sources, config, kernel)
        self.metadata.update(body_count=len(system.bodies), quadrature=asdict(config.integration),
            numerical_identity=_numerical_identity(config), riesz_storage='local_rwg_v1',
            kernel=kernel.receipt(), operator_identity=identity, operator_identity_payload=payload)

    def refresh(self):
        PeriodicMultipoleOperator.refresh(self)
        self._validate_periodic_binding()
        if self.integration != self.provider.config or any(
                local.integration != self.provider.config for local in self.body_handles):
            raise ValueError('Periodic FMM local integration binding changed')

    def build_rhs(self):
        self.refresh()
        validate_periodic_sources(self.sources, self.periodic_kernel.lattice, self.exterior)
        rhs = SystemOperator.build_rhs(self)
        self.metadata.update(self._periodic_metadata)
        self.metadata['rhs_identity'] = hashlib.sha256((self.metadata['operator_identity']+
            ':'+self.incident_carrier.identity).encode()).hexdigest()
        return rhs


def periodic_dyadic_operator_identity(system, exterior, config, kernel):
    """Bind the actual dyadic representation without scalar Ewald identities."""
    from .periodic_config import MoMDyadicPeriodicConfig
    from .periodic_dyadic import QuasiperiodicDyadicRegularKernel
    from .runtime import _representation_identity
    if (not isinstance(config, MoMConfig) or not isinstance(config.periodic, MoMDyadicPeriodicConfig)
            or not isinstance(kernel, QuasiperiodicDyadicRegularKernel)
            or kernel.medium is not exterior):
        raise TypeError('Periodic dyadic identity requires its exact configuration and exterior binding')
    system.assert_current()
    validate_periodic_cell(system, kernel.lattice)
    kernel._validate()
    def tensor(value):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'sha256': hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()}
    payload = {'version': 'periodic_dyadic_sampled_operator_identity_v1',
        **_representation_identity(config), 'numerical': _numerical_identity(config),
        'regular_kernel': kernel.payload(), 'exterior': sampled_medium_payload(exterior),
        'bodies': [{'structure_index': body.structure_index, 'region_id': body.region_id,
            'vertices': tensor(body.mesh.vertices), 'faces': tensor(body.mesh.faces),
            'interior': sampled_medium_payload(body.interior)} for body in system.bodies]}
    if config.operator_backend in ('hmatrix','h2'):
        payload.update(version='periodic_dyadic_hierarchical_operator_identity_v1',
            residual_metric_policy=periodic_dyadic_residual_metric())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest(), payload


class PeriodicDyadicDecodeHandle(SystemDecodeHandle):
    """Decode-only ownership for an actual periodic tensor exterior."""
    def __init__(self, system, exterior, sources, config, metadata, *, kernel, carrier=None):
        from types import SimpleNamespace
        from .tensor_carrier import compile_tensor_carrier
        from .periodic_config import MoMDyadicPeriodicConfig
        if (not isinstance(config.periodic, MoMDyadicPeriodicConfig)
                or config.dyadic_integration is None or kernel.medium is not exterior
                or not isinstance(exterior, TensorMedium)
                or any(not isinstance(body.interior, TensorMedium) for body in system.bodies)):
            raise TypeError('Periodic dyadic decoding requires tensor regions and explicit dyadic integration')
        system.assert_current()
        validate_periodic_cell(system, kernel.lattice)
        expected = config.periodic.compile(medium=exterior, integration=config.dyadic_integration,
                primary_cache_max_bytes=config.dyadic_cache_max_bytes//2)
        if expected.payload() != kernel.payload():
            raise ValueError('Periodic dyadic kernel differs from its declared numerical configuration')
        self.system, self.exterior, self.sources = system, exterior, tuple(sources)
        self.integration, self._public_config, self.metadata = config.integration, config, metadata
        self.kernel, self.periodic_kernel = FREE_SPACE, kernel
        self.incident_carrier = compile_tensor_carrier(self.sources, exterior) if carrier is None else carrier
        self.incident_carrier.validate_medium(exterior)
        self.incident_carrier.validate_lattice(kernel.lattice)
        if kernel.lattice.dimension == 3:
            raise ValueError('A nonzero real-Bloch 3D homogeneous Maxwell incident mode is a background spectral pole; projected/modal quantities require a separate contract')
        self.primary_kernel = DyadicGreenKernel(exterior, config.dyadic_integration,
                                                config.dyadic_cache_max_bytes//2)
        self.body_handles = tuple(SimpleNamespace(basis=body.basis, interior=body.interior,
            exterior=exterior, incident_carrier=self.incident_carrier, kernel=FREE_SPACE,
            integration=config.integration, metadata={'region_id': body.region_id},
            tensor_kernel=DyadicGreenKernel(body.interior, config.dyadic_integration,
                config.dyadic_cache_max_bytes//2)) for body in system.bodies)
        self._dyadic_binding = self._dyadic_state()

    def _dyadic_state(self):
        def kernel_state(kernel):
            if not isinstance(kernel, DyadicGreenKernel):
                raise ValueError('Periodic dyadic primary or interior kernel changed')
            return (id(kernel), id(kernel.medium), kernel.integration,
                    kernel.cache_max_bytes, id(kernel.mode_cache))
        return (id(self.system), id(self.exterior), id(self.incident_carrier),
            id(self.periodic_kernel), id(self._public_config), self.integration,
            tuple(id(source) for source in self.sources), kernel_state(self.primary_kernel),
            tuple((id(local), id(local.basis), id(local.interior), id(local.exterior),
                   id(local.incident_carrier), id(local.kernel), local.integration,
                   kernel_state(local.tensor_kernel)) for local in self.body_handles))

    def validate_tensor_state(self):
        if self._dyadic_state() != self._dyadic_binding or self.kernel is not FREE_SPACE:
            raise ValueError('Periodic dyadic operator or decoder binding changed')
        self.system.assert_current()
        self.periodic_kernel._validate()
        self.incident_carrier.validate_medium(self.exterior)
        self.incident_carrier.validate_lattice(self.periodic_kernel.lattice)
        for body in self.system.bodies:
            body.interior.assert_current()

    def far_field(self, *args, **kwargs):
        raise ValueError('Periodic tensor results require explicitly defined modal observables')

    def query_fields(self, coefficients, points, region_id, field_kind, trace_side=None, *, query_config=None):
        from .periodic_dyadic_fields import decode_periodic_dyadic
        return decode_periodic_dyadic(self, coefficients, points, region_id, field_kind,
                                     trace_side, query_config=query_config)


class PeriodicDyadicSystemOperator(DenseOperator, PeriodicDyadicDecodeHandle):
    """Complete dense dyadic assembly with a vacuum-impedance residual proxy."""
    def __init__(self, system, exterior, sources, config, *, kernel):
        import math
        from ..constants import EPSILON_0, MU_0
        from .periodic_dyadic_operator import assemble_periodic_dyadic_system
        if config.operator_backend != 'dense' or config.harmonic_balance is not None:
            raise NotImplementedError('Periodic dyadic construction currently requires the linear dense backend')
        metadata = {}
        PeriodicDyadicDecodeHandle.__init__(self, system, exterior, sources, config, metadata, kernel=kernel)
        capacity = _preflight_dense_capacity(system, config)
        identity, payload = periodic_dyadic_operator_identity(system, exterior, config, kernel)
        matrix, receipt = assemble_periodic_dyadic_system(system, reference=kernel,
            config=config.integration, exterior_kernel=self.primary_kernel,
            interior_kernels={body.region_id: local.tensor_kernel
                for body, local in zip(system.bodies, self.body_handles)})
        if matrix.dtype != config.dtype:
            raise ValueError('Periodic dyadic matrix and declared precision differ')
        grams = tuple(body.basis.mass_matrix() for body in system.bodies)
        impedance = math.sqrt(MU_0/EPSILON_0)
        if config.riesz_storage == 'compact_rwg_v1':
            riesz = BlockCompactRWGRiesz(tuple(CompactRWGRiesz(gram, impedance**2) for gram in grams))
        else:
            gram = torch.block_diag(*grams).to(matrix.dtype)
            riesz = torch.block_diag(gram/impedance**2, gram)
        from .runtime import _representation_identity
        metadata.update(receipt, **_representation_identity(config))
        metadata.update(operator_identity=identity, operator_identity_payload=payload,
            memory_preflight=capacity, numerical_identity=_numerical_identity(config),
            body_count=len(system.bodies), quadrature=asdict(config.integration),
            kernel_version=kernel.payload()['version'], kernel=kernel.payload(),
            riesz_storage='block_compact_rwg_v1' if config.riesz_storage == 'compact_rwg_v1' else 'dense',
            residual_metric='vacuum_impedance_balanced_L2_dual_proxy',
            trace_norm_certified=False, physical_qualification=False)
        DenseOperator.__init__(self, matrix, riesz, metadata)

    def refresh(self):
        self.validate_tensor_state()
        DenseOperator.refresh(self)

    def build_rhs(self):
        from .periodic_dyadic_operator import build_periodic_dyadic_rhs
        self.validate_tensor_state()
        rhs, receipt = build_periodic_dyadic_rhs(self.system, self.incident_carrier,
            reference=self.periodic_kernel, config=self.integration,
            spectral_config=self._public_config.dyadic_integration,
            interior_kernels={body.region_id: local.tensor_kernel
                for body, local in zip(self.system.bodies, self.body_handles)},
            rhs_cache_max_bytes=self._public_config.dyadic_cache_max_bytes-
                self._public_config.dyadic_cache_max_bytes//2)
        self.metadata.update(rhs=receipt, incident_carrier_identity=self.incident_carrier.identity,
            rhs_bodies=receipt['regions'], rhs_integration=asdict(self.integration),
            unknown_order=['all_J_scattered_then_all_M_scattered'],
            rhs_identity=hashlib.sha256((self.metadata['operator_identity']+':'+
                self.incident_carrier.identity).encode()).hexdigest())
        return rhs


def periodic_dyadic_residual_metric():
    """Version the fixed public tensor residual proxy, not a physical error norm."""
    import math
    from ..constants import EPSILON_0,MU_0
    return dict(version='vacuum_impedance_rwg_proxy_v1',
        impedance_squared_ohm2=math.sqrt(MU_0/EPSILON_0)**2,trace_norm_certified=False)


class _PeriodicDyadicHierarchyBinding:
    build_rhs=PeriodicDyadicSystemOperator.build_rhs

    def _bind_hierarchical_metadata(self,config):
        from .runtime import _representation_identity
        identity,payload=periodic_dyadic_operator_identity(self.system,self.exterior,config,self.periodic_kernel)
        self.metadata.update(_representation_identity(config))
        self.metadata.update(version='tensor_periodic_'+config.operator_backend+'_operator_v1',
            operator_identity=identity,operator_identity_payload=payload,
            numerical_identity=_numerical_identity(config),body_count=len(self.system.bodies),
            quadrature=asdict(config.integration),kernel_version=self.periodic_kernel.payload()['version'],
            kernel=self.periodic_kernel.payload(),riesz_storage='local_rwg_v1',
            residual_metric='vacuum_impedance_balanced_L2_dual_proxy',
            residual_metric_policy=periodic_dyadic_residual_metric(),
            trace_norm_certified=False,physical_qualification=False)

    def _validate_hierarchical_physics(self):
        self.validate_tensor_state()
        if (self.provider.system is not self.system or self.provider.exterior is not self.exterior
                or self.provider.reference is not self.periodic_kernel
                or self.provider.exterior_kernel is not self.primary_kernel
                or self.provider.config is not self.integration
                or self.metric.impedance_squared!=periodic_dyadic_residual_metric()['impedance_squared_ohm2']
                or any(kernel is not local.tensor_kernel for kernel,local in
                    zip(self.provider.interior_kernels,self.body_handles))):
            raise ValueError('Periodic tensor hierarchy and physical decode bindings disagree')


def _periodic_dyadic_provider(handle):
    from .periodic_dyadic_nearfield import PeriodicDyadicSystemBlocks
    return PeriodicDyadicSystemBlocks(handle.system,handle.periodic_kernel,handle.integration,
        exterior_kernel=handle.primary_kernel,interior_kernels={body.region_id:local.tensor_kernel
            for body,local in zip(handle.system.bodies,handle.body_handles)})


from .hmatrix import HierarchicalOperator
from .h2_operator import HierarchicalH2Operator


class PeriodicDyadicHierarchicalOperator(_PeriodicDyadicHierarchyBinding,HierarchicalOperator,PeriodicDyadicDecodeHandle):
    """Public physical binding of checked tensor-periodic selected H blocks."""
    def __init__(self,system,exterior,sources,config,*,kernel):
        if config.operator_backend!='hmatrix' or config.harmonic_balance is not None:
            raise ValueError('Periodic tensor H construction requires its linear hierarchical configuration')
        PeriodicDyadicDecodeHandle.__init__(self,system,exterior,sources,config,{},kernel=kernel)
        HierarchicalOperator.__init__(self,_periodic_dyadic_provider(self),config.hierarchical,
            metric_impedance=periodic_dyadic_residual_metric()['impedance_squared_ohm2']**.5)
        self._bind_hierarchical_metadata(config)
        self.refresh()

    def refresh(self):
        HierarchicalOperator.refresh(self)
        self._validate_hierarchical_physics()


class PeriodicDyadicHierarchicalH2Operator(_PeriodicDyadicHierarchyBinding,HierarchicalH2Operator,PeriodicDyadicDecodeHandle):
    """Public tensor-periodic H2 with shared physical kernels and exact carrier."""
    def __init__(self,system,exterior,sources,config,*,kernel):
        if config.operator_backend!='h2' or config.harmonic_balance is not None:
            raise ValueError('Periodic tensor H2 construction requires its linear nested configuration')
        PeriodicDyadicDecodeHandle.__init__(self,system,exterior,sources,config,{},kernel=kernel)
        h=HierarchicalOperator(_periodic_dyadic_provider(self),config.hierarchical,
            metric_impedance=periodic_dyadic_residual_metric()['impedance_squared_ohm2']**.5)
        HierarchicalH2Operator.__init__(self,h,config.h2)
        self._bind_hierarchical_metadata(config)
        self.refresh()

    def refresh(self):
        HierarchicalH2Operator.refresh(self)
        self._validate_hierarchical_physics()
