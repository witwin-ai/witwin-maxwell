"""Surface preparation and persistence within Scene / Simulation / Result."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import copy
import hashlib
import torch
from witwin.core import Sphere, Ellipsoid, Mesh
from ..monitors import PointMonitor
from ..sources import PointDipole

from .kernels import FREE_SPACE
from .linalg import CompactRWGRiesz, BlockCompactRWGRiesz, DenseOperator, StoppingRule, solve_reference
from .mesh import SurfaceMesh, RWGBasis, SurfaceSystem, sphere_mesh
from .primitive_mesh import FACETED_PRIMITIVES, EdgeGrading, SurfaceMeshConfig, primitive_surface
from .operator import IntegrationConfig, assemble_weak, assemble_system
from .physics import validate_scene, validate_scene_system, source_phasor
from .fields import FieldQueryConfig
from .harmonic_config import HarmonicBalanceConfig
from .linear_volume_state import LinearCoupledConfig
from .periodic_config import MoMPeriodicConfig, MoMDyadicPeriodicConfig
from .hmatrix import HMatrixConfig
from .h2 import H2Config
from .direct_config import DirectH2Config
from .fmm_config import FMMConfig, DirectionalFMMConfig
from .pfft_config import PFFTConfig, admit_pfft
from .dyadic_green import DyadicIntegrationConfig, DyadicGreenKernel
from .tensor_medium import TensorMedium, sampled_medium_payload
from .carrier import IncidentCarrier
from .lifting import FORMULATION_ID, CURRENT_KIND, KERNEL_DIFFERENCE_VERSION, build_lifted_rhs
from .efie_open import OpenPECRuntime, OpenSurfaceConfig, OpenSurfaceData
from .layered.config import LayeredSurfaceConfig
from .layered.runtime import LayeredSurfaceData
from .topology import EFIE_OPEN_PEC, scene_formulation


def _representation_identity(config=None):
    if config is not None and isinstance(config.periodic, MoMDyadicPeriodicConfig):
        return {'formulation_id': 'anisotropic_dgf_sie', 'current_kind': CURRENT_KIND,
                'kernel_id': 'checked_quasiperiodic_dyadic_regular_v1',
                'operator_backend': config.operator_backend,
                'rhs_version': 'periodic_tensor_analytic_lifted_rhs_v1',
                'decoder_version': 'periodic_dyadic_analytic_lifted_fields_v1'}
    if config is not None and config.dyadic_integration is not None:
        if config.periodic is not None:
            return {'formulation_id': 'anisotropic_dgf_sie', 'current_kind': CURRENT_KIND,
                    'kernel_id': f'quasiperiodic_ewald_{len(config.periodic.vectors_m)}d_tensor_interiors_v1',
                    'operator_backend': config.operator_backend,
                    'rhs_version': 'tensor_factored_spatial_contrast_candidate_v2',
                    'decoder_version': 'periodic_tensor_analytic_lifted_fields_v1'}
        return {'formulation_id': 'anisotropic_dgf_sie', 'current_kind': CURRENT_KIND,
                'kernel_id': 'scalar_exterior_tensor_interiors_v1', 'operator_backend': config.operator_backend,
                'rhs_version': 'tensor_factored_spatial_contrast_candidate_v2',
                'decoder_version': 'tensor_analytic_lift_principal_value_candidate_v2'}
    if config is not None and config.periodic is not None:
        return {'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
                'kernel_id': f'quasiperiodic_ewald_{len(config.periodic.vectors_m)}d', 'operator_backend': config.operator_backend,
                'rhs_version': KERNEL_DIFFERENCE_VERSION,
                'decoder_version': 'periodic_analytic_lifted_fields_v1'}
    return {'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
            'kernel_id': 'free_space_helmholtz', 'operator_backend': 'dense' if config is None else config.operator_backend,
            'rhs_version': KERNEL_DIFFERENCE_VERSION,
            'decoder_version': 'analytic_incident_lift_multibody_complete_field_v2'}


def _same_metadata(actual, expected):
    """Compare typed scalar metadata without bool/numeric equality aliases."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_metadata(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(_same_metadata(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _numerical_identity(config):
    result = {'subdivisions': config.subdivisions, 'dtype': str(config.dtype),
            'riesz_storage': config.riesz_storage, 'integration': asdict(config.integration),
            'stopping': asdict(config.stopping), 'query_config': asdict(config.query_config)}
    if config.dyadic_integration is not None:
        result.update(dyadic_integration=config.dyadic_integration.payload(), dyadic_cache_max_bytes=config.dyadic_cache_max_bytes)
    if config.periodic is not None:
        result['periodic'] = config.periodic.payload()
    if config.operator_backend in ('hmatrix', 'h2'):
        result.update(operator_backend=config.operator_backend, hierarchical=config.hierarchical.payload())
    if config.operator_backend == 'h2':
        result['h2'] = config.h2.payload()
    if config.operator_backend == 'direct_h2':
        result.update(operator_backend='direct_h2', direct_h2=config.direct_h2.payload())
    if config.operator_backend == 'fmm':
        result.update(operator_backend='fmm', fmm=config.fmm.payload())
    if config.surface_mesh is not None:
        result['surface_mesh'] = config.surface_mesh.payload()
    if config.operator_backend == 'pfft':
        result.update(operator_backend='pfft', pfft=config.pfft.payload())
    return result


@dataclass(frozen=True)
class MoMConfig:
    subdivisions: int = 1
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    stopping: StoppingRule = field(default_factory=lambda: StoppingRule(method='direct'))
    dtype: torch.dtype = torch.complex128
    query_config: FieldQueryConfig = field(default_factory=FieldQueryConfig)
    riesz_storage: str = 'dense'
    harmonic_balance: HarmonicBalanceConfig | None = None
    periodic: MoMPeriodicConfig | MoMDyadicPeriodicConfig | None = None
    operator_backend: str = 'dense'
    hierarchical: HMatrixConfig | None = None
    h2: H2Config | None = None
    direct_h2: DirectH2Config | None = None
    fmm: FMMConfig | DirectionalFMMConfig | None = None
    # Optional factory with bind(PreconditionerContext) -> object with apply(residual).
    # It changes only the Krylov path (flexible right GMRES), never the operator identity.
    right_preconditioner: object | None = None
    pfft: PFFTConfig | None = None
    dyadic_integration: DyadicIntegrationConfig | None = None
    dyadic_cache_max_bytes: int = 256*1024**2
    linear_coupled: LinearCoupledConfig | None = None
    # Sheet discretization for the open-surface PEC EFIE; other formulations reject it.
    open_surface: OpenSurfaceConfig | None = None
    # Mesh-size control for faceted Box/Cylinder/Cone/Prism bodies; None uses
    # SurfaceMeshConfig() defaults. ``subdivisions`` applies to Sphere/Ellipsoid.
    surface_mesh: SurfaceMeshConfig | None = None
    # Planar PEC sheets in a LayerStack background (layered MPIE network solver).
    layered: LayeredSurfaceConfig | None = None

    def __post_init__(self):
        if self.layered is not None and type(self.layered) is not LayeredSurfaceConfig:
            raise TypeError('layered must be LayeredSurfaceConfig or None')
        if self.open_surface is not None and not isinstance(self.open_surface, OpenSurfaceConfig):
            raise TypeError('open_surface must be OpenSurfaceConfig')
        if self.surface_mesh is not None and type(self.surface_mesh) is not SurfaceMeshConfig:
            raise TypeError('surface_mesh must be SurfaceMeshConfig or None')
        if self.linear_coupled is not None:
            if type(self.linear_coupled) is not LinearCoupledConfig:
                raise TypeError('linear_coupled must be LinearCoupledConfig')
            if self.harmonic_balance is not None:
                raise ValueError('Linear zero-induced volumes cannot include harmonic balance')
        if isinstance(self.periodic, MoMDyadicPeriodicConfig):
            if self.dyadic_integration is None:
                raise ValueError('Periodic dyadic exterior requires explicit primary dyadic integration')
            if self.operator_backend not in ('dense','hmatrix','h2') or self.harmonic_balance is not None:
                raise NotImplementedError('Periodic dyadic exterior requires linear dense/hmatrix/h2 execution')
        if self.dyadic_integration is not None:
            if not isinstance(self.dyadic_integration, DyadicIntegrationConfig):
                raise TypeError('dyadic_integration must be DyadicIntegrationConfig')
            if self.dyadic_integration.spatial_rule in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1') and self.periodic is not None:
                raise NotImplementedError('Factored SPD spatial tails require free-space execution')
            if self.operator_backend not in ('dense', 'hmatrix', 'h2'):
                raise NotImplementedError('Tensor interiors require dense/hmatrix/h2')
            if self.harmonic_balance is not None and self.periodic is not None:
                raise NotImplementedError('Mixed tensor harmonic balance requires free-space execution')
        if type(self.dyadic_cache_max_bytes) is not int or self.dyadic_cache_max_bytes < 2:
            raise ValueError('dyadic_cache_max_bytes must allow separate regional and RHS caches')
        if self.dyadic_integration is None and self.dyadic_cache_max_bytes != 256*1024**2:
            raise ValueError('Custom dyadic cache requires explicit dyadic_integration')
        if self.operator_backend not in ('dense', 'hmatrix', 'h2', 'direct_h2', 'fmm', 'pfft'):
            raise NotImplementedError('This MoM runtime supports dense or experimental scalar hmatrix/h2/direct_h2/fmm/pfft only')
        if self.operator_backend in ('hmatrix', 'h2', 'direct_h2', 'fmm', 'pfft'):
            if ((self.periodic is not None and self.operator_backend not in ('hmatrix', 'h2', 'fmm'))
                    or (self.harmonic_balance is not None and self.operator_backend not in ('hmatrix','h2'))):
                raise NotImplementedError('Hierarchical periodic and harmonic-balance compositions require dedicated integration')
            if self.harmonic_balance is not None and (not isinstance(self.stopping,StoppingRule)
                    or self.stopping.method not in ('gmres','block_gmres')
                    or self.stopping.recycle_size or self.stopping.preconditioner not in ('none','row_column_norm')):
                raise ValueError('Compressed harmonic balance requires unrecycled GMRES with none or row-column scaling')
            if self.operator_backend in ('hmatrix', 'h2'):
                if self.hierarchical is None:
                    object.__setattr__(self, 'hierarchical', HMatrixConfig())
                if not isinstance(self.hierarchical, HMatrixConfig):
                    raise TypeError('hierarchical must be HMatrixConfig')
            elif self.hierarchical is not None:
                raise ValueError('Direct construction does not use a reference hierarchical configuration')
            if self.riesz_storage != 'local_rwg_v1':
                raise ValueError('Hierarchical backends require riesz_storage=local_rwg_v1')
            if (not isinstance(self.stopping, StoppingRule) or self.stopping.method == 'direct'
                    or self.stopping.fallback != 'none'):
                raise ValueError('Hierarchical backends require iterative stopping without a direct fallback')
        elif self.hierarchical is not None or self.riesz_storage == 'local_rwg_v1':
            raise ValueError('Hierarchical configuration and local Riesz storage require operator_backend=hmatrix or h2')
        if self.operator_backend == 'direct_h2':
            if self.direct_h2 is None:
                object.__setattr__(self, 'direct_h2', DirectH2Config())
            if not isinstance(self.direct_h2, DirectH2Config):
                raise TypeError('direct_h2 must be DirectH2Config')
            if self.stopping.preconditioner not in ('none', 'rwg_diagonal'):
                raise ValueError('Direct H2 requires none or rwg_diagonal preconditioning')
        elif self.direct_h2 is not None:
            raise ValueError('Direct H2 configuration requires operator_backend=direct_h2')
        if self.operator_backend == 'fmm':
            if self.fmm is None:
                object.__setattr__(self, 'fmm', FMMConfig())
            if not isinstance(self.fmm, (FMMConfig, DirectionalFMMConfig)):
                raise TypeError('fmm must be FMMConfig or DirectionalFMMConfig')
            if isinstance(self.fmm, DirectionalFMMConfig):
                if self.periodic is not None:
                    raise NotImplementedError('Automatic directional FMM currently requires free space')
                if self.dtype != torch.complex128:
                    raise ValueError('Automatic directional FMM requires complex128 currents and float64 coordinates')
                if self.stopping.preconditioner not in ('none', 'rwg_diagonal'):
                    raise ValueError('FMM requires none or rwg_diagonal preconditioning')
            else:
                if self.fmm.expansion_center_policy in ('dyadic_cell_v1','binary_lattice_v1'):
                    if self.periodic is not None:
                        raise NotImplementedError('Dyadic cell FMM centers currently require free space')
                    if self.dtype != torch.complex128:
                        raise ValueError('Dyadic cell FMM centers require complex128 currents')
                if self.fmm.far_action_batch > 1:
                    if self.periodic is not None:
                        raise NotImplementedError('Batched compact FAR currently requires free space')
                    if self.dtype != torch.complex128:
                        raise ValueError('Batched compact FAR requires complex128 currents')
                if self.fmm.table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
                    if self.periodic is not None:
                        raise NotImplementedError('Shared compact FMM currently requires free space')
                    if self.dtype!=torch.complex128:
                        raise ValueError('Shared compact FMM requires complex128 currents and float64 coordinates')
                if self.fmm.near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
                    if self.periodic is not None:
                        raise NotImplementedError('Scalar D/K FMM near storage currently requires free space')
                    if self.dtype != torch.complex128:
                        raise ValueError('Scalar D/K FMM requires complex128 currents and float64 coordinates')
                if self.stopping.preconditioner not in ('none', 'rwg_diagonal'):
                    raise ValueError('FMM requires none or rwg_diagonal preconditioning')
                if self.fmm.table_storage == 'fft_ewald_v1':
                    if self.periodic is None:
                        raise ValueError('Ewald FFT FMM requires an explicit periodic kernel configuration')
                    if self.dtype != torch.complex128:
                        raise ValueError('Ewald FFT FMM requires complex128 currents and float64 coordinates')
                if self.periodic is not None and self.fmm.table_storage == 'fft_v1':
                    raise NotImplementedError('FFT FMM currently requires free space; screened Ewald FFT is not admitted')
                if self.periodic is not None and self.fmm.kernel_bound_rule != 'bivariate_cauchy_v1':
                    raise NotImplementedError('Real-support FMM kernel bounds require free space')
                if self.periodic is not None and any(value is not None for value in
                        (self.fmm.max_order, self.fmm.rwg_absolute_budget, self.fmm.kernel_error_budget)):
                    raise NotImplementedError('Periodic Ewald FMM currently requires fixed order without error-budget admission')
        elif self.fmm is not None:
            raise ValueError('FMM configuration requires operator_backend=fmm')
        if self.operator_backend == 'pfft':
            if self.pfft is None:
                object.__setattr__(self, 'pfft', PFFTConfig())
            if type(self.pfft) is not PFFTConfig:
                raise TypeError('pfft must be PFFTConfig')
            admit_pfft(self)
        elif self.pfft is not None:
            raise ValueError('pFFT configuration requires operator_backend=pfft')
        if self.operator_backend == 'h2':
            if self.h2 is None:
                object.__setattr__(self, 'h2', H2Config())
            if not isinstance(self.h2, H2Config):
                raise TypeError('h2 must be H2Config')
        elif self.h2 is not None:
            raise ValueError('H2 configuration requires operator_backend=h2')
        if self.operator_backend in ('hmatrix', 'h2', 'direct_h2', 'fmm'):
            solver_config = getattr(self, 'hierarchical' if self.operator_backend == 'hmatrix' else self.operator_backend)
            if self.stopping.rtol > 0.1 * solver_config.solve_dual_tolerance:
                raise ValueError('Hierarchical/FMM stopping.rtol must be <= 0.1 * solve_dual_tolerance: inner GMRES uses the Euclidean norm while admission uses the Riesz dual norm')
        if self.periodic is not None and not isinstance(self.periodic, (MoMPeriodicConfig, MoMDyadicPeriodicConfig)):
            raise TypeError('MoM periodic must be a supported scalar or dyadic periodic configuration')
        if self.periodic is not None and self.harmonic_balance is not None:
            raise NotImplementedError('Periodic harmonic balance requires coupled runtime integration')
        if self.harmonic_balance is not None and not isinstance(self.harmonic_balance,HarmonicBalanceConfig):
            raise TypeError('MoM harmonic_balance must be a HarmonicBalanceConfig')
        if (self.harmonic_balance is not None and self.operator_backend not in ('dense','hmatrix','h2')
                and (self.harmonic_balance.tensor_face_rule!='triangle' or self.harmonic_balance.tensor_kernel_batch!=128)):
            raise NotImplementedError('Explicit tensor volume quadrature policy requires dense or H/H2 HB preparation')
        if isinstance(self.subdivisions, bool) or not isinstance(self.subdivisions, int) or self.subdivisions < 0:
            raise ValueError('subdivisions must be a nonnegative integer')
        if not isinstance(self.integration, IntegrationConfig) or not isinstance(self.stopping, StoppingRule):
            raise TypeError('MoM integration and stopping must use IntegrationConfig and StoppingRule')
        if not isinstance(self.query_config, FieldQueryConfig):
            raise TypeError('MoM query_config must be a FieldQueryConfig')
        if self.dtype not in (torch.complex64, torch.complex128):
            raise ValueError('MoM dtype must be complex64 or complex128')
        if not isinstance(self.riesz_storage, str) or self.riesz_storage not in ('dense', 'compact_rwg_v1', 'local_rwg_v1'):
            raise ValueError('MoM riesz_storage must be dense, compact_rwg_v1 or local_rwg_v1')
        if self.right_preconditioner is not None:
            if not callable(getattr(self.right_preconditioner, 'bind', None)):
                raise TypeError('right_preconditioner must provide bind(context)')
            if (self.stopping.method != 'gmres' or self.stopping.recycle_size
                    or self.stopping.fallback != 'none'):
                raise ValueError('A right preconditioner requires method=gmres without recycling or fallback')
            if any(value is not None for value in (self.periodic, self.harmonic_balance,
                                                    self.linear_coupled, self.dyadic_integration)):
                raise NotImplementedError('Right preconditioners currently require the free-space scalar linear runtime')
            if self.fmm is not None and self.fmm.leaf_factor_kernel != 'none':
                raise ValueError('Use either FMM leaf factors or a right preconditioner')


def _signature(scene, exterior, interiors):
    shapes = []
    for index, structure in enumerate(scene.structures):
        if not structure.enabled:
            continue
        geometry = structure.geometry
        shape = {'structure_index': index, 'name': structure.name, 'geometry': geometry.kind,
                 'position': geometry.position.detach().cpu().tolist(),
                 'rotation': geometry.rotation.detach().cpu().tolist()}
        if isinstance(geometry, Mesh):
            shape.update({'vertices_sha256': hashlib.sha256(geometry.vertices.detach().cpu().numpy().tobytes()).hexdigest(),
                          'faces_sha256': hashlib.sha256(geometry.faces.detach().cpu().numpy().tobytes()).hexdigest(),
                          'scale': geometry.scale.detach().cpu().tolist(), 'recenter': geometry.recenter,
                          'fill_mode': geometry.fill_mode, 'surface_thickness': geometry.surface_thickness,
                          'input_vertex_dtype': str(geometry.vertices.dtype)})
        elif type(geometry) in FACETED_PRIMITIVES:
            for name in ('size', 'radius', 'height', 'axis', 'num_sides'):
                if hasattr(geometry, name):
                    value = getattr(geometry, name)
                    shape[name] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
        else:
            size = geometry.radius if isinstance(geometry, Sphere) else geometry.radii
            shape['size'] = size.detach().cpu().tolist()
        shapes.append(shape)
    if isinstance(exterior, TensorMedium):
        from .tensor_carrier import compile_tensor_carrier
        sources = compile_tensor_carrier(scene.sources, exterior).payload()
    else:
        sources = [{'kind': 'point_dipole', 'position': tuple(s.position), 'polarization': tuple(s.polarization),
                    'amplitude': source_phasor(s, exterior.frequency)} if isinstance(s, PointDipole) else
                   {'direction': tuple(s.direction), 'polarization': tuple(s.polarization),
                    'amplitude': source_phasor(s, exterior.frequency)} for s in scene.sources]
    return {'bodies': shapes, 'exterior': sampled_medium_payload(exterior), 'interiors': [sampled_medium_payload(m) for m in interiors],
            'sources': sources,
            'source_declarations': tuple(repr(source) for source in scene.sources),
            'domain': repr(scene.domain), 'boundary': repr(scene.boundary), 'symmetry': scene.symmetry,
            'monitors': tuple(repr(monitor) for monitor in scene.monitors)}


def _restore_surface_mesh(record):
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != set(SurfaceMeshConfig().payload()):
        raise ValueError('MoM surface_mesh payload is malformed')
    grading = record['edge_grading']
    if grading is not None:
        if not isinstance(grading, dict) or set(grading) != {'min_size', 'ratio'}:
            raise ValueError('MoM surface_mesh edge grading payload is malformed')
        grading = EdgeGrading(**grading)
    return SurfaceMeshConfig(**{**record, 'edge_grading': grading})


def compile_surface(scene, frequency, config):
    exterior, interior = validate_scene(scene, frequency)
    _validate_surface_queries(scene)
    geometry = next(s.geometry for s in scene.structures if s.enabled)
    return _compile_basis(scene, geometry, config, (exterior, interior)), exterior, interior


def _validate_surface_queries(scene):
    if torch.device(scene.device).type != 'cuda':
        raise ValueError('MoM preparation requires a CUDA Scene')
    if any(not isinstance(monitor, PointMonitor) for monitor in scene.monitors):
        raise ValueError('MoM supports PointMonitor E/H only; other monitors require explicit query APIs')
    names = [monitor.name for monitor in scene.monitors]
    if len(set(names)) != len(names):
        raise ValueError('MoM PointMonitor names must be unique')
    for monitor in scene.monitors:
        if any(component.lower() not in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz') for component in monitor.fields):
            raise ValueError('MoM PointMonitor supports E/H vector components only')


def _compile_basis(scene, geometry, config, media=None):
    """Compile one body. ``media`` (exterior, interior) sizes faceted primitives by wavelength."""
    if type(geometry) not in (Sphere, Ellipsoid, Mesh, *FACETED_PRIMITIVES):
        raise ValueError('MoM surface compilation supports Sphere, Ellipsoid, Box, Cylinder, Cone, Prism or '
                         f'explicit solid Mesh geometry; {type(geometry).__name__} is not supported')
    dtype = torch.float64 if config.dtype == torch.complex128 else torch.float32
    if type(geometry) in FACETED_PRIMITIVES:
        surface_mesh = SurfaceMeshConfig() if config.surface_mesh is None else config.surface_mesh
        return RWGBasis(primitive_surface(geometry, surface_mesh, media, device=scene.device, dtype=dtype))
    if isinstance(geometry, Mesh):
        if geometry.fill_mode != 'solid' or geometry.recenter or geometry.surface_thickness is not None:
            raise ValueError("MoM Mesh requires fill_mode='solid', recenter=False, and no surface thickness")
        vertices = geometry._transform_mesh_verts(geometry.vertices.to(device=scene.device, dtype=dtype) * geometry.scale.to(device=scene.device, dtype=dtype))
        faces = geometry.faces.to(device=scene.device)
        return RWGBasis(SurfaceMesh(vertices, faces))
    unit = sphere_mesh(subdivisions=config.subdivisions, device=scene.device, dtype=dtype)
    size = geometry.radius if isinstance(geometry, Sphere) else geometry.radii
    size = size.to(device=scene.device, dtype=dtype)
    if not bool(torch.isfinite(size).all() & (size > 0).all()):
        raise ValueError('MoM geometry dimensions must be positive and finite in metres')
    vertices = geometry._transform_mesh_verts(unit.vertices * size)
    mesh = SurfaceMesh(vertices, unit.faces)
    return RWGBasis(mesh)


def _point_dipole_scope(config):
    """Exterior dipoles use the linear dense free-space scalar lifting only."""
    return (config.operator_backend == 'dense' and config.periodic is None and config.dyadic_integration is None
            and config.harmonic_balance is None and config.linear_coupled is None)


def _validate_point_dipoles(scene, system, config):
    """Fail closed for dipoles inside, on, or too close to any body surface."""
    from .dipole import DIPOLE_MINIMUM_SOURCE_ORDER, validate_point_dipole_clearance
    dipoles = [source for source in scene.sources if isinstance(source, PointDipole)]
    if not dipoles:
        return
    order = min(config.integration.order, config.integration.singular_order)
    if order < DIPOLE_MINIMUM_SOURCE_ORDER:
        raise ValueError(f'MoM PointDipole sources require IntegrationConfig order and singular_order >= '
                         f'{DIPOLE_MINIMUM_SOURCE_ORDER}; lower source-panel rules under-integrate the '
                         'singular dipole trace at the admitted clearance.')
    for dipole in dipoles:
        for body in system.bodies:
            validate_point_dipole_clearance(dipole.position, body.mesh.triangles, body_name=body.name)


def compile_surfaces(scene, frequency, config):
    """Compile separately validated bodies in enabled Scene declaration order."""
    if config.open_surface is not None:
        raise ValueError('MoMConfig.open_surface applies only to scenes of open PEC sheets')
    exterior, interiors = validate_scene_system(scene, frequency,
        tensor_materials=config.dyadic_integration is not None,
        tensor_exterior=isinstance(config.periodic, MoMDyadicPeriodicConfig), dtype=config.dtype,
        point_dipoles=_point_dipole_scope(config))
    _validate_surface_queries(scene)
    enabled = [(index, structure) for index, structure in enumerate(scene.structures) if structure.enabled]
    entries = [(index, structure.name, _compile_basis(scene, structure.geometry, config, (exterior, interior)), interior)
               for (index, structure), interior in zip(enabled, interiors)]
    system = SurfaceSystem.create(entries)
    _validate_point_dipoles(scene, system, config)
    return system, exterior


@dataclass
class DecodeHandle:
    basis: RWGBasis
    exterior: object
    interior: object
    sources: tuple
    integration: IntegrationConfig
    metadata: dict
    incident_carrier: IncidentCarrier = field(init=False)
    kernel: object = field(default=FREE_SPACE, init=False)

    def __post_init__(self):
        self.incident_carrier = IncidentCarrier.compile(self.sources, self.exterior)


def _preflight_dense_capacity(basis, config):
    """Reject impossible resident storage before quadratic assembly allocations.

    This is a lower bound, not a workspace or fragmentation guarantee. PyTorch's
    unused reserved bytes are reusable; live tensors are already excluded by
    the driver's free-memory observation. No cache eviction is performed.
    """
    if isinstance(basis, SurfaceSystem):
        edges, faces = basis.total_edges, max(len(body.mesh.faces) for body in basis.bodies)
        gram_entries = sum(body.basis.n_edges**2 for body in basis.bodies)
        device = basis.device
    else:
        edges, faces = basis.n_edges, len(basis.mesh.faces)
        gram_entries = edges**2
        device = basis.mesh.device
    complex_bytes = 16 if config.dtype == torch.complex128 else 8
    real_bytes = complex_bytes // 2
    matrix = 4 * edges**2 * complex_bytes
    metric = (2 * gram_entries * real_bytes if config.riesz_storage == 'compact_rwg_v1'
              else 2 * matrix)
    factor = matrix if ((config.stopping.method == 'direct' or config.stopping.fallback == 'direct')
                        and config.stopping.direct_storage == 'resident') else 0
    # Classifier rows, columns, shared-count int64 arrays and near bool mask
    # coexist; geometry broadcasts and other temporaries are not included.
    pairs = faces**2 * (3 * 8 + 1)
    minimum = max(matrix + metric + factor, pairs)
    free, total = torch.cuda.mem_get_info(device)
    reusable = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    available = min(total, free + reusable)
    receipt = {'minimum_required_bytes': minimum, 'driver_free_bytes': free,
               'reusable_reserved_bytes': reusable, 'available_bytes': available,
               'matrix_bytes': matrix, 'metric_and_factor_bytes': metric,
               'direct_factor_bytes': factor, 'pair_storage_lower_bound_bytes': pairs,
               'workspace_included': False, 'device': str(device)}
    if minimum > available:
        raise torch.cuda.OutOfMemoryError(
            f'MoM dense capacity preflight: minimum {minimum} bytes exceeds available '
            f'{available} bytes on {device}; quadrature/workspace require additional memory')
    return receipt


class SurfaceOperator(DenseOperator):
    def __init__(self, basis, exterior, interior, sources, config, *, kernel=FREE_SPACE):
        if kernel.receipt() != FREE_SPACE.receipt():
            raise ValueError('Public analytic lifting currently requires the free-space kernel contract')
        sources = tuple(sources)
        carrier = IncidentCarrier.compile(sources, exterior)
        capacity = _preflight_dense_capacity(basis, config)
        matrix, metadata = assemble_weak(basis, exterior, interior, config.integration, kernel=kernel)
        mass = basis.mass_matrix()
        # Rows carry H and E testing units respectively. Impedance scaling puts
        # the two L2 dual residual proxies on a common electric-field scale.
        if config.riesz_storage == 'compact_rwg_v1':
            riesz = CompactRWGRiesz(mass, abs(exterior.eta)**2)
        else:
            mass = mass.to(matrix.dtype)
            riesz = torch.block_diag(mass / abs(exterior.eta)**2, mass)
        metadata.update({'residual_metric': 'impedance_balanced_L2_dual_proxy',
                         'trace_norm_certified': False, 'coordinate_units': 'm',
                         'memory_preflight': capacity})
        if config.riesz_storage != 'dense':
            metadata['riesz_storage'] = config.riesz_storage
        super().__init__(matrix, riesz, metadata)
        self.basis, self.exterior, self.interior = basis, exterior, interior
        self.sources, self.integration = tuple(sources), config.integration
        self.incident_carrier = carrier
        self.kernel = kernel


class SystemDecodeHandle:
    """A decode-only global body layout; restoring it never assembles a matrix."""
    def __init__(self, system, exterior, sources, config, metadata, *, kernel=FREE_SPACE, tensor_kernels=None):
        self.system, self.exterior, self.integration, self.kernel = system, exterior, config.integration, kernel
        self._public_config = config
        self.sources, self.metadata = tuple(sources), metadata
        self.incident_carrier = IncidentCarrier.compile(self.sources, exterior)
        self.body_handles = tuple(DecodeHandle(body.basis, exterior, body.interior, self.sources, config.integration,
            {'region_id': body.region_id, 'quadrature': asdict(config.integration)}) for body in system.bodies)
        tensor_kernels = {} if tensor_kernels is None else dict(tensor_kernels)
        for region, handle in enumerate(self.body_handles, 1):
            handle.incident_carrier = self.incident_carrier
            handle.kernel = kernel
            if isinstance(handle.interior, TensorMedium):
                handle.tensor_kernel = tensor_kernels.get(region)
                if handle.tensor_kernel is None:
                    handle.tensor_kernel = DyadicGreenKernel(handle.interior, config.dyadic_integration,
                        config.dyadic_cache_max_bytes//2)
                if (handle.tensor_kernel.medium is not handle.interior
                        or handle.tensor_kernel.integration != config.dyadic_integration
                        or handle.tensor_kernel.cache_max_bytes != config.dyadic_cache_max_bytes//2):
                    raise ValueError('Tensor construction and decoder kernels must share sampled identity and budgets')
                handle.tensor_rhs_cache_max_bytes = config.dyadic_cache_max_bytes-config.dyadic_cache_max_bytes//2
                from .dyadic_contrast import DyadicContrastKernel
                eye = torch.eye(3, device=system.device, dtype=config.dtype)
                outside = TensorMedium(exterior.frequency, exterior.epsilon_r*eye, exterior.mu_r*eye)
                handle.tensor_contrast_kernel = DyadicContrastKernel(outside, handle.interior,
                    config.dyadic_integration, handle.tensor_rhs_cache_max_bytes)
        self._tensor_decode_binding = self._tensor_state()

    def _tensor_state(self):
        from .dyadic_contrast import DyadicContrastKernel
        records = []
        for h in self.body_handles:
            if not isinstance(h.interior, TensorMedium):
                continue
            k, c = getattr(h, 'tensor_kernel', None), getattr(h, 'tensor_contrast_kernel', None)
            if not isinstance(k, DyadicGreenKernel) or not isinstance(c, DyadicContrastKernel):
                raise ValueError('Tensor decoder or contrast kernel binding changed')
            records.append((id(h), id(k), id(k.medium), k.integration, k.cache_max_bytes,
                id(k.mode_cache), k._isotropic, h.tensor_rhs_cache_max_bytes, id(c), id(c.exterior),
                id(c.interior), c.integration, c.cache_max_bytes, id(c.mode_cache)))
        return tuple(records)

    def validate_tensor_state(self):
        if self._tensor_state() != self._tensor_decode_binding:
            raise ValueError('Tensor decoder or contrast kernel binding changed')
        for h in self.body_handles:
            if isinstance(h.interior, TensorMedium):
                h.interior.assert_current()
                h.tensor_contrast_kernel.exterior.assert_current()

    def _record_rhs(self, handle, records):
        handle.metadata.update(_representation_identity(self._public_config))
        # Operators with a construction identity bind every RHS to it.
        if 'operator_identity' in self.metadata:
            handle.metadata['rhs_identity'] = hashlib.sha256((self.metadata['operator_identity']+
                ':'+handle.incident_carrier.identity).encode()).hexdigest()
        handle.metadata.update(incident_carrier_identity=handle.incident_carrier.identity,
                               rhs_bodies=records, rhs_integration=asdict(self.integration),
                               unknown_order=['all_J_scattered_then_all_M_scattered'])

    def build_rhs_batch(self, source_groups):
        """Lifted RHS rows [S,N] and decode-only handles for several source sets.

        Each group of sources is one physical RHS with its own carrier and
        decode handle, recorded exactly as ``build_rhs`` records its own.
        Scalar bodies integrate all carriers in one pass that shares
        classification, quadrature and difference kernels: dense backends use
        the global pair classes, compressed backends the bounded selected rows.
        """
        from .lifting import build_lifted_rhs_batch, build_selected_lifted_rhs_batch
        self.system.assert_current()
        self.validate_tensor_state()
        config = self._public_config
        if config.periodic is not None:
            raise NotImplementedError('Batched lifted RHS is not implemented for periodic lattices; '
                                      'use one prepared Simulation per source')
        source_groups = tuple(tuple(group) for group in source_groups)
        for group in source_groups:
            if any(isinstance(source, PointDipole) for source in group):
                if not _point_dipole_scope(config):
                    raise ValueError('MoM PointDipole sources require the dense free-space scalar lifting')
                # Same order and clearance admission as a prepared Scene source.
                from types import SimpleNamespace
                _validate_point_dipoles(SimpleNamespace(sources=group), self.system, config)
        tensor_kernels = {region: h.tensor_kernel for region, h in enumerate(self.body_handles, 1)
                          if isinstance(h.interior, TensorMedium)}
        decoders = tuple(SystemDecodeHandle(self.system, self.exterior, tuple(group), config,
                                            dict(self.metadata), kernel=self.kernel, tensor_kernels=tensor_kernels)
                         for group in source_groups)
        if not decoders:
            raise ValueError('Batched lifted RHS requires at least one source group')
        rhs = torch.empty((len(decoders), 2*self.system.total_edges), device=self.system.device, dtype=config.dtype)
        records = [[] for _ in decoders]
        for i, body in enumerate(self.system.bodies):
            handles = tuple(decoder.body_handles[i] for decoder in decoders)
            if isinstance(body.interior, TensorMedium):
                from .anisotropic_lifting import build_tensor_lifted_rhs
                parts = [build_tensor_lifted_rhs(h, interior_kernel=h.tensor_kernel) for h in handles]
                local, metadata = torch.stack([part[0] for part in parts]), [part[1] for part in parts]
            elif config.operator_backend == 'pfft':
                local, metadata = self.rhs_lifting[i].build_rhs_batch(handles)
            elif config.operator_backend != 'dense' and config.dyadic_integration is None:
                edges = torch.arange(body.basis.n_edges, device=self.system.device)
                local, metadata = build_selected_lifted_rhs_batch(handles, edges)
            else:
                local, metadata = build_lifted_rhs_batch(handles)
            n = body.basis.n_edges
            rhs[:, self.system.J_slices[i]], rhs[:, self.system.M_slices[i]] = local[:, :n], local[:, n:]
            for record, entry in zip(records, metadata):
                record.append({'region_id': i+1, **entry})
        for decoder, record in zip(decoders, records):
            self._record_rhs(decoder, record)
        return rhs, decoders

    @property
    def basis(self):
        if len(self.system.bodies) != 1:
            raise ValueError('Multiple bodies require an explicit body basis')
        return self.system.bodies[0].basis

    @property
    def interior(self):
        if len(self.system.bodies) != 1:
            raise ValueError('Multiple interiors require an explicit region_id')
        return self.system.bodies[0].interior


class SystemOperator(DenseOperator, SystemDecodeHandle):
    """Dense multi-body operator with separate compact body metrics and carriers."""
    def __init__(self, system, exterior, sources, config, *, kernel=FREE_SPACE):
        system.assert_current()
        if kernel.receipt() != FREE_SPACE.receipt():
            raise ValueError('Analytic multi-body lifting currently requires the free-space kernel')
        if any(body.interior.frequency != exterior.frequency for body in system.bodies):
            raise ValueError('Every region must use the common physical frequency')
        self.system, self.exterior, self.integration, self.kernel = system, exterior, config.integration, kernel
        self._public_config = config
        self.sources = tuple(sources)
        self.incident_carrier = IncidentCarrier.compile(self.sources, exterior)
        capacity = _preflight_dense_capacity(system, config)
        if config.dyadic_integration is not None:
            from .anisotropic_operator import assemble_generalized_system
            kernels = {b.region_id: DyadicGreenKernel(b.interior, config.dyadic_integration,
                config.dyadic_cache_max_bytes//2) for b in system.bodies if isinstance(b.interior, TensorMedium)}
            matrix, metadata = assemble_generalized_system(system, exterior, config.integration, interior_kernels=kernels)
            metadata.update(body_count=len(system.bodies), kernel_version=kernel.version,
                            quadrature=asdict(config.integration), physical_qualification=False)
        else:
            matrix, metadata = assemble_system(system, exterior, config.integration, kernel=kernel)
        grams = tuple(body.basis.mass_matrix() for body in system.bodies)
        if config.riesz_storage == 'compact_rwg_v1':
            riesz = BlockCompactRWGRiesz(tuple(CompactRWGRiesz(gram, abs(exterior.eta)**2) for gram in grams))
        else:
            gram = torch.block_diag(*grams).to(matrix.dtype)
            riesz = torch.block_diag(gram/abs(exterior.eta)**2, gram)
        metadata.update(residual_metric='impedance_balanced_L2_dual_proxy', trace_norm_certified=False,
                        coordinate_units='m', memory_preflight=capacity, numerical_identity=_numerical_identity(config))
        if config.riesz_storage == 'compact_rwg_v1': metadata['riesz_storage'] = 'block_compact_rwg_v1'
        super().__init__(matrix, riesz, metadata)
        SystemDecodeHandle.__init__(self, system, exterior, self.sources, config, self.metadata, kernel=kernel,
            tensor_kernels=kernels if config.dyadic_integration is not None else None)
        if config.dyadic_integration is not None:
            from .hierarchical_runtime import hierarchical_operator_identity
            identity, identity_payload = hierarchical_operator_identity(system, exterior, config)
            self.metadata.update(operator_identity=identity, operator_identity_payload=identity_payload)


    def refresh(self):
        self.system.assert_current()
        self.validate_tensor_state()
        super().refresh()

    def build_rhs(self):
        """Each exact incident carrier contributes only its local medium difference."""
        self.system.assert_current()
        rhs = torch.empty(self.shape[0], device=self.device, dtype=self.dtype)
        records = []
        self.validate_tensor_state()
        for i, handle in enumerate(self.body_handles):
            if isinstance(handle.interior, TensorMedium):
                from .anisotropic_lifting import build_tensor_lifted_rhs
                local, metadata = build_tensor_lifted_rhs(handle, interior_kernel=handle.tensor_kernel)
            elif self._public_config.operator_backend == 'pfft':
                batched, entries = self.rhs_lifting[i].build_rhs_batch((handle,))
                local, metadata = batched[0], entries[0]
            elif (self._public_config.operator_backend != 'dense'
                  and self._public_config.periodic is None
                  and self._public_config.dyadic_integration is None):
                from .lifting import build_selected_lifted_rhs
                edges = torch.arange(handle.basis.n_edges, device=self.device)
                local, metadata = build_selected_lifted_rhs(handle, edges)
            else:
                local, metadata = build_lifted_rhs(handle)
            n = handle.basis.n_edges
            rhs[self.system.J_slices[i]], rhs[self.system.M_slices[i]] = local[:n], local[n:]
            records.append({'region_id': i+1, **metadata})
        self._record_rhs(self, records)
        return rhs

@dataclass(frozen=True)
class SurfaceBodyData:
    """Read-only body identity and explicit slices of a physical global result."""
    parent: object = field(repr=False)
    index: int

    @property
    def body(self): return self.parent.handle.system.bodies[self.index]
    @property
    def region_id(self): return self.body.region_id
    @property
    def structure_index(self): return self.body.structure_index
    @property
    def name(self): return self.body.name
    @property
    def mesh(self): return self.body.mesh
    @property
    def global_edge_slice(self): return self.parent.handle.system.J_slices[self.index]
    @property
    def J_slice(self): return self.parent.handle.system.J_slices[self.index]
    @property
    def M_slice(self): return self.parent.handle.system.M_slices[self.index]
    @property
    def J_scattered(self): return self.parent.scattered_coefficients[self.J_slice]
    @property
    def M_scattered(self): return self.parent.scattered_coefficients[self.M_slice]

    def evaluate_total_currents(self, points, *, face_indices):
        self.parent._validate_carrier_binding()
        from .lifting_fields import evaluate_total_currents, system_local_coefficients
        return evaluate_total_currents(self.parent.handle.body_handles[self.index],
            system_local_coefficients(self.parent.handle, self.parent.scattered_coefficients, self.index), points, face_indices)


@dataclass
class SurfaceData:
    """One physical RHS with globally ordered scattered J/M and an exact carrier."""
    scattered_coefficients: torch.Tensor
    handle: SystemDecodeHandle | SystemOperator
    config: MoMConfig
    scene_signature: dict
    _incident_carrier: IncidentCarrier = field(init=False, repr=False)

    def __post_init__(self):
        self._incident_carrier = self.handle.incident_carrier
        self._prepared_system = self.handle.system
        self._prepared_exterior = self.handle.exterior
        self._prepared_sources = tuple(self.handle.sources)
        self._periodic_kernel = getattr(self.handle, 'periodic_kernel', None)
        self._prepared_numerical_identity = copy.deepcopy(_numerical_identity(self.config))

    def _validate_carrier_binding(self):
        if (self.handle.system is not self._prepared_system or self.handle.exterior is not self._prepared_exterior
                or tuple(self.handle.sources) != self._prepared_sources):
            raise ValueError('Result region or source binding changed after preparation')
        self.handle.system.assert_current()
        self.handle.validate_tensor_state()
        if (self.handle.integration != self.config.integration
                or any(local.integration != self.config.integration for local in self.handle.body_handles)):
            raise ValueError('Result integration configuration changed after preparation')
        if len(self.handle.body_handles) != len(self.handle.system.bodies) or any(
                local.basis is not body.basis or local.interior is not body.interior
                or local.exterior is not self._prepared_exterior or local.kernel is not FREE_SPACE
                for local, body in zip(self.handle.body_handles, self.handle.system.bodies)):
            raise ValueError('Result local region binding changed after preparation')
        if not _same_metadata(_numerical_identity(self.config), self._prepared_numerical_identity):
            raise ValueError('Result numerical configuration no longer matches its solved operator')
        if self.config.periodic is not None:
            if getattr(self.handle, 'periodic_kernel', None) is not self._periodic_kernel or self._periodic_kernel is None:
                raise ValueError('Periodic result kernel no longer matches its solved operator')
            self._periodic_kernel.lattice.assert_current()
        x = self.scattered_coefficients
        if (not isinstance(x, torch.Tensor) or x.shape != (2*self.handle.system.total_edges,)
                or x.device != self.handle.system.device or x.dtype != self.config.dtype
                or not bool(torch.isfinite(x).all())):
            raise ValueError('MoM scattered coefficients no longer match the global body layout')
        if (self.handle.incident_carrier is not self._incident_carrier
                or self.handle.metadata.get('incident_carrier_identity') != self._incident_carrier.identity
                or any(h.incident_carrier is not self._incident_carrier for h in self.handle.body_handles)):
            raise ValueError('MoM physical result carrier no longer matches its solved RHS')

    @property
    def coefficients(self):
        raise AttributeError('Lifted currents use scattered_coefficients; total currents require evaluate_total_currents.')
    @property
    def J(self):
        raise AttributeError('Use J_scattered or evaluate_total_currents; total J has no exact RWG coefficient view.')
    @property
    def M(self):
        raise AttributeError('Use M_scattered or evaluate_total_currents; total M has no exact RWG coefficient view.')
    @property
    def J_scattered(self): return self.scattered_coefficients[:self.handle.system.total_edges]
    @property
    def M_scattered(self): return self.scattered_coefficients[self.handle.system.total_edges:]
    @property
    def incident_carrier(self): return self._incident_carrier
    @property
    def bodies(self): return tuple(SurfaceBodyData(self, i) for i in range(len(self.handle.system.bodies)))
    @property
    def mesh(self): return self.handle.basis.mesh

    def evaluate_total_currents(self, points, *, face_indices, region_id='interior'):
        region = self.handle.system.region(region_id)
        if not region: raise ValueError('Total currents require an explicit body region')
        return self.bodies[region-1].evaluate_total_currents(points, face_indices=face_indices)

    def query_fields(self, points, region_id, field_kind, trace_side=None, *, query_config=None):
        self._validate_carrier_binding()
        if isinstance(self.config.periodic, MoMDyadicPeriodicConfig):
            return self.handle.query_fields(self.scattered_coefficients, points, region_id,
                field_kind, trace_side, query_config=query_config)
        if self.config.periodic is not None:
            from .periodic_fields import decode_periodic
            self.handle.last_periodic_field_log = None
            try:
                return decode_periodic(self.handle, self.scattered_coefficients, points, region_id, field_kind,
                    trace_side, kernel=self.handle.periodic_kernel,
                    query_config=self.config.query_config if query_config is None else query_config)
            finally:
                self.handle.last_field_query_log = self.handle.last_periodic_field_log
                self.handle.metadata['last_field_query'] = self.handle.last_field_query_log
        from .lifting_fields import decode_system
        return decode_system(self.handle, self.scattered_coefficients, points, region_id, field_kind,
            trace_side, query_config=self.config.query_config if query_config is None else query_config)

    def far_field(self, directions):
        if self.config.periodic is not None:
            raise ValueError('Periodic results require diffraction or Bloch observables, not isolated-scatterer far fields')
        from .lifting_fields import far_field_system
        self._validate_carrier_binding()
        return far_field_system(self.handle, self.scattered_coefficients, directions)

    def _periodic_observable_kernel(self):
        self._validate_carrier_binding()
        if isinstance(self.config.periodic, MoMDyadicPeriodicConfig):
            raise NotImplementedError('Periodic tensor modal observables require their own Maxwell-mode normalization')
        if self.config.periodic is None:
            raise ValueError('Periodic observables require a periodic MoM Result')
        return self.handle.periodic_kernel

    def rayleigh_orders(self, orders, *, side, plane_offset, quadrature_order=6):
        from .periodic_diffraction import rayleigh_total
        kernel = self._periodic_observable_kernel()
        return rayleigh_total(self.handle, self.scattered_coefficients, lattice=kernel.lattice,
            orders=orders, side=side, plane_offset=plane_offset, quadrature_order=quadrature_order)

    def cylindrical_orders(self, orders, directions, *, quadrature_order=6):
        from .periodic_cylindrical import cylindrical_orders
        kernel = self._periodic_observable_kernel()
        return cylindrical_orders(self.handle, self.scattered_coefficients, lattice=kernel.lattice,
            orders=orders, directions=directions, quadrature_order=quadrature_order)

    def bloch_spectrum(self, orders, *, quadrature_order=6):
        from .periodic_spectral import bloch_spectral_fields
        kernel = self._periodic_observable_kernel()
        return bloch_spectral_fields(self.handle, self.scattered_coefficients, kernel=kernel,
            orders=orders, quadrature_order=quadrature_order)

    def absorbed_power(self, points, volume_weights, *, region_id='interior', query_config=None):
        """Loss in the declared body [W] from explicit volume quadrature [m^3]."""
        from .observables import integrate_volume_dissipation
        region = self.handle.system.region(region_id)
        if not region: raise ValueError('Absorbed power requires an interior region')
        electric, magnetic = self.query_fields(points, region, 'total', query_config=query_config)
        return integrate_volume_dissipation(self.handle.system.bodies[region-1].interior, electric, magnetic, volume_weights)

    def absorbed_powers(self, volume_rules, *, query_config=None):
        """Return per-region and total absorption using one volume rule per body.

        volume_rules maps explicit region IDs/names to (points, weights). Every
        body must occur exactly once; the total is never inferred from balance.
        """
        if not isinstance(volume_rules, dict):
            raise TypeError('volume_rules must map each interior region to (points, weights)')
        rules = {}
        for key, rule in volume_rules.items():
            region = self.handle.system.region(key)
            if region == 0 or region in rules:
                raise ValueError('Absorption requires each interior region exactly once')
            if not isinstance(rule, (tuple, list)) or len(rule) != 2:
                raise ValueError('Each volume rule must be (points, weights)')
            rules[region] = rule
        if set(rules) != set(range(1, len(self.bodies)+1)):
            raise ValueError('Total absorption requires volume quadrature for every body')
        powers = {region: self.absorbed_power(*rule, region_id=region, query_config=query_config)
                  for region, rule in sorted(rules.items())}
        return {'per_region_W': powers, 'total_W': torch.stack(tuple(powers.values())).sum()}

    def plane_wave_cross_sections(self, directions, solid_angle_weights):
        from .observables import plane_wave_cross_sections
        self._validate_carrier_binding()
        return plane_wave_cross_sections(self.handle, self.scattered_coefficients, directions, solid_angle_weights,
                                         far_field_function=self.far_field)

    def snapshot(self):
        if self.config.operator_backend == 'pfft':
            raise NotImplementedError('pFFT results have no executable snapshot: the archive restores dense or '
                                      'hierarchical operator tensors, and a pFFT operator is rebuilt from the '
                                      'mesh; query or export fields before saving')
        self._validate_carrier_binding()
        bodies = []
        for view in self.bodies:
            body = view.body
            bodies.append({'region_id': body.region_id, 'structure_index': body.structure_index, 'name': body.name,
                'edge_offset': body.edge_offset, 'face_offset': body.face_offset, 'edge_count': body.basis.n_edges,
                'face_count': len(body.mesh.faces), 'J_slice': [view.J_slice.start, view.J_slice.stop],
                'M_slice': [view.M_slice.start, view.M_slice.stop], 'vertices': body.mesh.vertices.detach().cpu(),
                'faces': body.mesh.faces.detach().cpu(), 'interior': sampled_medium_payload(body.interior)})
        payload = {'schema_version': 3, 'payload_kind': 'physical_lifted_solution', 'executable': True,
                'basis': 'RWG', 'coefficient_order': ['all_J_scattered', 'all_M_scattered'],
                'coefficient_units': ['A/m', 'V/m'], 'coordinate_units': 'm', **_representation_identity(self.config),
                'scattered_coefficients': self.scattered_coefficients.detach().cpu(),
                'incident_carrier': self.incident_carrier.payload(), 'incident_carrier_identity': self.incident_carrier.identity,
                'body_table': bodies, 'exterior': sampled_medium_payload(self.handle.exterior),
                'integration': asdict(self.config.integration), 'query_config': asdict(self.config.query_config),
                'stopping': asdict(self.config.stopping), 'subdivisions': self.config.subdivisions,
                'riesz_storage': self.config.riesz_storage, 'dtype': str(self.config.dtype),
                'scene_signature': copy.deepcopy(self.scene_signature), 'operator_metadata': copy.deepcopy(self.handle.metadata)}
        if self.config.periodic is not None:
            payload.update(schema_version=7, periodic_config=self.config.periodic.payload())
        if self.config.operator_backend == 'hmatrix':
            payload.update(schema_version=8, hierarchical_config=self.config.hierarchical.payload())
        if self.config.operator_backend == 'h2':
            payload.update(schema_version=9, hierarchical_config=self.config.hierarchical.payload(), h2_config=self.config.h2.payload())
        if self.config.operator_backend == 'direct_h2':
            payload.update(schema_version=10, direct_h2_config=self.config.direct_h2.payload())
        if self.config.operator_backend == 'fmm':
            payload.update(schema_version=12 if self.config.periodic is not None else 11, fmm_config=self.config.fmm.payload())
        if self.config.periodic is not None and self.config.operator_backend in ('hmatrix', 'h2'):
            payload['schema_version'] = 17
        if self.config.dyadic_integration is not None:
            payload.update(schema_version=16 if self.config.periodic is not None else 13,
                           dyadic_integration=self.config.dyadic_integration.payload(),
                           dyadic_cache_max_bytes=self.config.dyadic_cache_max_bytes)
        if isinstance(self.config.periodic, MoMDyadicPeriodicConfig):
            payload.update(schema_version=18 if self.config.operator_backend=='dense' else 19,
                regular_kernel=self.handle.periodic_kernel.payload())
        if self.config.surface_mesh is not None:
            payload['surface_mesh'] = self.config.surface_mesh.payload()
        return payload

    @classmethod
    def restore(cls, payload, scene, frequency):
        if not isinstance(payload, dict): raise ValueError('MoM surface payload must be a dictionary')
        if payload.get('schema_version') == 20:
            from .linear_runtime import LinearSurfaceData
            return LinearSurfaceData.restore(payload, scene, frequency)
        periodic = MoMPeriodicConfig.restore(payload.get('periodic_config')) if payload.get('schema_version') in (7, 12, 16, 17) else None
        dyadic = payload.get('schema_version') in (18,19)
        if dyadic:
            periodic = MoMDyadicPeriodicConfig.restore(payload.get('periodic_config'))
        if (type(payload.get('schema_version')) is not int or payload['schema_version'] not in (3, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19)
                or payload.get('basis') != 'RWG' or payload.get('payload_kind') != 'physical_lifted_solution'
                or payload.get('executable') is not True):
            raise ValueError('Unsupported MoM surface payload schema; historical data requires explicit archive migration or a new solve')
        if dyadic and ((payload['schema_version']==18 and payload.get('operator_backend')!='dense')
                or (payload['schema_version']==19 and payload.get('operator_backend') not in ('hmatrix','h2'))):
            raise ValueError('Periodic dyadic schema and backend disagree')
        if payload['schema_version'] == 17 and payload.get('operator_backend') not in ('hmatrix', 'h2'):
            raise ValueError('Periodic hierarchical schema requires hmatrix or h2 backend')
        hierarchical = HMatrixConfig.restore(payload.get('hierarchical_config')) if payload['schema_version'] in (8, 9) or (payload['schema_version'] in (13, 16, 17, 19) and payload.get('operator_backend') in ('hmatrix', 'h2')) else None
        h2 = H2Config.restore(payload.get('h2_config')) if payload['schema_version'] == 9 or (payload['schema_version'] in (13, 16, 17, 19) and payload.get('operator_backend') == 'h2') else None
        direct_h2 = DirectH2Config.restore(payload.get('direct_h2_config')) if payload['schema_version'] == 10 else None
        fmm = FMMConfig.restore(payload.get('fmm_config')) if payload['schema_version'] in (11, 12) else None
        expected_keys = {'schema_version', 'payload_kind', 'executable', 'basis', 'coefficient_order',
            'coefficient_units', 'coordinate_units', 'scattered_coefficients', 'incident_carrier',
            'incident_carrier_identity', 'body_table', 'exterior', 'integration', 'query_config',
            'stopping', 'subdivisions', 'riesz_storage', 'dtype', 'scene_signature', 'operator_metadata',
            *_representation_identity()}
        if payload['schema_version'] in (13, 16, 18, 19):
            expected_keys.update(('dyadic_integration', 'dyadic_cache_max_bytes'))
        if dyadic:
            expected_keys.add('regular_kernel')
        if periodic is not None:
            expected_keys.add('periodic_config')
        if hierarchical is not None:
            expected_keys.add('hierarchical_config')
        if h2 is not None:
            expected_keys.add('h2_config')
        if direct_h2 is not None:
            expected_keys.add('direct_h2_config')
        if fmm is not None:
            expected_keys.add('fmm_config')
        surface_mesh = _restore_surface_mesh(payload.get('surface_mesh'))
        if surface_mesh is not None:
            expected_keys.add('surface_mesh')
        if set(payload) != expected_keys: raise ValueError('MoM surface payload has missing or unsupported fields')
        if (payload.get('coefficient_order') != ['all_J_scattered', 'all_M_scattered']
                or payload.get('coefficient_units') != ['A/m', 'V/m'] or payload.get('coordinate_units') != 'm'):
            raise ValueError('Invalid MoM coefficient or coordinate semantics')
        dtype = {'torch.complex64': torch.complex64, 'torch.complex128': torch.complex128}.get(payload['dtype'])
        config = MoMConfig(payload['subdivisions'], IntegrationConfig(**payload['integration']),
            StoppingRule(**payload['stopping']), dtype, FieldQueryConfig(**payload['query_config']),
            riesz_storage=payload['riesz_storage'], periodic=periodic,
            operator_backend='fmm' if fmm is not None else 'direct_h2' if direct_h2 is not None else 'h2' if h2 is not None else 'hmatrix' if hierarchical is not None else 'dense',
            hierarchical=hierarchical, h2=h2, direct_h2=direct_h2, fmm=fmm,
            dyadic_integration=DyadicIntegrationConfig(**payload['dyadic_integration']) if payload['schema_version'] in (13, 16, 18, 19) else None,
            dyadic_cache_max_bytes=payload.get('dyadic_cache_max_bytes', 256*1024**2), surface_mesh=surface_mesh)
        identity = _representation_identity(config)
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError('Unsupported MoM formulation, current kind, kernel, backend or decoder identity')
        metadata = payload.get('operator_metadata')
        storage = ('local_rwg_v1' if hierarchical is not None or direct_h2 is not None or fmm is not None else
                   'block_compact_rwg_v1' if config.riesz_storage == 'compact_rwg_v1' else 'dense')
        if not isinstance(metadata, dict) or metadata.get('riesz_storage', 'dense') != storage:
            raise ValueError('MoM snapshot Riesz storage metadata does not match its configuration')
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError('MoM snapshot operator representation metadata does not match')
        if metadata.get('unknown_order') != ['all_J_scattered_then_all_M_scattered']:
            raise ValueError('MoM snapshot operator coefficient order does not match')
        if (not _same_metadata(metadata.get('numerical_identity'), _numerical_identity(config))
                or not _same_metadata(metadata.get('rhs_integration'), asdict(config.integration))
                or not _same_metadata(metadata.get('quadrature'), asdict(config.integration))):
            raise ValueError('MoM snapshot numerical identity does not match its configuration')
        system, exterior = compile_surfaces(scene, frequency, config)
        if dyadic:
            from .periodic_dyadic import QuasiperiodicDyadicRegularKernel
            expected_kernel = periodic.compile(medium=exterior, integration=config.dyadic_integration,
                primary_cache_max_bytes=config.dyadic_cache_max_bytes//2)
            kernel = QuasiperiodicDyadicRegularKernel.from_payload(payload['regular_kernel'],
                medium=exterior, lattice=expected_kernel.lattice)
            if not _same_metadata(kernel.payload(), expected_kernel.payload()):
                raise ValueError('Saved dyadic regular kernel differs from declared configuration')
        else:
            kernel = FREE_SPACE if periodic is None else periodic.compile(device=system.device, dtype=config.dtype)
        if (type(metadata.get('body_count')) is not int or metadata['body_count'] != len(system.bodies)
                or metadata.get('kernel_version') != (kernel.payload()['version'] if dyadic else kernel.version)):
            raise ValueError('MoM snapshot body count or kernel version does not match')
        rhs_bodies = metadata.get('rhs_bodies')
        if (not isinstance(rhs_bodies, list) or len(rhs_bodies) != len(system.bodies)
                or any(not isinstance(record, dict) or type(record.get('region_id')) is not int
                       or record['region_id'] != i+1 for i, record in enumerate(rhs_bodies))):
            raise ValueError('MoM snapshot RHS body order does not match')
        signature = _signature(scene, exterior, tuple(b.interior for b in system.bodies))
        if not _same_metadata(payload['scene_signature'], signature):
            raise ValueError('MoM snapshot does not match the supplied Scene geometry, materials, or sources')
        if not _same_metadata(payload['exterior'], sampled_medium_payload(exterior)):
            raise ValueError('MoM snapshot sampled exterior does not match')
        if dyadic:
            from .tensor_carrier import TensorIncidentCarrier, compile_tensor_carrier
            carrier = TensorIncidentCarrier.from_payload(payload['incident_carrier'], device=scene.device, medium=exterior)
            if (carrier.identity != payload['incident_carrier_identity']
                    or carrier.identity != compile_tensor_carrier(scene.sources, exterior).identity):
                raise ValueError('Saved tensor carrier differs from the supplied Scene modes')
        else:
            carrier = IncidentCarrier.restore(payload['incident_carrier'], expected_identity=payload['incident_carrier_identity'],
                                               sources=scene.sources, exterior=exterior)
        if metadata.get('incident_carrier_identity') != carrier.identity:
            raise ValueError('MoM snapshot RHS carrier identity does not match')
        bodies = payload['body_table']
        if not isinstance(bodies, list) or len(bodies) != len(system.bodies):
            raise ValueError('MoM snapshot requires one explicit record per enabled body')
        for i, (record, body) in enumerate(zip(bodies, system.bodies)):
            expected = {'region_id': body.region_id, 'structure_index': body.structure_index, 'name': body.name,
                'edge_offset': body.edge_offset, 'face_offset': body.face_offset, 'edge_count': body.basis.n_edges,
                'face_count': len(body.mesh.faces), 'J_slice': [system.J_slices[i].start, system.J_slices[i].stop],
                'M_slice': [system.M_slices[i].start, system.M_slices[i].stop], 'interior': sampled_medium_payload(body.interior)}
            if not isinstance(record, dict) or set(record) != {*expected, 'vertices', 'faces'}:
                raise ValueError('MoM body record has missing or unsupported fields')
            if not _same_metadata({k: record[k] for k in expected}, expected):
                raise ValueError('MoM snapshot body order, slices or sampled material does not match')
            for key, reference in (('vertices', body.mesh.vertices), ('faces', body.mesh.faces)):
                value = record[key]
                if (not isinstance(value, torch.Tensor) or value.dtype != reference.dtype
                        or not torch.equal(value.to(scene.device), reference)):
                    raise ValueError('MoM snapshot surface mesh does not match the supplied Scene')
        coefficients = payload['scattered_coefficients']
        if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != (2*system.total_edges,)
                or coefficients.dtype != dtype or not bool(torch.isfinite(coefficients).all())):
            raise ValueError('Invalid MoM coefficient tensor')
        if dyadic:
            from .periodic_runtime import PeriodicDyadicDecodeHandle, periodic_dyadic_operator_identity
            handle = PeriodicDyadicDecodeHandle(system, exterior, scene.sources, config,
                copy.deepcopy(metadata), kernel=kernel, carrier=carrier)
            operator_id, operator_payload = periodic_dyadic_operator_identity(system, exterior, config, kernel)
            rhs_id = hashlib.sha256((operator_id+':'+carrier.identity).encode()).hexdigest()
            if payload['schema_version']==19:
                from .periodic_runtime import periodic_dyadic_residual_metric
                if (not _same_metadata(metadata.get('residual_metric_policy'),periodic_dyadic_residual_metric())
                        or metadata.get('residual_metric')!='vacuum_impedance_balanced_L2_dual_proxy'
                        or metadata.get('trace_norm_certified') is not False):
                    raise ValueError('Periodic tensor hierarchy residual metric identity changed')
            if (metadata.get('physical_qualification') is not False
                    or metadata.get('operator_identity') != operator_id or metadata.get('rhs_identity') != rhs_id
                    or not _same_metadata(metadata.get('operator_identity_payload'), operator_payload)
                    or not _same_metadata(metadata.get('kernel'), kernel.payload())):
                raise ValueError('Periodic dyadic snapshot operator, RHS or kernel identity does not match')
        else:
            handle = SystemDecodeHandle(system, exterior, scene.sources, config, copy.deepcopy(metadata))
        if periodic is not None and not dyadic:
            from .periodic_runtime import periodic_operator_identity
            from .periodic_operator import validate_periodic_sources
            validate_periodic_sources(scene.sources, kernel.lattice, exterior)
            kernel._guard(complex(exterior.k))
            operator_id, operator_payload = periodic_operator_identity(system, exterior, config, kernel)
            rhs_id = hashlib.sha256((operator_id+':'+carrier.identity).encode()).hexdigest()
            if (metadata.get('physical_qualification') is not False
                    or metadata.get('operator_identity') != operator_id or metadata.get('rhs_identity') != rhs_id
                    or not _same_metadata(metadata.get('operator_identity_payload'), operator_payload)
                    or not _same_metadata(metadata.get('kernel'), kernel.receipt())):
                raise ValueError('Periodic snapshot operator, RHS or kernel identity does not match')
            handle.periodic_kernel = kernel
        if periodic is None and (hierarchical is not None or direct_h2 is not None or fmm is not None or config.dyadic_integration is not None):
            from .hierarchical_runtime import hierarchical_operator_identity
            operator_id, operator_payload = hierarchical_operator_identity(system, exterior, config)
            rhs_id = hashlib.sha256((operator_id+':'+carrier.identity).encode()).hexdigest()
            if (metadata.get('physical_qualification') is not False
                    or metadata.get('operator_identity') != operator_id or metadata.get('rhs_identity') != rhs_id
                    or not _same_metadata(metadata.get('operator_identity_payload'), operator_payload)):
                raise ValueError('Hierarchical snapshot operator or RHS identity does not match')
        return cls(coefficients.to(scene.device), handle, config, signature)


class MoMRuntime:
    def __init__(self, simulation):
        self.simulation = simulation
        if simulation.has_trainable_parameters:
            raise NotImplementedError('MoM geometry/material derivatives are not supported; fixed-q residual_loss supports coefficient gradients')
        system, exterior = compile_surfaces(simulation.scene, simulation.frequency, simulation.config)
        if isinstance(simulation.config.periodic, MoMDyadicPeriodicConfig):
            from .periodic_runtime import (PeriodicDyadicSystemOperator,PeriodicDyadicHierarchicalOperator,
                PeriodicDyadicHierarchicalH2Operator)
            kernel = simulation.config.periodic.compile(medium=exterior,
                integration=simulation.config.dyadic_integration,
                primary_cache_max_bytes=simulation.config.dyadic_cache_max_bytes//2)
            cls={'dense':PeriodicDyadicSystemOperator,'hmatrix':PeriodicDyadicHierarchicalOperator,
                 'h2':PeriodicDyadicHierarchicalH2Operator}[simulation.config.operator_backend]
            self.handle = cls(system, exterior, simulation.scene.sources,
                simulation.config, kernel=kernel)
        elif simulation.config.operator_backend in ('hmatrix', 'h2') and simulation.config.periodic is not None:
            from .periodic_hierarchical_runtime import (PeriodicHierarchicalSystemOperator,
                                                        PeriodicHierarchicalH2SystemOperator)
            cls = (PeriodicHierarchicalH2SystemOperator if simulation.config.operator_backend == 'h2'
                   else PeriodicHierarchicalSystemOperator)
            kernel = simulation.config.periodic.compile(device=system.device, dtype=simulation.config.dtype)
            self.handle = cls(system, exterior, simulation.scene.sources, simulation.config, kernel=kernel)
        elif simulation.config.operator_backend == 'pfft':
            from .pfft_system import pfft_system_operator
            self.handle = pfft_system_operator(system, exterior, simulation.scene.sources, simulation.config)
        elif simulation.config.operator_backend == 'fmm' and simulation.config.periodic is not None:
            from .periodic_runtime import PeriodicMultipoleSystemOperator
            kernel = simulation.config.periodic.compile(device=system.device, dtype=simulation.config.dtype)
            self.handle = PeriodicMultipoleSystemOperator(system, exterior, simulation.scene.sources, simulation.config, kernel=kernel)
        elif simulation.config.operator_backend == 'fmm':
            from .hierarchical_runtime import MultipoleSystemOperator, DirectionalSystemOperator
            cls = DirectionalSystemOperator if isinstance(simulation.config.fmm, DirectionalFMMConfig) else MultipoleSystemOperator
            self.handle = cls(system, exterior, simulation.scene.sources, simulation.config)
        elif simulation.config.operator_backend == 'direct_h2':
            from .hierarchical_runtime import DirectH2SystemOperator
            self.handle = DirectH2SystemOperator(system, exterior, simulation.scene.sources, simulation.config)
        elif simulation.config.operator_backend == 'h2':
            from .hierarchical_runtime import HierarchicalH2SystemOperator
            self.handle = HierarchicalH2SystemOperator(system, exterior, simulation.scene.sources, simulation.config)
        elif simulation.config.operator_backend == 'hmatrix':
            from .hierarchical_runtime import HierarchicalSystemOperator
            self.handle = HierarchicalSystemOperator(system, exterior, simulation.scene.sources, simulation.config)
        elif simulation.config.periodic is None:
            self.handle = SystemOperator(system, exterior, simulation.scene.sources, simulation.config)
        else:
            from .periodic_runtime import PeriodicSystemOperator
            kernel = simulation.config.periodic.compile(device=system.device, dtype=simulation.config.dtype)
            self.handle = PeriodicSystemOperator(system, exterior, simulation.scene.sources, simulation.config, kernel=kernel)
        self.leaf_inverse = None
        if simulation.config.fmm is not None and simulation.config.fmm.leaf_factor_kernel != 'none':
            from .block_preconditioner import LeafBlockInverse
            fmm = simulation.config.fmm
            self.leaf_inverse = LeafBlockInverse(self.handle.provider, self.handle.tree,
                factor_kernel=fmm.leaf_factor_kernel, max_block_edges=fmm.leaf_factor_max_edges,
                retained_budget_bytes=fmm.leaf_factor_budget_bytes)
        self._leaf_inverse_binding = self.leaf_inverse
        self.right_preconditioner = self.right_preconditioner_receipt = None
        if simulation.config.right_preconditioner is not None:
            from .preconditioning import bind_right_preconditioner
            self.right_preconditioner, self.right_preconditioner_receipt = bind_right_preconditioner(
                simulation.config.right_preconditioner, self.handle, simulation.frequency)
        self.rhs = self.handle.build_rhs()
        self.scene_signature = _signature(simulation.scene, exterior, tuple(b.interior for b in system.bodies))
        self.prepared_config = simulation.config
        self.prepared_frequency = simulation.frequency

    def run(self):
        from ..result import Result
        sim = self.simulation
        exterior, interiors = validate_scene_system(sim.scene, sim.frequency,
            tensor_materials=sim.config.dyadic_integration is not None,
            tensor_exterior=isinstance(sim.config.periodic, MoMDyadicPeriodicConfig), dtype=sim.config.dtype,
            point_dipoles=_point_dipole_scope(sim.config))
        if (sim.config != self.prepared_config or sim.frequency != self.prepared_frequency
                or _signature(sim.scene, exterior, interiors) != self.scene_signature):
            raise ValueError('Scene or numerical configuration changed after MoM preparation')
        if self.leaf_inverse is not self._leaf_inverse_binding:
            raise ValueError('Prepared leaf preconditioner binding changed')
        if self.right_preconditioner is not None:
            action = self.right_preconditioner.apply
            if sim.config.operator_backend in ('hmatrix', 'h2', 'direct_h2', 'fmm', 'pfft'):
                coefficients, log = self.handle.solve(self.rhs, sim.config.stopping, right_preconditioner=action)
            else:
                coefficients, log = solve_reference(self.handle, self.rhs, sim.config.stopping,
                                                    right_preconditioner=action)
            log['right_preconditioner'] = dict(self.right_preconditioner_receipt)
        elif self.leaf_inverse is not None:
            coefficients, log = self.handle.solve(self.rhs, sim.config.stopping, right_inverse=self.leaf_inverse)
        elif sim.config.operator_backend in ('hmatrix', 'h2', 'direct_h2', 'fmm', 'pfft'):
            coefficients, log = self.handle.solve(self.rhs, sim.config.stopping)
        else:
            coefficients, log = solve_reference(self.handle, self.rhs, sim.config.stopping)
        data = SurfaceData(coefficients, self.handle, sim.config, self.scene_signature)
        monitors = {}
        if sim.scene.monitors:
            from .lifting_fields import classify_system_queries
            positions = tuple(dict.fromkeys(m.position for m in sim.scene.monitors))
            point_index = {position: index for index, position in enumerate(positions)}
            points = torch.tensor(positions, device=self.handle.system.device, dtype=self.handle.system.dtype)
            classified_points = points
            if sim.config.periodic is not None:
                import math
                lattice = self.handle.periodic_kernel.lattice
                translation = torch.floor(points@lattice.reciprocal.T/(2*math.pi)+.5)@lattice.vectors
                classified_points = points-translation
            regions, owners, distances = classify_system_queries(self.handle, coefficients, classified_points)
            if bool((owners != 0).any()):
                raise ValueError('MoM on-surface PointMonitors require explicit query_fields with trace_side')
            electric = torch.empty_like(points, dtype=coefficients.dtype)
            magnetic = torch.empty_like(electric)
            monitor_query_logs = {}
            for region in range(len(self.handle.system.bodies)+1):
                selected = regions == region
                if not bool(selected.any()):
                    continue
                try:
                    e, h = data.query_fields(points[selected], region, 'total')
                except RuntimeError as exc:
                    raise RuntimeError(f'MoM {region} PointMonitor batch failed: {exc}; query_log={getattr(self.handle, "last_field_query_log", None)}') from exc
                electric[selected], magnetic[selected] = e, h
                monitor_query_logs[region] = dict(self.handle.last_field_query_log)
            log['monitor_query_logs'] = monitor_query_logs
            log['unique_monitor_positions'] = len(positions)
            for monitor in sim.scene.monitors:
                index = point_index[monitor.position]
                region = int(regions[index])
                values = dict(zip(('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'),
                                  torch.cat((electric[index], magnetic[index]))))
                components = {name.capitalize(): values[name.capitalize()] for name in monitor.fields}
                monitors[monitor.name] = {'kind': 'point', 'fields': tuple(components),
                    'components': components, **components, 'position': monitor.position,
                    'frequency': sim.frequency, 'frequencies': (sim.frequency,),
                    'region_id': region,
                    'structure_index': None if region == 0 else self.handle.system.bodies[region-1].structure_index,
                    'structure_name': None if region == 0 else self.handle.system.bodies[region-1].name,
                    'field_kind': 'total', 'coordinate_units': 'm',
                    'phasor_convention': 'exp(-i omega t)', 'sampling': 'physical_position',
                    'distance_to_surface_m': float(distances[index])}
        return Result(method='mom', scene=sim.scene, frequency=sim.frequency, solver=self,
                      surface_data=data, solver_stats=log, monitors=monitors,
                      metadata={**sim.metadata, **self.handle.metadata,
                                'native_data': CURRENT_KIND, 'solver_converged': log['converged']})


# Result payload types: lifted PMCHWT surface data, open-sheet EFIE data and layered port currents.
SURFACE_DATA_TYPES = (SurfaceData, OpenSurfaceData, LayeredSurfaceData)


def select_runtime(simulation):
    """Linear surface runtime selected by the scene's body-kind formulation."""
    if scene_formulation([s for s in simulation.scene.structures if s.enabled]) == EFIE_OPEN_PEC:
        return OpenPECRuntime(simulation)
    return MoMRuntime(simulation)
