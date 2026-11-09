"""Open-surface PEC electric-field integral equation (dense, vacuum).

Formulation ``efie_open_pec`` (exp(-i omega t)): for the total induced surface
current J on zero-thickness perfectly conducting sheets,

    <f_m, E_s(J)> = -<f_m, E_inc>,   E_s = i*omega*mu*A(J) - grad phi(J),

tested with the same open RWG functions that expand J. The weak operator is
``i*omega*mu*(vector - div^T scalar_p0 div / k^2)`` from ``single_layer``.
Rows carry tested electric field [V*m]; unknowns are physical total currents
[A/m], so no incident lifting is involved and J is the radiating current.

Only the dense free-space backend with PlaneWave and exterior electric
PointDipole carriers is implemented. Every other backend, periodic, tensor,
harmonic or coupled configuration, and every scene mixing sheets with other
body kinds, is rejected before assembly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import hashlib
import math
import time

import torch

from ..constants import MU_0
from ..media import Material
from ..monitors import PointMonitor
from ..sources import PointDipole
from .carrier import IncidentCarrier
from .kernels import FREE_SPACE
from .linalg import DenseOperator, solve_reference
from .open_surface import OpenRWGBasis, compile_open_surface
from .operator import local_geometry
from .physics import compile_medium, source_phasor, validate_scene_structure
from .quadrature import triangle_rule
from .single_layer import SINGLE_LAYER_VERSION, assemble_single_layer, global_divergence
from .topology import EFIE_OPEN_PEC

FORMULATION_ID = 'efie_open_pec_exp_minus_iwt_v1'
CURRENT_KIND = 'total_electric_surface_current_A_per_m'
RHS_VERSION = 'efie_tested_incident_field_v1'
DECODER_VERSION = 'efie_single_layer_static_subtracted_fields_v1'


@dataclass(frozen=True)
class OpenSurfaceConfig:
    """Discretization controls specific to open conducting sheets.

    ``max_edge_length_m`` meshes zero-thickness Box/Cylinder primitives with
    at most this edge spacing along each side or radius; explicit surface
    Mesh geometry is used as supplied.
    """
    max_edge_length_m: float | None = None

    def __post_init__(self):
        value = self.max_edge_length_m
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError('max_edge_length_m must be a positive finite length in metres')

    def payload(self):
        return asdict(self)


def efie_identity(config):
    """Formulation, basis, kernel and backend identities, each separately versioned."""
    return {'formulation': EFIE_OPEN_PEC, 'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
            'basis_id': 'open_rwg_interior_edges_v1', 'test_space_id': 'RWG',
            'single_layer_version': SINGLE_LAYER_VERSION, 'kernel_id': FREE_SPACE.kernel_id,
            'kernel_version': FREE_SPACE.version, 'operator_backend': config.operator_backend,
            **({'pfft': config.pfft.payload()} if config.operator_backend == 'pfft' else {}),
            'rhs_version': RHS_VERSION, 'decoder_version': DECODER_VERSION}


def validate_efie_config(config):
    """Fail closed for every configuration outside the dense free-space EFIE."""
    unsupported = {'periodic': config.periodic, 'harmonic_balance': config.harmonic_balance,
                   'dyadic_integration': config.dyadic_integration, 'linear_coupled': config.linear_coupled,
                   'hierarchical': config.hierarchical, 'h2': config.h2, 'direct_h2': config.direct_h2,
                   'fmm': config.fmm}
    # MoMConfig attaches hierarchical, direct_h2 or fmm to every non-dense
    # backend, so this loop also rejects all of them.
    for name, value in unsupported.items():
        if value is not None:
            raise NotImplementedError(f'The open-surface PEC EFIE does not support MoMConfig.{name}; '
                                      'only the dense free-space backend is implemented')
    # The pFFT single layer uses the local RWG Gram metric (riesz_storage=local_rwg_v1).
    if config.operator_backend == 'dense' and config.riesz_storage != 'dense':
        raise ValueError('The open-surface PEC EFIE uses one dense RWG Gram residual metric (riesz_storage=dense)')
    if config.open_surface is not None and not isinstance(config.open_surface, OpenSurfaceConfig):
        raise TypeError('MoMConfig.open_surface must be an OpenSurfaceConfig')


@dataclass(frozen=True)
class OpenSurfaceBody:
    structure_index: int
    name: str | None
    basis: OpenRWGBasis
    edge_offset: int
    face_offset: int
    interior: None = None

    @property
    def mesh(self):
        return self.basis.mesh


class OpenSurfaceSystem:
    """Separately meshed sheets in Scene order, all radiating into one exterior."""

    current_components = ('J',)

    def __init__(self, bodies):
        if not isinstance(bodies, tuple) or not bodies:
            raise ValueError('OpenSurfaceSystem requires a nonempty tuple of bodies')
        edges = faces = 0
        aliases = {}
        for body in bodies:
            if body.edge_offset != edges or body.face_offset != faces:
                raise ValueError('Sheet offsets must be contiguous in Scene order')
            if body.mesh.device != bodies[0].mesh.device or body.mesh.dtype != bodies[0].mesh.dtype:
                raise ValueError('Every sheet must share CUDA device and coordinate precision')
            if body.name is not None:
                if body.name in aliases or body.name == 'exterior':
                    raise ValueError('Sheet names must be unique and not reserved')
                aliases[body.name] = len(aliases)
            edges += body.basis.n_edges
            faces += len(body.mesh.faces)
        self.bodies, self.total_edges, self.total_faces = bodies, edges, faces
        self.J_slices = tuple(slice(b.edge_offset, b.edge_offset+b.basis.n_edges) for b in bodies)
        self.M_slices = ()
        self.pair_gaps = self._validate_separation()

    @property
    def device(self):
        return self.bodies[0].mesh.device

    @property
    def dtype(self):
        return self.bodies[0].mesh.dtype

    def assert_current(self):
        for body in self.bodies:
            body.basis.assert_current()

    def entries(self):
        return [(body.basis, body.edge_offset, body.face_offset) for body in self.bodies]

    def _validate_separation(self):
        from .mesh import triangle_pair_distances
        gaps = []
        for i, left in enumerate(self.bodies):
            for j, right in enumerate(self.bodies[i+1:], i+1):
                a, b = left.mesh.triangles, right.mesh.triangles
                minimum = torch.tensor(torch.inf, dtype=self.dtype, device=self.device)
                for start in range(0, len(a)*len(b), 4096):
                    ids = torch.arange(start, min(start+4096, len(a)*len(b)), device=self.device)
                    minimum = torch.minimum(minimum, triangle_pair_distances(a[ids//len(b)], b[ids % len(b)]).min())
                scale = torch.maximum(left.mesh.vertices.abs().amax(), right.mesh.vertices.abs().amax())
                if not bool(minimum > 256*torch.finfo(self.dtype).eps*scale):
                    raise ValueError('Sheets touch or intersect; junctions between sheets are not implemented')
                gaps.append((i, j, float(minimum)))
        return tuple(gaps)


def _validate_dipole_clearance(scene, system, config):
    """Exterior electric dipoles keep the PMCHWT panel-clearance rule on every sheet."""
    from .dipole import DIPOLE_CLEARANCE_PANEL_RADII, DIPOLE_MINIMUM_SOURCE_ORDER
    from .fields import _triangle_distance
    dipoles = [source for source in scene.sources if isinstance(source, PointDipole)]
    if dipoles and min(config.integration.order, config.integration.singular_order) < DIPOLE_MINIMUM_SOURCE_ORDER:
        raise ValueError(f'MoM PointDipole sources require IntegrationConfig order and singular_order >= '
                         f'{DIPOLE_MINIMUM_SOURCE_ORDER}')
    for dipole in dipoles:
        for body in system.bodies:
            triangles = body.mesh.triangles
            point = triangles.new_tensor(tuple(float(x) for x in dipole.position))[None]
            distance = _triangle_distance(point, triangles)[0]
            radius = torch.linalg.vector_norm(triangles-triangles.mean(1)[:, None], dim=-1).amax(-1)
            ratio = float((distance/radius).min())
            if ratio < DIPOLE_CLEARANCE_PANEL_RADII:
                raise ValueError(f'PointDipole at {tuple(dipole.position)} is within {ratio:.3g} panel radii of '
                                 f'sheet {body.name!r}; at least {DIPOLE_CLEARANCE_PANEL_RADII:g} are required. '
                                 'Refine the sheet locally or move the dipole.')


def compile_open_pec_scene(scene, frequency, config):
    """Validate scope and mesh every enabled sheet in Scene order."""
    validate_efie_config(config)
    validate_scene_structure(scene, frequency, point_dipoles=True, formulation=EFIE_OPEN_PEC)
    from .runtime import _validate_surface_queries
    _validate_surface_queries(scene)
    exterior = compile_medium(Material(), frequency, background=True)
    dtype = torch.float64 if config.dtype == torch.complex128 else torch.float32
    resolution = None if config.open_surface is None else config.open_surface.max_edge_length_m
    bodies, edges, faces = [], 0, 0
    for index, structure in enumerate(scene.structures):
        if not structure.enabled:
            continue
        mesh = compile_open_surface(structure.geometry, device=scene.device, dtype=dtype,
                                    max_edge_length_m=resolution)
        basis = OpenRWGBasis(mesh)
        bodies.append(OpenSurfaceBody(index, structure.name, basis, edges, faces))
        edges += basis.n_edges
        faces += len(mesh.faces)
    system = OpenSurfaceSystem(tuple(bodies))
    _validate_dipole_clearance(scene, system, config)
    return system, exterior


def _preflight(system, config):
    edges, faces = system.total_edges, system.total_faces
    complex_bytes = 16 if config.dtype == torch.complex128 else 8
    # Resident single-layer vector block (becomes Z in place), Gram metric,
    # P0 scalar block, its product with div and an optional LU factor.
    factor = edges**2*complex_bytes if (config.stopping.method == 'direct' or config.stopping.fallback == 'direct') else 0
    minimum = 2*edges**2*complex_bytes+faces**2*complex_bytes+faces*edges*complex_bytes+factor
    free, total = torch.cuda.mem_get_info(system.device)
    reusable = max(0, torch.cuda.memory_reserved(system.device)-torch.cuda.memory_allocated(system.device))
    available = min(total, free+reusable)
    if minimum > available:
        raise torch.cuda.OutOfMemoryError(f'Open-surface EFIE dense preflight: {minimum} bytes exceed available '
                                          f'{available} bytes on {system.device}')
    return {'minimum_required_bytes': minimum, 'available_bytes': available, 'workspace_included': False}


class OpenPECOperator(DenseOperator):
    """Dense EFIE matrix Z [E, E] with the RWG L2 Gram as residual Riesz map."""

    current_components = ('J',)

    def __init__(self, system, exterior, sources, config, *, kernel=FREE_SPACE):
        if kernel.receipt() != FREE_SPACE.receipt():
            raise ValueError('The open-surface EFIE currently requires the free-space kernel')
        system.assert_current()
        capacity = _preflight(system, config)
        entries = system.entries()
        matrix, scalar, metadata = assemble_single_layer(entries, exterior.k, config.integration, kernel=kernel)
        divergence = global_divergence(entries, matrix.dtype)
        omega_mu = 2*math.pi*exterior.frequency*MU_0*exterior.mu_r
        matrix.addmm_(divergence.mT, scalar@divergence, alpha=-1/exterior.k**2)
        matrix.mul_(1j*omega_mu)
        del scalar
        gram = torch.block_diag(*(body.basis.mass_matrix() for body in system.bodies)).to(matrix.dtype)
        metadata.update(efie_identity(config), unknown_order=['all_sheet_J'], body_count=len(system.bodies),
                        residual_metric='rwg_L2_dual_proxy', trace_norm_certified=False,
                        coordinate_units='m', row_units='V*m', memory_preflight=capacity,
                        integration=asdict(config.integration), physical_qualification=False)
        super().__init__(matrix, gram, metadata)
        self.system, self.exterior, self.kernel = system, exterior, kernel
        self.integration, self.sources = config.integration, tuple(sources)
        self.incident_carrier = IncidentCarrier.compile(self.sources, exterior)

    def build_rhs(self):
        """-<f_m, E_inc> on every sheet with order max(6, integration.order)."""
        self.system.assert_current()
        order = max(6, self.integration.order)
        rhs = torch.zeros(self.shape[0], device=self.device, dtype=self.dtype)
        for body in self.system.bodies:
            basis = body.basis
            points, weights = triangle_rule(basis.mesh.triangles, order)
            electric, _ = self.incident_carrier.evaluate(points.reshape(-1, 3), self.exterior)
            values = basis.local_values(points).to(self.dtype)
            local = -torch.einsum('tqlc,tqc,tq->tl', values, electric.reshape(values.shape[0], -1, 3),
                                  weights.to(self.dtype))
            rhs.index_add_(0, (basis.face_edges+body.edge_offset).flatten(), local.flatten())
        self.metadata.update(incident_carrier_identity=self.incident_carrier.identity, rhs_order=order)
        return rhs


@dataclass
class _FieldHandle:
    """Per-sheet view in the layout expected by the shared panel potential integrator."""
    basis: OpenRWGBasis
    integration: object
    kernel: object = FREE_SPACE


def scattered_fields(handle, coefficients, points, *, order, singular_order):
    """Scattered E [V/m] and H [A/m] radiated by sheet currents at off-surface points.

    Near panels subtract the exact static panel potential and gradient and
    integrate the smooth remainder with a Duffy fan; far panels use a product
    rule. Boundary edges need no special case because every panel integral is
    complete and the current of an open RWG space is affine on each face.
    """
    from .fields import _potentials
    electric = torch.zeros(points.shape, device=points.device, dtype=coefficients.dtype)
    magnetic = torch.zeros_like(electric)
    for body, sl in zip(handle.system.bodies, handle.system.J_slices):
        integration = replace(handle.integration, order=order, singular_order=singular_order,
                              near_max_order=max(handle.integration.near_max_order, singular_order+2))
        local = _FieldHandle(body.basis, integration, handle.kernel)
        x = torch.cat((coefficients[sl], torch.zeros_like(coefficients[sl])))
        e, h = _potentials(local, x, points, handle.exterior, order=order, singular_order=singular_order)
        electric, magnetic = electric+e, magnetic+h
    return electric, magnetic


def surface_distances(system, points):
    """Minimum point-to-sheet distance [P] in metres."""
    from .fields import _triangle_distance
    return torch.stack([_triangle_distance(points, body.mesh.triangles).amin(-1) for body in system.bodies]).amin(0)


def query_open_fields(handle, coefficients, points, field_kind, query_config):
    """Adaptive successive-order field query; on-surface points are rejected."""
    if field_kind not in ('total', 'incident', 'scattered'):
        raise ValueError("field_kind must be total, incident or scattered")
    if (not isinstance(points, torch.Tensor) or points.device != handle.device or points.ndim != 2
            or points.shape[-1] != 3 or points.dtype != handle.system.dtype or not bool(torch.isfinite(points).all())):
        raise ValueError('Sheet field queries require finite [P, 3] CUDA points in the mesh precision')
    log = {'query_count': len(points), 'checks': [], 'converged': False,
           'estimator': 'successive_order_difference', 'global_error_certified': False}
    handle.last_field_query_log = log
    extent = max(float(torch.linalg.vector_norm(b.mesh.vertices.amax(0)-b.mesh.vertices.amin(0)))
                 for b in handle.system.bodies)
    distance = surface_distances(handle.system, points) if len(points) else points.new_zeros(0)
    if bool((distance <= 128*torch.finfo(points.dtype).eps*extent).any()):
        raise ValueError('Sheet field queries on the surface are undefined: the normal field jumps across '
                         'a charged sheet; move the point off the sheet')
    log['minimum_distance_m'] = float(distance.min()) if len(points) else None
    incident = handle.incident_carrier.evaluate(points, handle.exterior) if field_kind != 'scattered' else None
    if field_kind == 'incident':
        log.update(converged=True, final_order=None)
        return incident
    order = max(handle.integration.order, handle.integration.singular_order)
    result = scattered_fields(handle, coefficients, points, order=order, singular_order=order)
    if query_config.adaptive and len(points):
        eta = abs(handle.exterior.eta)
        while True:
            if order >= query_config.max_order:
                log['failure_reason'] = 'quadrature_order_budget_exhausted'
                raise RuntimeError('Sheet field quadrature did not converge within max_order')
            order = min(2*order, query_config.max_order)
            refined = scattered_fields(handle, coefficients, points, order=order, singular_order=order)
            error = ((refined[0]-result[0]).abs().square().sum(-1)
                     + eta**2*(refined[1]-result[1]).abs().square().sum(-1)).sqrt()
            norm = (refined[0].abs().square().sum(-1)+eta**2*refined[1].abs().square().sum(-1)).sqrt()
            log['checks'].append({'order': order, 'max_absolute_error_V_per_m': float(error.max()),
                                  'max_relative_error': float((error/norm.clamp_min(torch.finfo(norm.dtype).tiny)).max())})
            result = refined
            if bool((error <= query_config.absolute_tolerance+query_config.relative_tolerance*norm).all()):
                break
    log.update(converged=True, final_order=order, failure_reason=None)
    if field_kind == 'scattered':
        return result
    return result[0]+incident[0], result[1]+incident[1]


def far_field_open(handle, coefficients, directions):
    """Outgoing amplitudes F_E [V], F_H [A] with E ~ exp(ikr) F_E / r, phase origin at 0."""
    if (not isinstance(directions, torch.Tensor) or directions.device != handle.device or directions.ndim != 2
            or directions.shape[-1] != 3 or not bool(torch.isfinite(directions).all())):
        raise ValueError('Far-field directions must be finite [D, 3] CUDA unit vectors')
    norm = torch.linalg.vector_norm(directions, dim=-1)
    if not bool(torch.isclose(norm, torch.ones_like(norm), rtol=1e-6, atol=1e-7).all()):
        raise ValueError('Far-field directions must be unit vectors')
    medium, dtype = handle.exterior, coefficients.dtype
    integral = torch.zeros((len(directions), 3), device=handle.device, dtype=dtype)
    order = max(handle.integration.order, 6)
    for body, sl in zip(handle.system.bodies, handle.system.J_slices):
        points, weights = triangle_rule(body.mesh.triangles, order)
        current = body.basis.current(coefficients[sl], points)
        for start in range(0, len(directions), 128):
            direction = directions[start:start+128].to(points.dtype)
            phase = torch.exp(-1j*medium.k*torch.einsum('dc,tqc->dtq', direction, points))
            integral[start:start+128] += torch.einsum('dtq,tqc,tq->dc', phase.to(dtype), current, weights.to(dtype))
    n = directions.to(dtype)
    prefactor = 1j*medium.k/(4*math.pi)
    transverse = integral-n*(n*integral).sum(-1, keepdim=True)
    return prefactor*medium.eta*transverse, prefactor*torch.linalg.cross(n, integral)


def _open_signature(scene, exterior, config):
    shapes = []
    for index, structure in enumerate(scene.structures):
        if not structure.enabled:
            continue
        geometry = structure.geometry
        shape = {'structure_index': index, 'name': structure.name, 'geometry': geometry.kind,
                 'position': geometry.position.detach().cpu().tolist(),
                 'rotation': geometry.rotation.detach().cpu().tolist(), 'material': repr(structure.material)}
        for name in ('size', 'radius', 'height', 'scale'):
            value = getattr(geometry, name, None)
            if isinstance(value, torch.Tensor):
                shape[name] = value.detach().cpu().tolist()
        if hasattr(geometry, 'faces'):
            shape.update(vertices_sha256=hashlib.sha256(geometry.vertices.detach().cpu().numpy().tobytes()).hexdigest(),
                         faces_sha256=hashlib.sha256(geometry.faces.detach().cpu().numpy().tobytes()).hexdigest())
        shapes.append(shape)
    sources = [{'kind': type(s).__name__, 'declaration': repr(s), 'amplitude': source_phasor(s, exterior.frequency)}
               for s in scene.sources]
    return {'bodies': shapes, 'sources': sources, 'frequency': exterior.frequency,
            'monitors': tuple(repr(m) for m in scene.monitors), 'boundary': repr(scene.boundary),
            'open_surface': None if config.open_surface is None else config.open_surface.payload()}


@dataclass
class OpenSheetView:
    """Read-only per-sheet identity and slice of the global J coefficients."""
    parent: object = field(repr=False)
    index: int

    @property
    def body(self): return self.parent.handle.system.bodies[self.index]
    @property
    def structure_index(self): return self.body.structure_index
    @property
    def name(self): return self.body.name
    @property
    def mesh(self): return self.body.mesh
    @property
    def basis(self): return self.body.basis
    @property
    def J_slice(self): return self.parent.handle.system.J_slices[self.index]
    @property
    def J(self): return self.parent.coefficients[self.J_slice]


@dataclass
class OpenSurfaceData:
    """Total sheet currents of one solved EFIE RHS with its exact incident carrier."""
    coefficients: torch.Tensor
    handle: OpenPECOperator
    config: object
    scene_signature: dict

    def __post_init__(self):
        self._carrier = self.handle.incident_carrier
        self._system = self.handle.system

    def _validate(self):
        if self.handle.system is not self._system or self.handle.incident_carrier is not self._carrier:
            raise ValueError('Sheet result binding changed after the solve')
        self.handle.system.assert_current()
        x = self.coefficients
        if (not isinstance(x, torch.Tensor) or x.shape != (self.handle.system.total_edges,)
                or x.device != self.handle.device or x.dtype != self.config.dtype or not bool(torch.isfinite(x).all())):
            raise ValueError('Sheet coefficients no longer match the solved RWG layout')

    @property
    def formulation(self): return EFIE_OPEN_PEC
    @property
    def J(self): return self.coefficients
    @property
    def incident_carrier(self): return self._carrier
    @property
    def bodies(self): return tuple(OpenSheetView(self, i) for i in range(len(self.handle.system.bodies)))

    @property
    def mesh(self):
        if len(self.handle.system.bodies) != 1:
            raise ValueError('Multiple sheets require an explicit body view')
        return self.handle.system.bodies[0].mesh

    @property
    def basis(self):
        if len(self.handle.system.bodies) != 1:
            raise ValueError('Multiple sheets require an explicit body view')
        return self.handle.system.bodies[0].basis

    def evaluate_currents(self, points, *, face_indices, region_id=None):
        """Total J [A/m] at points [..., 3] lying on the given faces of a single-sheet result."""
        self._validate()
        index = 0 if region_id is None else next(i for i, b in enumerate(self.handle.system.bodies)
                                                    if region_id in (b.name, b.structure_index))
        body, sl = self.handle.system.bodies[index], self.handle.system.J_slices[index]
        _, triangles, slopes = local_geometry(body.basis)
        local = (self.coefficients[sl][body.basis.face_edges[face_indices]]*slopes[face_indices].to(self.coefficients.dtype))
        displacement = (points[..., None, :]-triangles[face_indices]).to(self.coefficients.dtype)
        return torch.einsum('...l,...lc->...c', local, displacement)

    def query_fields(self, points, region_id='exterior', field_kind='total', trace_side=None, *, query_config=None):
        self._validate()
        if region_id not in (0, 'exterior') or isinstance(region_id, bool):
            raise ValueError('Open sheets enclose no region; query the exterior (region_id=0)')
        if trace_side is not None:
            raise ValueError('On-sheet traces are not supported; query off-surface points')
        return query_open_fields(self.handle, self.coefficients, points, field_kind,
                                 self.config.query_config if query_config is None else query_config)

    def far_field(self, directions):
        self._validate()
        return far_field_open(self.handle, self.coefficients, directions)

    def radar_cross_section(self, directions):
        """Bistatic RCS [m^2] for exactly one incident plane wave."""
        carrier = self._carrier
        if len(carrier.waves) != 1 or carrier.dipoles:
            raise ValueError('RCS requires exactly one incident plane wave and no point dipoles')
        amplitude = abs(complex(*carrier.waves[0].amplitude))
        if amplitude == 0:
            raise ValueError('RCS is undefined for zero incident amplitude')
        electric, _ = self.far_field(directions)
        return 4*math.pi*electric.abs().square().sum(-1)/amplitude**2

    def plane_wave_cross_sections(self, directions, solid_angle_weights):
        from .observables import plane_wave_cross_sections
        self._validate()
        return plane_wave_cross_sections(self.handle, self.coefficients, directions, solid_angle_weights,
                                         far_field_function=self.far_field)

    def snapshot(self):
        raise NotImplementedError('Open-surface EFIE results do not support snapshot persistence: no payload '
                                  'schema binds the sheet meshes, open RWG layout and total-current carrier, '
                                  'so Result.save/load would not be verifiable; re-run the Simulation instead')


class OpenPECRuntime:
    """Prepared dense EFIE solve for scenes whose bodies are all open PEC sheets."""

    def __init__(self, simulation):
        if simulation.has_trainable_parameters:
            raise NotImplementedError('Open-surface EFIE geometry/material derivatives are not supported')
        self.simulation = simulation
        config = simulation.config
        system, exterior = compile_open_pec_scene(simulation.scene, simulation.frequency, config)
        if config.operator_backend == 'pfft':
            from .pfft_efie import pfft_open_pec_operator
            self.handle = pfft_open_pec_operator(system, exterior, simulation.scene.sources, config)
        else:
            self.handle = OpenPECOperator(system, exterior, simulation.scene.sources, config)
        self.right_preconditioner = self.right_preconditioner_receipt = None
        if config.right_preconditioner is not None:
            from .preconditioning import bind_right_preconditioner
            self.right_preconditioner, self.right_preconditioner_receipt = bind_right_preconditioner(
                config.right_preconditioner, self.handle, simulation.frequency)
        self.rhs = self.handle.build_rhs()
        self.scene_signature = _open_signature(simulation.scene, exterior, config)
        self.prepared_config, self.prepared_frequency = config, simulation.frequency

    def run(self):
        from ..result import Result
        sim = self.simulation
        exterior = compile_medium(Material(), sim.frequency, background=True)
        if (sim.config != self.prepared_config or sim.frequency != self.prepared_frequency
                or _open_signature(sim.scene, exterior, sim.config) != self.scene_signature):
            raise ValueError('Scene or numerical configuration changed after MoM preparation')
        torch.cuda.synchronize(self.handle.device)
        started = time.perf_counter()
        if self.right_preconditioner is not None:
            coefficients, log = solve_reference(self.handle, self.rhs, sim.config.stopping,
                                                right_preconditioner=self.right_preconditioner.apply)
            log['right_preconditioner'] = dict(self.right_preconditioner_receipt)
        else:
            coefficients, log = solve_reference(self.handle, self.rhs, sim.config.stopping)
        torch.cuda.synchronize(self.handle.device)
        log['solve_seconds'] = time.perf_counter()-started
        data = OpenSurfaceData(coefficients, self.handle, sim.config, self.scene_signature)
        monitors = {}
        if sim.scene.monitors:
            positions = tuple(dict.fromkeys(m.position for m in sim.scene.monitors))
            points = torch.tensor(positions, device=self.handle.device, dtype=self.handle.system.dtype)
            electric, magnetic = data.query_fields(points)
            index = {position: i for i, position in enumerate(positions)}
            distances = surface_distances(self.handle.system, points)
            for monitor in sim.scene.monitors:
                i = index[monitor.position]
                values = dict(zip(('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'), torch.cat((electric[i], magnetic[i]))))
                components = {name.capitalize(): values[name.capitalize()] for name in monitor.fields}
                monitors[monitor.name] = {'kind': 'point', 'fields': tuple(components), 'components': components,
                    **components, 'position': monitor.position, 'frequency': sim.frequency,
                    'frequencies': (sim.frequency,), 'region_id': 0, 'field_kind': 'total',
                    'coordinate_units': 'm', 'phasor_convention': 'exp(-i omega t)',
                    'sampling': 'physical_position', 'distance_to_surface_m': float(distances[i])}
        return Result(method='mom', scene=sim.scene, frequency=sim.frequency, solver=self, surface_data=data,
                      solver_stats=log, monitors=monitors,
                      metadata={**sim.metadata, **copy.deepcopy(self.handle.metadata),
                                'native_data': CURRENT_KIND, 'solver_converged': log['converged']})
