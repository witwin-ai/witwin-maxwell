"""Explicit offline archival migration; no legacy numerical runtime path."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import io
from pathlib import Path

import torch

from .fields import FieldQueryConfig
from .linalg import StoppingRule
from .operator import IntegrationConfig
from .runtime import MoMConfig, _same_metadata, compile_surface
from witwin.core import Mesh, Sphere
from .physics import source_phasor


def _historical_signature(scene, exterior, interior):
    geometry = next(s.geometry for s in scene.structures if s.enabled)
    source_data = [{'direction': tuple(s.direction), 'polarization': tuple(s.polarization),
                    'amplitude': source_phasor(s, exterior.frequency)} for s in scene.sources]
    shape = {'geometry': geometry.kind, 'position': geometry.position.detach().cpu().tolist(),
             'rotation': geometry.rotation.detach().cpu().tolist()}
    if isinstance(geometry, Mesh):
        shape.update({'vertices_sha256': hashlib.sha256(geometry.vertices.detach().cpu().numpy().tobytes()).hexdigest(),
                      'faces_sha256': hashlib.sha256(geometry.faces.detach().cpu().numpy().tobytes()).hexdigest(),
                      'scale': geometry.scale.detach().cpu().tolist(), 'recenter': geometry.recenter,
                      'fill_mode': geometry.fill_mode, 'surface_thickness': geometry.surface_thickness,
                      'input_vertex_dtype': str(geometry.vertices.dtype)})
    else:
        size = geometry.radius if isinstance(geometry, Sphere) else geometry.radii
        shape['size'] = size.detach().cpu().tolist()
    return {**shape, 'exterior': asdict(exterior), 'interior': asdict(interior), 'sources': source_data,
            'source_declarations': tuple(repr(source) for source in scene.sources),
            'domain': repr(scene.domain), 'boundary': repr(scene.boundary), 'symmetry': scene.symmetry,
            'monitors': tuple(repr(monitor) for monitor in scene.monitors)}


def migrate_surface_v1_to_archive(source_path, *, scene, frequency):
    """Validate a trusted historical snapshot and retain a non-executable v2 archive.

    Input may be an inner surface payload or an outer Result snapshot. Like
    Result.load, this reads trusted torch serialization, not untrusted uploads.
    The source bytes and old coefficient bytes remain unchanged. Re-solving in
    the lifted formulation is a separate explicit Simulation operation.
    """
    source_bytes = Path(source_path).read_bytes()
    source = torch.load(io.BytesIO(source_bytes), map_location='cpu', weights_only=False)
    if not isinstance(source, dict):
        raise ValueError('Historical MoM snapshot must be a dictionary')
    payload = source.get('surface_data', source)
    if not isinstance(payload, dict) or type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
        raise ValueError('Archive migration accepts only an explicit inner MoM schema v1')
    if (payload.get('basis') != 'RWG' or payload.get('coefficient_order') != ['J', 'M']
            or payload.get('coefficient_units') != ['A/m', 'V/m'] or payload.get('coordinate_units') != 'm'):
        raise ValueError('Historical MoM current or coordinate semantics are invalid')
    dtype = {'torch.complex64': torch.complex64, 'torch.complex128': torch.complex128}.get(payload.get('dtype'))
    config = MoMConfig(payload['subdivisions'], IntegrationConfig(**payload['integration']),
        StoppingRule(**payload['stopping']), dtype, FieldQueryConfig(**payload['query_config']),
        riesz_storage=payload.get('riesz_storage', 'dense'))
    metadata = payload.get('operator_metadata')
    if not isinstance(metadata, dict) or metadata.get('riesz_storage', 'dense') != config.riesz_storage:
        raise ValueError('Historical MoM Riesz storage metadata does not match')
    declared_formulation = metadata.get('formulation_id')
    if declared_formulation not in (None, 'pmchwt_exp_minus_iwt_total_trace_v1'):
        raise ValueError('Historical payload does not declare the total-RWG formulation')
    basis, exterior, interior = compile_surface(scene, frequency, config)
    if not _same_metadata(payload.get('scene_signature'), _historical_signature(scene, exterior, interior)):
        raise ValueError('Historical snapshot does not match the supplied Scene and frequency')
    mesh = basis.mesh
    for key, expected in (('vertices', mesh.vertices), ('faces', mesh.faces)):
        value = payload.get(key)
        if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                or not torch.equal(value.to(expected.device), expected)):
            raise ValueError('Historical snapshot mesh does not match the supplied Scene')
    coefficients = payload.get('coefficients')
    if (not isinstance(coefficients, torch.Tensor) or coefficients.dtype != dtype
            or coefficients.shape != (2*basis.n_edges,) or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Historical coefficient tensor is invalid')
    index = next(i for i, structure in enumerate(scene.structures) if structure.enabled)
    identity_keys = ('formulation_id', 'decoder_version', 'source_version')
    return {'schema_version': 2, 'payload_kind': 'historical_total_rwg_archive', 'executable': False,
        'migration_version': 'total_rwg_v1_archive_migration_v1',
        'source_payload_sha256': hashlib.sha256(source_bytes).hexdigest(), 'source_bytes': source_bytes,
        'basis': 'RWG', 'current_kind': 'total_rwg_v1', 'coefficient_order': ['J', 'M'],
        'coefficient_units': ['A/m', 'V/m'], 'coordinate_units': 'm',
        'coefficients': coefficients.clone(), 'dtype': str(dtype),
        'body_table': [{'region_id': 1, 'structure_index': index, 'name': scene.structures[index].name,
            'edge_offset': 0, 'face_offset': 0, 'edge_count': basis.n_edges, 'face_count': len(mesh.faces),
            'J_slice': [0, basis.n_edges], 'M_slice': [basis.n_edges, 2*basis.n_edges],
            'vertices': payload['vertices'].clone(), 'faces': payload['faces'].clone(), 'interior': asdict(interior)}],
        'exterior': asdict(exterior), 'scene_signature': copy.deepcopy(payload['scene_signature']),
        'declared_historical_identities': {key: metadata.get(key) for key in identity_keys},
        'unknown_historical_identities': [key for key in identity_keys if metadata.get(key) is None],
        'historical_identities_independently_verified': False,
        'original_surface_payload': copy.deepcopy(payload),
        'physical_qualification': False,
        'notice': 'Exact historical archive only. New lifted currents require a new solve and new evidence.'}
