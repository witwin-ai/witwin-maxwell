"""Historical bytes may be archived but cannot be relabeled as lifted solutions."""
from dataclasses import asdict
import hashlib
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SurfaceData, MoMConfig, compile_surface
from witwin.maxwell.mom.migration import migrate_surface_v1_to_archive, _historical_signature

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_offline_archive_preserves_bytes_and_rejects_execution(tmp_path):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25))], sources=[mw.PlaneWave()])
    config = MoMConfig(subdivisions=0)
    basis, exterior, interior = compile_surface(scene, 3e8, config)
    # An arbitrary historical current is valid storage, not a qualified solve.
    coefficients = torch.arange(2*basis.n_edges, dtype=torch.float64).to(torch.complex128)
    coefficients[0] = complex(-0., -0.)
    payload = {'schema_version': 1, 'basis': 'RWG', 'coefficient_order': ['J', 'M'],
        'coefficient_units': ['A/m', 'V/m'], 'coordinate_units': 'm', 'coefficients': coefficients,
        'vertices': basis.mesh.vertices.cpu(), 'faces': basis.mesh.faces.cpu(),
        'integration': asdict(config.integration), 'query_config': asdict(config.query_config),
        'stopping': asdict(config.stopping), 'subdivisions': 0, 'dtype': str(config.dtype),
        'scene_signature': _historical_signature(scene, exterior, interior), 'operator_metadata': {}}
    source = tmp_path/'historical.pt'
    torch.save(payload, source)
    original = source.read_bytes()
    archive = migrate_surface_v1_to_archive(source, scene=scene, frequency=3e8)
    assert archive['source_bytes'] == original == source.read_bytes()
    assert archive['source_payload_sha256'] == hashlib.sha256(original).hexdigest()
    assert torch.equal(archive['coefficients'].view(torch.uint8), coefficients.view(torch.uint8))
    assert archive['executable'] is False and archive['physical_qualification'] is False
    assert archive['unknown_historical_identities'] == ['formulation_id', 'decoder_version', 'source_version']
    with pytest.raises(ValueError, match='historical'):
        SurfaceData.restore(archive, scene, 3e8)
    changed = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=3.))], sources=[mw.PlaneWave()])
    with pytest.raises(ValueError, match='does not match'):
        migrate_surface_v1_to_archive(source, scene=changed, frequency=3e8)
