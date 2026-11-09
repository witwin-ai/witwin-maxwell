"""Self-created geometry is retained ownership; caller-shared geometry is separate."""
import json
import pytest
import torch
from test_fmm_directional_rwg import _basis
from witwin.maxwell.mom.operator import PreparedRWGGeometry
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalRWGProjection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_directional_rwg_owned_and_shared_geometry_exact_inventory_preflight_and_mutation(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as module
    basis = _basis((0., 0., 0.))
    center = basis.mesh.vertices.mean(0); direction = center.new_tensor([1., 0., 0.])
    edges = torch.tensor([7, 2, 11], device='cuda'); order = 3; count = 20
    map_bytes = 48+len(edges)*8+2*len(edges)*count*4*16
    shared_geometry = PreparedRWGGeometry(basis)
    geometry_bytes = sum(t.untyped_storage().nbytes() for t in shared_geometry.tensors())
    assert geometry_bytes == len(basis.mesh.faces)*16*8
    owned = DirectionalRWGProjection(basis, edges, center, direction, 5.+.2j, order,
        quadrature_order=4, retained_budget_bytes=map_bytes+geometry_bytes)
    shared = DirectionalRWGProjection(basis, edges, center, direction, 5.+.2j, order,
        quadrature_order=4, retained_budget_bytes=map_bytes, geometry=shared_geometry)
    for obj, expected in ((owned, map_bytes+geometry_bytes), (shared, map_bytes)):
        tensors = obj.tensors()
        assert len({t.untyped_storage().data_ptr() for t in tensors}) == len(tensors)
        assert all(t._base is None and t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in tensors)
        assert sum(t.untyped_storage().nbytes() for t in tensors) == obj.retained_bytes == obj.planned_retained_bytes == expected
    assert owned.receipt()['owned_geometry_included']
    assert not owned.receipt()['caller_shared_geometry_excluded']
    assert shared.receipt()['caller_shared_geometry_excluded']
    assert not shared.receipt()['owned_geometry_included']
    for a, b in zip(owned.tensors()[:7], shared.tensors(), strict=True):torch.testing.assert_close(a, b, rtol=0, atol=0)
    def forbidden(*args, **kwargs):raise AssertionError('Failed retained budget reached geometry/integration')
    monkeypatch.setattr(module, 'PreparedRWGGeometry', forbidden)
    monkeypatch.setattr(module, '_rwg_weighted_moments', forbidden)
    with pytest.raises(MemoryError, match='retained budget'):
        DirectionalRWGProjection(basis, edges, center, direction, 5.+.2j, order,
            quadrature_order=4, retained_budget_bytes=map_bytes+geometry_bytes-1)
    owned.geometry.tensors()[0].add_(1)
    with pytest.raises(ValueError, match='changed'):owned.assert_current()
    shared_geometry.tensors()[1].add_(1)
    with pytest.raises(ValueError, match='changed'):shared.assert_current()
    record = dict(map_bytes=map_bytes, own_geometry_bytes=geometry_bytes,
        owned_retained_bytes=owned.retained_bytes, shared_map_retained_bytes=shared.retained_bytes,
        owned_tensor_count=len(owned.tensors()), shared_tensor_count=len(shared.tensors()),
        preflight_before_geometry=True, geometry_mutation_rejected=True,
        numerical_maps_bit_exact=True, physical_qualification=False)
    (tmp_path/'ownership.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
