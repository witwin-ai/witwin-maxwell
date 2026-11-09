"""Selected analytic lifting retains dense RHS semantics and row ordering."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.runtime import SystemDecodeHandle
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.lifting import build_lifted_rhs, build_selected_lifted_rhs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_selected_lifted_rows_match_full_analytic_rhs(monkeypatch):
    basis = RWGBasis(sphere_mesh(radius=.07, subdivisions=0, device='cuda'))
    system = SurfaceSystem.create([(0, 'body', basis, compile_medium(mw.Material(eps_r=2., mu_r=1.2), 3e8))])
    config = mw.MoMConfig(integration=IntegrationConfig(order=3, singular_order=4, pair_batch=7, far_pair_batch=11))
    handle = SystemDecodeHandle(system, compile_medium(mw.Material(), 3e8), [mw.PlaneWave()], config, {}).body_handles[0]
    full, _ = build_lifted_rhs(handle)
    import witwin.maxwell.mom.lifting as lifting
    def forbidden(*args, **kwargs):
        raise AssertionError('Selected RHS must not classify all global pairs')
    monkeypatch.setattr(lifting, 'classify_all_pairs', forbidden)
    selected = torch.tensor([17, 0, 4], device='cuda')
    actual, receipt = build_selected_lifted_rhs(handle, selected)
    torch.testing.assert_close(actual, full[torch.cat((selected, selected+basis.n_edges))], rtol=2e-12, atol=1e-16)
    assert receipt['maximum_classification_pairs'] <= 11
    assert sum(receipt['triangle_pair_counts']) == receipt['test_faces']*receipt['source_faces']
    with pytest.raises(ValueError, match='unique'):
        build_selected_lifted_rhs(handle, selected.repeat(2))
