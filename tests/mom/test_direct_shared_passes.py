"""One global exterior traversal across many disjoint material bodies."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.direct_system import DirectSystemActions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_three_body_global_exterior_uses_one_projection(monkeypatch):
    exterior = compile_medium(mw.Material(), 3e8)
    entries = []
    for i, center in enumerate(((-.5, 0., 0.), (.5, 0., 0.), (0., 1., 0.))):
        basis = RWGBasis(sphere_mesh(radius=.01, center=center, subdivisions=0, device='cuda'))
        interior = compile_medium(mw.Material(eps_r=1.5+.2*i, mu_r=1.1+.1*i), 3e8)
        entries.append((i, str(i), basis, interior))
    provider = ScalarSystemBlocks(SurfaceSystem.create(entries), exterior, IntegrationConfig(order=3, singular_order=3))
    operator = DirectSystemActions(provider, order=6, quadrature_order=6, leaf_size=8)
    assert operator.exterior.test.tree.size == 90
    assert len(operator.interiors) == 3 and len(operator.exterior.far) >= 2
    assert all(not block.far for block in operator.interiors)
    reference, _ = assemble_system(provider.system, exterior, provider.config)
    x = torch.randn((2, 3, 180), device='cuda', dtype=torch.complex128, requires_grad=True)
    y = torch.randn_like(x)
    calls = {'up': 0, 'down': 0}
    def counted(function, kind):
        def wrapper(value):
            calls[kind] += 1
            return function(value)
        return wrapper
    projection = operator.exterior.test
    monkeypatch.setattr(projection, 'upward', counted(projection.upward, 'up'))
    monkeypatch.setattr(projection, 'downward', counted(projection.downward, 'down'))
    actual = operator.apply(x)
    assert calls == {'up': 1, 'down': 1}
    actual_adjoint = operator.adjoint(y)
    assert calls == {'up': 2, 'down': 2}
    torch.testing.assert_close(actual, x@reference.mT, rtol=1e-8, atol=1e-14)
    torch.testing.assert_close(actual_adjoint, y@reference.conj(), rtol=1e-8, atol=1e-14)
    loss = (actual.conj()*y).sum().real
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], actual_adjoint, rtol=2e-12, atol=1e-15)
    assert operator.receipt()['far_projection_passes'] == 'one_global_exterior_plus_body_local_interiors'


def test_global_far_coupling_groups_multiple_bodies_on_both_sides():
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=1.5), 3e8)
    entries = []
    for i, center in enumerate(((-1., .02, 0.), (-1., .04, 0.), (1., .02, 0.), (1., .04, 0.))):
        basis = RWGBasis(sphere_mesh(radius=.003, center=center, subdivisions=0, device='cuda'))
        entries.append((i, str(i), basis, interior))
    provider = ScalarSystemBlocks(SurfaceSystem.create(entries), exterior, IntegrationConfig(order=3, singular_order=3))
    operator = DirectSystemActions(provider, order=6, quadrature_order=6, leaf_size=8)
    projection = operator.exterior.test
    grouped = []
    for a, b, kernel in operator.exterior.far:
        rows = projection.tree.permutation[projection.starts[a]:projection.stops[a]]
        cols = projection.tree.permutation[projection.starts[b]:projection.stops[b]]
        if len((rows//30).unique()) > 1 and len((cols//30).unique()) > 1:
            grouped.append((a, b))
    assert len(grouped) >= 2
    rows, cols = torch.arange(60, 120, device='cuda'), torch.arange(60, device='cuda')
    reference = provider(rows, cols)
    source_ids = torch.cat((cols, cols+120))
    target_ids = torch.cat((rows, rows+120))
    value = torch.zeros((120, 240), device='cuda', dtype=torch.complex128)
    value[:, source_ids] = torch.eye(120, device='cuda', dtype=torch.complex128)
    actual = operator.apply(value)[:, target_ids].mT
    rs = torch.tensor([abs(exterior.eta)]*60+[1.]*60, device='cuda', dtype=torch.float64)
    cs = torch.tensor([1.]*60+[abs(exterior.eta)]*60, device='cuda', dtype=torch.float64)
    error = float(torch.linalg.vector_norm(rs[:, None]*(actual-reference)*cs)/torch.linalg.vector_norm(rs[:, None]*reference*cs))
    print('grouped multibody far error', error, 'grouped node pairs', len(grouped))
    assert error < 1e-8
