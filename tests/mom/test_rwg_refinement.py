"""Planar nested RWG transfer preserves fields, divergence, metric and adjoint."""
import pytest
import torch
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.rwg_refinement import RWGRefinement
from witwin.maxwell.mom.operator import local_geometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_exact_nested_rwg_transfer_and_current_gradient():
    coarse = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    transfer = RWGRefinement(coarse)
    fine, parents = transfer.fine, transfer.parent_faces
    torch.manual_seed(23)
    q = torch.randn((2, coarse.n_edges), device='cuda', dtype=torch.complex128, requires_grad=True)
    lifted = transfer.apply(q)
    bary = coarse.mesh.vertices.new_tensor([[.2, .3, .5], [.4, .1, .5]])
    points = torch.einsum('qv,fvc->fqc', bary, fine.mesh.triangles)
    _, free, slopes = local_geometry(coarse)
    values = slopes[parents, None, :, None]*(points[:, :, None]-free[parents, None])
    expected = torch.einsum('bfl,fqlc->bfqc', q[:, coarse.face_edges[parents]], values.to(q.dtype))
    actual = torch.einsum('bfl,fqlc->bfqc', lifted[:, fine.face_edges], fine.local_values(points).to(q.dtype))
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-12)
    torch.testing.assert_close(lifted@fine.divergence.to(q.dtype).T,
                              (q@coarse.divergence.to(q.dtype).T)[:, parents], rtol=2e-12, atol=1e-11)
    probe = torch.randn_like(lifted)
    gradient, = torch.autograd.grad(lifted, q, probe)
    torch.testing.assert_close(gradient, transfer.adjoint(probe), rtol=1e-13, atol=1e-13)
    p = transfer.apply(torch.eye(coarse.n_edges, device='cuda', dtype=torch.float64)).T
    torch.testing.assert_close(p.T@fine.mass_matrix()@p, coarse.mass_matrix(), rtol=2e-12, atol=1e-15)
    assert transfer.columns.shape == (fine.n_edges, 3)
    transfer.weights.add_(1)
    with pytest.raises(ValueError, match='changed'):
        transfer.apply(q)
