"""Pair partitioning preserves cell ordering, field derivatives and trace guards."""
import pytest
import torch
from witwin.maxwell.mom import vie_operator as vie
from witwin.maxwell.mom.fields import _triangle_distance, _pair_triangle_distance
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_partitioned_volume_pairs_preserve_values_gradients_and_edge_rejection(monkeypatch):
    vertices = torch.tensor([[0.,0.,0.], [.1,0.,0.], [0.,.1,0.], [0.,0.,.1]],
                            device='cuda', dtype=torch.float64)
    coarse = TetrahedralVolumeMesh(vertices, torch.tensor([[0,1,2,3]], device='cuda'), region_id=1)
    vertices = torch.cat((vertices, vertices.mean(0, keepdim=True)))
    cells = torch.cat((coarse.boundary.faces[:,[0,2,1]],
                       torch.full((4,1),4,device='cuda',dtype=torch.int64)),dim=1)
    mesh = TetrahedralVolumeMesh(vertices, cells, region_id=1)
    points = torch.tensor([[.011,.019,.033],[.013,.017,.041],[.2,-.1,.2]],
                          device='cuda', dtype=torch.float64, requires_grad=True)
    reference = vie.cell_green_potentials(mesh, points, 7+.4j, order=8)
    derivative, = torch.autograd.grad(sum(v.real.square().sum() for v in reference), points)
    monkeypatch.setattr(vie, '_MAX_PAIR_SAMPLES', 3*8**2)
    actual = vie.cell_green_potentials(mesh, points, 7+.4j, order=8)
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-14)
    gradient, = torch.autograd.grad(sum(v.real.square().sum() for v in actual), points)
    torch.testing.assert_close(gradient, derivative, rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match='edge'):
        vie.cell_green_potentials(mesh, (vertices[0:1]+vertices[1:2])/2, 7+.4j,
                                 order=8, principal_value=True)
    empty = vie.cell_green_potentials(mesh, points[:0], 7+.4j, order=8, principal_value=True)
    assert all(value.shape[0] == 0 for value in empty)
    triangles = mesh.boundary.triangles
    full = _triangle_distance(points, triangles)
    pairs = _pair_triangle_distance(points[:,None].expand(-1,len(triangles),-1).reshape(-1,3),
                                   triangles.repeat(len(points),1,1)).reshape_as(full)
    torch.testing.assert_close(pairs, full, rtol=0, atol=0)
