import math
import pytest
import torch

from witwin.maxwell.mom.quadrature import duffy_fan, gauss_rule, green, triangle_rule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_gauss_and_triangle_polynomials():
    nodes, weights = gauss_rule(5, device="cuda")
    for power in range(10):
        torch.testing.assert_close((weights*nodes**power).sum(), torch.tensor(1/(power+1), device="cuda", dtype=torch.float64))
    tri = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]]], device="cuda", dtype=torch.float64)
    points, weights = triangle_rule(tri, 5)
    torch.testing.assert_close(weights.sum(), torch.tensor(3., device="cuda", dtype=torch.float64))
    torch.testing.assert_close((weights[..., None]*points).sum(1), torch.tensor([[2., 3., 0.]], device="cuda", dtype=torch.float64))


@pytest.mark.parametrize("observer", [[.2, .3, 0.], [3., -1., .4]])
def test_signed_fan_integrates_area_and_centroid(observer):
    tri = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]]], device="cuda", dtype=torch.float64)
    points, weights = duffy_fan(torch.tensor([observer], device="cuda", dtype=torch.float64), tri, 6)
    torch.testing.assert_close(weights.sum(), torch.tensor(3., device="cuda", dtype=torch.float64))
    torch.testing.assert_close((weights[..., None]*points).sum(1), torch.tensor([[2., 3., 0.]], device="cuda", dtype=torch.float64))


def test_self_singularity_against_analytic_vertex_integral():
    tri = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], device="cuda", dtype=torch.float64)
    # Integral over the right triangle of 1/r from its right-angle vertex.
    observer = torch.zeros((1, 3), device="cuda", dtype=torch.float64)
    points, weights = duffy_fan(observer, tri, 24)
    # Degenerate fan pieces have zero weight and can contain zero-distance nodes.
    active = weights != 0
    value, _ = green((observer[:, None]-points)[active], 0.)
    integral = (4*math.pi*value*weights[active]).sum().real
    expected = math.sqrt(2)*math.asinh(1)
    assert abs(integral.item()-expected) < 1e-12
