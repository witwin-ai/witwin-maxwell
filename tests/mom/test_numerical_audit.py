"""Independent numerical audit regressions for invalid geometry and devices."""

import pytest
import torch

from witwin.maxwell.mom.mesh import SurfaceMesh, _has_intersections, sphere_mesh
from witwin.maxwell.mom.quadrature import gauss_rule


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical audit required")


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_closed_topology_does_not_allow_geometrically_pinched_vertices(scale):
    mesh = sphere_mesh(radius=scale, subdivisions=0)
    vertices = mesh.vertices.clone()
    # Opposite vertex stars have distinct global IDs and share no incident edge.
    vertices[0] = vertices[3]
    with pytest.raises(ValueError, match="coincident vertices"):
        SurfaceMesh(vertices, mesh.faces)


@pytest.mark.parametrize("second", [
    [[.2, .2, 0.], [.2, .2, 1.], [.6, .2, 1.]],
    [[.5, 0., 0.], [.5, -.2, 1.], [.7, -.2, 1.]],
    [[.5, 0., 0.], [.5, -.2, 0.], [.7, -.2, 0.]],
])
def test_nonincident_face_and_edge_endpoint_contacts_are_rejected(second):
    triangles = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], second], device="cuda", dtype=torch.float64)
    assert _has_intersections(triangles)


def test_conforming_shared_edge_and_vertex_are_not_self_intersections():
    first = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    for second in (
        [[1., 0., 0.], [0., 0., 0.], [.5, -.2, 0.]],
        [[0., 0., 0.], [-1., 0., 1.], [0., -1., 0.]],
    ):
        triangles = torch.tensor([first, second], device="cuda", dtype=torch.float64)
        assert not _has_intersections(triangles)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required to audit implicit-device cache")
def test_gauss_cache_resolves_implicit_cuda_device_before_lookup():
    with torch.cuda.device(0):
        first, _ = gauss_rule(3, device="cuda")
    with torch.cuda.device(1):
        second, weights = gauss_rule(3, device="cuda")
        assert second.device == weights.device == torch.device("cuda:1")
        explicit, _ = gauss_rule(3, device="cuda:1")
        assert explicit.data_ptr() == second.data_ptr()
    assert first.device == torch.device("cuda:0")
    torch.testing.assert_close(first, second.to(first.device))


@pytest.mark.host_validation
def test_reference_error_uses_solid_angle_not_gauss_sample_count():
    import numpy as np
    from benchmark.mom_convergence import far_field_error_metrics
    reference = np.array([[1., 0., 0.], [1., 0., 0.]])
    numerical = np.array([[2., 0., 0.], [1., 0., 0.]])
    report = far_field_error_metrics(numerical, reference, np.array([.1, .9])*4*np.pi)
    assert report["far_field_relative_error"] == pytest.approx(np.sqrt(.1))
    assert report["far_field_sample_relative_error"] == pytest.approx(np.sqrt(.5))
    assert report["far_field_error_norm"] == "solid_angle_weighted_relative_L2"
    zero = far_field_error_metrics(numerical, np.zeros_like(reference), np.array([.1, .9])*4*np.pi)
    assert zero["far_field_relative_error"] is None
    assert zero["far_field_absolute_L2_error"] > 0
