from types import SimpleNamespace

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium, incident_fields
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak, build_rhs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def system():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0))
    medium = compile_medium(mw.Material(), 3e8)
    matrix, metadata = assemble_weak(basis, medium, medium, IntegrationConfig(order=4, singular_order=6))
    return basis, medium, matrix, metadata


def test_actual_operator_adjoint_and_zero_source(system):
    from witwin.maxwell.mom.linalg import DenseOperator, apply_A, apply_A_adjoint, residual_loss
    basis, medium, a, metadata = system
    mass = basis.mass_matrix().to(a.dtype)
    handle = DenseOperator(a, torch.block_diag(mass, mass))
    x = torch.randn(a.shape[0], device=a.device, dtype=a.dtype, requires_grad=True)
    y = torch.randn_like(x)
    torch.testing.assert_close(torch.vdot(apply_A(handle, x), y), torch.vdot(x, apply_A_adjoint(handle, y)))
    assert torch.count_nonzero(build_rhs(basis, medium, [])) == 0
    assert sum(metadata["triangle_pair_counts"]) == len(basis.mesh.faces)**2
    loss = residual_loss(handle, x, torch.zeros_like(x))
    grad, = torch.autograd.grad(loss, x)
    expected = a.mH@torch.linalg.solve(handle.test_riesz, a@x)
    torch.testing.assert_close(grad, expected)


def test_no_contrast_trace_discretization(system):
    basis, medium, a, metadata = system
    source = mw.PlaneWave()
    b = build_rhs(basis, medium, [source])
    x = torch.linalg.solve(a, b)
    points, weights = basis.mesh.quadrature(6)
    e, h = incident_fields(source, points, medium)
    normals = basis.mesh.normals[:, None].expand_as(points).to(e.dtype)
    trace = [torch.linalg.cross(normals, h), -torch.linalg.cross(normals, e)]
    values = basis.local_values(points).to(e.dtype)
    mass = basis.mass_matrix().to(e.dtype)
    projected = []
    for target in trace:
        rhs = torch.zeros(basis.n_edges, device=e.device, dtype=e.dtype)
        rhs.index_add_(0, basis.face_edges.flatten(), torch.einsum("tqlc,tqc,tq->tl", values, target, weights.to(e.dtype)).flatten())
        projected.append(torch.linalg.solve(mass, rhs))
    target = torch.cat(projected)
    scale = torch.cat([torch.full_like(x[:basis.n_edges], medium.eta), torch.ones_like(x[basis.n_edges:])])
    error = torch.linalg.vector_norm((x-target)*scale)/torch.linalg.vector_norm(target*scale)
    # A coarse 20-face polyhedron is not a G0 reference; this detects signs/units.
    assert error < .35, error.item()
    assert torch.linalg.vector_norm(a@x-b)/torch.linalg.vector_norm(b) < 1e-12
