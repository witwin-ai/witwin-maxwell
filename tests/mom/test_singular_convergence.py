"""Independent moments and kernel convergence for joint touching rules."""

import math
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig, _adaptive_near_integrals, _pair_integrals, assemble_weak
from witwin.maxwell.mom.quadrature import (
    classify_pairs, duffy_fan, green, touching_pair_rule, triangle_rule,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def pair(category):
    left = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    right, ids = {
        0: (left, [0, 1, 2]),
        1: ([[1., 0., 0.], [0., 0., 0.], [0.1, -0.5, 0.8]], [1, 0, 3]),
        2: ([[0., 0., 0.], [-0.6, 0.1, 0.7], [0., -1., 0.]], [0, 3, 4]),
    }[category]
    return (torch.tensor([left], device="cuda", dtype=torch.float64),
            torch.tensor([right], device="cuda", dtype=torch.float64),
            torch.tensor([[0, 1, 2]], device="cuda"), torch.tensor([ids], device="cuda"))


@pytest.mark.parametrize("category", [0, 1, 2])
def test_joint_rules_preserve_area_and_first_moments(category):
    left, right, li, ri = pair(category)
    x, y, w = touching_pair_rule(left, right, li, ri, category, 6)
    area = lambda t: torch.linalg.vector_norm(torch.linalg.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), dim=-1)/2
    exact = area(left)*area(right)
    torch.testing.assert_close(w.sum(-1), exact)
    torch.testing.assert_close((x*w[..., None]).sum(1), exact[:, None]*left.mean(1))
    torch.testing.assert_close((y*w[..., None]).sum(1), exact[:, None]*right.mean(1))
    torch.testing.assert_close(torch.einsum("bqi,bqj,bq->bij", x, y, w), exact[:, None, None]*left.mean(1)[:, :, None]*right.mean(1)[:, None, :])
    assert torch.linalg.vector_norm(x-y, dim=-1).min() > 0
    assert (w > 0).all()


def test_self_static_scalar_against_closed_form():
    left, right, li, ri = pair(0)
    # Exact pair integral of 1/|x-y| on the unit right triangle.
    exact = (2+math.sqrt(2))*math.asinh(1)/3
    errors = []
    for order in (3, 6, 12):
        x, y, w = touching_pair_rule(left, right, li, ri, 0, order)
        integral = (w/torch.linalg.vector_norm(x-y, dim=-1)).sum().item()
        errors.append(abs(integral-exact))
    assert errors[2] < 1e-9
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("category", [1, 2])
def test_touching_scalar_and_gradient_converge_and_reciprocate(category):
    left, right, li, ri = pair(category)
    def integrate(order, reverse=False):
        args = (right, left, ri, li) if reverse else (left, right, li, ri)
        x, y, w = touching_pair_rule(*args, category, order)
        g, grad = green(x-y, 1.7+0.2j)
        return (g*w).sum(), (grad*w[..., None]).sum((0, 1))
    reference = integrate(18)
    errors = []
    for order in (4, 8, 12):
        scalar, gradient = integrate(order)
        errors.append(max(abs((scalar-reference[0]).item()), torch.linalg.vector_norm(gradient-reference[1]).item()))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-6
    reverse = integrate(18, reverse=True)
    torch.testing.assert_close(reference[0], reverse[0], atol=2e-10, rtol=2e-8)
    torch.testing.assert_close(reference[1], -reverse[1], atol=2e-8, rtol=2e-6)


def test_all_five_actual_rwg_blocks_converge():
    basis = RWGBasis(sphere_mesh(subdivisions=1))
    count = len(basis.mesh.faces)
    rows = torch.zeros(count, dtype=torch.long, device="cuda")
    cols = torch.arange(count, device="cuda")
    classes = classify_pairs(basis.mesh, rows, cols)
    medium = SimpleNamespace(k=2.1+0.1j)
    for category in range(5):
        col = cols[classes == category][:1]
        assert len(col) == 1
        def blocks(order):
            return torch.cat([value.flatten() for value in _pair_integrals(basis, rows[:1], col, medium, order, order, category != 4, category)])
        reference = blocks(16)
        errors = [torch.linalg.vector_norm(blocks(order)-reference).item()/torch.linalg.vector_norm(reference).item() for order in (4, 8, 12)]
        assert errors[2] < 3e-4, (category, errors)
        assert errors[2] < errors[0], (category, errors)


def test_projected_fan_integrates_constant_outside_panel():
    left, _, _, _ = pair(0)
    observer = torch.tensor([[1.3, 0.7, 0.2]], dtype=torch.float64, device="cuda")
    _, weights = duffy_fan(observer, left, 4)
    torch.testing.assert_close(weights.sum(1), torch.tensor([0.5], dtype=torch.float64, device="cuda"))
    on_edge = torch.tensor([[0.5, 0., 0.]], dtype=torch.float64, device="cuda")
    points, weights = duffy_fan(on_edge, left, 3)
    kernel, _ = green(on_edge[:, None]-points, 1.)
    assert torch.isfinite(kernel*weights).all()


def test_near_adaptation_reports_cost_and_rejects_unmet_budget():
    basis = RWGBasis(sphere_mesh(subdivisions=1))
    cols = torch.arange(len(basis.mesh.faces), device="cuda")
    rows = torch.zeros_like(cols)
    chosen = cols[classify_pairs(basis.mesh, rows, cols) == 3][:3]
    medium = SimpleNamespace(k=2.1+0.1j)
    config = IntegrationConfig(singular_order=4, near_tolerance=1e-5, near_max_order=16)
    d, k, stats = _adaptive_near_integrals(basis, rows[:3], chosen, medium, config)
    assert stats["pair_evaluations"] >= 6
    assert stats["maximum_order"] > 4
    assert stats["maximum_accepted_relative_difference"] <= 1e-5
    d_ref, k_ref = _pair_integrals(basis, rows[:3], chosen, medium, 18, 18, True, 3)
    torch.testing.assert_close(d, d_ref, atol=1e-8, rtol=2e-4)
    torch.testing.assert_close(k, k_ref, atol=1e-8, rtol=2e-4)
    with pytest.raises(RuntimeError, match="Panel quadrature failed tolerance"):
        _adaptive_near_integrals(basis, rows[:3], chosen, medium, IntegrationConfig(singular_order=2, near_tolerance=1e-14, near_max_order=4))


def test_actual_assembled_operator_obeys_arbitrary_edge_sign_transform():
    basis = RWGBasis(sphere_mesh(subdivisions=0))
    medium = SimpleNamespace(k=2.1+0.1j, frequency=1e8, epsilon_r=2.+0.2j, mu_r=1.)
    config = IntegrationConfig(order=3, singular_order=4, near_tolerance=1e-4)
    original, _ = assemble_weak(basis, medium, medium, config)
    signs = torch.where(torch.arange(basis.n_edges, device="cuda") % 3 == 0, -1, 1)
    basis.face_signs *= signs[basis.face_edges]
    transformed, _ = assemble_weak(basis, medium, medium, config)
    both = signs.repeat(2)
    torch.testing.assert_close(transformed, original*both[:, None]*both[None], atol=2e-12, rtol=1e-12)
