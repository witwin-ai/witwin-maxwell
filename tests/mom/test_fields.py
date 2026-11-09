"""Analytic Huygens checks and CUDA current-vector differentiation."""

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.fields import FieldQueryConfig, decode, far_field, poynting_flux, radar_cross_section
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.kernels import FREE_SPACE
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import compile_medium, incident_fields


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="MoM CUDA field evaluation required")


def _handle(subdivisions=1, *, order=6):
    basis = RWGBasis(sphere_mesh(radius=0.08, subdivisions=subdivisions))
    medium = compile_medium(mw.Material(), 4e8)
    return SimpleNamespace(basis=basis, exterior=medium, interior=medium, kernel=FREE_SPACE,
                           sources=(mw.PlaneWave(),), integration=IntegrationConfig(order=order, singular_order=order+2))


def _project_plane_wave_trace(handle):
    basis = handle.basis
    points, weights = basis.mesh.quadrature(6)
    e, h = incident_fields(handle.sources[0], points, handle.exterior)
    normal = basis.mesh.normals[:, None].expand_as(points).to(e.dtype)
    j = torch.linalg.cross(normal, h)
    m = -torch.linalg.cross(normal, e)
    values = basis.local_values(points).to(e.dtype)
    mass = basis.mass_matrix().to(e.dtype)
    projected = []
    for current in (j, m):
        local = torch.einsum("tqlc,tqc,tq->tl", values, current, weights.to(e.dtype))
        rhs = torch.zeros(basis.n_edges, dtype=e.dtype, device=e.device).scatter_add(0, basis.face_edges.flatten(), local.flatten())
        projected.append(torch.linalg.solve(mass, rhs))
    return torch.cat(projected)


def test_rwg_huygens_plane_wave_interior_and_exterior_converge():
    errors = []
    for level in (0, 1, 2):
        handle = _handle(level)
        x = _project_plane_wave_trace(handle)
        query_i = torch.tensor([[0.01, 0.002, 0.003]], device="cuda", dtype=torch.float64)
        query_o = torch.tensor([[0.15, 0.07, 0.02]], device="cuda", dtype=torch.float64)
        ei, hi = decode(handle, x, query_i, "interior", "total")
        expected_e, expected_h = incident_fields(handle.sources[0], query_i, handle.exterior)
        eo, ho = decode(handle, x, query_o, "exterior", "scattered")
        errors.append(float(torch.linalg.vector_norm(ei - expected_e) + abs(handle.exterior.eta) * torch.linalg.vector_norm(hi - expected_h)
                            + torch.linalg.vector_norm(eo) + abs(handle.exterior.eta) * torch.linalg.vector_norm(ho)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.035


def test_zero_currents_incident_total_and_region_query_rejection():
    handle = _handle(0)
    x = torch.zeros(2 * handle.basis.n_edges, device="cuda", dtype=torch.complex128)
    query = torch.tensor([[0.3, 0.01, 0.02]], device="cuda", dtype=torch.float64)
    e, h = decode(handle, x, query, 0, "total")
    ei, hi = incident_fields(handle.sources[0], query, handle.exterior)
    torch.testing.assert_close(e, ei)
    torch.testing.assert_close(h, hi)
    with pytest.raises(ValueError, match="region_id"):
        decode(handle, x, query, 1, "total")
    with pytest.raises(ValueError, match="on-surface"):
        decode(handle, x, handle.basis.mesh.centroids[:1], 0, "total")
    with pytest.raises(ValueError, match="on-surface"):
        decode(handle, x, query, 0, "total", trace_side="exterior")
    with pytest.raises(ValueError, match="CUDA"):
        decode(handle, x, query.cpu(), 0, "total")


def test_far_field_matches_large_radius_decoding_and_impedance():
    handle = _handle(1)
    torch.manual_seed(55)
    x = torch.randn(2 * handle.basis.n_edges, dtype=torch.complex128, device="cuda")
    x[handle.basis.n_edges:] *= handle.exterior.eta
    direction = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64, device="cuda")
    fe, fh = far_field(handle, x, direction)
    radius = 1e5
    e, h = decode(handle, x, radius * direction, 0, "scattered")
    phase = torch.exp(torch.tensor(-1j * handle.exterior.k * radius, device="cuda", dtype=x.dtype))
    torch.testing.assert_close(radius * phase * e, fe, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(radius * phase * h, fh, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(fh, torch.linalg.cross(direction.to(fe.dtype), fe) / handle.exterior.eta)
    assert bool((radar_cross_section(fe, 1j) >= 0).all())


def test_decoder_current_gradient_matches_real_and_imaginary_perturbations():
    handle = _handle(0, order=4)
    x = torch.randn(2 * handle.basis.n_edges, dtype=torch.complex128, device="cuda", requires_grad=True)
    query = torch.tensor([[0.14, 0.05, 0.07]], dtype=torch.float64, device="cuda")
    def objective(current):
        e, h = decode(handle, current, query, 0, "scattered")
        return e.abs().square().sum() + abs(handle.exterior.eta)**2 * h.abs().square().sum()
    objective(x).backward()
    for direction in (1., 1j):
        perturb = torch.zeros_like(x)
        perturb[3] = 1e-5 * direction
        finite_difference = (objective(x.detach() + perturb) - objective(x.detach() - perturb)) / 2e-5
        expected = (x.grad[3].conj() * direction).real
        torch.testing.assert_close(finite_difference, expected, rtol=1e-6, atol=1e-6)


def test_near_panel_duffy_quadrature_converges_with_order():
    handle = _handle(0)
    torch.manual_seed(42)
    x = torch.randn(2 * handle.basis.n_edges, device="cuda", dtype=torch.complex128)
    query = handle.basis.mesh.centroids[:1] + 0.005 * handle.basis.mesh.normals[:1]
    values = []
    for order in (4, 8, 16, 32):
        handle.integration = IntegrationConfig(order=order, singular_order=order, near_max_order=order+2)
        e, h = decode(handle, x, query, 0, "scattered", query_config=FieldQueryConfig(adaptive=False))
        values.append(torch.cat((e.flatten(), abs(handle.exterior.eta) * h.flatten())))
    errors = [torch.linalg.vector_norm(value - values[-1]) for value in values[:-1]]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / torch.linalg.vector_norm(values[-1]) < 1e-6


def test_poynting_flux_matches_plane_wave_energy():
    handle = _handle(0)
    points = torch.zeros((2, 3), device="cuda", dtype=torch.float64)
    e, h = incident_fields(handle.sources[0], points, handle.exterior, amplitude=2j)
    normals = torch.tensor([[0., 0., 1.], [0., 0., 1.]], device="cuda", dtype=torch.float64)
    weights = torch.tensor([0.1, 0.2], device="cuda", dtype=torch.float64)
    flux = poynting_flux(e, h, normals, weights)
    assert float(flux) == pytest.approx(0.5 * 4 * 0.3 / handle.exterior.eta.real)


def test_actual_potential_approaches_explicit_one_sided_surface_trace():
    handle = _handle(0)
    torch.manual_seed(46)
    x = torch.randn(2 * handle.basis.n_edges, device="cuda", dtype=torch.complex128)
    query, normal = handle.basis.mesh.centroids[:1], handle.basis.mesh.normals[:1]
    for region, kind, side, sign in ((0, "scattered", "exterior", 1), (1, "total", "interior", -1)):
        et, ht = decode(handle, x, query, region, kind, trace_side=side)
        errors = []
        for distance in (0.01, 0.005, 0.0025):
            e, h = decode(handle, x, query + sign*distance*normal, region, kind)
            errors.append(torch.linalg.vector_norm(e-et) + abs(handle.exterior.eta)*torch.linalg.vector_norm(h-ht))
        assert errors[2] < 0.6*errors[1] < 0.4*errors[0]


def test_surface_huygens_trace_converges_and_rejects_edges():
    errors = []
    for level in (0, 1, 2):
        handle = _handle(level)
        x = _project_plane_wave_trace(handle)
        query = handle.basis.mesh.centroids[:3]
        ei, hi = decode(handle, x, query, 1, "total", trace_side="interior")
        eo, ho = decode(handle, x, query, 0, "scattered", trace_side="exterior")
        e, h = incident_fields(handle.sources[0], query, handle.exterior)
        errors.append(float(torch.linalg.vector_norm(ei-e) + abs(handle.exterior.eta)*torch.linalg.vector_norm(hi-h)
                            + torch.linalg.vector_norm(eo) + abs(handle.exterior.eta)*torch.linalg.vector_norm(ho)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.16
    vertices = handle.basis.mesh.vertices
    edge = vertices[handle.basis.edges[:1]].mean(1)
    for query in (vertices[:1], edge):
        with pytest.raises(ValueError, match="edge and vertex"):
            decode(handle, x, query, 0, "scattered", trace_side="exterior")


def test_adaptive_field_error_and_exhausted_budget_are_logged():
    handle = _handle(0)
    torch.manual_seed(42)
    x = torch.randn(2 * handle.basis.n_edges, device="cuda", dtype=torch.complex128)
    query = handle.basis.mesh.centroids[:1] + 0.005*handle.basis.mesh.normals[:1]
    decode(handle, x, query, 0, "scattered")
    log = handle.last_field_query_log
    assert log["converged"] and log["checks"] and log["final_order"] > handle.integration.singular_order
    assert log["seconds"] > 0 and log["peak_allocated_bytes"] > 0
    with pytest.raises(RuntimeError, match="did not converge"):
        decode(handle, x, query, 0, "scattered", query_config=FieldQueryConfig(max_order=10, relative_tolerance=1e-12))
    assert not handle.last_field_query_log["converged"]
    assert handle.last_field_query_log["failure_reason"] == "quadrature_order_budget_exhausted"


def test_surface_trace_retains_current_autograd():
    handle = _handle(0)
    x = torch.randn(2 * handle.basis.n_edges, device="cuda", dtype=torch.complex128, requires_grad=True)
    query = handle.basis.mesh.centroids[:1]
    def objective(current):
        e, _ = decode(handle, current, query, 0, "scattered", trace_side="exterior")
        return e.abs().square().sum()
    objective(x).backward()
    perturbation = torch.zeros_like(x)
    perturbation[0] = 1e-5j
    derivative = (objective(x.detach()+perturbation)-objective(x.detach()-perturbation))/2e-5
    torch.testing.assert_close(derivative, x.grad[0].imag, rtol=2e-6, atol=1e-6)


def test_off_plane_static_gradient_solid_angle_sign_against_direct_quadrature():
    from witwin.maxwell.mom.fields import _static_panel_gradient
    from witwin.maxwell.mom.quadrature import green, triangle_rule
    triangle = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], device="cuda", dtype=torch.float64)
    query = torch.tensor([[.2, .3, .2], [.2, .3, -.2], [1.4, .5, .1]], device="cuda", dtype=torch.float64)
    triangles = triangle.expand(3, -1, -1)
    points, weights = triangle_rule(triangles, 64)
    _, gradient = green(query[:, None]-points, 0.)
    numerical = (gradient*weights[..., None]).sum(1).real
    analytic = _static_panel_gradient(query, triangles)
    torch.testing.assert_close(analytic, numerical, rtol=1e-10, atol=1e-12)
    assert analytic[0, 2] < 0 < analytic[1, 2]


def test_very_near_face_static_subtraction_converges_without_extreme_order():
    handle = _handle(0)
    torch.manual_seed(142)
    x = torch.randn(2*handle.basis.n_edges, device="cuda", dtype=torch.complex128)
    point = handle.basis.mesh.centroids[:1] + 1e-6*handle.basis.mesh.normals[:1]
    e, h = decode(handle, x, point, 0, "scattered")
    assert torch.isfinite(e).all() and torch.isfinite(h).all()
    assert handle.last_field_query_log["converged"]
    assert handle.last_field_query_log["final_order"] <= 64


def test_far_query_batches_preserve_fields_and_actual_coefficient_gradients():
    from dataclasses import replace
    handle = _handle(1, order=4)
    torch.manual_seed(532)
    x = torch.randn(2*handle.basis.n_edges, device="cuda", dtype=torch.complex128, requires_grad=True)
    points = torch.randn(20, 3, device="cuda", dtype=torch.float64)
    points = .8*points/torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    outputs, gradients = [], []
    def objective(current):
        e, h = decode(handle, current, points, "exterior", "scattered")
        return e, h, e.abs().square().sum()+abs(handle.exterior.eta)**2*h.abs().square().sum()
    for batch in (64, 1024):
        handle.integration = replace(handle.integration, far_pair_batch=batch)
        e, h, loss = objective(x)
        outputs.append((e, h))
        gradients.append(torch.autograd.grad(loss, x)[0])
    for first, second in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(first, second, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-11, atol=2e-10)
    for direction in (1., 1j):
        perturbation = torch.zeros_like(x)
        perturbation[5] = 1e-4*direction
        difference = (objective(x.detach()+perturbation)[2]-objective(x.detach()-perturbation)[2])/2e-4
        torch.testing.assert_close(difference, (gradients[1][5].conj()*direction).real, rtol=2e-6, atol=2e-7)


def test_high_order_far_query_batch_bounds_quadrature_workspace(monkeypatch):
    from dataclasses import replace
    import witwin.maxwell.mom.fields as fields
    handle = _handle(1, order=4)
    handle.integration = replace(handle.integration, order=64, singular_order=64, near_max_order=66, far_pair_batch=100000)
    x = torch.ones(2*handle.basis.n_edges, device="cuda", dtype=torch.complex128)
    points = torch.tensor([[.8, .2, .1]], device="cuda", dtype=torch.float64)
    original, sample_counts = fields.triangle_rule, []
    def observed_rule(triangles, order):
        result = original(triangles, order)
        sample_counts.append(result[1].numel())
        return result
    monkeypatch.setattr(fields, "triangle_rule", observed_rule)
    e, h = decode(handle, x, points, "exterior", "scattered", query_config=FieldQueryConfig(adaptive=False))
    assert torch.isfinite(e).all() and torch.isfinite(h).all()
    assert sample_counts and max(sample_counts) <= 262144 and len(sample_counts) > 1
