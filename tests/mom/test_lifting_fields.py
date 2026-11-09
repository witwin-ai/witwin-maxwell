"""Carrier field identity, explicit total-current ownership and normal jumps."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import compile_surface, DecodeHandle
from witwin.maxwell.mom.lifting_fields import (
    carrier_difference_potentials, carrier_difference_jump, evaluate_total_currents,
)
from witwin.maxwell.mom.physics import incident_fields
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def handle(epsilon, *, dtype=torch.complex128, sources=None):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=epsilon))],
                     sources=[mw.PlaneWave()] if sources is None else sources)
    config = mw.MoMConfig(subdivisions=0, dtype=dtype, integration=IntegrationConfig(order=4, singular_order=6))
    basis, exterior, interior = compile_surface(scene, 3e8, config)
    return DecodeHandle(basis, exterior, interior, tuple(scene.sources), config.integration, {})


def test_lifted_observables_retain_the_prepared_rhs_carrier():
    from witwin.maxwell.mom.lifting import build_lifted_rhs
    from witwin.maxwell.mom.lifting_fields import decode_lifted
    op = handle(1.01)
    y = torch.zeros(2*op.basis.n_edges, device=op.basis.mesh.device, dtype=torch.complex128)
    points = torch.tensor([[.2, .1, .3]], device=y.device, dtype=torch.float64)
    interior = points*.05
    face_ids = torch.tensor([0], device=y.device)
    face_points = op.basis.mesh.triangles[face_ids].mean(1)
    rhs, metadata = build_lifted_rhs(op)
    baseline = [decode_lifted(op, y, points, 'exterior', 'total'),
                decode_lifted(op, y, points, 'exterior', 'incident'),
                decode_lifted(op, y, interior, 'interior', 'total'),
                evaluate_total_currents(op, y, face_points, face_ids)]
    # A physical RHS owns its prepared carrier, not a later mutable source list.
    op.sources = (mw.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1)),)
    actual = [decode_lifted(op, y, points, 'exterior', 'total'),
              decode_lifted(op, y, points, 'exterior', 'incident'),
              decode_lifted(op, y, interior, 'interior', 'total'),
              evaluate_total_currents(op, y, face_points, face_ids)]
    for expected_pair, actual_pair in zip(baseline, actual):
        for expected, value in zip(expected_pair, actual_pair):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    repeated, repeated_metadata = build_lifted_rhs(op)
    # CUDA index_add reductions may reorder roundoff on otherwise identical RHS.
    scale = torch.linalg.vector_norm(rhs)
    assert float(torch.linalg.vector_norm(repeated-rhs)/scale) < 1e-13
    assert metadata['incident_carrier_identity'] == repeated_metadata['incident_carrier_identity'] == op.incident_carrier.identity


def test_zero_material_carrier_difference_is_exact_zero_near_and_on_surface():
    op = handle(1.)
    centroids = op.basis.mesh.centroids[:2]
    off = .3*centroids
    points = torch.cat((centroids, off))
    faces = torch.tensor([0, 1, -1, -1], device=points.device)
    for field in carrier_difference_potentials(op, points, 6, faces):
        assert torch.count_nonzero(field) == 0
    for field in carrier_difference_jump(op, centroids, faces[:2]):
        assert torch.count_nonzero(field) == 0


def test_total_current_has_exact_carrier_and_current_vector_gradient():
    op = handle(2.25)
    faces = torch.tensor([0, 3], device=op.basis.mesh.device)
    points = op.basis.mesh.centroids[faces]
    y = torch.zeros(2*op.basis.n_edges, device=points.device, dtype=torch.complex128, requires_grad=True)
    j, m = evaluate_total_currents(op, y, points, faces)
    e, h = incident_fields(op.sources[0], points, op.exterior)
    normal = op.basis.mesh.normals[faces].to(e.dtype)
    torch.testing.assert_close(j, torch.linalg.cross(normal, h))
    torch.testing.assert_close(m, -torch.linalg.cross(normal, e))
    loss = j.abs().square().sum()+m.abs().square().sum()
    derivative, = torch.autograd.grad(loss, y)
    assert bool(torch.isfinite(derivative).all()) and bool((derivative.abs() > 0).any())
    direction = torch.polar(torch.ones_like(y.real), torch.linspace(.1, 2., len(y), device=y.device, dtype=y.real.dtype))
    epsilon = 1e-5
    def objective(value):
        return sum(field.abs().square().sum() for field in evaluate_total_currents(op, value, points, faces))
    finite_difference = (objective(epsilon*direction)-objective(-epsilon*direction))/(2*epsilon)
    analytical = (derivative.conj()*direction).real.sum()
    torch.testing.assert_close(analytical, finite_difference, rtol=1e-7, atol=1e-9)
    with pytest.raises(ValueError, match='declared faces'):
        evaluate_total_currents(op, y, points+.001*normal.real, faces)
    with pytest.raises(ValueError, match='face index'):
        evaluate_total_currents(op, y, points, faces+100)


def test_carrier_normal_jump_matches_two_sided_field_limits():
    op = handle(2.25)
    faces = torch.tensor([0, 7], device=op.basis.mesh.device)
    points = op.basis.mesh.centroids[faces]
    normals = op.basis.mesh.normals[faces]
    jump = carrier_difference_jump(op, points, faces)
    errors = []
    no_faces = torch.full_like(faces, -1)
    for distance in (.001, .0001):
        outside = carrier_difference_potentials(op, points+distance*normals, 16, no_faces)
        inside = carrier_difference_potentials(op, points-distance*normals, 16, no_faces)
        residuals = [a-b-c for a, b, c in zip(outside, inside, jump)]
        error = torch.sqrt(residuals[0].abs().square().sum()+abs(op.exterior.eta)**2*residuals[1].abs().square().sum())
        scale = torch.sqrt(jump[0].abs().square().sum()+abs(op.exterior.eta)**2*jump[1].abs().square().sum())
        errors.append(float(error/scale))
    assert errors[1] < errors[0]/3
    assert errors[1] < .01


def test_lifted_decoder_zero_contrast_matches_incident_inside_outside_and_traces():
    from witwin.maxwell.mom.lifting_fields import decode_lifted
    op = handle(1.)
    y = torch.zeros(2*op.basis.n_edges, device=op.basis.mesh.device, dtype=torch.complex128)
    base = op.basis.mesh.centroids[:2]
    for points, region, trace in ((.3*base, 'interior', None), (3*base, 'exterior', None),
                                  (base, 'interior', 'interior'), (base, 'exterior', 'exterior')):
        actual = decode_lifted(op, y, points, region, 'total', trace)
        expected = incident_fields(op.sources[0], points, op.exterior)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    with pytest.raises(ValueError, match='region_id'):
        decode_lifted(op, y, base, True, 'total')
    with pytest.raises(ValueError, match='interior queries'):
        decode_lifted(op, y, .3*base, 'interior', 'scattered')


def test_nonzero_lifted_interior_limits_and_current_vector_jacobian():
    from witwin.maxwell.mom.fields import decode, FieldQueryConfig
    from witwin.maxwell.mom.lifting_fields import decode_lifted
    op = handle(2.25)
    mesh = op.basis.mesh
    values = torch.linspace(-.002, .002, 2*op.basis.n_edges, device=mesh.device, dtype=mesh.dtype)
    y = torch.complex(values, values.flip(0)*.3).requires_grad_()
    faces = torch.tensor([0, 7], device=mesh.device)
    points = mesh.centroids[faces]
    # Fixed common quadrature isolates the limiting jump and coefficient
    # Jacobian from the separate adaptive quadrature decision process.
    config = FieldQueryConfig(adaptive=False)
    trace = decode_lifted(op, y, points, 'interior', 'total', 'interior', query_config=config)
    errors = []
    for distance in (.001, .0001):
        fields = decode_lifted(op, y, points-distance*mesh.normals[faces], 'interior', 'total', query_config=config)
        difference = [a-b for a, b in zip(fields, trace)]
        error = torch.sqrt(difference[0].abs().square().sum()+abs(op.interior.eta)**2*difference[1].abs().square().sum())
        scale = torch.sqrt(trace[0].abs().square().sum()+abs(op.interior.eta)**2*trace[1].abs().square().sum())
        errors.append(float((error/scale).detach()))
    assert errors[1] < errors[0]/3 and errors[1] < .01
    queries = .3*points
    total = decode_lifted(op, y, queries, 'interior', 'total', query_config=config)
    linear = decode(op, y, queries, 'interior', 'total', query_config=config)
    def objective(fields):
        return fields[0].real.sum()+.37*fields[0].imag.sum()+abs(op.interior.eta)*fields[1].imag.sum()
    actual, = torch.autograd.grad(objective(total), y, retain_graph=True)
    expected, = torch.autograd.grad(objective(linear), y)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_nonzero_adaptive_lifted_decoder_success_and_exhaustion():
    from witwin.maxwell.mom.fields import FieldQueryConfig
    from witwin.maxwell.mom.lifting_fields import decode_lifted
    op = handle(2.25)
    values = torch.linspace(-.002, .002, 2*op.basis.n_edges, device=op.basis.mesh.device, dtype=torch.float64)
    y = torch.complex(values, .3*values.flip(0))
    points = .3*op.basis.mesh.centroids[:2]
    fields = decode_lifted(op, y, points, 'interior', 'total')
    assert all(bool(torch.isfinite(value).all()) for value in fields)
    assert op.last_field_query_log['converged'] and op.last_field_query_log['checks']
    assert op.last_field_query_log['estimator'] == 'complete_lifted_field_successive_order_difference'
    with pytest.raises(RuntimeError, match='did not converge'):
        decode_lifted(op, y, points, 'interior', 'total', query_config=FieldQueryConfig(
            relative_tolerance=1e-14, absolute_tolerance=1e-15, max_order=8))
    assert op.last_field_query_log['failure_reason'] == 'quadrature_order_budget_exhausted'
    with pytest.raises(ValueError, match='must exceed'):
        decode_lifted(op, y, points, 'interior', 'total', query_config=FieldQueryConfig(max_order=6))
    assert op.last_field_query_log['failure_reason'] == 'initial_order_leaves_no_refinement'


def test_multisource_lifted_fields_complex64_match_complex128():
    from witwin.maxwell.mom.lifting_fields import decode_lifted
    sources = [mw.PlaneWave(source_time=mw.CW(frequency=3e8, phase=.37)),
               mw.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1),
                            source_time=mw.CW(frequency=3e8, phase=-.51, amplitude=.31))]
    outputs = []
    for dtype in (torch.complex128, torch.complex64):
        op = handle(2.25, dtype=dtype, sources=sources)
        values = torch.linspace(-.002, .002, 2*op.basis.n_edges,
                                device=op.basis.mesh.device, dtype=op.basis.mesh.dtype)
        y = torch.complex(values, values.flip(0)*.3)
        points = torch.tensor([[.01, .02, .01], [-.02, .01, .01]], device=y.device, dtype=values.dtype)
        fields = decode_lifted(op, y, points, 'interior', 'total')
        assert op.last_field_query_log['converged']
        outputs.append(fields)
    for reference, actual in zip(*outputs):
        assert actual.dtype == torch.complex64
        error = torch.linalg.vector_norm(actual.to(torch.complex128)-reference)/torch.linalg.vector_norm(reference)
        assert float(error) < 2e-4
