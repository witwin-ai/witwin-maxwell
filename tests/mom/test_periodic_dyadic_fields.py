"""Tensor image field integration against scalar divergence-form Ewald fields."""
import copy
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
from witwin.maxwell.mom.periodic_fields import regular_surface_fields
from witwin.maxwell.mom.periodic_dyadic_fields import regular_dyadic_surface_fields

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')



def interior_reference(lattice, medium, kind, monkeypatch):
    from witwin.maxwell.mom.periodic_dyadic import (
        QuasiperiodicDyadicGreenKernel, QuasiperiodicDyadicRegularKernel)
    if kind == 'spectral_2d':
        lattice = LatticeSpec(torch.cat((lattice.vectors,
            lattice.vectors.new_tensor([[0., 1., 0.]]))), lattice.bloch)
    if kind == 'image':
        reference = LossyDyadicImageReference(lattice, medium, 1)
    else:
        reference = QuasiperiodicDyadicRegularKernel(
            QuasiperiodicDyadicGreenKernel(lattice, medium), 8, .12, 3, 1e-4, 16,
            normal=lattice.vectors.new_tensor([0., 0., 1.]) if kind == 'spectral_1d' else None)
    def forbidden(*args, **kwargs):
        pytest.fail('Local interior recovery must not evaluate exterior periodic images')
    monkeypatch.setattr(type(reference), 'evaluate', forbidden)
    return reference


def test_tensor_image_fields_match_independent_scalar_ewald_and_current_gradient():
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = handle.system.bodies[0].mesh.vertices
    exterior = compile_medium(mw.Material(sigma_e=.04, sigma_m=2000.), handle.exterior.frequency)
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    tensor = TensorMedium(exterior.frequency, exterior.epsilon_r*eye, exterior.mu_r*eye)
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    reference = LossyDyadicImageReference(lattice, tensor, 6)
    kernel = EwaldHelmholtzKernel(lattice, 3., real_shells=3, reciprocal_shells=8, quadrature_order=128)
    coefficients = torch.arange(2*handle.system.total_edges, device=sample.device,
                                dtype=sample.dtype).to(torch.complex128)
    coefficients = ((coefficients+1)/31+1j*(coefficients.flip(0)+1)/23).requires_grad_()
    points = sample.new_tensor([[.1, .12, .15], [.4, .2, .1]])
    actual, receipt = regular_dyadic_surface_fields(handle.system, coefficients, points,
        reference=reference, order=6, face_batch=2, query_batch=1)
    scalar = copy.copy(handle)
    scalar.exterior = exterior
    expected = regular_surface_fields(scalar, coefficients, points, kernel, 6)
    balanced = lambda fields: torch.cat((fields[0], abs(exterior.eta)*fields[1]), dim=-1)
    error = (balanced(actual)-balanced(expected)).norm()/balanced(expected).norm()
    assert error < 2e-5
    actual_grad, = torch.autograd.grad(balanced(actual).abs().square().sum(), coefficients)
    expected_grad, = torch.autograd.grad(balanced(expected).abs().square().sum(), coefficients)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=1e-12)
    assert not receipt['primary_included'] and not receipt['complete_physical_fields']
    assert not receipt['physical_qualification']
    with pytest.raises(ValueError, match='all-J/all-M'):
        regular_dyadic_surface_fields(handle.system, coefficients[:-1], points, reference=reference)
    print({'balanced_field_relative_difference': float(error.detach())})


def test_complete_exterior_fields_scalar_limit_bloch_phase_and_region_guards():
    from types import SimpleNamespace
    from witwin.maxwell.mom.fields import _potentials
    from witwin.maxwell.mom.kernels import FREE_SPACE
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic_fields import exterior_dyadic_surface_fields
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = handle.system.bodies[0].mesh.vertices
    exterior = compile_medium(mw.Material(sigma_e=.04, sigma_m=2000.), handle.exterior.frequency)
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    tensor = TensorMedium(exterior.frequency, exterior.epsilon_r*eye, exterior.mu_r*eye)
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    reference = LossyDyadicImageReference(lattice, tensor, 6)
    coefficients = torch.linspace(.1, 1., 2*handle.system.total_edges,
                                   device=sample.device).to(torch.complex128)
    points = sample.new_tensor([[.12, .6, .7]])
    translation = 4*lattice.vectors[0]
    queries = torch.cat((points, points+translation))
    actual, receipt = exterior_dyadic_surface_fields(handle.system, coefficients, queries,
        reference=reference, order=6)
    scalar = copy.copy(handle)
    scalar.exterior = exterior
    kernel = EwaldHelmholtzKernel(lattice, 3., real_shells=3, reciprocal_shells=8, quadrature_order=128)
    e, h = regular_surface_fields(scalar, coefficients, points, kernel, 6)
    for i, body in enumerate(handle.system.bodies):
        local = torch.cat((coefficients[handle.system.J_slices[i]], coefficients[handle.system.M_slices[i]]))
        free = SimpleNamespace(basis=body.basis, integration=IntegrationConfig(order=6), kernel=FREE_SPACE)
        pe, ph = _potentials(free, local, points, exterior)
        e, h = e+pe, h+ph
    for field, expected in zip(actual, (e, h)):
        torch.testing.assert_close(field[:1], expected, rtol=2e-6, atol=1e-12)
        torch.testing.assert_close(field[1:], field[:1]*torch.exp(1j*(translation@lattice.bloch)),
                                   rtol=1e-11, atol=1e-14)
    assert receipt['primary_included'] and not receipt['incident_included']
    assert not receipt['physical_qualification']
    inside = sample.mean(0, keepdim=True)+translation
    with pytest.raises(ValueError, match='body-interior'):
        exterior_dyadic_surface_fields(handle.system, coefficients, inside, reference=reference)
    on_surface = handle.system.bodies[0].mesh.triangles[:1].mean(1)
    with pytest.raises(ValueError, match='trace_side'):
        exterior_dyadic_surface_fields(handle.system, coefficients, on_surface, reference=reference)


def test_half_cell_boundary_has_unique_bloch_reduction_with_short_image_sum():
    from witwin.maxwell.mom.periodic_dyadic_fields import exterior_dyadic_surface_fields
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = handle.system.bodies[0].mesh.vertices
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    tensor = TensorMedium(handle.exterior.frequency, (1+.1j)*eye, (1+.1j)*eye)
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    reference = LossyDyadicImageReference(lattice, tensor, 1)
    coefficients = torch.ones(2*handle.system.total_edges, device=sample.device, dtype=torch.complex128)
    points = sample.new_tensor([[-.5, .6, .7], [.5, .6, .7], [1.5, .6, .7]])
    actual, receipt = exterior_dyadic_surface_fields(handle.system, coefficients, points,
        reference=reference, order=3)
    for field in actual:
        phase = torch.exp(1j*lattice.bloch[0])
        torch.testing.assert_close(field[1:], field[:-1]*phase, rtol=1e-12, atol=1e-14)
    assert not receipt['image_cutoff_converged']


@pytest.mark.parametrize('regular_kind', ('image', 'spectral_1d', 'spectral_2d'))
def test_tensor_interior_recovers_incident_field_without_other_body_currents(monkeypatch, regular_kind):
    from witwin.maxwell.mom.tensor_carrier import TensorIncidentCarrier
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.fields import FieldQueryConfig
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic_fields import interior_dyadic_surface_fields
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = handle.system.bodies[0].mesh.vertices
    scalar = compile_medium(mw.Material(sigma_e=.01, sigma_m=50.), handle.exterior.frequency)
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    medium = TensorMedium(scalar.frequency, scalar.epsilon_r*eye, scalar.mu_r*eye)
    q = eye.new_tensor([[.2, 0., (scalar.k**2-.2**2)**.5]])
    electric = eye.new_tensor([[0., 1., 0.]])
    magnetic = torch.linalg.cross(q, electric)/(scalar.omega*scalar.mu)
    carrier = TensorIncidentCarrier(medium, q, electric, magnetic)
    system = SurfaceSystem.create([(body.structure_index, body.name, body.basis, medium)
                                   for body in handle.system.bodies])
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    reference = interior_reference(lattice, medium, regular_kind, monkeypatch)
    coefficients = eye.new_zeros(2*system.total_edges)
    points = torch.cat((sample.mean(0, keepdim=True), sample.mean(0, keepdim=True)+lattice.vectors[0]))
    kwargs = dict(carrier=carrier, reference=reference, region_id=1,
        config=IntegrationConfig(order=4, singular_order=4),
        query_config=FieldQueryConfig(relative_tolerance=1e-6, absolute_tolerance=1e-9, max_order=32))
    actual, receipt = interior_dyadic_surface_fields(system, coefficients, points, **kwargs)
    expected = carrier.evaluate(points, medium)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=2e-6, atol=2e-8 if value is actual[0] else 1e-10)
    coefficients[system.J_slices[1]] = 1.
    coefficients[system.M_slices[1]] = 2.
    other, _ = interior_dyadic_surface_fields(system, coefficients, points, **kwargs)
    for value, target in zip(other, actual):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    assert receipt['local_recovery']['converged']
    assert not receipt['interior_images_added'] and not receipt['cross_body_interior_terms']


@pytest.mark.parametrize('regular_kind', ('image', 'spectral_1d', 'spectral_2d'))
def test_tensor_background_interior_trace_matches_analytic_sheet_jump(monkeypatch, regular_kind):
    from witwin.maxwell.mom.tensor_carrier import TensorIncidentCarrier
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.fields import FieldQueryConfig
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic_fields import interior_dyadic_surface_fields
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body = handle.system.bodies[0]
    sample = body.mesh.vertices
    scalar = compile_medium(mw.Material(sigma_e=.01, sigma_m=50.), handle.exterior.frequency)
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    medium = TensorMedium(scalar.frequency, scalar.epsilon_r*eye, scalar.mu_r*eye)
    q = eye.new_tensor([[.2, 0., (scalar.k**2-.2**2)**.5]])
    e = eye.new_tensor([[0., 1., 0.]])
    carrier = TensorIncidentCarrier(medium, q, e, torch.linalg.cross(q, e)/(scalar.omega*scalar.mu))
    system = SurfaceSystem.create([(body.structure_index, body.name, body.basis, medium)])
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    points = body.mesh.centroids[:1]+lattice.vectors[0]
    actual, receipt = interior_dyadic_surface_fields(system, eye.new_zeros(2*system.total_edges), points,
        carrier=carrier, reference=interior_reference(lattice, medium, regular_kind, monkeypatch),
        region_id=1, trace_side='interior', config=IntegrationConfig(order=4, singular_order=4),
        query_config=FieldQueryConfig(relative_tolerance=1e-6, absolute_tolerance=1e-9, max_order=64))
    expected = carrier.evaluate(points, medium)
    for i, (value, target) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(value, target, rtol=2e-6, atol=2e-8 if i == 0 else 1e-10)
    assert receipt['local_recovery']['principal_value']
    assert receipt['local_recovery']['converged']


def test_exterior_tensor_trace_matches_scalar_principal_value_and_jump():
    from types import SimpleNamespace
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.fields import _potentials, _trace_jump
    from witwin.maxwell.mom.kernels import FREE_SPACE
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic_fields import exterior_dyadic_surface_fields
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body = handle.system.bodies[0]
    system = SurfaceSystem.create([(body.structure_index, body.name, body.basis, body.interior)])
    sample = body.mesh.vertices
    exterior = compile_medium(mw.Material(sigma_e=.04, sigma_m=2000.), handle.exterior.frequency)
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    medium = TensorMedium(exterior.frequency, exterior.epsilon_r*eye, exterior.mu_r*eye)
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    coefficients = torch.linspace(.1, 1., 2*system.total_edges, device=sample.device,
                                   dtype=sample.dtype).to(eye.dtype)
    points = body.mesh.centroids[:1]
    actual, receipt = exterior_dyadic_surface_fields(system, coefficients, points,
        reference=LossyDyadicImageReference(lattice, medium, 6), order=32, trace_side='exterior')
    local = SimpleNamespace(basis=body.basis, integration=IntegrationConfig(order=32, singular_order=32), kernel=FREE_SPACE)
    faces = torch.zeros(1, device=sample.device, dtype=torch.int64)
    primary = _potentials(local, coefficients, points, exterior, surface_faces=faces)
    jump = _trace_jump(local, coefficients, points, faces, exterior)
    scalar = SimpleNamespace(system=system, exterior=exterior, integration=local.integration)
    kernel = EwaldHelmholtzKernel(lattice, 3., real_shells=3, reciprocal_shells=8, quadrature_order=128)
    images = regular_surface_fields(scalar, coefficients, points, kernel, 6)
    expected = tuple(p+.5*j+i for p, j, i in zip(primary, jump, images))
    for i, (value, target) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(value, target, rtol=2e-6, atol=1e-8 if i == 0 else 1e-10)
    assert receipt['trace_side'] == 'exterior'
    assert not receipt['physical_qualification']
