"""Complete-support tensor potentials versus scalar fields and native gradients."""
from types import SimpleNamespace
import pytest
import torch
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.mesh import RWGBasis
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.anisotropic_fields import tensor_current_potentials

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_tensor_potentials_scalar_reduction_and_current_gradient():
    from witwin.core import PhysicalMaterial
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.fields import _potentials
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.kernels import FreeSpaceHelmholtzKernel
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (2+.1j)*eye, (1.3+.2j)*eye)
    scalar = compile_medium(PhysicalMaterial(eps_r=2+.1j, mu_r=1.3+.2j), 3e8)
    generator = torch.Generator(device='cuda').manual_seed(92)
    current = torch.randn(2*basis.n_edges, generator=generator, device='cuda', dtype=torch.complex128, requires_grad=True)
    points = torch.tensor([[.4, .3, .5], [-.5, .3, -.4]], device='cuda', dtype=torch.float64)
    electric, magnetic, receipt = tensor_current_potentials(basis, current, points, DyadicGreenKernel(medium), order=8)
    handle = SimpleNamespace(basis=basis, integration=IntegrationConfig(order=8), kernel=FreeSpaceHelmholtzKernel())
    expected = _potentials(handle, current, points, scalar)
    for actual, reference in zip((electric, magnetic), expected):
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-11)
    actual_gradient, = torch.autograd.grad(electric.abs().square().sum()+magnetic.abs().square().sum(), current)
    expected_gradient, = torch.autograd.grad(sum(v.abs().square().sum() for v in expected), current)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-10, atol=1e-11)
    assert receipt['surface_error_certified'] is False
    with pytest.raises(ValueError, match='trace_side'):
        tensor_current_potentials(basis, current, basis.mesh.triangles[:1].mean(1), DyadicGreenKernel(medium))


def test_tensor_surface_refinement_matches_higher_order_and_retains_failure():
    from witwin.maxwell.mom.anisotropic_fields import evaluate_tensor_current_potentials, TensorFieldConvergenceError
    from witwin.maxwell.mom.fields import FieldQueryConfig
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    kernel = DyadicGreenKernel(TensorMedium(3e8, (2+.1j)*eye, (1.3+.2j)*eye))
    generator = torch.Generator(device='cuda').manual_seed(109)
    current = torch.randn(2*basis.n_edges, generator=generator, device='cuda', dtype=torch.complex128)
    points = torch.tensor([[.4, .3, .5]], device='cuda', dtype=torch.float64)
    e, h, receipt = evaluate_tensor_current_potentials(basis, current, points, kernel, initial_order=2,
        query_config=FieldQueryConfig(relative_tolerance=1e-7, absolute_tolerance=1e-10, max_order=16))
    reference_e, reference_h, _ = tensor_current_potentials(basis, current, points, kernel, order=20)
    torch.testing.assert_close(e, reference_e, rtol=1e-7, atol=1e-10)
    torch.testing.assert_close(h, reference_h, rtol=1e-7, atol=1e-12)
    assert receipt['converged'] and not receipt['surface_error_certified']
    with pytest.raises(TensorFieldConvergenceError) as caught:
        evaluate_tensor_current_potentials(basis, current, points, kernel, initial_order=1,
            query_config=FieldQueryConfig(relative_tolerance=1e-12, absolute_tolerance=0., max_order=2))
    failure = caught.value.receipt
    assert not failure['converged'] and failure['final_order'] == 2
    assert failure['checks'][-1]['E']['failed_query_count'] == 1
    assert failure['failure_reason'] == 'surface_quadrature_order_budget_exhausted'


def test_tensor_sheet_jump_satisfies_four_maxwell_sheet_conditions():
    from witwin.maxwell.mom.anisotropic_fields import tensor_sheet_jump
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon = eye.new_tensor([[2+.2j, .3+.1j, .2], [.3-.1j, 3+.2j, .1], [.2, .1, 1+.2j]])
    mu = eye.new_tensor([[1+.1j, .2j, .1], [-.2j, 2+.1j, .3], [.1, .3, 1.5+.1j]])
    medium = TensorMedium(3e8, epsilon, mu, reciprocal=False)
    normals = torch.tensor([[.6, 0, .8], [0, 1., 0]], device='cuda', dtype=torch.float64)
    n = normals.to(eye.dtype)
    j = torch.linalg.cross(n, eye.new_tensor([[1, 2j, 3], [.2j, .3, .1]]))
    m = torch.linalg.cross(n, eye.new_tensor([[.2j, .1, .4], [1, 2, .5j]]))
    divj, divm = eye.new_tensor([.3+.1j, -.2j]), eye.new_tensor([.1j, -.3])
    e, h = tensor_sheet_jump(normals, j, m, divj, divm, medium)
    torch.testing.assert_close(torch.linalg.cross(n, h), j, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(torch.linalg.cross(n, e), -m, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(torch.einsum('ni,ij,nj->n', n, medium.epsilon, e), divj/(1j*medium.omega), rtol=1e-13, atol=1e-24)
    torch.testing.assert_close(torch.einsum('ni,ij,nj->n', n, medium.mu, h), divm/(1j*medium.omega), rtol=1e-12, atol=1e-21)


def test_tensor_rwg_jump_reduces_to_scalar_and_rejects_wrong_face():
    from witwin.core import PhysicalMaterial
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.fields import _trace_jump
    from witwin.maxwell.mom.anisotropic_fields import tensor_rwg_trace_jump
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (2+.1j)*eye, (1.3+.2j)*eye)
    scalar = compile_medium(PhysicalMaterial(eps_r=2+.1j, mu_r=1.3+.2j), 3e8)
    generator = torch.Generator(device='cuda').manual_seed(111)
    current = torch.randn(2*basis.n_edges, generator=generator, device='cuda', dtype=torch.complex128)
    faces = torch.tensor([0, 3, 10], device='cuda')
    points = basis.mesh.triangles[faces].mean(1)
    actual = tensor_rwg_trace_jump(basis, current, points, faces, medium)
    expected = _trace_jump(SimpleNamespace(basis=basis), current, points, faces, scalar)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-13, atol=1e-12)
    with pytest.raises(ValueError, match='declared surface triangle'):
        tensor_rwg_trace_jump(basis, current, points, faces+1, medium)


def test_analytic_incident_sheet_reproduces_interior_and_extinction_outside():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.carrier import IncidentCarrier
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.anisotropic_fields import tensor_incident_potentials
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    carrier = IncidentCarrier.compile([mw.PlaneWave()], exterior)
    handle = SimpleNamespace(basis=basis, exterior=exterior, incident_carrier=carrier)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    kernel = DyadicGreenKernel(TensorMedium(3e8, eye, eye))
    points = torch.tensor([[.01, .02, 0.], [.4, .3, .5]], device='cuda', dtype=torch.float64)
    electric, magnetic, receipt = tensor_incident_potentials(handle, points, kernel, order=16)
    expected_e, expected_h = carrier.evaluate(points[:1], exterior)
    # The outward equivalent sheet radiates minus the incident solution inside
    # and vanishes outside the closed surface (the extinction identity).
    torch.testing.assert_close(electric[:1], -expected_e, rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(magnetic[:1], -expected_h, rtol=2e-7, atol=1e-10)
    assert torch.linalg.vector_norm(electric[1]) < 1e-9
    assert torch.linalg.vector_norm(magnetic[1]) < 1e-11
    assert receipt['incident_carrier_identity'] == carrier.identity
