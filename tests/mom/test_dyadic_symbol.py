"""Independent isotropic transverse/longitudinal and tensor covariance checks."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.dyadic_green import maxwell_resolvent, maxwell_symbol, cross_matrix

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_isotropic_resolvent_matches_longitudinal_and_transverse_formula():
    dtype, device, frequency = torch.complex128, 'cuda', 3e8
    k0 = 2*math.pi*frequency*math.sqrt(EPSILON_0*MU_0)
    q = torch.tensor([[.3, .2, .7], [1.2, -.4, .3]], device=device, dtype=dtype)
    eye = torch.eye(3, device=device, dtype=dtype)
    eps, mu = 2+.1j, 1.3+.2j
    inverse = maxwell_resolvent(k0*q, eps*eye, mu*eye, frequency)
    squared = (q*q).sum(-1)
    longitudinal = q[..., :, None]*q[..., None, :]/squared[:, None, None]
    transverse = eye-longitudinal
    denom = eps*mu-squared
    ee = longitudinal/eps + mu*transverse/denom[:, None, None]
    hh = -longitudinal/mu-eps*transverse/denom[:, None, None]
    eh = cross_matrix(q)/denom[:, None, None]
    torch.testing.assert_close(inverse[..., :3, :3], ee, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(inverse[..., 3:, 3:], hh, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(inverse[..., :3, 3:], eh, rtol=1e-13, atol=1e-14)


def test_general_tensor_rotation_covariance_and_maxwell_equations():
    dtype = torch.complex128
    eps = torch.tensor([[2+.2j, .3, .1], [.3, 3+.1j, 0], [.1, 0, 1+.1j]], device='cuda', dtype=dtype)
    mu = torch.tensor([[1+.1j, .2j, 0], [-.2j, 1+.1j, .1], [0, .1, 2+.1j]], device='cuda', dtype=dtype)
    q = torch.tensor([[.3+.1j, .5, 1.2]], device='cuda', dtype=dtype)
    rotation = torch.tensor([[.6, 0, .8], [0, 1, 0], [-.8, 0, .6]], device='cuda', dtype=dtype)
    global_rotation = torch.block_diag(rotation, rotation)
    original = maxwell_resolvent(q, eps, mu, 3e8)
    rotated = maxwell_resolvent(q@rotation.T, rotation@eps@rotation.T, rotation@mu@rotation.T, 3e8)
    torch.testing.assert_close(rotated, global_rotation@original@global_rotation.T, rtol=1e-13, atol=1e-14)
    symbol = maxwell_symbol(q, eps, mu, 3e8)
    torch.testing.assert_close(symbol@original, torch.eye(6, device='cuda', dtype=dtype)[None], rtol=1e-13, atol=1e-14)


def test_physical_current_source_signs_and_onsager_bias_reversal():
    from witwin.maxwell.mom.dyadic_green import current_green_spectrum
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    mu = (1+.1j)*eye
    mu[0, 1], mu[1, 0] = .3j, -.3j
    medium = TensorMedium(3e8, (2+.1j)*eye, mu, reciprocal=False)
    reversed_medium = TensorMedium(3e8, medium.epsilon_r, mu.T, reciprocal=False)
    q = torch.tensor([[.3, -.2, 1.2]], device='cuda', dtype=torch.complex128)
    green = current_green_spectrum(q, medium)
    source = torch.randn(6, device='cuda', dtype=torch.complex128)
    electric, magnetic = (green@source).split(3, dim=-1)
    electric_current, magnetic_current = source[:3], source[3:]
    torch.testing.assert_close(1j*torch.linalg.cross(q, electric),
        1j*medium.omega*(medium.mu@magnetic[..., None]).squeeze(-1)-magnetic_current, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(1j*torch.linalg.cross(q, magnetic),
        -1j*medium.omega*(medium.epsilon@electric[..., None]).squeeze(-1)+electric_current, rtol=1e-12, atol=1e-12)
    reversed_green = current_green_spectrum(-q, reversed_medium)
    parity = torch.diag(eye.new_tensor([1., 1., 1., -1., -1., -1.]))
    torch.testing.assert_close(green, parity@reversed_green.mT@parity, rtol=1e-12, atol=1e-12)
    same_bias = current_green_spectrum(-q, medium)
    assert torch.linalg.vector_norm(green-parity@same_bias.mT@parity) > 1e-5


def test_outgoing_plane_residues_match_scalar_weyl_spectrum():
    from witwin.maxwell.mom.dyadic_green import outgoing_plane_spectrum
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (2+.1j)*eye, (1.3+.05j)*eye)
    transverse = torch.tensor([[.2, .3], [2.5, .4], [0., 0.]], device='cuda', dtype=torch.float64)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    kz = k0*torch.sqrt(medium.epsilon_r[0, 0]*medium.mu_r[0, 0]-transverse.square().sum(-1))
    q = torch.cat((k0*transverse.to(kz.dtype), kz[:, None]), -1)
    scalar = 1j*torch.exp(1j*kz*.2)/(2*kz)
    projector = eye-q[..., :, None]*q[..., None, :]/(medium.omega**2*medium.epsilon[0, 0]*medium.mu[0, 0])
    ee = 1j*medium.omega*medium.mu[0, 0]*projector*scalar[:, None, None]
    hh = 1j*medium.omega*medium.epsilon[0, 0]*projector*scalar[:, None, None]
    curl = 1j*cross_matrix(q)*scalar[:, None, None]
    expected = torch.cat((torch.cat((ee, -curl), -1), torch.cat((curl, hh), -1)), -2)
    actual = outgoing_plane_spectrum(transverse, .2, medium)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-12)


def test_spatial_weyl_integral_matches_independent_closed_form():
    from witwin.maxwell.mom.dyadic_green import integrate_spatial_green
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps, mu = 2+.8j, 1.3+.5j
    medium = TensorMedium(3e8, eps*eye, mu*eye)
    points = torch.tensor([[.1, -.2, .15]], device='cuda', dtype=torch.float64)
    radius = torch.linalg.vector_norm(points, dim=-1)
    direction = (points/radius[:, None]).to(eye.dtype)
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*complex(eps)**.5*complex(mu)**.5
    g = torch.exp(1j*k*radius)/(4*math.pi*radius)
    first = g*(1j*k-1/radius)
    second = g*(-k*k-2j*k/radius+2/radius.square())
    nn = direction[..., :, None]*direction[..., None, :]
    hessian = second[:, None, None]*nn+(first/radius)[:, None, None]*(eye-nn)
    dyadic = g[:, None, None]*eye+hessian/(k*k)
    curl = cross_matrix(first[:, None]*direction)
    ee = 1j*medium.omega*medium.mu[0, 0]*dyadic
    hh = 1j*medium.omega*medium.epsilon[0, 0]*dyadic
    expected = torch.cat((torch.cat((ee, -curl), -1), torch.cat((curl, hh), -1)), -2)
    actual = integrate_spatial_green(points, medium, radial_order=96, azimuthal_order=32)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize('epsilon', [2.+.1j, -2.+.1j, -2.+0j])
def test_static_singularity_has_correct_scalar_branch(epsilon):
    from witwin.maxwell.mom.dyadic_green import electrostatic_kernel
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, epsilon*eye, eye)
    p = torch.tensor([[.1, -.2, .3]], device='cuda', dtype=torch.float64)
    r = torch.linalg.vector_norm(p, dim=-1)
    g, grad, hessian = electrostatic_kernel(p, medium)
    expected = 1/(4*math.pi*epsilon*r)
    torch.testing.assert_close(g, expected.to(g.dtype), rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(grad, -p*expected[:, None]/r[:, None]**2, rtol=1e-13, atol=1e-14)
    nn = (p[..., :, None]*p[..., None, :]/r[:, None, None]**2).to(eye.dtype)
    torch.testing.assert_close(hessian, (3*nn-eye)*expected[:, None, None]/r[:, None, None]**2,
                               rtol=1e-13, atol=1e-14)


def test_anisotropic_static_hessian_satisfies_pde_and_autograd():
    from witwin.maxwell.mom.dyadic_green import electrostatic_kernel
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps = torch.tensor([[2+.2j, .3+.01j, .1], [.3+.01j, 3+.1j, .2], [.1, .2, 1+.3j]], device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, eps, eye)
    p = torch.tensor([[.1, -.2, .3]], device='cuda', dtype=torch.float64, requires_grad=True)
    g, gradient, hessian = electrostatic_kernel(p, medium)
    real = torch.autograd.grad(g.real.sum(), p, create_graph=True, retain_graph=True)[0]
    imag = torch.autograd.grad(g.imag.sum(), p, create_graph=True, retain_graph=True)[0]
    torch.testing.assert_close(real+1j*imag, gradient, rtol=1e-12, atol=1e-13)
    rows = []
    for i in range(3):
        dr = torch.autograd.grad(real[0, i], p, retain_graph=True)[0]
        di = torch.autograd.grad(imag[0, i], p, retain_graph=True)[0]
        rows.append(dr+1j*di)
    torch.testing.assert_close(torch.stack(rows, 1), hessian, rtol=1e-12, atol=1e-13)
    contraction = torch.einsum('ij,nji->n', eps, hessian)
    torch.testing.assert_close(contraction, torch.zeros_like(contraction), rtol=0, atol=1e-12)


def test_static_plane_subtraction_matches_independent_scalar_coulomb_spectrum():
    from witwin.maxwell.mom.dyadic_green import electrostatic_plane_spectrum
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps, mu = -2+.1j, 1.3+.2j
    medium = TensorMedium(3e8, eps*eye, mu*eye)
    transverse = torch.tensor([[.3, -.2], [2., 1.]], device='cuda', dtype=torch.float64)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    rho = torch.linalg.vector_norm(transverse, dim=-1)
    q = k0*torch.cat((transverse.to(eye.dtype), 1j*rho[:, None]), -1)
    actual = electrostatic_plane_spectrum(transverse, .2, medium)
    expected = torch.zeros_like(actual)
    for offset, coefficient, vacuum in ((0, eps, EPSILON_0), (3, mu, MU_0)):
        potential = torch.exp(-k0*rho*.2)/(2*coefficient*k0*rho)
        hessian = -q[..., :, None]*q[..., None, :]*potential[:, None, None]
        expected[:, offset:offset+3, offset:offset+3] = 1j*hessian/(medium.omega*vacuum)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_plane_modes_reuse_eigenspaces_for_batched_queries(monkeypatch):
    from witwin.maxwell.mom.dyadic_green import compile_plane_modes
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (2+.8j)*eye, (1.3+.5j)*eye)
    transverse = torch.tensor([[.2, .3], [2.5, .4], [0., 0.]], device='cuda', dtype=torch.float64)
    modes = compile_plane_modes(transverse, medium)
    def forbidden(*args, **kwargs): raise AssertionError('A query must not repeat the eigenproblem')
    monkeypatch.setattr(torch.linalg, 'eig', forbidden)
    points = torch.tensor([[0., 0., .2], [.1, -.2, .3]], device='cuda', dtype=torch.float64)
    weights = torch.tensor([.3, .2, .5], device='cuda', dtype=torch.float64)
    actual = modes.integrate(points, weights, query_batch=1)
    for i, point in enumerate(points):
        spectrum = modes.spectrum(float(point[2]))
        phase = torch.exp(1j*(modes.transverse@point[:2]))
        expected = (spectrum*(phase*weights)[:, None, None]).sum(0)
        torch.testing.assert_close(actual[i], expected, rtol=1e-12, atol=1e-13)
    modes.residues.add_(1.)
    with pytest.raises(ValueError, match='changed'): modes.spectrum(.2)


def test_spatial_queries_share_modes_and_match_closed_form(monkeypatch):
    from witwin.maxwell.mom import dyadic_green as dg
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps, mu = 2+.8j, 1.3+.5j
    medium = TensorMedium(3e8, eps*eye, mu*eye)
    points = torch.tensor([[.01, -.02, .2], [.02, .01, .22], [-.03, .02, .24]], device='cuda', dtype=torch.float64)
    calls = []
    original = dg.compile_plane_modes
    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(dg, 'compile_plane_modes', counted)
    actual = dg.integrate_spatial_green(points, medium, radial_order=64, azimuthal_order=12)
    assert calls == [64*12]
    radius = torch.linalg.vector_norm(points, dim=-1)
    direction = (points/radius[:, None]).to(eye.dtype)
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*complex(eps)**.5*complex(mu)**.5
    g = torch.exp(1j*k*radius)/(4*math.pi*radius)
    first = g*(1j*k-1/radius)
    second = g*(-k*k-2j*k/radius+2/radius.square())
    nn = direction[..., :, None]*direction[..., None, :]
    hessian = second[:, None, None]*nn+(first/radius)[:, None, None]*(eye-nn)
    dyadic = g[:, None, None]*eye+hessian/(k*k)
    curl = dg.cross_matrix(first[:, None]*direction)
    ee = 1j*medium.omega*medium.mu[0, 0]*dyadic
    hh = 1j*medium.omega*medium.epsilon[0, 0]*dyadic
    expected = torch.cat((torch.cat((ee, -curl), -1), torch.cat((curl, hh), -1)), -2)
    torch.testing.assert_close(actual, expected, rtol=5e-7, atol=1e-9)


def test_isotropic_specialization_preserves_full_fields_and_near_source_remainder():
    from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps, mu = 2+.1j, 1.3+.2j
    medium = TensorMedium(3e8, eps*eye, mu*eye)
    kernel = DyadicGreenKernel(medium)
    points = torch.tensor([[.01, .02, .03], [.2, .1, -.3]], device='cuda', dtype=torch.float64)
    actual, receipt = kernel.evaluate(points)
    r = torch.linalg.vector_norm(points, dim=-1)
    n = (points/r[:, None]).to(eye.dtype)
    nn = n[..., :, None]*n[..., None, :]
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*eps**.5*mu**.5
    g = torch.exp(1j*k*r)/(4*math.pi*r)
    gp = g*(1j*k-1/r)
    gpp = g*(-k*k-2j*k/r+2/r**2)
    hessian = gpp[:, None, None]*nn+(gp/r)[:, None, None]*(eye-nn)
    d = g[:, None, None]*eye+hessian/k**2
    ee, hh = 1j*medium.omega*MU_0*mu*d, 1j*medium.omega*EPSILON_0*eps*d
    curl = cross_matrix(gp[:, None]*n)
    expected = torch.cat((torch.cat((ee, -curl), -1), torch.cat((curl, hh), -1)), -2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert receipt['kernel_version'] == 'analytic_isotropic_dyadic_v1'
    tiny = torch.tensor([[0., 0., 1e-11]], device='cuda', dtype=torch.float64)
    regular, receipt = kernel.evaluate_regular(tiny)
    leading = 1j*medium.omega*MU_0*mu/(8*math.pi)*(eye+torch.diag(eye.new_tensor([0, 0, 1])))
    torch.testing.assert_close(tiny[0, 2]*regular[0, :3, :3], leading, rtol=1e-8, atol=1e-12)
    assert receipt['static_diagonal_restored'] is False


def test_anisotropic_spatial_kernel_matches_transformed_isotropic_solution():
    """A real coordinate stretch gives an independent anisotropic Green oracle.

    For r=A r', epsilon=mu=s A A^T/det(A), physical current Green
    tensors equal blockdiag(A^-T) G_s(r') blockdiag(A^-1).
    This checks the general spectral path, including off-diagonal tensors.
    """
    from witwin.maxwell.mom.dyadic_green import integrate_spatial_green, isotropic_dyadic_green
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    rotation = eye.new_tensor([[.8, 0, .6], [0, 1, 0], [-.6, 0, .8]])
    stretch = rotation@torch.diag(eye.new_tensor([.8, 1.2, 1.]))@rotation.T
    inverse = torch.linalg.inv(stretch)
    constitutive = stretch@stretch.T/torch.linalg.det(stretch)
    scalar = 1+.8j
    medium = TensorMedium(3e8, scalar*constitutive, scalar*constitutive)
    reference_medium = TensorMedium(3e8, scalar*eye, scalar*eye)
    points = torch.tensor([[.01, -.02, .2], [.02, .01, .22]], device='cuda', dtype=torch.float64)
    transform = torch.block_diag(inverse, inverse)
    reference, _ = isotropic_dyadic_green(points@inverse.real.T, reference_medium)
    expected = transform.T@reference@transform
    actual = integrate_spatial_green(points, medium, radial_order=96, azimuthal_order=32)
    # Balance E/H units before comparing all 36 tensor entries together.
    eta0 = math.sqrt(MU_0/EPSILON_0)
    balance = torch.diag(eye.new_tensor([1, 1, 1, eta0, eta0, eta0]))
    error = torch.linalg.matrix_norm(balance@(actual-expected)@balance)/torch.linalg.matrix_norm(balance@expected@balance)
    assert bool((error < 2e-6).all()), error


def test_spectral_constitutive_contrast_preserves_zero_and_weak_signal():
    from witwin.maxwell.mom.dyadic_green import current_green_spectrum, current_green_spectrum_difference
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps, mu = 2+.2j, 1.3+.1j
    exterior = TensorMedium(3e8, eps*eye, mu*eye)
    q = eye.new_tensor([[.3, .4, .5], [.7, -.2, .8]])
    zero = current_green_spectrum_difference(q, exterior, exterior)
    assert torch.count_nonzero(zero) == 0
    for contrast in (.2, 1e-12):
        interior = TensorMedium(3e8, (eps+contrast)*eye, mu*eye)
        actual = current_green_spectrum_difference(q, exterior, interior)
        if contrast == .2:
            expected = current_green_spectrum(q, interior)-current_green_spectrum(q, exterior)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        else:
            # Independent longitudinal eigenvalue: G_EJ=-i/(omega*epsilon),
            # so its finite contrast is i*delta/(omega*epsilon0*eps_i*eps_e).
            direction = q/torch.sqrt((q*q).sum(-1, keepdim=True))
            longitudinal = torch.einsum('ni,nij,nj->n', direction, actual[:, :3, :3], direction)
            sampled_delta = interior.epsilon_r[0, 0]-exterior.epsilon_r[0, 0]
            expected = 1j*sampled_delta/(exterior.omega*EPSILON_0*interior.epsilon_r[0, 0]*eps)
            torch.testing.assert_close(longitudinal, expected.expand_as(longitudinal), rtol=1e-12, atol=0)
    parameter = torch.tensor(0., device='cuda', dtype=torch.float64, requires_grad=True)
    interior = TensorMedium(3e8, (eps+parameter)*eye, mu*eye)
    zero_with_gradient = current_green_spectrum_difference(q, exterior, interior)
    derivative, = torch.autograd.grad(zero_with_gradient[:, :3, :3].imag.sum(), parameter)
    assert torch.isfinite(derivative) and derivative.abs() > 0
