"""Three-dimensional outgoing layer sums against independent scalar Ewald."""
from dataclasses import replace
import math
import runpy
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import cross_matrix
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('loss', (0., .2))
def test_stacked_scalar_limit_matches_independent_3d_ewald_hessian(loss):
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    lattice = LatticeSpec(eye.real.new_tensor([[.7, 0., 0.], [.1, .8, 0.], [.05, .1, -.9]]),
                          eye.real.new_tensor([.2, .1, .05]))
    eps, mu = 1+1j*loss, 1+.5j*loss
    medium = TensorMedium(3e8, eps*eye, mu*eye)
    kernel = QuasiperiodicDyadicGreenKernel(lattice, medium, mode_batch=1024)
    points = eye.real.new_tensor([[.13, .07, .4], [.1, -.12, -.35]], requires_grad=True)
    value, receipt = kernel.evaluate_stacked(points, shells=12)
    scalar = EwaldHelmholtzKernel(lattice, 3.5, real_shells=3, reciprocal_shells=5,
                                quadrature_order=96, image_batch=256)
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*eps**.5*mu**.5
    green, gradient = scalar.evaluate(points, k)
    rows = []
    for axis in range(3):
        real, = torch.autograd.grad(gradient[:, axis].real.sum(), points, retain_graph=True)
        imag, = torch.autograd.grad(gradient[:, axis].imag.sum(), points, retain_graph=True)
        rows.append(real+1j*imag)
    potential = green[:, None, None]*eye+torch.stack(rows, 1)/k**2
    curl = cross_matrix(gradient)
    expected = torch.cat((torch.cat((1j*medium.omega*MU_0*mu*potential, -curl), -1),
        torch.cat((curl, 1j*medium.omega*EPSILON_0*eps*potential), -1)), -2)
    torch.testing.assert_close(value, expected, rtol=2e-7, atol=1e-7)
    assert not receipt['heat_filter_applied'] and not receipt['physical_qualification']


def test_stacked_full_tensor_bloch_transpose_identity_and_native_bias_gradient():
    problem = runpy.run_path('tests/mom/test_periodic_dyadic_nonreciprocal.py')['problem']
    kernel = problem(3)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .4], [.08, -.03, -.35]])
    value, _ = kernel.evaluate_stacked(points, shells=2)
    for row in kernel.lattice.vectors:
        actual, _ = kernel.evaluate_stacked(points+row, shells=2)
        torch.testing.assert_close(actual, value*torch.exp(1j*(row@kernel.lattice.bloch)), rtol=1e-10, atol=1e-10)
    opposite = problem(3, bias=-.2)
    opposite = replace(opposite, lattice=LatticeSpec(kernel.lattice.vectors, -kernel.lattice.bloch))
    transposed, _ = opposite.evaluate_stacked(-points, shells=2)
    signs = points.new_tensor([1., 1., 1., -1., -1., -1.])
    torch.testing.assert_close(value, signs[:, None]*transposed.mT*signs[None, :], rtol=1e-9, atol=1e-9)
    bias = points.new_tensor(.2, requires_grad=True)
    def objective(b):
        return problem(3, bias=b).evaluate_stacked(points[:1], shells=1)[0].real.sum()
    derivative, = torch.autograd.grad(objective(bias), bias)
    step = 1e-5
    expected = (objective(bias.detach()+step)-objective(bias.detach()-step))/(2*step)
    torch.testing.assert_close(derivative, expected, rtol=2e-5, atol=2e-5)


def test_stacked_source_plane_and_resonant_geometric_denominator_reject():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    k0 = 2*math.pi*3e8*math.sqrt(EPSILON_0*MU_0)
    lattice = LatticeSpec(eye.real*(2*math.pi/k0), eye.real.new_zeros(3))
    kernel = QuasiperiodicDyadicGreenKernel(lattice, TensorMedium(3e8, eye, eye))
    with pytest.raises(ValueError, match='spectral pole'):
        kernel.evaluate_stacked(eye.real.new_tensor([[.1, .1, .4]]), shells=0)
    for point in (eye.real.new_zeros(1, 3), 100*lattice.vectors[2:3]):
        with pytest.raises(ValueError, match='source-plane'):
            kernel.evaluate_stacked(point, shells=1)
    with pytest.raises(ValueError, match='nonnegative'):
        kernel.evaluate_stacked(eye.real.new_tensor([[.1, .1, .4]]), shells=-1)


def test_admitted_thin_skew_lattice_avoids_gram_conditioning_loss():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    sample = eye.real
    medium = TensorMedium(3e8, (1+1j)*eye, (1+1j)*eye)
    skew = LatticeSpec(sample.new_tensor([[1., 0., 0.], [1., 1e-8, 0.], [0., 0., 1.]]), sample.new_zeros(3))
    orthogonal = LatticeSpec(sample.new_tensor([[1., 0., 0.], [0., 1e-8, 0.], [0., 0., 1.]]), sample.new_zeros(3))
    points = sample.new_tensor([[.1, 1e-9, .4]])
    # The zero transverse order has the same area and wavevector in both cells.
    value, _ = QuasiperiodicDyadicGreenKernel(skew, medium).evaluate_stacked(points, shells=0)
    expected, _ = QuasiperiodicDyadicGreenKernel(orthogonal, medium).evaluate_stacked(points, shells=0)
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-10)


def test_stacked_transverse_cutoff_rejects_coarse_budget_and_accepts_refinement():
    from witwin.maxwell.mom.periodic_dyadic import PeriodicDyadicConvergenceError
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    lattice = LatticeSpec(eye.real.new_tensor([[.7, 0., 0.], [.1, .8, 0.], [.05, .1, .9]]),
                          eye.real.new_tensor([.2, .1, .05]))
    kernel = QuasiperiodicDyadicGreenKernel(lattice, TensorMedium(3e8, (1+.2j)*eye, (1+.1j)*eye), mode_batch=512)
    points = eye.real.new_tensor([[.13, .07, .4], [.1, -.12, -.35]])
    with pytest.raises(PeriodicDyadicConvergenceError) as error:
        kernel.estimate_stacked(points, shells=0, relative_tolerance=0., absolute_tolerance=1e-5)
    assert error.value.receipt['failed_axis'] == 'transverse_reciprocal_tail'
    assert error.value.receipt['compared_shells'] == [0, 1]
    value, receipt = kernel.estimate_stacked(points, shells=8, relative_tolerance=0., absolute_tolerance=1e-5)
    expected, _ = kernel.evaluate_stacked(points, shells=8)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert receipt['compared_shells'] == [8, 16] and receipt['cutoff_converged']
    assert not receipt['certified_error_bound'] and not receipt['limiting_absorption_verified']
    assert not receipt['global_resolvent_verified'] and not receipt['physical_qualification']
    with pytest.raises(ValueError, match='finite'):
        kernel.estimate_stacked(points, shells=8, absolute_tolerance=float('nan'))
