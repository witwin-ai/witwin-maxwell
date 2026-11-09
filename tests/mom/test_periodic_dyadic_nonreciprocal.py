"""Finite passive nonreciprocal Bloch maps and transposed-medium identities."""
from dataclasses import replace
import math
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel, LossyDyadicImageReference
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(dimension, bias=.2, loss=.3):
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    gyro = eye.new_tensor([[0., 1j, -.3j], [-1j, 0., .2j], [.3j, -.2j, 0.]])
    epsilon = eye.new_tensor([[2., .1, .05], [.1, 2.6, .08], [.05, .08, 3.1]]) + 1j*loss*eye + .4*bias*gyro
    mu = eye.new_tensor([[1.1, .03, 0.], [.03, 1.4, .04], [0., .04, 1.7]]) + 1j*loss*eye + bias*gyro
    vectors = eye.real.new_tensor([[.7, 0., 0.], [.1, .8, 0.], [.05, .1, .9]])[:dimension]
    bloch = vectors.new_tensor([.2, .1 if dimension > 1 else 0., .05 if dimension == 3 else 0.])
    return QuasiperiodicDyadicGreenKernel(LatticeSpec(vectors, bloch),
        TensorMedium(3e8, epsilon, mu, reciprocal=False), mode_batch=64)


def evaluate(kernel, points, normal=None):
    if kernel.lattice.dimension == 3:
        return kernel.evaluate_heat(points, shells=1, heat_time=.04)
    if kernel.lattice.dimension == 2:
        return kernel.evaluate_open(points, shells=1)
    if normal is None:
        normal = points.new_tensor([0., 0., 1.])
    return kernel.evaluate_line(points, normal=normal, shells=1,
        transverse_cutoff=16., quadrature_order=24, transverse_map='sinh')


@pytest.mark.parametrize('dimension', (1, 2, 3))
def test_transposed_medium_bloch_identity_and_proper_rotation(dimension):
    kernel = problem(dimension)
    sample = kernel.lattice.vectors
    points = sample.new_tensor([[.12, .07, .4], [.08, -.03, -.35]])
    value, receipt = evaluate(kernel, points)
    reversed_bias = problem(dimension, bias=-.2)
    opposite = replace(reversed_bias, lattice=LatticeSpec(sample, -kernel.lattice.bloch))
    transposed, _ = evaluate(opposite, -points)
    signs = sample.new_tensor([1., 1., 1., -1., -1., -1.])
    torch.testing.assert_close(value, signs[:, None]*transposed.mT*signs[None, :], rtol=2e-8, atol=2e-8)
    same_bias, _ = evaluate(replace(kernel, lattice=opposite.lattice), -points)
    assert torch.linalg.matrix_norm(value-signs[:, None]*same_bias.mT*signs[None, :]).max() > 1e-5
    translated, _ = evaluate(kernel, points+sample[0])
    torch.testing.assert_close(translated, value*torch.exp(1j*(sample[0]@kernel.lattice.bloch)),
                               rtol=2e-9, atol=2e-9)
    angle = .4
    rotation = sample.new_tensor([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                                 [-math.sin(angle), 0., math.cos(angle)]])
    r = rotation.to(value.dtype)
    rotated = replace(kernel, lattice=LatticeSpec(sample@rotation.T, rotation@kernel.lattice.bloch),
        medium=TensorMedium(kernel.medium.frequency, r@kernel.medium.epsilon_r@r.T,
            r@kernel.medium.mu_r@r.T, reciprocal=False))
    actual, _ = evaluate(rotated, points@rotation.T, rotation@sample.new_tensor([0., 0., 1.]))
    transform = torch.block_diag(r, r)
    torch.testing.assert_close(actual, transform@value@transform.T, rtol=2e-8, atol=2e-8)
    assert not receipt['physical_qualification']


def test_nonreciprocal_modal_maxwell_equations_adjoint_and_bias_gradient():
    bias = torch.tensor(.2, device='cuda', dtype=torch.float64, requires_grad=True)
    kernel = problem(3, bias)
    orders = torch.tensor([[0, 0, 0], [1, -1, 0]], device='cuda')
    torch.manual_seed(71)
    source = torch.randn(2, 6, device='cuda', dtype=torch.complex128, requires_grad=True)
    covector = torch.randn_like(source)
    fields = kernel.apply_modes(orders, source)
    modes = kernel.spectral_modes(orders)
    q = modes['wavevectors'].to(fields.dtype)
    e, h = fields[:, :3], fields[:, 3:]
    medium = kernel.medium
    torch.testing.assert_close(1j*torch.linalg.cross(q, e)-1j*medium.omega*(h@medium.mu.T),
        -source[:, 3:]/kernel.lattice.cell_measure, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(1j*torch.linalg.cross(q, h)+1j*medium.omega*(e@medium.epsilon.T),
        source[:, :3]/kernel.lattice.cell_measure, rtol=1e-10, atol=1e-10)
    objective = torch.vdot(covector.flatten(), fields.flatten()).real
    gradient, derivative = torch.autograd.grad(objective, (source, bias))
    adjoint = kernel.apply_modes(orders, covector, adjoint=True)
    torch.testing.assert_close(gradient, adjoint, rtol=1e-11, atol=1e-11)
    def scalar(b):
        value = problem(3, b).apply_modes(orders, source.detach())
        return torch.vdot(covector.flatten(), value.flatten()).real
    step = 1e-5
    numerical = (scalar(bias.detach()+step)-scalar(bias.detach()-step))/(2*step)
    torch.testing.assert_close(derivative, numerical, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('dimension', (1, 2))
def test_open_nonreciprocal_native_bias_gradient(dimension):
    bias = torch.tensor(.2, device='cuda', dtype=torch.float64, requires_grad=True)
    points = bias.new_tensor([[.12, .07, .4]])
    def objective(b):
        return evaluate(problem(dimension, b), points)[0].real.sum()
    derivative, = torch.autograd.grad(objective(bias), bias)
    step = 1e-5
    numerical = (objective(bias.detach()+step)-objective(bias.detach()-step))/(2*step)
    torch.testing.assert_close(derivative, numerical, rtol=2e-5, atol=2e-5)


def test_image_reference_preserves_nonreciprocal_material_and_loss_guard(monkeypatch):
    kernel = problem(1)
    received = []
    # Routing control only: actual spatial accuracy is a separate acceptance.
    def spatial(self, points):
        assert not self.medium.reciprocal
        torch.testing.assert_close(self.medium.epsilon_r, kernel.medium.epsilon_r)
        torch.testing.assert_close(self.medium.mu_r, kernel.medium.mu_r)
        received.append(points.clone())
        value = self.medium.epsilon_r.new_zeros(len(points), 6, 6)
        value[:, :3, :3] = self.medium.epsilon_r
        return value, {'converged': True, 'manufactured_routing_control': True}
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', spatial)
    reference = LossyDyadicImageReference(kernel.lattice, kernel.medium, 1)
    value, receipt = reference.evaluate(kernel.lattice.vectors.new_zeros(1, 3), exclude_primary=True)
    assert sum(len(p) for p in received) == 2
    angle = kernel.lattice.vectors[0]@kernel.lattice.bloch
    torch.testing.assert_close(value[0, :3, :3], 2*torch.cos(angle)*kernel.medium.epsilon_r)
    assert receipt['primary_image_excluded'] and not receipt['physical_qualification']
    lossless = problem(1, loss=0.)
    with pytest.raises(ValueError, match='strictly positive'):
        LossyDyadicImageReference(lossless.lattice, lossless.medium, 1)


def test_axial_circular_channels_match_independent_gyromagnetic_response():
    from witwin.maxwell.constants import EPSILON_0, MU_0
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon, diagonal, bias = 2.+.4j, 1.4+.3j, .2
    gyro = eye.new_tensor([[0., -1j, 0.], [1j, 0., 0.], [0., 0., 0.]])
    lattice = LatticeSpec(1.1*eye.real, eye.real.new_tensor([0., 0., .2]))
    medium = TensorMedium(3e8, epsilon*eye, diagonal*eye+bias*gyro, reciprocal=False)
    kernel = QuasiperiodicDyadicGreenKernel(lattice, medium)
    reverse = replace(kernel, medium=TensorMedium(3e8, epsilon*eye,
        diagonal*eye-bias*gyro, reciprocal=False))
    orders = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, -1]], device='cuda')
    q = (orders.to(eye.real.dtype)@lattice.reciprocal+lattice.bloch)[:, 2]
    amplitudes = []
    for handedness in (-1, 1):
        polarization = eye.new_tensor([1., handedness*1j, 0.])/math.sqrt(2)
        # For axial propagation, mu acts on this channel by diagonal+s*bias.
        mu_channel = diagonal+handedness*bias
        denominator = q.square()-medium.omega**2*EPSILON_0*MU_0*epsilon*mu_channel
        electric_amplitude = 1j*medium.omega*MU_0*mu_channel/denominator/lattice.cell_measure
        amplitudes.append(electric_amplitude)
        j = polarization.expand(len(orders), -1)
        zero = torch.zeros_like(j)
        electric_source = torch.cat((j, zero), -1)
        actual = kernel.apply_modes(orders, electric_source)
        expected = torch.cat((electric_amplitude[:, None]*j,
            (handedness*q/denominator/lattice.cell_measure)[:, None]*j), -1)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        magnetic_source = torch.cat((zero, j), -1)
        expected = torch.cat(((-handedness*q/denominator/lattice.cell_measure)[:, None]*j,
            (1j*medium.omega*EPSILON_0*epsilon/denominator/lattice.cell_measure)[:, None]*j), -1)
        torch.testing.assert_close(kernel.apply_modes(orders, magnetic_source), expected,
                                   rtol=1e-11, atol=1e-11)
        opposite = polarization.conj().expand(len(orders), -1)
        reversed_field = reverse.apply_modes(orders, torch.cat((opposite, zero), -1))
        amplitude = torch.einsum('ni,ni->n', opposite.conj(), reversed_field[:, :3])
        torch.testing.assert_close(amplitude, electric_amplitude, rtol=1e-11, atol=1e-11)
        # Real gyrotropic bias is Hermitian; it must not contribute absorption.
        expected_loss = .5*medium.omega*(EPSILON_0*epsilon.imag+MU_0*diagonal.imag)
        loss = medium.dissipated_power_density(j, j)
        torch.testing.assert_close(loss, torch.full_like(loss, expected_loss), rtol=1e-12, atol=1e-12)
    assert (amplitudes[0]-amplitudes[1]).abs().max() > 1e-5


@pytest.mark.parametrize('dimension', (1, 2))
def test_actual_nonreciprocal_spatial_kernel_matches_independent_images(dimension):
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic import _cutoff_check
    kernel = replace(problem(dimension, loss=2.), mode_batch=512)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .4], [.08, -.03, -.35]])
    tolerance = .002
    if dimension == 1:
        value, spectral_receipt = kernel.estimate_line(points,
            normal=points.new_tensor([0., 0., 1.]), shells=8, transverse_cutoff=64.,
            quadrature_order=64, transverse_map='sinh',
            relative_tolerance=0., absolute_tolerance=tolerance/8)
    else:
        value, spectral_receipt = kernel.estimate_open(points, shells=8,
            relative_tolerance=0., absolute_tolerance=tolerance/8)
    config = DyadicIntegrationConfig(8, 8, 1024, 128,
        relative_tolerance=0., absolute_tolerance=tolerance/256)
    primary, primary_receipt = DyadicGreenKernel(kernel.medium, config).evaluate(points)
    regular, image_receipt = LossyDyadicImageReference(kernel.lattice, kernel.medium,
        1, integration=config).estimate_regular(points,
            relative_tolerance=0., absolute_tolerance=tolerance/8)
    expected = primary+regular
    check = _cutoff_check(expected, value, .005, 1e-5)
    assert check['passed'], check
    assert spectral_receipt['cutoff_converged'] and primary_receipt['converged']
    assert image_receipt['image_cutoff_converged']
    assert not spectral_receipt['physical_qualification']
    print({'dimension': dimension, 'spatial_comparison': check,
           'spectral_receipt': spectral_receipt, 'image_tail': image_receipt['cutoff_check']})
