"""Explicit scaled-forcing projection at requested 3D Maxwell spectral points."""
import math
from dataclasses import replace
import runpy
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel
from witwin.maxwell.mom.dyadic_green import maxwell_symbol

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def resonant():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    k0 = 2*math.pi*3e8*math.sqrt(EPSILON_0*MU_0)
    lattice = LatticeSpec(eye.real*(2*math.pi/k0), eye.real.new_zeros(3))
    return QuasiperiodicDyadicGreenKernel(lattice, TensorMedium(3e8, eye, eye))


def test_resonant_projection_matches_independent_axial_minimum_norm_response():
    kernel = resonant()
    orders = torch.tensor([[1, 0, 0]], device='cuda')
    with pytest.raises(ValueError, match='spectral pole'):
        kernel.spectral_modes(orders)
    result = kernel.projected_modes(orders, rank_relative_tolerance=1e-10)
    assert result['retained_rank'].tolist() == [4]
    matrix = result['projected_current_to_field'][0]
    inverse = matrix.new_zeros(6, 6)
    inverse[0, 0], inverse[3, 3] = 1., -1.
    inverse[1, 1] = inverse[1, 5] = .25
    inverse[5, 1] = inverse[5, 5] = -.25
    inverse[2, 2] = inverse[4, 2] = .25
    inverse[2, 4] = inverse[4, 4] = -.25
    eta = math.sqrt(MU_0/EPSILON_0)
    k0 = kernel.medium.omega*math.sqrt(EPSILON_0*MU_0)
    source_scale = matrix.new_tensor([-1j*eta/k0]*3+[1j/k0]*3)
    field_scale = matrix.new_tensor([1.]*3+[1/eta]*3)
    expected = field_scale[:, None]*inverse*source_scale[None, :]/kernel.lattice.cell_measure
    torch.testing.assert_close(matrix, expected, rtol=1e-12, atol=1e-12)
    symbol = maxwell_symbol(result['wavevectors'].to(matrix.dtype), kernel.medium.epsilon_r,
        kernel.medium.mu_r, kernel.medium.frequency)[0]
    left, right = result['removed_forcing_projector'][0], result['homogeneous_field_projector'][0]
    eye = torch.eye(6, device=matrix.device, dtype=matrix.dtype)
    torch.testing.assert_close(symbol@inverse, eye-left, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(inverse@symbol, eye-right, rtol=1e-12, atol=1e-12)
    for projector in (left, right):
        torch.testing.assert_close(projector.mH, projector, rtol=0, atol=1e-12)
        torch.testing.assert_close(projector@projector, projector, rtol=1e-12, atol=1e-12)
    # A forcing in the removed subspace cannot become an ordinary forced solution.
    forcing = left[:, 1]
    assert forcing.norm() > .1
    torch.testing.assert_close(inverse@forcing, torch.zeros_like(forcing), rtol=0, atol=1e-12)
    assert not result['unprojected_forced_solution'] and not result['limiting_absorption_verified']
    assert not result['exact_nullspace_certified'] and not result['physical_qualification']
    # Source-vector differentiation remains native, although material rank derivatives reject.
    currents = matrix.new_ones(6, requires_grad=True)
    gradient, = torch.autograd.grad((matrix@currents).abs().square().sum(), currents)
    torch.testing.assert_close(gradient, 2*matrix.mH@(matrix@currents.detach()), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('reciprocal', (True, False))
def test_off_pole_projected_map_agrees_with_full_tensor_resolvent(reciprocal):
    kernel = runpy.run_path('tests/mom/test_periodic_dyadic_nonreciprocal.py')['problem'](3)
    if reciprocal:
        kernel = replace(kernel, medium=TensorMedium(kernel.medium.frequency,
            (kernel.medium.epsilon_r+kernel.medium.epsilon_r.T)/2,
            (kernel.medium.mu_r+kernel.medium.mu_r.T)/2))
    orders = torch.tensor([[0, 0, 0], [1, -1, 2]], device='cuda')
    result = kernel.projected_modes(orders, rank_relative_tolerance=1e-10)
    expected = kernel.spectral_modes(orders)
    torch.testing.assert_close(result['projected_current_to_field'], expected['current_to_field'],
                               rtol=2e-12, atol=1e-12)
    assert result['retained_rank'].tolist() == [6, 6]
    assert torch.count_nonzero(result['removed_forcing_projector']) == 0
    assert torch.count_nonzero(result['homogeneous_field_projector']) == 0


def test_projection_policy_input_guards_and_empty_orders():
    kernel = resonant()
    orders = torch.empty(0, 3, device='cuda', dtype=torch.int64)
    result = kernel.projected_modes(orders, rank_relative_tolerance=1e-10)
    assert result['projected_current_to_field'].shape == (0, 6, 6)
    assert result['retained_rank'].dtype == torch.int64
    for tolerance in (0., True, 1., float('nan')):
        with pytest.raises(ValueError, match='Rank tolerance'):
            kernel.projected_modes(orders, rank_relative_tolerance=tolerance)
    with pytest.raises(ValueError, match='unique'):
        kernel.projected_modes(torch.zeros(2, 3, device='cuda', dtype=torch.int64), rank_relative_tolerance=1e-10)
    epsilon = kernel.medium.epsilon_r.clone().requires_grad_()
    differentiable = replace(kernel, medium=TensorMedium(kernel.medium.frequency, epsilon, kernel.medium.mu_r))
    with pytest.raises(NotImplementedError, match='rank-changing'):
        differentiable.projected_modes(orders, rank_relative_tolerance=1e-10)
    kernel.medium.epsilon_r[0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        kernel.projected_modes(orders, rank_relative_tolerance=1e-10)


def test_rank_deficient_gyromagnetic_projection_matches_circular_channels():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon, diagonal, bias = 2., 1.4, .2
    gyro = eye.new_tensor([[0., -1j, 0.], [1j, 0., 0.], [0., 0., 0.]])
    medium = TensorMedium(3e8, epsilon*eye, diagonal*eye+bias*gyro, reciprocal=False)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    axial = math.sqrt(epsilon*(diagonal+bias))
    lattice = LatticeSpec((2*math.pi/(k0*axial))*eye.real, eye.real.new_zeros(3))
    kernel = QuasiperiodicDyadicGreenKernel(lattice, medium)
    orders = torch.tensor([[0, 0, 1]], device='cuda')
    with pytest.raises(ValueError, match='spectral pole'):
        kernel.spectral_modes(orders)
    result = kernel.projected_modes(orders, rank_relative_tolerance=1e-10)
    assert result['retained_rank'].tolist() == [5]
    expected_inverse = eye.new_zeros(6, 6)
    for handedness in (1, -1):
        p = eye.new_tensor([1., handedness*1j, 0.])/math.sqrt(2)
        channel = eye.new_tensor([[epsilon, -handedness*1j*axial],
                                 [-handedness*1j*axial, -(diagonal+handedness*bias)]])
        if handedness == 1:
            # Independent rank-one identity B+ = B^H / ||B||_F^2.
            inverse = channel.mH/channel.abs().square().sum()
        else:
            determinant = axial**2-epsilon*(diagonal+handedness*bias)
            inverse = eye.new_tensor([[-(diagonal+handedness*bias), handedness*1j*axial],
                                      [handedness*1j*axial, epsilon]])/determinant
        for row in range(2):
            for col in range(2):
                expected_inverse[3*row:3*row+3, 3*col:3*col+3] += inverse[row, col]*p[:, None]*p.conj()[None, :]
    expected_inverse[2, 2], expected_inverse[5, 5] = 1/epsilon, -1/diagonal
    eta = math.sqrt(MU_0/EPSILON_0)
    source_scale = eye.new_tensor([-1j*eta/k0]*3+[1j/k0]*3)
    field_scale = eye.new_tensor([1.]*3+[1/eta]*3)
    expected = field_scale[:, None]*expected_inverse*source_scale[None, :]/lattice.cell_measure
    torch.testing.assert_close(result['projected_current_to_field'][0], expected, rtol=1e-11, atol=1e-11)
    symbol = maxwell_symbol(result['wavevectors'].to(eye.dtype), medium.epsilon_r, medium.mu_r, medium.frequency)[0]
    identity = torch.eye(6, device=eye.device, dtype=eye.dtype)
    left, right = result['removed_forcing_projector'][0], result['homogeneous_field_projector'][0]
    torch.testing.assert_close(left, identity-symbol@expected_inverse, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(right, identity-expected_inverse@symbol, rtol=1e-11, atol=1e-11)
    assert torch.linalg.matrix_norm(left-right) > .1
