"""One-dimensional tensor Bloch residues, physical normalization and cutoff guards."""
import math
from dataclasses import replace
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel, PeriodicDyadicConvergenceError
from witwin.maxwell.mom.dyadic_green import cross_matrix
from test_periodic_dyadic import problem

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def line_problem(parameter=None):
    base = problem(parameter)
    vectors = base.lattice.vectors[:1]
    return QuasiperiodicDyadicGreenKernel(LatticeSpec(vectors, .2*vectors[0]), base.medium)


def test_line_scalar_limit_matches_independent_ewald_hessian():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    vectors = eye.real.new_tensor([[.7, 0., 0.]])
    lattice = LatticeSpec(vectors, vectors.new_tensor([.2, 0., 0.]))
    epsilon, mu = 1+.4j, 1+.3j
    medium = TensorMedium(3e8, epsilon*eye, mu*eye)
    kernel = QuasiperiodicDyadicGreenKernel(lattice, medium)
    points = vectors.new_tensor([[.13, .07, .45], [.1, -.12, -.5]], requires_grad=True)
    normal = vectors.new_tensor([0., 0., 1.])
    actual, receipt = kernel.evaluate_line(points, normal=normal, shells=8,
        transverse_cutoff=64., quadrature_order=384)
    scalar = EwaldHelmholtzKernel(lattice, 3.5, real_shells=4,
        reciprocal_shells=10, quadrature_order=128)
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*epsilon**.5*mu**.5
    green, gradient = scalar.evaluate(points, k)
    rows = []
    for axis in range(3):
        real, = torch.autograd.grad(gradient[:, axis].real.sum(), points, retain_graph=True)
        imag, = torch.autograd.grad(gradient[:, axis].imag.sum(), points, retain_graph=True)
        rows.append(real+1j*imag)
    potential = green[:, None, None]*eye+torch.stack(rows, 1)/k**2
    curl = cross_matrix(gradient)
    expected = torch.cat((torch.cat((1j*medium.omega*MU_0*mu*potential, -curl), -1),
        torch.cat((curl, 1j*medium.omega*EPSILON_0*epsilon*potential), -1)), -2)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-7)
    assert not receipt['physical_qualification'] and not receipt['cutoff_converged']
    assert not receipt['primary_singularity_separated']


def test_line_anisotropic_maxwell_bloch_reciprocity_and_rotation():
    kernel = line_problem()
    sample = kernel.lattice.vectors
    normal = sample.new_tensor([0., 0., 1.])
    points = sample.new_tensor([[.12, .07, .45]], requires_grad=True)
    settings = dict(normal=normal, shells=2, transverse_cutoff=20., quadrature_order=32)
    value, _ = kernel.evaluate_line(points, **settings)
    def curl(field):
        rows = []
        for axis in range(3):
            real, = torch.autograd.grad(field[:, axis].real.sum(), points, retain_graph=True)
            imag, = torch.autograd.grad(field[:, axis].imag.sum(), points, retain_graph=True)
            rows.append(real+1j*imag)
        derivative = torch.stack(rows, 1)
        return torch.stack((derivative[:, 2, 1]-derivative[:, 1, 2],
            derivative[:, 0, 2]-derivative[:, 2, 0], derivative[:, 1, 0]-derivative[:, 0, 1]), -1)
    for column in (0, 3):
        e, h = value[:, :3, column], value[:, 3:, column]
        torch.testing.assert_close(curl(e), 1j*kernel.medium.omega*(h@kernel.medium.mu.T), rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(curl(h), -1j*kernel.medium.omega*(e@kernel.medium.epsilon.T), rtol=1e-8, atol=1e-8)
    shifted, _ = kernel.evaluate_line(points.detach()+sample[0], **settings)
    torch.testing.assert_close(shifted, value*torch.exp(1j*(sample[0]@kernel.lattice.bloch)), rtol=1e-10, atol=1e-10)
    opposite = replace(kernel, lattice=LatticeSpec(sample, -kernel.lattice.bloch))
    reciprocal, _ = opposite.evaluate_line(-points.detach(), **settings)
    signs = sample.new_tensor([1, 1, 1, -1, -1, -1])
    torch.testing.assert_close(value, signs[:, None]*reciprocal.transpose(-1, -2)*signs[None, :], rtol=1e-9, atol=1e-9)
    angle = .4
    rotation = sample.new_tensor([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                                  [-math.sin(angle), 0., math.cos(angle)]])
    r = rotation.to(value.dtype)
    medium = TensorMedium(kernel.medium.frequency, r@kernel.medium.epsilon_r@r.T,
                           r@kernel.medium.mu_r@r.T)
    rotated = QuasiperiodicDyadicGreenKernel(LatticeSpec(sample@rotation.T,
        rotation@kernel.lattice.bloch), medium)
    actual, _ = rotated.evaluate_line(points.detach()@rotation.T,
        **{**settings, 'normal': rotation@normal})
    transform = torch.block_diag(r, r)
    torch.testing.assert_close(actual, transform@value@transform.T, rtol=1e-9, atol=1e-9)


def test_line_native_material_gradient_matches_centered_difference():
    parameter = torch.tensor(.1, device='cuda', dtype=torch.float64, requires_grad=True)
    kernel = line_problem(parameter)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .5]])
    settings = dict(normal=points.new_tensor([0., 0., 1.]), shells=1,
                    transverse_cutoff=16., quadrature_order=24)
    def objective(p):
        value, _ = line_problem(p).evaluate_line(points, **settings)
        return value.real.sum()
    derivative, = torch.autograd.grad(objective(parameter), parameter)
    step = 1e-5
    reference = (objective(parameter.detach()+step)-objective(parameter.detach()-step))/(2*step)
    torch.testing.assert_close(derivative, reference, rtol=1e-6, atol=1e-6)


def test_line_guards_preserve_source_plane_and_material_ownership():
    kernel = line_problem()
    sample = kernel.lattice.vectors
    points = sample.new_tensor([[.1, .2, .3]])
    settings = dict(normal=sample.new_tensor([0., 0., 1.]), shells=1,
                    transverse_cutoff=16., quadrature_order=16)
    for update in ({'normal': sample[0]/sample[0].norm()}, {'normal': settings['normal']*2},
                   {'shells': True}, {'transverse_cutoff': float('nan')}, {'quadrature_order': 0}):
        with pytest.raises(ValueError):
            kernel.evaluate_line(points, **{**settings, **update})
    with pytest.raises(ValueError, match='source-plane'):
        kernel.evaluate_line(sample[:1], **settings)
    with pytest.raises(ValueError, match='1D lattice'):
        problem().evaluate_line(points, **settings)
    kernel.medium.epsilon_r[0, 0] += .1
    with pytest.raises(ValueError, match='changed'):
        kernel.evaluate_line(points, **settings)


@pytest.mark.parametrize('failed_axis', ('quadrature', 'transverse_tail', 'reciprocal_tail',
                                         'refined_quadrature', 'combined', None))
def test_line_estimator_checks_each_budget_and_combined_difference(monkeypatch, failed_axis):
    kernel = line_problem()
    points = kernel.lattice.vectors.new_tensor([[.1, .2, .3]])
    budgets = [(1, 8., 8), (1, 8., 16), (1, 16., 32), (2, 16., 32), (2, 16., 64)]
    axes = ['quadrature', 'transverse_tail', 'reciprocal_tail', 'refined_quadrature']
    observed = []
    def evaluate(self, displacement, *, normal, shells, transverse_cutoff, quadrature_order):
        key = (shells, transverse_cutoff, quadrature_order)
        index = budgets.index(key); observed.append(key)
        output = self.medium.epsilon_r.new_zeros(len(displacement), 6, 6)
        if failed_axis == 'combined':
            output[:, 0, 0] = .4*index
        elif failed_axis in axes and index >= axes.index(failed_axis)+1:
            output[:, 0, 0] = 2.
        return output, {'cutoff_converged': False, 'physical_qualification': False}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'evaluate_line', evaluate)
    settings = dict(normal=points.new_tensor([0., 0., 1.]), shells=1,
                    transverse_cutoff=8., quadrature_order=8,
                    relative_tolerance=0., absolute_tolerance=1.)
    if failed_axis is not None:
        with pytest.raises(PeriodicDyadicConvergenceError) as failure:
            kernel.estimate_line(points, **settings)
        assert failure.value.receipt['failed_axis'] == failed_axis
        assert not failure.value.receipt['cutoff_converged']
    else:
        result, receipt = kernel.estimate_line(points, **settings)
        assert receipt['cutoff_converged'] and not receipt['certified_error_bound']
        assert torch.equal(result, torch.zeros_like(result))
    assert observed == budgets[:len(observed)]


def test_line_estimator_rejects_overflowed_acceptance_threshold(monkeypatch):
    kernel = line_problem()
    points = kernel.lattice.vectors.new_tensor([[.1, .2, .3]])
    def evaluate(self, displacement, **settings):
        output = self.medium.epsilon_r.new_zeros(len(displacement), 6, 6)
        output[:, 0, 0] = settings['quadrature_order']
        return output, {'cutoff_converged': False, 'physical_qualification': False}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'evaluate_line', evaluate)
    with pytest.raises(PeriodicDyadicConvergenceError) as failure:
        kernel.estimate_line(points, normal=points.new_tensor([0., 0., 1.]), shells=1,
            transverse_cutoff=8., quadrature_order=8,
            relative_tolerance=1e308, absolute_tolerance=1e308)
    assert failure.value.receipt['failed_axis'] == 'quadrature'
    assert not failure.value.receipt['cutoff_checks']['quadrature']['passed']


@pytest.mark.parametrize('resolved', (True, False))
def test_line_actual_tensor_spectral_refinement_preserves_unresolved_failure(resolved):
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    sample = eye.real.new_tensor([[.7, 0., 0.]])
    epsilon = eye.new_tensor([[2., .1, 0.], [.1, 3., .1], [0., .1, 4.]])+2j*eye
    mu = eye.new_tensor([[1.2, .05, 0.], [.05, 1.4, .1], [0., .1, 1.6]])+1.5j*eye
    kernel = QuasiperiodicDyadicGreenKernel(LatticeSpec(sample, sample.new_tensor([.2, 0., 0.])),
                                           TensorMedium(3e8, epsilon, mu))
    points = sample.new_tensor([[.12, .07, .8]])
    settings = dict(normal=sample.new_tensor([0., 0., 1.]), shells=6,
        transverse_cutoff=64., quadrature_order=64 if resolved else 2,
        relative_tolerance=1e-5 if resolved else 1e-14, absolute_tolerance=1e-9 if resolved else 0.)
    if resolved:
        value, receipt = kernel.estimate_line(points, **settings)
        assert torch.isfinite(value).all() and receipt['cutoff_converged']
        assert all(check['passed'] for check in receipt['cutoff_checks'].values())
        assert not receipt['certified_error_bound'] and not receipt['physical_qualification']
    else:
        with pytest.raises(PeriodicDyadicConvergenceError) as failure:
            kernel.estimate_line(points, **settings)
        assert not failure.value.receipt['cutoff_converged']
