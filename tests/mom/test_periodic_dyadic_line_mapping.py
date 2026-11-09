"""Finite transverse sinh quadrature preserves normalization and material derivatives."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import cross_matrix
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel
from test_periodic_dyadic_line import line_problem

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_sinh_line_scalar_limit_matches_independent_ewald_hessian():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    lattice = LatticeSpec(eye.real.new_tensor([[.7, 0., 0.]]), eye.real.new_tensor([.2, 0., 0.]))
    eps, mu = 1+.4j, 1+.3j
    medium = TensorMedium(3e8, eps*eye, mu*eye)
    kernel = QuasiperiodicDyadicGreenKernel(lattice, medium, mode_batch=1024)
    points = eye.real.new_tensor([[.13, .07, .45], [.1, -.12, -.5]], requires_grad=True)
    actual, receipt = kernel.evaluate_line(points, normal=points.new_tensor([0., 0., 1.]),
        shells=8, transverse_cutoff=64., quadrature_order=192, transverse_map='sinh')
    scalar = EwaldHelmholtzKernel(lattice, 3.5, real_shells=4, reciprocal_shells=10, quadrature_order=128)
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
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-7)
    assert receipt['version'] == 'outgoing_line_sinh_residues_v1'
    assert receipt['transverse_map'] == 'sinh' and not receipt['cutoff_converged']


def test_sinh_line_native_tensor_material_gradient():
    parameter = torch.tensor(.1, device='cuda', dtype=torch.float64, requires_grad=True)
    kernel = line_problem(parameter)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .5]])
    def objective(p):
        value, _ = line_problem(p).evaluate_line(points, normal=points.new_tensor([0., 0., 1.]),
            shells=1, transverse_cutoff=16., quadrature_order=64, transverse_map='sinh')
        return value.real.sum()
    gradient, = torch.autograd.grad(objective(parameter), parameter)
    step = 1e-5
    reference = (objective(parameter.detach()+step)-objective(parameter.detach()-step))/(2*step)
    torch.testing.assert_close(gradient, reference, rtol=1e-6, atol=1e-6)


def test_sinh_transverse_measure_and_input_guards(monkeypatch):
    import witwin.maxwell.mom.periodic_dyadic as pd
    kernel = line_problem()
    sample = kernel.lattice.vectors
    kernel = QuasiperiodicDyadicGreenKernel(
        LatticeSpec(sample.new_tensor([[.7, 0., 0.]]), sample.new_tensor([.2, 0., 0.])), kernel.medium)
    points = kernel.lattice.vectors.new_tensor([[.1, .2, .5]])
    from witwin.maxwell.mom.dyadic_green import PlaneModes
    class ConstantModes(PlaneModes):
        def integrate(self, queries, weights, **kwargs):
            return kernel.medium.epsilon_r.new_ones(len(queries), 6, 6)*weights.sum()
    def constant_modes(transverse, medium, **kwargs):
        return ConstantModes(transverse, medium.epsilon_r.new_zeros(len(transverse), 2),
            medium.epsilon_r.new_zeros(len(transverse), 2, 6, 6), medium)
    monkeypatch.setattr(pd, 'compile_plane_modes', constant_modes)
    settings = dict(normal=points.new_tensor([0., 0., 1.]), shells=2,
                    transverse_cutoff=128., quadrature_order=96, transverse_map='sinh')
    value, _ = kernel.evaluate_line(points, **settings)
    expected = value.new_full(value.shape, 5*128/(math.pi*float(kernel.lattice.cell_measure)))
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='Transverse map'):
        kernel.evaluate_line(points, **{**settings, 'transverse_map': 'invalid'})
    with pytest.raises(ValueError, match='Transverse map'):
        kernel.estimate_line(points, **{**settings, 'transverse_map': 'invalid'})
