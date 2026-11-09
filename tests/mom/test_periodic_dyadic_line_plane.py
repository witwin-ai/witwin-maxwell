"""One-dimensional spectral budgets feed the shared primary-excluded limit."""
from dataclasses import replace
import pytest
import torch
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel, PeriodicDyadicConvergenceError
from test_periodic_dyadic_plane_limit import problem

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def line_problem():
    base = problem()
    return replace(base, lattice=LatticeSpec(base.lattice.vectors[:1],
                                             base.lattice.vectors.new_tensor([.2, 0., 0.])))


@pytest.mark.parametrize('case', ['quadratic', 'quadrature', 'transverse', 'reciprocal', 'roundoff', 'limit'])
def test_line_plane_limits_preserve_independent_gates_and_extrapolation(monkeypatch, case):
    kernel = line_problem()
    points = kernel.lattice.vectors.new_zeros(1, 3)
    def primary(self, queries):
        assert self.integration.relative_tolerance == 0.
        assert self.integration.absolute_tolerance == 1e-3/8
        value = self.medium.epsilon_r.new_zeros(len(queries), 6, 6)
        value[:, 0, 0] = 1e20 if case == 'roundoff' else 1e8
        return value, {'converged': True}
    def periodic(self, queries, *, normal, shells, transverse_cutoff, quadrature_order):
        value = self.medium.epsilon_r.new_zeros(len(queries), 6, 6)
        z = queries@normal
        value[:, 0, 0] = (1e20 if case == 'roundoff' else 1e8)+2+z+100*z.square()
        if case == 'quadrature': value[:, 0, 0] += z.sign()/quadrature_order
        if case == 'transverse': value[:, 0, 0] += z.sign()/transverse_cutoff
        if case == 'reciprocal': value[:, 0, 0] += z.sign()/shells
        if case == 'limit': value[:, 0, 0] += 1/z.abs()
        return value, {'cutoff_converged': False}
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', primary)
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'evaluate_line', periodic)
    settings = dict(normal=points.new_tensor([0., 0., 1.]), height=.1,
        shells=1, transverse_cutoff=8., quadrature_order=8, max_halvings=3,
        absolute_tolerance=1e-3)
    if case == 'quadratic':
        value, receipt = kernel.estimate_line_plane_regular(points, **settings)
        torch.testing.assert_close(value[:, 0, 0], value.new_tensor([2.]), rtol=0., atol=1e-7)
        assert receipt['converged'] and receipt['periodic_dimension'] == 1
        assert len(receipt['limit_checks']) == 2
        assert all(level['line_spectrum']['cutoff_converged'] for level in receipt['levels'])
    else:
        with pytest.raises(PeriodicDyadicConvergenceError) as failure:
            kernel.estimate_line_plane_regular(points, **settings)
        receipt = failure.value.receipt
        assert not receipt['converged']
        if case in ('quadrature', 'transverse', 'reciprocal'):
            assert receipt['failure_reason'] == 'line_spectral_budget_exhausted'
            assert receipt['line_failure']['failed_axis'] == {
                'quadrature': 'quadrature', 'transverse': 'transverse_tail',
                'reciprocal': 'reciprocal_tail'}[case]
        else:
            assert receipt['failure_reason'] == ('primary_subtraction_precision_budget_exhausted'
                if case == 'roundoff' else 'source_plane_limit_budget_exhausted')


def test_line_plane_normal_and_dimension_guards():
    kernel = line_problem()
    points = kernel.lattice.vectors.new_zeros(1, 3)
    settings = dict(normal=points.new_tensor([0., 0., 1.]), height=.1,
        shells=1, transverse_cutoff=8., quadrature_order=8, max_halvings=3,
        absolute_tolerance=1e-3)
    for change in ({'normal': points.new_tensor([1., 0., 0.])},
                   {'normal': points.new_tensor([0., 0., 2.])},
                   {'quadrature_order': True}, {'transverse_cutoff': float('nan')},
                   {'shells': True}):
        with pytest.raises(ValueError):
            kernel.estimate_line_plane_regular(points, **{**settings, **change})
    with pytest.raises(ValueError, match='1D lattice'):
        problem().estimate_line_plane_regular(points, **settings)
    with pytest.raises(ValueError, match='Nonprimary'):
        kernel.estimate_line_plane_regular(3*kernel.lattice.vectors, **settings)
