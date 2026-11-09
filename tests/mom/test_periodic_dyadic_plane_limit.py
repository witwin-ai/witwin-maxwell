"""Primary-subtracted source-plane limits with independent spectral gates."""
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel, DyadicIntegrationConfig
from witwin.maxwell.mom.periodic_dyadic import (
    QuasiperiodicDyadicGreenKernel, LossyDyadicImageReference,
    PeriodicDyadicConvergenceError, _cutoff_check)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(dimension=2):
    vectors = torch.tensor([[.7, 0., 0.], [0., .8, 0.]], device='cuda', dtype=torch.float64)
    if dimension == 3:
        vectors = torch.cat((vectors, vectors.new_tensor([[.05, .1, .9]])))
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (1+2j)*eye, (1+2j)*eye)
    return QuasiperiodicDyadicGreenKernel(
        LatticeSpec(vectors, vectors.new_tensor([.2, .1, 0.])), medium, mode_batch=1024)


@pytest.mark.parametrize('dimension', (2, 3))
@pytest.mark.parametrize('case', ['finite', 'quadratic', 'spectral', 'limit', 'roundoff'])
def test_plane_limit_keeps_primary_and_opposite_side_cancellation_out_of_admission(monkeypatch, case, dimension):
    kernel = problem(dimension)
    estimate = kernel.estimate_stacked_plane_regular if dimension == 3 else kernel.estimate_open_plane_regular
    points = kernel.lattice.vectors.new_zeros(1, 3)
    budgets = []
    def primary(self, queries):
        budgets.append((self.integration.relative_tolerance, self.integration.absolute_tolerance))
        value = self.medium.epsilon_r.new_zeros(len(queries), 6, 6)
        value[:, 0, 0] = 1e20 if case == 'roundoff' else 1e8
        return value, {'converged': True}
    def periodic(self, queries, *, shells):
        value = self.medium.epsilon_r.new_zeros(len(queries), 6, 6)
        z = queries[:, 2]
        value[:, 0, 0] = (1e20 if case == 'roundoff' else 1e8)+2+z
        if case == 'quadratic':
            value[:, 0, 0] += 100*z.square()
        if case == 'spectral':
            value[:, 0, 0] += z.sign()/shells
        if case == 'limit':
            value[:, 0, 0] += 1/z.abs()
        return value, {}
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', primary)
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'evaluate_stacked' if dimension == 3 else 'evaluate_open', periodic)
    settings = dict(height=.1, shells=1, max_shells=2, max_halvings=3, absolute_tolerance=1e-3)
    if case in ('finite', 'quadratic'):
        value, receipt = estimate(points, **settings)
        torch.testing.assert_close(value[:, 0, 0], value.new_tensor([2.]), rtol=0., atol=1e-7)
        assert receipt['converged'] and len(receipt['limit_checks']) == 2
        assert receipt['primary_image_excluded'] and not receipt['physical_qualification']
    else:
        with pytest.raises(PeriodicDyadicConvergenceError) as failure:
            estimate(points, **settings)
        receipt = failure.value.receipt
        assert not receipt['converged']
        assert receipt['failure_reason'] == {'spectral': 'reciprocal_budget_exhausted',
            'limit': 'source_plane_limit_budget_exhausted',
            'roundoff': 'primary_subtraction_precision_budget_exhausted'}[case]
    assert all(relative == 0. and absolute == 1e-3/8 for relative, absolute in budgets)


def test_plane_limit_rejects_nonprimary_singularities_and_invalid_budgets():
    kernel = problem()
    points = kernel.lattice.vectors.new_zeros(1, 3)
    settings = dict(height=.1, shells=1, max_shells=2, max_halvings=3, absolute_tolerance=1e-3)
    for update in ({'height': 0.}, {'shells': True}, {'max_shells': 1},
                   {'max_halvings': 1}, {'absolute_tolerance': float('inf')}):
        with pytest.raises(ValueError):
            kernel.estimate_open_plane_regular(points, **{**settings, **update})
    with pytest.raises(ValueError, match='Nonprimary'):
        kernel.estimate_open_plane_regular(8*kernel.lattice.vectors[:1], **settings)
    with pytest.raises(ValueError, match='source-plane coordinates'):
        kernel.estimate_open_plane_regular(points+points.new_tensor([0., 0., .1]), **settings)


def test_actual_plane_limit_matches_independent_primary_excluded_images(tmp_path):
    kernel = problem()
    points = kernel.lattice.vectors.new_tensor([[0., 0., 0.], [.1, .07, 0.]])
    tolerance = .02
    try:
        value, receipt = kernel.estimate_open_plane_regular(points, height=.12,
            shells=8, max_shells=512, max_halvings=3, absolute_tolerance=tolerance)
    except PeriodicDyadicConvergenceError as error:
        torch.save(error.receipt, tmp_path/'plane-limit-failure.pt')
        raise AssertionError(error.receipt) from error
    expected, reference = LossyDyadicImageReference(kernel.lattice, kernel.medium, 6).estimate_regular(
        points, relative_tolerance=0., absolute_tolerance=1e-6)
    assert _cutoff_check(expected, value, 0., tolerance)['passed']
    assert reference['image_cutoff_converged'] and receipt['converged']
    assert all(check['passed'] for level in receipt['levels']
               for check in level['spectral_checks'][-1:])
    assert not receipt['certified_error_bound']


def test_actual_tensor_primary_budget_failure_is_preserved():
    base = problem()
    epsilon = base.medium.epsilon_r.clone()
    epsilon[1, 1] += .3
    kernel = QuasiperiodicDyadicGreenKernel(base.lattice,
        TensorMedium(3e8, epsilon, base.medium.mu_r))
    points = base.lattice.vectors.new_zeros(1, 3)
    with pytest.raises(PeriodicDyadicConvergenceError) as failure:
        kernel.estimate_open_plane_regular(points, height=.1, shells=1, max_shells=2,
            max_halvings=3, absolute_tolerance=1e-12,
            integration=DyadicIntegrationConfig(2, 2, 4, 4))
    receipt = failure.value.receipt
    assert receipt['failure_reason'] == 'primary_spectral_budget_exhausted'
    assert not receipt['primary_failure']['converged'] and not receipt['converged']
