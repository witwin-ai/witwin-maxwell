"""True shared block Krylov space, rank deflation, and physical residuals."""
from dataclasses import asdict

import pytest
import torch

from witwin.maxwell.mom.linalg import DenseOperator, StoppingRule, apply_A, correct, solve_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def operator(n=18, dtype=torch.complex128):
    torch.manual_seed(837)
    matrix = torch.randn(n, n, dtype=dtype, device='cuda')/n**.5
    matrix += 3*torch.eye(n, dtype=dtype, device='cuda')
    return DenseOperator(matrix, torch.eye(n, dtype=dtype, device='cuda'))


def assert_residuals(op, b, x, log):
    rows = log.get('rhs_logs', [log])
    residual = torch.linalg.vector_norm(apply_A(op, x)-b, dim=-1).reshape(-1)
    # BLAS may use different reduction paths for an active subblock and the
    # independent full-batch check; bound absolute roundoff by the operand scale.
    scale = torch.linalg.vector_norm(apply_A(op, x).abs()+b.abs(), dim=-1).reshape(-1)
    for actual, row, magnitude in zip(residual, rows, scale):
        roundoff = 8*torch.finfo(op.matrix.real.dtype).eps*float(magnitude)
        assert row['absolute_residual'] == pytest.approx(float(actual), rel=2e-4, abs=roundoff)
        assert row['converged'] == (float(actual) <= row['threshold'])
    assert log['matvecs'] == log['arnoldi_vector_actions'] + log['residual_vector_actions']
    assert sum(row['residual_checks'] for row in rows) == log['residual_checks']


def test_shared_block_space_succeeds_where_independent_one_step_cannot():
    matrix = torch.diag(torch.tensor([1., 4., 9., 16.], dtype=torch.complex128, device='cuda'))
    op = DenseOperator(matrix, torch.eye(4, dtype=matrix.dtype, device=matrix.device))
    b = torch.tensor([[1., 1., 0., 0.], [1., -1., 0., 0.]], dtype=op.dtype, device=op.device)
    _, scalar = correct(op, b, torch.zeros_like(b), StoppingRule(maxiter=1, preconditioner='none'))
    x, block = correct(op, b, torch.zeros_like(b), StoppingRule(method='block_gmres', maxiter=1, preconditioner='none'))
    assert not scalar['converged'] and block['converged']
    assert block['block_iterations'] == 1 and block['arnoldi_vector_actions'] == 2
    assert block['block_operator_products'] == 3
    assert_residuals(op, b, x, block)
    torch.testing.assert_close(x, torch.linalg.solve(matrix, b.mT).mT, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('preconditioner', ['none', 'row_norm', 'row_column_norm'])
def test_arbitrary_initial_block_and_cached_setup_match_direct(dtype, preconditioner):
    op = operator(dtype=dtype)
    b = torch.randn(2, 2, 18, device=op.device, dtype=dtype)
    guess = 3*torch.randn_like(b)
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-10
    rule = StoppingRule(method='block_gmres', restart=3, maxiter=50, rtol=tolerance,
                        preconditioner=preconditioner)
    x, log = correct(op, b, guess, rule)
    assert log['converged'] and log['restarts'] > 0, log
    assert_residuals(op, b, x, log)
    expected = torch.linalg.solve(op.matrix, b.reshape(-1, 18).mT).mT.reshape(b.shape)
    torch.testing.assert_close(x, expected, rtol=10*tolerance, atol=10*tolerance)
    _, repeated = correct(op, b, torch.zeros_like(b), rule)
    assert repeated['converged'] and repeated['cache_reused'] == (preconditioner != 'none')


def test_zero_duplicate_correlated_and_weak_rhs_lock_independently():
    op = operator(n=12)
    b = torch.randn(6, 12, device=op.device, dtype=op.dtype)
    b[0] = 0
    b[2] = (2+3j)*b[1]
    b[3] = 1e-9*b[3]
    b[4] = b[1]+1e-8*b[4]
    initial = torch.zeros_like(b)
    initial[5] = torch.linalg.solve(op.matrix, b[5])
    x, log = correct(op, b, initial, StoppingRule(method='block_gmres', restart=12, rtol=1e-10, atol=1e-22))
    assert log['converged'], log
    assert log['rhs_logs'][0]['iterations'] == 0
    assert log['rhs_logs'][5]['iterations'] == 0
    assert log['block_rank_history'][0]['retained_rank'] == 3
    assert torch.equal(x[5], initial[5])
    assert_residuals(op, b, x, log)


def test_zero_rhs_with_nonzero_guess_and_more_rhs_than_dimension():
    op = operator(n=5)
    b = torch.randn(8, 5, device=op.device, dtype=op.dtype)
    b[0] = 0
    x, log = correct(op, b, torch.randn_like(b), StoppingRule(method='block_gmres', atol=1e-11))
    assert log['converged'] and log['maximum_block_basis_columns'] <= 5
    assert log['rhs_logs'][0]['relative_residual'] is None
    assert_residuals(op, b, x, log)


def test_budget_stagnation_singular_breakdown_and_explicit_fallback():
    op = operator()
    b = torch.randn(3, 18, device=op.device, dtype=op.dtype)
    rule = StoppingRule(method='block_gmres', maxiter=1, preconditioner='none')
    x, failed = correct(op, b, torch.zeros_like(b), rule)
    assert not failed['converged'] and all(r['reason'] == 'maxiter' for r in failed['rhs_logs'])
    assert_residuals(op, b, x, failed)
    _, stagnated = correct(op, b, torch.zeros_like(b), StoppingRule(method='block_gmres',
        restart=1, stagnation_window=1, stagnation_rtol=.99, preconditioner='none'))
    assert not stagnated['converged'] and any(r['reason'] == 'stagnation' for r in stagnated['rhs_logs'])
    x, fallback = correct(op, b, torch.zeros_like(b), StoppingRule(**{**asdict(rule), 'fallback':'direct'}))
    assert fallback['converged'] and fallback['fallback_used']
    assert not fallback['gmres_attempt']['converged']
    assert fallback['matvecs'] == fallback['gmres_attempt']['matvecs']+fallback['fallback_log']['matvecs']
    matrix = torch.diag(torch.tensor([1., 0.], dtype=op.dtype, device=op.device))
    singular = DenseOperator(matrix, torch.eye(2, dtype=op.dtype, device=op.device))
    rhs = torch.tensor([[1., 0.], [0., 1.]], dtype=op.dtype, device=op.device)
    x, failure = correct(singular, rhs, torch.zeros_like(rhs), StoppingRule(method='block_gmres'))
    assert not failure['converged'] and failure['rhs_logs'][0]['converged']
    assert failure['rhs_logs'][1]['reason'] == 'breakdown'
    assert_residuals(singular, rhs, x, failure)


def test_single_rhs_and_configuration_roundtrip():
    op = operator(n=6)
    b = torch.randn(6, dtype=op.dtype, device=op.device)
    rule = StoppingRule(method='block_gmres', restart=6)
    restored = StoppingRule(**asdict(rule))
    x, log = correct(op, b, torch.randn_like(b), restored)
    assert log['converged'] and log['rhs_count'] == 1
    assert_residuals(op, b, x, log)
    with pytest.raises(ValueError, match='recycled correction'):
        StoppingRule(method='block_gmres', recycle_size=2)


def test_actual_lossy_pmchwt_multi_source_block_matches_cached_lu():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs

    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.004))],
                     sources=[mw.PlaneWave()])
    rule = StoppingRule(method='block_gmres', preconditioner='row_column_norm', restart=50,
                        maxiter=100, stagnation_window=60)
    prepared = mw.Simulation.mom(scene, frequency=3e8,
        config=mw.MoMConfig(subdivisions=1, integration=IntegrationConfig(order=3, singular_order=4), stopping=rule)).prepare()
    op = prepared.solver.handle
    sources = [mw.PlaneWave(direction=d, polarization=p) for d,p in (
        ((0,0,1),(1,0,0)), ((0,0,1),(0,1,0)), ((1,0,0),(0,0,1)),
        ((1,0,0),(0,1,0)), ((0,1,0),(1,0,0)), ((0,1,0),(0,0,1)))]
    b = torch.stack([build_rhs(op.basis, op.exterior, [source], order=4) for source in sources])
    expected, direct_log = solve_reference(op, b)
    torch.manual_seed(192)
    x, log = correct(op, b, .1*torch.randn_like(b), rule)
    assert direct_log['converged'] and log['converged'] and not log['fallback_used'], log
    assert_residuals(op, b, x, log)
    assert torch.linalg.vector_norm(x-expected)/torch.linalg.vector_norm(expected) < 2e-7


def test_full_rank_exhaustion_restarts_to_resolve_weak_rhs_after_large_guess():
    op = operator(n=17)
    b = torch.randn(6, 17, dtype=op.dtype, device=op.device)
    b[-1] *= 1e-12
    x, log = correct(op, b, torch.randn_like(b), StoppingRule(method='block_gmres',
        rtol=3e-9, atol=1e-24, maxiter=30, restart=10))
    assert log['converged'] and log['restarts'] >= 1, log
    assert_residuals(op, b, x, log)
    assert log['rhs_logs'][-1]['relative_residual'] <= 3e-9


def test_svd_failure_is_an_explicit_unsuccessful_solve(monkeypatch):
    op = operator(n=4)
    b = torch.ones(2, 4, dtype=op.dtype, device=op.device)
    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError('injected CUDA SVD nonconvergence')
    monkeypatch.setattr(torch.linalg, 'svd', fail)
    _, log = correct(op, b, torch.zeros_like(b), StoppingRule(method='block_gmres'))
    assert not log['converged']
    assert all(row['reason'] == 'factorization_failure' for row in log['rhs_logs'])
    assert log['matvecs'] == 2 and log['block_iterations'] == 0
