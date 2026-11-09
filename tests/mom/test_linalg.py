"""CUDA execution, complex adjoint, metric gradients, and solve certificates."""
import pytest
import torch
from witwin.maxwell.mom.linalg import (DenseOperator, StoppingRule, apply_A,
    apply_A_adjoint, apply_test_riesz_inverse, condition_diagnostics, correct,
    residual_loss, solve_reference)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def operator(dtype=torch.complex128, n=12):
    torch.manual_seed(51)
    a = torch.randn(n, n, dtype=dtype, device='cuda') + 5 * torch.eye(n, dtype=dtype, device='cuda')
    p = torch.randn(n, n, dtype=dtype, device='cuda')
    return DenseOperator(a, p @ p.mH + torch.eye(n, dtype=dtype, device='cuda'))


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_batched_adjoint_and_native_metric_gradient(dtype):
    op = operator(dtype)
    tol = 3e-5 if dtype == torch.complex64 else 1e-11
    x = torch.randn(2, 3, 12, dtype=dtype, device='cuda', requires_grad=True)
    y = torch.randn_like(x)
    torch.testing.assert_close((apply_A(op, x).conj() * y).sum(),
        (x.conj() * apply_A_adjoint(op, y)).sum(), rtol=tol, atol=tol)
    b = torch.randn_like(x)
    loss = residual_loss(op, x, b)
    gradient, = torch.autograd.grad(loss, x)
    expected = apply_A_adjoint(op, apply_test_riesz_inverse(op, apply_A(op, x) - b))
    torch.testing.assert_close(gradient, expected, rtol=tol, atol=tol)
    assert gradient.is_cuda and loss > 0


def test_real_and_imaginary_finite_differences():
    op = operator(n=4)
    x = torch.randn(4, dtype=op.dtype, device='cuda', requires_grad=True)
    b = torch.randn_like(x)
    grad, = torch.autograd.grad(residual_loss(op, x, b), x)
    for component in [1., 1j]:
        d = torch.zeros_like(x)
        d[2] = component
        fd = (residual_loss(op, x + 1e-6 * d, b) - residual_loss(op, x - 1e-6 * d, b)) / 2e-6
        torch.testing.assert_close(fd, (grad.conj() * d).sum().real, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_direct_and_gmres_arbitrary_guesses(dtype):
    op = operator(dtype)
    b = torch.randn(12, dtype=dtype, device='cuda')
    tol = 1e-5 if dtype == torch.complex64 else 1e-8
    direct, direct_log = solve_reference(op, b, StoppingRule(method='direct', rtol=tol))
    assert direct_log['converged']
    again, reuse_log = solve_reference(op, b, StoppingRule(method='direct', rtol=tol))
    assert reuse_log['cache_reused']
    torch.testing.assert_close(direct, again)
    for guess in [torch.zeros_like(b), direct + .1 * torch.randn_like(b), 10 * torch.randn_like(b)]:
        x, log = correct(op, b, guess, StoppingRule(rtol=tol, restart=12))
        assert log['converged'], log
        actual = torch.linalg.vector_norm(op.matrix @ x - b) / torch.linalg.vector_norm(b)
        assert actual <= tol
        assert log['relative_residual'] == pytest.approx(float(actual), rel=1e-5, abs=1e-15)
        assert log['matvecs'] >= log['iterations'] + log['residual_checks']
        assert log['device'].startswith('cuda') and log['peak_memory_bytes'] > 0
        assert log['setup_seconds'] >= 0 and log['solve_seconds'] > 0


def test_batched_rhs_restarts_and_zero_rhs_absolute_floor():
    op = operator(n=8)
    b = torch.randn(2, 3, 8, dtype=op.dtype, device=op.device)
    x, log = correct(op, b, torch.zeros_like(b), StoppingRule(restart=4, maxiter=200))
    assert log['converged'], log
    assert log['rhs_count'] == 6 and log['restarts'] > 0
    torch.testing.assert_close(apply_A(op, x), b, rtol=1e-7, atol=1e-7)
    b = torch.zeros(8, dtype=op.dtype, device=op.device)
    x, log = correct(op, b, torch.randn_like(b), StoppingRule(restart=8, atol=1e-11))
    assert log['converged'], log
    assert log['relative_residual'] is None and log['absolute_residual'] <= 1e-11
    x, log = correct(op, b, b.clone())
    assert log['converged'] and log['iterations'] == 0


def test_budget_and_stagnation_never_report_success():
    op = operator()
    b = torch.randn(12, dtype=op.dtype, device=op.device)
    _, log = correct(op, b, torch.zeros_like(b), StoppingRule(maxiter=1))
    assert not log['converged'] and log['reason'] == 'maxiter'
    _, log = correct(op, b, torch.zeros_like(b), StoppingRule(restart=1, stagnation_window=1, stagnation_rtol=.99))
    assert not log['converged'] and log['reason'] == 'stagnation'


def test_singular_failure_and_small_residual_large_error_diagnostic():
    eye = torch.eye(2, dtype=torch.complex128, device='cuda')
    a = torch.diag(torch.tensor([1., 0.], dtype=torch.complex128, device='cuda'))
    op = DenseOperator(a, eye)
    b = torch.ones(2, dtype=op.dtype, device=op.device)
    _, log = solve_reference(op, b)
    assert not log['converged'] and log['reason'] == 'singular_matrix'
    a[1, 1] = 1e-8
    exact = torch.ones_like(b)
    approximate = torch.tensor([1., 0.], dtype=op.dtype, device=op.device)
    assert torch.linalg.vector_norm(apply_A(op, approximate - exact)) < 2e-8
    assert torch.linalg.vector_norm(approximate - exact) == 1
    diagnostics = condition_diagnostics(op)
    assert diagnostics['condition_number'] == pytest.approx(1e8)
    assert not diagnostics['certified_stability_bound']


def test_cache_invalidation_and_guards():
    op = operator(n=3)
    b = torch.ones(3, dtype=op.dtype, device=op.device)
    solve_reference(op, b)
    op.matrix.mul_(2)
    x, log = solve_reference(op, b)
    assert not log['cache_reused'] and log['converged']
    torch.testing.assert_close(apply_A(op, x), b)
    old = apply_test_riesz_inverse(op, b)
    op.test_riesz.mul_(2)
    torch.testing.assert_close(apply_test_riesz_inverse(op, b), old / 2)
    with pytest.raises(ValueError, match='positive definite'):
        DenseOperator(op.matrix, -op.test_riesz)
    with pytest.raises(ValueError, match='CUDA'):
        DenseOperator(op.matrix.cpu(), op.test_riesz.cpu())
    with pytest.raises(ValueError, match='finite'):
        correct(op, b * float('nan'), b)
    with pytest.raises(ValueError):
        StoppingRule(rtol=-1)


def test_residual_norm_invariant_under_test_row_scaling():
    op = operator(n=5)
    x = torch.randn(3, 5, dtype=op.dtype, device=op.device)
    b = torch.randn_like(x)
    d = torch.diag(torch.tensor([.01, .1, 1., 10., 100.], dtype=op.dtype, device=op.device))
    scaled = DenseOperator(d @ op.matrix, d @ op.test_riesz @ d.mH)
    torch.testing.assert_close(residual_loss(scaled, x, b @ d.mT), residual_loss(op, x, b),
                               rtol=1e-11, atol=1e-11)


def test_inconsistent_gmres_system_reports_failure():
    a = torch.zeros((3, 3), dtype=torch.complex128, device='cuda')
    op = DenseOperator(a, torch.eye(3, dtype=a.dtype, device=a.device))
    b = torch.ones(3, dtype=a.dtype, device=a.device)
    solution, log = correct(op, b, torch.zeros_like(b))
    assert not log['converged']
    assert log['reason'] in ('breakdown', 'nonfinite_residual', 'nonfinite_iterate')
    assert bool(torch.isfinite(solution).all())


def test_tiny_rhs_uses_fixed_absolute_tolerance():
    op = operator(n=3)
    b = torch.full((3,), 1e-15, dtype=op.dtype, device=op.device)
    _, log = correct(op, b, torch.zeros_like(b), StoppingRule(atol=1e-12))
    assert log['converged'] and log['iterations'] == 0
    assert log['relative_residual'] == pytest.approx(1)
    assert log['threshold'] == 1e-12


def test_mutated_operator_guards_run_before_any_cpu_numerics(monkeypatch):
    op = operator(n=3)
    vector = torch.ones(3, dtype=op.dtype, device=op.device)
    original = op.test_riesz
    op.test_riesz = original.cpu()
    def forbidden_cholesky(*args, **kwargs):
        raise AssertionError('invalid replacement reached numerical Cholesky')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, 'cholesky_ex', forbidden_cholesky)
        with pytest.raises(ValueError, match='CUDA device'):
            apply_test_riesz_inverse(op, vector)
    op.test_riesz = original.clone()
    op.test_riesz[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        apply_test_riesz_inverse(op, vector)
    op.test_riesz = original
    op.matrix = op.matrix.cpu()
    with pytest.raises(ValueError, match='CUDA'):
        apply_A(op, vector.cpu())
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, 'svdvals', lambda *args: pytest.fail('invalid matrix reached numerical SVD'))
        with pytest.raises(ValueError, match='CUDA'):
            condition_diagnostics(op)
    op.matrix = op.matrix.cuda()
    op.matrix[0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        apply_A_adjoint(op, vector)


def test_disabled_preconditioner_has_no_setup_cache_and_lu_reports_none():
    op = operator(n=3)
    b = torch.ones(3, dtype=op.dtype, device=op.device)
    for _ in range(2):
        _, log = correct(op, b, torch.zeros_like(b), StoppingRule(preconditioner='none'))
        assert log['converged'] and not log['cache_reused']
        assert op._scale is None
    _, log = solve_reference(op, b)
    assert log['preconditioner'] == 'none'


def test_actual_pmchwt_metric_gradient_real_imaginary_finite_differences():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak, build_rhs
    from witwin.maxwell.mom.physics import compile_medium
    basis = RWGBasis(sphere_mesh(radius=0.1, subdivisions=0))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2, sigma_e=0.01), 3e8)
    a, _ = assemble_weak(basis, exterior, interior, IntegrationConfig(order=3, singular_order=4))
    mass = basis.mass_matrix().to(a.dtype)
    op = DenseOperator(a, torch.block_diag(mass, mass))
    b = build_rhs(basis, exterior, [mw.PlaneWave()])
    x = torch.randn(a.shape[0], dtype=a.dtype, device=a.device, requires_grad=True)
    grad, = torch.autograd.grad(residual_loss(op, x, b), x)
    for component in (1., 1j):
        direction = torch.zeros_like(x)
        direction[4] = component
        step = 1e-5
        difference = (residual_loss(op, x+step*direction, b)-residual_loss(op, x-step*direction, b))/(2*step)
        expected = (grad.conj()*direction).sum().real
        torch.testing.assert_close(difference, expected, atol=1e-5, rtol=2e-7)
    reference, reference_log = solve_reference(op, b)
    corrected, corrected_log = correct(op, b, reference+0.01*torch.randn_like(reference), StoppingRule(restart=len(b), maxiter=2*len(b)))
    assert reference_log['converged'] and corrected_log['converged'], corrected_log
    assert torch.linalg.vector_norm(a@corrected-b)/torch.linalg.vector_norm(b) <= 1e-8
    recycle_rule = StoppingRule(restart=len(b), maxiter=2*len(b), recycle_size=2)
    _, first = correct(op, b, torch.zeros_like(b), recycle_rule)
    reused, second = correct(op, b, torch.zeros_like(b), recycle_rule)
    assert first['converged'] and second['converged']
    assert second['iterations'] == 0 and second['matvecs'] < first['matvecs']
    assert torch.linalg.vector_norm(a@reused-b)/torch.linalg.vector_norm(b) <= 1e-8


def test_recycled_initial_augmentation_spans_rhs_and_preserves_true_residual():
    op = operator(n=12)
    rule = StoppingRule(restart=12, recycle_size=4)
    rhs = torch.randn(3, 12, dtype=op.dtype, device=op.device)
    solutions, first_log = correct(op, rhs, torch.zeros_like(rhs), rule)
    assert first_log['converged']
    assert op._recycle_u.shape == (12, 3)
    assert first_log['recycle_update_matvecs'] == 3
    assert first_log['recycle_projection_matvecs'] == 2
    torch.testing.assert_close(op.matrix@op._recycle_u, op._recycle_c, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(op._recycle_c.mH@op._recycle_c, torch.eye(3, dtype=op.dtype, device=op.device), atol=1e-12, rtol=1e-12)
    coefficients = torch.tensor([0.7+0.2j, -0.1j, 1.3], dtype=op.dtype, device=op.device)
    combined = coefficients@rhs
    solution, log = correct(op, combined, torch.zeros_like(combined), rule)
    assert log['converged'] and log['iterations'] == 0
    assert log['recycle_dimension_before'] == 3
    assert log['recycle_projection_matvecs'] == 1
    assert not log['recycle_direction_added']
    assert log['residual_history'][0]['phase'] == 'before_recycle_projection'
    assert log['residual_checks'] == len(log['residual_history'])
    torch.testing.assert_close(solution, coefficients@solutions, atol=1e-11, rtol=1e-10)
    assert torch.linalg.vector_norm(op.matrix@solution-combined)/torch.linalg.vector_norm(combined) <= rule.rtol
    # A new arbitrary residual outside the retained span still receives GMRES.
    fresh = torch.randn_like(combined)
    initial = 5*torch.randn_like(combined)
    solution, outside_log = correct(op, fresh, initial, rule)
    assert outside_log['converged'] and outside_log['iterations'] > 0
    assert outside_log['recycle_dimension_after'] == 4
    assert outside_log['recycle_storage_bytes'] == 2*12*4*op.matrix.element_size()


def test_recycle_capacity_mutation_invalidation_and_failure_do_not_leak_cache():
    op = operator(n=7)
    b = torch.randn(7, dtype=op.dtype, device=op.device)
    rule = StoppingRule(restart=7, recycle_size=2)
    for _ in range(3):
        _, log = correct(op, torch.randn_like(b), torch.zeros_like(b), rule)
        assert log['converged'] and log['recycle_dimension_after'] <= 2
    previous = op._recycle_c.clone()
    _, failed = correct(op, torch.randn_like(b), torch.randn_like(b), StoppingRule(maxiter=1, recycle_size=2))
    assert not failed['converged'] and failed['recycle_update_matvecs'] == 0
    torch.testing.assert_close(previous, op._recycle_c)
    op.matrix.mul_(1.1)
    _, updated = correct(op, b, torch.zeros_like(b), rule)
    assert updated['converged'] and updated['recycle_dimension_before'] == 0
    assert updated['recycle_projection_matvecs'] == 0
    _, reused = correct(op, b, torch.zeros_like(b), StoppingRule(restart=7, recycle_size=1))
    assert reused['converged'] and reused['recycle_dimension_after'] == 1
    with pytest.raises(ValueError, match='recycled correction'):
        StoppingRule(method='direct', recycle_size=2)


def test_two_sided_scaling_preserves_physical_unknowns_and_invalidates_cache():
    op = operator(n=12)
    columns = torch.logspace(-6, 6, 12, dtype=op.matrix.real.dtype, device=op.device)
    op.matrix = op.matrix*columns[None]
    b = torch.randn(12, dtype=op.dtype, device=op.device)
    rule = StoppingRule(preconditioner='row_column_norm', restart=12, maxiter=100)
    solution, first = correct(op, b, torch.randn_like(b), rule)
    assert first['converged'], first
    torch.testing.assert_close(apply_A(op, solution), b, atol=1e-8, rtol=1e-8)
    assert first['preconditioner'] == 'row_column_norm'
    assert op._column_scale.is_cuda and op._balanced_row_scale.is_cuda
    _, second = correct(op, b, torch.zeros_like(b), rule)
    assert second['converged'] and second['cache_reused']
    old = op._column_scale.clone()
    op.matrix.mul_(2)
    _, updated = correct(op, b, torch.zeros_like(b), rule)
    assert updated['converged'] and not updated['cache_reused']
    torch.testing.assert_close(op._column_scale, old/2)


@pytest.mark.parametrize('batch', [False, True])
def test_explicit_direct_fallback_retains_failed_attempt_and_full_cost(batch):
    op = operator(n=12)
    b = torch.randn((2, 12) if batch else (12,), dtype=op.dtype, device=op.device)
    rule = StoppingRule(maxiter=1, fallback='direct')
    solution, log = correct(op, b, torch.zeros_like(b), rule)
    assert log['converged'] and log['reason'] == 'fallback_converged'
    assert log['fallback_used'] and not log['gmres_attempt']['converged']
    assert log['gmres_attempt']['reason'] == ('batch_failure' if batch else 'maxiter')
    assert log['fallback_log']['method'] == 'direct'
    assert log['matvecs'] == log['gmres_attempt']['matvecs']+log['fallback_log']['matvecs']
    assert log['total_seconds'] >= log['gmres_attempt']['total_seconds']+log['fallback_log']['total_seconds']
    torch.testing.assert_close(apply_A(op, solution), b, atol=1e-10, rtol=1e-10)
    if not batch:
        assert len(log['residual_history']) == log['residual_checks']
        assert log['residual_history'][-1]['solver_phase'] == 'direct_fallback'
        assert log['relative_residual'] <= rule.rtol


def test_singular_fallback_preserves_failure_and_has_no_success_claim():
    matrix = torch.zeros((3, 3), dtype=torch.complex128, device='cuda')
    op = DenseOperator(matrix, torch.eye(3, dtype=matrix.dtype, device=matrix.device))
    b = torch.ones(3, dtype=matrix.dtype, device=matrix.device)
    _, log = correct(op, b, torch.zeros_like(b), StoppingRule(fallback='direct'))
    assert not log['converged'] and log['reason'] == 'fallback_failed'
    assert log['fallback_log']['reason'] == 'singular_matrix'
    assert log['fallback_used']
    with pytest.raises(ValueError, match='fallback'):
        StoppingRule(method='direct', fallback='direct')


def test_two_sided_scaling_resolves_actual_pmchwt_row_only_budget_failure():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak, build_rhs
    from witwin.maxwell.mom.physics import compile_medium
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=1))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.25, sigma_e=.004), 3e8)
    matrix, _ = assemble_weak(basis, exterior, interior, IntegrationConfig(order=3, singular_order=4))
    mass = basis.mass_matrix().to(matrix.dtype)
    op = DenseOperator(matrix, torch.block_diag(mass/abs(exterior.eta)**2, mass))
    b = build_rhs(basis, exterior, [mw.PlaneWave()])
    torch.manual_seed(77)
    initial = torch.randn_like(b)
    settings = dict(maxiter=350, restart=80, stagnation_window=80)
    _, row_log = correct(op, b, initial, StoppingRule(preconditioner='row_norm', **settings))
    solution, balanced_log = correct(op, b, initial, StoppingRule(preconditioner='row_column_norm', **settings))
    assert not row_log['converged'] and row_log['reason'] == 'maxiter'
    assert balanced_log['converged'], balanced_log
    assert balanced_log['iterations'] < row_log['iterations']
    assert torch.linalg.vector_norm(matrix@solution-b)/torch.linalg.vector_norm(b) <= 1e-8
