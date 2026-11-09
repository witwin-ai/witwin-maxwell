"""Exact CUDA metric storage, independent dense oracles and public integration."""
from dataclasses import asdict, replace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import (CompactRWGRiesz, BlockCompactRWGRiesz, DenseOperator, StoppingRule,
    apply_A_adjoint, apply_test_riesz_inverse, residual_loss, solve_reference)
from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_explicit_storage_configuration():
    assert mw.MoMConfig().riesz_storage == 'dense'
    assert mw.MoMConfig(riesz_storage='compact_rwg_v1').riesz_storage == 'compact_rwg_v1'
    for invalid in ('diagonal', 'auto', None, False, [], 1):
        with pytest.raises(ValueError, match='riesz_storage'):
            mw.MoMConfig(riesz_storage=invalid)


@pytest.mark.parametrize('configuration,marker', [('compact_rwg_v1', 'dense'),
    ('compact_rwg_v1', None), (None, 'compact_rwg_v1'), ('dense', 'compact_rwg_v1')])
def test_restore_rejects_conflicting_storage_provenance_before_compile(monkeypatch, configuration, marker):
    from witwin.maxwell.mom import runtime
    config = mw.MoMConfig()
    payload = dict(schema_version=3, basis='RWG', coefficient_order=['all_J_scattered', 'all_M_scattered'],
        coefficient_units=['A/m', 'V/m'], coordinate_units='m', subdivisions=0,
        dtype='torch.complex128', integration=asdict(config.integration),
        stopping=asdict(config.stopping), query_config=asdict(config.query_config), operator_metadata={},
        payload_kind='physical_lifted_solution', executable=True, scattered_coefficients=None,
        incident_carrier=None, incident_carrier_identity=None, body_table=None, exterior=None,
        scene_signature=None, **runtime._representation_identity())
    if configuration is not None: payload['riesz_storage'] = configuration
    if marker is not None: payload['operator_metadata']['riesz_storage'] = marker
    monkeypatch.setattr(runtime, 'compile_surfaces', lambda *args: pytest.fail('conflicting storage reached geometry compilation'))
    with pytest.raises(ValueError, match='Riesz storage metadata|missing or unsupported fields'):
        runtime.SurfaceData.restore(payload, None, 3e8)


def test_historical_dense_storage_requires_explicit_archive_migration(monkeypatch):
    from witwin.maxwell.mom import runtime
    config = mw.MoMConfig()
    payload = dict(schema_version=1, basis='RWG', coefficient_order=['J', 'M'],
        coefficient_units=['A/m', 'V/m'], coordinate_units='m', subdivisions=0,
        dtype='torch.complex128', integration=asdict(config.integration),
        stopping=asdict(config.stopping), query_config=asdict(config.query_config), operator_metadata={})
    monkeypatch.setattr(runtime, 'compile_surfaces', lambda *args: pytest.fail('historical payload reached runtime compilation'))
    with pytest.raises(ValueError, match='historical data requires explicit archive migration'):
        runtime.SurfaceData.restore(payload, None, 3e8)


def system(dtype=torch.complex128, n=5):
    torch.manual_seed(7301)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    a = torch.randn(2*n, 2*n, device='cuda', dtype=dtype)
    a = a + 8 * torch.eye(2*n, device='cuda', dtype=dtype)
    p = torch.randn(n, n, device='cuda', dtype=real_dtype)
    g = p @ p.mT + torch.eye(n, device='cuda', dtype=real_dtype)
    return a, g


@cuda
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_inverse_and_native_gradient_match_independent_full_metric(dtype):
    a, g = system(dtype)
    eta2 = 7.25
    op = DenseOperator(a, CompactRWGRiesz(g, eta2))
    full_metric = torch.block_diag(g / eta2, g).to(dtype)
    r = torch.randn(2, 3, 10, device='cuda', dtype=dtype)
    expected = torch.linalg.solve(full_metric, r.reshape(-1, 10).mT).mT.reshape(r.shape)
    tol = 2e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(apply_test_riesz_inverse(op, r), expected, rtol=tol, atol=tol)
    x = r.clone().requires_grad_()
    b = torch.randn_like(x)
    gradient, = torch.autograd.grad(residual_loss(op, x, b), x)
    weak_residual = x.detach() @ a.mT - b
    inverse = torch.linalg.solve(full_metric, weak_residual.reshape(-1, 10).mT).mT.reshape(r.shape)
    torch.testing.assert_close(gradient, inverse @ a.conj(), rtol=tol, atol=tol)
    assert op._riesz_factor.shape == g.shape and op._riesz_factor.dtype == g.dtype
    assert not isinstance(op.test_riesz, torch.Tensor)


@cuda
def test_compact_repeated_backward_gram_scale_matrix_and_vector_gradients():
    a, g = system(n=3)
    a.requires_grad_(); g.requires_grad_()
    eta2 = torch.tensor(3., dtype=g.dtype, device=g.device, requires_grad=True)
    op = DenseOperator(a, CompactRWGRiesz(g, eta2))
    x = torch.randn(2, 6, dtype=a.dtype, device=a.device, requires_grad=True)
    b = torch.randn_like(x)
    for _ in range(2):
        loss = residual_loss(op, x, b)
        actual = torch.autograd.grad(loss, (a, g, eta2, x))
        metric = torch.block_diag(g / eta2, g).to(a.dtype)
        r = x @ a.mT - b
        inv = torch.linalg.solve(metric, r.mT).mT
        oracle = .5 * (r.conj() * inv).real.sum()
        expected = torch.autograd.grad(oracle, (a, g, eta2, x))
        for left, right in zip(actual, expected):
            # Cholesky differentiates the symmetric Gram representation.
            if left.shape == g.shape:
                right = (right + right.mT) / 2
            torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-11)
    assert op._riesz_factor.grad_fn is None


@cuda
def test_metric_mutation_replacement_and_lu_cache_are_independent():
    a, g = system(n=3)
    scale = torch.tensor(4., dtype=g.dtype, device=g.device)
    metric = CompactRWGRiesz(g, scale)
    op = DenseOperator(a, metric)
    b = torch.randn(8, 6, dtype=a.dtype, device=a.device)
    expected = apply_test_riesz_inverse(op, b)
    _, first = solve_reference(op, b)
    factor = op._riesz_factor
    g.mul_(2)
    torch.testing.assert_close(apply_test_riesz_inverse(op, b), expected / 2)
    assert op._riesz_factor is not factor
    scale.mul_(3)
    scaled = expected / 2
    scaled[:, :3] *= 3
    torch.testing.assert_close(apply_test_riesz_inverse(op, b), scaled)
    _, repeated = solve_reference(op, b)
    assert first['converged'] and repeated['converged'] and repeated['cache_reused']
    metric.gram = g.clone()
    metric.impedance_squared = 12.
    torch.testing.assert_close(apply_test_riesz_inverse(op, b), scaled)
    op.test_riesz = torch.block_diag(g / 12, g).to(a.dtype)
    torch.testing.assert_close(apply_test_riesz_inverse(op, b), scaled)
    op.test_riesz = CompactRWGRiesz(g.clone(), 12.)
    torch.testing.assert_close(apply_test_riesz_inverse(op, b), scaled)
    a.mul_(2)
    _, changed = solve_reference(op, b)
    assert changed['converged'] and not changed['cache_reused']


@cuda
@pytest.mark.parametrize('change', ['matrix_shape', 'matrix_precision', 'scale_overflow', 'scale_underflow'])
def test_compact_revalidates_operator_binding_and_representable_scaling(monkeypatch, change):
    a, g = system(n=3)
    op = DenseOperator(a, CompactRWGRiesz(g, 3.))
    if change == 'matrix_shape':
        op.matrix = torch.eye(12, dtype=a.dtype, device=a.device)
    elif change == 'matrix_precision':
        op.matrix = a.to(torch.complex64)
    elif change == 'scale_overflow':
        a, g = system(torch.complex64, n=3)
        op = DenseOperator(a, CompactRWGRiesz(g, 3.))
        op.test_riesz.impedance_squared = 1e90
    else:
        a, g = system(torch.complex64, n=3)
        op = DenseOperator(a, CompactRWGRiesz(g, 3.))
        op.test_riesz.impedance_squared = 1e-90
    monkeypatch.setattr(torch, 'cholesky_solve', lambda *args, **kwargs: pytest.fail('invalid metric reached numerical inverse'))
    with pytest.raises(ValueError):
        apply_test_riesz_inverse(op, torch.ones(op.shape[0], dtype=op.dtype, device=op.device))


@cuda
@pytest.mark.parametrize('change', ['host', 'complex', 'wrong_shape', 'nan', 'indefinite',
    'asymmetric', 'zero_scale', 'nan_scale', 'complex_scale', 'host_scale', 'vector_scale', 'bool_scale'])
def test_replaced_compact_metric_rejects_invalid_data_before_factorization(monkeypatch, change):
    a, g = system(n=3)
    metric = CompactRWGRiesz(g, 2.)
    op = DenseOperator(a, metric)
    if change == 'host': metric.gram = g.cpu()
    elif change == 'complex': metric.gram = g.to(a.dtype)
    elif change == 'wrong_shape': metric.gram = g[:2, :2]
    elif change == 'nan': g[0, 0] = float('nan')
    elif change == 'indefinite': metric.gram = -g
    elif change == 'asymmetric': g[0, 1] += 3
    elif change == 'zero_scale': metric.impedance_squared = 0.
    elif change == 'nan_scale': metric.impedance_squared = float('nan')
    elif change == 'complex_scale': metric.impedance_squared = torch.tensor(2+1j, dtype=a.dtype, device=a.device)
    elif change == 'host_scale': metric.impedance_squared = torch.tensor(2., dtype=g.dtype)
    elif change == 'vector_scale': metric.impedance_squared = torch.ones(1, dtype=g.dtype, device=g.device)
    else: metric.impedance_squared = True
    calls = []
    original = torch.linalg.cholesky_ex
    def factor(value, *args, **kwargs):
        assert value.is_cuda and value.dtype == g.dtype
        calls.append(value.shape)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', factor)
    with pytest.raises(ValueError):
        apply_test_riesz_inverse(op, torch.ones(6, dtype=a.dtype, device=a.device))
    assert len(calls) == (1 if change == 'indefinite' else 0)


@cuda
def test_adjoint_never_conjugates_the_matrix(monkeypatch):
    a, g = system(n=5)
    a.requires_grad_()
    op = DenseOperator(a, CompactRWGRiesz(g, 3.))
    y = torch.randn(8, 10, dtype=a.dtype, device=a.device, requires_grad=True)
    expected = y @ a.conj()
    original = torch.Tensor.conj
    def checked(value):
        assert value is not a
        return original(value)
    monkeypatch.setattr(torch.Tensor, 'conj', checked)
    actual = apply_A_adjoint(op, y)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert not actual.is_conj()
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (a, y))
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (a, y))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


@cuda
def test_public_compact_path_eight_rhs_fields_and_persistence(tmp_path, monkeypatch):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.01))],
        sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=4))
    dense_prepared = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()
    dense_handle = dense_prepared.solver.handle
    dense_result = dense_prepared.run()
    compact_config = replace(config, riesz_storage='compact_rwg_v1')
    factors = []
    original_factor = torch.linalg.cholesky_ex
    def factor(value, *args, **kwargs):
        factors.append((tuple(value.shape), value.dtype))
        assert value.shape == (30, 30) and value.dtype == torch.float64
        return original_factor(value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'block_diag', lambda *args: pytest.fail('compact path materialized block metric'))
        patch.setattr(torch.linalg, 'cholesky_ex', factor)
        prepared = mw.Simulation.mom(scene, frequency=3e8, config=compact_config).prepare()
        result = prepared.run()
    compact_handle = prepared.solver.handle
    assert factors == [((30, 30), torch.float64)]
    assert isinstance(dense_handle.test_riesz, torch.Tensor)
    assert isinstance(compact_handle.test_riesz, BlockCompactRWGRiesz)
    torch.testing.assert_close(compact_handle.matrix, dense_handle.matrix, rtol=2e-13, atol=2e-13)
    sources = [mw.PlaneWave(direction=d, polarization=p, source_time=mw.CW(frequency=3e8, phase=.17*i))
        for i, (d, p) in enumerate([((0,0,1),(1,0,0)), ((0,0,1),(0,1,0)),
            ((0,0,-1),(1,0,0)), ((0,0,-1),(0,1,0)), ((1,0,0),(0,1,0)),
            ((1,0,0),(0,0,1)), ((0,1,0),(1,0,0)), ((0,1,0),(0,0,1))])]
    rhs = torch.stack([build_rhs(compact_handle.basis, compact_handle.exterior, [s], order=3) for s in sources])
    reference, _ = solve_reference(dense_handle, rhs)
    solved, log = solve_reference(compact_handle, rhs)
    assert log['converged'] and log['rhs_count'] == 8 and log['cache_reused']
    torch.testing.assert_close(solved, reference, rtol=2e-11, atol=2e-12)
    actual_residual = torch.linalg.vector_norm(solved @ compact_handle.matrix.mT - rhs, dim=-1) / torch.linalg.vector_norm(rhs, dim=-1)
    assert bool((actual_residual < 1e-8).all())
    points = torch.tensor([[.3,.2,.4]], dtype=torch.float64, device='cuda')
    for left, right in zip(result.query_fields(points, region_id='exterior', field_kind='total'),
                           dense_result.query_fields(points, region_id='exterior', field_kind='total')):
        torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-12)
    path = tmp_path / 'compact.pt'
    result.save(path)
    restored = mw.Result.load(path, scene=scene)
    assert restored.surface_currents.config.riesz_storage == 'compact_rwg_v1'
    assert restored.surface_currents.handle.metadata['riesz_storage'] == 'block_compact_rwg_v1'
    torch.testing.assert_close(restored.surface_currents.scattered_coefficients, result.surface_currents.scattered_coefficients)


@cuda
def test_storage_selection_cannot_change_after_prepare():
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2))], sources=[mw.PlaneWave()])
    simulation = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(subdivisions=0,
        integration=IntegrationConfig(order=2, singular_order=3), riesz_storage='compact_rwg_v1'))
    prepared = simulation.prepare()
    simulation.config = replace(simulation.config, riesz_storage='dense')
    with pytest.raises(ValueError, match='configuration changed'):
        prepared.run()
