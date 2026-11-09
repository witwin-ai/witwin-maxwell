"""Overflowed field error metrics must never count as convergence evidence."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def policy(kind, *, adaptive=True):
    return FieldQueryConfig(relative_tolerance=1e308 if kind=='threshold' else 2.,
        absolute_tolerance=1e308 if kind=='threshold' else 1., max_order=2, adaptive=adaptive)


def values(points, order, kind):
    electric = torch.zeros_like(points, dtype=torch.complex128)
    electric[:, 0] = 1e200 if kind=='norm' else float(order)
    assert torch.isfinite(electric).all()
    return electric, torch.zeros_like(electric)


@pytest.mark.parametrize('kind', ('threshold', 'norm', 'finite'))
@pytest.mark.parametrize('tensor', (False, True))
def test_free_field_admission_requires_finite_error_norm_and_threshold(kind, tensor, monkeypatch):
    points = torch.zeros((1, 3), device='cuda', dtype=torch.float64)
    config = policy(kind)
    if tensor:
        from witwin.maxwell.mom.anisotropic_fields import _adaptive_tensor_fields, TensorFieldConvergenceError
        def evaluate(order):
            return *values(points, order, kind), {'order': order}
        def run():
            return _adaptive_tensor_fields(evaluate, points, config, 1)
        error = TensorFieldConvergenceError
    else:
        from witwin.maxwell.mom import fields
        handle = SimpleNamespace(basis=SimpleNamespace(mesh=SimpleNamespace(device=points.device)),
                                 integration=IntegrationConfig(order=1, singular_order=1), metadata={})
        monkeypatch.setattr(fields, '_potentials',
            lambda *args, order, **kwargs: values(points, order, kind))
        def run():
            result = fields._evaluate_adaptive(handle, points.new_zeros(2, dtype=torch.complex128),
                points, SimpleNamespace(eta=1.), torch.tensor([-1], device='cuda'), config)
            return *result, handle.last_field_query_log
        error = RuntimeError
    if kind=='finite':
        _, _, receipt = run()
        assert receipt['converged']
    else:
        with pytest.raises(error):
            run()
        if not tensor:
            assert not handle.last_field_query_log['converged']


def periodic_handle(tensor):
    from witwin.maxwell.mom.runtime import SystemDecodeHandle
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    from witwin.maxwell.mom.physics import compile_medium
    from test_generalized_system_blocks import basis_at
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = (TensorMedium(3e8, torch.diag(eye.new_tensor([2., 3., 4.])), eye) if tensor
              else compile_medium(mw.Material(eps_r=2.), 3e8))
    basis = basis_at(0.)
    system = SurfaceSystem.create([(0, 'body', basis, medium)])
    config = mw.MoMConfig(integration=IntegrationConfig(order=1, singular_order=1),
        dyadic_integration=mw.DyadicIntegrationConfig() if tensor else None,
        periodic=mw.MoMPeriodicConfig(((2., 0., 0.),)))
    handle = SystemDecodeHandle(system, compile_medium(mw.Material(), 3e8), [], config, {})
    return handle, config.periodic.compile(device='cuda', dtype=torch.complex128)


@pytest.mark.parametrize('kind', ('threshold', 'norm', 'finite'))
@pytest.mark.parametrize('tensor', (False, True))
def test_periodic_surface_admission_requires_finite_metrics(kind, tensor, monkeypatch):
    from witwin.maxwell.mom import periodic_fields
    handle, kernel = periodic_handle(tensor)
    point = handle.system.bodies[0].mesh.vertices.mean(0, keepdim=True)
    monkeypatch.setattr(periodic_fields, 'decode_system',
        lambda local, *args, **kwargs: values(point, local.integration.order, kind))
    coefficients = point.new_zeros(2*handle.system.total_edges, dtype=torch.complex128)
    def run():
        return periodic_fields.decode_periodic(handle, coefficients, point, 1, 'total',
                                               kernel=kernel, query_config=policy(kind))
    if kind=='finite':
        run()
        assert handle.last_periodic_field_log['converged']
    else:
        with pytest.raises(periodic_fields.PeriodicFieldConvergenceError) as failure:
            run()
        assert not failure.value.receipt['converged']
        assert not failure.value.receipt['checks'][-1]['passed']


@pytest.mark.parametrize('kind', ('threshold', 'norm', 'finite'))
def test_periodic_ewald_admission_requires_finite_complete_and_correction_norms(kind, monkeypatch):
    from witwin.maxwell.mom import periodic_fields
    handle, kernel = periodic_handle(False)
    point = kernel.lattice.vectors.new_tensor([[.2, .3, .4]])
    monkeypatch.setattr(periodic_fields, 'decode_system',
        lambda *args, **kwargs: values(point, 1, kind))
    def correction(local, coefficients, wrapped, current, order):
        depth = current.real_shells+current.reciprocal_shells+current.quadrature_order
        return values(point, depth, kind)
    monkeypatch.setattr(periodic_fields, 'regular_surface_fields', correction)
    coefficients = point.new_zeros(2*handle.system.total_edges, dtype=torch.complex128)
    def run():
        return periodic_fields.decode_periodic(handle, coefficients, point, 0, 'scattered',
            kernel=kernel, query_config=policy(kind, adaptive=False), ewald_check=True)
    if kind=='finite':
        run()
        assert handle.last_periodic_field_log['ewald_truncation_verified']
        assert not handle.last_periodic_field_log['converged']  # Surface error remains unestimated.
    else:
        with pytest.raises(periodic_fields.PeriodicFieldConvergenceError) as failure:
            run()
        assert failure.value.receipt['failure_reason']=='independent_ewald_estimate_failed'
        assert not failure.value.receipt['ewald_truncation_verified']


@pytest.mark.parametrize('overflowed_reference', ('complete_field', 'correction'))
def test_periodic_ewald_minimum_cannot_hide_one_overflowed_norm(monkeypatch, overflowed_reference):
    from witwin.maxwell.mom import periodic_fields
    handle, kernel = periodic_handle(False)
    point = kernel.lattice.vectors.new_tensor([[.2, .3, .4]])
    def field(value):
        electric = torch.zeros_like(point, dtype=torch.complex128)
        electric[:, 0] = value
        return electric, torch.zeros_like(electric)
    primary = 1e200 if overflowed_reference=='complete_field' else -1e200
    correction = 2. if overflowed_reference=='complete_field' else 1e200
    monkeypatch.setattr(periodic_fields, 'decode_system', lambda *args, **kwargs: field(primary))
    monkeypatch.setattr(periodic_fields, 'regular_surface_fields', lambda *args, **kwargs: field(correction))
    coefficients = point.new_zeros(2*handle.system.total_edges, dtype=torch.complex128)
    with pytest.raises(periodic_fields.PeriodicFieldConvergenceError) as failure:
        periodic_fields.decode_periodic(handle, coefficients, point, 0, 'scattered',
            kernel=kernel, query_config=policy('finite', adaptive=False), ewald_check=True)
    assert failure.value.receipt['failure_reason']=='independent_ewald_estimate_failed'
    for check in failure.value.receipt['ewald_checks']:
        assert check['absolute_difference']==[0.]
        assert all(math_value < float('inf') for math_value in check['threshold'])
        assert not check['passed']
