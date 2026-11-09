"""Periodic dense adapter and operator/RHS cache identity separation."""
from dataclasses import replace
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_runtime import PeriodicSystemOperator, periodic_operator_identity
from witwin.maxwell.mom.linalg import solve_reference, StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_periodic_operator_shared_solver_fields_and_kernel_binding():
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = original.system.bodies[0].mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_zeros(3))
    kernel = EwaldHelmholtzKernel(lattice, 3., reciprocal_shells=4, quadrature_order=128)
    config = mw.MoMConfig(integration=original.integration, riesz_storage='compact_rwg_v1',
                          query_config=FieldQueryConfig(adaptive=False))
    operator = PeriodicSystemOperator(original.system, original.exterior,
        iter(original.sources), config, kernel=kernel)
    rhs = operator.build_rhs()
    assert rhs.norm() > 0
    coefficients, log = solve_reference(operator, rhs, StoppingRule(method='direct'))
    assert log['converged']
    error = (operator.apply(coefficients)-rhs).norm()/rhs.norm()
    assert error < 1e-11
    assert operator.metadata['kernel_id'] == 'quasiperiodic_ewald_1d'
    assert not operator.metadata['physical_qualification']
    assert all(body.kernel is FREE_SPACE for body in operator.body_handles)
    points = sample.new_tensor([[.1, .1, .15], [1.1, .1, .15]])
    fields = operator.query_fields(coefficients, points, 0, 'scattered')
    assert not operator.last_field_query_log['adaptive']
    for field in fields:
        assert torch.isfinite(field).all() and field.norm() > 0
        torch.testing.assert_close(field[:1], field[1:], rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match='diffraction'):
        operator.far_field(points)
    operator.periodic_kernel = replace(kernel, eta=4.)
    with pytest.raises(ValueError, match='changed'):
        operator.apply(coefficients)
    operator.periodic_kernel = kernel
    operator.sources = (mw.PlaneWave(source_time=mw.CW(frequency=original.exterior.frequency, amplitude=2.)),)
    with pytest.raises(ValueError, match='source binding'):
        operator.build_rhs()
    print({'periodic_adapter_algebraic_relative_residual': float(error)})


def test_periodic_cache_identity_includes_bloch_lattice_and_kernel_tolerances():
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = original.system.bodies[0].mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_zeros(3))
    kernel = EwaldHelmholtzKernel(lattice, 3., reciprocal_shells=4, quadrature_order=128)
    config = mw.MoMConfig(integration=original.integration)
    identity, payload = periodic_operator_identity(original.system, original.exterior, config, kernel)
    assert identity == periodic_operator_identity(original.system, original.exterior, config, kernel)[0]
    alternatives = [replace(kernel, eta=4.), replace(kernel, reciprocal_shells=5),
                    replace(kernel, pole_relative_tolerance=1e-8),
                    replace(kernel, lattice=LatticeSpec(lattice.vectors, sample.new_tensor([.2, 0., 0.]))),
                    replace(kernel, lattice=LatticeSpec(1.2*lattice.vectors, lattice.bloch))]
    for other in alternatives:
        assert identity != periodic_operator_identity(original.system, original.exterior, config, other)[0]
    assert 'incident_carrier_identity' not in payload
    assert identity != periodic_operator_identity(original.system, original.exterior,
        replace(config, integration=replace(config.integration, order=5)), kernel)[0]


def test_amplitude_changes_share_operator_identity_but_not_rhs_identity():
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = original.system.bodies[0].mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_zeros(3))
    kernel = EwaldHelmholtzKernel(lattice, 3., reciprocal_shells=4, quadrature_order=128)
    config = mw.MoMConfig(integration=original.integration, riesz_storage='compact_rwg_v1')
    operators = [PeriodicSystemOperator(original.system, original.exterior,
        [mw.PlaneWave(source_time=mw.CW(frequency=original.exterior.frequency, amplitude=amplitude))], config, kernel=kernel) for amplitude in (1., 2.)]
    rhs = [operator.build_rhs() for operator in operators]
    assert operators[0].metadata['operator_identity'] == operators[1].metadata['operator_identity']
    assert operators[0].metadata['rhs_identity'] != operators[1].metadata['rhs_identity']
    torch.testing.assert_close(rhs[1], 2*rhs[0], rtol=1e-12, atol=1e-14)
    coefficients, log = solve_reference(operators[0], torch.stack(rhs), StoppingRule(method='direct'))
    assert log['converged']
    torch.testing.assert_close(coefficients[1], 2*coefficients[0], rtol=1e-12, atol=1e-13)
