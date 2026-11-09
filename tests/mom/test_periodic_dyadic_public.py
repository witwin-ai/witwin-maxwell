"""Public tensor-exterior routing and exact recovery with manufactured matrices.

These checks exercise real declarations, compilers, solve protocol and payloads.
Manufactured matrix assembly is not a physical scattering accuracy control.
"""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.runtime import SurfaceData
from test_tensor_carrier_persistence import carrier
from test_generalized_system_blocks import basis_at

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(dimension=2):
    original = carrier(False)
    material = mw.Material(
        epsilon_tensor=mw.ComplexTensor3x3(original.medium.epsilon_r.detach().cpu().tolist()),
        mu_tensor=mw.ComplexTensor3x3(original.medium.mu_r.detach().cpu().tolist(), reciprocal=False))
    bodies = []
    for x in (-.2, .2):
        basis = basis_at(x)
        bodies.append(mw.Structure(mw.Mesh(basis.mesh.vertices, basis.mesh.faces,
            recenter=False, fill_mode='solid'), material))
    source = mw.MaxwellPlaneModes(original.medium.frequency, original.wavevectors,
        original.electric, original.magnetic, original.origin)
    beta = [float(original.wavevectors[0, 0].real),
            float(original.wavevectors[0, 1].real) if dimension == 2 else 0., 0.]
    periodic = mw.MoMDyadicPeriodicConfig(
        vectors_m=((1., 0., 0.), (0., 1., 0.))[:dimension], bloch_rad_per_m=beta,
        shells=4, max_shells=8, height_m=.1, max_halvings=3, absolute_tolerance=.02,
        normal=(0., 0., 1.) if dimension == 1 else None, cache_max_bytes=0)
    config = mw.MoMConfig(integration=IntegrationConfig(order=1, singular_order=2),
        query_config=mw.FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1',
        periodic=periodic, dyadic_integration=mw.DyadicIntegrationConfig(), dyadic_cache_max_bytes=2)
    return mw.Scene(background=material, structures=bodies, sources=[source]), config


@pytest.mark.parametrize('dimension', [1, 2])
def test_public_dyadic_solve_incident_and_schema18_restore(dimension, monkeypatch, tmp_path):
    import witwin.maxwell.mom.periodic_dyadic_operator as assembly
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicRegularKernel
    scene, config = problem(dimension)
    calls = []
    def manufactured(system, *, reference, config, exterior_kernel, interior_kernels):
        assert isinstance(reference, QuasiperiodicDyadicRegularKernel)
        assert exterior_kernel.medium is reference.medium
        assert all(interior_kernels[body.region_id].medium is body.interior for body in system.bodies)
        calls.append(reference.kernel_id)
        return torch.eye(2*system.total_edges, device=system.device,
                         dtype=reference.medium.epsilon_r.dtype), {'physical_qualification': False}
    monkeypatch.setattr(assembly, 'assemble_periodic_dyadic_system', manufactured)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    assert calls == ['checked_quasiperiodic_dyadic_regular_v1']
    assert torch.count_nonzero(result.surface_currents.scattered_coefficients) == 0
    assert result.stats()['converged']
    points = torch.tensor([[.1, .2, .3], [1.1, .2, .3]], device='cuda', dtype=torch.float64)
    fields = result.query_fields(points, region_id=0, field_kind='incident')
    phase = torch.exp(1j*points.new_tensor(config.periodic.bloch_rad_per_m[0]))
    for field in fields:
        torch.testing.assert_close(field[1], phase*field[0], rtol=1e-12, atol=1e-13)
    payload = result.surface_currents.snapshot()
    assert payload['schema_version'] == 18
    assert payload['kernel_id'] == 'checked_quasiperiodic_dyadic_regular_v1'
    assert payload['operator_metadata']['physical_qualification'] is False
    path = tmp_path/'dyadic.pt'
    result.save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Restore must neither assemble nor solve')
    monkeypatch.setattr(assembly, 'assemble_periodic_dyadic_system', forbidden)
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored = mw.Result.load(path, scene=scene)
        assert restored.surface_currents.config == config
        assert torch.equal(restored.surface_currents.scattered_coefficients,
                           result.surface_currents.scattered_coefficients)
        for actual, expected in zip(restored.query_fields(points, region_id=0, field_kind='incident'), fields):
            assert torch.equal(actual, expected)
        for mutation in ('schema', 'regular', 'carrier', 'exterior', 'decoder', 'operator'):
            damaged = copy.deepcopy(payload)
            if mutation == 'schema': damaged['schema_version'] = 16
            elif mutation == 'regular': damaged['regular_kernel']['regular']['height'] *= 2
            elif mutation == 'carrier': damaged['incident_carrier']['origin_m'][0] += .1
            elif mutation == 'exterior': damaged['exterior']['epsilon_r_real_imag'][0][0][0] += .1
            elif mutation == 'decoder': damaged['decoder_version'] = 'periodic_tensor_analytic_lifted_fields_v1'
            else: damaged['operator_metadata']['operator_identity'] = 'changed'
            with pytest.raises(ValueError):
                SurfaceData.restore(damaged, scene, 3e8)
    with pytest.raises(NotImplementedError, match='normalization'):
        restored.surface_currents.rayleigh_orders([[0, 0]], side='+', plane_offset=.3)


def test_public_dyadic_complete_field_refinement_and_component_failure_logs(monkeypatch):
    """Manufactured fields check API composition and failure ownership only."""
    import witwin.maxwell.mom.periodic_dyadic_operator as assembly
    import witwin.maxwell.mom.periodic_dyadic_fields as fields
    from witwin.maxwell.mom.anisotropic_fields import TensorFieldConvergenceError
    from witwin.maxwell.mom.periodic_dyadic import PeriodicDyadicConvergenceError
    from witwin.maxwell.mom.dyadic_green import DyadicConvergenceError
    scene, config = problem()
    def matrix(system, *, reference, **kwargs):
        return torch.eye(2*system.total_edges, device=system.device,
                         dtype=reference.medium.epsilon_r.dtype), {'physical_qualification': False}
    monkeypatch.setattr(assembly, 'assemble_periodic_dyadic_system', matrix)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    data = result.surface_currents
    points = torch.tensor([[.1, .2, .3]], device='cuda', dtype=torch.float64)
    orders = []
    def manufactured(system, coefficients, points, *, order, **kwargs):
        orders.append(order)
        electric = torch.ones_like(points, dtype=coefficients.dtype)*order
        magnetic = electric/1000
        return (electric, magnetic), {'manufactured': True, 'surface_order': order}
    monkeypatch.setattr(fields, 'exterior_dyadic_surface_fields', manufactured)
    scattered = result.query_fields(points, region_id=0, field_kind='scattered')
    total = result.query_fields(points, region_id=0, field_kind='total')
    incident = result.query_fields(points, region_id=0, field_kind='incident')
    for a, b, c in zip(total, scattered, incident):
        torch.testing.assert_close(a, b+c, rtol=0, atol=0)
    assert data.handle.metadata['last_field_query']['representation'] == 'analytic_tensor_incident_modes'
    orders.clear()
    with pytest.raises(TensorFieldConvergenceError) as error:
        result.query_fields(points, region_id=0, field_kind='scattered',
            query_config=mw.FieldQueryConfig(max_order=4, relative_tolerance=0., absolute_tolerance=1e-9))
    assert orders == [2, 4]
    assert data.handle.last_field_query_log is error.value.receipt
    assert data.handle.metadata['last_field_query']['failure_reason'] == 'surface_quadrature_order_budget_exhausted'
    for error_type in (PeriodicDyadicConvergenceError, DyadicConvergenceError):
        failure = {'failed_axis': 'test_component', 'failure_reason': 'test_budget_exhausted'}
        def failed(*args, **kwargs):
            if error_type is DyadicConvergenceError:
                raise error_type('Injected primary integration failure', failure)
            raise error_type(failure)
        monkeypatch.setattr(fields, 'exterior_dyadic_surface_fields', failed)
        with pytest.raises(error_type):
            result.query_fields(points, region_id=0, field_kind='total')
        receipt = data.handle.metadata['last_field_query']
        assert not receipt['converged'] and not receipt['physical_qualification']
        assert receipt['component_failure'] == failure
        assert receipt['component_failure'] is not failure
        assert receipt['field_kind'] == 'total' and receipt['region_id'] == 0
