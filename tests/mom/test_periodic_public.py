"""Public scalar periodic Scene/Simulation/Result identity and exact restore."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.runtime import SurfaceData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_public_periodic_solution_roundtrip_and_identity_tamper(tmp_path, monkeypatch):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.07), mw.Material(eps_r=2.))],
                     sources=[mw.PlaneWave()])
    periodic = mw.MoMPeriodicConfig(vectors_m=((1., 0., 0.),), reciprocal_shells=4, quadrature_order=128)
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2, singular_order=3),
        query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1', periodic=periodic)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    assert result.stats()['converged']
    native = result.surface_currents.snapshot()
    assert native['kernel_id'] == 'quasiperiodic_ewald_1d'
    assert native['operator_metadata']['physical_qualification'] is False
    points = torch.tensor([[.2, .2, .3], [1.2, .2, .3]], device='cuda', dtype=torch.float64)
    fields = result.query_fields(points, region_id='exterior', field_kind='total')
    for value in fields:
        torch.testing.assert_close(value[:1], value[1:], rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match='diffraction'):
        result.far_field(points[:1])
    path = tmp_path/'periodic.pt'
    result.save(path)
    payload = torch.load(path, weights_only=False)
    assert payload['surface_data']['schema_version'] == 7
    assert payload['surface_data']['periodic_config'] == periodic.payload()
    import witwin.maxwell.mom.periodic_runtime as runtime
    def forbid(*args, **kwargs):
        raise AssertionError('Snapshot restore must not assemble the periodic matrix')
    monkeypatch.setattr(runtime, 'assemble_periodic_system', forbid)
    restored = mw.Result.load(path, scene=scene)
    assert torch.equal(restored.surface_currents.scattered_coefficients, result.surface_currents.scattered_coefficients)
    for actual, expected in zip(restored.query_fields(points, region_id='exterior', field_kind='total'), fields):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    orders = torch.zeros((1, 1), device='cuda', dtype=torch.int64)
    directions = points.new_tensor([[0., 1., 0.]])
    before = result.cylindrical_orders(orders, directions)
    after = restored.cylindrical_orders(orders, directions)
    assert before['physical_qualification'] is False
    torch.testing.assert_close(before['electric'], after['electric'], rtol=0, atol=0)
    corrupted = copy.deepcopy(payload['surface_data'])
    corrupted['operator_metadata']['physical_qualification'] = True
    with pytest.raises(ValueError, match='identity'):
        SurfaceData.restore(corrupted, scene, 3e8)
    for key, value in (('eta_per_m', 4.), ('real_shells', 4), ('bloch_rad_per_m', [.1, 0., 0.])):
        corrupted = copy.deepcopy(payload['surface_data'])
        corrupted['periodic_config'][key] = value
        with pytest.raises(ValueError, match='identity|sector|match'):
            SurfaceData.restore(corrupted, scene, 3e8)
    for physical in (result, restored):
        data = physical.surface_currents
        original_config = data.config
        data.config = replace(original_config, periodic=None)
        with pytest.raises(ValueError, match='configuration'):
            physical.query_fields(points, region_id='exterior', field_kind='total')
        with pytest.raises(ValueError, match='configuration'):
            physical.far_field(points[:1])
        data.config = original_config
        local = data.handle.body_handles[0]
        interior = local.interior
        local.interior = data.handle.exterior
        with pytest.raises(ValueError, match='local region binding'):
            physical.query_fields(points, region_id='exterior', field_kind='total')
        local.interior = interior
    corrupted = copy.deepcopy(payload['surface_data'])
    corrupted['kernel_id'] = 'free_space_helmholtz'
    with pytest.raises(ValueError, match='identity'):
        SurfaceData.restore(corrupted, scene, 3e8)


def test_periodic_declaration_is_owned_versioned_and_rejects_invalid_values():
    vectors = [[1., 0., 0.]]
    periodic = mw.MoMPeriodicConfig(vectors, [0., 0., 0.])
    vectors[0][0] = 4.
    assert periodic.vectors_m == ((1., 0., 0.),)
    assert mw.MoMPeriodicConfig.restore(periodic.payload()) == periodic
    for kwargs in ({'real_shells': True}, {'eta_per_m': float('nan')}, {'bloch_rad_per_m': (True, 0., 0.)}):
        with pytest.raises(ValueError):
            mw.MoMPeriodicConfig(((1., 0., 0.),), **kwargs)
    with pytest.raises(ValueError, match='independent'):
        mw.MoMPeriodicConfig(((1., 0., 0.), (2., 0., 0.))).compile(device='cuda')
    with pytest.raises(TypeError, match='periodic'):
        mw.MoMConfig(periodic={})


def test_public_two_dimensional_diffraction_and_translated_point_monitor():
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.07), mw.Material(eps_r=2.))],
        sources=[mw.PlaneWave()],
        monitors=[mw.PointMonitor(name='translated', position=(1., 0., 0.), fields=('Ex', 'Hy'))])
    periodic = mw.MoMPeriodicConfig(((1., 0., 0.), (0., 1., 0.)), reciprocal_shells=4, quadrature_order=128)
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2, singular_order=3),
        query_config=FieldQueryConfig(adaptive=False), periodic=periodic)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    orders = torch.zeros((1, 2), device='cuda', dtype=torch.int64)
    spectrum = result.rayleigh_orders(orders, side=1, plane_offset=.2)
    assert spectrum['physical_qualification'] is False
    assert torch.isfinite(spectrum['electric']).all()
    # The monitor lies inside the translated copy of the body, not exterior.
    assert result.monitor('translated')['region_id'] == 1
