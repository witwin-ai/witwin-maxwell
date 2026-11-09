"""Public Ewald FMM solve, field identity and decode-only exact storage."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.runtime import SurfaceData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_public_ewald_fmm_dense_reference_and_exact_restore(tmp_path, monkeypatch):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01), mw.Material(eps_r=2., mu_r=1.2))],
                     sources=[mw.PlaneWave()])
    periodic = mw.MoMPeriodicConfig(((1., 0., 0.),), eta_per_m=2., real_shells=2,
                                   reciprocal_shells=2, quadrature_order=32)
    dense_config = mw.MoMConfig(subdivisions=0, periodic=periodic,
        integration=IntegrationConfig(order=4, singular_order=3),
        query_config=FieldQueryConfig(adaptive=False))
    config = replace(dense_config, operator_backend='fmm', riesz_storage='local_rwg_v1',
        fmm=mw.FMMConfig(order=6, table_storage='compact', separation_ratio=.25),
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', fallback='none', rtol=1e-12, atol=0., maxiter=240, restart=120))
    dense = mw.Simulation.mom(scene, frequency=9e7, config=dense_config).run()
    prepared = mw.Simulation.mom(scene, frequency=9e7, config=config).prepare()
    result = prepared.run()
    assert result.stats()['converged'], result.stats()
    current = result.surface_currents.scattered_coefficients
    reference = dense.surface_currents.scattered_coefficients
    torch.testing.assert_close(current, reference, rtol=1e-8, atol=1e-12)
    points = torch.tensor([[.1, .1, .15], [1.1, .1, .15]], device='cuda', dtype=torch.float64)
    fields = result.query_fields(points, region_id='exterior', field_kind='total')
    for actual, expected in zip(fields, dense.query_fields(points, region_id='exterior', field_kind='total')):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)
        torch.testing.assert_close(actual[:1], actual[1:], rtol=1e-11, atol=1e-12)
    path = tmp_path/'ewald_fmm.pt'
    result.save(path)
    payload = result.surface_currents.snapshot()
    assert payload['schema_version'] == 12 and payload['operator_backend'] == 'fmm'
    assert payload['operator_metadata']['physical_qualification'] is False
    assert payload['operator_metadata']['operator_identity_payload']['backend'] == 'fmm'
    import witwin.maxwell.mom.periodic_runtime as runtime
    def forbid(*args, **kwargs):
        raise AssertionError('Restore must not construct an operator')
    monkeypatch.setattr(runtime.PeriodicMultipoleSystemOperator, '__init__', forbid)
    monkeypatch.setattr(runtime, 'assemble_periodic_system', forbid)
    restored = mw.Result.load(path, scene=scene)
    assert torch.equal(restored.surface_currents.scattered_coefficients, current)
    for actual, expected in zip(restored.query_fields(points, region_id='exterior', field_kind='total'), fields):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, value in [('schema_version', 11), ('operator_backend', 'dense')]:
        corrupt = copy.deepcopy(payload)
        corrupt[key] = value
        with pytest.raises(ValueError):
            SurfaceData.restore(corrupt, scene, 9e7)
    corrupt = copy.deepcopy(payload)
    corrupt['fmm_config']['parameters']['order'] = 7
    with pytest.raises(ValueError, match='identity'):
        SurfaceData.restore(corrupt, scene, 9e7)
    handle = result.surface_currents.handle
    handle.periodic_kernel = replace(handle.periodic_kernel, eta=3.)
    with pytest.raises(ValueError, match='changed'):
        handle.apply(current)
    print({'public_ewald_fmm_current_relative_error': float((current-reference).norm()/reference.norm())})


def test_public_two_dimensional_multibody_bloch_solution():
    from witwin.maxwell.mom.physics import compile_medium
    frequency = 9e7
    exterior = compile_medium(mw.Material(), frequency)
    direction = (.6, 0., .8)
    beta = (exterior.k.real*direction[0], 0., 0.)
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
        mw.Material(eps_r=eps, mu_r=mu)) for x, eps, mu in ((-.05, 2., 1.2), (.05, 3., 1.1))],
        sources=[mw.PlaneWave(direction=direction, polarization=(0., 1., 0.))])
    periodic = mw.MoMPeriodicConfig(((1., .2, 0.), (0., 1., 0.)), beta,
        eta_per_m=2., real_shells=2, reciprocal_shells=2, quadrature_order=32)
    base = mw.MoMConfig(subdivisions=0, periodic=periodic,
        integration=IntegrationConfig(order=4, singular_order=3),
        query_config=FieldQueryConfig(adaptive=False))
    config = replace(base, operator_backend='fmm', riesz_storage='local_rwg_v1',
        fmm=mw.FMMConfig(order=8, table_storage='compact', separation_ratio=.25),
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
    reference = mw.Simulation.mom(scene, frequency=frequency, config=base).run()
    result = mw.Simulation.mom(scene, frequency=frequency, config=config).run()
    assert result.stats()['converged'], result.stats()
    current = result.surface_currents.scattered_coefficients
    original = reference.surface_currents.handle
    rhs = original.build_rhs()
    residual = float((original.apply(current)-rhs).norm()/rhs.norm())
    error = float((current-reference.surface_currents.scattered_coefficients).norm()/
                  reference.surface_currents.scattered_coefficients.norm())
    assert residual < 1e-8 and error < 1e-8
    point = torch.tensor([[.15, .12, .2]], device='cuda', dtype=torch.float64)
    lattice = result.surface_currents.handle.periodic_kernel.lattice
    translations = lattice.vectors
    points = torch.cat((point, point+translations))
    phase = torch.exp(1j*(translations@lattice.bloch))
    for actual, expected in zip(result.query_fields(points, region_id='exterior', field_kind='total'),
                                reference.query_fields(points, region_id='exterior', field_kind='total')):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)
        torch.testing.assert_close(actual[1:], phase[:, None]*actual[:1], rtol=1e-11, atol=1e-12)
    restored = SurfaceData.restore(result.surface_currents.snapshot(), scene, frequency)
    assert len(restored.bodies) == 2
    torch.testing.assert_close(restored.scattered_coefficients, current, rtol=0, atol=0)
    incompatible = replace(config, periodic=replace(periodic, bloch_rad_per_m=(0., 0., 0.)))
    with pytest.raises(ValueError, match='Bloch'):
        mw.Simulation.mom(scene, frequency=frequency, config=incompatible).prepare()
    print({'public_2d_multibody_original_residual': residual, 'current_relative_error': error})


def test_public_three_dimensional_plane_wave_rejects_resolvent_pole():
    from witwin.maxwell.mom.physics import compile_medium
    frequency = 9e7
    k = compile_medium(mw.Material(), frequency).k.real
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01), mw.Material(eps_r=2.))],
                     sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, operator_backend='fmm', riesz_storage='local_rwg_v1',
        stopping=StoppingRule(preconditioner='rwg_diagonal'),
        periodic=mw.MoMPeriodicConfig(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                                     (0., 0., k)))
    with pytest.raises(ValueError, match='spectral pole'):
        mw.Simulation.mom(scene, frequency=frequency, config=config).prepare()
