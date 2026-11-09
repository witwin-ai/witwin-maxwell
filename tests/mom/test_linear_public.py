"""Public linear volume degeneration and decode-only native restoration."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.volume_h_snapshot import _digest
from test_linear_volume_state import system_and_volumes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_public_volume_degenerates_exactly_and_restores_without_solve(tmp_path, monkeypatch):
    _, volumes = system_and_volumes(torch.float64)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(v.boundary.vertices, v.boundary.faces,
        recenter=False, fill_mode='solid'), mw.Material(eps_r=2.+i)) for i,v in enumerate(volumes)],
        sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8))])
    config = mw.MoMConfig(integration=IntegrationConfig(order=2, singular_order=2),
        query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1')
    reference = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    simulation = mw.Simulation.mom(scene, frequency=3e8,
        config=replace(config, linear_coupled=mw.LinearCoupledConfig(volumes)))
    prepared = simulation.prepare()
    # The zero-source contract compares the same regional operator. Independent
    # CUDA assembly can differ at roundoff even with identical declarations.
    independent = reference
    reference = prepared.solver.surface_runtime.run()
    # Compare repeatability in the vector norm used by the diagnostic; tiny
    # individual components can have large relative cancellation errors.
    expected_coefficients = independent.surface_currents.scattered_coefficients
    repeat_error = (reference.surface_currents.scattered_coefficients-expected_coefficients).norm()
    assert float(repeat_error/expected_coefficients.norm()) <= 64*torch.finfo(config.dtype).eps
    result = prepared.run()
    assert result.stats()['converged']
    actual = result.surface_currents
    assert torch.equal(actual.scattered_coefficients, reference.surface_currents.scattered_coefficients)
    assert bool(actual.scattered_coefficients.any())
    points = volumes[0].vertices.new_tensor([[.2, -.1, .1]])
    expected = reference.query_fields(points, region_id=0, field_kind='total')
    for a,b in zip(result.query_fields(points, region_id=0, field_kind='total'), expected):
        assert torch.equal(a,b)
    path = tmp_path/'linear.pt'
    result.save(path)
    payload = torch.load(path, weights_only=False)
    assert payload['surface_data']['schema_version'] == 20
    assert payload['metadata']['formulation_id'] == 'coupled_sie_vie_linear'
    def forbidden(*args, **kwargs): raise AssertionError('Restore must not assemble or solve')
    with monkeypatch.context() as patch:
        patch.setattr(SystemOperator, '__init__', forbidden)
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored = mw.Result.load(path, scene=scene)
    assert torch.equal(restored.surface_currents.scattered_coefficients, actual.scattered_coefficients)
    for a,b in zip(restored.query_fields(points, region_id=0, field_kind='total'), expected):
        assert torch.equal(a,b)
    for mutation in ('source', 'volume', 'representation', 'metadata'):
        bad = copy.deepcopy(payload)
        inner = bad['surface_data']
        if mutation == 'source': inner['volume_state']['regions'][0]['additional_current'][0,0] = 1
        elif mutation == 'volume': inner['volume_state']['regions'][1]['vertices'] += .01
        elif mutation == 'representation': inner['representation']['material_partition'] = 'double_count'
        else: bad['metadata']['formulation_id'] = 'pmchwt_scalar_sie'
        inner['sha256'] = _digest({k:v for k,v in inner.items() if k != 'sha256'})
        bad_path = tmp_path/(mutation+'.pt');torch.save(bad, bad_path)
        with pytest.raises(ValueError): mw.Result.load(bad_path, scene=scene)
    actual.linear_volume_state.currents[0].add_(1)
    with pytest.raises(ValueError, match='changed'): result.query_fields(points, region_id=0, field_kind='total')


def test_prepared_and_restore_physical_binding_reject_replacement(tmp_path):
    from witwin.maxwell.mom.linear_volume_state import LinearVolumeState
    _, volumes = system_and_volumes(torch.float64)
    structures = [mw.Structure(mw.Mesh(v.boundary.vertices, v.boundary.faces,
        recenter=False, fill_mode='solid'), mw.Material(eps_r=2.+i)) for i,v in enumerate(volumes)]
    source = mw.PlaneWave(source_time=mw.CW(frequency=3e8))
    scene = mw.Scene(structures=structures, sources=[source])
    config = mw.MoMConfig(integration=IntegrationConfig(order=2, singular_order=2),
        query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1',
        linear_coupled=mw.LinearCoupledConfig(volumes))
    simulation = mw.Simulation.mom(scene, frequency=3e8, config=config)
    prepared = simulation.prepare()
    runtime = prepared.solver
    original = runtime.volume_state
    runtime.volume_state = LinearVolumeState(runtime.handle.system, config.linear_coupled, dtype=config.dtype)
    with pytest.raises(ValueError, match='changed'): runtime.run()
    runtime.volume_state = original
    simulation.config = replace(config, subdivisions=config.subdivisions+1)
    with pytest.raises(ValueError, match='changed'): runtime.run()
    simulation.config = config
    simulation.frequency = 3.1e8
    with pytest.raises(ValueError, match='changed'): runtime.run()
    simulation.frequency = 3e8
    result = prepared.run()
    path = tmp_path/'bound.pt';result.save(path)
    other_source = mw.PlaneWave(source_time=mw.CW(frequency=3e8, amplitude=2.))
    wrong_source = mw.Scene(structures=structures, sources=[other_source])
    wrong_material = mw.Scene(structures=[mw.Structure(s.geometry, mw.Material(eps_r=4.+i))
        for i,s in enumerate(structures)], sources=[source])
    for changed_scene in (wrong_source, wrong_material):
        with pytest.raises(ValueError): mw.Result.load(path, scene=changed_scene)
    payload = torch.load(path, weights_only=False)
    payload['frequency'] = 3.1e8
    torch.save(payload, tmp_path/'frequency.pt')
    with pytest.raises(ValueError): mw.Result.load(tmp_path/'frequency.pt', scene=scene)
