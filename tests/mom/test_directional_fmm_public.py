"""Public automatic DAG solve and constructor-free exact result restoration."""
import copy
import hashlib
import json
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_config import DirectionalFMMConfig
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan, AutomaticDirectionalDAGPlan
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator, MultipoleOperator
from witwin.maxwell.mom.runtime import SurfaceData, SystemOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_public_automatic_directional_three_body_solve_original_residual_and_exact_restore(tmp_path, monkeypatch):
    scene = mw.Scene(structures=[mw.Structure(
        mw.Sphere(radius=.02, position=(i, 1e-6*(i%2), 0.)),
        mw.Material(eps_r=2.+i, mu_r=1.3+.1*i, sigma_e=.01, sigma_m=.02))
        for i in range(3)], sources=[mw.PlaneWave()])
    fmm = DirectionalFMMConfig(leaf_size=4, separation_ratio=.15,
        cone_subdivisions=(2**20,), max_contexts=4096,
        tensor_workspace_budget_bytes=512*1024**2,
        leaf_factor_kernel='original', leaf_factor_max_edges=16)
    config = mw.MoMConfig(subdivisions=0,
        integration=IntegrationConfig(order=6, singular_order=3, pair_batch=4,
            far_pair_batch=16, near_max_order=32),
        operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
    simulation = mw.Simulation.mom(scene, frequency=3e8, config=config)
    prepared = simulation.prepare()
    handle = prepared.solver.handle
    actions = handle.actions
    regions = (actions.exterior, *actions.interiors)
    assert handle.metadata['automatic_cone_configuration']
    assert not handle.metadata['prescribed_complete_system']
    assert actions.ideal_rwg_truncation_bound()['absolute_bound'] <= fmm.global_ideal_rwg_error_budget
    assert regions[0].test.plan_guard.selected_positive_links > 0
    count = len(actions.interiors)
    for index, region in enumerate(regions):
        plan = region.test.plan_guard
        assert isinstance(plan, AutomaticDirectionalDAGPlan)
        budget = fmm.global_ideal_rwg_error_budget/(2 if index == 0 else 2*count**.5)
        assert plan.region_plan.ideal_rwg_error_budget == budget
    result = prepared.run()
    stats = result.stats()
    assert stats['converged'] and stats['error_budget']['checked']
    assert not stats['fallback_used'] and not stats['original_global_residual_certified']
    coefficients = result.surface_currents.scattered_coefficients
    path = tmp_path/'directional-public-result.pt'
    result.save(path)
    snapshot = result.surface_currents.snapshot()
    assert snapshot['fmm_config'] == fmm.payload()
    def forbidden(*args, **kwargs):
        raise AssertionError('Restoring physical directional currents rebuilt or solved the operator')
    with monkeypatch.context() as patch:
        for cls in (DirectionalSystemActions, DirectionalMultipoleOperator,
                    MultipoleOperator, AutomaticDirectionalRegionPlan,
                    AutomaticDirectionalDAGPlan, SystemOperator):
            patch.setattr(cls, '__init__', forbidden)
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored = mw.Result.load(path, scene=scene)
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,
            coefficients, rtol=0, atol=0)
        assert restored.surface_currents.config.fmm == fmm
        assert restored.surface_currents.snapshot()['fmm_config'] == fmm.payload()
        for field, value in (('max_candidates', 256), ('phase_relative_budget', 1e-11),
                             ('cone_subdivisions', [2]), ('global_ideal_rwg_error_budget', 1e-3)):
            bad = copy.deepcopy(snapshot)
            bad['fmm_config']['parameters'][field] = value
            with pytest.raises(ValueError): SurfaceData.restore(bad, scene, 3e8)
    (tmp_path/'public.json').write_text(json.dumps(dict(stats=stats,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        configuration=fmm.payload(), root_DAG_receipts=[r.test.plan_guard.receipt() for r in regions],
        exact_current_restore=True, original_global_residual_certified=False,
        physics_qualified=False, resources_qualified=False, qualification=False), indent=2, default=str)+'\n')


def test_public_automatic_embedding_capacity_rejected_before_plans(monkeypatch, tmp_path):
    from test_directional_automatic_dag import _physical_plan
    provider = _physical_plan().basis
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid embedding capacity reached a regional plan')
    with monkeypatch.context() as patch:
        patch.setattr(AutomaticDirectionalRegionPlan, '__init__', forbidden)
        with pytest.raises(MemoryError, match='before regional plans'):
            DirectionalSystemActions.from_automatic(provider,
                replace(DirectionalFMMConfig(), embedding_budget_bytes=16*provider.system.total_edges-1))
    (tmp_path/'capacity.json').write_text(json.dumps(dict(before_regional_plans=True,
        no_maps_or_coefficients_constructed=True, qualification=False), indent=2)+'\n')


def test_public_automatic_global_bound_rejection_precedes_all_maps(monkeypatch, tmp_path):
    from test_directional_automatic_dag import _physical_plan
    from witwin.maxwell.mom import rwg_interpolation_error
    provider = _physical_plan().basis
    config = DirectionalFMMConfig(leaf_size=4, separation_ratio=.15,
        cone_subdivisions=(2**20,), max_candidates=1, max_contexts=4096)
    original = rwg_interpolation_error.directional_system_ideal_bound
    observed = []
    def refused(*args, **kwargs):
        bound = original(*args, **kwargs)
        observed.append(bound['absolute_bound'])
        return bound | dict(absolute_bound=config.global_ideal_rwg_error_budget*1.01)
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected global admission reached maps/near coefficient construction')
    with monkeypatch.context() as patch:
        patch.setattr(rwg_interpolation_error, 'directional_system_ideal_bound', refused)
        patch.setattr(AutomaticDirectionalDAGPlan, 'build_region', forbidden)
        with pytest.raises(ValueError, match='before maps'):
            DirectionalSystemActions.from_automatic(provider, config)
    assert len(observed) == 1 and observed[0] <= config.global_ideal_rwg_error_budget
    (tmp_path/'global_preflight.json').write_text(json.dumps(dict(
        actual_admitted_bound=observed[0], injected_rejection_control=True,
        all_maps_and_near_factories_not_reached=True, qualification=False), indent=2)+'\n')
