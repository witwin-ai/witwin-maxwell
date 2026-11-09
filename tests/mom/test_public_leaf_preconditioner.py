"""Public leaf solves recover physical currents and preserve exact configuration."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.runtime import SurfaceData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('periodic', [False, True])
def test_public_leaf_solve_original_admission_and_storage(periodic, tmp_path):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01), mw.Material(eps_r=2.))],
                     sources=[mw.PlaneWave()])
    lattice = (mw.MoMPeriodicConfig(((1., 0., 0.),), eta_per_m=2., real_shells=2,
        reciprocal_shells=2, quadrature_order=32) if periodic else None)
    dense_config = mw.MoMConfig(subdivisions=0, periodic=lattice,
        integration=IntegrationConfig(order=4, singular_order=3))
    reference = mw.Simulation.mom(scene, frequency=9e7, config=dense_config).run()
    config = replace(dense_config, operator_backend='fmm', riesz_storage='local_rwg_v1',
        fmm=mw.FMMConfig(order=6, table_storage='compact', separation_ratio=.25,
                        leaf_factor_kernel='primary', leaf_factor_max_edges=16, near_action_batch=7),
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
    prepared = mw.Simulation.mom(scene, frequency=9e7, config=config).prepare()
    actions = prepared.solver.handle.actions
    exterior_regions = actions.exterior.regions if periodic else (actions.exterior,)
    for region in (*exterior_regions, *actions.interiors):
        assert region.near.action_batch == 7
    result = prepared.run()
    stats = result.stats()
    assert stats['converged'] and stats['error_budget']['checked'], stats
    assert stats['leaf_preconditioner']['factor_kernel'] == 'primary'
    assert stats['returned_coordinates'] == 'physical_scattered_currents'
    current = result.surface_currents.scattered_coefficients
    print({'periodic': periodic, 'iterations': stats.get('iterations'),
           'original_sample': stats['error_budget']['residual_report']['original_sampled'],
           'current_relative_difference': float((current-reference.surface_currents.scattered_coefficients).norm()/
                                                reference.surface_currents.scattered_coefficients.norm())})
    expected = reference.surface_currents.scattered_coefficients
    # Compare both physical current families, so larger M units cannot hide J error.
    for actual_block, expected_block in zip(current.chunk(2), expected.chunk(2)):
        relative = float((actual_block-expected_block).norm()/expected_block.norm())
        print({'current_family_relative_error': relative,
               'maximum_absolute_error': float((actual_block-expected_block).abs().max())})
        assert relative < 1e-8
    original = reference.surface_currents.handle
    rhs = original.build_rhs()
    assert float((original.apply(current)-rhs).norm()/rhs.norm()) < 1e-8
    path = tmp_path/'leaf_result.pt'
    result.save(path)
    restored = mw.Result.load(path, scene=scene)
    assert restored.surface_currents.config.fmm == config.fmm
    assert torch.equal(restored.surface_currents.scattered_coefficients, current)
    corrupt = copy.deepcopy(result.surface_currents.snapshot())
    corrupt['fmm_config']['parameters']['leaf_factor_kernel'] = 'original'
    with pytest.raises(ValueError, match='identity'):
        SurfaceData.restore(corrupt, scene, 9e7)

    prepared.solver.leaf_inverse = None
    with pytest.raises(ValueError, match='binding changed'):
        prepared.solver.run()
