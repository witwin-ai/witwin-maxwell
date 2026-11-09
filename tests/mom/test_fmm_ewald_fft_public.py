"""Fresh public periodic FFT producer and constructor-free exact restoration."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.runtime import SurfaceData, SystemOperator
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_public_ewald_fft_original_residual_and_exact_state_restore(tmp_path, monkeypatch):
    dx = .17/compile_medium(mw.Material(), 9e7).k.real
    wave = mw.PlaneWave(direction=(dx, 0., math.sqrt(1-dx*dx)), polarization=(0., 1., 0.))
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
        mw.Material(eps_r=2.+i, mu_r=1.2+.1*i)) for i, x in enumerate((-.2, .2))], sources=[wave])
    periodic = mw.MoMPeriodicConfig(((1., 0., 0.),), bloch_rad_per_m=(.17, 0., 0.),
        eta_per_m=2., real_shells=2, reciprocal_shells=2, quadrature_order=32)
    fmm = mw.FMMConfig(order=8, near_storage='full_matrix', table_storage='fft_ewald_v1', shift_storage='separable_v1',
        leaf_size=30, separation_ratio=.25, leaf_factor_kernel='primary', leaf_factor_max_edges=16)
    config = mw.MoMConfig(subdivisions=0, periodic=periodic, integration=IntegrationConfig(order=4, singular_order=3),
        operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', fallback='none',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
    result = mw.Simulation.mom(scene, frequency=9e7, config=config).run()
    stats = result.stats()
    assert stats['converged'] and stats['error_budget']['checked']
    actions = result.surface_currents.handle.actions
    assert actions.capacity_plan['translated_images'] > 0
    assert actions.capacity_plan['table_storage'] == 'fft_ewald_v1'
    assert actions.receipt()['interiors_periodic'] is False
    current = result.surface_currents.scattered_coefficients
    path = tmp_path/'fft_ewald_v1-fmm.pt'; result.save(path)
    record = dict(artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), stats=stats,
        capacity_plan=actions.capacity_plan, fmm_config=fmm.payload(), periodic_config=periodic.payload(),
        exact_current_restore=False, original_reference_scope='configured_ewald_quadrature_and_cutoffs',
        physical_qualification=False, complexity_qualification=False, infinite_periodic_certified=False)
    (tmp_path/'public.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    def forbidden(*args, **kwargs): raise AssertionError('Periodic FFT replay rebuilt or solved the operator')
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
    from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
    with monkeypatch.context() as patch:
        for cls in (MultipoleOperator, MultipoleSystemActions, PeriodicMultipoleOperator, PeriodicMultipoleSystemActions, SystemOperator):
            patch.setattr(cls, '__init__', forbidden)
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored = mw.Result.load(path, scene=scene)
        assert restored.surface_currents.config.fmm == fmm
        assert restored.surface_currents.config.periodic == periodic
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients, current, rtol=0, atol=0)
        assert restored.surface_currents.snapshot()['fmm_config'] == fmm.payload()
        snapshot = result.surface_currents.snapshot()
        for field, value in (('table_storage', 'compact'), ('kernel_bound_rule', 'real_support_min_v1'), ('shift_storage', 'dense')):
            bad = copy.deepcopy(snapshot); bad['fmm_config']['parameters'][field] = value
            error = NotImplementedError if field == 'kernel_bound_rule' else ValueError
            with pytest.raises(error): SurfaceData.restore(bad, scene, 9e7)
        for field, value in (('version', 'invalid'), ('real_shells', 3), ('bloch_rad_per_m', [.18, 0., 0.])):
            bad = copy.deepcopy(snapshot); bad['periodic_config'][field] = value
            with pytest.raises(ValueError): SurfaceData.restore(bad, scene, 9e7)
    record['exact_current_restore'] = True
    (tmp_path/'public.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    print(record, flush=True)


def test_saved_ewald_fft_producer_exact_recovery_and_all_binding_guards(tmp_path, monkeypatch):
    source = os.environ.get('GSP_EWALD_FFT_RECOVERY_RESULT')
    digest = os.environ.get('GSP_EWALD_FFT_RECOVERY_RESULT_SHA256')
    if source is None and digest is None:
        pytest.skip('Completed Ewald FFT solve artifact required for recovery control')
    assert source and digest, 'Recovery artifact and SHA256 must be supplied together'
    path = Path(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    # Independently declare the complete problem before loading saved state.
    dx = .17/compile_medium(mw.Material(), 9e7).k.real
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
        mw.Material(eps_r=2.+i, mu_r=1.2+.1*i)) for i, x in enumerate((-.2, .2))],
        sources=[mw.PlaneWave(direction=(dx, 0., math.sqrt(1-dx*dx)), polarization=(0., 1., 0.))])
    periodic = mw.MoMPeriodicConfig(((1., 0., 0.),), bloch_rad_per_m=(.17, 0., 0.),
        eta_per_m=2., real_shells=2, reciprocal_shells=2, quadrature_order=32)
    fmm = mw.FMMConfig(order=8, near_storage='full_matrix', table_storage='fft_ewald_v1', shift_storage='separable_v1',
        leaf_size=30, separation_ratio=.25, leaf_factor_kernel='primary', leaf_factor_max_edges=16)
    config = mw.MoMConfig(subdivisions=0, periodic=periodic, integration=IntegrationConfig(order=4, singular_order=3),
        operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', fallback='none',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
    from witwin.maxwell.mom.runtime import _numerical_identity
    expected_identity = _numerical_identity(config)
    def forbidden(*args, **kwargs): raise AssertionError('Recovery rebuilt or solved a FMM operator')
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
    from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
    for cls in (MultipoleOperator, MultipoleSystemActions, PeriodicMultipoleOperator, PeriodicMultipoleSystemActions, SystemOperator):
        monkeypatch.setattr(cls, '__init__', forbidden)
    monkeypatch.setattr(torch.linalg, 'solve', forbidden)
    saved = torch.load(path, map_location='cuda', weights_only=False)['surface_data']
    result = mw.Result.load(path, scene=scene)
    assert _numerical_identity(result.surface_currents.config) == expected_identity
    torch.testing.assert_close(result.surface_currents.scattered_coefficients,
        saved['scattered_coefficients'], rtol=0, atol=0)
    stats = result.stats()
    assert stats['converged'] and stats['error_budget']['checked']
    assert stats['error_budget']['compressed_dual_pass'] and stats['error_budget']['original_sampled_pass']
    for field, value, error in (('table_storage', 'compact', ValueError),
            ('kernel_bound_rule', 'real_support_min_v1', NotImplementedError), ('shift_storage', 'dense', ValueError)):
        bad = copy.deepcopy(saved); bad['fmm_config']['parameters'][field] = value
        with pytest.raises(error): SurfaceData.restore(bad, scene, 9e7)
    for field, value in (('version', 'invalid'), ('real_shells', 3), ('bloch_rad_per_m', [.18, 0., 0.])):
        bad = copy.deepcopy(saved); bad['periodic_config'][field] = value
        with pytest.raises(ValueError): SurfaceData.restore(bad, scene, 9e7)
    record = dict(artifact_sha256=digest, exact_current_restore=True, numerical_identity=expected_identity,
        saved_solve_converged=True, saved_error_budget=stats['error_budget'], fresh_solve_executed=False,
        producer_collector_status='FAILED_negative_exception_assertion_after_completed_solve',
        physical_qualification=False, complexity_qualification=False, infinite_periodic_certified=False)
    (tmp_path/'recovery.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    print(record, flush=True)
