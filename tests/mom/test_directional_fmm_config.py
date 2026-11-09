"""Strict public directional configuration and selected factory admission."""
import copy
import json
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule


def test_directional_configuration_canonical_roundtrip_and_all_fields_bound(tmp_path):
    config = mw.DirectionalFMMConfig(cone_subdivisions=(2, 4), max_candidates=256,
        global_ideal_rwg_error_budget=3e-5, leaf_factor_kernel='original')
    payload = json.loads(json.dumps(config.payload()))
    assert mw.DirectionalFMMConfig.restore(payload) == config
    assert mw.FMMConfig.restore(payload) == config
    assert config.solve_config().original_sample_tolerance == 1e-8
    for field in payload['parameters']:
        bad = copy.deepcopy(payload)
        del bad['parameters'][field]
        with pytest.raises(ValueError, match='incomplete'):
            mw.FMMConfig.restore(bad)
    for version in ('cartesian_fmm_config_v7', 'cartesian_fmm_fft_config_v1', 'unknown'):
        bad = copy.deepcopy(payload); bad['version'] = version
        with pytest.raises(ValueError):
            mw.FMMConfig.restore(bad)
    bad = copy.deepcopy(payload); bad['parameters']['unbound'] = 1
    with pytest.raises(ValueError): mw.DirectionalFMMConfig.restore(bad)
    for field, value in (('cone_subdivisions', (2, 4)), ('kernel_error_budget', None)):
        bad = copy.deepcopy(payload); bad['parameters'][field] = value
        with pytest.raises(ValueError): mw.DirectionalFMMConfig.restore(bad)
    (tmp_path/'configuration.json').write_text(json.dumps(dict(payload=payload,
        all_parameters_required=True,JSON_roundtrip=True,qualification=False), indent=2)+'\n')


def test_directional_configuration_rejects_invalid_orders_modes_budgets_and_catalogs():
    config = mw.DirectionalFMMConfig()
    invalid = [dict(min_order=True), dict(max_order=13), dict(min_order=8, max_order=7),
        dict(quadrature_order=7), dict(max_families=0), dict(max_candidates=True),
        dict(max_contexts=0), dict(regional_near_budget_bytes=False), dict(embedding_budget_bytes=0),
        dict(cone_subdivisions=[2]), dict(cone_subdivisions=(2, 2)),
        dict(cone_subdivisions=(True,)), dict(cone_subdivisions=(2**21,)),
        dict(phase_relative_budget=1), dict(global_ideal_rwg_error_budget=float('nan')),
        dict(separation_ratio=1), dict(leaf_factor_kernel='primary'),
        dict(solve_dual_tolerance=float('inf'))]
    for change in invalid:
        with pytest.raises(ValueError): replace(config, **change)
    with pytest.raises(TypeError): replace(config, kernel_error_budget={})


def test_directional_public_MoM_scope_and_configuration_identity():
    config = mw.DirectionalFMMConfig()
    options = dict(operator_backend='fmm', fmm=config, riesz_storage='local_rwg_v1',
                   stopping=StoppingRule(preconditioner='rwg_diagonal'))
    accepted = mw.MoMConfig(**options)
    assert accepted.fmm is config
    with pytest.raises(ValueError): mw.MoMConfig(fmm=config)
    with pytest.raises(NotImplementedError): mw.MoMConfig(**options, periodic=mw.MoMPeriodicConfig(((1., 0., 0.),)))
    with pytest.raises(ValueError): mw.MoMConfig(**options, dtype=torch.complex64)
    with pytest.raises(ValueError): mw.MoMConfig(**dict(options, stopping=StoppingRule(method='direct')))
    with pytest.raises(ValueError): mw.MoMConfig(**dict(options, stopping=StoppingRule(fallback='direct')))
    from witwin.maxwell.mom.runtime import _numerical_identity
    changed = replace(accepted, fmm=replace(config, max_candidates=257))
    assert _numerical_identity(accepted) != _numerical_identity(changed)
