"""Strict D/K numerical identities with explicit storage policies."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule


def test_all_old_payload_and_construction_bytes_and_strict_scalar_dk_versions():
    path, digest = os.environ.get('GSP_ORIGINAL_FMM_CONFIG'), os.environ.get('GSP_ORIGINAL_FMM_CONFIG_SHA256')
    if path is None and digest is None:pytest.skip('Original configuration source pair required')
    assert path and digest
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    name = 'witwin.maxwell.mom._original_scalar_dk_fmm_config'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module;spec.loader.exec_module(module)
    rules = ('bivariate_cauchy_v1', 'real_support_min_v1', 'positive_tail_min_v1', 'first_wave_min_v1',
             'direct_positive_tail_min_v1', 'direct_first_wave_min_v1')
    for storage in ('dense', 'compact', 'fft_v1', 'fft_ewald_v1'):
        for rule in rules:
            for shift in ('dense', 'separable_v1'):
                kwargs = dict(table_storage=storage, kernel_bound_rule=rule, shift_storage=shift)
                current, original = mw.FMMConfig(**kwargs, near_storage='full_matrix'), module.FMMConfig(**kwargs)
                expected = original.payload()
                expected['parameters'].update(near_storage='full_matrix', shift_storage=shift,
                    near_coefficient_workspace_budget_bytes=128*1024**2,
                    near_coefficient_cache_budget_bytes=128*1024**2)
                assert current.payload() == expected
                assert current.construction() == original.construction()
                assert mw.FMMConfig.restore(current.payload()) == current
                if storage == 'fft_ewald_v1':continue
                changed = mw.FMMConfig(**kwargs, near_storage='scalar_dk_v1')
                payload = changed.payload();assert payload['version'] == 'cartesian_fmm_scalar_dk_config_v1'
                assert changed.construction()['near_storage'] == 'scalar_dk_v1'
                assert mw.FMMConfig.restore(payload) == changed
                for field in payload['parameters']:
                    bad = copy.deepcopy(payload);del bad['parameters'][field]
                    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
                bad = copy.deepcopy(payload);bad['parameters']['near_storage'] = 'full_matrix'
                with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
                bad = current.payload();bad['parameters']['near_storage'] = 'scalar_dk_v1'
                with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
                bad = copy.deepcopy(payload);bad['version'] = current.payload()['version']
                with pytest.raises(ValueError):mw.FMMConfig.restore(bad)


def test_scalar_dk_native_and_periodic_contract_rejects_before_public_setup():
    for value in (True, None, 'scalar', 'full'):
        with pytest.raises(ValueError):mw.FMMConfig(near_storage=value)
    with pytest.raises(NotImplementedError, match='free space'):
        mw.FMMConfig(near_storage='scalar_dk_v1', table_storage='fft_ewald_v1')
    config = mw.FMMConfig(near_storage='scalar_dk_v1')
    stopping = StoppingRule(preconditioner='rwg_diagonal')
    with pytest.raises(ValueError, match='complex128'):
        mw.MoMConfig(operator_backend='fmm', fmm=config, dtype=torch.complex64, riesz_storage='local_rwg_v1', stopping=stopping)
    with pytest.raises(NotImplementedError, match='free space'):
        mw.MoMConfig(operator_backend='fmm', fmm=config, periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)),
                     riesz_storage='local_rwg_v1', stopping=stopping)
