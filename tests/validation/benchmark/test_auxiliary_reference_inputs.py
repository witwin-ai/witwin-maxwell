"""Fail-closed physical input and separate power-budget regressions."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmark.mom_auxiliary_acceptance import (
    BUDGET, COMPARISON_CONTRACT, power_changes, validate_mom_input,
)


def write_report(tmp_path, contract=None):
    report = dict(frequency_hz=3e8, semiaxes_m=[.1, .08, .065], cases=[])
    if contract is not None:
        report['physical_contract'] = contract
    path = tmp_path / 'mom.json'
    path.write_text(json.dumps(report))
    return path


def reviewed_fixture(tmp_path):
    path = write_report(tmp_path)
    report = json.loads(path.read_text())
    hashes = {}
    for name in ('benchmark/mom_ellipsoid_convergence.py', 'benchmark/mom_fdtd.py'):
        target = tmp_path / 'snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# Frozen reviewed source fixture: ' + name)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    report['source_sha256'] = hashes
    path.write_text(json.dumps(report))
    entry = dict(physical_contract=COMPARISON_CONTRACT,
                 review=dict(reviewed_by='regression fixture', evidence_notes='Fixture source review'),
                 source_snapshot_directory='snapshot', source_sha256=hashes)
    sidecar = tmp_path / 'review.json'
    sidecar.write_text(json.dumps(dict(schema='mom-reviewed-physical-contract-v1',
        reports={hashlib.sha256(path.read_bytes()).hexdigest():entry})))
    return path, sidecar


def test_same_geometry_without_material_and_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='complete physical_contract'):
        validate_mom_input(write_report(tmp_path))


def test_complete_explicit_contract_is_accepted(tmp_path):
    _, identity = validate_mom_input(write_report(tmp_path, COMPARISON_CONTRACT))
    assert identity['contract_evidence']['kind'] == 'explicit_report_metadata'


@pytest.mark.parametrize('change', ['epsilon', 'phase', 'missing_mu'])
def test_same_geometry_with_wrong_or_incomplete_physics_is_rejected(tmp_path, change):
    contract = deepcopy(COMPARISON_CONTRACT)
    if change == 'epsilon':
        contract['epsilon_r'] = [4, 0]
    elif change == 'phase':
        contract['source']['phase_radians'] = 0
    else:
        del contract['mu_r']
    with pytest.raises(ValueError, match='does not match'):
        validate_mom_input(write_report(tmp_path, contract))


def test_reviewed_legacy_report_requires_exact_report_and_source_bytes(tmp_path):
    path, sidecar = reviewed_fixture(tmp_path)
    before = path.read_bytes()
    _, identity = validate_mom_input(path, sidecar)
    assert identity['contract_evidence']['kind'] == 'reviewed_legacy_sidecar'
    assert path.read_bytes() == before
    path.write_bytes(before + b' ')
    with pytest.raises(ValueError, match='SHA256 is absent'):
        validate_mom_input(path, sidecar)
    path.write_bytes(before)
    (tmp_path / 'snapshot/benchmark/mom_fdtd.py').write_text('# changed source')
    with pytest.raises(ValueError, match='snapshot is missing or has changed'):
        validate_mom_input(path, sidecar)


def test_field_budget_does_not_implicitly_pass_power_budget():
    # A 0.04% amplitude change passes the 0.05% field budget, but its squared
    # power changes by 0.080016%, which must fail the independent power gate.
    amplitude = 1 + .8 * BUDGET
    first = {'far_reference_powers': {'scattering_W': amplitude**2, 'extinction_W': 1}}
    second = {'far_reference_powers': {'scattering_W': 1, 'extinction_W': 1}}
    assert amplitude - 1 < BUDGET
    assert power_changes(first, second)['scattering_W'] > BUDGET
