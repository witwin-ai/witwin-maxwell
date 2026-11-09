"""Completed independent drives can be reused without appending partial fields."""
import hashlib
import json
import pytest
from scripts.dev.fdtd.printed_line_external_reference import (
    completed_drive_action,completed_drive_provenance,validate_completed_execution,
)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completed_drive_policy_is_explicit_and_rejects_partial_data(tmp_path):
    assert completed_drive_action(tmp_path,reuse_complete=True)=='run'
    (tmp_path/'raw').mkdir()
    with pytest.raises(ValueError,match='incomplete drive'):
        completed_drive_action(tmp_path,reuse_complete=True)
    (tmp_path/'simulation.xml').write_text('<physical/>')
    with pytest.raises(ValueError,match='execution timing'):
        completed_drive_action(tmp_path,reuse_raw=True)
    (tmp_path/'execution.json').write_text('{}')
    assert completed_drive_action(tmp_path,reuse_complete=True)=='reuse'
    assert completed_drive_action(tmp_path,reuse_raw=True)=='reuse'
    with pytest.raises(ValueError,match='one explicit'):
        completed_drive_action(tmp_path,reuse_raw=True,reuse_complete=True)


@pytest.mark.parametrize('change',[{'steps':9},{'dt_s':2e-12},{'seconds':0.},{'seconds':float('nan')},{'xml_sha256':'changed'}])
def test_completed_execution_requires_exact_xml_timestep_count_and_original_positive_timing(tmp_path,change):
    xml=tmp_path/'simulation.xml';xml.write_text('<physical/>')
    execution={'steps':10,'dt_s':1e-12,'seconds':4.,'xml_sha256':digest(xml)}
    validate_completed_execution(execution,xml,1e-12,10)
    execution.update(change)
    with pytest.raises(ValueError,match='Retained execution identity'):
        validate_completed_execution(execution,xml,1e-12,10)


def provenance(tmp_path):
    executed=tmp_path/'original-driver.py';executed.write_text('original execution driver\n')
    timing=tmp_path/'execution.json';timing.write_text('{"seconds":4503.184962479398}\n')
    current={'maxwell_source_sha256':'original numerical source','fixture_sha256':'fixed physical fixture',
        'reference':{'binding_sha256':'pinned binding','library_sha256':'pinned library'}}
    record=dict(current,execution_sha256=digest(timing),executed_driver_path=str(executed),executed_driver_sha256=digest(executed))
    (tmp_path/'reuse-provenance.json').write_text(json.dumps(record))
    return current,record


@pytest.mark.parametrize('field',['maxwell_source_sha256','fixture_sha256','reference'])
def test_completed_provenance_rejects_changed_physical_source_or_reference_build(tmp_path,field):
    current,record=provenance(tmp_path)
    assert completed_drive_provenance(tmp_path,current)==record
    current[field]='changed'
    with pytest.raises(ValueError,match='provenance differs'):
        completed_drive_provenance(tmp_path,current)


def test_completed_provenance_preserves_original_driver_and_execution_timing(tmp_path):
    current,record=provenance(tmp_path)
    (tmp_path/'original-driver.py').write_text('changed driver\n')
    with pytest.raises(ValueError,match='executed driver'):
        completed_drive_provenance(tmp_path,current)
    (tmp_path/'execution.json').write_text('{"seconds":0.}\n')
    with pytest.raises(ValueError,match='timing sidecar'):
        completed_drive_provenance(tmp_path,current)
