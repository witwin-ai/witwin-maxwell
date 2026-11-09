"""Exact current configuration union without replaying closed physical fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import witwin.maxwell as mw


def test_new_binary_current_config_union_identity_without_numeric_replay(tmp_path):
    source=os.environ.get('GSP_CLOSED_BINARY_PUBLIC_RECEIPT')
    digest=os.environ.get('GSP_CLOSED_BINARY_PUBLIC_RECEIPT_SHA256')
    if source is None and digest is None:
        pytest.skip('Requires the pinned closed binary public receipt')
    assert source and digest
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    raw=json.loads(Path(source).read_text())
    original=raw['fmm_config']
    assert original['version']=='cartesian_fmm_binary_lattice_config_v1'
    assert 'near_coefficient_workspace_budget_bytes' not in original['parameters']
    cfg=mw.FMMConfig.restore(original)
    assert cfg.payload()==original
    assert cfg.expansion_center_policy=='binary_lattice_v1'
    assert cfg.near_storage=='scalar_reciprocal_dk_v1'
    assert cfg.construction()['expansion_center_policy']=='binary_lattice_v1'
    # The new combined configuration binds every actual control explicitly.
    combined=copy.deepcopy(original['parameters'])
    combined.update(near_storage='scalar_reciprocal_streamed_dk_v1',
        near_coefficient_workspace_budget_bytes=6144)
    streamed=mw.FMMConfig.restore(dict(version='cartesian_fmm_streamed_reciprocal_dk_config_v1',parameters=combined))
    payload=streamed.payload()
    assert payload['version']=='cartesian_fmm_streamed_reciprocal_dk_config_v1'
    assert payload['parameters']['expansion_center_policy']=='binary_lattice_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==streamed
    assert streamed.construction()['near_coefficient_workspace_budget_bytes']==6144
    for version in ('cartesian_fmm_binary_lattice_config_v1','cartesian_fmm_dyadic_centers_config_v1'):
        bad=copy.deepcopy(payload);bad['version']=version
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    bad=copy.deepcopy(original);bad['parameters']['near_coefficient_workspace_budget_bytes']=6144
    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    bad=copy.deepcopy(original);bad['parameters']['expansion_center_policy']='dyadic_cell_v1'
    with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    directional=mw.DirectionalFMMConfig(near_storage='scalar_reciprocal_streamed_dk_v1',
        near_coefficient_workspace_budget_bytes=7168)
    assert mw.FMMConfig.restore(json.loads(json.dumps(directional.payload())))==directional
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    (tmp_path/'binary-current-config-union.json').write_text(json.dumps(dict(
        closed_binary_receipt_sha256=digest,closed_binary_payload_exact=True,
        new_combined_payload_exact=True,directional_streamed_dispatch_exact=True,
        no_reference_or_producer_or_archive_or_planner_replay=True,
        combined_binary_streamed_actions_qualified=False,
        production_qualification=False),indent=2)+'\n')
