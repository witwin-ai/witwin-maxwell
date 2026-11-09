"""Actual read-only extra native V/I observations, never a physical release."""
from dataclasses import replace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase,DRIVE_BASIS,build_case,run_drive,execution_identity

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA execution required')

@pytest.mark.parametrize('graph',[False,True])
def test_actual_dual_VI_preserves_source_fields_and_primary_samples(graph,tmp_path):
    actual_identity=execution_identity()
    case=replace(PilotCase(),dx=.00125,longitudinal_dl=.0025,probe_reference_dx=.0025,
        axial_half_length=.06,padding=.02,separation=.06,positions=(-.015,0.,.015),frequencies=(.8e9,1e9,1.2e9))
    dual_case=replace(case,record_legacy_native_vi=True)
    primary_folder=tmp_path/'primary';dual_folder=tmp_path/'dual-resumed'
    plain=run_drive(case,DRIVE_BASIS[0],primary_folder,graph=graph,resume=False,planned_steps=1024)
    dual=run_drive(dual_case,DRIVE_BASIS[0],dual_folder,graph=graph,resume=True,planned_steps=1024,stop_step=512)
    assert plain['port_operator_sha256']==dual['port_operator_sha256']
    assert plain['actual_dt_s']==dual['actual_dt_s']
    for key in ('source_sha256','grid_sha256','material_sha256'):
        assert plain['contract'][key]==dual['contract'][key]
    assert plain['planned_steps']==dual['planned_steps']==1024
    assert plain['actual_field_graph']==dual['actual_field_graph']==graph
    assert plain['actual_observer_graph']==dual['actual_observer_graph']==graph
    a=torch.load(primary_folder/'final-fields.pt',map_location='cpu',weights_only=False)
    b=torch.load(dual_folder/'final-fields.pt',map_location='cpu',weights_only=False)
    assert a.keys()==b.keys()
    for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    np.testing.assert_array_equal(plain['physical_state'],dual['physical_state'])
    assert plain['legacy_native_physical_state'] is None
    assert dual['legacy_native_physical_state'].shape==(3,2,3)
    assert np.isfinite(dual['legacy_native_physical_state']).all()
    assert np.max(np.abs(dual['legacy_native_physical_state']))>0
    scene=build_case(dual_case)[0]
    result=mw.Result.load(dual_folder/'result.pt',scene=scene,map_location='cpu')
    contract=dual['legacy_native_VI_observation_contract'];actual=[]
    for pair in contract['integration_rows']:
        values=[]
        for row in pair:
            sums=[]
            for frequency in case.frequencies:
                value=sum(weight*complex(result.monitor(name,frequency=frequency)['components'][component]) for name,component,weight in row)
                sums.append(value)
            values.append(sums)
        actual.append(values)
    np.testing.assert_array_equal(np.asarray(actual),dual['legacy_native_physical_state'])
    for report in (plain,dual):
        assert report['all_source_identity_unchanged']
        assert report['source_before']==report['source_after']==actual_identity
