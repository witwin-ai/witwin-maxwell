"""Acceptance calculations preserve frozen limits and explicit control references."""
import json

import pytest
import torch

from benchmark.mom_cross_assessment import assess, norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_far_norm_uses_solid_angle_weights():
    values = torch.tensor([[1+1j,0,0],[2+0j,0,0]],device='cuda',dtype=torch.complex128)
    weights = torch.tensor([.1,.9],device='cuda',dtype=torch.float64)
    assert norm(values,weights) == pytest.approx((.1*2+.9*4)**.5)


def test_pml_refinement_keeps_explicit_reference_and_failed_coarse_control(tmp_path):
    sweeps=[]
    for depth,amplitude in ((.2,1.02),(.25,1.005),(.3,1.0)):
        value={'real':[[amplitude,0,0]],'imag':[[0,0,0]]}
        sweeps.append({'parameter':'pml','value':depth,'status':'measured',
                       'fields':{key:value for key in ('total_E','total_H','scattered_E','scattered_H')}})
    path=tmp_path/'controls.json'
    path.write_text(json.dumps({'cases':[],'reference_pml_m':.3,'sensitivity_sweeps':sweeps}))
    result=assess(path)
    controls=result['independent_fdtd_controls']
    assert len(controls)==2
    assert all(row['reference_value']==.3 for row in controls)
    assert all(not check['passed'] for check in controls[0]['checks'])
    assert all(check['passed'] for check in controls[1]['checks'])
    assert controls[0]['checks'][0]['limit']==.01


@pytest.mark.host_validation
def test_unrecognized_measurement_is_unassessed(tmp_path):
    path=tmp_path/'unknown.json'
    path.write_text(json.dumps({'cases':[{'status':'measured','family':'future_measurement'}]}))
    row=assess(path)['rows'][0]
    assert row['assessment_status']=='unassessed'
    assert row['all_recorded_limits_passed'] is None
