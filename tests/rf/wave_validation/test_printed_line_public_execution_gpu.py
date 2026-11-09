"""Small genuine completeResult native91056 workflow qualification."""
from dataclasses import replace
import pytest,torch
from scripts.dev.fdtd.printed_line_public_execution import column_case,run_public_column

def compact_case():
    return replace(column_case(0),axial_half_length=.06,separation=.06,
        padding=.01,positions=(-.015,0.,.015),duration=4e-9,checkpoint_time=2e-9)


@pytest.mark.parametrize('kind,graph,resume',[('cpw',False,False),('cpw',True,True),('differential_pair',True,True)])
def test_actual_public_complete_small_column_native_heads_and_disk(tmp_path,request,kind,graph,resume):
    """Explicit4ns unit horizon, not a renamed64ns prefix or accuracy gate."""
    if not request.config.getoption('--gpu',default=False)or not torch.cuda.is_available():
        pytest.skip('Needs explicit --gpu and CUDA; no implicit GPU execution.')
    report=run_public_column(kind,compact_case(),0,tmp_path/'actual',graph=graph,resume=resume)
    assert report['functional_passed']and report['other_columns_run']==0
    assert report['clock']['requested_duration_s']==4e-9
    assert report['reverse_and_disk_bitwise']and not report['physical_mode_attenuation_accuracy_qualified']
    assert report['native_Point_plane_primitive_proof']['actual_Point_plane_weighted_primitive_parity']
    assert report['initial_independent_queries']['primary_checks']==6
    assert report['initial_independent_queries']['constitutive_side_checks']==12
    assert report['restored_independent_queries']['independently_verified']
