"""Dual observation contracts; CPU compilation, no field advance."""
from dataclasses import asdict,replace
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase,build_case,canonical_case_payload
from scripts.dev.fdtd.printed_line_broadband import encode
from tests.rf.wave_validation.test_loss_material_control_preparation import _rows_by_native_key

@pytest.fixture(scope='module')
def contour_helpers():
    # The printed-line builders load their contour helpers from this evidence directory.
    if not (Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-22').exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22')


@pytest.fixture(scope='module')
def dual_geometry(contour_helpers):
    case=replace(PilotCase(),dx=.00125,longitudinal_dl=.0025,probe_reference_dx=.0025,
        axial_half_length=.06,padding=.02,separation=.06,positions=(-.015,0.,.015),frequencies=(.8e9,1e9,1.2e9))
    return case,build_case(case),build_case(replace(case,record_legacy_native_vi=True))


def test_secondary_names_rows_and_points_preserve_the_primary(dual_geometry):
    case,plain,dual=dual_geometry
    assert plain[1:5]==dual[1:5]
    for name,value in plain[5].items():assert dual[5][name]==value
    secondary=dual[0].metadata['legacy_native_VI_observation_contract']
    assert secondary['role']=='secondary_mesh_native'
    assert all(name.startswith('legacy_') for name in secondary['expected_native_probes'])
    assert not set(plain[5]).intersection(secondary['expected_native_probes'])
    roles=dual[0].metadata['VI_observation_roles']
    assert roles['primary']['state_key']=='physical_state' and roles['primary']['functional']=='fixed_reference'
    assert roles['secondary']['state_key']=='legacy_native_physical_state'
    # The fine loop has a different actual physical contour; its integral must
    # not be silently relabelled as the fixed reference functional.
    assert _rows_by_native_key(plain[4],plain[5])!=_rows_by_native_key(secondary['integration_rows'],secondary['expected_native_probes'])
    assert secondary['grid_spacing_m']==case.dx


@pytest.mark.usefixtures('contour_helpers')
def test_coarse_native_secondary_is_exactly_the_original_functional():
    case=replace(PilotCase(),probe_reference_dx=.0025,record_legacy_native_vi=True)
    result=build_case(case);secondary=result[0].metadata['legacy_native_VI_observation_contract']
    assert _rows_by_native_key(result[4],result[5])==_rows_by_native_key(secondary['integration_rows'],secondary['expected_native_probes'])


@pytest.mark.parametrize('changes',[{'record_legacy_native_vi':1},{'record_legacy_native_vi':True,'probe_reference_dx':None}])
def test_invalid_secondary_declaration_rejects(changes):
    with pytest.raises(ValueError,match='explicit boolean'):
        build_case(replace(PilotCase(),**changes))


@pytest.fixture(scope='module')
def compiled_preflights(contour_helpers, tmp_path_factory):
    folder=tmp_path_factory.mktemp('dual-vi-cpu');records=[]
    for enabled in (False,True):
        output=folder/('dual.json' if enabled else 'primary.json')
        command=[sys.executable,'-m','scripts.dev.fdtd.printed_line_loss_control_preflight','--metal','copper','--sigma','.01','--dx','.00125','--longitudinal-dl','.0025','--probe-reference-dx','.0025','--output',str(output)]
        if enabled:command.append('--record-legacy-native-vi')
        environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='';environment['PYTHONDONTWRITEBYTECODE']='1'
        process=subprocess.run(command,env=environment,capture_output=True,text=True)
        assert process.returncode==0,process.stdout+process.stderr
        records.append(json.loads(output.read_text()))
    return records


def test_actual_compiled_grid_material_clock_and_all_Norton_stamp_fields_are_bitwise_equal(compiled_preflights):
    plain,dual=compiled_preflights
    for key in ('grid_node_sha256','domain_bounds_m','ROI_bounds_m','CPU_compiled_time_step_forecast_s','planned_steps_forecast','checkpoint_step_forecast',
                'physical_source_ports','material_native_components','source_basis','source_group_names','waveform','physical_VI_integration_rows','fixed_physical_VI_contract','native_observation_schema','copper_compiled_support'):
        assert plain[key]==dual[key],key
    assert plain['legacy_native_VI_contract'] is None and dual['legacy_native_VI_contract'] is not None
    for record in (plain,dual):
        assert record['source_before']==record['source_after']
        assert record['CUDA_initialized'] is False and record['CUDA_initialization_unchanged']
        assert record['actual_field_advance'] is False and record['actual_CUDA_preparation'] is False


@pytest.mark.parametrize('omit',[{'record_legacy_native_vi'}, {'record_legacy_native_vi','metal','longitudinal_dl','probe_reference_dx'}, set()])
def test_explicit_known_saved_case_schemas_read_without_changing_original_payload(omit):
    original=encode(asdict(PilotCase()));saved={name:value for name,value in original.items() if name not in omit}
    frozen=json.dumps(saved,sort_keys=True)
    assert canonical_case_payload(saved)==original
    assert json.dumps(saved,sort_keys=True)==frozen


@pytest.mark.parametrize('mutation',['missing_physical','partial_old_options','unknown','nonboolean'])
def test_case_compatibility_does_not_relax_other_physical_or_unknown_fields(mutation):
    payload=encode(asdict(PilotCase()))
    if mutation=='missing_physical':payload.pop('sigma')
    elif mutation=='partial_old_options':payload.pop('metal')
    elif mutation=='unknown':payload['extra_source_shape']='invented'
    else:payload['record_legacy_native_vi']=0
    with pytest.raises(ValueError):canonical_case_payload(payload)


@pytest.fixture(autouse=True)
def CUDA_initialization_is_unchanged():
    initial=torch.cuda.is_initialized();yield
    assert torch.cuda.is_initialized()==initial

@pytest.fixture
def archived_printed_lines():
    path=Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-28'
    if not path.exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-28')


@pytest.mark.usefixtures('archived_printed_lines')
@pytest.mark.parametrize('name,changes',[
    ('first64-actual-text',{}),
    ('material-control-actual-text/pec_sigma',{'metal':'pec','longitudinal_dl':.0025}),
    ('material-control-actual-text/copper_zero',{'sigma':0.,'longitudinal_dl':.0025}),
])
def test_actual_archived_execution_cases_keep_known_physical_contracts(name,changes):
    path=Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-28'/name/'runtime/drive0/execution.json'
    original=path.read_bytes();record=json.loads(original)
    assert canonical_case_payload(record['case'])==encode(asdict(replace(PilotCase(),**changes)))
    assert path.read_bytes()==original


@pytest.mark.parametrize('field,value',[('dx',.00125),('sigma',.02),('duration',32e-9),('padding',.04)])
def test_schema_default_reading_never_hides_a_changed_physical_value(field,value):
    payload=encode(asdict(PilotCase()));payload.pop('record_legacy_native_vi');payload[field]=value
    assert canonical_case_payload(payload)!=encode(asdict(PilotCase()))
