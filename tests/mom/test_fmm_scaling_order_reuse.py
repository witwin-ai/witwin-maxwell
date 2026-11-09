"""Order reuse preserves source, physical cases and original admission gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

path = Path(__file__).resolve().parents[2]/'scripts/dev/measure_mom_fmm_scaling.py'
spec = importlib.util.spec_from_file_location('scaling_order_reuse', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def selection(family, level, order=6):
    return dict(version='scalar_fmm_reused_order_selection_v1',
        parameters=runner.case_parameters(family, level), plan_hash=runner.plan_hash(),
        production_source_hash=runner.production_source_hash(), selected_order=order,
        production_qualification=False, source_kind='fresh_current_order_plan',
        source_path=None, source_sha256=None)


def closed_bundle(tmp_path):
    selections = {}
    for family, order in [('fixed_ka',11), ('growing_ka',12)]:
        item = selection(family,3,order);parameters=item['parameters']
        raw = dict(order_admitted=True,family=family,frequency_Hz=parameters['frequency'],
            edges=3840,kernel_bound_rule=parameters['kernel_bound_rule'],
            order_plan=dict(selected_order=order,kernel_bound_rule=parameters['kernel_bound_rule'],
                kernel_budget=parameters['kernel_budget'],attempts=[dict(order=order,kernel_pass=True,
                    unavailable_bounds=0,green_absolute_max=1e-7,gradient_component_absolute_max=1e-6)]))
        path=tmp_path/(family+'.json');path.write_text(json.dumps(raw))
        item.update(source_kind='closed_original_admission',source_path=str(path),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        selections[family+':3']=item
    path=tmp_path/'orders.json';path.write_text(json.dumps(dict(
        version='scalar_fmm_closed_order_manifest_v1',plan_hash=runner.plan_hash(),
        production_source_hash=runner.production_source_hash(),selections=selections)))
    return path


def test_closed_receipts_bind_each_family_and_original_gates(tmp_path):
    path=closed_bundle(tmp_path);bundle=runner.load_closed_order_manifest(path)
    assert bundle['selections']['fixed_ka:3']['selected_order']==11
    assert bundle['selections']['growing_ka:3']['selected_order']==12
    assert runner.PLAN['kernel_bound_rules']['growing_ka']=='direct_first_wave_min_v1'
    assert runner.PLAN['native_action_tolerance']==1e-11


@pytest.mark.parametrize('mutation', ['missing_large','source_sha','production','plan','order_bool',
    'wrong_frequency','wrong_rule','kernel_fail','unavailable','green_over','gradient_nan','qualification'])
def test_closed_receipts_reject_changed_admission(tmp_path,mutation):
    path=closed_bundle(tmp_path);bundle=json.loads(path.read_text());item=bundle['selections']['growing_ka:3']
    if mutation=='missing_large':del bundle['selections']['growing_ka:3']
    elif mutation=='source_sha':item['source_sha256']='0'*64
    elif mutation=='production':bundle['production_source_hash']='0'*64
    elif mutation=='plan':item['plan_hash']='0'*64
    elif mutation=='order_bool':item['selected_order']=True
    elif mutation=='qualification':item['production_qualification']=True
    else:
        source=Path(item['source_path']);raw=json.loads(source.read_text());final=raw['order_plan']['attempts'][-1]
        if mutation=='wrong_frequency':raw['frequency_Hz']=3e8
        elif mutation=='wrong_rule':raw['order_plan']['kernel_bound_rule']='positive_tail_min_v1'
        elif mutation=='kernel_fail':final['kernel_pass']=False
        elif mutation=='unavailable':final['unavailable_bounds']=1
        elif mutation=='green_over':final['green_absolute_max']=1.01e-6
        elif mutation=='gradient_nan':final['gradient_component_absolute_max']=float('nan')
        source.write_text(json.dumps(raw));item['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):runner.load_closed_order_manifest(path)


def first_record(tmp_path):
    item=selection('growing_ka',1,8);parameters=item['parameters']
    record=dict(identity='id',revision='rev',source_hash='sha',case=15,status='measured',
        family='growing_ka',subdivision=1,repetition=0,selected_order=8,order_selection=item,
        edges=parameters['edges'],frequency=parameters['frequency'],native_actions_gate_passed=True,
        adjoint_relative_error=1e-15,current_gradient_relative_error=0.,
        **{k:parameters[k] for k in ('kernel_bound_rule','table_storage','near_storage','shift_storage')})
    path=tmp_path/'case-15.json';path.write_text(json.dumps(record));return path


def test_repetition_reuses_first_case_without_changing_measurement_identity(tmp_path):
    path=first_record(tmp_path);item=runner.reused_case_order(path,16,'id','rev','sha')
    assert item['selected_order']==8 and item['source_kind']=='current_first_repetition'
    assert item['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not item['production_qualification']


@pytest.mark.parametrize('field,value', [('case',12),('source_hash','other'),('status','failed'),
    ('selected_order',9),('frequency',3e8),('edges',60),('kernel_bound_rule','positive_tail_min_v1'),
    ('adjoint_relative_error',-1.),('current_gradient_relative_error',None)])
def test_repetition_rejects_wrong_case_or_failed_native_gate(tmp_path,field,value):
    path=first_record(tmp_path);record=json.loads(path.read_text());record[field]=value;path.write_text(json.dumps(record))
    with pytest.raises(ValueError):runner.reused_case_order(path,16,'id','rev','sha')


def test_coordinator_does_not_replay_failed_first_case(tmp_path,monkeypatch):
    def identity(order_manifest=None):
        assert order_manifest is None
        return 'id','rev','sha'
    monkeypatch.setattr(runner,'source_identity',identity)
    path=tmp_path/'case-00.json';path.write_text(json.dumps(dict(
        identity='id',revision='rev',source_hash='sha',case=0,status='failed')))
    def forbidden(*args,**kwargs):raise AssertionError('failed admission was replayed')
    monkeypatch.setattr(runner.subprocess,'run',forbidden)
    monkeypatch.setattr(sys,'argv',['runner','--case','1','--output',str(tmp_path)])
    runner.main();record=json.loads((tmp_path/'case-01.json').read_text())
    assert record['status']=='failed' and record['not_launched']


def test_coordinator_rejects_foreign_failed_first_case(tmp_path,monkeypatch):
    def identity(order_manifest=None):
        assert order_manifest is None
        return 'id','rev','sha'
    monkeypatch.setattr(runner,'source_identity',identity)
    (tmp_path/'case-00.json').write_text(json.dumps(dict(
        identity='id',revision='foreign',source_hash='sha',case=0,status='failed')))
    monkeypatch.setattr(sys,'argv',['runner','--case','1','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='identity'):runner.main()
