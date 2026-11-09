"""Host-only eligibility and fail-closed acceptance tests; no numerical oracle."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import pytest

PATH=Path(__file__).with_name('sie_combined_full_acceptance_v1.py')
SPEC=importlib.util.spec_from_file_location('combined_acceptance',PATH)
h=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(h)


def test_actual_candidate_has_exact_two_scientific_changes():
    row=h.verify_provider_pair(PATH.parents[1],Path(os.environ.get('WITWIN_SIE_COMPACT_RIESZ_BASELINE', Path(tempfile.gettempdir())/'witwin-maxwell-compact-riesz')))
    assert {n for n in row['candidate'] if row['candidate'][n]!=row['baseline'][n]}==set(h.REVIEWED_DELTAS)


@pytest.fixture
def history(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();folder=old/'witwin/maxwell/sie';folder.mkdir(parents=True)
    def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    git(old,'init','-q');git(old,'config','user.name','Xingyu Chen');git(old,'config','user.email','fixture@example.invalid')
    for name in ('operator.py','linalg.py','fields.py'):(folder/name).write_text('old=1\n')
    git(old,'add','.');git(old,'commit','-qm','Create independent source fixture')
    base=git(old,'rev-parse','HEAD');new=tmp_path/'new';subprocess.run(['git','clone','-q',str(old),str(new)],check=True)
    wanted={}
    for name in ('operator.py','linalg.py'):
        p=new/'witwin/maxwell/sie'/name;p.write_text('reviewed=2\n');wanted['sie/'+name]=h.sha(p.read_bytes())
    monkeypatch.setattr(h,'BASE_COMMIT',base);monkeypatch.setattr(h,'REVIEWED_DELTAS',wanted)
    return old,new,git


@pytest.mark.parametrize('mutation',['operator','linalg','other','extra','missing','baseline','baseline_head','unrelated'])
def test_two_delta_guard_never_allows_arbitrary_sources(history,mutation):
    old,new,git=history
    h.verify_provider_pair(new,old)
    if mutation in ('operator','linalg','other'):
        name={'operator':'operator.py','linalg':'linalg.py','other':'fields.py'}[mutation]
        (new/'witwin/maxwell/sie'/name).write_text('unreviewed=3\n')
    elif mutation=='extra':(new/'witwin/maxwell/sie/new.cuh').write_text('new header')
    elif mutation=='missing':(new/'witwin/maxwell/sie/fields.py').unlink()
    elif mutation=='baseline':(old/'witwin/maxwell/sie/fields.py').write_text('changed old=3\n')
    elif mutation=='baseline_head':git(old,'commit','--allow-empty','-qm','Change reference epoch')
    else:
        git(new,'config','user.name','Xingyu Chen');git(new,'config','user.email','fixture@example.invalid')
        git(new,'checkout','--orphan','other');git(new,'add','.');git(new,'commit','-qm','Create unrelated fixture')
    with pytest.raises(ValueError):h.verify_provider_pair(new,old)


def valid_receipts():
    nodes=['a','b'];phases=[{'nodeid':n,'when':p,'outcome':'passed','duration':.01,'wasxfail':None} for n in nodes for p in ('setup','call','teardown')]
    tests=[{'nodeid':n,'seconds':.03,'measurement_errors':[],'cuda':{d:dict(allocation_events=1,allocated_before=1,allocated_after=2,peak_allocated_bytes=3,peak_reserved_bytes=4) for d in ('0','1')}} for n in nodes]
    return nodes,phases,tests


@pytest.mark.parametrize('mutation',['xpass','skip','missing_phase','duplicate','wrong_order','bad_duration','missing_test','unknown_device','counter_missing','negative','peak','no_second_device_work','no_required_work','measurement','pytest','source'])
def test_terminal_verdict_rejects_incomplete_or_failed_evidence(mutation):
    nodes,phases,tests=valid_receipts();exitcode=0;source=True
    if mutation=='xpass':phases[1]['wasxfail']='historical expected failure'
    elif mutation=='skip':phases[1]['outcome']='skipped'
    elif mutation=='missing_phase':phases.pop()
    elif mutation=='duplicate':phases.append(copy.deepcopy(phases[0]))
    elif mutation=='wrong_order':tests.reverse()
    elif mutation=='bad_duration':phases[0]['duration']=float('nan')
    elif mutation=='missing_test':tests.pop()
    elif mutation=='unknown_device':tests[0]['cuda']['2']=tests[0]['cuda'].pop('1')
    elif mutation=='counter_missing':tests[0]['cuda']['0']['allocation_events']=None
    elif mutation=='negative':tests[0]['cuda']['0']['allocation_events']=-1
    elif mutation=='peak':tests[0]['cuda']['0']['peak_allocated_bytes']=0
    elif mutation=='no_second_device_work':
        for row in tests:row['cuda']['1']['allocation_events']=0
    elif mutation=='no_required_work':
        for row in tests[0]['cuda'].values():row['allocation_events']=0
    elif mutation=='measurement':tests[0]['measurement_errors']=['unavailable']
    elif mutation=='pytest':exitcode=1
    elif mutation=='source':source=False
    assert not h.grade(nodes,phases,tests,exitcode,source,['0','1'],['a'])['passed']


def test_complete_two_device_evidence_is_engineering_only():
    nodes,phases,tests=valid_receipts();assert h.grade(nodes,phases,tests,0,True,['0','1'],['a'])['passed']


@pytest.mark.parametrize('mutation',['native','core','benchmark','header','test','config'])
def test_full_inventory_binds_all_inputs(tmp_path,mutation):
    root=tmp_path/'provider';core=tmp_path/'site/witwin/core';core.mkdir(parents=True)
    files={
        'native':root/'witwin/maxwell/sie/operator.py', 'core':core/'mesh.py',
        'benchmark':root/'benchmark/sie_reference.py','header':core/'kernel.cuh',
        'test':root/'tests/sie/nested/test_more.py','config':root/'pytest.ini'}
    for p in files.values():p.parent.mkdir(parents=True,exist_ok=True);p.write_text('first bytes')
    helper=root/'benchmark/bound.py';helper.write_text('binding')
    before=h.inventory(root,core,helper);files[mutation].write_text('changed bytes')
    assert h.inventory(root,core,helper)!=before


@pytest.mark.parametrize('fault',['before','after','protocol'])
def test_real_hook_retains_partial_records_and_original_protocol_error(tmp_path,fault):
    import time
    tree=ast.parse(PATH.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='pytest_runtest_protocol');node.decorator_list=[]
    calls=[]
    def stats(i):
        calls.append(i)
        if (fault=='before' and len(calls)==1) or (fault=='after' and len(calls)==3):raise RuntimeError('counter unavailable')
        return {'allocation.all.allocated':10+len(calls)}
    cuda=SimpleNamespace(synchronize=lambda i:None,reset_peak_memory_stats=lambda i:None,memory_stats=stats,memory_allocated=lambda i:4,max_memory_allocated=lambda i:8,max_memory_reserved=lambda i:16)
    env={'time':time,'devices':['0','1'],'torch':SimpleNamespace(cuda=cuda)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(PATH),'exec'),env)
    evidence=SimpleNamespace(tests=[],active=None,active_started=None)
    iterator=env['pytest_runtest_protocol'](evidence,SimpleNamespace(nodeid='actual'),None);next(iterator)
    if fault=='protocol':
        with pytest.raises(RuntimeError,match='original'):iterator.throw(RuntimeError('original protocol'))
    else:
        with pytest.raises(StopIteration):next(iterator)
    assert evidence.active is None and len(evidence.tests)==1 and set(evidence.tests[0]['cuda'])=={'0','1'}
    assert evidence.tests[0]['seconds']>0
    if fault!='protocol':assert evidence.tests[0]['measurement_errors']


def test_helper_executes_only_the_single_verified_read(tmp_path,monkeypatch):
    helper=tmp_path/'binding.py';helper.write_text('marker="trusted first buffer"\n')
    trusted=helper.read_bytes();wanted=h.sha(trusted)
    monkeypatch.setattr(h,'BINDING_HELPER_SHA256',wanted)
    original=Path.read_bytes;calls=[]
    def alternating(path):
        if path==helper:
            calls.append(path)
            return trusted if len(calls)==1 else b'raise RuntimeError("unverified second read executed")\n'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',alternating)
    module=h.load_binding_helper(helper,wanted)
    assert module.marker=='trusted first buffer' and len(calls)==1


def test_unverified_helper_bytes_never_execute(tmp_path):
    helper=tmp_path/'binding.py';helper.write_text('raise AssertionError("untrusted source executed")\n')
    with pytest.raises(ValueError,match='Wrong reviewed'):h.load_binding_helper(helper,h.BINDING_HELPER_SHA256)
