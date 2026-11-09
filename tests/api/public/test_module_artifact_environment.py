"""Real CPU subprocess contracts for exact module-scoped artifact environments."""
from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,sys,time
import pytest

ROOT=Path(os.environ['PCB_MODULE_ENV_SOURCE']) if 'PCB_MODULE_ENV_SOURCE' in os.environ else Path(__file__).resolve().parents[3]
CONTROL=Path(os.environ.get('PCB_MODULE_ENV_CONTROL',str(ROOT/'scripts/dev/fdtd/regression_shards')))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope='session')
def package_copy(tmp_path_factory):
    root=tmp_path_factory.mktemp('real-package')
    shutil.copytree(ROOT/'witwin/maxwell',root/'witwin/maxwell',ignore=shutil.ignore_patterns('__pycache__'))
    return root


@pytest.fixture
def project(tmp_path,package_copy):
    source=tmp_path/'source';source.mkdir()
    shutil.copytree(package_copy/'witwin',source/'witwin',copy_function=os.link)
    (source/'tests').mkdir();(tmp_path/'inputs').mkdir()
    return tmp_path,source


def policy(project,modules):
    root,source=project;files={};entries={}
    for name,cases in modules.items():
        data=root/'inputs'/f'{name}.json';data.write_text(json.dumps(cases));files[str(data)]=sha(data)
        entries[f'tests/test_{name}.py']={'environment':{'GSP_TEST_RESOURCE':str(data),'GSP_TEST_RESOURCE_SHA256':sha(data)}}
    path=root/'policy.json';path.write_text(json.dumps(dict(schema='module-scoped-artifact-environment/1',files=files,modules=entries)))
    return path


def write(source,name,body):
    path=source/'tests'/f'test_{name}.py';path.write_text(body);return path


def launch(project,path,*,action='run',resume=False,node_timeout=10):
    root,source=project
    argv=[sys.executable,str(CONTROL/'run_shards.py'),action,'--source-root',str(source),'--output',str(root/'result'),'--module-environment',str(path),'--node-timeout',str(node_timeout),'--shard-timeout','90','--termination-grace','.2','--poll-seconds','.05']
    if resume:argv+=['--resume']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',GSP_TEST_RESOURCE='must-not-leak',GSP_UNSCOPED='must-not-leak',PCB_SCOPE_PROBE=str(root/'probe.json'))
    log=root/f'controller-{time.time_ns()}.log'
    stream=log.open('wb');process=subprocess.Popen(argv,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);stream.close()
    return process,log


def finish(process,log,expected=0):
    try:code=process.wait(timeout=60)
    except BaseException:
        os.kill(process.pid,signal.SIGTERM);process.wait(timeout=10);raise
    assert code==expected,log.read_text()


def resource_module(expected):
    return f'''import hashlib,json,os,pytest
from pathlib import Path
IMPORTED=os.environ['GSP_TEST_RESOURCE']
CASES=json.loads(Path(IMPORTED).read_text())
@pytest.mark.parametrize('case',CASES)
def test_resource(case):
    assert os.environ['GSP_TEST_RESOURCE']==IMPORTED
    assert hashlib.sha256(Path(IMPORTED).read_bytes()).hexdigest()==os.environ['GSP_TEST_RESOURCE_SHA256']
    assert case in {expected!r}
    assert 'GSP_UNSCOPED' not in os.environ
'''


def test_actual_collection_and_execution_preserve_conflicting_module_parameters(project):
    root,source=project;write(source,'a',resource_module(['a0','a1']));write(source,'b',resource_module(['b0']))
    write(source,'unscoped',"import os\ndef test_unscoped():assert not any(k.startswith('GSP_') for k in os.environ)\n")
    path=policy(project,{'a':['a0','a1'],'b':['b0']});process,log=launch(project,path);finish(process,log)
    summary=json.loads((root/'result/summary.json').read_text());contract=json.loads((root/'result/contract.json').read_text())
    assert summary['counts']=={'passed':4} and summary['collected']==4
    assert contract['selection']==['tests/test_a.py::test_resource[a0]','tests/test_a.py::test_resource[a1]','tests/test_b.py::test_resource[b0]','tests/test_unscoped.py::test_unscoped']
    identities=[json.loads(p.read_text()) for p in (root/'result').glob('attempt-*/identity-before.json')]
    assert len(identities)==3
    assert {r['gsp_environment'].get('GSP_TEST_RESOURCE') for r in identities}=={str(root/'inputs/a.json'),str(root/'inputs/b.json'),None}
    assert not contract['environment']['GSP_input_files']
    assert os.environ.get('GSP_TEST_RESOURCE')!='must-not-leak'


@pytest.mark.parametrize('mutation',['missing','sha','sha_pair','orphan','unregistered','module_escape','missing_module'])
def test_invalid_artifact_policy_refuses_before_collection(project,mutation):
    root,source=project;write(source,'a','def test_ok():pass\n');path=policy(project,{'a':[1]});raw=json.loads(path.read_text());entry=raw['modules']['tests/test_a.py']
    if mutation=='missing':(root/'inputs/a.json').unlink()
    elif mutation=='sha':raw['files'][str(root/'inputs/a.json')]='0'*64
    elif mutation=='sha_pair':entry['environment']['GSP_TEST_RESOURCE_SHA256']='0'*64
    elif mutation=='orphan':entry['environment']['GSP_UNKNOWN_SHA256']='0'*64
    elif mutation=='unregistered':entry['environment']['CUDA_VISIBLE_DEVICES']='1'
    elif mutation=='module_escape':raw['modules']['../bad.py']=entry
    elif mutation=='missing_module':raw['modules']['tests/test_missing.py']=entry
    path.write_text(json.dumps(raw));process,log=launch(project,path);finish(process,log,1)
    assert not (root/'result').exists()
    assert 'ValueError' in log.read_text()


def test_nonstandard_original_source_sha_pair_is_explicitly_supported(project):
    root,source=project;write(source,'a','import os\ndef test_ok():assert os.environ["GSP_ORIGINAL_NEARFIELD_SOURCE"]\n');path=policy(project,{'a':[1]});raw=json.loads(path.read_text());entry=raw['modules']['tests/test_a.py'];values=entry['environment'];entry['environment']={'GSP_ORIGINAL_NEARFIELD_SOURCE':values['GSP_TEST_RESOURCE'],'GSP_ORIGINAL_NEARFIELD_SHA256':values['GSP_TEST_RESOURCE_SHA256']};entry['sha_pairs']={'GSP_ORIGINAL_NEARFIELD_SHA256':'GSP_ORIGINAL_NEARFIELD_SOURCE'};path.write_text(json.dumps(raw));process,log=launch(project,path);finish(process,log)
    assert json.loads((root/'result/summary.json').read_text())['counts']=={'passed':1}


def test_module_import_cannot_silently_mutate_scoped_environment(project):
    root,source=project;write(source,'a','import os\nos.environ["GSP_UNDECLARED"]="bad"\ndef test_ok():pass\n');path=policy(project,{'a':[1]});process,log=launch(project,path);finish(process,log,1)
    summary=json.loads((root/'result/summary.json').read_text());assert summary['state']=='controller_error_not_qualification'
    assert 'Complete collection did not succeed' in log.read_text()


def test_original_failure_skip_and_xfail_outcomes_are_not_rewritten(project):
    root,source=project;write(source,'a','import os,pytest\ndef test_pass():assert os.environ["GSP_TEST_RESOURCE"]\ndef test_fail():assert False,"original failure"\ndef test_skip():pytest.skip("original missing optional evidence")\n@pytest.mark.xfail(reason="original limitation")\ndef test_xfail():assert False\n');path=policy(project,{'a':[1]});process,log=launch(project,path);finish(process,log,1)
    summary=json.loads((root/'result/summary.json').read_text());assert summary['counts']=={'passed':1,'failed':1,'skipped':1,'xfailed':1}
    ledger=json.loads((root/'result/node-outcomes.json').read_text());assert 'original failure' in ledger['tests/test_a.py::test_fail']['reports'][1]['original_detail']
    assert summary['all_nodes_terminal'] and not summary['all_nodes_passed']


def test_changed_artifact_refuses_resume_of_a_successful_collection(project):
    root,source=project;write(source,'a',resource_module([1]));path=policy(project,{'a':[1]});process,log=launch(project,path,action='collect');finish(process,log)
    (root/'inputs/a.json').write_text('[2]');process,log=launch(project,path,resume=True);finish(process,log,1)
    assert 'Missing or changed module artifact' in log.read_text()
    assert not list((root/'result').glob('attempt-*'))


def test_controller_interrupt_and_resume_keep_same_module_input(project):
    root,source=project
    write(source,'a',"import json,os,time\nfrom pathlib import Path\ndef test_resume():\n p=Path(os.environ['PCB_SCOPE_PROBE'])\n old=json.loads(p.read_text()) if p.exists() else []\n old.append(os.environ['GSP_TEST_RESOURCE']);p.write_text(json.dumps(old))\n if len(old)==1:time.sleep(20)\n assert len(old)==2\n")
    write(source,'b',resource_module([2]));path=policy(project,{'a':[1],'b':[2]});process,log=launch(project,path)
    deadline=time.monotonic()+40
    while time.monotonic()<deadline and not (root/'probe.json').exists() and process.poll() is None:time.sleep(.05)
    assert (root/'probe.json').exists(),log.read_text();os.kill(process.pid,signal.SIGTERM);finish(process,log,143)
    assert json.loads((root/'result/summary.json').read_text())['state']=='interrupted_resumable'
    process,log=launch(project,path,resume=True);finish(process,log)
    summary=json.loads((root/'result/summary.json').read_text());assert summary['counts']=={'passed':2}
    assert json.loads((root/'probe.json').read_text())==[str(root/'inputs/a.json')]*2
    attempts=[json.loads(p.read_text()) for p in (root/'result').glob('attempt-*/attempt-result.json')]
    assert any(any(d['outcome']=='interrupted' for d in r['decisions'].values()) for r in attempts)


def test_runtime_artifact_environment_leak_is_not_a_passing_shard(project):
    root,source=project;write(source,'a','import os\ndef test_changes_environment():os.environ["GSP_UNDECLARED"]="leak"\n');path=policy(project,{'a':[1]});process,log=launch(project,path);finish(process,log,1)
    assert json.loads((root/'result/summary.json').read_text())['state']=='controller_error_not_qualification'
    reports=[json.loads(line) for p in (root/'result').glob('attempt-*/events.jsonl') for line in p.read_text().splitlines()]
    assert any(e['event']=='test_report' and e['phase']=='call' and e['outcome']=='passed' for e in reports)
    assert 'Missing/mismatched actual child import identity' in log.read_text()


def test_valid_but_different_module_environment_cannot_resume_collection(project):
    root,source=project;write(source,'a',resource_module([1]));write(source,'b',resource_module([2]));path=policy(project,{'a':[1],'b':[2]});process,log=launch(project,path,action='collect');finish(process,log)
    raw=json.loads(path.read_text());raw['modules']['tests/test_a.py']=raw['modules']['tests/test_b.py'];path.write_text(json.dumps(raw));process,log=launch(project,path,resume=True);finish(process,log,1)
    assert 'Resume config or selection differs' in log.read_text()
    assert not list((root/'result').glob('attempt-*'))


def test_timeout_preserves_nonpass_and_later_module_inputs(project):
    root,source=project;write(source,'a','import os,time\ndef test_hang():\n assert os.environ["GSP_TEST_RESOURCE"]\n time.sleep(20)\n');write(source,'b',resource_module([2]));path=policy(project,{'a':[1],'b':[2]});process,log=launch(project,path,node_timeout=.3);finish(process,log,1)
    summary=json.loads((root/'result/summary.json').read_text());assert summary['counts']=={'timeout':1,'passed':1}
    assert summary['all_nodes_terminal'] and not summary['all_nodes_passed']


def test_abrupt_exit_preserves_crash_and_later_module_inputs(project):
    root,source=project;write(source,'a','import os\ndef test_crash():\n assert os.environ["GSP_TEST_RESOURCE"]\n os._exit(17)\n');write(source,'b',resource_module([2]));path=policy(project,{'a':[1],'b':[2]});process,log=launch(project,path);finish(process,log,1)
    summary=json.loads((root/'result/summary.json').read_text());assert summary['counts']=={'process_crashed':1,'passed':1}
    assert summary['infrastructure_events']
