"""Combined assembly/storage/bounded-validation affected-provider acceptance; no physical label."""
from pathlib import Path
import argparse,contextlib,hashlib,subprocess,importlib.abc,importlib.machinery,importlib.util,json,math,os,sys,time
import xml.etree.ElementTree as ET

SCOPE=['tests/sie','tests/api/public/test_public_api.py']
BASE_COMMIT='e95ab35a87e4259f329fbf8b0dfaecbea45eb0c7'
REVIEWED_DELTAS={
 'sie/operator.py':'eb491a09d5858d92acaddd6b79be20943e8fbcda2e7dc9fac0c99f42638b87d4',
 'sie/linalg.py':'7aedf3c1962c4321c8f4575c50857734aacd0e853c20278387d703e299ea69bf',
 'sie/mesh.py':'3a9ff417ba454f04044b0d5927db8c7bf239197f7253b783f58c98132a55477d'}
BINDING_HELPER_SHA256='f996a2d65ecddec6d25c42c1ea195fa8bc14dcbab6e3dfbd4711ad2675b69725'
PYTHON='/home/xingyu/miniconda3/envs/maxwell/bin/python'
SOURCE_SUFFIXES={'.py','.cpp','.cu','.cuh','.h','.so'}

def scientific_inventory(root):
 root=Path(root)
 return {str(p.relative_to(root)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
         for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts}

def verify_provider_pair(candidate,baseline):
 candidate,baseline=Path(candidate).resolve(),Path(baseline).resolve()
 heads={name:subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        for name,root in [('candidate',candidate),('baseline',baseline)]}
 if heads['baseline']!=BASE_COMMIT:raise ValueError('Baseline HEAD differs from reviewed e95')
 if subprocess.run(['git','-C',str(candidate),'merge-base','--is-ancestor',BASE_COMMIT,heads['candidate']],capture_output=True).returncode:
  raise ValueError('Combined candidate does not descend from e95')
 old,new=scientific_inventory(baseline/'witwin/maxwell'),scientific_inventory(candidate/'witwin/maxwell')
 tracked=subprocess.check_output(['git','-C',str(baseline),'ls-tree','-r','--name-only',BASE_COMMIT],text=True).splitlines()
 expected={n.removeprefix('witwin/maxwell/') for n in tracked if n.startswith('witwin/maxwell/') and Path(n).suffix in SOURCE_SUFFIXES}
 if set(old)!=expected or set(new)!=expected:raise ValueError('Scientific inventory differs from exact e95')
 for name in expected:
  blob=subprocess.check_output(['git','-C',str(baseline),'show',BASE_COMMIT+':witwin/maxwell/'+name])
  original=sha(blob)
  if original!=old[name]['sha256']:raise ValueError('Baseline bytes differ from e95: '+name)
  if new[name]['sha256']!=REVIEWED_DELTAS.get(name,original):raise ValueError('Unreviewed combined scientific source: '+name)
 return {'baseline':old,'candidate':new,'heads':heads,'reviewed_scientific_deltas':REVIEWED_DELTAS.copy(),'protocol':'combined-assembly-direct-storage-v2'}

def load_binding_helper(driver,expected):
 data=Path(driver).read_bytes()
 if expected!=BINDING_HELPER_SHA256 or sha(data)!=BINDING_HELPER_SHA256:raise ValueError('Wrong reviewed equivalence driver')
 spec=importlib.util.spec_from_file_location('reviewed_equivalence_driver',driver)
 module=importlib.util.module_from_spec(spec)
 exec(compile(data,str(driver),'exec'),module.__dict__)
 return module

def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def canonical(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
def inventory(root,core,driver):
 paths=set(p.resolve() for p in (root/'benchmark').rglob('*.py') if '__pycache__' not in p.parts)
 for base in [root/'witwin/maxwell',core,root/'tests/sie']:
  paths.update(p.resolve() for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))
 for path in [root/'tests/api/public/test_public_api.py',root/'tests/conftest.py',root/'tests/api/conftest.py',root/'tests/api/public/conftest.py',root/'pyproject.toml',root/'setup.cfg',root/'pytest.ini',root/'tox.ini',Path(__file__).resolve(),driver,root/'docs/assessments/sie_single_buffer_equivalence_protocol_v3.md',root/'docs/reference/sie_direct_storage.md',root/'docs/assessments/sie_combined_full_acceptance_v1.md',root/'docs/assessments/sie_combined_full_acceptance_v2.md',root/'docs/assessments/sie_bounded_validation_v1.md']:
  if path.is_file():paths.add(path.resolve())
 # RECORD and installed distribution metadata are inputs, not a build claim.
 for dist in core.parents[1].glob('*core*.dist-info'):
  paths.update(p.resolve() for p in dist.iterdir() if p.is_file())
 return {str(p):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(paths)}
class BenchmarkSources(importlib.abc.MetaPathFinder):
 """Bind only this candidate's benchmark namespace and Python modules."""
 def __init__(self,root,bound):
  if any(n=='benchmark' or n.startswith('benchmark.') for n in sys.modules):raise ValueError('benchmark already imported before archive binding')
  self.root=(Path(root)/'benchmark').resolve();self.bound=bound;self.namespaces={};self.loaded={};self.package_paths={}
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='benchmark' and not fullname.startswith('benchmark.'):return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  if spec is None:raise ValueError('Unarchived benchmark module resolution')
  owner=self
  if spec.origin is None:
   locations=tuple(str(Path(p).resolve()) for p in (spec.submodule_search_locations or []))
   if fullname!='benchmark' or locations!=(str(self.root),):raise ValueError('Foreign benchmark namespace path')
   class NamespaceLoader(importlib.abc.Loader):
    def create_module(self,spec):return None
    def exec_module(self,module):owner.namespaces[fullname]=(module,locations)
   spec.loader=NamespaceLoader();return spec
  origin=str(Path(spec.origin).resolve())
  if not Path(origin).is_relative_to(self.root) or origin not in self.bound.files or not origin.endswith('.py'):raise ValueError('Unarchived benchmark import origin')
  class Loader(importlib.abc.Loader):
   def create_module(self,spec):return None
   def exec_module(self,module):
    row=owner.bound.files[origin];data=Path(row['archive']).read_bytes()
    if sha(data)!=row['sha256'] or sha(Path(origin).read_bytes())!=row['sha256']:raise ValueError('Benchmark source changed before first execution')
    module.__file__=origin;exec(compile(data,origin,'exec'),module.__dict__)
    owner.loaded[fullname]=(module,origin);owner.bound.modules[fullname]=(module,origin)
    if hasattr(module,'__path__'):owner.package_paths[fullname]=(module,(str(Path(origin).parent),))
  spec.loader=Loader();return spec
 def verify(self):
  actual={n for n in sys.modules if n=='benchmark' or n.startswith('benchmark.')}
  if actual!=set(self.loaded)|set(self.namespaces):raise ValueError('Unbound benchmark module appeared')
  for origin,row in self.bound.files.items():
   if Path(origin).is_relative_to(self.root) and (sha(Path(origin).read_bytes())!=row['sha256'] or sha(Path(row['archive']).read_bytes())!=row['sha256']):raise ValueError('Benchmark source/archive changed')
  for name,(obj,origin) in self.loaded.items():
   if sys.modules.get(name) is not obj or self.bound.modules.get(name)!=(obj,origin):raise ValueError('Benchmark module object replaced')
  for name,(obj,locations) in (self.namespaces|self.package_paths).items():
   if sys.modules.get(name) is not obj or tuple(str(Path(p).resolve()) for p in obj.__path__)!=locations:raise ValueError('Benchmark namespace replaced')
  return {'loaded_modules':{n:{'origin':o,'sha256':self.bound.files[o]['sha256'],'module_object_id':id(obj)} for n,(obj,o) in self.loaded.items()},'namespaces':{n:{'paths':list(p),'module_object_id':id(obj)} for n,(obj,p) in self.namespaces.items()}}

def preserve_partial(report,evidence):
 if evidence is None:return
 report.update(ordered_nodeids=evidence.nodes,phases=evidence.phases,per_test=evidence.tests,collection=evidence.collection,deselected=evidence.deselected,warnings=evidence.warnings)
 if evidence.active is not None:
  report['incomplete_test_protocol']={**evidence.active,'partial_elapsed_seconds':time.monotonic()-evidence.active_started,'scope':'interrupted protocol; only available measurements retained'}

def check_catalog(catalog,source,nodeids=None):
 if catalog.get('scope')!=SCOPE or catalog.get('source_inventory_sha256')!=canonical(source):raise ValueError('Unreviewed source/config/test inventory')
 nodes=catalog.get('ordered_nodeids')
 if not isinstance(nodes,list) or not nodes or len(nodes)!=len(set(nodes)) or any(not isinstance(x,str) or not x for x in nodes):raise ValueError('Malformed or duplicate frozen node IDs')
 if nodeids is not None and nodes!=nodeids:raise ValueError('Actual ordered collection differs from frozen catalog')

def grade(nodes,phases,per_test,exitstatus,source_ok,devices,required_cuda_nodes=()):
 errors=[]
 if exitstatus!=0:errors.append('pytest nonzero exit')
 if not source_ok:errors.append('source or initial module binding changed')
 if len(nodes)!=len(set(nodes)) or not nodes:errors.append('empty/duplicate collection')
 if [x['nodeid'] for x in per_test]!=nodes:errors.append('executed test protocols differ from collection')
 expected=[(n,p) for n in nodes for p in ('setup','call','teardown')]
 if [(x['nodeid'],x['when']) for x in phases]!=expected:errors.append('missing/duplicate/out-of-order phases')
 if any(type(x.get('duration')) not in (int,float) or not math.isfinite(x['duration']) or x['duration']<0 for x in phases):errors.append('malformed phase duration')
 if any(x['outcome']!='passed' or x.get('wasxfail') is not None for x in phases):errors.append('failed/skipped/xfail/xpass phase remains')
 observed={d:0 for d in devices}
 for t in per_test:
  if t.get('measurement_errors'):errors.append('CUDA measurement failed')
  if set(t.get('cuda',{}))!=set(devices):errors.append('missing requested device observation')
  if type(t.get('seconds')) not in (float,int) or not math.isfinite(t['seconds']) or t['seconds']<=0:errors.append('malformed test duration')
  for d,row in t.get('cuda',{}).items():
   if d not in observed or any(type(row.get(k)) is not int or row[k]<0 for k in ('allocation_events','allocated_before','allocated_after','peak_allocated_bytes','peak_reserved_bytes')):errors.append('malformed actual CUDA counters');continue
   if row['peak_allocated_bytes']<max(row['allocated_before'],row['allocated_after']):errors.append('inconsistent actual CUDA peak')
   observed[d]+=row['allocation_events']
 if any(v<=0 for v in observed.values()):errors.append('requested device has no actual test allocation evidence')
 if any(n not in nodes for n in required_cuda_nodes):errors.append('Required CUDA node absent from collection')
 by_node={r['nodeid']:r for r in per_test}
 for node in required_cuda_nodes:
  row=by_node.get(node,{})
  if sum(v.get('allocation_events',0) for v in row.get('cuda',{}).values() if type(v.get('allocation_events')) is int)<=0:errors.append('Required direct CUDA node has no actual allocation: '+node)
 return {'passed':not errors,'errors':errors,'actual_test_allocation_events_by_device':observed}

def verify_junit(path,nodes):
 from _pytest.junitxml import mangle_test_address
 cases=ET.parse(path).findall('.//testcase')
 expected=[]
 for node in nodes:
  parts=mangle_test_address(node);expected.append(('.'.join(parts[:-1]),parts[-1]))
 if [(x.get('classname'),x.get('name')) for x in cases]!=expected:raise ValueError('JUnit exact ordered testcase identities differ')
 if any(x.find(k) is not None for x in cases for k in ('failure','error','skipped')):raise ValueError('JUnit contains a nonpassing testcase')
 return {'exact_ordered_testcases':len(cases),'sha256':sha(Path(path).read_bytes())}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['collect','run'],required=True)
 parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--equivalence-driver-sha256',required=True)
 parser.add_argument('--catalog',type=Path);parser.add_argument('--reviewed-catalog-sha256');parser.add_argument('--expected-device-count',type=int,default=2)
 args=parser.parse_args()
 if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise ValueError('Wrong pinned Python')
 if args.expected_device_count!=2:raise ValueError('Full affected acceptance requires both reviewed CUDA devices')
 root=args.candidate.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 started=time.time();report={'mode':args.mode,'scope':SCOPE,'status':'failed','engineering_acceptance_passed':False,'H2_G0':'not_assessed','pid':os.getpid(),'python':sys.executable,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_unix':started};code=1;bound=None;e=None;bench_bound=None
 try:
  driver=root/'benchmark/sie_single_buffer_equivalence_v3.py'
  m=load_binding_helper(driver,args.equivalence_driver_sha256)
  pair=verify_provider_pair(root,args.baseline);report['provider_pair']=pair
  sys.path.insert(0,str(root));os.chdir(root)
  core_spec=importlib.util.find_spec('witwin.core')
  if core_spec is None or not core_spec.origin:raise ValueError('Actual installed core unavailable')
  core=Path(core_spec.origin).resolve().parent;before=inventory(root,core,driver)
  if before[str(driver)]['sha256']!=BINDING_HELPER_SHA256:raise ValueError('Executed helper changed before archive binding')
  bound=m.BoundSources([Path(x) for x in before],out/'source');sys.meta_path.insert(0,bound)
  bench_bound=BenchmarkSources(root,bound);sys.meta_path.insert(0,bench_bound)
  write(out/'source-before.json',before);write(out/'executed-arguments.json',vars(args)|{'candidate':str(args.candidate),'baseline':str(args.baseline),'output':str(args.output),'catalog':str(args.catalog) if args.catalog else None})
  if args.mode=='run':
   if args.catalog is None or not args.reviewed_catalog_sha256 or sha(args.catalog.read_bytes())!=args.reviewed_catalog_sha256:raise ValueError('Missing exact independently reviewed collection catalog')
   catalog=read(args.catalog);check_catalog(catalog,before);write(out/'executed-catalog.json',catalog)
  import pytest,torch
  if args.mode=='run' and (not torch.cuda.is_available() or torch.cuda.device_count()!=args.expected_device_count):raise ValueError('Requested actual CUDA device count unavailable')
  devices=[str(i) for i in range(args.expected_device_count)] if args.mode=='run' else []
  torch_build={'version':torch.__version__,'git':torch.version.git_version,'cuda':torch.version.cuda}
  if args.mode=='run' and catalog.get('torch_build')!=torch_build:raise ValueError('Torch build differs from reviewed collection')
  report['runtime']={'torch_build':torch_build,'core_origin':str(core),'devices':{d:torch.cuda.get_device_name(int(d)) for d in devices}}
  class Evidence:
   def __init__(self):self.nodes=[];self.phases=[];self.tests=[];self.collection=[];self.deselected=[];self.warnings=[];self.collection_binding=None;self.active=None;self.active_started=None;self.required_cuda_nodes=[]
   def pytest_collection_finish(self,session):
    self.nodes=[item.nodeid for item in session.items]
    self.required_cuda_nodes=[item.nodeid for item in session.items if item.path.name in ('test_direct_restore.py','test_combined_direct_storage.py') and any(mark.kwargs.get('reason')=='CUDA required' for mark in item.iter_markers('skipif'))]
    if len(self.required_cuda_nodes)!=22:pytest.exit('Direct-storage CUDA catalog changed',returncode=2)
    if args.mode=='run' and self.required_cuda_nodes!=catalog.get('required_cuda_nodeids'):pytest.exit('Required CUDA catalog changed',returncode=2)
    if args.mode=='run':
     try:check_catalog(catalog,before,self.nodes)
     except ValueError as error:pytest.exit(str(error),returncode=2)
    self.collection_binding=bound.verify();self.benchmark_collection_binding=bench_bound.verify()
    self.initial_ids={name:id(obj) for name,(obj,_) in bound.modules.items()}
   def pytest_collectreport(self,report):self.collection.append({'nodeid':report.nodeid,'outcome':report.outcome,'longrepr':str(report.longrepr) if report.failed else None})
   def pytest_deselected(self,items):self.deselected.extend(x.nodeid for x in items)
   def pytest_warning_recorded(self,warning_message,when,nodeid,location):self.warnings.append({'message':str(warning_message.message),'when':when,'nodeid':nodeid})
   def pytest_runtest_logreport(self,report):self.phases.append({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome,'duration':report.duration,'wasxfail':getattr(report,'wasxfail',None),'longrepr':str(report.longrepr) if report.failed or report.skipped else None})
   @pytest.hookimpl(hookwrapper=True)
   def pytest_runtest_protocol(self,item,nextitem):
    started=time.monotonic();row={'nodeid':item.nodeid,'cuda':{},'measurement_errors':[]};counts={};self.active=row;self.active_started=started
    def observe(phase,device,action):
     try:return action()
     except BaseException as error:
      row['measurement_errors'].append({'phase':phase,'device':device,'type':type(error).__name__,'message':str(error)})
      return None
    for d in devices:
     i=int(d);row['cuda'][d]={}
     observe('synchronize_before',d,lambda:torch.cuda.synchronize(i))
     observe('reset_peak',d,lambda:torch.cuda.reset_peak_memory_stats(i))
     counts[d]=observe('counter_before',d,lambda:torch.cuda.memory_stats(i)['allocation.all.allocated'])
     row['cuda'][d]['allocated_before']=observe('allocated_before',d,lambda:torch.cuda.memory_allocated(i))
    try:
     yield
    finally:
     for d in devices:
      i=int(d)
      observe('synchronize_after',d,lambda:torch.cuda.synchronize(i))
      row['cuda'][d]['allocation_events']=observe('counter_after',d,lambda:torch.cuda.memory_stats(i)['allocation.all.allocated']-counts[d])
      row['cuda'][d]['allocated_after']=observe('allocated_after',d,lambda:torch.cuda.memory_allocated(i))
      row['cuda'][d]['peak_allocated_bytes']=observe('peak_allocated',d,lambda:torch.cuda.max_memory_allocated(i))
      row['cuda'][d]['peak_reserved_bytes']=observe('peak_reserved',d,lambda:torch.cuda.max_memory_reserved(i))
     row['seconds']=time.monotonic()-started;self.tests.append(row);self.active=None

  e=Evidence();command=SCOPE+['--gpu','-q','-p','no:cacheprovider',f'--junitxml={out}/junit.xml']
  if args.mode=='collect':command+=['--collect-only']
  report['pytest_args']=command
  with (out/'pytest.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=pytest.main(command,plugins=[e])
  report.update(pytest_exitstatus=int(result),ordered_nodeids=e.nodes,phases=e.phases,per_test=e.tests,collection=e.collection,deselected=e.deselected,warnings=e.warnings)
  after=inventory(root,core,driver);final_binding=bound.verify();benchmark_final=bench_bound.verify();write(out/'benchmark-collection-bindings.json',e.benchmark_collection_binding);write(out/'benchmark-final-bindings.json',benchmark_final);write(out/'source-after.json',after);write(out/'initial-collection-bindings.json',e.collection_binding);write(out/'final-bindings.json',final_binding)
  if verify_provider_pair(root,args.baseline)!=pair:raise ValueError('Provider revision/source changed during full run')
  source_ok=before==after and all(id(bound.modules[n][0])==v for n,v in e.initial_ids.items())
  if not source_ok:raise ValueError('Final source or collection module binding changed')
  if e.deselected or any(x['outcome']!='passed' for x in e.collection):raise ValueError('Collection failure/deselection/skip remains')
  report['junit_verification']=verify_junit(out/'junit.xml',e.nodes if args.mode=='run' else [])
  if args.mode=='collect':
   if int(result)!=0 or e.phases or e.tests or torch.cuda.is_initialized():raise ValueError('Collection preflight did not remain host-only and complete')
   frozen={'scope':SCOPE,'source_inventory_sha256':canonical(before),'ordered_nodeids':e.nodes,'provider_heads':pair['heads'],'role':'collection_and_source_candidate_requires_independent_review','harness_sha256':sha(Path(__file__).read_bytes()),'torch_build':torch_build,'required_cuda_nodeids':e.required_cuda_nodes,'protocol':'combined-assembly-direct-storage-v2','H2_G0':'not_assessed'}
   check_catalog(frozen,before);write(out/'candidate-catalog.json',frozen);report['status']='collected_no_tests_run';code=0
  else:
   verdict=grade(e.nodes,e.phases,e.tests,int(result),source_ok,devices,e.required_cuda_nodes);report['verdict']=verdict
   report['engineering_acceptance_passed']=verdict['passed'];report['status']='passed' if verdict['passed'] else 'failed';code=0 if verdict['passed'] else 1
  report['source_inventory_sha256']=canonical(before)
 except BaseException as error:report['status']='failed';report['engineering_acceptance_passed']=False;report['failure']={'type':type(error).__name__,'message':str(error)};code=1
 finally:
  preserve_partial(report,e)
  report['ended_unix']=time.time();report['elapsed_seconds']=report['ended_unix']-started;report['cost_scope']='Per-test process peaks and overlapping invocation duration; shared devices, no speed or n24 fit qualification'
  report['artifacts']={str(x.relative_to(out)):{'sha256':sha(x.read_bytes()),'bytes':x.stat().st_size} for x in out.iterdir() if x.is_file()}
  write(out/'execution.json',report)
 print(json.dumps({'status':report['status'],'engineering_acceptance_passed':report['engineering_acceptance_passed'],'tests':len(report.get('ordered_nodeids',[])),'failure':report.get('failure')}));return code
if __name__=='__main__':raise SystemExit(main())
