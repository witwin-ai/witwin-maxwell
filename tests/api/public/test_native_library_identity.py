"""Real CPU ELF library identity through torch.ops, with no solver/GPU claim."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,shutil,subprocess,sys
import pytest
ROOT=Path(os.environ['PCB_NATIVE_IDENTITY_SOURCE']) if 'PCB_NATIVE_IDENTITY_SOURCE'in os.environ else Path(__file__).resolve().parents[3]
CONTROL=Path(os.environ.get('PCB_NATIVE_IDENTITY_CONTROL',str(ROOT/'scripts/dev/fdtd/regression_shards')))
CHILD=r'''
from pathlib import Path
import ctypes,hashlib,json,sys
import torch,witwin.core as core
from shard_common import runtime_identity
source,control,library,alias,output,mode=map(Path,sys.argv[1:7])
assert not torch.cuda.is_initialized()
first=library.resolve();torch.ops.load_library(str(library))
if str(mode)=='repeat':
 torch.ops.load_library(str(alias));torch.ops.load_library(str(first))
probe=ctypes.CDLL(str(first));probe.pcb_identity_probe.restype=ctypes.c_int
assert probe.pcb_identity_probe()==731
second=None;second_sha=None
if str(mode)=='two':
 second=(first.parent/'second'/first.name).resolve();torch.ops.load_library(str(second))
 second_probe=ctypes.CDLL(str(second));second_probe.pcb_identity_probe.restype=ctypes.c_int
 assert second_probe.pcb_identity_probe()==947
 second_sha=hashlib.sha256(second.read_bytes()).hexdigest()
module_paths={str(Path(p).resolve())for value in sys.modules.values()if isinstance((p:=getattr(value,'__file__',None)),str)}
assert str(first)not in module_paths
if second is not None:assert str(second)not in module_paths
registered={str(Path(p).resolve())for p in torch.ops.loaded_libraries}
assert str(first)in registered
actual_sha=hashlib.sha256(first.read_bytes()).hexdigest()
if str(mode)=='deleted':first.unlink()
request=dict(source_root=str(source),control_root=str(control),external_paths=[])
identity=runtime_identity(request)
assert not torch.cuda.is_initialized()
output.write_text(json.dumps(dict(actual_library=str(first),library_sha256=actual_sha,
 absent_from_sys_modules=True,registered_by_real_torch_ops=True,
 recorded=identity['loaded_native_binaries'].get(str(first)),
 canonical_records=[key for key in identity['loaded_native_binaries']if Path(key).resolve()==first],
 core_distribution=identity['versions']['core_distribution'],core_module_version=core.__version__,
 second_library=str(second)if second is not None else None,second_sha256=second_sha,second_recorded=identity['loaded_native_binaries'].get(str(second))if second is not None else None,CUDA_initialized=False),indent=2)+'\n')
'''

@pytest.fixture
def compiled(tmp_path):
 if sys.platform!='linux':pytest.skip('Actual ELF CPU shared-library contract is Linux-specific.')
 cc=shutil.which('cc')or shutil.which('gcc')
 if not cc:pytest.skip('A system C compiler is required for this real library-loader contract.')
 source=tmp_path/'identity.c';source.write_text('int pcb_identity_probe(void) { return 731; }\n')
 library=tmp_path/'libactual_identity.so';command=[cc,'-shared','-fPIC','-O0',str(source),'-o',str(library)]
 result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
 alias=tmp_path/'alias.so';alias.symlink_to(library.name)
 return library,alias

def probe(tmp_path,compiled,mode='once'):
 library,alias=compiled;output=tmp_path/'actual.json'
 if mode=='two':
  second=tmp_path/'second';second.mkdir();(second/'identity.c').write_text('int pcb_identity_probe(void) { return 947; }\n')
  result=subprocess.run([shutil.which('cc')or shutil.which('gcc'),'-shared','-fPIC','-O0',str(second/'identity.c'),'-o',str(second/library.name)],capture_output=True,text=True)
  assert result.returncode==0,result.stderr
 environment=dict(os.environ,PYTHONPATH=str(CONTROL)+os.pathsep+str(ROOT),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 result=subprocess.run([sys.executable,'-c',CHILD,str(ROOT),str(CONTROL),str(library),str(alias),str(output),mode],cwd=ROOT,env=environment,capture_output=True,text=True)
 (tmp_path/'child.stdout').write_text(result.stdout);(tmp_path/'child.stderr').write_text(result.stderr)
 return result,output

def test_torch_ops_library_recorded_when_absent_from_sys_modules(tmp_path,compiled):
 result,output=probe(tmp_path,compiled);assert result.returncode==0,result.stderr;value=json.loads(output.read_text())
 assert value['absent_from_sys_modules']and value['registered_by_real_torch_ops']
 assert value['recorded']==value['library_sha256']
 assert value['actual_library']==str(compiled[0].resolve())and value['core_distribution']==value['core_module_version']==importlib.metadata.distribution('witwin').version


def test_real_alias_and_repeated_load_have_one_canonical_hash_record(tmp_path,compiled):
 result,output=probe(tmp_path,compiled,'repeat');assert result.returncode==0,result.stderr;value=json.loads(output.read_text())
 assert value['recorded']==hashlib.sha256(compiled[0].read_bytes()).hexdigest()
 assert value['canonical_records']==[str(compiled[0].resolve())]


def test_loaded_deleted_library_fails_closed_instead_of_silently_omitting_identity(tmp_path,compiled):
 result,output=probe(tmp_path,compiled,'deleted')
 assert result.returncode!=0 and 'FileNotFoundError'in result.stderr and not output.exists()


def test_same_basename_distinct_real_libraries_keep_both_paths_and_actual_hashes(tmp_path,compiled):
 result,output=probe(tmp_path,compiled,'two');assert result.returncode==0,result.stderr;value=json.loads(output.read_text())
 first=compiled[0].resolve();second=(tmp_path/'second'/first.name).resolve()
 assert first.name==second.name and first!=second
 assert value['actual_library']==str(first)and value['second_library']==str(second)
 assert value['recorded']==hashlib.sha256(first.read_bytes()).hexdigest()
 assert value['second_recorded']==value['second_sha256']==hashlib.sha256(second.read_bytes()).hexdigest()
 assert value['recorded']!=value['second_recorded']
