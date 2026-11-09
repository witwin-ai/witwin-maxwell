"""Host metadata contracts use original receipts, never a field/reference campaign."""
from pathlib import Path
import copy,hashlib,importlib.metadata,importlib.util,json,os,shutil,subprocess,sys,tempfile
import witwin.core as core
import pytest

SOURCE=Path(__file__).resolve().parents[3]
CONTROL=SOURCE/'scripts/dev/fdtd/consumer_metadata'
CORE=Path(core.__file__).resolve().parents[2]
ARCHIVE=SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-08-mom/consumer-preparation.provenance.json'
# The pre-rename consumer epoch: the rename-receipt mapping tests derive against it.
PRE_RENAME=SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-07-rfic/consumer-preparation.provenance.json'
if not ARCHIVE.exists():
 pytest.skip('repository evidence not in this checkout: '+str(ARCHIVE.relative_to(SOURCE)),allow_module_level=True)
BASELINE=json.loads(ARCHIVE.read_text())['baseline']
# The archive records absolute paths from the worktree it was made in; the files live beside the
# archive record and their contents stay bound by the pinned SHAs below.
OLD=ARCHIVE.parent/Path(BASELINE['bridge']).name
PROOF=ARCHIVE.parent/Path(BASELINE['provenance']).name
# Closed witnesses embed their original /tmp and worktree paths; the receipts carry the reviewed same-SHA relocation.
ROOTS=json.loads(PROOF.read_text())['relocated_roots']


def load(name):
 spec=importlib.util.spec_from_file_location(name,CONTROL/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
bridge=load('derive_consumer_bridge');closure=load('helper_closure')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
moved=lambda p:bridge.relocate(p,ROOTS)


@pytest.fixture(autouse=True)
def isolated_import_state():
 """Pinned loaders insert their (often frozen) source roots at sys.path[0]. Restore the import
 state after each test so later tests resolve top-level packages such as `scripts` normally."""
 path,modules=list(sys.path),set(sys.modules)
 yield
 sys.path[:]=path
 for name in [n for n in sys.modules if n not in modules and n.split('.')[0]=='scripts']:del sys.modules[name]


@pytest.fixture(scope='session')
def receipt():return bridge.derive(**derive_args())


def derive_args(archive=None):
 distribution=importlib.metadata.distribution('witwin')
 metadata=next(Path(distribution.locate_file(f)) for f in distribution.files if str(f).endswith('.dist-info/METADATA'))
 baseline,old,proof=BASELINE,OLD,PROOF
 if archive is not None:
  baseline=json.loads(archive.read_text())['baseline'];old=archive.parent/Path(baseline['bridge']).name;proof=archive.parent/Path(baseline['provenance']).name
 return dict(source=SOURCE,core_target=CORE,expected_source=bridge.package(SOURCE/'witwin/maxwell')['source_sha256'],expected_core=bridge.package(CORE/'witwin/core')['source_sha256'],expected_core_metadata=sha(metadata),baseline_bridge=old,baseline_sha=baseline['sha256'],baseline_provenance=proof,provenance_sha=baseline['provenance_sha256'])


def test_genuine_closed_receipts_are_consumed_without_relabelling_provider(receipt):
 before={name:sha(name) for name in receipt['inputs_sha256']};result=bridge.derive(**derive_args());after={name:sha(name) for name in receipt['inputs_sha256']}
 proof=json.loads(PROOF.read_text());lineage=json.loads((ARCHIVE.parent/'historical-lineage.json').read_text());old=json.loads(OLD.read_text())
 assert before==after and result['baseline']['historical_unique_pins']==BASELINE['historical_unique_pins']==len(proof['inputs_sha256'])==868 and result['baseline']['direct_original_witnesses']==24
 assert lineage['original_unique_pins']==len(lineage['inputs_sha256'])==455 and lineage['direct_original_witnesses']==len(lineage['original_records'])==24
 assert lineage['witnesses_included_in_historical_pins'] is True and lineage['predecessor']==proof['predecessor']
 assert sha(moved(lineage['predecessor']['bridge']))==lineage['predecessor']['sha256'] and sha(moved(lineage['predecessor']['provenance']))==lineage['predecessor']['provenance_sha256']
 assert lineage['inputs_sha256']==json.loads(moved(lineage['predecessor']['provenance']).read_text())['inputs_sha256']
 assert all(proof['inputs_sha256'][str(moved(name))]==expected for name,expected in lineage['inputs_sha256'].items())
 bridge.check_pins({str(moved(name)):expected for name,expected in lineage['inputs_sha256'].items()})
 assert not bridge.fragile_inputs([*proof['inputs_sha256'],proof['source_root'],*ROOTS.values()]) and result['relocated_roots']==ROOTS
 assert not result['current_native_actions_executed'] and not result['provider_capability_requalified'] and not result['production_qualification']
 assert result['candidate']['production_source_hash']==result['consumer_loader_source_hash']==bridge.helper(SOURCE).production_source_hash()
 assert result['preserved_baseline_consumer_hash']==old['production_source_hash']
 assert result['candidate']['records']==old['records']==lineage['original_records']
 assert len(result['current_selections'])==8
 assert all(s['original_order_selection']['production_source_hash']!=s['production_source_hash'] and s['production_qualification'] is False for s in result['current_selections'].values())
 assert all(s['original_order_selection']==proof['current_selections'][name]['original_order_selection'] for name,s in result['current_selections'].items())
 previous=proof['current_package_before']['files'];current=result['source_before']['files']
 changed=sorted(name for name in previous.keys()|current.keys() if previous.get(name)!=current.get(name))
 assert result['compatibility']['unchanged_sie_files']==227 and result['compatibility']['changed_core_files']==[]
 assert result['compatibility']['changed_package_files']==changed
 assert result['compatibility']['changed_shared_or_public_files']==[name for name in changed if not name.startswith(('fdtd/','execution/'))]
 assert result['compatibility']['original_provider_comparison']['changed_sie_files_vs_current']
 assert result['original_provider_source_hash']!=result['consumer_loader_source_hash']


@pytest.mark.parametrize('slot',['expected_source','expected_core','expected_core_metadata'])
def test_wrong_actual_source_or_core_identity_is_rejected(slot):
 args=derive_args();args[slot]='0'*64
 with pytest.raises(ValueError,match='identity'):bridge.derive(**args)


def test_missing_original_raw_file_is_not_silently_accepted(tmp_path):
 path=tmp_path/'raw.json';path.write_text('{"original":true}');expected=sha(path);path.unlink()
 with pytest.raises(ValueError,match='Missing or changed original input'):bridge.check_pins({str(path):expected})


def test_changed_preserved_header_is_rejected_without_touching_history(tmp_path):
 args=derive_args();value=json.loads(OLD.read_text());value['production_source_hash']='0'*64;changed=tmp_path/'not-original.json';changed.write_text(json.dumps(value));args.update(baseline_bridge=changed,baseline_sha=sha(changed))
 with pytest.raises(ValueError,match='Baseline is not unchanged'):bridge.derive(**args)
 assert sha(OLD)==derive_args()['baseline_sha']


def fragile_cases(tmp_path):
 distribution=importlib.metadata.distribution('witwin');files=[Path(distribution.locate_file(f)).resolve() for f in distribution.files]
 core_source=next(p for p in files if p.parts[-3:]==('witwin','core','__init__.py'));metadata=next(p for p in files if p.name=='METADATA' and p.parent.name.endswith('.dist-info'))
 checkout=tmp_path/'.worktrees/old-checkout/witness.json';checkout.parent.mkdir(parents=True);checkout.write_text('{"witness":1}')
 handle,temporary=tempfile.mkstemp(suffix='.json');os.close(handle);temporary=Path(temporary);temporary.write_text('{"witness":2}')
 return {'live core source':(core_source,'the live environment site-packages'),'live dist-info':(metadata,'the live environment site-packages'),'worktree':(checkout,'a .worktrees checkout'),'temporary':(temporary,'a temporary directory')}


def doctored_provenance(tmp_path,extra=None,source_root=None,drop_root=None):
 args=derive_args();value=json.loads(PROOF.read_text());assert not bridge.fragile_inputs(value['inputs_sha256'])
 value['inputs_sha256'].update(extra or {})
 if source_root:value['source_root']=str(source_root)
 if drop_root:del value['relocated_roots'][drop_root]
 changed=tmp_path/'doctored.provenance.json';changed.write_text(json.dumps(value));args.update(baseline_provenance=changed,provenance_sha=sha(changed));return args


@pytest.mark.parametrize('case',['live core source','live dist-info','worktree','temporary'])
def test_historical_pin_on_fragile_location_is_rejected_by_the_guard(tmp_path,case):
 path,reason=fragile_cases(tmp_path)[case];pin={str(path):sha(path)}
 bridge.check_pins(pin)  # the file exists with its pinned SHA, so only the location guard can reject it
 try:
  with pytest.raises(ValueError,match='Historical input names '+reason):bridge.derive(**doctored_provenance(tmp_path,pin))
 finally:
  if case=='temporary':path.unlink()


def test_fragile_pins_are_accepted_once_the_guard_is_removed(tmp_path,monkeypatch):
 cases=fragile_cases(tmp_path);pins={str(path):sha(path) for path,_ in cases.values()}
 try:
  args=doctored_provenance(tmp_path,pins);assert {name for name,_ in bridge.fragile_inputs(pins)}==set(pins)
  monkeypatch.setattr(bridge,'fragile_inputs',lambda names:[]);result=bridge.derive(**args)
  assert all(result['inputs_sha256'][name]==value for name,value in pins.items())
 finally:cases['temporary'][0].unlink()


def test_unarchived_temporary_source_root_is_rejected_by_the_guard(tmp_path):
 archived=Path(json.loads(PROOF.read_text())['source_root']);old=next(old for old,new in ROOTS.items() if archived.is_relative_to(new));original=Path(old)/archived.relative_to(ROOTS[old])
 assert original.is_relative_to('/tmp') and moved(original)==archived
 # With its relocation entry present the historical name reads the archived tree; without it the guard refuses the /tmp root.
 with pytest.raises(ValueError,match='Historical input names a temporary directory'):bridge.derive(**doctored_provenance(tmp_path,source_root=original,drop_root=old))


def test_current_consumer_loader_rejects_a_relabelled_original_provider_header(tmp_path,receipt):
 current=bridge.helper(SOURCE,ROOTS);value=copy.deepcopy(receipt['candidate']);original=json.loads(moved(value['records']['3']['path']).read_text());original['order_selection']['production_source_hash']=current.production_source_hash();raw=tmp_path/'corrupt-provider-copy.json';raw.write_text(json.dumps(original));value['records']['3']=dict(path=str(raw),sha256=sha(raw));candidate=tmp_path/'consumer.json';candidate.write_text(json.dumps(value))
 with pytest.raises(ValueError,match='Original order differs'):current.load_radial_order_bridge(candidate)


@pytest.mark.parametrize('epoch',['post-rename','pre-rename'])
def test_numerical_provider_mutation_is_rejected_even_with_new_overall_hash(tmp_path,receipt,epoch):
 source=renamed_copy(tmp_path);path=source/'witwin/maxwell/mom/physics.py';path.write_text(path.read_text()+'\nUNQUALIFIED_NUMERICAL_CHANGE = True\n')
 with pytest.raises(ValueError,match='consumer-epoch SIE files changed'):bridge.derive(**copy_args(source,ARCHIVE if epoch=='post-rename' else PRE_RENAME))


def test_pre_rename_consumer_epoch_is_mapped_through_the_receipt():
 # The renamed live tree derives against the frozen pre-rename epoch only through the pinned tool.
 result=bridge.derive(**derive_args(PRE_RENAME))
 assert result['compatibility']['unchanged_sie_files']==227 and result['compatibility']['changed_core_files']==[]


def renamed_copy(tmp_path,receipt=True):
 """Live package plus the rename tool (and receipt) that map it onto the frozen consumer epoch."""
 source=tmp_path/'source';shutil.copytree(SOURCE/'witwin/maxwell',source/'witwin/maxwell',ignore=shutil.ignore_patterns('__pycache__'))
 (source/'scripts/dev').mkdir(parents=True);shutil.copy2(SOURCE/bridge.RENAME_TOOL,source/bridge.RENAME_TOOL)
 if receipt:shutil.copy2(SOURCE/bridge.RENAME_RECEIPT,source/bridge.RENAME_RECEIPT)
 return source


def copy_args(source,archive=PRE_RENAME):
 args=derive_args(archive);args.update(source=source,expected_source=bridge.package(source/'witwin/maxwell')['source_sha256']);return args


def test_one_numeric_token_off_the_rename_is_rejected(tmp_path):
 # tool(old) and the live file differ in a single numeric literal: not a rename, a numerical change.
 import re
 source=renamed_copy(tmp_path);path=source/'witwin/maxwell/mom/physics.py';text=path.read_text()
 match=next(m for m in re.finditer(r'(?<![\w.])(\d+)\.(\d+)(?![\w.])',text))
 digit=match.group(2)[-1];path.write_text(text[:match.end()-1]+('1' if digit!='1' else '2')+text[match.end():])
 with pytest.raises(ValueError,match='consumer-epoch SIE files changed'):bridge.derive(**copy_args(source))


@pytest.mark.parametrize('change',['extra','missing'])
def test_extra_or_missing_renamed_solver_file_is_rejected(tmp_path,change):
 source=renamed_copy(tmp_path)
 if change=='extra':(source/'witwin/maxwell/mom/unreviewed.py').write_text('UNREVIEWED = True\n')
 else:(source/'witwin/maxwell/mom/physics.py').unlink()
 with pytest.raises(ValueError,match='consumer-epoch SIE files changed'):bridge.derive(**copy_args(source))


def test_absent_rename_receipt_is_rejected(tmp_path):
 with pytest.raises(ValueError,match='Rename receipt missing'):bridge.derive(**copy_args(renamed_copy(tmp_path,receipt=False)))


def test_altered_rename_receipt_is_rejected(tmp_path):
 # Mutate a solver file and relabel its receipt row to the mutated bytes: the tool does not reproduce the row.
 source=renamed_copy(tmp_path);path=source/'witwin/maxwell/mom/physics.py';path.write_text(path.read_text()+'\nUNQUALIFIED_NUMERICAL_CHANGE = True\n')
 receipt_path=source/bridge.RENAME_RECEIPT;receipt=json.loads(receipt_path.read_text())
 row=next(r for r in receipt['files'] if r['new_path']=='witwin/maxwell/mom/physics.py');row['new_sha256']=sha(path);receipt_path.write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='not reproduced by the rename tool'):bridge.derive(**copy_args(source))


def forge_receipt_row(source,rel):
 # Relabel both sides of the row to the mutated live bytes (adding the row if the receipt lacks one):
 # the frozen epoch file no longer matches old_sha256, so the live file must be compared directly.
 path=source/rel;receipt_path=source/bridge.RENAME_RECEIPT;receipt=json.loads(receipt_path.read_text())
 row=next((r for r in receipt['files'] if r['new_path']==rel),None)
 if row is None:row=dict(mode='text',old_path=rel,new_path=rel);receipt['files'].append(row)
 row['old_sha256']=row['new_sha256']=sha(path);receipt_path.write_text(json.dumps(receipt))


def test_forged_old_and_new_receipt_digests_on_a_solver_file_are_rejected(tmp_path):
 source=renamed_copy(tmp_path);path=source/'witwin/maxwell/mom/physics.py';path.write_text(path.read_text()+'\nUNQUALIFIED_NUMERICAL_CHANGE = True\n')
 forge_receipt_row(source,'witwin/maxwell/mom/physics.py')
 with pytest.raises(ValueError,match='consumer-epoch SIE files changed'):bridge.derive(**copy_args(source,PRE_RENAME))


def test_forged_old_and_new_receipt_digests_on_a_protected_file_are_rejected(tmp_path):
 rel='witwin/maxwell/'+bridge.PROTECTED_FILES[0]
 source=renamed_copy(tmp_path);path=source/rel;path.write_text(path.read_text()+'\nUNQUALIFIED_NUMERICAL_CHANGE = True\n')
 forge_receipt_row(source,rel)
 with pytest.raises(ValueError,match='Prescribed shared numerical file changed'):bridge.derive(**copy_args(source,PRE_RENAME))


@pytest.fixture(scope='session')
def package_copy(tmp_path_factory):
 root=tmp_path_factory.mktemp('real-package');shutil.copytree(SOURCE/'witwin/maxwell',root/'witwin/maxwell',ignore=shutil.ignore_patterns('__pycache__'));return root


@pytest.fixture
def helper_case(tmp_path,package_copy):
 source=tmp_path/'source';(source/'tests').mkdir(parents=True);shutil.copytree(package_copy/'witwin',source/'witwin',copy_function=os.link);raw=tmp_path/'original.json';raw.write_text('{"value":17}');other=tmp_path/'different.json';other.write_text('{"value":99}')
 (source/'tests/test_leaf.py').write_text("import json,os\nfrom pathlib import Path\ndef original():\n path=os.environ.get('GSP_REFERENCE')\n if not path:raise AssertionError('Unqualified original fallback attempted')\n return json.loads(Path(path).read_text())['value']\n")
 (source/'tests/test_middle.py').write_text('from test_leaf import original\ndef fetch():return original()\n')
 caller=source/'tests/test_caller.py';caller.write_text('from test_middle import fetch\ndef test_declared_original():assert fetch()==17\n');assert 'GSP_' not in caller.read_text()
 (source/'tests/test_unrelated.py').write_text("import os\ndef test_unrelated():assert not any(key.startswith('GSP_') for key in os.environ)\n")
 policy=dict(schema='module-scoped-artifact-environment/1',files={str(raw):sha(raw),str(other):sha(other)},modules={'tests/test_leaf.py':dict(environment={'GSP_REFERENCE':str(raw)})})
 return tmp_path,source,raw,other,policy


def run_case(root,source,policy):
 path=root/'policy.json';path.write_text(json.dumps(policy));output=root/'result';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CORE),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',GSP_REFERENCE='poison-global')
 command=[sys.executable,str(SOURCE/'scripts/dev/fdtd/regression_shards/run_shards.py'),'run','--source-root',str(source),'--output',str(output),'--module-environment',str(path)]
 run=subprocess.run(command,cwd=source,env=env,capture_output=True,text=True,timeout=300);(root/'controller.log').write_text(run.stdout+run.stderr);return run,output


def test_no_keyword_caller_gets_only_its_transitive_original_bundle(helper_case):
 root,source,raw,other,policy=helper_case;result=closure.derive(source,policy)
 assert result['modules']['tests/test_caller.py']['environment']=={'GSP_REFERENCE':str(raw)}
 assert 'tests/test_unrelated.py' not in result['modules']
 assert set(result['helper_dependency_source_pins'])=={'tests/test_caller.py','tests/test_middle.py','tests/test_leaf.py'}
 run,output=run_case(root,source,result);assert run.returncode==0,run.stderr;summary=json.loads((output/'summary.json').read_text());assert summary['counts']=={'passed':2}
 actual=[json.loads(p.read_text())['gsp_environment'] for p in output.glob('attempt-*/identity-before.json')];assert {'GSP_REFERENCE':str(raw)} in actual and {} in actual


def test_no_bundle_caller_cannot_claim_fallback_as_original_reuse(helper_case):
 root,source,raw,other,policy=helper_case;run,output=run_case(root,source,policy);assert run.returncode==1;ledger=json.loads((output/'node-outcomes.json').read_text());assert ledger['tests/test_caller.py::test_declared_original']['outcome']=='failed';assert 'Unqualified original fallback attempted' in str(ledger)


def test_conflicting_original_requirements_inside_one_consumer_reject(helper_case):
 root,source,raw,other,policy=helper_case;policy['modules']['tests/test_caller.py']=dict(environment={'GSP_REFERENCE':str(other)})
 with pytest.raises(ValueError,match='Conflicting helper artifact key'):closure.derive(source,policy)


def test_missing_genuine_helper_raw_input_rejects(helper_case):
 root,source,raw,other,policy=helper_case;raw.unlink()
 with pytest.raises(ValueError,match='Missing or changed original helper artifact'):closure.derive(source,policy)


def test_missing_transitive_declared_matrix_rejects(helper_case):
 root,source,raw,other,policy=helper_case;raw.write_text(json.dumps(dict(path=str(root/'missing-matrix.pt'),sha256='0'*64)));policy['files'][str(raw)]=sha(raw)
 with pytest.raises(ValueError,match='Missing or changed declared original artifact'):closure.derive(source,policy)


def test_changed_transitive_helper_source_rejects_before_execution(helper_case):
 root,source,raw,other,policy=helper_case;result=closure.derive(source,policy);path=source/'tests/test_middle.py';path.write_text(path.read_text()+'\nCHANGED = True\n')
 with pytest.raises(ValueError,match='Changed helper-consumer source'):closure.verify_source_pins(result,source)
 run,output=run_case(root,source,result);assert run.returncode!=0 and not output.exists();assert 'Missing or changed module artifact' in run.stderr


def test_two_imported_helpers_cannot_merge_conflicting_originals(helper_case):
 root,source,raw,other,policy=helper_case
 (source/'tests/test_leaf_second.py').write_text("import json,os\nfrom pathlib import Path\ndef second():return json.loads(Path(os.environ['GSP_REFERENCE']).read_text())['value']\n")
 (source/'tests/test_middle.py').write_text('from test_leaf import original\nfrom test_leaf_second import second\ndef fetch():return original()+second()\n')
 policy['modules']['tests/test_leaf_second.py']=dict(environment={'GSP_REFERENCE':str(other)})
 with pytest.raises(ValueError,match='Conflicting helper artifact key'):closure.derive(source,policy)


def test_declared_original_matrix_sha_is_pinned_without_regeneration(helper_case):
 root,source,raw,other,policy=helper_case;matrix=root/'retained-matrix.bin';matrix.write_bytes(b'original independent matrix bytes');raw.write_text(json.dumps(dict(path=str(matrix),sha256=sha(matrix))));policy['files'][str(raw)]=sha(raw);result=closure.derive(source,policy)
 assert result['files'][str(matrix)]==sha(matrix)
 matrix.write_bytes(b'changed original')
 with pytest.raises(ValueError,match='Missing or changed original helper artifact'):closure.verify_artifact_pins(result)


def test_wrong_source_missing_reviewed_helper_module_is_rejected(helper_case):
 root,source,raw,other,policy=helper_case;(source/'tests/test_leaf.py').unlink()
 with pytest.raises(ValueError,match='Source lacks reviewed policy module'):closure.derive(source,policy)


def test_rename_tool_rewritten_to_reproduce_a_tampered_file_is_rejected(tmp_path):
 # Tamper a solver file, teach the live tool to emit exactly those bytes and relabel the receipt:
 # only the pinned tool identity stops this.
 source=renamed_copy(tmp_path);path=source/'witwin/maxwell/mom/physics.py';path.write_text(path.read_text()+'\nUNQUALIFIED_NUMERICAL_CHANGE = True\n')
 tool=source/bridge.RENAME_TOOL
 tool.write_text(tool.read_text()+"""
_reviewed_transform_bytes = transform_bytes
def transform_bytes(path, data):
    new = _reviewed_transform_bytes(path, data)
    return new + b'\\nUNQUALIFIED_NUMERICAL_CHANGE = True\\n' if path == 'witwin/maxwell/sie/physics.py' else new
""")
 receipt_path=source/bridge.RENAME_RECEIPT;receipt=json.loads(receipt_path.read_text())
 row=next(r for r in receipt['files'] if r['new_path']=='witwin/maxwell/mom/physics.py');row['new_sha256']=sha(path);receipt_path.write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='identity'):bridge.derive(**copy_args(source))
 # With the pin bypassed the tampered tool would map the tampered file onto the frozen epoch.
 import unittest.mock
 pre=PRE_RENAME.parent/Path(json.loads(PRE_RENAME.read_text())['baseline']['provenance']).name
 old_source=bridge.relocate(json.loads(pre.read_text())['source_root'],json.loads(pre.read_text())['relocated_roots'])
 with unittest.mock.patch.object(bridge,'RENAME_TOOL_SHA256',sha(tool)):
  view=bridge.pre_rename_view(source,old_source,bridge.package(source/'witwin/maxwell')['files'])
 assert 'mom/physics.py' not in view and 'sie/physics.py' in view


def test_modified_loader_with_matching_receipt_row_is_rejected(tmp_path):
 # The loader is compared with tool(frozen loader), never with the live receipt row alone.
 source=renamed_copy(tmp_path);live=SOURCE/bridge.loader(SOURCE).relative_to(SOURCE);target=source/live.relative_to(SOURCE)
 target.write_text(live.read_text()+'\nUNREVIEWED_LOADER_CHANGE = True\n')
 receipt_path=source/bridge.RENAME_RECEIPT;receipt=json.loads(receipt_path.read_text())
 row=next(r for r in receipt['files'] if r['new_path']==str(live.relative_to(SOURCE)));row['new_sha256']=sha(target);receipt_path.write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='Consumer loader contract changed'):bridge.derive(**copy_args(source))
