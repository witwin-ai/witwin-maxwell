"""Host metadata contracts use original receipts, never a field/reference campaign."""
from pathlib import Path
import copy,hashlib,importlib.metadata,importlib.util,json,os,shutil,subprocess,sys,tempfile
import witwin.core as core
import pytest

SOURCE=Path(__file__).resolve().parents[3]
CONTROL=SOURCE/'scripts/dev/fdtd/consumer_metadata'
CORE=Path(core.__file__).resolve().parents[2]
ARCHIVE=SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-14-core047/consumer-preparation.provenance.json'
# The pre-rename consumer epoch: the rename-receipt mapping tests derive against it.
PRE_RENAME=SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-07-rfic/consumer-preparation.provenance.json'
# The receipt-mapping tests need a post-rename tree that is exactly tool(pre-rename epoch): the frozen
# source of the 11-08-mom rename re-receipt (read-only store). The live tree carries later feature
# changes (R4b, layered round 1) and is covered by the ARCHIVE-based tests.
RENAMED_EPOCH=Path(json.loads((SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-08-mom/consumer-baseline.provenance.json').read_text())['source_root'])
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
# The pre-rename consumer epoch was frozen with witwin-core 0.4.6 at 2ea14d8 (its baseline provenance names the
# read-only store as core_target and its package as actual_core). Its rename-mapping and loader-contract
# derivations run against that store, not the live core, which later epochs may change (round 7: 7ad8bd3).
PRE_RENAME_PROOF=json.loads((PRE_RENAME.parent/Path(json.loads(PRE_RENAME.read_text())['baseline']['provenance']).name).read_text())
PRE_RENAME_CORE=Path(PRE_RENAME_PROOF['core_target'])
PRE_RENAME_CORE_METADATA=PRE_RENAME_CORE/'witwin-0.4.6.dist-info/METADATA'
assert PRE_RENAME_CORE==Path('/mnt/data/xingyu/amazon_data/consumer-metadata-pinned-inputs-20261107/core046')
assert bridge.package(PRE_RENAME_CORE/'witwin/core')==PRE_RENAME_PROOF['actual_core']
assert PRE_RENAME_PROOF['actual_core']['source_sha256']=='e42437ce4de422a1fd63c57a01e82154d6bcd51930e8d972f7385da6bef351d0'
assert hashlib.sha256(PRE_RENAME_CORE_METADATA.read_bytes()).hexdigest()==PRE_RENAME_PROOF['core_metadata_sha256']=='683590ee771a88806dbebf9583ec05ab760bb6eed41963f74b8a780e8fa2f81e'
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
 assert before==after and result['baseline']['historical_unique_pins']==BASELINE['historical_unique_pins']==len(proof['inputs_sha256'])==1293 and result['baseline']['direct_original_witnesses']==24
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
 # The 254 consumer-epoch MoM files: 208 core files pinned per file, the layered files pinned by the
 # feature-group signature of the receipt chain head (an empty chain admits exactly the archive's group).
 groups=result['compatibility']['feature_groups'];archived=json.loads(PROOF.read_text())['current_package_before']['files'];chain=bridge.group_chain(SOURCE,'layered')
 assert result['compatibility']['unchanged_sie_files']==208 and result['compatibility']['changed_core_files']==[]
 assert set(groups)=={'layered'} and groups['layered']['chain_length']==len(chain) and groups['layered']['head']==(f'{chain[-1][0]}@{chain[-1][1]}' if chain else None)
 assert groups['layered']['archive_signature']==bridge.group_signature({k:v for k,v in archived.items() if k.startswith('mom/layered/')})
 assert groups['layered']['signature']==(chain[-1][2]['signature'] if chain else groups['layered']['archive_signature'])
 assert result['actual_imports']['rebinding']==[] and result['actual_imports']['audited_modules']>0
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
 # Each epoch starts from the tree that matches it exactly, so only the mutation is rejected.
 source=renamed_copy(tmp_path,root=SOURCE if epoch=='post-rename' else RENAMED_EPOCH);path=source/'witwin/maxwell/mom/physics.py';path.write_text(path.read_text()+'\nUNQUALIFIED_NUMERICAL_CHANGE = True\n')
 with pytest.raises(ValueError,match='consumer-epoch SIE files changed'):bridge.derive(**copy_args(source,ARCHIVE if epoch=='post-rename' else PRE_RENAME))


def test_pre_rename_consumer_epoch_is_mapped_through_the_receipt():
 # The renamed epoch derives against the frozen pre-rename epoch only through the pinned tool.
 result=bridge.derive(**copy_args(RENAMED_EPOCH,PRE_RENAME))
 assert result['compatibility']['unchanged_sie_files']==227 and result['compatibility']['changed_core_files']==[]


def renamed_copy(tmp_path,receipt=True,root=RENAMED_EPOCH):
 """Writable copy of a renamed package (default: the renamed epoch) plus the live rename tool
 (and receipt) that map it onto the frozen pre-rename consumer epoch."""
 source=tmp_path/'source';shutil.copytree(root/'witwin/maxwell',source/'witwin/maxwell',ignore=shutil.ignore_patterns('__pycache__'),copy_function=shutil.copyfile)
 for path in [source,*source.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
 (source/'scripts/dev').mkdir(parents=True);shutil.copy2(SOURCE/bridge.RENAME_TOOL,source/bridge.RENAME_TOOL)
 if receipt:shutil.copy2(SOURCE/bridge.RENAME_RECEIPT,source/bridge.RENAME_RECEIPT)
 return source


def copy_args(source,archive=PRE_RENAME):
 args=derive_args(archive);args.update(source=source,expected_source=bridge.package(source/'witwin/maxwell')['source_sha256'])
 if archive==PRE_RENAME:
  args.update(core_target=PRE_RENAME_CORE,expected_core=PRE_RENAME_PROOF['actual_core']['source_sha256'],expected_core_metadata=sha(PRE_RENAME_CORE_METADATA))
 return args


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


def test_changed_check_structures_scope_is_rejected(tmp_path):
 # scene.py:_check_structures (the immutable Scene's structure contract, successor of Scene.add_structure) is a
 # protected scope: a changed body is refused even when the rest of the package matches the archive.
 source=renamed_copy(tmp_path,root=SOURCE);path=source/'witwin/maxwell/scene.py';text=path.read_text()
 old='raise TypeError("Maxwell Scene structures must be witwin.core.Structure instances.")'
 assert text.count(old)==1;path.write_text(text.replace(old,old.replace('instances.','instances (changed).')))
 with pytest.raises(ValueError,match='Prescribed shared numerical scope changed: scene.py:_check_structures'):bridge.derive(**copy_args(source,ARCHIVE))


def test_check_structures_keeps_the_add_structure_contract(tmp_path):
 # The predecessor epoch (the round-7 frozen consumer source, mutable Scene) and the live immutable Scene enforce
 # the same structure contract: the same TypeError message for a non-Structure, and the given order kept.
 frozen=Path(json.loads((SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-12-round7/consumer-baseline.provenance.json').read_text())['source_root'])
 assert 'add_structure' in bridge.scopes(frozen/'witwin/maxwell/scene.py')['Scene.add_structure']
 code=("import json,sys;import witwin.maxwell as mw;assert mw.__file__.startswith(sys.argv[1]),mw.__file__\n"
       "from witwin.core import Box,Structure\n"
       "a,b=(Structure(geometry=Box(position=(x,0.,0.),size=(.1,.1,.1)),material=mw.Material(eps_r=2.),name=n) for x,n in ((0.,'a'),(.2,'b')))\n"
       "{build}\ntry:\n {bad}\n message=None\nexcept TypeError as error:message=str(error)\n"
       "print(json.dumps(dict(order=[s.name for s in scene.structures],message=message)))\n")
 builds={'old':(frozen,"scene=mw.Scene(domain=mw.Domain(bounds=((-1,1),)*3),device='cpu');scene.add_structure(a);scene.add_structure(b)",
                "mw.Scene(domain=mw.Domain(bounds=((-1,1),)*3),device='cpu').add_structure(object())"),
         'new':(SOURCE,"scene=mw.Scene(domain=mw.Domain(bounds=((-1,1),)*3),structures=(a,b))",
                "mw.Scene(domain=mw.Domain(bounds=((-1,1),)*3),structures=(a,object()))")}
 out={}
 for key,(root,build,bad) in builds.items():
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
  run=subprocess.run([sys.executable,'-c',code.format(build=build,bad=bad),str(root)],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=300)
  assert run.returncode==0,run.stderr;out[key]=json.loads(run.stdout.strip().splitlines()[-1])
 assert out['old']==out['new']=={'order':['a','b'],'message':'Maxwell Scene structures must be witwin.core.Structure instances.'}


# Direct scope comparison (bridge.compare_scope) for the scene.py pin: the immutable Scene's structure check and its
# dispatch path (construction and updated()) on one side, the predecessor Scene.add_structure on the other.
ROUND7_FROZEN=Path(json.loads((SOURCE/'docs/assessments/fdtd-consumer-metadata-bridge-2026-11-12-round7/consumer-baseline.provenance.json').read_text())['source_root'])
SCENE_PINS=bridge.PROTECTED_SCOPES['scene.py']
DISPATCH_EDITS={'Scene.__post_init__':('self._check(_SCENE_CHECKS)','self._check(set(_SCENE_CHECKS) - {"structures"})'),
                'Scene._check':('            if name in names:','            if name in names and name != "structures":'),
                'Scene.updated':('scene._check(dirty)','scene._check(dirty - {"structures"})')}


def scene_scopes(root,edit=None):
 text=(root/'witwin/maxwell/scene.py').read_text()
 if edit:
  old,new=edit;assert text.count(old)==1,old;text=text.replace(old,new)
 return bridge.scopes(None,text)


def test_scene_pin_is_the_check_and_its_dispatch_path():
 assert SCENE_PINS==('_check_structures','Scene.__post_init__','Scene._check','Scene.updated')
 assert all(bridge.PREDECESSOR_SCOPES[name]=='Scene.add_structure' for name in SCENE_PINS)
 # Function scopes cannot pin the dispatch table itself; it must route structures to the pinned check.
 import witwin.maxwell.scene as scene
 assert scene._SCENE_CHECKS['structures'] is scene._check_structures


def test_scene_scope_old_old_compares_the_predecessor():
 old=scene_scopes(ROUND7_FROZEN);assert not set(SCENE_PINS)&set(old)
 hashes={bridge.compare_scope('scene.py',name,old,copy.deepcopy(old)) for name in SCENE_PINS}
 assert hashes=={('Scene.add_structure',hashlib.sha256(old['Scene.add_structure'].encode()).hexdigest())}


def test_scene_scope_new_new_compares_each_pin():
 live=scene_scopes(SOURCE);assert 'Scene.add_structure' not in live
 for name in SCENE_PINS:
  assert bridge.compare_scope('scene.py',name,live,copy.deepcopy(live))==(name,hashlib.sha256(live[name].encode()).hexdigest())


@pytest.mark.parametrize('name',SCENE_PINS)
def test_scene_scope_mixed_pair_is_refused(name):
 old,live=scene_scopes(ROUND7_FROZEN),scene_scopes(SOURCE)
 for left,right in ((old,live),(live,old)):
  with pytest.raises(ValueError,match=f'Prescribed shared numerical scope changed: scene.py:{name}'):bridge.compare_scope('scene.py',name,left,right)


def test_scene_scope_changed_check_body_is_refused():
 live=scene_scopes(SOURCE)
 old='raise TypeError("Maxwell Scene structures must be witwin.core.Structure instances.")'
 changed=scene_scopes(SOURCE,(old,old.replace('instances.','instances (changed).')))
 with pytest.raises(ValueError,match='Prescribed shared numerical scope changed: scene.py:_check_structures'):bridge.compare_scope('scene.py','_check_structures',live,changed)
 assert all(bridge.compare_scope('scene.py',name,live,changed)[0]==name for name in SCENE_PINS[1:])


@pytest.mark.parametrize('name',tuple(DISPATCH_EDITS))
def test_scene_scope_changed_dispatch_is_refused(name):
 # Each edit skips the structure check on one path (construction, the check loop, updated()).
 live,changed=scene_scopes(SOURCE),scene_scopes(SOURCE,DISPATCH_EDITS[name])
 with pytest.raises(ValueError,match=f'Prescribed shared numerical scope changed: scene.py:{name}'):bridge.compare_scope('scene.py',name,live,changed)
 assert bridge.compare_scope('scene.py','_check_structures',live,changed)[0]=='_check_structures'


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
 source=renamed_copy(tmp_path);live=RENAMED_EPOCH/bridge.loader(RENAMED_EPOCH).relative_to(RENAMED_EPOCH);target=source/live.relative_to(RENAMED_EPOCH)
 target.write_text(live.read_text()+'\nUNREVIEWED_LOADER_CHANGE = True\n')
 receipt_path=source/bridge.RENAME_RECEIPT;receipt=json.loads(receipt_path.read_text())
 row=next(r for r in receipt['files'] if r['new_path']==str(live.relative_to(RENAMED_EPOCH)));row['new_sha256']=sha(target);receipt_path.write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='Consumer loader contract changed'):bridge.derive(**copy_args(source))


# Feature-group receipts (design D-P1 consumer bridge). Each test works on a writable copy of the live
# tree: the package, the rename tool and receipt, the radial loader, the merge-declaration manifest and
# the committed layered evidence. Admitting a change commits the group to the copy's own git repository
# (the acquisition commit), appends a numerical-lock renewal of that commit to the copy's evidence,
# issues the receipt with issue_group_receipt.py and declares it in the copy's manifest.
issuer=load('issue_group_receipt')
GROUP_COPY=('scripts/dev/mom_rename.py','scripts/dev/mom_rename_receipt.json','scripts/dev/measure_mom_fmm_scaling.py',bridge.MANIFEST,*bridge.GROUP_EVIDENCE['layered'])
LIVE_CHAIN=len(bridge.group_chain(SOURCE,'layered'))
T1_FIELD='fixture_r6_native_evidence.json:coarse/none/metrics/L_1GHz_H'


@pytest.fixture(scope='session')
def group_base(tmp_path_factory):
 root=tmp_path_factory.mktemp('group-base');shutil.copytree(SOURCE/'witwin/maxwell',root/'witwin/maxwell',ignore=shutil.ignore_patterns('__pycache__'))
 for name in GROUP_COPY:(root/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,root/name)
 if (SOURCE/bridge.GROUP_RECEIPTS/'layered').is_dir():shutil.copytree(SOURCE/bridge.GROUP_RECEIPTS/'layered',root/bridge.GROUP_RECEIPTS/'layered')
 return root


def git(source,*args):
 return subprocess.run(['git','-C',str(source),'-c','user.name=consumer-bridge-test','-c','user.email=test@invalid','-c','commit.gpgsign=false',*args],capture_output=True,text=True,check=True).stdout.strip()


@pytest.fixture
def group_tree(tmp_path,group_base):
 """A copy of the live tree and its receipt chain whose layered group is committed to a fresh repository;
 the live repository's objects are its alternates, so the live chain's acquisition commits resolve."""
 source=tmp_path/'source';shutil.copytree(group_base,source);git(source,'init','-q')
 objects=Path(git(SOURCE,'rev-parse','--path-format=absolute','--git-common-dir'))/'objects';(source/'.git/objects/info/alternates').write_text(str(objects)+'\n')
 git(source,'add','witwin/maxwell/mom/layered');git(source,'commit','-q','-m','live group');return source


def mutate(source,name,text):
 path=source/'witwin/maxwell/mom/layered'/name;path.write_text(path.read_text()+text);return path


def renew_lock(source,tmp,movers=(T1_FIELD,),t2=0,accepted=()):
 """Commit the group (acquisition commit) and append the evidence_lock.py renewal of that commit."""
 git(source,'add','-A','witwin/maxwell/mom/layered');git(source,'commit','-q','--allow-empty','-m','layered change');commit=git(source,'rev-parse','HEAD')
 table=dict(bound=bridge.T1_STATEMENT,acquisition='<evidence-root>/acq-'+commit[:8],acquisition_commit=commit,
            locked_movers=[dict(field=f,tier='T1',max_rel_diff=3e-12,delta_over_bound=0.03) for f in movers]+[dict(field=a['field'],tier='accepted_T2',max_rel_diff=2e-9,delta_over_bound=20.) for a in accepted],
            accepted_t2=list(accepted),unaccepted_t2=['x'] if t2 else [])
 path=tmp/'evidence'/('acq-'+commit[:8])/'tier_table.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(table))
 for name in bridge.GROUP_EVIDENCE['layered']:
  data=json.loads((source/name).read_text());file=Path(name).name
  fields=[m['field'] for m in table['locked_movers'] if m['field'].startswith(file+':')]
  data['numerical_lock']['renewals'].append(dict(acquisition=table['acquisition'],acquisition_commit=commit,tier_table=table['acquisition']+'/tier_table.json',tier_table_sha256=sha(path),
       summary=dict(T0=1,T1=sum(m['tier']=='T1' for m in table['locked_movers'] if m['field'] in fields),accepted_T2=sum(a['field'] in fields for a in accepted),T2=t2),
       renewed_fields=fields,accepted_t2=[a for a in accepted if a['field'] in fields]))
  (source/name).write_text(json.dumps(data,indent=2)+'\n')
 return commit


def declare(source,branch,name,digest_):
 with (source/bridge.MANIFEST).open('a') as stream:
  stream.write(f'\n[[merge_declarations]]\nbranch = "{branch}"\npytest = "none"\nbenchmark = "none"\nnote = "fixture"\nconsumer_group_receipt = "{name}@{digest_}"\n')


def admit(source,tmp,branch='mom/layered-fixture',approval=None,**lock):
 renew_lock(source,tmp,**lock);issued=issuer.issue(source,'layered',ARCHIVE,branch,approval,tmp/'evidence');declare(source,branch,issued['path'],issued['sha256']);return issued


def forge(source,name,edit):
 """Edit a receipt and re-declare its new SHA (a reviewed-looking but tampered chain head)."""
 path=source/name;old=sha(path);value=json.loads(path.read_text());edit(value);path.write_text(json.dumps(value,indent=1,sort_keys=True)+'\n')
 manifest=source/bridge.MANIFEST;manifest.write_text(manifest.read_text().replace(old,sha(path)))


def write_receipt(source,tmp,branch='mom/layered-fixture',approval=None):
 """A receipt written by hand for the current tree, without the issuing tool's refusals."""
 receipt,_,chain,_=issuer.build_receipt(source,'layered',ARCHIVE,branch,approval,tmp/'evidence')
 path=source/bridge.GROUP_RECEIPTS/'layered'/f'{len(chain)+1:04d}.json';path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(receipt,indent=1,sort_keys=True)+'\n');name=path.relative_to(source).as_posix();declare(source,branch,name,sha(path));return name


def archive_snapshot():
 proof=json.loads(PROOF.read_text())
 return bridge.archive_group(bridge.relocate(proof['source_root'],proof['relocated_roots']),proof['current_package_before'],'layered')


def test_group_numeric_mutation_without_receipt_is_rejected(group_tree):
 mutate(group_tree,'assembly.py','\nUNQUALIFIED_NUMERICAL_CHANGE = 1.0\n')
 with pytest.raises(ValueError,match='Feature-group signature differs'):bridge.derive(**copy_args(group_tree,ARCHIVE))


def test_group_one_numeric_token_is_rejected(group_tree):
 import re
 path=group_tree/'witwin/maxwell/mom/layered/loss.py';text=path.read_text()
 match=next(m for m in re.finditer(r'(?<![\w.])(\d+)\.(\d+)(?![\w.])',text));digit=match.group(2)[-1]
 path.write_text(text[:match.end()-1]+('1' if digit!='1' else '2')+text[match.end():])
 with pytest.raises(ValueError,match='Feature-group signature differs'):bridge.derive(**copy_args(group_tree,ARCHIVE))


@pytest.mark.parametrize('change',['extra','extra-data','missing'])
def test_extra_or_missing_group_file_is_rejected(group_tree,change):
 # The group file set is every file under the prefix, not only code suffixes.
 layered=group_tree/'witwin/maxwell/mom/layered'
 if change=='extra':(layered/'unreviewed.py').write_text('UNREVIEWED = True\n')
 elif change=='extra-data':(layered/'unreviewed_table.npz').write_bytes(b'\x93NUMPY unreviewed table')
 else:(layered/'sweep.py').unlink()
 with pytest.raises(ValueError,match='Feature-group file set differs'):bridge.derive(**copy_args(group_tree,ARCHIVE))


@pytest.mark.parametrize('field',['files','predecessor'])
def test_relabelled_group_receipt_is_rejected(group_tree,tmp_path,field):
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n');issued=admit(group_tree,tmp_path)
 path=mutate(group_tree,'assembly.py','\nUNQUALIFIED_NUMERICAL_CHANGE = 1.0\n');receipt_path=group_tree/issued['path'];value=json.loads(receipt_path.read_text())
 if field=='files':
  # Relabel the head to the mutated bytes without a new declaration.
  value['files']['mom/layered/assembly.py']=sha(path);value['signature']=bridge.group_signature(value['files']);receipt_path.write_text(json.dumps(value,indent=1,sort_keys=True)+'\n')
  with pytest.raises(ValueError,match='Feature-group receipt is not declared'):bridge.derive(**copy_args(group_tree,ARCHIVE))
 else:
  path.write_text(path.read_text().replace('\nUNQUALIFIED_NUMERICAL_CHANGE = 1.0\n',''))
  forge(group_tree,issued['path'],lambda v:v.update(predecessor_signature='0'*64))
  with pytest.raises(ValueError,match='Feature-group receipt chain broken: predecessor mismatch'):bridge.derive(**copy_args(group_tree,ARCHIVE))


def test_group_receipt_unbound_to_numerical_lock_is_rejected(group_tree,tmp_path):
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n');issued=admit(group_tree,tmp_path)
 forge(group_tree,issued['path'],lambda v:v['numerical_lock'].update(tier_table_sha256='0'*64))
 with pytest.raises(ValueError,match='Numerical lock binding differs'):bridge.derive(**copy_args(group_tree,ARCHIVE))


def test_group_receipt_with_unaccepted_t2_is_rejected(group_tree,tmp_path):
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n')
 renew_lock(group_tree,tmp_path,t2=1)
 with pytest.raises(SystemExit,match='Unaccepted T2'):issuer.issue(group_tree,'layered',ARCHIVE,'mom/layered-fixture',None,tmp_path/'evidence')
 write_receipt(group_tree,tmp_path)
 with pytest.raises(ValueError,match='Numerical lock has unaccepted T2'):bridge.derive(**copy_args(group_tree,ARCHIVE))


def test_group_receipt_with_accepted_t2_needs_a_user_approval(group_tree,tmp_path):
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n')
 accepted=[dict(field='fixture_r4b_closed_comparison.json:cases/0/samples/0/L_H',approval='user 2026-11-09 fixture',analysis='fixture')]
 renew_lock(group_tree,tmp_path,accepted=accepted)
 with pytest.raises(SystemExit,match='Accepted T2 entries need --approval'):issuer.issue(group_tree,'layered',ARCHIVE,'mom/layered-fixture',None,tmp_path/'evidence')
 name=write_receipt(group_tree,tmp_path)
 with pytest.raises(ValueError,match='Numerical lock has accepted T2 without a user approval'):bridge.derive(**copy_args(group_tree,ARCHIVE))
 forge(group_tree,name,lambda v:v.update(approval='user approval 2026-11-09 fixture'))
 check=bridge.check_group_chain(group_tree,'layered',archive_snapshot(),json.loads(PROOF.read_text())['current_package_before']['source_sha256'])
 assert check['chain_length']==LIVE_CHAIN+1 and check['signature']==bridge.group_signature(bridge.group_files(group_tree,'layered'))


def test_group_receipt_on_a_different_acquisition_tree_is_rejected(group_tree,tmp_path):
 base=git(group_tree,'rev-parse','HEAD');mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n');issued=admit(group_tree,tmp_path)
 commit=json.loads((group_tree/issued['path']).read_text())['numerical_lock']['acquisition_commit']
 # Point the receipt and the committed evidence at the archive-group commit, whose layered tree differs.
 for name in bridge.GROUP_EVIDENCE['layered']:(group_tree/name).write_text((group_tree/name).read_text().replace(commit,base))
 forge(group_tree,issued['path'],lambda v:v.update(json.loads(json.dumps(v).replace(commit,base))))
 with pytest.raises(ValueError,match='Acquisition tree differs from the receipt'):bridge.derive(**copy_args(group_tree,ARCHIVE))


@pytest.mark.parametrize('case',['snapshot','unknown commit'])
def test_group_receipt_without_its_acquisition_commit_fails_closed(group_tree,tmp_path,case):
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n');issued=admit(group_tree,tmp_path)
 if case=='snapshot':shutil.rmtree(group_tree/'.git')
 else:
  commit=json.loads((group_tree/issued['path']).read_text())['numerical_lock']['acquisition_commit'];missing='f'*40
  for name in bridge.GROUP_EVIDENCE['layered']:(group_tree/name).write_text((group_tree/name).read_text().replace(commit,missing))
  forge(group_tree,issued['path'],lambda v:v.update(json.loads(json.dumps(v).replace(commit,missing))))
 with pytest.raises(ValueError,match='Acquisition commit is not available'):bridge.derive(**copy_args(group_tree,ARCHIVE))


REBINDINGS={
 'function':'\nfrom witwin.maxwell.mom import physics as _core\n_core.compile_medium = lambda *args, **kwargs: None\n',
 'class attribute':'\nfrom witwin.maxwell.mom import runtime as _core\n_core.MoMConfig.__repr__ = lambda self: "rebound"\n',
 'partial alias':'\nimport functools\nfrom witwin.maxwell.mom import physics as _core\nvars(_core)["compile_medium"] = functools.partial(_core.compile_medium, background=True)\n',
}


@pytest.mark.parametrize('case',list(REBINDINGS))
def test_group_rebinding_core_numerics_is_rejected(group_tree,tmp_path,case):
 # sweep.py is imported by witwin.maxwell; its receipt is genuine, so only the import probe can refuse.
 mutate(group_tree,'sweep.py',REBINDINGS[case]);admit(group_tree,tmp_path)
 assert bridge.check_group_chain(group_tree,'layered',archive_snapshot(),json.loads(PROOF.read_text())['current_package_before']['source_sha256'])['chain_length']==LIVE_CHAIN+1
 with pytest.raises(ValueError,match='Core module rebinding by a feature group' if case!='partial alias' else 'Core module rebinding'):bridge.derive(**copy_args(group_tree,ARCHIVE))


def test_group_interface_widening_is_rejected(group_tree):
 path=group_tree/'witwin/maxwell/mom/nearfield.py';path.write_text(path.read_text()+'\nfrom .layered import green as _layered_green\n')
 with pytest.raises(ValueError,match='consumer-epoch SIE files changed'):bridge.derive(**copy_args(group_tree,ARCHIVE))
 # Without the core pin, the frozen interface still refuses the new edge.
 with pytest.raises(ValueError,match='Feature-group interface changed'):bridge.check_group_chain(group_tree,'layered',archive_snapshot(),json.loads(PROOF.read_text())['current_package_before']['source_sha256'])
 assert ['mom/nearfield.py','witwin.maxwell.mom.layered',['green as _layered_green']] in bridge.group_interface(group_tree,'layered')


def test_group_receipt_on_fragile_location_is_rejected(group_tree,tmp_path):
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n');issued=admit(group_tree,tmp_path)
 handle,temporary=tempfile.mkstemp(suffix='.json');os.close(handle);shutil.copyfile(group_tree/issued['path'],temporary)
 try:
  manifest=group_tree/bridge.MANIFEST;manifest.write_text(manifest.read_text().replace(issued['path']+'@',temporary+'@'))
  with pytest.raises(ValueError,match='Historical input names a temporary directory'):bridge.derive(**copy_args(group_tree,ARCHIVE))
 finally:Path(temporary).unlink()


def test_group_op_namespace_change_is_rejected(group_tree,tmp_path):
 mutate(group_tree,'cuda/binding.cpp','\nTORCH_LIBRARY_FRAGMENT(witwin_maxwell_mom,m) {}\n');renew_lock(group_tree,tmp_path)
 with pytest.raises(SystemExit,match='conventional re-receipt'):issuer.issue(group_tree,'layered',ARCHIVE,'mom/layered-fixture',None,tmp_path/'evidence')
 write_receipt(group_tree,tmp_path)
 with pytest.raises(ValueError,match='Feature-group op namespaces changed'):bridge.derive(**copy_args(group_tree,ARCHIVE))


def test_valid_group_receipt_is_admitted(group_tree,tmp_path):
 # Two successive layered-only changes, each with a renewed lock, a receipt and a declaration.
 mutate(group_tree,'assembly.py','\nADMITTED_CHANGE = 1.0\n');first=admit(group_tree,tmp_path,branch='mom/layered-first')
 with pytest.raises(SystemExit,match='nothing to admit'):issuer.issue(group_tree,'layered',ARCHIVE,'mom/layered-again',None,tmp_path/'evidence')
 mutate(group_tree,'loss.py','\nADMITTED_SECOND_CHANGE = 2.0\n');second=admit(group_tree,tmp_path/'second',branch='mom/layered-second')
 assert first['path'].endswith(f'/{LIVE_CHAIN+1:04d}.json') and second['path'].endswith(f'/{LIVE_CHAIN+2:04d}.json')
 assert T1_FIELD in second['note'] and 'max_rel_diff 3e-12' in second['note'] and bridge.T1_STATEMENT in second['note'] and 'approval T1-auto' in second['note']
 result=bridge.derive(**copy_args(group_tree,ARCHIVE));group=result['compatibility']['feature_groups']['layered']
 receipt=json.loads((group_tree/second['path']).read_text())
 assert result['compatibility']['unchanged_sie_files']==208 and group['chain_length']==LIVE_CHAIN+2 and group['head']==second['path']+'@'+second['sha256']
 assert group['signature']==receipt['signature']!=group['archive_signature']==archive_snapshot()['signature']
 assert receipt['predecessor_receipt']==first['path'] and receipt['predecessor_receipt_sha256']==first['sha256']
 assert receipt['numerical_lock']['t1_bound']==dict(relative=1e-10,statement=bridge.T1_STATEMENT) and receipt['approval']=='T1-auto'
 assert result['actual_imports']['rebinding']==[] and not result['production_qualification']
 # The first receipt alone no longer matches the live group.
 (group_tree/second['path']).unlink()
 with pytest.raises(ValueError,match='Feature-group signature differs'):bridge.check_group_chain(group_tree,'layered',archive_snapshot(),json.loads(PROOF.read_text())['current_package_before']['source_sha256'])
