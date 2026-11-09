"""Consumer-only metadata from immutable closed witnesses, never provider requalification."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,os,subprocess,sys,sysconfig,tempfile,time

CODE_SUFFIXES={'.py','.cu','.cpp','.h','.cuh'}
PROTECTED_FILES=('constants.py','materials.py','network_math.py','rational.py','waveforms.py')
PROTECTED_SCOPES={'media.py':('Material.evaluate_static','Material.evaluate_at_frequency'),
                  'sources.py':('evaluate_source_time','evaluate_source_time_normalization','polarization_vector'),
                  'scene.py':('Scene.add_structure',),'simulation.py':('Simulation.sie',)}
# The frozen consumer epoch predates the SIE -> MoM rename. Its solver files, protected scopes
# and loader are compared with the live tree through scripts/dev/mom_rename.py: a live file is
# unchanged when its bytes equal the receipt's tool(old bytes), a scope when it equals the
# scope of tool(old file).
RENAME_TOOL='scripts/dev/mom_rename.py';RENAME_RECEIPT='scripts/dev/mom_rename_receipt.json'
# The tool is the trust anchor: pinned to its bytes as committed with the rename (2296c629).
# A regeneration that edits the tool must update this pin in the same commit.
RENAME_TOOL_SHA256='4ab2bf7247b51579b6e6fa3e771666b3a084f9d771f0a0a698bb6dd43a6487d3'
RENAMED_SCOPES={'Simulation.sie':'Simulation.mom'}
LOADERS=('scripts/dev/measure_mom_fmm_scaling.py','scripts/dev/measure_sie_fmm_scaling.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def package(root):
    root=Path(root);files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in CODE_SUFFIXES}
    return dict(files=files,source_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())


def check_pins(pins):
    for name,expected in pins.items():
        path=Path(name)
        if not path.is_file() or digest(path)!=expected:raise ValueError(f'Missing or changed original input: {name}')


def fragile_inputs(names):
    """Historical inputs must name persistent copies: never the running interpreter's site-packages, a .worktrees checkout or a temporary directory."""
    live={Path(sysconfig.get_path(key)).resolve() for key in ('purelib','platlib')};temporary={Path('/tmp').resolve(),Path(tempfile.gettempdir()).resolve()};found=[]
    for name in names:
        path=Path(name).resolve()
        if any(path.is_relative_to(root) for root in live):found.append((name,'the live environment site-packages'))
        elif '.worktrees' in path.parts:found.append((name,'a .worktrees checkout'))
        elif any(path.is_relative_to(root) for root in temporary):found.append((name,'a temporary directory'))
    return sorted(found)


def relocate(path,roots):
    """Map a historical absolute path through reviewed same-SHA root moves (longest whole-component prefix)."""
    path=Path(path);matches=[Path(old) for old in roots if path.is_relative_to(old)]
    if not matches:return path
    old=max(matches,key=lambda item:len(item.parts));return Path(roots[str(old)])/path.relative_to(old)


def scopes(path,text=None):
    result={}
    def visit(node,prefix=()):
        for item in ast.iter_child_nodes(node):
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                name=(*prefix,item.name);result['.'.join(name)]=ast.dump(item,include_attributes=False);visit(item,name)
            else:visit(item,prefix)
    visit(ast.parse(Path(path).read_text() if text is None else text));return result


def loader(source):
    """Radial-order consumer loader of a tree, before or after the rename."""
    return next(Path(source)/name for name in LOADERS if (Path(source)/name).is_file())


def rename_tool(source):
    path=Path(source)/RENAME_TOOL
    if not path.is_file():raise ValueError(f'Rename tool missing: {RENAME_TOOL}')
    if digest(path)!=RENAME_TOOL_SHA256:raise ValueError('Rename tool identity differs from the reviewed pin')
    spec=importlib.util.spec_from_file_location('consumer_rename_tool_'+digest(path)[:12],path);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool);return tool


def rename_rows(source):
    path=Path(source)/RENAME_RECEIPT
    if not path.is_file():raise ValueError(f'Rename receipt missing: {RENAME_RECEIPT}')
    return {row['old_path']:row for row in json.loads(path.read_text())['files']}


def pre_rename_view(source,old_source,files):
    """Live package digests in the pre-rename layout.

    A live file takes its pre-rename name and digest only when its receipt row is reproduced
    from the frozen consumer-epoch bytes: sha256(old)==old_sha256, tool path map old->new and
    sha256(tool(old))==new_sha256==live digest. A receipt row that the tool does not reproduce
    is rejected; a live file that differs from tool(old) keeps its new name and so shows as a
    changed consumer-epoch file.
    """
    tool=rename_tool(source);rows={row['new_path'].removeprefix('witwin/maxwell/'):row for row in rename_rows(source).values()
          if row['old_path'].startswith('witwin/maxwell/')}
    view={}
    for name,value in files.items():
        row=rows.get(name);old=Path(old_source)/row['old_path'] if row is not None else None
        if row is None or not old.is_file():
            view[name]=value;continue
        data=old.read_bytes()
        if digest(old)!=row['old_sha256']:
            # Not the receipt's pre-rename bytes (e.g. a post-rename epoch): compare directly.
            view[name]=value;continue
        if (tool.map_path(row['old_path'])!=row['new_path']
                or hashlib.sha256(tool.transform_bytes(row['old_path'],data)).hexdigest()!=row['new_sha256']):
            raise ValueError(f'Rename receipt row is not reproduced by the rename tool: {row["old_path"]}')
        if value==row['new_sha256']:view[row['old_path'].removeprefix('witwin/maxwell/')]=row['old_sha256']
        else:view[name]=value
    return view


def reproduced(source,old_rel,old_path,live_path):
    """The pinned tool maps the frozen file onto the live file (path and bytes)."""
    tool=rename_tool(source)
    return (Path(source)/tool.map_path(old_rel)==Path(live_path)
            and hashlib.sha256(tool.transform_bytes(old_rel,Path(old_path).read_bytes())).hexdigest()==digest(live_path))


def renamed_scope(source,old_path,rel_path,name):
    """Scope ``name`` of tool(old file), spelled with its renamed qualified name."""
    text=rename_tool(source).transform_bytes('witwin/maxwell/'+rel_path,Path(old_path).read_bytes()).decode()
    return scopes(None,text).get(RENAMED_SCOPES.get(name,name))


def helper(source,roots=None):
    path=loader(source);spec=importlib.util.spec_from_file_location('closed_witness_consumer_'+digest(path)[:12],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Closed witnesses embed their original absolute paths; the unchanged loader reads them through the reviewed relocation, and every read stays SHA-bound.
    # The substitution relies on the loader using Path only as a constructor (no isinstance, classmethods or annotations).
    if roots:module.Path=lambda *parts:relocate(Path(*parts),roots)
    module.configure_radial_series();return module


def import_probe(source,core_target):
    code='''from pathlib import Path
import hashlib,importlib.metadata,json,torch,witwin.core,witwin.maxwell
core=importlib.metadata.distribution('witwin')
metadata={str(Path(core.locate_file(f)).resolve()):hashlib.sha256(Path(core.locate_file(f)).read_bytes()).hexdigest() for f in core.files if '.dist-info/' in str(f) and Path(core.locate_file(f)).is_file()}
print(json.dumps(dict(maxwell=witwin.maxwell.__file__,core=witwin.core.__file__,core_module_version=witwin.core.__version__,core_distribution_version=core.version,core_metadata_files=metadata,cuda_initialized=torch.cuda.is_initialized())))
'''
    env=dict(os.environ,PYTHONPATH=str(source)+os.pathsep+str(core_target),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    run=subprocess.run([sys.executable,'-c',code],cwd='/tmp',env=env,capture_output=True,text=True,check=True);value=json.loads(run.stdout)
    if Path(value['maxwell']).resolve()!=Path(source)/'witwin/maxwell/__init__.py' or Path(value['core']).resolve()!=Path(core_target)/'witwin/core/__init__.py' or value['cuda_initialized']:raise ValueError('Actual imports do not match the declared frozen source and genuine core target')
    return value


def compatibility(source,old_source,current,previous,core,old_core):
    source=Path(source);old_source=Path(old_source);view=pre_rename_view(source,old_source,current['files']);old_sie={k:v for k,v in previous['files'].items() if k.startswith(('sie/','mom/'))};new_sie={k:v for k,v in view.items() if k.startswith(('sie/','mom/'))}
    if not old_sie or new_sie!=old_sie:raise ValueError('Prescribed consumer-epoch SIE files changed; metadata cannot requalify them')
    for path in PROTECTED_FILES:
        if view.get(path)!=previous['files'].get(path):raise ValueError(f'Prescribed shared numerical file changed: {path}')
    scope_hashes={}
    for path,names in PROTECTED_SCOPES.items():
        left=scopes(old_source/'witwin/maxwell'/path);right=scopes(source/'witwin/maxwell'/path)
        for name in names:
            # A post-rename consumer epoch already spells the renamed scope.
            live=RENAMED_SCOPES.get(name,name);old=name if name in left else live
            if old not in left or live not in right:raise ValueError(f'Prescribed shared numerical scope changed: {path}:{name}')
            if left[old]!=right[live] and (old==live or renamed_scope(source,old_source/'witwin/maxwell'/path,path,name)!=right[live]):raise ValueError(f'Prescribed shared numerical scope changed: {path}:{name}')
            scope_hashes[path+':'+name]=hashlib.sha256(left[old].encode()).hexdigest()
    core_protected={k:v for k,v in old_core['files'].items() if k not in ('__init__.py','structure.py')}
    if not core_protected or any(core['files'].get(k)!=v for k,v in core_protected.items()):raise ValueError('Prescribed shared core numerical file changed')
    changed=[name for name in sorted(current['files'].keys()|previous['files'].keys()) if current['files'].get(name)!=previous['files'].get(name)]
    return dict(comparison_baseline_source_root=str(old_source),comparison_baseline_source_sha256=previous['source_sha256'],comparison_baseline_role='preserved unqualified consumer, not the original raw-witness provider',unchanged_sie_files=len(old_sie),unchanged_shared_numerical_files={k:view[k] for k in PROTECTED_FILES},unchanged_shared_scopes=scope_hashes,unchanged_core_numerical_files=core_protected,changed_package_files=changed,changed_shared_or_public_files=[k for k in changed if not k.startswith(('fdtd/','execution/'))],changed_core_files=[k for k in sorted(core['files'].keys()|old_core['files'].keys()) if core['files'].get(k)!=old_core['files'].get(k)],overall_changes_observer_only=False,provider_capability_requalified=False,scope='Exact prescribed file/scope identity only. Shared constructor, mapped-material, source and public orchestration changes are enumerated; no blanket observer-only or numerical-equivalence claim.')


def derive(source,core_target,expected_source,expected_core,expected_core_metadata,baseline_bridge,baseline_sha,baseline_provenance,provenance_sha,output=None):
    source=Path(source).resolve();core_target=Path(core_target).resolve();baseline_bridge=Path(baseline_bridge).resolve();baseline_provenance=Path(baseline_provenance).resolve();started=time.time()
    if digest(baseline_bridge)!=baseline_sha or digest(baseline_provenance)!=provenance_sha:raise ValueError('Preserved consumer bridge/provenance SHA differs')
    current=package(source/'witwin/maxwell');core=package(core_target/'witwin/core')
    if current['source_sha256']!=expected_source:raise ValueError('Wrong frozen consumer source identity')
    if core['source_sha256']!=expected_core:raise ValueError('Wrong genuine core source identity')
    old=json.loads(baseline_bridge.read_text());proof=json.loads(baseline_provenance.read_text());roots=dict(proof.get('relocated_roots',{}));old_source=relocate(proof['source_root'],roots);previous=proof['current_package_before'];pins=dict(proof['inputs_sha256'])
    if proof.get('current_native_actions_executed') is not False or proof.get('production_qualification') is not False or proof.get('output_sha256')!=baseline_sha or proof.get('current_package_after')!=previous:raise ValueError('Baseline is not unchanged unqualified consumer metadata')
    if fragile:=fragile_inputs([*pins,str(old_source),*roots.values()]):raise ValueError(f'Historical input names {fragile[0][1]}; pin a persistent archived copy: {fragile[0][0]}')
    if package(old_source/'witwin/maxwell')!=previous:raise ValueError('Preserved baseline consumer source changed')
    check_pins(pins);old_helper=helper(old_source,roots)
    if old.get('production_source_hash')!=old_helper.production_source_hash() or old.get('production_qualification') is not False:raise ValueError('Preserved old consumer header changed')
    old_selections=old_helper.load_radial_order_bridge(baseline_bridge)['selections']
    check=compatibility(source,old_source,current,previous,core,proof['actual_core'])
    original_input=json.loads(relocate(old['original_input']['path'],roots).read_text())
    provider_root=relocate(original_input['provider'],roots);provider=package(provider_root/'witwin/maxwell')
    provider_sie={name:value for name,value in provider['files'].items() if name.startswith('sie/')}
    current_sie={name:value for name,value in pre_rename_view(source,old_source,current['files']).items() if name.startswith(('sie/','mom/'))}
    provider_hash=hashlib.sha256()
    for path in sorted((provider_root/'witwin/maxwell').rglob('*.py')):
        provider_hash.update(str(path.relative_to(provider_root)).encode());provider_hash.update(path.read_bytes())
    check['original_provider_comparison']=dict(source_root=str(provider_root),original_provider_production_source_hash=provider_hash.hexdigest(),changed_sie_files_vs_current=[name for name in sorted(provider_sie.keys()|current_sie.keys()) if provider_sie.get(name)!=current_sie.get(name)],provider_capability_requalified=False,scope='Original witness provider and preserved consumer are distinct epochs. Prescribed file continuity applies only relative to that consumer; earlier provider differences are not erased or newly qualified.')
    actual=import_probe(source,core_target)
    if actual['core_module_version']!=actual['core_distribution_version']:raise ValueError('Genuine core module and distribution versions differ')
    metadata_files=actual['core_metadata_files'];meta=[value for key,value in metadata_files.items() if key.endswith('.dist-info/METADATA')]
    if meta!=[expected_core_metadata]:raise ValueError('Genuine core wheel METADATA identity differs')
    current_helper=helper(source,roots)
    live_loader,old_loader=loader(source),loader(old_source)
    if digest(live_loader)!=digest(old_loader) and not reproduced(source,old_loader.relative_to(old_source).as_posix(),old_loader,live_loader):raise ValueError('Consumer loader contract changed; explicit review required')
    current_helper.configure_radial_series()
    candidate=dict(version='scalar_fmm_radial_order_bridge_v1',plan_hash=current_helper.plan_hash(),production_source_hash=current_helper.production_source_hash(),production_qualification=False,**{key:copy.deepcopy(old[key]) for key in ('original_manifest','original_input','original_receipt','original_terminal','records')})
    if set(candidate['records'])!={str(i) for i in range(24)}:raise ValueError('All24 closed original witnesses are required')
    for item in [candidate[k] for k in ('original_manifest','original_input','original_receipt','original_terminal')]+list(candidate['records'].values()):
        name=str(relocate(item['path'],roots));expected=item['sha256']
        if pins.get(name)!=expected:raise ValueError('Original closed witness differs from preserved input pins')
    pins[str(baseline_bridge)]=baseline_sha;pins[str(baseline_provenance)]=provenance_sha;pins.update(metadata_files)
    pins[str(loader(source))]=digest(loader(source))
    if output is not None:
        output=Path(output)
        with output.open('x') as stream:json.dump(candidate,stream,indent=2);stream.write('\n')
        candidate_path=output;temp=None
    else:
        temp=tempfile.TemporaryDirectory(prefix='consumer-only-bridge-');candidate_path=Path(temp.name)/'candidate.json';candidate_path.write_text(json.dumps(candidate))
    try:
        selections=current_helper.load_radial_order_bridge(candidate_path)['selections']
        if len(selections)!=8 or any(v.get('production_qualification') is not False for v in selections.values()):raise ValueError('Consumer selection must remain explicitly unqualified')
        for name,value in selections.items():
            if value['original_order_selection']!=old_selections[name]['original_order_selection']:raise ValueError('An original provider selection was relabelled')
        for item in candidate['records'].values():
            record=json.loads(relocate(item['path'],roots).read_text());selection=record['order_selection']
            if selection.get('source_path') is not None:
                path=str(relocate(selection['source_path'],roots));expected=selection['source_sha256']
                if digest(path)!=expected:raise ValueError('Original referenced admission changed')
                pins[path]=expected
        check_pins(pins)
        if package(source/'witwin/maxwell')!=current or package(core_target/'witwin/core')!=core or package(old_source/'witwin/maxwell')!=previous or package(provider_root/'witwin/maxwell')!=provider:raise ValueError('Code changed while deriving consumer metadata')
        return dict(schema='explicit-consumer-only-bridge-derivation/1',started_unix_s=started,source_root=str(source),core_target=str(core_target),source_before=current,source_after=package(source/'witwin/maxwell'),core_before=core,core_after=package(core_target/'witwin/core'),actual_imports=actual,baseline=dict(bridge=str(baseline_bridge),sha256=baseline_sha,provenance=str(baseline_provenance),provenance_sha256=provenance_sha,historical_unique_pins=len(proof['inputs_sha256']),direct_original_witnesses=24,witnesses_included_in_historical_pins=True),inputs_sha256=pins,relocated_roots=roots,compatibility=check,consumer_loader_source_hash=candidate['production_source_hash'],preserved_baseline_consumer_hash=old['production_source_hash'],original_provider_source_hash=check['original_provider_comparison']['original_provider_production_source_hash'],candidate=candidate,current_selections=selections,output=str(output) if output else None,output_sha256=digest(output) if output else None,current_native_actions_executed=False,provider_capability_requalified=False,production_qualification=False,final_release_binding=False,scope='Legacy production_source_hash at the NEW outer header identifies this consumer loader only. Every original selection/provider hash/closed receipt remains unchanged. No provider solve, kernel action, oracle generation or resource campaign executed.')
    finally:
        if temp:temp.cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','core-target','baseline-bridge','baseline-provenance'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-source','expected-core','expected-core-metadata','baseline-sha256','provenance-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--output',type=Path,help='New preparation-only metadata path; omitted means read-only temporary validation')
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists() or a.output and a.output.exists():raise ValueError('Use new report/output paths; old evidence is never overwritten')
    r=derive(a.source_root,a.core_target,a.expected_source,a.expected_core,a.expected_core_metadata,a.baseline_bridge,a.baseline_sha256,a.baseline_provenance,a.provenance_sha256,a.output);a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(historical_unique_pins=r['baseline']['historical_unique_pins'],original_witnesses=24,all_current_inputs=len(r['inputs_sha256']),changed_files=len(r['compatibility']['changed_package_files']),current_native_actions_executed=False,production_qualification=False)))
