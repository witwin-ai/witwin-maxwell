"""Consumer-only metadata from immutable closed witnesses, never provider requalification."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,os,re,subprocess,sys,sysconfig,tempfile,time,tomllib

CODE_SUFFIXES={'.py','.cu','.cpp','.h','.cuh'}
PROTECTED_FILES=('constants.py','materials.py','network_math.py','rational.py','waveforms.py')
PROTECTED_SCOPES={'media.py':('Material.evaluate_static','Material.evaluate_at_frequency'),
                  'sources.py':('evaluate_source_time','evaluate_source_time_normalization','polarization_vector'),
                  'scene.py':('_check_structures','Scene.__post_init__','Scene._check','Scene.updated'),
                  'simulation.py':('Simulation.sie',)}
# The immutable Scene (0.7.0 phase 7a) replaced Scene.add_structure by the constructor/updated() check
# _check_structures: the same structure type contract (the same TypeError), the collection keeps the given
# order. The pin covers what add_structure covered, the check and the store: _check_structures is the check,
# and the dispatch path that runs it and stores its result is pinned with it (construction:
# Scene.__post_init__ -> Scene._check; updated(): Scene.updated -> Scene._check, which stores the checked
# value). PREDECESSOR_SCOPES is the second part of the same transition: consumer epochs frozen before it
# (pre-rename 11-07, renamed 11-08, ..., round 7) only have Scene.add_structure, and the tests and
# qualifications that derive two such trees against each other still have to compare that scope. When
# neither tree has a pinned scope, its predecessor is compared instead; a pair with one spelling on each side
# (old epoch vs immutable Scene) is refused, so only a conventional re-receipt crosses the transition.
# User decisions 2026-11-12 (RFIC line owner session, AskUserQuestion): '改为钩住 _check_structures（推荐）',
# then '也保护分发路径（推荐）'.
PREDECESSOR_SCOPES={name:'Scene.add_structure' for name in PROTECTED_SCOPES['scene.py']}
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
# Feature groups: consumer-epoch subtrees pinned by a signature receipt chain instead of per file
# (docs/assessments/consumer-feature-groups/README.md). The consumer imports the group but does not
# run its numerics; the group is pinned by signature, its interface with the core is frozen and the
# import probe checks that importing it rebinds no core object. Adding a group, widening a prefix or
# changing the exclusions, op namespaces or evidence files below is a bridge-source change and needs
# a conventional re-receipt.
FEATURE_GROUPS={'layered':'mom/layered/'}
# The group file set is every file under the prefix except never-tracked interpreter caches.
GROUP_EXCLUDED_DIRS=frozenset({'__pycache__'});GROUP_EXCLUDED_SUFFIXES=frozenset({'.pyc'})
# Custom-op namespaces a group registers (TORCH_LIBRARY* / torch.library in the group sources).
GROUP_OP_NAMESPACES={'layered':['witwin_maxwell_mom_layered']}
# Committed evidence whose numerical_lock (scripts/dev/rfic/evidence_lock.py) binds a group receipt.
GROUP_EVIDENCE={'layered':('tests/mom/layered/data/fixture_r6_native_evidence.json','tests/mom/layered/data/fixture_r6b_shunt_evidence.json','tests/mom/layered/data/fixture_r4b_closed_comparison.json')}
GROUP_RECEIPTS='docs/assessments/consumer-feature-groups';MANIFEST='scripts/dev/integration_suites.toml'
RECEIPT_SCHEMA='consumer-feature-group-receipt/1'
# T1 of numerical_equivalence_standard.md for fp64 run-level evidence: max|new-ref| <= T1_BOUND*max|ref|
# per field (evidence_lock.py BOUND). A receipt states the bound it was judged with.
T1_BOUND=1e-10
T1_STATEMENT=f'fp64 run level: max|new-ref| <= {T1_BOUND:g}*max|ref| per field'
OP_REGISTRATION=re.compile(r'''TORCH_LIBRARY(?:_FRAGMENT|_IMPL)?\s*\(\s*(\w+)|(?:\bLibrary|\bcustom_op|torch\.library\.(?:define|impl|register_fake|register_kernel|register_autograd))\s*\(\s*["'](\w+)(?:::|["'])''')


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


def group_files(root,group):
    """Every file of a feature group, keyed like package(): {path relative to witwin/maxwell: sha256}."""
    base=Path(root)/'witwin/maxwell';top=base/FEATURE_GROUPS[group]
    return {p.relative_to(base).as_posix():digest(p) for p in sorted(top.rglob('*'))
            if p.is_file() and not GROUP_EXCLUDED_DIRS&set(p.relative_to(top).parts) and p.suffix not in GROUP_EXCLUDED_SUFFIXES}


def group_signature(files):return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def group_module(group):return 'witwin.maxwell.'+FEATURE_GROUPS[group].rstrip('/').replace('/','.')


def within(name,module):return name==module or name.startswith(module+'.')


def group_interface(root,group):
    """AST import edges from package files outside every group into GROUP: [[file, group module, [names]]].

    Covers module-level and function-local imports, relative imports, `from package import group`
    and constant-string importlib.import_module/__import__ calls."""
    base=Path(root)/'witwin/maxwell';target=group_module(group);edges={}
    for path in sorted(base.rglob('*.py')):
        rel=path.relative_to(base)
        if '__pycache__' in rel.parts or rel.as_posix().startswith(tuple(FEATURE_GROUPS.values())):continue
        package=['witwin','maxwell',*rel.with_suffix('').parts][:-1];text=path.read_bytes()
        # Every edge spells the group's last name component; files that never do are not parsed.
        if target.rsplit('.',1)[1].encode() not in text:continue
        for node in ast.walk(ast.parse(text)):
            found=[]
            if isinstance(node,ast.Import):found=[(a.name,'import'+(f' as {a.asname}' if a.asname else '')) for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                module='.'.join(package[:len(package)-node.level+1]+([node.module] if node.module else [])) if node.level else node.module
                for a in node.names:
                    label=a.name+(f' as {a.asname}' if a.asname else '')
                    found.append((module,label) if within(module,target) else (module+'.'+a.name,'import'+(f' as {a.asname}' if a.asname else '')))
            elif (isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)
                  and getattr(node.func,'attr',getattr(node.func,'id',None)) in ('import_module','__import__')):
                value=node.args[0].value
                if within(value,target) or '.'+target.rsplit('.',1)[1] in value:found=[(value,'dynamic import')]
            for module,label in found:
                if within(module,target):edges.setdefault((rel.as_posix(),module),set()).add(label)
    return [[name,module,sorted(labels)] for (name,module),labels in sorted(edges.items())]


def op_namespaces(root,group):
    """Custom-op namespaces registered by the group's sources."""
    top=Path(root)/'witwin/maxwell'/FEATURE_GROUPS[group]
    return sorted({m.group(1) or m.group(2) for name in group_files(root,group) for m in OP_REGISTRATION.finditer((Path(root)/'witwin/maxwell'/name).read_text(errors='replace'))}) if top.is_dir() else []


def archive_group(old_source,previous,group):
    """The group as admitted by the consumer archive: its frozen source, bound to current_package_before."""
    files=group_files(old_source,group)
    if {k:v for k,v in files.items() if Path(k).suffix in CODE_SUFFIXES}!={k:v for k,v in previous['files'].items() if k.startswith(FEATURE_GROUPS[group])}:
        raise ValueError(f'Preserved baseline consumer source changed: feature group {group}')
    return dict(files=files,signature=group_signature(files),interface=group_interface(old_source,group),op_namespaces=op_namespaces(old_source,group))


def group_chain(source,group):
    """Receipts of GROUP in order: [(repository path, sha256, receipt)]. No directory is an empty chain."""
    directory=Path(source)/GROUP_RECEIPTS/group
    paths=sorted(directory.iterdir()) if directory.is_dir() else []
    if [p.name for p in paths]!=[f'{i:04d}.json' for i in range(1,len(paths)+1)]:
        raise ValueError(f'Feature-group receipt chain broken: {GROUP_RECEIPTS}/{group} must hold only 0001.json..NNNN.json')
    return [(f'{GROUP_RECEIPTS}/{group}/{p.name}',digest(p),json.loads(p.read_text())) for p in paths]


def group_declarations(source):
    """consumer_group_receipt values of [[merge_declarations]]: {(receipt path, sha256): branch or commit}."""
    path=Path(source)/MANIFEST;found={}
    for entry in tomllib.loads(path.read_text()).get('merge_declarations',[]) if path.is_file() else []:
        if 'consumer_group_receipt' not in entry:continue
        name,_,sha=str(entry['consumer_group_receipt']).rpartition('@')
        if not name or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError(f'Malformed consumer_group_receipt declaration: {entry["consumer_group_receipt"]}')
        if Path(name).is_absolute() and (fragile:=fragile_inputs([name])):raise ValueError(f'Historical input names {fragile[0][1]}; pin a persistent archived copy: {name}')
        if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith(GROUP_RECEIPTS+'/'):
            raise ValueError(f'Feature-group receipt declaration must name a repository receipt under {GROUP_RECEIPTS}: {name}')
        found[(name,sha)]=entry.get('branch',entry.get('commit'))
    return found


def acquisition_tree(source,commit,group):
    """The group's tracked files at COMMIT; fails closed when the commit is not in SOURCE's repository."""
    source=Path(source).resolve();git=['git','-C',str(source)]
    top=subprocess.run([*git,'rev-parse','--show-toplevel'],capture_output=True,text=True)
    if top.returncode or Path(top.stdout.strip()).resolve()!=source:
        raise ValueError(f'Acquisition commit is not available: {source} is not a git checkout (snapshot or copy); the feature-group receipt cannot be bound')
    if not re.fullmatch('[0-9a-f]{40}',str(commit)) or subprocess.run([*git,'cat-file','-e',f'{commit}^{{commit}}'],capture_output=True).returncode:
        raise ValueError(f'Acquisition commit is not available: {commit} (shallow clone or unfetched); the feature-group receipt cannot be bound')
    top=Path(FEATURE_GROUPS[group]);files={}
    for row in filter(None,subprocess.run([*git,'ls-tree','-r','-z',commit,'--','witwin/maxwell/'+FEATURE_GROUPS[group]],capture_output=True,check=True).stdout.split(b'\0')):
        meta,name=row.split(b'\t',1);mode,kind,obj=meta.decode().split();name=name.decode().removeprefix('witwin/maxwell/')
        if GROUP_EXCLUDED_DIRS&set(Path(name).relative_to(top).parts) or Path(name).suffix in GROUP_EXCLUDED_SUFFIXES:continue
        if kind!='blob' or mode not in ('100644','100755'):raise ValueError(f'Acquisition tree differs from the receipt: {name} is not a regular file at {commit}')
        files[name]=hashlib.sha256(subprocess.run([*git,'cat-file','blob',obj],capture_output=True,check=True).stdout).hexdigest()
    return files


def check_numerical_lock(source,group,name,receipt,head):
    """Check 4: the receipt is bound to the committed evidence lock and to the tree it was acquired on."""
    lock=receipt.get('numerical_lock') or {};evidence=lock.get('evidence') or {}
    if sorted(evidence)!=sorted(GROUP_EVIDENCE[group]):raise ValueError(f'Numerical lock binding differs: {name} must bind exactly {sorted(GROUP_EVIDENCE[group])}')
    for path,record in sorted(evidence.items()):
        if (record.get('summary') or {}).get('T2')!=0:raise ValueError(f'Numerical lock has unaccepted T2: {name} {path} {record.get("summary")}')
        if record.get('acquisition_commit')!=lock.get('acquisition_commit') or record.get('tier_table_sha256')!=lock.get('tier_table_sha256'):
            raise ValueError(f'Numerical lock binding differs: {name} {path} names another acquisition or tier table')
        committed=Path(source)/path
        if not committed.is_file():raise ValueError(f'Numerical lock binding differs: committed evidence missing: {path}')
        renewals=json.loads(committed.read_text()).get('numerical_lock',{}).get('renewals',[])
        if (renewals[-1:]!=[record]) if head else (record not in renewals):
            raise ValueError(f'Numerical lock binding differs: {path} does not carry the renewal of {name}'+(' as its latest' if head else ''))
    accepted=[entry for record in evidence.values() for entry in record.get('accepted_t2') or []]
    if accepted and (receipt.get('approval') in (None,'','T1-auto') or any(not str(entry.get('approval') or '').strip() for entry in accepted)):
        raise ValueError(f'Numerical lock has accepted T2 without a user approval: {name}')
    if not accepted and receipt.get('approval')!='T1-auto':raise ValueError(f'Numerical lock binding differs: {name} approval must be T1-auto without accepted T2')
    if lock.get('t1_bound')!=dict(relative=T1_BOUND,statement=T1_STATEMENT):raise ValueError(f'Numerical lock T1 bound differs from the standard: {name}')
    movers=lock.get('t1_movers') or [];fields={f for record in evidence.values() for f in record.get('renewed_fields') or []}-{entry.get('field') for entry in accepted}
    if sorted(m.get('field') for m in movers)!=sorted(fields) or any(m.get('tier')!='T1' or not 0<=m.get('delta_over_bound',-1)<=1 for m in movers):
        raise ValueError(f'Numerical lock binding differs: {name} T1 movers do not match the renewed fields within the T1 bound')
    if acquisition_tree(source,lock.get('acquisition_commit'),group)!=receipt.get('files'):
        raise ValueError(f'Acquisition tree differs from the receipt: {name} at {lock.get("acquisition_commit")}')


def check_group_chain(source,group,archive,anchor):
    """Checks 2-5 of a feature group against its receipt chain; an empty chain admits only the archive's group.

    ANCHOR is the archive's frozen consumer source identity (current_package_before source_sha256):
    a conventional re-receipt that changes the epoch breaks the chain, which then restarts empty."""
    chain=group_chain(source,group);declared=group_declarations(source);predecessor=(archive['signature'],None,None)
    for name,sha,receipt in chain:
        if receipt.get('schema')!=RECEIPT_SCHEMA or receipt.get('group')!=group or receipt.get('prefix')!=FEATURE_GROUPS[group]:raise ValueError(f'Feature-group receipt chain broken: {name} is not a {group} receipt')
        if receipt.get('archive_source_sha256')!=anchor:raise ValueError(f'Feature-group receipt chain broken: {name} is anchored to another consumer epoch')
        if (receipt.get('predecessor_signature'),receipt.get('predecessor_receipt'),receipt.get('predecessor_receipt_sha256'))!=predecessor:raise ValueError(f'Feature-group receipt chain broken: predecessor mismatch at {name}')
        if group_signature(receipt.get('files') or {})!=receipt.get('signature'):raise ValueError(f'Feature-group receipt chain broken: {name} signature does not match its files')
        if declared.get((name,sha),False)!=receipt.get('merge_declaration'):raise ValueError(f'Feature-group receipt is not declared by its merge declaration: {name}@{sha}')
        predecessor=(receipt['signature'],name,sha)
    head=chain[-1][2]['files'] if chain else archive['files'];live=group_files(source,group);where='chain head '+chain[-1][0] if chain else 'consumer archive (empty chain)'
    if live.keys()!=head.keys():raise ValueError(f'Feature-group file set differs from the {where}: {group} extra {sorted(live.keys()-head.keys())}, missing {sorted(head.keys()-live.keys())}')
    if group_signature(live)!=group_signature(head):raise ValueError(f'Feature-group signature differs from the {where}: {group} {[k for k in sorted(live) if live[k]!=head[k]]}')
    for index,(name,sha,receipt) in enumerate(chain):check_numerical_lock(source,group,name,receipt,index==len(chain)-1)
    snapshots=[('consumer archive',archive)]+[(name,receipt) for name,sha,receipt in chain]+[('live tree',dict(interface=group_interface(source,group),op_namespaces=op_namespaces(source,group)))]
    for name,snapshot in snapshots:
        if snapshot.get('interface')!=archive['interface']:raise ValueError(f'Feature-group interface changed: {name}: {group} {snapshot.get("interface")} != {archive["interface"]}')
        if snapshot.get('op_namespaces')!=archive['op_namespaces'] or archive['op_namespaces']!=GROUP_OP_NAMESPACES[group]:
            raise ValueError(f'Feature-group op namespaces changed: {name}: {group} {snapshot.get("op_namespaces")} (reviewed {GROUP_OP_NAMESPACES[group]})')
    return dict(prefix=FEATURE_GROUPS[group],files=len(live),signature=group_signature(live),archive_signature=archive['signature'],chain_length=len(chain),
                head=f'{chain[-1][0]}@{chain[-1][1]}' if chain else None,interface=archive['interface'],op_namespaces=archive['op_namespaces'])


def helper(source,roots=None):
    path=loader(source);spec=importlib.util.spec_from_file_location('closed_witness_consumer_'+digest(path)[:12],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Closed witnesses embed their original absolute paths; the unchanged loader reads them through the reviewed relocation, and every read stays SHA-bound.
    # The substitution relies on the loader using Path only as a constructor (no isinstance, classmethods or annotations).
    if roots:module.Path=lambda *parts:relocate(Path(*parts),roots)
    module.configure_radial_series();return module


# Import-time rebinding audit (check 6), run inside import_probe's CPU subprocess before witwin is
# imported. Every core module (witwin.maxwell outside the feature groups, and witwin.core) is
# watched from its creation: attribute writes from other code are recorded with the writer's frames
# up to its module body, and each module's top-level objects (with function code/defaults and the
# attributes of the classes it defines) are remembered by id() right after its own body ran. After
# importing the package, every consumer-epoch module and every group module, the probe reports
# writes from other code, ids that changed since the module's own body (names the module declares
# global are its own lazy state), replaced sys.modules entries or parent attributes, and objects
# defined by a core module or class that come from group code.
REBINDING_AUDIT='''
import ast,functools,importlib,importlib.abc,importlib.machinery,json,sys,types
from pathlib import Path
_source,_core,_groups=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),json.loads(sys.argv[3])
_group_dirs=[(_source/'witwin/maxwell'/prefix).resolve() for prefix in _groups]
_group_modules=['witwin.maxwell.'+prefix.rstrip('/').replace('/','.') for prefix in _groups]
_watched=[(_source/'witwin/maxwell').resolve(),(_core/'witwin/core').resolve()]
_writes,_snapshots,_keep=[],{},[]
def _in_group(filename):
    return isinstance(filename,str) and Path(filename).is_absolute() and any(Path(filename).resolve().is_relative_to(d) for d in _group_dirs)
def _core_file(filename):
    return isinstance(filename,str) and Path(filename).is_absolute() and any(Path(filename).resolve().is_relative_to(d) for d in _watched) and not _in_group(filename)
def _note(module,name,value,frame):
    if isinstance(value,types.ModuleType) and value.__name__==module.__name__+'.'+name:return
    files=[]
    while frame is not None:
        files.append(frame.f_code.co_filename)
        if frame.f_code.co_name=='<module>':break
        frame=frame.f_back
    own=getattr(module,'__file__',None)
    if files and own and Path(files[0]).is_absolute() and Path(files[0]).resolve()==Path(own).resolve():return
    _writes.append(dict(kind='setattr',module=module.__name__,name=name,writer=files[0] if files else None,group=any(map(_in_group,files))))
class _Audited(types.ModuleType):
    def __setattr__(self,name,value):
        _note(self,name,value,sys._getframe(1));types.ModuleType.__setattr__(self,name,value)
    def __delattr__(self,name):
        _note(self,name,None,sys._getframe(1));types.ModuleType.__delattr__(self,name)
def _ids(module):
    out={}
    for key,value in list(vars(module).items()):
        out[key]=id(value);_keep.append(value)
        items=[(key,value)]
        if isinstance(value,type) and value.__module__==module.__name__:items+=[(key+'.'+k,v) for k,v in list(vars(value).items())]
        for name,item in items:
            out[name]=id(item);_keep.append(item)
            if isinstance(item,types.FunctionType):
                for attr in ('__code__','__defaults__','__kwdefaults__'):out[name+'.'+attr]=id(getattr(item,attr));_keep.append(getattr(item,attr))
    return out
class _Loader(importlib.abc.Loader):
    def __init__(self,inner):self.inner=inner
    def create_module(self,spec):return self.inner.create_module(spec)
    def exec_module(self,module):
        if type(module) is types.ModuleType:module.__class__=_Audited
        self.inner.exec_module(module)
        _snapshots[module.__name__]=(module,_ids(module))
class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path,target=None):
        if not name.startswith(('witwin.maxwell.','witwin.core.')) and name not in ('witwin.maxwell','witwin.core'):return None
        spec=importlib.machinery.PathFinder.find_spec(name,path)
        if spec is not None and isinstance(spec.loader,importlib.machinery.SourceFileLoader) and _core_file(spec.origin):spec.loader=_Loader(spec.loader)
        return spec
sys.meta_path.insert(0,_Finder())
def _from_group(value):
    for _ in range(16):
        if isinstance(value,(staticmethod,classmethod,types.MethodType)):value=value.__func__
        elif isinstance(value,(functools.partial,functools.partialmethod)):value=value.func
        elif isinstance(value,property):return any(_from_group(f) for f in (value.fget,value.fset,value.fdel) if f is not None)
        elif isinstance(value,types.FunctionType):return _in_group(value.__code__.co_filename)
        elif isinstance(value,type):return any(value.__module__==m or str(value.__module__).startswith(m+'.') for m in _group_modules)
        elif '__wrapped__' in getattr(value,'__dict__',{}):value=value.__dict__['__wrapped__']
        else:return False
    return False
def _defined(tree):
    names=set()
    def visit(body):
        for node in body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):names.add(node.name)
            elif isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                for target in (node.targets if isinstance(node,ast.Assign) else [node.target]):
                    names.update(n.id for n in ast.walk(target) if isinstance(n,ast.Name))
            for field in ('body','orelse','finalbody','handlers'):
                if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and isinstance(getattr(node,field,None),list):visit(getattr(node,field))
    visit(tree.body);return names
def _rebinding():
    found=list(_writes);written={(w['module'],w['name']) for w in _writes}
    for name,(module,before) in sorted(_snapshots.items()):
        if sys.modules.get(name) is not module:found.append(dict(kind='sys.modules',module=name,name=None,writer=None,group=None));continue
        parent,_,child=name.rpartition('.')
        # A watched parent's own attributes are covered by its snapshot (a package may bind a name over
        # its submodule); an unwatched parent (namespace package) has no code of its own.
        if parent not in _snapshots and parent in sys.modules and child in vars(sys.modules[parent]) and vars(sys.modules[parent])[child] is not module:
            found.append(dict(kind='parent attribute',module=parent,name=child,writer=None,group=None))
        tree=ast.parse(Path(module.__file__).read_bytes()) if getattr(module,'__file__',None) else ast.Module(body=[],type_ignores=[])
        lazy={n for node in ast.walk(tree) if isinstance(node,ast.Global) for n in node.names}
        after=_ids(module)
        for key in sorted(before.keys()|after.keys()):
            top=key.split('.')[0]
            if before.get(key)==after.get(key) or top in lazy or (name,top) in written:continue
            value=vars(module).get(top)
            if '.' not in key and isinstance(value,types.ModuleType) and value.__name__==name+'.'+key:continue
            found.append(dict(kind='rebound after import',module=name,name=key,writer=None,group=None))
        objects=[(key,vars(module).get(key)) for key in sorted(_defined(tree)) if key in vars(module)]
        objects+=[(key+'.'+k,v) for key,value in list(vars(module).items()) if isinstance(value,type) and value.__module__==name for k,v in list(vars(value).items())]
        found+=[dict(kind='group object',module=name,name=key,writer=None,group=True) for key,value in objects if _from_group(value)]
    return found
'''


def import_probe(source,core_target):
    code='''from pathlib import Path
import hashlib,importlib.metadata,json,torch,witwin.core,witwin.maxwell
core=importlib.metadata.distribution('witwin')
metadata={str(Path(core.locate_file(f)).resolve()):hashlib.sha256(Path(core.locate_file(f)).read_bytes()).hexdigest() for f in core.files if '.dist-info/' in str(f) and Path(core.locate_file(f)).is_file()}
# Every consumer-epoch module and every feature-group module is imported, so group import-time code runs.
modules=sorted('.'.join(p.relative_to(_source).with_suffix('').parts[:-1] if p.name=='__init__.py' else p.relative_to(_source).with_suffix('').parts)
               for top in ('sie','mom') for p in (_source/'witwin/maxwell'/top).rglob('*.py') if '__pycache__' not in p.parts)
for name in modules:importlib.import_module(name)
print(json.dumps(dict(maxwell=witwin.maxwell.__file__,core=witwin.core.__file__,core_module_version=witwin.core.__version__,core_distribution_version=core.version,core_metadata_files=metadata,cuda_initialized=torch.cuda.is_initialized(),audited_modules=len(_snapshots),imported_consumer_modules=len(modules),rebinding=_rebinding())))
'''
    env=dict(os.environ,PYTHONPATH=str(source)+os.pathsep+str(core_target),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    run=subprocess.run([sys.executable,'-c',REBINDING_AUDIT+code,str(source),str(core_target),json.dumps(sorted(FEATURE_GROUPS.values()))],cwd='/tmp',env=env,capture_output=True,text=True,check=True);value=json.loads(run.stdout)
    if Path(value['maxwell']).resolve()!=Path(source)/'witwin/maxwell/__init__.py' or Path(value['core']).resolve()!=Path(core_target)/'witwin/core/__init__.py' or value['cuda_initialized']:raise ValueError('Actual imports do not match the declared frozen source and genuine core target')
    if found:=value['rebinding']:
        first=next((item for item in found if item['group']),found[0])
        raise ValueError(('Core module rebinding by a feature group' if first['group'] else 'Core module rebinding after import')+f": {first['kind']} {first['module']}.{first['name']} (writer {first['writer']}); {len(found)} finding(s)")
    return value


def compare_scope(path,name,left,right,renamed=None):
    """(compared name, sha256) of pinned scope ``name`` of ``path``; ``left``/``right`` are scopes() of the
    baseline and live file, ``renamed(name)`` the scope of tool(baseline file). Refuses any other pair."""
    if name not in left and name not in right and name in PREDECESSOR_SCOPES:name=PREDECESSOR_SCOPES[name]
    # A post-rename consumer epoch already spells the renamed scope.
    live=RENAMED_SCOPES.get(name,name);old=name if name in left else live
    if old not in left or live not in right:raise ValueError(f'Prescribed shared numerical scope changed: {path}:{name}')
    if left[old]!=right[live] and (old==live or renamed is None or renamed(name)!=right[live]):raise ValueError(f'Prescribed shared numerical scope changed: {path}:{name}')
    return name,hashlib.sha256(left[old].encode()).hexdigest()


def compatibility(source,old_source,current,previous,core,old_core):
    source=Path(source);old_source=Path(old_source);view=pre_rename_view(source,old_source,current['files'])
    # A feature group leaves the per-file comparison only against an archive that admitted it under
    # its live prefix; a pre-rename archive has no chain and keeps its group files pinned per file.
    groups=[group for group,prefix in FEATURE_GROUPS.items() if any(k.startswith(prefix) for k in previous['files'])]
    grouped=tuple(FEATURE_GROUPS[group] for group in groups)
    old_sie={k:v for k,v in previous['files'].items() if k.startswith(('sie/','mom/')) and not k.startswith(grouped)};new_sie={k:v for k,v in view.items() if k.startswith(('sie/','mom/')) and not k.startswith(grouped)}
    if not old_sie or new_sie!=old_sie:raise ValueError('Prescribed consumer-epoch SIE files changed; metadata cannot requalify them')
    for path in PROTECTED_FILES:
        if view.get(path)!=previous['files'].get(path):raise ValueError(f'Prescribed shared numerical file changed: {path}')
    scope_hashes={}
    for path,names in PROTECTED_SCOPES.items():
        left=scopes(old_source/'witwin/maxwell'/path);right=scopes(source/'witwin/maxwell'/path)
        for name in names:
            name,value=compare_scope(path,name,left,right,lambda name:renamed_scope(source,old_source/'witwin/maxwell'/path,path,name))
            scope_hashes[path+':'+name]=value
    core_protected={k:v for k,v in old_core['files'].items() if k not in ('__init__.py','structure.py')}
    if not core_protected or any(core['files'].get(k)!=v for k,v in core_protected.items()):raise ValueError('Prescribed shared core numerical file changed')
    feature_groups={group:check_group_chain(source,group,archive_group(old_source,previous,group),previous['source_sha256']) for group in groups}
    changed=[name for name in sorted(current['files'].keys()|previous['files'].keys()) if current['files'].get(name)!=previous['files'].get(name)]
    return dict(comparison_baseline_source_root=str(old_source),comparison_baseline_source_sha256=previous['source_sha256'],comparison_baseline_role='preserved unqualified consumer, not the original raw-witness provider',unchanged_sie_files=len(old_sie),feature_groups=feature_groups,unchanged_shared_numerical_files={k:view[k] for k in PROTECTED_FILES},unchanged_shared_scopes=scope_hashes,unchanged_core_numerical_files=core_protected,changed_package_files=changed,changed_shared_or_public_files=[k for k in changed if not k.startswith(('fdtd/','execution/'))],changed_core_files=[k for k in sorted(core['files'].keys()|old_core['files'].keys()) if core['files'].get(k)!=old_core['files'].get(k)],overall_changes_observer_only=False,provider_capability_requalified=False,scope='Exact prescribed file/scope identity only. Shared constructor, mapped-material, source and public orchestration changes are enumerated; no blanket observer-only or numerical-equivalence claim.')


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
