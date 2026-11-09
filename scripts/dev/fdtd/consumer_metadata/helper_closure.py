"""Explicit test-helper artifact inheritance from actual static imports and calls."""
from pathlib import Path
import argparse,ast,copy,hashlib,json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_keys(node):
    found=set()
    for item in ast.walk(node):
        value=None
        if isinstance(item,ast.Call) and ast.unparse(item.func) in ('os.environ.get','os.getenv') and item.args:value=item.args[0]
        elif isinstance(item,ast.Subscript) and ast.unparse(item.value)=='os.environ':value=item.slice
        if isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value.startswith('GSP_'):found.add(value.value)
        elif isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add) and isinstance(value.left,ast.Constant) and isinstance(value.left.value,str) and value.left.value.startswith('GSP_'):found.add(value.left.value)
    return found


def index_source(source):
    source=Path(source).resolve();modules={};functions={}
    for path in sorted((source/'tests').rglob('test*.py')):
        rel=str(path.relative_to(source));tree=ast.parse(path.read_text());defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};modules[rel]=dict(path=path,tree=tree,functions=defs,imports={})
        for name,node in defs.items():functions[(rel,name)]=node
    def resolve(caller,module,level):
        parent=(source/caller).parent
        if level:
            for _ in range(level-1):parent=parent.parent
            target=parent/((module or '').replace('.','/')+'.py')
        else:
            target=parent/((module or '').replace('.','/')+'.py')
            if not target.is_file():target=source/((module or '').replace('.','/')+'.py')
        if target.is_file() and target.resolve().is_relative_to(source):return str(target.resolve().relative_to(source))
        return None
    for rel,record in modules.items():
        # Function-local imports are included only in their actual owning function.
        for node in record['tree'].body:
            imports=[]
            if isinstance(node,(ast.ImportFrom,ast.Import)):imports=[node]
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):imports=[n for n in ast.walk(node) if isinstance(n,(ast.ImportFrom,ast.Import))]
            owner=node.name if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) else None
            for item in imports:
                if isinstance(item,ast.ImportFrom):
                    target=resolve(rel,item.module,item.level)
                    if target in modules:
                        for name in item.names:
                            if name.name!='*':record['imports'][(owner,name.asname or name.name)]=(target,name.name)
                else:
                    for name in item.names:
                        target=resolve(rel,name.name,0)
                        if target in modules:record['imports'][(owner,name.asname or name.name)]=(target,None)
    graph={}
    for key,node in functions.items():
        rel,name=key;record=modules[rel];edges=set()
        def imported(alias):return record['imports'].get((name,alias),record['imports'].get((None,alias)))
        for item in ast.walk(node):
            if not isinstance(item,ast.Call):continue
            target=None
            if isinstance(item.func,ast.Name):
                alias=item.func.id;target=imported(alias)
                if target is None and alias in record['functions']:target=(rel,alias)
            elif isinstance(item.func,ast.Attribute) and isinstance(item.func.value,ast.Name):
                match=imported(item.func.value.id)
                if match and match[1] is None:target=(match[0],item.func.attr)
            if target in functions:edges.add(target)
        # Pytest calls named fixtures without a Python call expression in the test.
        for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:
            target=imported(arg.arg) or ((rel,arg.arg) if arg.arg in record['functions'] else None)
            if target in functions and any('fixture' in ast.unparse(d) for d in functions[target].decorator_list):edges.add(target)
        graph[key]=sorted(edges)
    return modules,functions,graph


def verify_artifact_pins(policy):
    for name,expected in policy['files'].items():
        path=Path(name)
        if not path.is_file() or digest(path)!=expected:raise ValueError(f'Missing or changed original helper artifact: {name}')


def pin_declared_artifacts(value,files):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            path=Path(value['path']).resolve();expected=value['sha256']
            if not path.is_file() or digest(path)!=expected:raise ValueError(f'Missing or changed declared original artifact: {path}')
            files[str(path)]=expected
        for child in value.values():pin_declared_artifacts(child,files)
    elif isinstance(value,list):
        for child in value:pin_declared_artifacts(child,files)


def derive(source,policy):
    verify_artifact_pins(policy)
    source=Path(source).resolve()
    for rel in policy['modules']:
        if not (source/rel).is_file():raise ValueError(f'Source lacks reviewed policy module: {rel}')
    result=copy.deepcopy(policy);modules,functions,graph=index_source(source);traces=[];unresolved=[];pins={}
    for caller,record in modules.items():
        roots=[(caller,name) for name in record['functions'] if name.startswith('test_')]
        visited=set();queue=[(key,[key]) for key in roots];requirements={};details=[];sha_pairs={}
        while queue:
            key,chain=queue.pop(0)
            if key in visited:continue
            visited.add(key);keys=env_keys(functions[key])
            if key[0]!=caller and keys:
                owner=policy['modules'].get(key[0]);values=owner.get('environment',{}) if owner else {}
                selected={k:v for k,v in values.items() if any(k==read or read.endswith('_') and k.startswith(read) for read in keys)}
                if not selected:
                    unresolved.append(dict(caller=caller,reader=list(key),keys=sorted(keys),reason='No reviewed original bundle for this reachable helper reader.'));continue
                for key_name,value in list(selected.items()):
                    if key_name+'_SHA256' in values:selected[key_name+'_SHA256']=values[key_name+'_SHA256']
                    for sha_key,path_key in owner.get('sha_pairs',{}).items():
                        if path_key==key_name:
                            selected[sha_key]=values[sha_key];sha_pairs[sha_key]=path_key
                for key_name,value in selected.items():
                    existing=requirements.get(key_name,result['modules'].get(caller,{}).get('environment',{}).get(key_name))
                    if existing is not None and existing!=value:raise ValueError(f'Conflicting helper artifact key in {caller}: {key_name}')
                    requirements[key_name]=value
                    if not key_name.endswith('_SHA256') and Path(value).suffix=='.json':pin_declared_artifacts(json.loads(Path(value).read_text()),result['files'])
                paths=sorted({part[0] for part in chain})
                for path in paths:pins[path]=digest(source/path)
                details.append(dict(chain=[list(p) for p in chain],reader_keys=sorted(keys),environment_keys=sorted(selected)))
            queue.extend((target,[*chain,target]) for target in graph[key])
        if requirements:
            entry=result['modules'].setdefault(caller,dict(environment={},qualification=False,source_kind='explicit_import_call_helper_closure'))
            entry['environment'].update(requirements);entry['helper_dependency_paths']=details
            if sha_pairs:entry.setdefault('sha_pairs',{}).update(sha_pairs)
            for rel,_ in visited:pins[rel]=digest(source/rel)
            traces.append(dict(caller=caller,requirements=requirements,paths=details))
    result['helper_dependency_source_pins']=pins
    for rel,value in pins.items():result['files'][str((source/rel).resolve())]=value
    result['helper_dependency_review']=dict(aliases=traces,unresolved=unresolved,scope='Static direct imports/calls and named fixtures only; no global environment inheritance. Original artifacts, provider hashes and receipt gates remain unchanged. Dynamic Python dependencies are not inferred or qualified.')
    verify_source_pins(result,source)
    return result


def verify_source_pins(policy,source):
    source=Path(source).resolve()
    for rel,expected in policy.get('helper_dependency_source_pins',{}).items():
        path=(source/rel).resolve()
        if not path.is_relative_to(source) or not path.is_file() or digest(path)!=expected:raise ValueError(f'Changed helper-consumer source: {rel}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--policy-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if digest(a.policy)!=a.policy_sha256:raise ValueError('Input policy SHA differs')
    if a.output.exists():raise ValueError('Use a new output; old policy/evidence is never overwritten')
    result=derive(a.source_root,json.loads(a.policy.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(modules=len(result['modules']),helper_consumers=len(result['helper_dependency_review']['aliases']),source_pins=len(result['helper_dependency_source_pins']),unresolved=len(result['helper_dependency_review']['unresolved']))))
