"""Bound per-module artifact inputs; no global environment union or outcome changes."""
from contextlib import contextmanager
from pathlib import Path,PurePosixPath
import json,os,re
from shard_common import digest

SCHEMA='module-scoped-artifact-environment/1'


def artifact_environment(environment):
    return {key:value for key,value in environment.items() if key.startswith('GSP_')}


def load_policy(path,source):
    source=Path(source).resolve()
    raw=json.loads(Path(path).read_text())
    if raw.get('schema')!=SCHEMA or not isinstance(raw.get('modules'),dict) or not isinstance(raw.get('files'),dict):
        raise ValueError('Invalid module-scoped artifact environment schema')
    for name,expected in raw['files'].items():
        item=Path(name)
        if not item.is_absolute() or str(item.resolve())!=name or not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Artifact pins require canonical absolute paths and SHA256')
        if not item.is_file() or digest(item)!=expected:
            raise ValueError(f'Missing or changed module artifact: {name}')
    for module,entry in raw['modules'].items():
        parsed=PurePosixPath(module)
        if parsed.is_absolute() or '..' in parsed.parts or str(parsed)!=module or not module.endswith('.py') or '::' in module:
            raise ValueError(f'Invalid exact module path: {module}')
        actual=(source/module).resolve()
        if not actual.is_relative_to(source) or not actual.is_file():
            raise ValueError(f'Module does not exist inside source: {module}')
        values=entry.get('environment') if isinstance(entry,dict) else None
        if not isinstance(values,dict):
            raise ValueError('Each module requires an explicit artifact environment')
        for key,value in values.items():
            if not isinstance(key,str) or not re.fullmatch('GSP_[A-Z0-9_]+',key) or any(word in key for word in ('API_KEY','TOKEN','PASSWORD','SECRET')) or not isinstance(value,str):
                raise ValueError('Only declared GSP artifact input strings are allowed')
            if key.endswith('_SHA256'):
                path_key=entry.get('sha_pairs',{}).get(key,key[:-7])
                original=values.get(path_key)
                if not isinstance(original,str) or not re.fullmatch('[0-9a-f]{64}',value) or raw['files'].get(str(Path(original).resolve()))!=value:
                    raise ValueError(f'Orphan or incorrect artifact SHA pair: {key}')
            else:
                item=Path(value)
                if not item.is_absolute() or str(item.resolve()) not in raw['files']:
                    raise ValueError(f'Artifact environment path lacks a pinned input: {key}')
    return raw


def module_key(path,source):
    actual=Path(path).resolve();source=Path(source).resolve()
    return str(actual.relative_to(source)) if actual.is_relative_to(source) else None


def for_module(base,policy,module):
    if policy is None:return dict(base)
    result={key:value for key,value in base.items() if not key.startswith('GSP_')}
    result.update(policy['modules'].get(module,{}).get('environment',{}))
    return result


@contextmanager
def collection_scope(policy,module):
    previous=artifact_environment(os.environ)
    expected=artifact_environment(for_module(os.environ,policy,module))
    for key in list(os.environ):
        if key.startswith('GSP_'):del os.environ[key]
    os.environ.update(expected)
    try:
        yield expected
    finally:
        for key in list(os.environ):
            if key.startswith('GSP_'):del os.environ[key]
        os.environ.update(previous)
