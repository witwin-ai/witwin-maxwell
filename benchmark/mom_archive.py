"""Create or verify an exact source/environment/numerical-evidence archive.

Usage: python -m benchmark.mom_archive create --report result.json --output run.zip
       python -m benchmark.mom_archive verify run.zip
Archive validity means integrity, never physical acceptance or gate certification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _command(argv, cwd):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def _provenance_records(value, location='$'):
    """Find explicitly recorded hashes, including resumed execution segments."""
    if isinstance(value, dict):
        if isinstance(value.get('source_sha256'), dict):
            yield location, value['source_sha256']
        for key, child in value.items():
            if key != 'source_sha256' and isinstance(child, (dict, list)):
                yield from _provenance_records(child, f'{location}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from _provenance_records(child, f'{location}[{index}]')


def create_archive(output, reports, *, allow_dirty=False):
    root = Path(_command(['git', 'rev-parse', '--show-toplevel'], Path.cwd()))
    status = _command(['git', 'status', '--porcelain'], root)
    if status and not allow_dirty:
        raise ValueError('Source worktree is dirty; commit the stage or explicitly use --allow-dirty')
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = {}
    source_hashes = {}
    for name in filter(None, paths):
        path = root / name
        if path.is_symlink():
            raise ValueError(f'Archive requires regular tracked source files: {name}')
        if not path.exists():
            continue  # A deleted tracked file is represented by the dirty diff.
        data = path.read_bytes()
        files['source/' + name] = data
        source_hashes[name] = _digest(data)
    evidence = []
    for index, report_path in enumerate(reports):
        path = Path(report_path)
        data = path.read_bytes()
        report = json.loads(data)
        if not isinstance(report, dict):
            raise ValueError(f'Evidence must be a JSON report object: {path}')
        records = []
        for location, declared in _provenance_records(report):
            comparison = {}
            for name, expected in declared.items():
                name = str(name)
                relative = name
                if Path(name).is_absolute():
                    try:
                        relative = str(Path(name).relative_to(root))
                    except ValueError:
                        comparison[name] = 'outside_source_tree'
                        continue
                observed = source_hashes.get(relative)
                comparison[name] = 'match' if observed == expected else 'missing' if observed is None else 'mismatch'
            records.append({'location': location, 'source_checks': comparison,
                            'declared_files_match': bool(comparison) and all(v == 'match' for v in comparison.values())})
        archived_name = f'evidence/{index:03d}-{path.name}'
        files[archived_name] = data
        evidence.append({'file': archived_name, 'provenance_records': records,
                         'declared_files_match': bool(records) and all(r['declared_files_match'] for r in records),
                         'source_coverage': 'declared files and segments only; unrecorded rows remain unverified' if records else 'not recorded by report',
                         'physical_acceptance': 'requires independent review of scope, errors and budgets'})
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()
                      if d.metadata.get('Name'))
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': packages,
                   'execution_environment': {key: os.environ.get(key) for key in
                     ('CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'MAX_JOBS')}}
    try:
        environment['gpu'] = _command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                      '--format=csv,noheader'], root)
    except (OSError, subprocess.CalledProcessError) as error:
        environment['gpu_probe_error'] = str(error)
    files['environment.json'] = (json.dumps(environment, indent=2) + '\n').encode()
    files['source.diff'] = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=root)
    manifest = {'schema_version': 1, 'source_revision': _command(['git', 'rev-parse', 'HEAD'], root),
                'source_dirty': bool(status), 'source_status': status, 'evidence': evidence,
                'complete_G0': False, 'gate_status': 'No gate certification is inferred by archiving',
                'files': {name: {'sha256': _digest(data), 'bytes': len(data)} for name, data in files.items()}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
    verify_archive(output)
    return manifest


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Archive contains duplicate entries')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported evidence archive schema')
        expected_names = set(manifest['files']) | {'manifest.json'}
        if set(names) != expected_names:
            raise ValueError('Archive entries do not match manifest')
        for name, expected in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'Unsafe archive path: {name}')
            data = archive.read(name)
            if len(data) != expected['bytes'] or _digest(data) != expected['sha256']:
                raise ValueError(f'Archive integrity check failed: {name}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    create = sub.add_parser('create')
    create.add_argument('--report', type=Path, action='append', required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--allow-dirty', action='store_true')
    verify = sub.add_parser('verify')
    verify.add_argument('archive', type=Path)
    args = parser.parse_args(argv)
    manifest = (create_archive(args.output, args.report, allow_dirty=args.allow_dirty)
                if args.action == 'create' else verify_archive(args.archive))
    print(json.dumps({'integrity_verified': True, 'source_revision': manifest['source_revision'],
                      'source_dirty': manifest['source_dirty'], 'evidence': manifest['evidence']}, indent=2))


if __name__ == '__main__':
    main()
