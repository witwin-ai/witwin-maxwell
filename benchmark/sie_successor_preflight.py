"""Source-bound collection and dual-CUDA regression for the successor SIE epoch."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCOPE = [
    'tests/mom',
    'tests/api/public',
    'tests/sources/definitions',
    'tests/fdtd/test_batch_result_scene.py',
    'tests/fdtd/multi_gpu/test_result_persistence.py',
]
HELPERS = {
    'sie_single_buffer_equivalence_v3.py':
        'f996a2d65ecddec6d25c42c1ea195fa8bc14dcbab6e3dfbd4711ad2675b69725',
    'sie_combined_full_acceptance_v2.py':
        '4b686988b2850645c2f7901946dc438f3427a8227fe4bb2248e5b4ae7b5f578a',
}
EVIDENCE_SKIP = 'repository evidence not in this checkout: '
BUILD_ENVIRONMENT = ('CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'CC', 'CXX',
                     'MAX_JOBS', 'TORCH_CUDA_ARCH_LIST', 'WITWIN_CUDA_GENCODE_ARCHES',
                     'WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR',
                     'WITWIN_MAXWELL_FDTD_CUDA_PREBUILT',
                     'WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT')
PROTOCOL = 'gsp-successor-collection-v2'


def build_environment():
    return {name: os.environ.get(name) for name in BUILD_ENVIRONMENT}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def inventory(root, core):
    """Include all provider/tests/harness inputs, even untracked additions."""
    paths = set()
    for base in (root / 'witwin', root / 'tests', root / 'benchmark', core):
        paths.update(p.resolve() for p in base.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts
                     and p.suffix in {'.py', '.cu', '.cpp', '.cuh', '.h', '.so'})
    for name in ('pyproject.toml', 'pytest.ini', 'setup.cfg', 'setup.py', 'tox.ini',
                 'conftest.py'):
        if (root / name).is_file():
            paths.add((root / name).resolve())
    return {str(p): {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
            for p in sorted(paths)}


def load_helper(root, name):
    path = root / 'benchmark' / name
    data = path.read_bytes()
    if digest(data) != HELPERS[name]:
        raise ValueError('Historical binding helper changed: ' + name)
    spec = importlib.util.spec_from_file_location('_successor_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def is_provider(name):
    return name == 'witwin.core' or name.startswith(('witwin.core.', 'witwin.maxwell'))


class ProviderNamespaces(importlib.abc.MetaPathFinder):
    """Bind provider namespace packages that have no Python source file."""
    def __init__(self, root, core):
        self.roots = {'witwin.maxwell': root / 'witwin/maxwell', 'witwin.core': core}
        self.modules = {}

    def find_spec(self, fullname, path=None, target=None):
        if not is_provider(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is not None:
            return None
        base = 'witwin.core' if fullname.startswith('witwin.core') else 'witwin.maxwell'
        expected = self.roots[base].joinpath(*fullname[len(base):].strip('.').split('.'))
        locations = tuple(str(Path(p).resolve()) for p in spec.submodule_search_locations or ())
        if locations != (str(expected.resolve()),):
            raise ValueError('Foreign provider namespace: ' + fullname)
        owner = self

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                owner.modules[fullname] = (module, locations)

        spec.loader = Loader()
        return spec


def verify_provider_bindings(bound, namespaces=None):
    namespaces = namespaces or {}
    for name, (module, locations) in namespaces.items():
        if sys.modules.get(name) is not module or tuple(module.__path__) != locations:
            raise ValueError('Provider namespace replaced: ' + name)

    actual = {name for name in sys.modules if is_provider(name)}
    expected = {name for name in bound.modules if is_provider(name)} | set(namespaces)
    if actual != expected:
        raise ValueError('Unbound or missing provider module: '
                         f'unbound={sorted(actual - expected)}, missing={sorted(expected - actual)}')
    return bound.verify() | {'namespaces': {n: list(paths) for n, (_, paths) in namespaces.items()}}


def validate_catalog(catalog, before, head, build):
    if (catalog.get('status') != 'collected_not_qualified'
            or catalog.get('protocol') != PROTOCOL
            or catalog.get('scope') != SCOPE or catalog.get('head') != head
            or catalog.get('source_inventory_sha256') != digest(json.dumps(before, sort_keys=True).encode())
            or catalog.get('torch_build') != build
            or catalog.get('build_environment') != build_environment()):
        raise ValueError('Reviewed catalog source, scope, HEAD or build differs')
    nodes = catalog.get('ordered_nodeids')
    if (not isinstance(nodes, list) or not nodes
            or any(not isinstance(node, str) or not node for node in nodes)
            or len(nodes) != len(set(nodes))):
        raise ValueError('Empty or duplicate reviewed node catalog')
    required = catalog.get('required_cuda_nodeids')
    if (not isinstance(required, list) or not required
            or any(node not in nodes for node in required) or len(required) != len(set(required))):
        raise ValueError('Invalid required CUDA catalog')
    host = catalog.get('host_validation_nodeids')
    if (not isinstance(host, list) or any(node not in nodes for node in host)
            or len(host) != len(set(host)) or set(host) & set(required)):
        raise ValueError('Invalid host-validation catalog')


def execution_role(item):
    if item.get_closest_marker('host_validation') is not None:
        return 'host_validation'
    if any('cuda' in str(mark.kwargs.get('reason', '')).lower()
           for mark in item.iter_markers('skipif')):
        return 'cuda_allocation'
    return 'general'


def host_validation_errors(per_test, host_nodes):
    by_node = {row['nodeid']: row for row in per_test}
    return ['Host validation allocated CUDA memory: ' + node for node in host_nodes
            if node not in by_node or not by_node[node].get('cuda')
            or any(row.get('allocation_events') != 0 for row in by_node[node]['cuda'].values())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('collect', 'run'), default='collect')
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--reviewed-catalog-sha256')
    args = parser.parse_args(argv)
    root, out = args.candidate.resolve(), args.output.resolve()
    if Path(__file__).resolve() != root / 'benchmark/sie_successor_preflight.py':
        raise ValueError('Execute the selected candidate harness')
    if out.is_relative_to(root):
        raise ValueError('Collection artifacts must be outside the candidate')
    out.mkdir(parents=True, exist_ok=False)
    report = {'protocol': PROTOCOL, 'status': 'failed',
              'engineering_acceptance_passed': False,
              'physical_acceptance_passed': False, 'scope': SCOPE,
              'mode': args.mode, 'python': sys.executable,
              'build_environment': build_environment(),
              'requires': 'independent physics and robustness qualification beyond regression'}
    code = 1
    collection = None
    started = time.time()
    try:
        if any(os.environ.get(name) for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS')):
            raise ValueError('Ambient pytest options/plugins are not allowed')
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        report['pytest_plugin_autoload'] = False
        sys.path.insert(0, str(root))
        os.chdir(root)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        core_spec = importlib.util.find_spec('witwin.core')
        if core_spec is None or core_spec.origin is None:
            raise ValueError('Installed core unavailable')
        core = Path(core_spec.origin).resolve().parent
        before = inventory(root, core)
        write(out / 'source-before.json', before)
        binding = load_helper(root, 'sie_single_buffer_equivalence_v3.py')
        historical = load_helper(root, 'sie_combined_full_acceptance_v2.py')
        bound = binding.BoundSources([Path(p) for p in before], out / 'source')
        archived = {p: {'sha256': row['sha256'], 'bytes': row['bytes']}
                    for p, row in bound.files.items()}
        if archived != before:
            raise ValueError('Source changed between inventory and archive')
        benchmarks = historical.BenchmarkSources(root, bound)
        namespaces = ProviderNamespaces(root, core)
        sys.meta_path[:0] = [benchmarks, namespaces, bound]
        import pytest
        import torch
        build = {'version': torch.__version__, 'cuda': torch.version.cuda,
                 'git': torch.version.git_version}
        devices, catalog = [], None
        if args.mode == 'run':
            catalog_bytes = args.catalog.read_bytes() if args.catalog is not None else b''
            if (not catalog_bytes or args.reviewed_catalog_sha256 is None
                    or digest(catalog_bytes) != args.reviewed_catalog_sha256):
                raise ValueError('An exact independently reviewed catalog is required')
            catalog = json.loads(catalog_bytes)
            validate_catalog(catalog, before, head, build)
            with (out / 'reviewed-catalog.json').open('xb') as stream:
                stream.write(catalog_bytes)
            report['reviewed_catalog_sha256'] = digest(catalog_bytes)
            if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
                raise ValueError('Successor regression requires two actual CUDA devices')
            devices = ['0', '1']
            report['devices'] = {d: {'name': torch.cuda.get_device_name(int(d)),
                                     'total_memory': torch.cuda.get_device_properties(int(d)).total_memory}
                                 for d in devices}

        class Collection:
            def __init__(self):
                self.nodes, self.errors, self.deselected = [], [], []
                self.evidence_skips = []
                self.phases, self.tests = [], []
                self.required_cuda = []
                self.host_validation = []
                self.initial_binding, self.initial_objects = None, {}

            def pytest_collection_finish(self, session):
                self.nodes = [item.nodeid for item in session.items]
                self.required_cuda = [item.nodeid for item in session.items
                                      if execution_role(item) == 'cuda_allocation']
                self.host_validation = [item.nodeid for item in session.items
                                        if execution_role(item) == 'host_validation']
                if catalog is not None and self.nodes != catalog['ordered_nodeids']:
                    pytest.exit('Executed collection differs from reviewed catalog', returncode=2)
                if catalog is not None and self.required_cuda != catalog['required_cuda_nodeids']:
                    pytest.exit('Required CUDA nodes differ from reviewed catalog', returncode=2)
                if catalog is not None and self.host_validation != catalog['host_validation_nodeids']:
                    pytest.exit('Host validation nodes differ from reviewed catalog', returncode=2)
                self.initial_binding = verify_provider_bindings(bound, namespaces.modules)
                self.initial_objects = {name: id(module) for name, (module, _) in bound.modules.items()}

            def pytest_collectreport(self, report):
                # A checkout without repository evidence skips those modules at collection.
                if report.outcome == 'skipped' and EVIDENCE_SKIP in str(report.longrepr):
                    self.evidence_skips.append({'nodeid': report.nodeid, 'detail': str(report.longrepr)})
                elif report.outcome != 'passed':
                    self.errors.append({'nodeid': report.nodeid,
                                        'outcome': report.outcome,
                                        'detail': str(report.longrepr)})

            def pytest_deselected(self, items):
                self.deselected.extend(item.nodeid for item in items)

            def pytest_runtest_logreport(self, report):
                self.phases.append({'nodeid': report.nodeid, 'when': report.when,
                                    'outcome': report.outcome, 'duration': report.duration,
                                    'wasxfail': getattr(report, 'wasxfail', None),
                                    'longrepr': str(report.longrepr) if report.failed or report.skipped else None})

            @pytest.hookimpl(hookwrapper=True)
            def pytest_runtest_protocol(self, item, nextitem):
                started_test = time.monotonic()
                row = {'nodeid': item.nodeid, 'cuda': {}, 'measurement_errors': []}
                counters = {}

                def observe(device, label, action):
                    try:
                        return action()
                    except Exception as error:
                        row['measurement_errors'].append({'device': device, 'phase': label,
                                                          'message': str(error)})
                        return None

                for d in devices:
                    i = int(d)
                    observe(d, 'synchronize_before', lambda: torch.cuda.synchronize(i))
                    observe(d, 'reset_peak', lambda: torch.cuda.reset_peak_memory_stats(i))
                    counters[d] = observe(d, 'counter_before', lambda: torch.cuda.memory_stats(i)['allocation.all.allocated'])
                    row['cuda'][d] = {'allocated_before': observe(d, 'allocated_before', lambda: torch.cuda.memory_allocated(i))}
                try:
                    yield
                finally:
                    for d in devices:
                        i = int(d)
                        observe(d, 'synchronize_after', lambda: torch.cuda.synchronize(i))
                        row['cuda'][d].update(
                            allocation_events=observe(d, 'allocation_events', lambda: torch.cuda.memory_stats(i)['allocation.all.allocated'] - counters[d]),
                            allocated_after=observe(d, 'allocated_after', lambda: torch.cuda.memory_allocated(i)),
                            peak_allocated_bytes=observe(d, 'peak_allocated', lambda: torch.cuda.max_memory_allocated(i)),
                            peak_reserved_bytes=observe(d, 'peak_reserved', lambda: torch.cuda.max_memory_reserved(i)))
                    row['seconds'] = time.monotonic() - started_test
                    self.tests.append(row)

        collection = Collection()
        command = SCOPE + ['--gpu', '-q', '-p', 'no:cacheprovider',
                           '--confcutdir=' + str(root)]
        command += ['--collect-only'] if args.mode == 'collect' else [f'--junitxml={out}/junit.xml']
        with (out / 'collection.log').open('x') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = pytest.main(command, plugins=[collection])
        after = inventory(root, core)
        libraries = {}
        native_archive = out / 'native-binaries'
        native_archive.mkdir()
        for name in sorted(torch.ops.loaded_libraries):
            path = Path(name).resolve()
            data = path.read_bytes()
            sha = digest(data)
            archived = native_archive / sha
            if not archived.exists():
                archived.write_bytes(data)
            row = {'sha256': sha, 'bytes': len(data), 'archive': str(archived)}
            ninja = path.parent / 'build.ninja'
            if ninja.is_file():
                text = ninja.read_bytes()
                ninja_archive = native_archive / digest(text)
                if not ninja_archive.exists():
                    ninja_archive.write_bytes(text)
                row['build_ninja_sha256'] = digest(text)
            libraries[str(path)] = row
        write(out / 'native-builds.json', libraries)
        write(out / 'source-after.json', after)
        write(out / 'provider-bindings.json', verify_provider_bindings(bound, namespaces.modules))
        write(out / 'benchmark-bindings.json', benchmarks.verify())
        write(out / 'collection-bindings.json', collection.initial_binding)
        report.update(head=head, core_origin=str(core), pytest_args=command,
                      pytest_exitstatus=int(result), ordered_nodeids=collection.nodes,
                      collection_errors=collection.errors, deselected=collection.deselected,
                      collection_evidence_skips=collection.evidence_skips,
                      torch_build=build,
                      cuda_available=torch.cuda.is_available(),
                      cuda_device_count=torch.cuda.device_count(),
                      source_inventory_sha256=digest(json.dumps(before, sort_keys=True).encode()))
        if before != after or head != subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True).strip():
            raise ValueError('Source or HEAD changed during collection')
        if ((result and args.mode == 'collect') or collection.errors or collection.deselected or not collection.nodes
                or len(collection.nodes) != len(set(collection.nodes))):
            raise ValueError('Incomplete or duplicate collection')
        if any(id(bound.modules[name][0]) != identity for name, identity in collection.initial_objects.items()):
            raise ValueError('Collection module object changed during execution')
        if args.mode == 'collect':
            if torch.cuda.is_initialized() or collection.phases or collection.tests:
                raise ValueError('Collection initialized CUDA or executed tests')
            report['status'] = 'collected_not_qualified'
        else:
            verdict = historical.grade(collection.nodes, collection.phases, collection.tests,
                                        int(result), True, devices, collection.required_cuda)
            verdict['errors'].extend(host_validation_errors(collection.tests, collection.host_validation))
            verdict['passed'] = not verdict['errors']
            report['verdict'] = verdict
            report['junit_verification'] = historical.verify_junit(out / 'junit.xml', collection.nodes)
            if not verdict['passed']:
                raise ValueError('Successor CUDA regression grading failed')
            report['engineering_acceptance_passed'] = True
            report['status'] = 'regression_passed_physics_not_qualified'
        code = 0
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if collection is not None:
            report.update(ordered_nodeids=collection.nodes, phases=collection.phases,
                          per_test=collection.tests, collection_errors=collection.errors,
                          deselected=collection.deselected, required_cuda_nodeids=collection.required_cuda,
                          host_validation_nodeids=collection.host_validation)
        report.update(started_unix=started, ended_unix=time.time())
        write(out / 'execution.json', report)
    print(json.dumps({'status': report['status'], 'tests': len(report.get('ordered_nodeids', [])),
                      'failure': report.get('failure')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
