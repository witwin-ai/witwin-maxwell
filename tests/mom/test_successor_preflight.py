"""Successor collection must preserve evidence and reject source drift."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[2] / 'benchmark/sie_successor_preflight.py'
SPEC = importlib.util.spec_from_file_location('successor_preflight', PATH)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_inventory_detects_added_removed_and_modified_inputs(tmp_path):
    package = tmp_path / 'witwin'
    package.mkdir()
    source = package / 'solver.py'
    source.write_text('value = 1\n')
    baseline = preflight.inventory(tmp_path, tmp_path / 'core')
    source.write_text('value = 2\n')
    assert preflight.inventory(tmp_path, tmp_path / 'core') != baseline
    source.write_text('value = 1\n')
    header = package / 'kernel.cuh'
    header.write_text('// new native input\n')
    assert set(preflight.inventory(tmp_path, tmp_path / 'core')) - set(baseline) == {str(header)}
    header.unlink()
    assert preflight.inventory(tmp_path, tmp_path / 'core') == baseline
    source.unlink()
    assert not preflight.inventory(tmp_path, tmp_path / 'core')


def test_historical_helper_mutation_rejected_before_execution(tmp_path):
    base = tmp_path / 'benchmark'
    base.mkdir()
    (base / 'sie_single_buffer_equivalence_v3.py').write_text(
        'raise RuntimeError("untrusted input executed")\n')
    with pytest.raises(ValueError, match='Historical binding helper changed'):
        preflight.load_helper(tmp_path, 'sie_single_buffer_equivalence_v3.py')


def test_reports_are_exclusive(tmp_path):
    target = tmp_path / 'execution.json'
    preflight.write(target, {'engineering_acceptance_passed': False})
    with pytest.raises(FileExistsError):
        preflight.write(target, {'engineering_acceptance_passed': True})
    assert json.loads(target.read_text())['engineering_acceptance_passed'] is False


def test_foreign_candidate_rejected_before_creating_output(tmp_path):
    target = tmp_path / 'output'
    with pytest.raises(ValueError, match='selected candidate harness'):
        preflight.main(['--candidate', str(tmp_path), '--output', str(target)])
    assert not target.exists()


def test_unarchived_provider_module_is_rejected(monkeypatch):
    monkeypatch.setattr(preflight, 'sys', SimpleNamespace(
        modules={'witwin.maxwell.injected': object()}))
    bound = SimpleNamespace(modules={}, verify=lambda: {})
    with pytest.raises(ValueError, match='Unbound or missing provider'):
        preflight.verify_provider_bindings(bound)


def test_root_conftest_addition_changes_inventory(tmp_path):
    before = preflight.inventory(tmp_path, tmp_path / 'core')
    config = tmp_path / 'conftest.py'
    config.write_text('def pytest_collection_modifyitems(items): items.clear()\n')
    assert set(preflight.inventory(tmp_path, tmp_path / 'core')) - set(before) == {str(config)}


def test_provider_and_benchmark_bindings_can_coexist(monkeypatch):
    modules = {'witwin.maxwell.mom': object(), 'benchmark.example': object()}
    monkeypatch.setattr(preflight, 'sys', SimpleNamespace(modules=modules))
    bound = SimpleNamespace(modules=modules, verify=lambda: {'verified': True})
    assert preflight.verify_provider_bindings(bound) == {'verified': True, 'namespaces': {}}


def test_namespace_path_replacement_is_rejected(monkeypatch):
    module = SimpleNamespace(__path__=['/foreign/mom'])
    monkeypatch.setattr(preflight, 'sys', SimpleNamespace(modules={'witwin.maxwell.mom': module}))
    with pytest.raises(ValueError, match='namespace replaced'):
        preflight.verify_provider_bindings(SimpleNamespace(modules={}),
                                           {'witwin.maxwell.mom': (module, ('/reviewed/mom',))})


@pytest.mark.parametrize('change', ['head', 'scope', 'source', 'build', 'environment', 'nodes', 'required'])
def test_run_rejects_catalog_drift(change):
    before, build = {'solver.py': {'sha256': 'abc', 'bytes': 4}}, {'version': 'reviewed'}
    catalog = dict(status='collected_not_qualified', protocol=preflight.PROTOCOL,
                   scope=preflight.SCOPE, head='reviewed', torch_build=build,
                   build_environment=preflight.build_environment(),
                   source_inventory_sha256=preflight.digest(json.dumps(before, sort_keys=True).encode()),
                   ordered_nodeids=['tests/mom/test.py::test_cuda'],
                   host_validation_nodeids=[],
                   required_cuda_nodeids=['tests/mom/test.py::test_cuda'])
    preflight.validate_catalog(catalog, before, 'reviewed', build)
    if change == 'head':
        catalog['head'] = 'another'
    elif change == 'scope':
        catalog['scope'] = ['tests/mom']
    elif change == 'source':
        catalog['source_inventory_sha256'] = 'unreviewed'
    elif change == 'build':
        catalog['torch_build'] = {'version': 'another'}
    elif change == 'environment':
        catalog['build_environment'] = {'CUDA_HOME': '/unreviewed'}
    elif change == 'nodes':
        catalog['ordered_nodeids'] *= 2
    else:
        catalog['required_cuda_nodeids'] = []
    with pytest.raises(ValueError):
        preflight.validate_catalog(catalog, before, 'reviewed', build)


def test_environment_requirement_is_not_an_allocation_requirement():
    gpu = SimpleNamespace(kwargs={'reason': 'CUDA required'})
    item = SimpleNamespace(get_closest_marker=lambda name: None, iter_markers=lambda name: [gpu])
    assert preflight.execution_role(item) == 'cuda_allocation'
    item.get_closest_marker = lambda name: object() if name == 'host_validation' else None
    assert preflight.execution_role(item) == 'host_validation'


@pytest.mark.parametrize('count', [None, 1, -1])
def test_host_validation_cannot_hide_cuda_execution(count):
    row = {'nodeid': 'guard', 'cuda': {'0': {'allocation_events': count}, '1': {'allocation_events': 0}}}
    assert preflight.host_validation_errors([row], ['guard'])
    row['cuda']['0']['allocation_events'] = 0
    assert preflight.host_validation_errors([row], ['guard']) == []
    assert preflight.host_validation_errors([], ['guard'])
