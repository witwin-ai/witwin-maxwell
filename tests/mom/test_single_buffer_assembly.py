"""Host validation contracts; numerical equivalence uses the explicit driver."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[2] / 'benchmark/sie_single_buffer_equivalence_v3.py'
# Local checkout of the compact-Riesz provider baseline.
COMPACT_RIESZ_BASELINE = Path(os.environ.get('WITWIN_SIE_COMPACT_RIESZ_BASELINE', Path(tempfile.gettempdir())/'witwin-maxwell-compact-riesz'))
SPEC = importlib.util.spec_from_file_location('single_buffer_equivalence_driver', PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def metadata():
    return {'triangle_pair_counts': [20, 60, 120, 300, 5900],
            'assembly_seconds': 1., 'dtype': 'torch.complex128',
            'quadrature_error_estimate': {'near_disjoint': {'maximum_order': 8}},
            'matrix_bytes': 230400}


def test_protocol_freezes_both_precisions_three_shapes_and_multiple_batches():
    p = driver.PROTOCOL
    assert p['shapes'] == ['sphere', 'ellipsoid', 'harmonic']
    assert p['dtypes'] == ['complex64', 'complex128']
    assert len(p['batches']) == 2 and all(0 < n < 80 for b in p['batches'] for n in b)
    assert p['rhs_count'] == 8 and p['subdivisions'] == 1
    assert p['tolerances']['complex128']['comparison'] < 1e-10


def test_measured_timing_difference_is_not_scientific_difference():
    old, new = metadata(), metadata()
    new['assembly_seconds'] = 14.
    assert driver.validate_metadata(old, new) == old['triangle_pair_counts']


@pytest.mark.parametrize('field,value', [('dtype', 'torch.complex64'),
    ('quadrature_error_estimate', {'near_disjoint': {'maximum_order': 16}}),
    ('matrix_bytes', 1), ('triangle_pair_counts', [20, 60, 120, 299, 5901])])
def test_metadata_mutation_rejected(field, value):
    old, new = metadata(), metadata()
    new[field] = value
    with pytest.raises(ValueError, match='metadata differs'):
        driver.validate_metadata(old, new)


@pytest.mark.parametrize('counts', [[20, 60, 120, 0, 6200], [20, 60, 120, 300],
    [20, 60, 120, True, 5900], [20, 60, 120, 300., 5900]])
def test_missing_near_execution_not_equivalence(counts):
    old, new = metadata(), metadata()
    old['triangle_pair_counts'] = new['triangle_pair_counts'] = counts
    with pytest.raises(ValueError, match='category'):
        driver.validate_metadata(old, new)


def test_binding_restored_after_native_failure():
    old, new = object(), object()
    runtime = SimpleNamespace(assemble_weak=old)
    with pytest.raises(RuntimeError):
        with driver.assembly_binding(runtime, new):
            assert runtime.assemble_weak is new
            raise RuntimeError('injected native failure')
    assert runtime.assemble_weak is old


def test_source_inventory_detects_add_remove_and_mutation(tmp_path):
    source = tmp_path/'operator.py'
    source.write_text('x = 1\n')
    original = driver.source_map(tmp_path)
    source.write_text('x = 2\n')
    assert driver.source_map(tmp_path) != original
    source.write_text('x = 1\n')
    header = tmp_path/'kernel.cuh'
    header.write_text('native header\n')
    assert set(driver.source_map(tmp_path))-set(original) == {'kernel.cuh'}
    header.unlink()
    assert driver.source_map(tmp_path) == original
    source.unlink()
    assert driver.source_map(tmp_path) == {}


def test_preloaded_provider_cannot_be_relabelled(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'sys', SimpleNamespace(modules={'witwin.core.identity': object()}))
    with pytest.raises(ValueError, match='already imported'):
        driver.BoundSources([], tmp_path/'archive')


@pytest.mark.parametrize('change', ['source', 'archive', 'module'])
def test_bound_archive_and_live_module_mutations_fail(tmp_path, monkeypatch, change):
    module = object()
    monkeypatch.setattr(driver, 'sys', SimpleNamespace(modules={'bound.fake': module}))
    source = tmp_path/'bound.py'
    source.write_text('answer = 17\n')
    bound = driver.BoundSources([source], tmp_path/'archive')
    bound.modules['bound.fake'] = (module, str(source))
    before = bound.verify()
    if change == 'source':
        source.write_text('answer = 18\n')
    elif change == 'archive':
        Path(before['files'][str(source)]['archive']).write_text('answer = 18\n')
    else:
        driver.sys.modules['bound.fake'] = object()
    with pytest.raises(ValueError, match='changed|replaced'):
        bound.verify()


def test_baseline_wrong_bytes_rejected_before_import(tmp_path):
    path = tmp_path/'operator.py'
    path.write_text('raise RuntimeError("untrusted code executed")\n')
    with pytest.raises(ValueError, match='baseline operator hash'):
        driver.load_baseline(path)


def test_reports_are_exclusive_and_nonfinite_values_rejected(tmp_path):
    path = tmp_path/'report.json'
    driver.write_json(path, {'passed': False})
    with pytest.raises(FileExistsError):
        driver.write_json(path, {'passed': True})
    with pytest.raises(ValueError):
        driver.write_json(tmp_path/'nonfinite.json', {'error': float('nan')})
    assert json.loads(path.read_text()) == {'passed': False}


def test_actual_fresh_entrypoint_preflight_no_cuda(tmp_path):
    # Current source gets a new epoch; historical exact-e95 remains immutable.
    candidate = PATH.parents[1]
    successor = candidate/'benchmark/sie_successor_preflight.py'
    result = subprocess.run([sys.executable, str(successor),
        '--candidate', str(candidate), '--output', str(tmp_path/'preflight')],
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path/'preflight/execution.json').read_text())
    assert report['status'] == 'collected_not_qualified'
    assert report['engineering_acceptance_passed'] is False
    assert report['physical_acceptance_passed'] is False
    assert report['ordered_nodeids'] and not report['collection_errors']
    before = json.loads((tmp_path/'preflight/source-before.json').read_text())
    after = json.loads((tmp_path/'preflight/source-after.json').read_text())
    assert before == after
    bound = json.loads((tmp_path/'preflight/provider-bindings.json').read_text())
    for name in ('operator', 'linalg', 'mesh'):
        path = candidate/f'witwin/maxwell/mom/{name}.py'
        module = bound['loaded_modules'][f'witwin.maxwell.mom.{name}']
        assert Path(module['origin']) == path
        assert module['sha256'] == driver.digest(path.read_bytes())
        assert module['sha256'] == before[str(path)]['sha256']
    assert 'witwin.core' in bound['loaded_modules']


@pytest.fixture(scope='module')
def historical_driver(tmp_path_factory):
    """Replay unchanged V3 in a local immutable assembly-epoch checkout."""
    root = tmp_path_factory.mktemp('historical-assembly')/'provider'
    commit = '4ba0bb5bde0e8f2c2d01d461135f6ecaeb5b0f9c'
    subprocess.run(['git','clone','-q','--shared','--no-checkout',str(PATH.parents[1]),str(root)],check=True)
    subprocess.run(['git','-C',str(root),'checkout','-q','--detach',commit],check=True)
    head = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert head == commit
    actual = root/'benchmark/sie_single_buffer_equivalence_v3.py'
    assert driver.digest(actual.read_bytes()) == 'f996a2d65ecddec6d25c42c1ea195fa8bc14dcbab6e3dfbd4711ad2675b69725'
    pair = driver.verify_provider_pair(root,COMPACT_RIESZ_BASELINE)
    assert pair['heads']['candidate']==commit
    yield actual
    assert driver.verify_provider_pair(root,COMPACT_RIESZ_BASELINE)==pair


@pytest.mark.parametrize('mode', ['preflight', 'synthetic_numerical_summary'])
@pytest.mark.parametrize('phase', ['provider', 'core', 'archive'])
def test_final_source_failure_after_successful_work_never_passes(tmp_path, mode, phase, historical_driver):
    """Real source preflight; numerical-summary rows are explicit host doubles."""
    out = tmp_path/'injected-final-failure'
    script = r'''
import importlib.util, json, pathlib, sys
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('tested_driver', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
path, output, mode, phase = map(str, sys.argv[1:])
root = pathlib.Path(path).parents[1]
real_pair = m.verify_provider_pair
pair_calls = 0
def pair(*args):
    global pair_calls
    pair_calls += 1
    if pair_calls == 2 and phase == 'provider':
        raise ValueError('injected final provider drift')
    return real_pair(*args)
m.verify_provider_pair = pair
real_map = m.source_map
def source_map(root):
    result = real_map(root)
    if pair_calls == 2 and phase == 'core' and str(root).endswith('/witwin/core'):
        return {**result, 'injected.py': {'sha256': '0'*64, 'bytes': 0}}
    return result
m.source_map = source_map
real_bound = m.BoundSources.verify
def verify(self):
    if pair_calls == 2 and phase == 'archive':
        raise ValueError('injected final source archive drift')
    return real_bound(self)
m.BoundSources.verify = verify
args = ['--candidate', str(root), '--baseline', @COMPACT_RIESZ_BASELINE@, '--output', output]
if mode == 'preflight':
    args += ['--preflight']
else:
    import torch
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda: 'HOST TEST DOUBLE, NO CUDA EXECUTION'
    torch.cuda.get_device_properties = lambda n: SimpleNamespace(total_memory=1)
    def fake_case(shape, dtype, batch, baseline, target, *, progress):
        result = {'passed': True, 'scope': 'host negative test double; not numerical evidence'}
        m.write_json(target/'result.json', result)
        return result
    m.run_case = fake_case
code = m.main(args)
import torch
print(json.dumps({'return_code': code, 'cuda_initialized': torch.cuda.is_initialized()}))
sys.exit(code)
'''.replace('@COMPACT_RIESZ_BASELINE@', repr(str(COMPACT_RIESZ_BASELINE)))
    child = subprocess.run([sys.executable, '-c', script, str(historical_driver), str(out), mode, phase],
                           text=True, capture_output=True, timeout=90)
    assert child.returncode == 1, child.stdout + child.stderr
    report = json.loads((out/'execution.json').read_text())
    assert report['status'] == 'failed' and 'failure' in report
    assert report['source_verification']['after'] == 'not_completed'
    assert not (out/'source-after.json').exists()
    assert len(report['cases']) == (0 if mode == 'preflight' else 12)
    assert json.loads(child.stdout.splitlines()[-1])['cuda_initialized'] is False


def test_actual_entrypoint_retains_failed_case_partial_cost(tmp_path, historical_driver):
    out = tmp_path/'injected-case-failure'
    script = r'''
import importlib.util, pathlib, sys
from types import SimpleNamespace
import torch
spec = importlib.util.spec_from_file_location('tested_driver', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.cuda.is_available = lambda: True
torch.cuda.get_device_name = lambda: 'HOST FAILURE DOUBLE'
torch.cuda.get_device_properties = lambda n: SimpleNamespace(total_memory=1)
def fail(shape, dtype, batch, baseline, target, *, progress):
    progress['phase'] = 'new_public_prepare'
    progress['allocation']['old'] = {'status': 'succeeded', 'assembly_wall_seconds': .001}
    progress['allocation']['new'] = {'status': 'failed', 'assembly_wall_seconds': .002, 'allocated_before': 123}
    raise RuntimeError('injected assembly failure after old measured phase')
m.run_case = fail
root = pathlib.Path(sys.argv[1]).parents[1]
sys.exit(m.main(['--candidate', str(root), '--baseline', @COMPACT_RIESZ_BASELINE@, '--output', sys.argv[2]]))
'''.replace('@COMPACT_RIESZ_BASELINE@', repr(str(COMPACT_RIESZ_BASELINE)))
    child = subprocess.run([sys.executable, '-c', script, str(historical_driver), str(out)], text=True, capture_output=True, timeout=90)
    assert child.returncode == 1, child.stdout+child.stderr
    report = json.loads((out/'execution.json').read_text())
    assert report['status'] == 'failed' and len(report['cases']) == 1
    row = report['cases'][0]
    attempt = out/row['id']/'attempt.json'
    assert driver.digest(attempt.read_bytes()) == row['attempt_sha256']
    attempt = json.loads(attempt.read_text())
    assert attempt['status'] == 'failed' and attempt['phase'] == 'new_public_prepare'
    assert attempt['case_wall_seconds'] > 0
    assert attempt['allocation']['old']['assembly_wall_seconds'] == .001
    assert attempt['allocation']['new']['allocated_before'] == 123
    assert report['source_verification']['after'] == 'not_completed'


@pytest.mark.parametrize('missing_after', [False, True])
def test_assembly_observer_retains_exception_measurements(monkeypatch, missing_after):
    import torch
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    calls = 0
    def allocated():
        nonlocal calls
        calls += 1
        if missing_after and calls == 2:
            raise RuntimeError('injected unavailable CUDA observation')
        return 456
    monkeypatch.setattr(torch.cuda, 'memory_allocated', allocated)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 789)
    def failure(*args):
        raise RuntimeError('injected native assembly exception')
    measurements = {}
    observed = driver.observe_assembly(failure, measurements)
    with pytest.raises(RuntimeError, match='native assembly exception'):
        observed(SimpleNamespace(n_edges=120), None, None, None)
    assert measurements['status'] == 'failed'
    assert measurements['assembly_wall_seconds'] > 0
    assert measurements['allocated_before'] == 456
    assert measurements['peak_allocated'] == 789
    assert measurements['allocated_after'] == (None if missing_after else 456)
    assert bool(measurements['measurement_errors']) == missing_after
    assert measurements['failure']['message'] == 'injected native assembly exception'
    assert not driver.allocation_measurement_complete(measurements)


@pytest.mark.parametrize('change', [{'allocated_after': None}, {'measurement_errors': ['unavailable']},
    {'peak_allocated': 1}, {'status': 'failed'}, {'assembly_wall_seconds': float('nan')}])
def test_incomplete_memory_measurement_never_passes(change):
    row = {'status': 'succeeded', 'allocated_before': 10, 'allocated_after': 20,
           'peak_allocated': 30, 'assembly_wall_seconds': .01, 'measurement_errors': []}
    assert driver.allocation_measurement_complete(row)
    row.update(change)
    assert not driver.allocation_measurement_complete(row)


@pytest.fixture
def provider_pair(tmp_path, monkeypatch):
    """Small real Git histories exercise revision eligibility without numerics."""
    baseline = tmp_path/'baseline'
    candidate = tmp_path/'candidate'
    baseline.mkdir()
    def git(root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git(baseline, 'init', '-q')
    git(baseline, 'config', 'user.name', 'Xingyu Chen')
    git(baseline, 'config', 'user.email', 'fixture@example.invalid')
    folder = baseline/'witwin/maxwell/sie'
    folder.mkdir(parents=True)
    (folder/'operator.py').write_text('reviewed_old = 1\n')
    (folder/'linalg.py').write_text('unchanged = 1\n')
    git(baseline, 'add', 'witwin')
    git(baseline, 'commit', '-qm', 'Create test source history')
    base = git(baseline, 'rev-parse', 'HEAD')
    subprocess.run(['git', 'clone', '-q', str(baseline), str(candidate)], check=True)
    git(candidate, 'config', 'user.name', 'Xingyu Chen')
    git(candidate, 'config', 'user.email', 'fixture@example.invalid')
    (candidate/'witwin/maxwell/sie/operator.py').write_text('reviewed_new = 1\n')
    monkeypatch.setattr(driver, 'BASE_COMMIT', base)
    monkeypatch.setattr(driver, 'BASE_OPERATOR_SHA256', driver.digest((folder/'operator.py').read_bytes()))
    monkeypatch.setattr(driver, 'CANDIDATE_OPERATOR_SHA256', driver.digest((candidate/'witwin/maxwell/sie/operator.py').read_bytes()))
    return candidate, baseline, git


def test_exact_scientific_candidate_remains_replayable_after_commit(provider_pair):
    candidate, baseline, git = provider_pair
    before = driver.verify_provider_pair(candidate, baseline)
    git(candidate, 'add', 'witwin')
    git(candidate, 'commit', '-qm', 'Commit reviewed test change')
    after = driver.verify_provider_pair(candidate, baseline)
    assert before['candidate'] == after['candidate']
    assert before['baseline'] == after['baseline']
    assert before['heads']['candidate'] != after['heads']['candidate']
    assert after['heads']['baseline'] == driver.BASE_COMMIT


@pytest.mark.parametrize('mutation', ['shared_dependency', 'candidate_operator', 'candidate_added_source', 'baseline_head', 'unrelated_candidate'])
def test_revision_eligibility_never_relaxes_scientific_bytes(provider_pair, mutation):
    candidate, baseline, git = provider_pair
    if mutation == 'shared_dependency':
        for root in (baseline, candidate):
            (root/'witwin/maxwell/sie/linalg.py').write_text('unreviewed = 2\n')
    elif mutation == 'candidate_operator':
        (candidate/'witwin/maxwell/sie/operator.py').write_text('wrong = 3\n')
    elif mutation == 'candidate_added_source':
        (candidate/'witwin/maxwell/sie/unreviewed.cu').write_text('unreviewed\n')
    elif mutation == 'baseline_head':
        git(baseline, 'commit', '--allow-empty', '-qm', 'Change baseline revision')
    else:
        git(candidate, 'checkout', '--orphan', 'unrelated')
        git(candidate, 'add', 'witwin')
        git(candidate, 'commit', '-qm', 'Create unrelated test history')
    with pytest.raises(ValueError):
        driver.verify_provider_pair(candidate, baseline)
