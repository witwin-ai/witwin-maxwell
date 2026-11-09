"""Host adversarial controls for exact provider source and shard coverage."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'provider_epoch_v3_control', ROOT / 'benchmark/sie_combined_full_acceptance_v3.py')
epoch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(epoch)


def test_frozen_provider_rejects_mutated_and_added_runtime_source(tmp_path, monkeypatch):
    # Check out the frozen provider commit, so the control also holds on later revisions.
    frozen = tmp_path / 'frozen'
    subprocess.run(['git', 'clone', '-q', '--shared', '--no-checkout', str(ROOT), str(frozen)], check=True)
    subprocess.run(['git', '-C', str(frozen), 'sparse-checkout', 'set', 'witwin/maxwell'], check=True)
    subprocess.run(['git', '-C', str(frozen), 'checkout', '-q', '--detach', epoch.SOURCE_COMMIT], check=True)
    assert len(epoch.verify_source(frozen)['reviewed_scientific_deltas']) == 369
    shutil.copytree(frozen / 'witwin/maxwell', tmp_path / 'witwin/maxwell')
    original = epoch.commit_inventory
    monkeypatch.setattr(epoch, 'commit_inventory', lambda root, revision: original(frozen, revision))
    path = tmp_path / 'witwin/maxwell/sie/physics.py'
    data = path.read_bytes()
    path.write_bytes(data + b'\nUNREVIEWED_NUMERICAL_CHANGE = True\n')
    with pytest.raises(ValueError, match='Live provider package differs'):
        epoch.verify_source(tmp_path)
    path.write_bytes(data)
    (path.parent / 'unreviewed.py').write_text('UNREVIEWED = True\n')
    with pytest.raises(ValueError, match='Live provider package differs'):
        epoch.verify_source(tmp_path)


def test_aggregate_refuses_missing_and_duplicate_shard_evidence(tmp_path):
    node = 'host-control'
    phases = [dict(nodeid=node, when=phase, outcome='passed', duration=0.)
              for phase in ('setup', 'call', 'teardown')]
    observed = [dict(nodeid=node, seconds=.01, measurement_errors=[], cuda={'0': dict(
        allocation_events=0, allocated_before=0, allocated_after=0,
        peak_allocated_bytes=0, peak_reserved_bytes=0)})]
    assert epoch.grade([node], phases, observed, 0, True, ['0'])['passed']
    assert not epoch.grade([node], phases, observed, 0, True, ['0'], [node])['passed']
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'shards': [{'id': 0}, {'id': 1}]}))
    args = SimpleNamespace(catalog=catalog, reviewed_catalog_sha256=epoch.sha(catalog.read_bytes()),
                           shard_results=[], output=tmp_path / 'aggregate')
    with pytest.raises(ValueError, match='exact ordered shards'):
        epoch.aggregate(args)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'execution.json').write_text(json.dumps({'shard_id': 0}))
    args.shard_results = [run, run]
    with pytest.raises(ValueError, match='exact ordered shards'):
        epoch.aggregate(args)
    assert not args.output.exists()
