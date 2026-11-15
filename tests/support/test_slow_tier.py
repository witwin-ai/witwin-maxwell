"""Slow/perf tier selection, the slow-tier budget guard and the slow list itself."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tests.support.slow_tier import listed, read_slow_list

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))}

SAMPLE = '''
import time
import pytest

def test_fast():
    pass

@pytest.mark.slow
def test_slow():
    time.sleep(0.3)

@pytest.mark.perf
def test_perf():
    pass

def test_unmarked_sleeper():
    time.sleep(0.3)
'''


def run(directory, *args):
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-rA", *args],
        cwd=directory, capture_output=True, text=True, env=ENV)
    outcomes = {name: outcome for outcome, name in re.findall(r"^(PASSED|FAILED) \S+::(\w+)", child.stdout, re.MULTILINE)}
    return child, outcomes


@pytest.fixture
def sample(tmp_path):
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["tests.support.slow_tier"]\n')
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "test_sample.py").write_text(SAMPLE)
    return tmp_path


def test_directory_runs_deselect_slow_and_perf(sample):
    _, outcomes = run(sample, "pkg")
    assert outcomes == {"test_fast": "PASSED", "test_unmarked_sleeper": "PASSED"}
    _, outcomes = run(sample, "pkg", "--run-slow")
    assert set(outcomes) == {"test_fast", "test_slow", "test_unmarked_sleeper"}
    _, outcomes = run(sample, "pkg", "-m", "slow")
    assert set(outcomes) == {"test_slow"}
    _, outcomes = run(sample, "pkg", "--run-slow", "--run-perf")
    assert set(outcomes) == {"test_fast", "test_slow", "test_perf", "test_unmarked_sleeper"}


def test_named_files_keep_slow_but_not_perf(sample):
    _, outcomes = run(sample, "pkg/test_sample.py")
    assert set(outcomes) == {"test_fast", "test_slow", "test_unmarked_sleeper"}
    _, outcomes = run(sample, "pkg/test_sample.py::test_perf")
    assert outcomes == {}


MILESTONE_SAMPLE = '''
import pytest

def test_small():
    pass

@pytest.mark.mom_milestone
def test_large():
    pass

@pytest.mark.mom_milestone
@pytest.mark.parametrize("n", [1, 2])
def test_large_param(n):
    pass
'''


def test_mom_milestone_needs_the_flag_the_marker_expression_or_the_named_node(sample):
    (sample / "pkg" / "test_milestone.py").write_text(MILESTONE_SAMPLE)
    _, outcomes = run(sample, "pkg/test_milestone.py", "--run-slow")
    assert set(outcomes) == {"test_small"}
    _, outcomes = run(sample, "pkg/test_milestone.py", "--run-mom-milestone")
    assert set(outcomes) == {"test_small", "test_large", "test_large_param"}
    _, outcomes = run(sample, "pkg", "-m", "mom_milestone")
    assert set(outcomes) == {"test_large", "test_large_param"}
    _, outcomes = run(sample, "pkg/test_milestone.py::test_large", "pkg/test_milestone.py::test_large_param")
    assert set(outcomes) == {"test_large", "test_large_param"}
    _, outcomes = run(sample / "pkg", "test_milestone.py::test_large_param[2]")
    assert set(outcomes) == {"test_large_param"}


def test_budget_fails_only_unmarked_tests_over_budget(sample):
    child, outcomes = run(sample, "pkg", "--run-slow", "--slow-budget", "0.2")
    assert outcomes == {"test_fast": "PASSED", "test_slow": "PASSED", "test_unmarked_sleeper": "FAILED"}
    assert "slow-tier budget" in child.stdout and child.returncode == 1


def test_slow_list_is_sorted_unique_and_names_existing_files():
    entries = read_slow_list()
    assert entries == sorted(set(entries))
    assert not [entry for entry in entries if not (ROOT / entry.split("::")[0]).is_file()]
    assert listed("a.py::t[1]", ["a.py::t"]) and listed("a.py::t", ["a.py"]) and not listed("a.py::t2", ["a.py::t"])


def test_every_slow_entry_still_collects():
    """Renamed or removed tests must leave the list; files that cannot be collected here are not judged."""
    entries = read_slow_list()
    files = sorted({entry.split("::")[0] for entry in entries})
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "--collect-only", "-q",
         "--run-slow", "--run-mom-milestone", "--continue-on-collection-errors", *files],
        cwd=ROOT, capture_output=True, text=True, env=ENV)
    nodes = [line for line in child.stdout.splitlines() if "::" in line]
    broken = set(re.findall(r"^ERROR (\S+\.py)", child.stdout, re.MULTILINE))
    stale = [entry for entry in entries if entry.split("::")[0] not in broken and not
             any(listed(node, [entry]) for node in nodes)]
    assert nodes and not stale, stale
