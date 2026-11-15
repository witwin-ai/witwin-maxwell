"""Pytest plugin for the slow and perf tiers.

``slow`` marks tests measured over 60 s on CPU or GPU. They come from an explicit
``@pytest.mark.slow`` or from a node id (or node-id prefix) listed in
``tests/slow_tests.txt``. A run whose arguments name only directories (the
default ``pytest`` or ``pytest tests``) deselects them; ``--run-slow`` or a ``-m``
expression that mentions ``slow`` keeps them. A test file or node named on the
command line always runs its slow tests, so explicit gate commands are unchanged.

``perf`` marks wall-clock budget tests that need an otherwise idle card. They are
deselected unless ``--run-perf`` is given or ``-m`` mentions ``perf``, even when
their file is named explicitly.

``mom_milestone`` marks the large MoM dense-oracle originals. Each has a regression-tier
version at a fraction of the cost with the same assertions; the original runs only in the
milestone tier (integration batch ``--tier milestone``, triggered by MoM source changes).
They are deselected unless ``--run-mom-milestone`` is given, ``-m`` mentions
``mom_milestone``, or the node itself (not just its file) is named on the command line.

``--slow-budget SECONDS`` turns a passing test that is not marked ``slow`` into a
failure once its setup plus call time exceeds SECONDS, so new slow tests are noticed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SLOW_LIST = Path(__file__).resolve().parents[1] / "slow_tests.txt"


def read_slow_list(path=SLOW_LIST):
    """Node ids or node-id prefixes, one per line; text after ``#`` is a comment."""
    entries = []
    for line in path.read_text().splitlines():
        entry = line.split(" #", 1)[0].strip()
        if entry and not entry.startswith("#"):
            entries.append(entry)
    return entries


def listed(nodeid, entries):
    return any(nodeid == entry or nodeid.startswith((entry + "::", entry + "[")) for entry in entries)


def pytest_addoption(parser):
    group = parser.getgroup("tiers")
    group.addoption("--run-slow", action="store_true", default=False,
                    help="include tests marked slow (deselected from directory-level runs by default)")
    group.addoption("--run-perf", action="store_true", default=False,
                    help="include wall-clock perf tests; run them only on an idle device")
    group.addoption("--run-mom-milestone", action="store_true", default=False,
                    help="include the large MoM dense-oracle originals (milestone tier)")
    group.addoption("--slow-budget", type=float, default=None, metavar="SECONDS",
                    help="fail any passing test not marked slow whose setup+call time exceeds SECONDS")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: measured over 60 s; deselected from directory-level runs unless --run-slow")
    config.addinivalue_line("markers", "perf: wall-clock budget on an idle device; deselected unless --run-perf")
    config.addinivalue_line("markers", "mom_milestone: large MoM dense-oracle original; deselected unless "
                                       "--run-mom-milestone, -m mom_milestone or the node is named")


def _named_files(config):
    root = config.invocation_params.dir
    files = set()
    for arg in config.args:
        path = (root / arg.split("::", 1)[0]).resolve()
        if path.is_file():
            files.add(path)
    return files


def _named_nodes(config):
    """Node ids named on the command line, rewritten relative to the rootdir like ``item.nodeid``."""
    root = config.invocation_params.dir
    nodes = set()
    for arg in config.args:
        if "::" in arg:
            path, rest = arg.split("::", 1)
            relative = (root / path).resolve().relative_to(config.rootpath.resolve()).as_posix()
            nodes.add(relative + "::" + rest)
    return nodes


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # tryfirst: markers must exist before -m deselection runs.
    entries = read_slow_list()
    for item in items:
        if listed(item.nodeid, entries):
            item.add_marker(pytest.mark.slow)
    expression = config.option.markexpr or ""
    keep_slow = config.getoption("--run-slow") or "slow" in expression
    keep_perf = config.getoption("--run-perf") or "perf" in expression
    keep_milestone = config.getoption("--run-mom-milestone") or "mom_milestone" in expression
    named = _named_files(config)
    nodes = _named_nodes(config)
    selected, deselected = [], []
    for item in items:
        drop = ((not keep_perf and item.get_closest_marker("perf") is not None) or
                (not keep_slow and item.get_closest_marker("slow") is not None and item.path not in named) or
                (not keep_milestone and item.get_closest_marker("mom_milestone") is not None and
                 not any(item.nodeid == node or item.nodeid.startswith(node + "[") for node in nodes)))
        (deselected if drop else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    budget = item.config.getoption("--slow-budget")
    if budget is None or item.get_closest_marker("slow") is not None:
        return report
    if report.when == "setup":
        item.stash[_SETUP_SECONDS] = report.duration
    elif report.when == "call" and report.passed:
        seconds = item.stash.get(_SETUP_SECONDS, 0.0) + report.duration
        if seconds > budget:
            report.outcome = "failed"
            report.longrepr = (f"slow-tier budget: setup+call took {seconds:.1f} s > {budget:g} s without the "
                               "slow marker; list the node in tests/slow_tests.txt or speed it up")
    return report


_SETUP_SECONDS = pytest.StashKey[float]()
