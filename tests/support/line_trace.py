"""Opt-in line tracer for tier-equivalence evidence and the MoM executed-sources trigger.

Load with ``-p tests.support.line_trace --line-trace=<output.jsonl>``. For every test node it
records the source lines executed inside functions (setup and call; import-time module bodies
are skipped) of files under ``--line-trace-prefix`` (comma-separated, relative to the repository
root, default ``witwin/maxwell,tests``) and appends one JSON line
``{"nodeid": ..., "lines": {relative_file: [lines]}}`` as soon as the node finishes, so a run cut
off by a deadline keeps every finished node. The file is truncated when the session starts. Comparing the line sets of a regression-tier test with its milestone original shows
whether the small case reaches the same code; ``integration_batch.py executed-sources`` turns the
files into the MoM trigger manifest. Autograd multithreading is disabled while tracing so backward
functions run, and are traced, on the calling thread; this does not change any computed value.
"""
from __future__ import annotations

import json
import sys
import threading
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class _Tracer:
    def __init__(self, prefixes):
        self.prefixes = prefixes
        self.lines = defaultdict(set)

    def call(self, frame, event, arg):
        code = frame.f_code
        return (self.line if code.co_name != "<module>" and code.co_filename.startswith(self.prefixes)
                and code.co_filename != __file__ else None)

    def line(self, frame, event, arg):
        if event == "line":
            self.lines[frame.f_code.co_filename].add(frame.f_lineno)
        return self.line


def pytest_addoption(parser):
    group = parser.getgroup("line-trace")
    group.addoption("--line-trace", metavar="PATH", help="write executed function lines per test node to PATH")
    group.addoption("--line-trace-prefix", default="witwin/maxwell,tests",
                    help="comma-separated repository-relative directories to trace")


def pytest_configure(config):
    output = config.getoption("--line-trace")
    if output:
        Path(output).write_text("")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    import torch

    prefixes = item.config.getoption("--line-trace-prefix").split(",")
    tracer = _Tracer(tuple(str(ROOT / prefix) + "/" for prefix in prefixes))
    previous = torch.autograd.is_multithreading_enabled()
    torch.autograd.set_multithreading_enabled(False)
    threading.settrace(tracer.call)
    sys.settrace(tracer.call)
    try:
        return (yield)
    finally:
        sys.settrace(None)
        threading.settrace(None)
        torch.autograd.set_multithreading_enabled(previous)
        output = item.config.getoption("--line-trace")
        if output:
            lines = {Path(name).relative_to(ROOT).as_posix(): sorted(numbers)
                     for name, numbers in sorted(tracer.lines.items())}
            with open(output, "a") as handle:
                handle.write(json.dumps(dict(nodeid=item.nodeid, lines=lines)) + "\n")
