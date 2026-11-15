"""The opt-in line tracer appends function-body lines per node and nothing at import time."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))}

SAMPLE = '''
from tests.support.slow_tier import listed

def test_listed():
    assert listed("a.py::t", ["a.py"])

def test_nothing():
    pass
'''


def test_line_trace_records_function_lines_per_node(tmp_path):
    (tmp_path / "test_sample.py").write_text(SAMPLE)
    output = tmp_path / "trace.jsonl"
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "tests.support.line_trace", "-q",
         f"--line-trace={output}", "--line-trace-prefix=tests/support", "test_sample.py"],
        cwd=tmp_path, capture_output=True, text=True, env=ENV)
    assert child.returncode == 0, child.stdout + child.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    trace = {record["nodeid"]: record["lines"] for record in records}
    assert len(records) == len(trace)
    assert set(trace) == {"test_sample.py::test_listed", "test_sample.py::test_nothing"}
    assert trace["test_sample.py::test_nothing"] == {}
    lines = trace["test_sample.py::test_listed"]["tests/support/slow_tier.py"]
    source = (ROOT / "tests/support/slow_tier.py").read_text().splitlines()
    start = next(i for i, line in enumerate(source, 1) if line.startswith("def listed("))
    # Only the body of listed() ran; the module body (imports, definitions) is not recorded.
    assert lines and all(start < n <= start + 2 for n in lines)


def test_line_trace_keeps_finished_nodes_when_the_run_is_cut_off(tmp_path):
    (tmp_path / "test_cut.py").write_text(SAMPLE + "\nimport os\n\ndef test_zz_cut():\n    os._exit(3)\n")
    output = tmp_path / "trace.jsonl"
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "tests.support.line_trace", "-q",
         f"--line-trace={output}", "--line-trace-prefix=tests/support", "test_cut.py"],
        cwd=tmp_path, capture_output=True, text=True, env=ENV)
    assert child.returncode == 3
    assert [json.loads(line)["nodeid"] for line in output.read_text().splitlines()] == [
        "test_cut.py::test_listed", "test_cut.py::test_nothing"]
