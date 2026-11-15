"""Wall-clock budgets of the layered MoM GPU tests, checked on an otherwise idle card.

The tests/mom/layered correctness modules below record each test's wall time (fixture setup
through teardown) as a JUnit property and assert only their memory caps, because
time on a shared device says nothing about correctness. This perf test reruns one module
in a fresh interpreter and requires every recorded time to stay under 120 s.
Run it with ``--run-perf`` only when no other job uses the device.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.perf, pytest.mark.slow]

BUDGET_SECONDS = 120
MODULES = (
    "test_layered_adaptive.py",
    "test_layered_admittance_fast.py",
    "test_layered_body_preconditioner.py",
    "test_layered_closed.py",
    "test_layered_fixture_fa.py",
    "test_layered_fixture_fb.py",
    "test_layered_fixture_r6.py",
    "test_layered_fixture_r6b.py",
    "test_layered_frontend.py",
    "test_layered_gradients.py",
    "test_layered_hmatrix.py",
    "test_layered_loss.py",
    "test_layered_matrix_free.py",
)


def recorded_seconds(case):
    """Per-test wall times, plus the shared adaptive solve recorded by its first consumer."""
    for prop in case.iter("property"):
        name, value = prop.get("name"), prop.get("value")
        if name in ("seconds", "gpu_seconds"):
            yield name, float(value)
        elif name == "adaptive_solve_resources":
            yield name, float(ast.literal_eval(value)["seconds"])


def plugin_arguments(config):
    """Forward the caller's -p plugins (for example a per-process memory cap) to the child."""
    args = list(config.invocation_params.args)
    forwarded = []
    for index, arg in enumerate(args):
        if arg == "-p" and index + 1 < len(args) and not args[index + 1].startswith("no:"):
            forwarded += ["-p", args[index + 1]]
        elif arg.startswith("-p") and len(arg) > 2 and not arg[2:].startswith("no:"):
            forwarded.append(arg)
    return forwarded


@pytest.mark.parametrize("module", MODULES)
def test_layered_module_stays_within_wall_budget(module, request, tmp_path):
    junit = tmp_path / "junit.xml"
    root = Path(__file__).resolve().parents[2]
    child = subprocess.run(
        [sys.executable, "-m", "pytest", str(root / "tests/mom/layered" / module), "--run-slow", "-p", "no:cacheprovider",
         "-q", f"--junitxml={junit}", f"--basetemp={tmp_path / 'run'}", *plugin_arguments(request.config)],
        cwd=root, env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))},
        capture_output=True, text=True)
    assert junit.exists(), child.stdout[-4000:] + child.stderr[-4000:]
    times = {(case.get("name"), name): seconds
             for case in ET.parse(junit).getroot().iter("testcase")
             for name, seconds in recorded_seconds(case)}
    assert times, f"{module} recorded no wall times"
    over = {key: seconds for key, seconds in times.items() if seconds >= BUDGET_SECONDS}
    assert not over, over
