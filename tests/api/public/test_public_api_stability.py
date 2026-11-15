"""The public-API inventory matches the live namespaces, and experimental objects warn once.

Regenerate the inventory with ``python scripts/dev/public_api.py --write`` after adding,
moving, removing or re-marking a public object.
"""
from __future__ import annotations

import csv
import importlib.util
import re
import subprocess
import sys

import pytest

import witwin.maxwell as mw
from witwin.maxwell._stability import FEATURES, feature_of, stability

from tests._fresh_interpreter import ROOT

pytestmark = pytest.mark.host_validation


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("public_api", ROOT / "scripts/dev/public_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inventory():
    with open(ROOT / "docs/reference/public-api.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_inventory_lists_every_public_object_with_its_marker(generator):
    inventory = _inventory()
    assert [tuple(row[key] for key in ("namespace", "name", "kind", "stability")) for row in inventory] == \
        generator.rows(), "stale docs/reference/public-api.csv; run scripts/dev/public_api.py --write"
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", row["since"]) for row in inventory)
    assert {row["stability"] for row in inventory} == {"stable", "experimental"}


def test_feature_list_stability_table_is_current(generator):
    text = (ROOT / "FEATURE_LIST.md").read_text()
    assert generator.render_feature_list(text) == text, \
        "stale FEATURE_LIST API Stability table; run scripts/dev/public_api.py --write"


def test_markers_name_a_feature_and_q5_objects_are_stable(generator):
    experimental = [value for _, _, value in generator.public_objects() if stability(value) == "experimental"]
    assert experimental and {feature_of(value) for value in experimental} <= set(FEATURES)
    for name in ("ModeSource", "ModeMonitor", "ModePort", "TFSF", "LumpedPort", "AxisPath"):
        assert stability(getattr(mw, name)) == "stable"
    # A stable subclass does not inherit its base's marker.
    assert stability(mw.media.fit.DispersionFitReport) == "stable"
    assert stability(mw.rf.rational.FitReport) == "experimental"
    assert mw.rf.WavePort.__stability__ == "experimental" and mw.Simulation.fdfd.__stability__ == "experimental"
    assert not hasattr(mw.circuits.CircuitDevice, "__stability__")  # protocol members stay unchanged
    assert [entry for entry, _ in generator.experimental_entry_points()] == [
        "Simulation.batch", "Simulation.capacitance", "Simulation.electrostatic", "Simulation.fdfd", "Simulation.mom"]


def _run(code: str, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *options, "-c", code], cwd=ROOT, capture_output=True, text=True)


def test_experimental_object_warns_once_per_process_and_only_for_outside_callers():
    code = (
        "import warnings, witwin.maxwell as mw\n"
        "from witwin.maxwell.errors import ExperimentalWarning\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    exec('mw.circuits.CircuitNode(\"inside\")', {'__name__': 'witwin.maxwell.probe', 'mw': mw})\n"
        "    assert caught == [], caught\n"
        "    mw.circuits.CircuitNode('a')\n"
        "    mw.circuits.CircuitNode('b')\n"
        "    mw.rf.rational.RationalFitConfig()\n"
        "    mw.CW(frequency=1e9)\n"
        "messages = [str(item.message) for item in caught]\n"
        "assert all(item.category is ExperimentalWarning for item in caught), caught\n"
        "assert len(messages) == 2 and messages[0].startswith('CircuitNode is experimental (circuit'), messages\n"
        "assert caught[0].filename == '<string>', caught[0].filename\n"
        "assert messages[1].startswith('RationalFitConfig is experimental'), messages\n"
    )
    process = _run(code)
    assert process.returncode == 0, process.stdout + process.stderr


def _pytest(*options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *options, "-p", "no:cacheprovider",
         "tests/api/public/test_public_api_stability.py::test_constructs_an_experimental_object"],
        cwd=ROOT, capture_output=True, text=True)


def test_warning_filters_apply_and_the_suite_configuration_holds_under_w_error():
    process = _run("import witwin.maxwell as mw; mw.circuits.CircuitNode('a')", "-W", "error")
    assert process.returncode != 0 and "ExperimentalWarning: CircuitNode is experimental" in process.stderr
    # The tests/conftest.py filter holds when Python escalates warnings. (pytest's own -W takes
    # precedence over the configuration: add -W ignore::witwin.maxwell.errors.ExperimentalWarning.)
    process = _pytest("-W", "error", "-m", "pytest", "-q")
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-4000:]


def test_constructs_an_experimental_object():
    assert mw.circuits.CircuitNode("a").name == "a"
