"""Documentation capability tables are generated from the registry and cannot drift from it.

Regenerate with ``python scripts/dev/capability_tables.py --write`` after a
registry, field-manifest, native-inventory or census change.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

import witwin.maxwell.capability as capability
from witwin.maxwell.capability import CONFIG_KEYS, FEATURE_KEYS, METHODS, SCENE_KEYS, Support, capabilities
from witwin.maxwell.capability.tables import TABLES
from witwin.maxwell.fdtd.adjoint.capabilities import NATIVE_ADJOINT_CAPABILITIES

pytestmark = pytest.mark.host_validation

ROOT = Path(__file__).resolve().parents[3]


def _generator():
    spec = importlib.util.spec_from_file_location("capability_tables", ROOT / "scripts/dev/capability_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_tables_match_the_registry(capsys):
    generator = _generator()
    assert set(generator.render()) == {
        "FEATURE_LIST.md", "docs/reference/gradient-capability-matrix.md",
        "docs/reference/fdtd-capability-guard-census.md", "docs/reference/capability-registry.md",
        "docs/reference/scene-field-consumption.md"}
    assert generator.main([]) == 0, "stale capability tables; run scripts/dev/capability_tables.py --write\n" + (
        capsys.readouterr().out)


def test_every_feature_list_section_carries_its_table():
    text = (ROOT / "FEATURE_LIST.md").read_text()
    sections = {"fdfd": "## FDFD Solver Workflow", "known-limitations": "## Known Limitations",
                "mom": "## Experimental surface integral workflows", "electrostatics": "## Electrostatics (experimental)"}
    for name, heading in sections.items():
        section = text.split(heading, 1)[1].split("\n## ", 1)[0]
        assert f"<!-- capability-table: {name} -->" in section, heading


def test_census_registry_rows_name_refusal_rows():
    census = json.loads((ROOT / "docs/reference/fdtd-capability-guards.json").read_text())
    moved = [entry for entry in census["retired"] if entry.get("registry_rows")]
    assert moved
    for entry in moved:
        assert entry["positive_tests"], entry["id"]
        for item in entry["registry_rows"]:
            method, key = item.split(":", 1)
            row = capabilities(method).get(key)
            assert row is not None and row.support is not Support.SUPPORTED, (entry["id"], item)
    assert not any(entry.get("registry_rows") for entry in census["active"])


def test_gradient_matrix_native_labels_name_inventory_entries():
    text = (ROOT / "docs/reference/gradient-capability-matrix.md").read_text()
    variants = text.split("## Native variants", 1)[1].split("\n\n", 2)[1]
    labels = re.findall(r"\| `native_(\w+)` \|", variants)
    assert labels
    assert [label for label in labels if label.upper() not in NATIVE_ADJOINT_CAPABILITIES] == []


def test_registry_vocabulary():
    assert METHODS == tuple(TABLES) == ("fdtd", "fdfd", "mom", "mom.layered", "electrostatic", "capacitance")
    assert FEATURE_KEYS == (*SCENE_KEYS, *CONFIG_KEYS) and len(set(FEATURE_KEYS)) == len(FEATURE_KEYS)
    assert all(hasattr(capability, name) for name in capability.__all__)
    for method in METHODS:
        assert set(capabilities(method)) <= set(FEATURE_KEYS), method
