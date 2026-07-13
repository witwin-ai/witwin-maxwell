from __future__ import annotations

import numpy as np

from benchmark.metrics import flux_incident_normalized_error
from benchmark.runner import _pick_flux_error
from benchmark.validation_catalog import VALIDATION_CASES, inventory_markdown


def test_plan_catalog_has_unique_case_names_and_all_families():
    names = [case.name for case in VALIDATION_CASES]
    assert len(names) == len(set(names))
    assert {case.family for case in VALIDATION_CASES} == {
        "sources", "media", "boundaries", "grid_geometry", "postprocess", "fdfd"
    }


def test_inventory_reports_every_planned_case_as_registered():
    report = inventory_markdown()
    assert "`astigmatic_beam`" in report
    assert "Registered" in report
    assert all(case.registered for case in VALIDATION_CASES)


def test_every_planned_case_builds_with_source_and_display_monitor():
    for case in VALIDATION_CASES:
        scenario = __import__("benchmark.scenes", fromlist=["SCENARIOS"]).SCENARIOS[case.name]
        scene = scenario.builder()
        assert scene.sources, case.name
        assert any(monitor.name == scenario.display_monitor for monitor in scene.monitors), case.name


def test_flux_error_uses_incident_power_not_near_zero_monitor_reference():
    maxwell = {"incident": {"flux": np.array([10.0])}, "blocked": {"flux": np.array([0.02])}}
    tidy3d = {"incident": {"flux": np.array([10.0])}, "blocked": {"flux": np.array([0.01])}}
    assert _pick_flux_error(maxwell, tidy3d) == 0.001


def test_incident_normalized_flux_error_handles_zero_scale():
    assert flux_incident_normalized_error([0.0], [0.0], incident_power=0.0) == 0.0
    assert np.isinf(flux_incident_normalized_error([1.0], [0.0], incident_power=0.0))
