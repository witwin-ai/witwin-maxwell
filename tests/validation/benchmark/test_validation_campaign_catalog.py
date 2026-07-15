from __future__ import annotations

import numpy as np
import torch

from benchmark.metrics import flux_incident_normalized_error
from benchmark.runner import _pick_flux_error
from benchmark.scenes import SCENARIOS
from benchmark.scenes.coverage import COVERAGE_SCENARIOS
from benchmark.validation_catalog import VALIDATION_CASES, inventory_markdown
from witwin.maxwell.compiler.sources import compile_fdfd_sources
import witwin.maxwell as mw


def test_plan_catalog_has_unique_case_names_and_all_families():
    names = [case.name for case in VALIDATION_CASES]
    assert len(names) == len(set(names))
    assert {case.family for case in VALIDATION_CASES} == {
        "sources", "media", "boundaries", "grid_geometry", "postprocess", "fdfd"
    }


def test_every_coverage_scenario_is_registered_and_catalogued():
    catalogued = {case.name for case in VALIDATION_CASES}
    for scenario in COVERAGE_SCENARIOS:
        assert scenario.name in SCENARIOS, scenario.name
        assert scenario.name in catalogued, scenario.name
        assert SCENARIOS[scenario.name] is scenario, scenario.name


def test_inventory_reports_every_planned_case_as_registered():
    report = inventory_markdown()
    assert "`astigmatic_beam`" in report
    assert "Registered" in report
    assert all(case.registered for case in VALIDATION_CASES)


def test_every_planned_case_builds_with_source_and_display_monitor():
    for case in VALIDATION_CASES:
        scenario = __import__("benchmark.scenes", fromlist=["SCENARIOS"]).SCENARIOS[case.name]
        scene = scenario.builder()
        assert scene.resolved_sources(), case.name
        assert any(monitor.name == scenario.display_monitor for monitor in scene.resolved_monitors()), case.name


def test_named_feature_scenarios_exercise_the_claimed_public_objects():
    custom = SCENARIOS["custom_pole_uniform_slab"].builder()
    assert isinstance(custom.structures[0].material.lorentz_poles[0], mw.CustomLorentzPole)

    perturbation = SCENARIOS["perturbation_uniform_slab"].builder()
    assert isinstance(perturbation.structures[0].material, mw.PerturbationMedium)

    medium2d = SCENARIOS["static_medium2d_sheet"].builder()
    assert type(medium2d.structures[0].material) is mw.Medium2D

    material_region = SCENARIOS["material_region_slab"].builder()
    assert len(material_region.material_regions) == 1
    assert torch.unique(material_region.material_regions[0].density).numel() == 1

    mode_port = SCENARIOS["mode_port_straight_wg"].builder()
    assert len(mode_port.ports) == 2
    assert len(mode_port.resolved_sources()) == 1

    ricker = SCENARIOS["ricker_axis_x_anisotropic"].builder()
    assert isinstance(ricker.sources[0].source_time, mw.RickerWavelet)
    assert ricker.sources[0].direction == (-1.0, 0.0, 0.0)

    mesh = SCENARIOS["explicit_mesh_scatter"].builder()
    assert type(mesh.structures[0].geometry) is mw.Mesh

    auto = SCENARIOS["autogrid_override_refinement"].builder().grid
    assert auto.override_structures
    assert auto.layer_refinement is not None

    asymmetric = SCENARIOS["asymmetric_boundary_faces"].builder().boundary
    assert asymmetric.face_kind("x", "low") == "pec"
    assert asymmetric.face_kind("x", "high") == "pml"


def test_tfsf_vacuum_observes_only_outside_the_total_field_box():
    scene = SCENARIOS["tfsf_vacuum"].builder()
    source = scene.sources[0]
    z_lo, z_hi = source.injection.bounds[2]
    flux_positions = [
        monitor.position
        for monitor in scene.monitors
        if isinstance(monitor, mw.FluxMonitor)
    ]
    assert flux_positions
    assert all(position < z_lo or position > z_hi for position in flux_positions)


def test_scalar_coverage_scenarios_select_load_bearing_observables():
    assert SCENARIOS["point_monitor_probe"].scalar_observable == "point_probe_values"
    assert SCENARIOS["mode_monitor_two_planes"].scalar_observable == "mode_plane_ratio"
    assert SCENARIOS["mode_port_straight_wg"].scalar_observable == "mode_port_transmission"
    assert SCENARIOS["permittivity_monitor_slab"].scalar_observable == "permittivity_stats"
    assert SCENARIOS["time_monitor_vacuum"].scalar_observable == "time_monitor_traces"


def test_fdfd_campaign_cases_compile_for_the_declared_backend():
    for scenario in SCENARIOS.values():
        if scenario.solver != "fdfd":
            continue
        scene = scenario.builder()
        assert scenario.reference_solver == "fdtd"
        assert scenario.run_time_factor >= 20.0
        assert scenario.compare_magnitude
        assert compile_fdfd_sources(scene, default_frequency=scenario.frequencies[0])


def test_flux_error_uses_incident_power_not_near_zero_monitor_reference():
    maxwell = {"incident": {"flux": np.array([10.0])}, "blocked": {"flux": np.array([0.02])}}
    tidy3d = {"incident": {"flux": np.array([10.0])}, "blocked": {"flux": np.array([0.01])}}
    assert _pick_flux_error(maxwell, tidy3d) == 0.001


def test_incident_normalized_flux_error_handles_zero_scale():
    assert flux_incident_normalized_error([0.0], [0.0], incident_power=0.0) == 0.0
    assert np.isinf(flux_incident_normalized_error([1.0], [0.0], incident_power=0.0))
