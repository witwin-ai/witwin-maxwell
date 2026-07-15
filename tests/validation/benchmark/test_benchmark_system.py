from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from benchmark import cache as benchmark_cache
from benchmark import plotting as benchmark_plotting
from benchmark import report as benchmark_report
from benchmark import runner as benchmark_runner
from benchmark import paths as benchmark_paths
from benchmark.grid_convergence import GridSample, estimate_observed_order, render_markdown
from benchmark.metrics import (
    align_arrays,
    align_plane_fields,
    best_fit_field_scale,
    phase_align_field,
    significant_field_mask,
    vector_field_comparison,
)
from benchmark.models import ScenarioMetrics
from benchmark.scenes import SCENARIOS, build_scene
from benchmark.tidy3d_scene import benchmark_physical_bounds, prepare_tidy3d_benchmark_scene
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def test_benchmark_scenes_build():
    assert {
        "dipole_vacuum",
        "planewave_vacuum",
        "dipole_dielectric_box",
        "planewave_dielectric_sphere",
        "dielectric_slab",
        "metal_sphere",
        "multi_dielectric",
        "lorentz_resonator",
        "dipole_ey",
        "dipole_offcenter",
        "high_eps_box",
        "dielectric_sphere",
        "dipole_dielectric_sphere",
        "dipole_two_freq",
    }.issubset(set(SCENARIOS))
    for name in SCENARIOS:
        scene = build_scene(name)
        assert scene.resolved_sources(), name
        assert len(scene.resolved_monitors()) >= 5, name
        scenario = SCENARIOS[name]
        assert scenario.display_monitor in {
            monitor.name for monitor in scene.resolved_monitors()
        }, name


def test_group7_scenarios_use_supported_source_boundary_and_monitor_contracts():
    lossy = build_scene("lossy_metal_slab")
    metal = lossy.structures[0].geometry
    metal_lower = metal.position - metal.size / 2
    metal_upper = metal.position + metal.size / 2
    assert metal_lower[0] < lossy.domain.bounds[0][0]
    assert metal_upper[0] > lossy.domain.bounds[0][1]
    prepared_lossy = prepare_scene(lossy)
    assert np.min(np.abs(prepared_lossy.z_nodes64 - float(metal_lower[2]))) < 1.0e-7

    periodic = build_scene("periodic_grating")
    assert periodic.boundary.axis_kind("x") == "periodic"
    assert periodic.boundary.axis_kind("y") == "periodic"
    assert periodic.boundary.axis_kind("z") == "pml"
    assert isinstance(periodic.sources[0].injection, mw.TFSF)

    bloch = build_scene("bloch_oblique")
    assert bloch.boundary.axis_kind("x") == "bloch"
    assert bloch.boundary.axis_kind("y") == "bloch"
    assert bloch.boundary.axis_kind("z") == "pml"
    assert all(value > 0.0 for value in bloch.boundary.bloch_wavevector[:2])

    for name in ("pec_cavity", "pmc_cavity"):
        cavity = build_scene(name)
        assert any(monitor.name == "resonance_probe" for monitor in cavity.monitors)
        assert SCENARIOS[name].scalar_observable == "cavity_resonance"
    pmc = build_scene("pmc_cavity")
    assert isinstance(pmc.sources[0], mw.CustomCurrentSource)
    assert set(pmc.sources[0].current_dataset.components) == {"Mz"}
    assert SCENARIOS["pmc_cavity"].display_component == "Hz"

    diffraction = build_scene("grating_diffraction")
    assert any(isinstance(monitor, mw.DiffractionMonitor) for monitor in diffraction.monitors)
    for name in ("sphere_rcs", "antenna_directivity"):
        scene = build_scene(name)
        assert any(isinstance(monitor, mw.ClosedSurfaceMonitor) for monitor in scene.monitors)


def test_plane_slab_continues_through_transverse_external_pml():
    scene = build_scene("nonuniform_custom_grid")
    prepared = prepare_scene(scene)
    geometry = scene.structures[0].geometry
    lower = geometry.position - geometry.size / 2
    upper = geometry.position + geometry.size / 2

    assert float(lower[0]) <= float(prepared.x_nodes64[0])
    assert float(upper[0]) >= float(prepared.x_nodes64[-1])
    assert float(lower[1]) <= float(prepared.y_nodes64[0])
    assert float(upper[1]) >= float(prepared.y_nodes64[-1])

    model = prepared.compile_materials()
    z_index = int(np.argmin(np.abs(prepared.z_nodes64)))
    eps_slice = model["eps_components"]["x"][:, :, z_index]
    assert torch.allclose(eps_slice, torch.full_like(eps_slice, 3.0), atol=1.0e-4)


def test_source_normalized_tfsf_cache_uses_standard_spatial_unit_conversion():
    scene = build_scene("periodic_grating")
    monitors = {
        "field": {
            "kind": "field",
            "position": 250_000.0,
            "x": np.array([-640_000.0, 640_000.0]),
            "fields": {"Ey": np.ones((2, 1), dtype=np.complex64)},
        },
        "flux": {"kind": "flux", "flux": np.array([2.0])},
    }

    scaled = benchmark_runner._rescale_tidy3d_fields(
        monitors,
        scene=scene,
        normalize_source=True,
    )

    np.testing.assert_allclose(scaled["field"]["fields"]["Ey"], 1.0e6)
    np.testing.assert_allclose(scaled["flux"]["flux"], 2.0)
    assert scaled["field"]["position"] == pytest.approx(0.25)
    np.testing.assert_allclose(scaled["field"]["x"], (-0.64, 0.64))


def test_vector_field_comparison_uses_one_global_phase_and_area_weights():
    u = np.array([-0.4, -0.1, 0.2, 0.6])
    v = np.array([-0.3, 0.0, 0.5])
    reference = np.zeros((3, u.size, v.size), dtype=np.complex128)
    reference[0] = 1.0 + u[:, None] + 0.25j * v[None, :]
    reference[1] = (0.5 - 0.2j) * reference[0]
    reference[2] = (-0.1 + 0.7j) * reference[0]
    actual = 2.5 * np.exp(0.4j) * reference

    comparison = vector_field_comparison(actual, reference, coords=(u, v))

    assert comparison["valid"] is True
    assert comparison["overlap"] == pytest.approx(1.0)
    assert comparison["energy_ratio"] == pytest.approx(2.5)
    assert comparison["field_shape_l2"] == pytest.approx(0.0, abs=1.0e-14)
    assert comparison["phase"] == pytest.approx(np.exp(-0.4j))


def test_vector_field_comparison_detects_orthogonal_and_relative_phase_errors():
    coords = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.5, 0.5, 4))
    ey = np.zeros((3, 5, 4), dtype=np.complex128)
    ez = np.zeros_like(ey)
    ey[1] = 1.0
    ez[2] = 1.0
    orthogonal = vector_field_comparison(ey, ez, coords=coords)
    assert orthogonal["valid"] is True
    assert orthogonal["overlap"] == pytest.approx(0.0)

    reference = ey + ez
    wrong_relative_phase = ey + 1j * ez
    relative_phase = vector_field_comparison(wrong_relative_phase, reference, coords=coords)
    assert relative_phase["overlap"] == pytest.approx(np.sqrt(0.5))
    assert relative_phase["field_shape_l2"] > 0.7


def test_vector_field_comparison_rejects_zero_fields_without_nan_metrics():
    zeros = np.zeros((3, 2, 2), dtype=np.complex128)
    comparison = vector_field_comparison(
        zeros,
        zeros,
        coords=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    )
    assert comparison == {"valid": False, "reason": "zero vector field"}


def test_tidy3d_cache_keeps_only_publicly_requested_monitor_fields():
    scene = build_scene("pmc_cavity")
    monitors = {
        "resonance_probe": {
            "kind": "field",
            "fields": {
                "Ez": np.asarray([9.0, 1.0, 2.0]),
                "Hz": np.asarray([1.0, 5.0, 2.0]),
            },
        }
    }

    scaled = benchmark_runner._rescale_tidy3d_fields(monitors, scene=scene)

    assert set(scaled["resonance_probe"]["fields"]) == {"Hz"}
    np.testing.assert_allclose(
        scaled["resonance_probe"]["fields"]["Hz"],
        np.asarray([1.0, 5.0, 2.0]) * 1.0e6,
    )


def test_lorentz_resonator_keeps_the_point_source_outside_the_dispersive_body():
    scene = build_scene("lorentz_resonator")
    source = scene.sources[0]
    cylinder = scene.structures[0].geometry

    radial_distance = math.hypot(
        float(source.position[0] - cylinder.position[0]),
        float(source.position[1] - cylinder.position[1]),
    )
    axial_distance = abs(float(source.position[2] - cylinder.position[2]))
    assert radial_distance > cylinder.radius or axial_distance > 0.5 * cylinder.height


@pytest.mark.parametrize("name", ("ring_resonator_s21", "waveguide_s_matrix"))
def test_s_parameter_benchmarks_compare_the_output_transverse_mode(name):
    scene = build_scene(name)
    field_monitor = next(monitor for monitor in scene.monitors if monitor.name == "field")

    assert isinstance(field_monitor, mw.PlaneMonitor)
    assert field_monitor.axis == "x"
    assert field_monitor.position == pytest.approx(0.35)
    assert not any(isinstance(monitor, mw.FluxMonitor) for monitor in scene.monitors)
    assert not SCENARIOS[name].compare_flux


def test_benchmark_cache_key_supports_geometry_and_source_objects():
    scene = build_scene("dielectric_slab")
    cache_key = benchmark_runner._benchmark_cache_key(scene, frequencies=(2.0e9,), run_time_factor=15.0)
    assert isinstance(cache_key, str)
    assert len(cache_key) == 64


def test_reference_refresh_cli_forwards_force_refresh(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        benchmark_runner,
        "generate_tidy3d_references",
        lambda names, *, force_refresh=False: captured.update(
            names=names,
            force_refresh=force_refresh,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--references-only",
            "--refresh-references",
            "dipole_vacuum",
        ],
    )

    benchmark_runner.main()

    assert captured == {"names": ["dipole_vacuum"], "force_refresh": True}


def test_reference_refresh_cli_requires_references_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--refresh-references", "dipole_vacuum"],
    )

    with pytest.raises(SystemExit, match="requires --references-only"):
        benchmark_runner.main()


def test_modulated_benchmark_uses_carrier_referenced_field_spectra():
    scenario = SCENARIOS["modulated_slab"]

    assert not scenario.normalize_source
    assert scenario.spectral_reference_index == 1
    assert not scenario.compare_flux


def test_mode_benchmark_cache_key_tracks_export_contract(monkeypatch):
    scene = build_scene("mode_source_wg")
    original = benchmark_runner._benchmark_cache_key(
        scene, frequencies=(2.0e9,), run_time_factor=15.0
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_MODE_EXPORT_CONTRACT_VERSION",
        benchmark_runner._MODE_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene, frequencies=(2.0e9,), run_time_factor=15.0
    )
    assert changed != original


def test_mesh_benchmark_cache_key_tracks_export_contract(monkeypatch):
    scene = build_scene("autogrid_ring")
    original = benchmark_runner._benchmark_cache_key(
        scene, frequencies=(2.0e9,), run_time_factor=8.0
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_MESH_EXPORT_CONTRACT_VERSION",
        benchmark_runner._MESH_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene, frequencies=(2.0e9,), run_time_factor=8.0
    )
    assert changed != original


def test_cone_benchmark_cache_key_tracks_geometry_export_contract(monkeypatch):
    scenario = SCENARIOS["cone_scatter"]
    scene = build_scene("cone_scatter")
    original = benchmark_runner._benchmark_cache_key(
        scene, scenario.frequencies, scenario.run_time_factor
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_GEOMETRY_EXPORT_CONTRACT_VERSION",
        benchmark_runner._GEOMETRY_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene, scenario.frequencies, scenario.run_time_factor
    )
    assert changed != original


def test_material_benchmark_cache_key_tracks_export_contract(monkeypatch):
    scenario = SCENARIOS["debye_slab"]
    scene = build_scene("debye_slab")
    original = benchmark_runner._benchmark_cache_key(
        scene, scenario.frequencies, scenario.run_time_factor
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_MATERIAL_EXPORT_CONTRACT_VERSION",
        benchmark_runner._MATERIAL_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene, scenario.frequencies, scenario.run_time_factor
    )
    assert changed != original


def test_mixed_undamped_pole_families_use_material_export_contract():
    material = mw.Material(
        drude_poles=(mw.DrudePole(plasma_frequency=3.0e9, gamma=0.2e9),),
        lorentz_poles=(
            mw.LorentzPole(
                delta_eps=0.4,
                resonance_frequency=2.0e9,
                gamma=0.0,
            ),
        ),
    )
    assert benchmark_runner._material_uses_export_contract(material)


def test_directional_source_benchmark_cache_key_tracks_export_contract(monkeypatch):
    scenario = SCENARIOS["full_tensor_slab"]
    scene = build_scene("full_tensor_slab")
    original = benchmark_runner._benchmark_cache_key(
        scene, scenario.frequencies, scenario.run_time_factor
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_DIRECTIONAL_SOURCE_EXPORT_CONTRACT_VERSION",
        benchmark_runner._DIRECTIONAL_SOURCE_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene, scenario.frequencies, scenario.run_time_factor
    )
    assert changed != original


def test_default_normal_plane_wave_does_not_use_directional_source_contract():
    scene = build_scene("planewave_vacuum")
    assert not benchmark_runner._directional_source_uses_export_contract(scene.sources[0])


def test_normalized_tfsf_cache_key_tracks_source_export_contract(monkeypatch):
    scenario = SCENARIOS["tfsf_vacuum"]
    scene = build_scene("tfsf_vacuum")
    original = benchmark_runner._benchmark_cache_key(
        scene,
        scenario.frequencies,
        scenario.run_time_factor,
        normalize_source=True,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_TFSF_SOURCE_EXPORT_CONTRACT_VERSION",
        benchmark_runner._TFSF_SOURCE_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene,
        scenario.frequencies,
        scenario.run_time_factor,
        normalize_source=True,
    )
    assert changed != original


def test_continuous_wave_cache_key_tracks_source_time_export_contract(monkeypatch):
    scenario = SCENARIOS["modulated_slab"]
    scene = build_scene("modulated_slab")
    original = benchmark_runner._benchmark_cache_key(
        scene,
        scenario.frequencies,
        scenario.run_time_factor,
        normalize_source=False,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_SOURCE_TIME_EXPORT_CONTRACT_VERSION",
        benchmark_runner._SOURCE_TIME_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene,
        scenario.frequencies,
        scenario.run_time_factor,
        normalize_source=False,
    )
    assert changed != original


def test_raw_time_monitor_cache_key_tracks_gaussian_export_contract(monkeypatch):
    scenario = SCENARIOS["time_monitor_vacuum"]
    scene = build_scene("time_monitor_vacuum")
    original = benchmark_runner._benchmark_cache_key(
        scene,
        scenario.frequencies,
        scenario.run_time_factor,
        normalize_source=False,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_SOURCE_TIME_EXPORT_CONTRACT_VERSION",
        benchmark_runner._SOURCE_TIME_EXPORT_CONTRACT_VERSION + 1,
    )
    changed = benchmark_runner._benchmark_cache_key(
        scene,
        scenario.frequencies,
        scenario.run_time_factor,
        normalize_source=False,
    )
    assert changed != original


def test_incident_scene_signature_ignores_names_but_tracks_launch_physics():
    vacuum = build_scene("planewave_vacuum")
    slab = build_scene("dielectric_slab")
    assert benchmark_runner._incident_scene_signature(vacuum, (2.0e9,)) == (
        benchmark_runner._incident_scene_signature(slab, (2.0e9,))
    )

    changed_source = slab.sources[0]
    slab.sources[0] = mw.PlaneWave(
        direction=changed_source.direction,
        polarization=changed_source.polarization,
        source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.4e9),
    )
    assert benchmark_runner._incident_scene_signature(vacuum, (2.0e9,)) != (
        benchmark_runner._incident_scene_signature(slab, (2.0e9,))
    )


def test_incident_scene_signature_ignores_unit_transverse_plane_wave_orientation():
    vacuum = build_scene("planewave_vacuum")
    rotated = build_scene("planewave_vacuum")
    source = rotated.sources[0]
    rotated.sources[0] = mw.PlaneWave(
        direction=source.direction,
        polarization=(0.8, 0.6, 0.0),
        source_time=source.source_time,
        injection=source.injection,
        injection_axis=source.injection_axis,
    )
    assert benchmark_runner._incident_scene_signature(vacuum, (2.0e9,)) == (
        benchmark_runner._incident_scene_signature(rotated, (2.0e9,))
    )


def test_incident_scene_signature_ignores_plane_wave_amplitude_and_phase():
    unit = build_scene("planewave_periodic_vacuum")
    driven = build_scene("planewave_periodic_vacuum")
    source = driven.sources[0]
    driven.sources[0] = mw.PlaneWave(
        direction=source.direction,
        polarization=source.polarization,
        source_time=mw.GaussianPulse(
            frequency=source.source_time.frequency,
            fwidth=source.source_time.fwidth,
            amplitude=7.0,
            phase=0.83,
        ),
        injection=source.injection,
        injection_axis=source.injection_axis,
    )

    assert benchmark_runner._incident_scene_signature(unit, (2.0e9,)) == (
        benchmark_runner._incident_scene_signature(driven, (2.0e9,))
    )


def test_incident_scene_signature_preserves_boundary_and_direction_requirements():
    periodic = build_scene("planewave_periodic_vacuum")
    pml = build_scene("planewave_vacuum")
    angled = build_scene("planewave_periodic_vacuum")
    source = angled.sources[0]
    angled.sources[0] = mw.PlaneWave(
        direction=(0.6, 0.0, 0.8),
        polarization=(0.0, 1.0, 0.0),
        source_time=source.source_time,
        injection=source.injection,
        injection_axis="z",
    )

    periodic_signature = benchmark_runner._incident_scene_signature(periodic, (2.0e9,))
    assert periodic_signature != benchmark_runner._incident_scene_signature(pml, (2.0e9,))
    assert periodic_signature != benchmark_runner._incident_scene_signature(angled, (2.0e9,))


def test_cached_incident_power_scales_unit_reference_by_amplitude_squared(monkeypatch):
    scene = build_scene("planewave_periodic_vacuum")
    source = scene.sources[0]
    scene.sources[0] = mw.PlaneWave(
        direction=source.direction,
        polarization=source.polarization,
        source_time=mw.GaussianPulse(
            frequency=source.source_time.frequency,
            fwidth=source.source_time.fwidth,
            amplitude=3.0,
            phase=1.2,
        ),
        injection=source.injection,
        injection_axis=source.injection_axis,
    )
    loaded = []

    def fake_load(name, *, expected_cache_key):
        loaded.append((name, expected_cache_key))
        return {"incident": {"flux": np.asarray([0.4])}}

    monkeypatch.setattr(benchmark_runner, "load_tidy3d_result", fake_load)

    raw_power = benchmark_runner._cached_plane_wave_incident_power(
        scene,
        (2.0e9,),
        normalize_source=False,
    )
    normalized_power = benchmark_runner._cached_plane_wave_incident_power(
        scene,
        (2.0e9,),
        normalize_source=True,
    )

    assert raw_power == pytest.approx(0.4 * 3.0**2)
    assert normalized_power == pytest.approx(0.4 * 3.0**2)
    assert [name for name, _ in loaded] == [
        "planewave_periodic_vacuum",
        "planewave_periodic_vacuum",
    ]


@pytest.mark.parametrize(
    ("scenario_name", "amplitude"),
    [
        ("kerr_slab", 3.0e7),
        ("kerr_slab_strong", 1.0e8),
        ("tpa_slab", 1.0e5),
    ],
)
def test_nonlinear_plane_wave_uses_matching_periodic_incident_reference(
    monkeypatch,
    scenario_name,
    amplitude,
):
    scene = build_scene(scenario_name)
    loaded = []

    def fake_load(name, *, expected_cache_key):
        loaded.append((name, expected_cache_key))
        return {"incident": {"flux": np.asarray([0.25])}}

    monkeypatch.setattr(benchmark_runner, "load_tidy3d_result", fake_load)

    power = benchmark_runner._cached_plane_wave_incident_power(
        scene,
        (2.0e9,),
        normalize_source=False,
    )

    assert power == pytest.approx(0.25 * amplitude**2)
    assert [name for name, _ in loaded] == ["planewave_periodic_vacuum"]


def test_tfsf_incident_power_uses_physical_box_aperture():
    scene = build_scene("tfsf_vacuum")
    impedance = 4.0e-7 * np.pi * 299_792_458.0

    power = benchmark_runner._tfsf_plane_wave_incident_power(
        scene,
        normalize_source=True,
    )

    assert power == pytest.approx(0.5 * 0.6 * 0.6 / impedance)


def test_flux_error_uses_empty_scene_incident_power_for_reflective_cases():
    maxwell = {
        "reflected": {"flux": np.array([-0.016])},
        "transmitted": {"flux": np.array([0.029])},
    }
    tidy3d = {
        "reflected": {"flux": np.array([-0.043])},
        "transmitted": {"flux": np.array([0.030])},
    }

    error = benchmark_runner._pick_flux_error(maxwell, tidy3d, incident_power=0.585)

    assert error == pytest.approx(0.027 / 0.585)


def test_full_tensor_benchmark_launches_a_transverse_eigenpolarization():
    scene = build_scene("full_tensor_slab")
    tensor = np.asarray(scene.structures[0].material.epsilon_tensor.rows)
    polarization = np.asarray(scene.sources[0].polarization)
    transverse = polarization[:2]
    response = tensor[:2, :2] @ transverse

    np.testing.assert_allclose(response, (response @ transverse) * transverse, rtol=1e-9, atol=1e-9)
    assert abs(polarization[1]) > 0.3


def test_directional_field_comparison_excludes_the_soft_source_stencil():
    scene = build_scene("planewave_vacuum")
    source_position = benchmark_runner.soft_plane_wave_coordinate(scene, "z", 1.0)
    spacing = 0.025
    x = np.array((-0.1, 0.1))
    z = source_position + spacing * np.array((-1.5, -0.5, 0.5, 1.5, 2.5))
    reference = np.ones((x.size, z.size), dtype=np.complex128)
    actual = reference * np.exp(0.4j)
    actual[:, 2] = 100.0  # first downstream cell contains the source stencil

    compared_actual, compared_reference = benchmark_runner._comparison_fields(
        scene,
        "y",
        (x, z),
        actual,
        reference,
    )

    assert compared_actual.size == 2 * 2
    np.testing.assert_allclose(compared_actual, compared_reference)


def test_tfsf_field_comparison_removes_only_the_global_reference_phase():
    scene = build_scene("periodic_grating")
    reference = np.array(
        ((1.0 + 0.5j, -0.5 + 0.25j), (0.2 - 0.1j, 2.0 - 0.5j)),
        dtype=np.complex128,
    )
    actual = reference * np.exp(0.73j)

    compared_actual, compared_reference = benchmark_runner._comparison_fields(
        scene,
        "y",
        (np.array((-0.1, 0.1)), np.array((-0.2, 0.2))),
        actual,
        reference,
    )

    np.testing.assert_allclose(compared_actual, compared_reference)
    assert compared_actual.size == reference.size


def test_spectral_phase_anchor_preserves_sideband_relative_phase():
    scene = build_scene("periodic_grating")
    coords = (np.array((-0.1, 0.1)), np.array((-0.2, 0.2)))
    reference = np.ones((2, 2), dtype=np.complex128)
    carrier = reference * np.exp(0.4j)
    raw_carrier, selected_reference = benchmark_runner._comparison_fields(
        scene,
        "y",
        coords,
        carrier,
        reference,
        align_phase=False,
    )
    _, carrier_phase = phase_align_field(raw_carrier, selected_reference)
    upper_sideband = carrier * 1j

    independently_aligned, _ = benchmark_runner._comparison_fields(
        scene,
        "y",
        coords,
        upper_sideband,
        reference,
    )
    carrier_anchored, anchored_reference = benchmark_runner._comparison_fields(
        scene,
        "y",
        coords,
        upper_sideband,
        reference,
        phase_factor=carrier_phase,
    )

    np.testing.assert_allclose(independently_aligned, reference.ravel())
    np.testing.assert_allclose(carrier_anchored, 1j * anchored_reference)
    assert np.linalg.norm(carrier_anchored - anchored_reference) > 1.0


def test_explicit_phase_factor_applies_without_a_directional_source():
    scene = mw.Scene(device="cpu")
    reference = np.ones((2, 2), dtype=np.complex128)
    actual = reference * np.exp(0.4j)
    phase_factor = np.exp(-0.4j)

    compared_actual, compared_reference = benchmark_runner._comparison_fields(
        scene,
        "y",
        (np.array((-0.1, 0.1)), np.array((-0.2, 0.2))),
        actual,
        reference,
        phase_factor=phase_factor,
    )

    np.testing.assert_allclose(compared_actual, compared_reference)


def test_resonance_diagnostic_uses_nearest_resonant_field_slice():
    frequencies = (1.0e9, 2.0e9, 3.0e9)
    per_frequency = [
        {"field_l2": 0.9},
        {"field_l2": 0.1},
        {"field_l2": 0.2},
    ]
    scalar_metrics = [
        {
            "observable": "resonance_frequency",
            "maxwell": 2.10e9,
            "tidy3d": 2.15e9,
        }
    ]

    assert (
        benchmark_runner._diagnostic_frequency_index(
            frequencies,
            per_frequency,
            scalar_metrics,
        )
        == 1
    )
    assert benchmark_runner._diagnostic_frequency_index(frequencies, per_frequency, []) == 0


def test_point_source_field_comparison_excludes_the_singular_source_disk():
    scene = mw.Scene(device="cpu").add_source(
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez")
    )
    coords = np.linspace(-0.2, 0.2, 9)
    reference = np.ones((coords.size, coords.size), dtype=np.complex128)
    actual = reference * np.exp(0.3j)
    actual[coords.size // 2, coords.size // 2] = 1.0e6

    compared_actual, compared_reference = benchmark_runner._comparison_fields(
        scene,
        "z",
        (coords, coords),
        actual,
        reference,
        monitor_position=0.0,
    )

    assert compared_actual.size < actual.size
    np.testing.assert_allclose(compared_actual, compared_reference)


def test_point_source_field_comparison_excludes_reference_weak_field_noise():
    scene = mw.Scene(device="cpu").add_source(
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez")
    )
    coords = np.linspace(-0.2, 0.2, 9)
    reference = np.full((coords.size, coords.size), 1.0e-6, dtype=np.complex128)
    reference[1:3, 1:3] = 1.0
    actual = reference * np.exp(0.3j)
    actual[6:, 6:] = np.exp(2.0j) * 1.0e-6

    compared_actual, compared_reference = benchmark_runner._comparison_fields(
        scene,
        "z",
        (coords, coords),
        actual,
        reference,
        monitor_position=0.0,
    )

    assert compared_actual.size == 4
    np.testing.assert_allclose(compared_actual, compared_reference)


def test_monitor_fields_can_be_normalized_to_one_spectral_reference():
    coords = np.linspace(-0.1, 0.1, 3)
    values = np.stack(
        (
            np.full((3, 3), 2.0 + 0.0j),
            np.full((3, 3), 4.0 + 0.0j),
        ),
        axis=-1,
    )
    monitors = {
        "field": {
            "axis": "z",
            "x": coords,
            "y": coords,
            "frequencies": (1.0, 2.0),
            "fields": {"Ex": values},
        }
    }

    normalized = benchmark_runner._normalize_monitor_fields_to_spectral_reference(
        monitors,
        monitor_name="field",
        component="Ex",
        reference_index=1,
    )

    np.testing.assert_allclose(normalized["field"]["fields"]["Ex"][..., 0], 0.5)
    np.testing.assert_allclose(normalized["field"]["fields"]["Ex"][..., 1], 1.0)
    np.testing.assert_allclose(monitors["field"]["fields"]["Ex"][..., 1], 4.0)


def test_align_arrays_center_crops():
    a = np.ones((10, 12))
    b = np.ones((8, 10))
    a_aligned, b_aligned = align_arrays(a, b)
    assert a_aligned.shape == (8, 10)
    assert b_aligned.shape == (8, 10)


def test_align_plane_fields_uses_physical_coordinates():
    source_x = np.linspace(-1.0, 1.0, 5)
    source_y = np.linspace(-1.0, 1.0, 5)
    reference_x = np.linspace(-0.5, 0.5, 3)
    reference_y = np.linspace(-0.5, 0.5, 3)

    source_grid_x, source_grid_y = np.meshgrid(source_x, source_y, indexing="ij")
    reference_grid_x, reference_grid_y = np.meshgrid(reference_x, reference_y, indexing="ij")
    source_field = source_grid_x + 2.0 * source_grid_y
    reference_field = reference_grid_x + 2.0 * reference_grid_y

    aligned_source, aligned_reference, (target_x, target_y) = align_plane_fields(
        source_field,
        reference_field,
        source_coords=(source_x, source_y),
        reference_coords=(reference_x, reference_y),
    )

    np.testing.assert_allclose(target_x, reference_x)
    np.testing.assert_allclose(target_y, reference_y)
    np.testing.assert_allclose(aligned_source, aligned_reference)


def test_phase_alignment_removes_only_global_phasor_on_significant_support():
    reference = np.array((0.0j, 1.0 + 2.0j, -0.5 + 0.25j))
    actual = 1.2 * reference * np.exp(-0.7j)
    support = significant_field_mask(reference)

    aligned, factor = phase_align_field(actual, reference, mask=support)

    np.testing.assert_allclose(aligned[support], 1.2 * reference[support])
    assert abs(factor) == pytest.approx(1.0)
    assert abs(aligned[1]) == pytest.approx(abs(actual[1]))


def test_best_fit_field_scale_separates_global_scale_from_shape_error():
    reference = np.array((1.0 + 2.0j, -0.5 + 0.25j, 0.2 - 0.1j))
    actual = (1.7 - 0.4j) * reference

    aligned, scale = best_fit_field_scale(actual, reference)

    np.testing.assert_allclose(aligned, reference)
    assert scale == pytest.approx(1.0 / (1.7 - 0.4j))


def test_scalar_observables_compare_point_probes_and_two_mode_planes():
    point_monitors = {
        "probe_0": {"fields": {"Ex": np.asarray([1.0 + 2.0j]), "Ey": np.asarray([3.0])}},
        "probe_1": {"fields": {"Ez": np.asarray([-0.5j])}},
    }
    point_values = benchmark_runner._scalar_observables(
        "point_probe_values", point_monitors, (1.0e9,)
    )
    assert set(point_values) == {"probe_0_Ex", "probe_0_Ey", "probe_1_Ez"}
    np.testing.assert_array_equal(point_values["probe_0_Ex"], [1.0 + 2.0j])

    mode_monitors = {
        "mode_mid": {
            "scalars": {
                "amplitude_forward": np.asarray([2.0 + 0.0j]),
                "effective_index": np.asarray([1.5]),
            }
        },
        "mode_out": {
            "scalars": {
                "amplitude_forward": np.asarray([1.0 + 1.0j]),
                "effective_index": np.asarray([1.5]),
            }
        },
    }
    mode_values = benchmark_runner._scalar_observables(
        "mode_plane_ratio", mode_monitors, (1.0e9,)
    )
    np.testing.assert_allclose(mode_values["forward_amplitude_ratio"], [0.5 + 0.5j])


def test_scalar_observables_return_permittivity_monitor_statistics():
    monitors = {
        "permittivity": {
            "scalars": {
                "eps_x_mean": np.asarray([2.0]),
                "eps_x_min": np.asarray([1.0]),
                "eps_x_max": np.asarray([4.0]),
            }
        }
    }

    values = benchmark_runner._scalar_observables(
        "permittivity_stats", monitors, (1.0e9,)
    )

    assert set(values) == {"eps_x_mean", "eps_x_min", "eps_x_max"}
    np.testing.assert_array_equal(values["eps_x_max"], [4.0])


def test_scalar_observables_resample_and_normalize_time_monitor_traces():
    monitors = {
        "field_time": {"fields": {"Ex": np.asarray([0.0, 2.0, 0.0])}},
        "flux_time": {"flux": np.asarray([0.0, -4.0, 0.0, 1.0])},
    }

    values = benchmark_runner._scalar_observables(
        "time_monitor_traces", monitors, (1.0e9,)
    )

    assert values["field_time_Ex"].shape == (128,)
    assert values["flux_time"].shape == (128,)
    assert np.max(np.abs(values["field_time_Ex"])) == pytest.approx(1.0)
    assert np.max(np.abs(values["flux_time"])) == pytest.approx(1.0)


def test_time_monitor_trace_comparison_uses_waveform_l2_not_frequency_count():
    monitors = {
        "field_time": {
            "t": np.asarray([0.0, 1.0, 2.0]),
            "fields": {"Ex": np.asarray([0.0, 2.0, 0.0])},
        },
        "flux_time": {
            "t": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "flux": np.asarray([0.0, -4.0, 0.0, 1.0]),
        },
    }

    metrics = benchmark_runner._compare_scalar_observables(
        "time_monitor_traces", monitors, monitors, (1.0e9,)
    )

    assert {item["observable"] for item in metrics} == {"field_time_Ex", "flux_time"}
    assert all(item["complex_error"] == pytest.approx(0.0) for item in metrics)
    assert all(item["phase_error"] is None for item in metrics)


def test_time_monitor_trace_comparison_uses_common_physical_time_window():
    full_times = np.arange(11, dtype=np.float64)
    truncated_times = np.arange(7, dtype=np.float64)
    full_trace = np.exp(-0.5 * ((full_times - 3.0) / 1.2) ** 2)
    truncated_trace = np.exp(-0.5 * ((truncated_times - 3.0) / 1.2) ** 2)
    maxwell = {
        "field_time": {"t": full_times, "fields": {"Ex": full_trace}},
        "flux_time": {"t": full_times, "flux": full_trace},
    }
    tidy3d = {
        "field_time": {"t": truncated_times, "fields": {"Ex": truncated_trace}},
        "flux_time": {"t": truncated_times, "flux": truncated_trace},
    }

    metrics = benchmark_runner._compare_scalar_observables(
        "time_monitor_traces", maxwell, tidy3d, (1.0,),
    )

    assert all(item["complex_error"] == pytest.approx(0.0) for item in metrics)
    assert all(item["time_window_s"] == pytest.approx(6.0) for item in metrics)


def test_time_lag_diagnostic_reports_bounded_sample_shift():
    reference = np.zeros(32)
    reference[10:14] = (0.25, 1.0, 0.5, 0.1)
    actual = np.roll(reference, 1)

    lag_s, error = benchmark_runner._time_lag_diagnostic(
        actual,
        reference,
        time_step=1.0e-10,
        frequencies=(5.0e8,),
    )

    assert abs(lag_s) == pytest.approx(1.0e-10)
    assert error == pytest.approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA FDTD runtime")
def test_time_monitor_vacuum_peaks_at_physical_propagation_delay():
    scenario = SCENARIOS["time_monitor_vacuum"]
    scene = build_scene("time_monitor_vacuum")
    _, monitors, _, _ = benchmark_runner._run_maxwell(
        scene,
        frequencies=scenario.frequencies,
        run_time_factor=scenario.run_time_factor,
        normalize_source=False,
    )
    source = scene.resolved_sources()[0]
    source_z = benchmark_runner.soft_plane_wave_coordinate(scene, "z", 1.0)

    field = monitors["field_time"]
    field_times = np.asarray(field["t"], dtype=np.float64)
    field_values = np.asarray(field["fields"]["Ex"])
    field_peak_time = float(field_times[np.argmax(np.abs(field_values))])
    expected_field_peak = float(source.source_time.delay) + (0.25 - source_z) / 299_792_458.0
    field_sample_step = float(np.median(np.diff(field_times)))

    flux = monitors["flux_time"]
    flux_times = np.asarray(flux["t"], dtype=np.float64)
    flux_values = np.asarray(flux["flux"])
    flux_peak_time = float(flux_times[np.argmax(np.abs(flux_values))])
    expected_flux_peak = float(source.source_time.delay) + (0.30 - source_z) / 299_792_458.0
    flux_sample_step = float(np.median(np.diff(flux_times)))

    assert np.all(np.diff(field_times) > 0.0)
    assert np.all(np.diff(flux_times) > 0.0)
    assert abs(field_peak_time - expected_field_peak) <= 1.5 * field_sample_step
    assert abs(flux_peak_time - expected_flux_peak) <= 1.5 * flux_sample_step


def test_grid_convergence_helpers_report_observed_order_and_performance():
    assert estimate_observed_order(0.09, 0.04, 1.5) == pytest.approx(2.0)
    samples = [
        GridSample(0.06, (17, 17, 17), 1.0, 0.2, 5000.0, 100, 64.0),
        GridSample(0.04, (25, 25, 25), 2.0, 0.3, 3333.0, 150, 96.0),
        GridSample(0.02666666666666667, (38, 38, 38), 4.0, 0.5, 2000.0, 225, 144.0),
    ]

    report = render_markdown(samples, 0.09, 0.04, 2.0, updated_at="now")

    assert "Observed order:** 2.0000" in report
    assert "coarse vs medium | 9.000000e-02" in report
    assert "Peak GPU (MiB)" in report


@pytest.mark.parametrize(
    ("coarse", "fine", "ratio"),
    ((0.0, 0.1, 1.5), (0.1, 0.0, 1.5), (0.1, 0.05, 1.0)),
)
def test_grid_convergence_rejects_invalid_inputs(coarse, fine, ratio):
    with pytest.raises(ValueError):
        estimate_observed_order(coarse, fine, ratio)


def test_select_monitor_plane_field_accepts_trailing_frequency_axis():
    coords = np.linspace(-0.1, 0.1, 4)
    plane = np.arange(16, dtype=np.float64).reshape(4, 4)
    monitor = {
        "axis": "z",
        "frequencies": (1.0, 2.0),
        "x": coords,
        "y": coords,
    }
    stacked = np.stack([plane, plane + 10.0], axis=-1)

    selected = benchmark_runner._select_monitor_plane_field(
        monitor,
        "Ex",
        stacked,
        freq_index=1,
    )

    np.testing.assert_allclose(selected, plane + 10.0)


def test_benchmark_cache_round_trip(tmp_path, monkeypatch):
    benchmark_paths.scenario_path_parts.cache_clear()
    monkeypatch.setattr(benchmark_paths, "CACHE_DIR", tmp_path)

    field = (np.random.randn(6, 7) + 1j * np.random.randn(6, 7)).astype(np.complex128)
    output_path = benchmark_cache.save_tidy3d_result(
        "dipole_vacuum",
        frequencies=(1.0e9,),
        monitors={
            "field_xy": {
                "kind": "field",
                "fields": {"Ez": field},
                "x": np.linspace(-1.0, 1.0, 6),
                "y": np.linspace(-1.0, 1.0, 7),
            },
            "flux_pos_z": {"kind": "flux", "flux": np.array([1.2])},
            "mode_out": {
                "kind": "mode",
                "scalars": {
                    "amplitude_forward": np.array([0.3 + 0.4j]),
                    "effective_index": np.array([1.8]),
                },
            },
        },
        cache_key="demo-cache-key",
    )
    assert output_path == tmp_path / "dipole" / "dipole_vacuum.h5"
    loaded = benchmark_cache.load_tidy3d_result("dipole_vacuum", expected_cache_key="demo-cache-key")
    np.testing.assert_allclose(loaded["field_xy"]["fields"]["Ez"], field)
    np.testing.assert_allclose(loaded["flux_pos_z"]["flux"], np.array([1.2]))
    np.testing.assert_allclose(
        loaded["mode_out"]["scalars"]["amplitude_forward"],
        np.array([0.3 + 0.4j]),
    )
    np.testing.assert_allclose(
        loaded["mode_out"]["scalars"]["effective_index"],
        np.array([1.8]),
    )


def test_scalar_observable_comparison_uses_complex_modal_ratios():
    maxwell = {
        "mode_in": {
            "scalars": {
                "amplitude_forward": np.array([2.0 + 0.0j]),
                "amplitude_backward": np.array([0.2j]),
            }
        },
        "mode_out": {
            "scalars": {
                "amplitude_forward": np.array([1.0 + 1.0j]),
                "effective_index": np.array([1.75]),
            }
        },
    }
    tidy3d = {
        "mode_in": {
            "scalars": {
                "amplitude_forward": np.array([4.0 + 0.0j]),
                "amplitude_backward": np.array([0.4j]),
            }
        },
        "mode_out": {
            "scalars": {
                "amplitude_forward": np.array([2.0 + 2.0j]),
                "effective_index": np.array([1.75]),
            }
        },
    }

    metrics = benchmark_runner._compare_scalar_observables(
        "waveguide_s_matrix", maxwell, tidy3d, (2.0e9,)
    )

    assert {item["observable"] for item in metrics} == {"S11", "S12", "S21", "S22", "n_eff"}
    assert max(float(item["complex_error"]) for item in metrics) == pytest.approx(0.0)


def test_ring_scalar_observable_reports_one_resonance_frequency():
    frequencies = tuple(np.linspace(1.8e9, 2.2e9, 9))
    monitors = {
        "mode_out": {
            "scalars": {
                "effective_index": np.full(len(frequencies), 1.965),
            }
        }
    }

    metrics = benchmark_runner._compare_scalar_observables(
        "ring_s21",
        monitors,
        monitors,
        frequencies,
    )

    resonance_rows = [
        item for item in metrics if item["observable"] == "resonance_frequency"
    ]
    assert len(resonance_rows) == 1
    assert resonance_rows[0]["maxwell"].real == pytest.approx(2.0e9)
    assert resonance_rows[0]["phase_error"] is None


def test_cavity_scalar_observable_uses_sub_bin_quadratic_peak():
    frequencies = tuple(np.linspace(100.0, 120.0, 5))
    expected_peak = 113.0
    spectrum = np.exp(-((np.asarray(frequencies) - expected_peak) / 7.0) ** 2)
    monitors = {
        "resonance_probe": {
            "fields": {"Ez": spectrum},
        }
    }

    observables = benchmark_runner._scalar_observables(
        "cavity_resonance",
        monitors,
        frequencies,
    )

    assert observables["resonance_frequency"][0] == pytest.approx(expected_peak)
    assert np.max(observables["normalized_probe_spectrum"]) == pytest.approx(1.0)


def test_cavity_scalar_observable_accepts_magnetic_dual_probe():
    frequencies = (1.0e8, 1.5e8, 2.0e8)
    monitors = {
        "resonance_probe": {
            "fields": {"Hz": np.asarray([1.0, 4.0, 2.0], dtype=np.complex128)}
        }
    }

    observables = benchmark_runner._scalar_observables(
        "cavity_resonance", monitors, frequencies
    )

    assert observables["normalized_probe_spectrum"] == pytest.approx((0.25, 1.0, 0.5))
    assert 1.0e8 < observables["resonance_frequency"][0] < 2.0e8


def test_real_scalar_comparison_plot_is_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_plotting, "ensure_directories", lambda: None)
    monkeypatch.setattr(
        benchmark_plotting,
        "scenario_plot_dir",
        lambda _name: tmp_path / "scalar_case",
    )
    metrics = [
        {
            "frequency": frequency,
            "observable": "normalized_probe_spectrum",
            "maxwell": maxwell,
            "tidy3d": tidy3d,
        }
        for frequency, maxwell, tidy3d in (
            (1.0e8, 0.2, 0.1),
            (1.5e8, 1.0, 0.9),
            (2.0e8, 0.3, 0.4),
        )
    ]
    metrics.append(
        {
            "frequency": 1.5e8,
            "observable": "resonance_frequency",
            "maxwell": 1.52e8,
            "tidy3d": 1.49e8,
        }
    )

    output = benchmark_plotting.save_scalar_comparison_plot(
        scenario_name="scalar_case", scalar_metrics=metrics
    )

    assert output == tmp_path / "scalar_case" / "scalar_comparison.png"
    assert output.is_file()


def test_time_trace_comparison_plot_is_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_plotting, "ensure_directories", lambda: None)
    monkeypatch.setattr(
        benchmark_plotting,
        "scenario_plot_dir",
        lambda _name: tmp_path / "time_case",
    )
    times = np.linspace(0.0, 2.0e-9, 21)

    output = benchmark_plotting.save_time_trace_comparison_plot(
        scenario_name="time_case",
        traces=[
            {
                "t": times,
                "maxwell": np.sin(2.0 * np.pi * 1.0e9 * times),
                "tidy3d": np.sin(2.0 * np.pi * 1.0e9 * times + 0.05),
                "label": "field_time/Ex",
                "ylabel": "normalized Ex",
            }
        ],
    )

    assert output == tmp_path / "time_case" / "time_trace_comparison.png"
    assert output.is_file()


def test_diffraction_scalar_observable_normalizes_common_orders():
    monitors = {
        "orders": {
            "scalars": {
                "orders_m": np.asarray([-2, -1, 0, 1, 2]),
                "orders_n": np.zeros(5, dtype=np.int64),
                "order_power": np.asarray([[1.0], [2.0], [4.0], [2.0], [1.0]]),
            }
        }
    }

    observables = benchmark_runner._scalar_observables(
        "diffraction_orders",
        monitors,
        (1.0e9,),
    )

    assert observables["eta_+0_0"][0] == pytest.approx(0.4)
    assert observables["eta_-1_0"][0] == pytest.approx(0.2)
    assert observables["eta_+3_0"][0] == pytest.approx(0.0)


def test_tidy3d_closed_surface_faces_feed_common_far_field_postprocess(monkeypatch):
    surface = mw.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(0.2, 0.2, 0.2),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(device="cpu").add_monitor(surface)
    coords = np.linspace(-0.1, 0.1, 3)
    monitors = {}
    for face in surface.faces:
        coord_names = benchmark_runner._PLANE_COORD_NAMES[face.axis]
        monitors[face.name] = {
            "kind": "field",
            "axis": face.axis,
            "position": face.plane_position,
            "frequencies": (1.0e9,),
            coord_names[0]: coords,
            coord_names[1]: coords,
            "fields": {
                component: np.ones((3, 3), dtype=np.complex128)
                for component in face.fields
            },
        }

    monkeypatch.setattr(
        benchmark_runner,
        "_far_field_scalar_summary",
        lambda currents: {"D_max": np.asarray([1.5])},
    )

    benchmark_runner._attach_tidy3d_surface_scalars(monitors, scene)

    np.testing.assert_allclose(monitors["huygens"]["scalars"]["D_max"], [1.5])


def test_benchmark_cache_rejects_mismatched_cache_key(tmp_path, monkeypatch):
    benchmark_paths.scenario_path_parts.cache_clear()
    monkeypatch.setattr(benchmark_paths, "CACHE_DIR", tmp_path)

    benchmark_cache.save_tidy3d_result(
        "dipole_vacuum",
        frequencies=(1.0e9,),
        monitors={},
        cache_key="cache-a",
    )

    with pytest.raises(ValueError, match="cache key mismatch"):
        benchmark_cache.load_tidy3d_result("dipole_vacuum", expected_cache_key="cache-b")


def test_extract_maxwell_monitors_prefers_native_component_payload():
    scene = mw.Scene(device="cpu").add_monitor(
        mw.PlaneMonitor(
            name="field_xy",
            axis="z",
            position=0.0,
            fields=("Ez",),
            frequencies=(1.5e9,),
        )
    )

    native = np.full((4, 4), 2.0, dtype=np.complex128)
    aligned = np.full((3, 3), 0.5, dtype=np.complex128)
    payload = {
        "kind": "plane",
        "axis": "z",
        "position": 0.0,
        "x": np.linspace(-0.1, 0.1, 3),
        "y": np.linspace(-0.1, 0.1, 3),
        "Ez": aligned,
        "components": {
            "Ez": {
                "data": native,
                "coords": (
                    np.linspace(-0.15, 0.15, 4),
                    np.linspace(-0.15, 0.15, 4),
                ),
            }
        },
    }

    class DummyResult:
        def monitor(self, name):
            assert name == "field_xy"
            return payload

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)
    np.testing.assert_allclose(extracted["field_xy"]["fields"]["Ez"], native)
    np.testing.assert_allclose(
        extracted["field_xy"]["field_coords"]["Ez"]["x"],
        np.linspace(-0.15, 0.15, 4),
    )


def test_extract_maxwell_monitors_also_preserves_collocated_vector_payload():
    scene = mw.Scene(device="cpu").add_monitor(
        mw.FinitePlaneMonitor(
            name="field",
            position=(0.0, 0.0, 0.0),
            size=(0.0, 0.4, 0.4),
            fields=("Ex", "Ey", "Ez"),
            frequencies=(1.5e9,),
        )
    )
    coords = np.linspace(-0.2, 0.2, 3)
    payload = {
        "y": coords,
        "z": coords,
        "Ex": np.ones((3, 3)),
        "Ey": 2.0 * np.ones((3, 3)),
        "Ez": 3.0 * np.ones((3, 3)),
        "components": {
            name: {
                "data": (index + 5.0) * np.ones((4, 4)),
                "coords": (np.linspace(-0.2, 0.2, 4), np.linspace(-0.2, 0.2, 4)),
            }
            for index, name in enumerate(("Ex", "Ey", "Ez"))
        },
    }

    class DummyResult:
        def monitor(self, name):
            assert name == "field"
            return payload

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)["field"]
    np.testing.assert_allclose(extracted["fields"]["Ex"], 5.0)
    np.testing.assert_allclose(extracted["collocated_fields"]["Ex"], 1.0)
    np.testing.assert_allclose(extracted["collocated_fields"]["Ez"], 3.0)
    np.testing.assert_allclose(extracted["collocated_coords"]["y"], coords)


def test_aligned_vector_field_comparison_uses_collocated_components():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    coords = np.linspace(-0.25, 0.25, 3)
    reference = {
        name: (index + 1.0) * np.ones((3, 3), dtype=np.complex128)
        for index, name in enumerate(("Ex", "Ey", "Ez"))
    }
    common = {"kind": "field", "axis": "x", "position": 0.0, "y": coords, "z": coords}
    tidy3d_monitor = {**common, "fields": reference}
    maxwell_monitor = {
        **common,
        "fields": {name: np.zeros((4, 4)) for name in reference},
        "collocated_fields": {
            name: 2.0 * np.exp(0.3j) * values for name, values in reference.items()
        },
        "collocated_coords": {"y": coords, "z": coords},
    }

    comparison, maxwell_vector, tidy3d_vector, aligned_coords = (
        benchmark_runner._aligned_vector_field_comparison(
            scene,
            maxwell_monitor,
            tidy3d_monitor,
            components=("Ex", "Ey", "Ez"),
            freq_index=0,
        )
    )

    assert comparison["overlap"] == pytest.approx(1.0)
    assert comparison["energy_ratio"] == pytest.approx(2.0)
    assert maxwell_vector.shape == tidy3d_vector.shape == (3, 3, 3)
    np.testing.assert_allclose(aligned_coords[0], coords)


def test_extract_fdfd_monitors_preserves_component_yee_coordinates():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    ).add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ex", "Ey", "Ez"),
            frequencies=(1.0e9,),
        )
    )
    prepared = prepare_scene(scene)
    fields = {
        "EX": torch.ones((prepared.x_half.numel(), prepared.y.numel(), prepared.z.numel())),
        "EY": torch.ones((prepared.x.numel(), prepared.y_half.numel(), prepared.z.numel())),
        "EZ": torch.ones((prepared.x.numel(), prepared.y.numel(), prepared.z_half.numel())),
    }

    class DummyResult:
        method = "fdfd"
        frequency = 1.0e9
        prepared_scene = prepared

        def tensor(self, name):
            return fields[name]

    result = DummyResult()
    result.fields = fields
    extracted = benchmark_runner._extract_maxwell_monitors(result, scene)["field"]

    for component in ("Ex", "Ey", "Ez"):
        coords = benchmark_runner._component_plane_coords(extracted, component)
        assert coords is not None
        assert extracted["fields"][component].shape == tuple(len(axis) for axis in coords)


def test_extract_maxwell_point_monitor_unwraps_component_data():
    scene = mw.Scene(device="cpu").add_monitor(
        mw.PointMonitor("probe", position=(0.0, 0.0, 0.0), fields=("Ez",))
    )
    payload = {
        "frequencies": (1.0, 2.0),
        "components": {
            "Ez": {"data": np.asarray([1.0 + 0.0j, 2.0 + 0.0j])},
        },
    }

    class DummyResult:
        def monitor(self, name):
            assert name == "probe"
            return payload

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)

    np.testing.assert_allclose(extracted["probe"]["fields"]["Ez"], [1.0, 2.0])
    assert extracted["probe"]["frequencies"] == (1.0, 2.0)


def test_extract_maxwell_mode_monitor_preserves_complex_frequency_series():
    frequencies = (1.8e9, 2.0e9)
    scene = mw.Scene(device="cpu").add_monitor(
        mw.ModeMonitor(
            "mode_out",
            position=(0.2, 0.0, 0.0),
            size=(0.0, 0.4, 0.5),
            polarization="Ez",
            frequencies=frequencies,
        )
    )

    class DummyResult:
        def raw_monitor(self, name):
            assert name == "mode_out"
            return {"frequencies": frequencies}

        def monitor(self, name, *, freq_index):
            assert name == "mode_out"
            return {
                "amplitude_forward": torch.tensor(1.0 + 0.1j * freq_index),
                "amplitude_backward": torch.tensor(0.01j * (freq_index + 1)),
                "effective_index": 1.7 + 0.05 * freq_index,
            }

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)

    assert extracted["mode_out"]["kind"] == "mode"
    np.testing.assert_allclose(
        extracted["mode_out"]["scalars"]["amplitude_forward"],
        np.array([1.0 + 0.0j, 1.0 + 0.1j]),
    )
    np.testing.assert_allclose(
        extracted["mode_out"]["scalars"]["effective_index"],
        np.array([1.7, 1.75]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensor payloads")
def test_extract_maxwell_monitors_converts_cuda_tensor_payloads_to_numpy():
    scene = mw.Scene(device="cpu").add_monitor(
        mw.PlaneMonitor(
            name="field_xy",
            axis="z",
            position=0.0,
            fields=("Ez",),
            frequencies=(1.5e9,),
        )
    )

    field = torch.full((4, 4), 2.0 + 0.5j, dtype=torch.complex64, device="cuda")
    coords = torch.linspace(-0.15, 0.15, 4, device="cuda")
    payload = {
        "kind": "plane",
        "axis": "z",
        "position": 0.0,
        "x": coords,
        "y": coords,
        "components": {
            "Ez": {
                "data": field,
                "coords": (coords, coords),
            }
        },
    }

    class DummyResult:
        def monitor(self, name):
            assert name == "field_xy"
            return payload

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)
    assert isinstance(extracted["field_xy"]["fields"]["Ez"], np.ndarray)
    assert isinstance(extracted["field_xy"]["field_coords"]["Ez"]["x"], np.ndarray)
    np.testing.assert_allclose(
        extracted["field_xy"]["fields"]["Ez"],
        field.detach().cpu().numpy(),
    )
    np.testing.assert_allclose(
        extracted["field_xy"]["field_coords"]["Ez"]["x"],
        coords.detach().cpu().numpy(),
    )


def test_extract_maxwell_monitors_trims_flux_to_physical_interior():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.2),
        boundary=mw.BoundarySpec.pml(num_layers=1),
        device="cpu",
    ).add_monitor(
        mw.FluxMonitor(
            name="flux_z",
            axis="z",
            position=0.0,
            frequencies=(1.0,),
        )
    )

    coords = np.linspace(-0.4, 0.4, 5)
    field = np.ones((5, 5), dtype=np.complex128)
    payload = {
        "kind": "plane",
        "axis": "z",
        "position": 0.0,
        "normal_direction": "+",
        "x": coords,
        "y": coords,
        "Ex": field,
        "Ey": np.zeros_like(field),
        "Hx": np.zeros_like(field),
        "Hy": field,
        "flux": 0.32,
        "power": 0.32,
        "components": {
            "Ex": {"data": field, "coords": (coords, coords)},
            "Ey": {"data": np.zeros_like(field), "coords": (coords, coords)},
            "Hx": {"data": np.zeros_like(field), "coords": (coords, coords)},
            "Hy": {"data": field, "coords": (coords, coords)},
        },
    }

    class DummyResult:
        def monitor(self, name):
            assert name == "flux_z"
            return payload

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)
    assert extracted["flux_z"]["raw_flux"] == pytest.approx(0.32)
    assert extracted["flux_z"]["flux"] == pytest.approx(0.18)
    assert extracted["flux_z"]["power"] == pytest.approx(0.18)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensor payloads")
def test_extract_maxwell_flux_monitors_accepts_cuda_tensor_payloads():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.2),
        boundary=mw.BoundarySpec.pml(num_layers=1),
        device="cpu",
    ).add_monitor(
        mw.FluxMonitor(
            name="flux_z",
            axis="z",
            position=0.0,
            frequencies=(1.0,),
        )
    )

    coords = torch.linspace(-0.4, 0.4, 5, device="cuda")
    field = torch.ones((5, 5), dtype=torch.complex64, device="cuda")
    payload = {
        "kind": "plane",
        "axis": "z",
        "position": 0.0,
        "normal_direction": "+",
        "x": coords,
        "y": coords,
        "Ex": field,
        "Ey": torch.zeros_like(field),
        "Hx": torch.zeros_like(field),
        "Hy": field,
        "flux": torch.tensor(0.32, dtype=torch.float32, device="cuda"),
        "power": torch.tensor(0.32, dtype=torch.float32, device="cuda"),
        "components": {
            "Ex": {"data": field, "coords": (coords, coords)},
            "Ey": {"data": torch.zeros_like(field), "coords": (coords, coords)},
            "Hx": {"data": torch.zeros_like(field), "coords": (coords, coords)},
            "Hy": {"data": field, "coords": (coords, coords)},
        },
    }

    class DummyResult:
        def monitor(self, name):
            assert name == "flux_z"
            return payload

    extracted = benchmark_runner._extract_maxwell_monitors(DummyResult(), scene)
    assert isinstance(extracted["flux_z"]["raw_flux"], np.ndarray)
    assert isinstance(extracted["flux_z"]["flux"], np.ndarray)
    assert extracted["flux_z"]["raw_flux"] == pytest.approx(0.32)
    assert extracted["flux_z"]["flux"] == pytest.approx(0.18)


def test_report_writer_updates_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "RESULTS.md"
    monkeypatch.setattr(benchmark_report, "RESULTS_MD", output_path)
    benchmark_paths.scenario_path_parts.cache_clear()

    dipole_result = ScenarioMetrics(
        name="dipole_vacuum",
        description="demo",
        frequencies=(1.5e9,),
        maxwell_time_s=1.23,
        tidy3d_cache_hit=True,
        field_l2=0.1,
        field_shape_l2=0.01,
        field_linf=0.2,
        field_corr=0.99,
        flux_error=0.05,
        compared_monitor="field_xy",
        compared_component="Ez",
        material_source_plot=Path("plots/dipole/material_source.png"),
        field_plot=Path("plots/dipole/field_comparison.png"),
        updated_at="2026-03-16T12:00:00-07:00",
        maxwell_ms_per_step=0.25,
        maxwell_steps_per_second=4000.0,
        maxwell_dft_samples=320,
        maxwell_peak_gpu_memory_mb=128.5,
        scalar_plot=Path("plots/dipole/scalar_comparison.png"),
        scalar_metrics=[
            {
                "frequency": 1.5e9,
                "observable": "S21",
                "maxwell": 0.8 + 0.1j,
                "tidy3d": 0.79 + 0.11j,
                "complex_error": 0.02,
                "magnitude_error": 0.01,
                "phase_error": 0.015,
            }
        ],
        per_frequency=[
            {
                "frequency": 1.5e9,
                "field_l2": 0.1,
                "field_shape_l2": 0.01,
                "field_linf": 0.2,
                "field_corr": 0.99,
            }
        ],
    )
    planewave_result = ScenarioMetrics(
        name="planewave_vacuum",
        description="demo 2",
        frequencies=(1.5e9,),
        maxwell_time_s=2.34,
        tidy3d_cache_hit=False,
        field_l2=0.2,
        field_shape_l2=0.02,
        field_linf=0.3,
        field_corr=0.98,
        flux_error=0.15,
        compared_monitor="field_xz",
        compared_component="Ex",
        material_source_plot=Path("plots/planewave/material_source.png"),
        field_plot=Path("plots/planewave/field_comparison.png"),
        updated_at="2026-03-16T12:10:00-07:00",
        per_frequency=[
            {
                "frequency": 1.5e9,
                "field_l2": 0.2,
                "field_shape_l2": 0.02,
                "field_linf": 0.3,
                "field_corr": 0.98,
            }
        ],
    )
    benchmark_report.write_results_markdown([dipole_result])
    benchmark_report.write_results_markdown([planewave_result])
    content = output_path.read_text(encoding="utf-8")
    assert "## Metric Guide" in content
    assert "Field L2 [smaller, <1e-1]" in content
    assert "Shape L2 [smaller]" in content
    assert "Field Corr [larger, >0.99]" in content
    assert "ms/step" in content
    assert "128.50" in content
    assert "## dipole" in content
    assert "## planewave" in content
    assert "| dipole_vacuum | demo | field_xy | Ez |" in content
    assert "| planewave_vacuum | demo 2 | field_xz | Ex |" in content
    assert "### dipole" in content
    assert "### planewave" in content
    assert "[material+source]" in content
    assert "[scalar comparison]" in content
    assert "## Scalar observables" in content
    assert "| dipole_vacuum | 1.50000000e+09 | S21 |" in content
    assert "## Per-frequency field metrics" in content
    assert "| dipole_vacuum | 1.50000000e+09 | 1.0000e-01 |" in content
    assert "| planewave_vacuum | 1.50000000e+09 | 2.0000e-01 |" in content


def test_material_source_plot_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    scene = build_scene("dipole_vacuum")
    output_path = benchmark_plotting.save_material_source_plot(
        scene=scene,
        scenario_name="dipole_vacuum",
    )
    assert output_path.exists()
    assert output_path == tmp_path / "dipole" / "dipole_vacuum" / "material_source.png"


def test_material_source_plot_resolves_mode_port_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    scene = build_scene("mode_port_straight_wg")
    assert not scene.sources
    assert scene.resolved_sources()

    output_path = benchmark_plotting.save_material_source_plot(
        scene=scene,
        scenario_name="mode_port_straight_wg",
    )

    assert output_path.exists()


def test_geometry_export_masks_detect_a_shifted_cone(monkeypatch):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    ).add_structure(
        mw.Structure(
            geometry=mw.Cone(
                position=(0.0, 0.0, -0.1),
                radius=0.12,
                height=0.2,
                axis="z",
            ),
            material=mw.Material(eps_r=2.0),
        )
    )
    maxwell_mask, exported_mask = benchmark_plotting._geometry_export_masks(scene)
    baseline_mismatch = np.count_nonzero(maxwell_mask ^ exported_mask)
    convert_geometry = benchmark_plotting._convert_geometry

    def shifted_convert_geometry(geometry, td, scale):
        return convert_geometry(geometry, td, scale).translated(0.2 * scale, 0.0, 0.0)

    monkeypatch.setattr(
        benchmark_plotting,
        "_convert_geometry",
        shifted_convert_geometry,
    )
    _, shifted_export = benchmark_plotting._geometry_export_masks(scene)

    assert np.count_nonzero(maxwell_mask ^ shifted_export) > baseline_mismatch


def test_prepare_tidy3d_benchmark_scene_preserves_material_regions():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cpu",
    ).add_material_region(
        mw.MaterialRegion(
            name="region",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            density=torch.ones((2, 2, 2), dtype=torch.float32),
            eps_bounds=(1.0, 4.0),
        )
    )

    trimmed = prepare_tidy3d_benchmark_scene(scene)

    assert trimmed is not scene
    assert trimmed.domain.bounds == scene.domain.bounds
    assert len(trimmed.material_regions) == 1
    assert trimmed.material_regions[0] == scene.material_regions[0]


def test_benchmark_scenes_keep_plane_monitors_inside_domain():
    for name in SCENARIOS:
        scene = build_scene(name)
        trimmed = prepare_tidy3d_benchmark_scene(scene)
        for monitor in trimmed.monitors:
            if not isinstance(monitor, mw.PlaneMonitor):
                continue
            axis_index = "xyz".index(monitor.axis)
            lo, hi = trimmed.domain.bounds[axis_index]
            assert lo <= monitor.position <= hi, (
                f"{name}:{monitor.name} is outside benchmark domain "
                f"({monitor.position} not in [{lo}, {hi}])"
            )


@pytest.mark.parametrize(
    ("geometry", "label"),
    [
        (mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)), "box"),
        (mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12), "sphere"),
    ],
    ids=["box", "sphere"],
)
def test_benchmark_scene_material_slices_match_exactly(geometry, label):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=25),
        device="cpu",
    ).add_structure(
        mw.Structure(geometry=geometry, material=mw.Material(eps_r=4.0), name=label)
    )
    trimmed = prepare_tidy3d_benchmark_scene(scene)
    prepared_scene = prepare_scene(scene)
    prepared_trimmed = prepare_scene(trimmed)

    scene_slice, scene_coords = benchmark_plotting.get_plane_slice_with_coords(
        prepared_scene,
        axis="z",
        position=0.0,
        values=prepared_scene.permittivity.detach().cpu().numpy(),
    )
    trimmed_slice, trimmed_coords = benchmark_plotting.get_plane_slice_with_coords(
        prepared_trimmed,
        axis="z",
        position=0.0,
        values=prepared_trimmed.permittivity.detach().cpu().numpy(),
    )
    np.testing.assert_allclose(scene_coords[0], trimmed_coords[0])
    np.testing.assert_allclose(scene_coords[1], trimmed_coords[1])
    np.testing.assert_allclose(scene_slice, trimmed_slice, atol=0.0, rtol=0.0)


def test_align_plane_monitor_fields_crops_to_physical_interior():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.2),
        boundary=mw.BoundarySpec.pml(num_layers=1),
        device="cpu",
    )

    coords = np.linspace(-0.4, 0.4, 5)
    interior = np.ones((3, 3), dtype=np.complex128)
    maxwell_field = np.pad(interior, pad_width=1, constant_values=5.0)
    tidy3d_field = np.pad(interior, pad_width=1, constant_values=7.0)
    monitor = {"axis": "z", "x": coords, "y": coords}

    aligned_maxwell, aligned_tidy3d = benchmark_runner._align_plane_monitor_fields(
        scene,
        monitor,
        monitor,
        component="Ex",
        maxwell_field=maxwell_field,
        tidy3d_field=tidy3d_field,
    )

    assert aligned_maxwell.shape == (3, 3)
    np.testing.assert_allclose(aligned_maxwell, interior)
    np.testing.assert_allclose(aligned_tidy3d, interior)


def test_tidy3d_plane_wave_source_is_exported_inside_physical_interior():
    scene = build_scene("planewave_vacuum")
    td_scene = prepare_tidy3d_benchmark_scene(scene)
    td_sim = td_scene.to_tidy3d(frequencies=(2.0e9,), run_time=1e-9)
    source = td_sim.sources[0]
    physical_bounds = benchmark_physical_bounds(scene)

    assert physical_bounds[2][0] * 1e6 <= source.center[2] <= physical_bounds[2][1] * 1e6
    assert np.isinf(source.size[0])
    assert np.isinf(source.size[1])
    assert source.size[2] == pytest.approx(0.0)


def test_field_comparison_plot_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    field = np.ones((8, 8))
    monitors = {
        f"plot_field_{axis}": {
            "fields": {component: field for component in ("Ex", "Ey", "Ez")}
        }
        for axis in ("x", "y", "z")
    }
    output_path = benchmark_plotting.save_field_comparison_plot(
        scenario_name="dipole_vacuum",
        maxwell_monitors=monitors,
        tidy3d_monitors=monitors,
    )
    assert output_path.exists()
    assert output_path == tmp_path / "dipole" / "dipole_vacuum" / "field_comparison.png"


def test_complex_field_diagnostic_plot_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    coords = np.linspace(-0.4, 0.4, 8)
    phase = np.exp(1j * np.linspace(0.0, np.pi, 8))[None, :]
    field = np.broadcast_to(phase, (8, 8))
    monitor = {
        "axis": "y",
        "frequencies": (2.0e9,),
        "x": coords,
        "z": coords,
        "fields": {"Ex": field},
    }
    scene = build_scene("planewave_vacuum")

    output_path = benchmark_plotting.save_complex_field_diagnostic_plot(
        scenario_name="planewave_vacuum",
        monitor_name="field_xz",
        component="Ex",
        maxwell_monitor=monitor,
        tidy3d_monitor=monitor,
        scene=scene,
    )

    assert output_path.exists()
    assert output_path == (
        tmp_path / "planewave" / "planewave_vacuum" / "complex_field_diagnostic.png"
    )


def test_spectral_field_diagnostic_plots_every_frequency(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)
    coords = np.linspace(-0.4, 0.4, 8)
    frequencies = (1.8e9, 2.0e9, 2.2e9)
    fields = np.stack(
        [np.full((8, 8), index + 1.0, dtype=np.complex128) for index in range(3)]
    )
    monitor = {
        "axis": "y",
        "frequencies": frequencies,
        "x": coords,
        "z": coords,
        "fields": {"Ex": fields},
    }
    metrics = [
        {
            "frequency": frequency,
            "field_l2": 0.1 * (index + 1),
            "field_shape_l2": 0.05 * (index + 1),
            "field_corr": 1.0 - 0.01 * index,
        }
        for index, frequency in enumerate(frequencies)
    ]
    selected_indices = []
    original_select = benchmark_plotting._select_plot_field

    def tracked_select(monitor_data, component, field_values, *, freq_index=0):
        selected_indices.append(freq_index)
        return original_select(
            monitor_data,
            component,
            field_values,
            freq_index=freq_index,
        )

    monkeypatch.setattr(benchmark_plotting, "_select_plot_field", tracked_select)
    output_path = benchmark_plotting.save_spectral_field_diagnostic_plot(
        scenario_name="modulated_slab",
        monitor_name="plot_field_y",
        component="Ex",
        maxwell_monitor=monitor,
        tidy3d_monitor=monitor,
        frequencies=frequencies,
        per_frequency=metrics,
        phase_factor=np.exp(-0.2j),
        scene=build_scene("modulated_slab"),
    )

    assert output_path.exists()
    assert output_path.name == "spectral_field_diagnostic.png"
    assert selected_indices == [0, 0, 1, 1, 2, 2]


def test_vector_field_comparison_plot_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)
    coords = np.linspace(-0.2, 0.2, 5)
    reference = np.ones((3, 5, 5), dtype=np.complex128)
    actual = 1.2 * np.exp(0.2j) * reference
    comparison = vector_field_comparison(actual, reference, coords=(coords, coords))

    output_path = benchmark_plotting.save_vector_field_comparison_plot(
        scenario_name="mode_source_higher_order",
        components=("Ex", "Ey", "Ez"),
        maxwell_vector=actual,
        reference_vector=reference,
        coords=(coords, coords),
        comparison=comparison,
    )

    assert output_path.exists()
    assert output_path.name == "vector_field_comparison.png"


def test_field_comparison_plot_smoke_with_multi_frequency_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    field = np.ones((2, 8, 8))
    monitors = {
        f"plot_field_{axis}": {
            "fields": {component: field for component in ("Ex", "Ey", "Ez")}
        }
        for axis in ("x", "y", "z")
    }
    output_path = benchmark_plotting.save_field_comparison_plot(
        scenario_name="dipole_two_freq",
        maxwell_monitors=monitors,
        tidy3d_monitors=monitors,
    )
    assert output_path.exists()
    assert output_path == tmp_path / "dipole" / "dipole_two_freq" / "field_comparison.png"


def test_field_comparison_plot_smoke_with_trailing_frequency_axis_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    field = np.stack([np.full((8, 8), value) for value in (1.0, 2.0, 3.0)], axis=-1)
    selected_fields = []
    original_plot_triplet = benchmark_plotting._plot_triplet

    def tracked_plot_triplet(*args, **kwargs):
        selected_fields.append(np.asarray(kwargs["left"]))
        return original_plot_triplet(*args, **kwargs)

    monkeypatch.setattr(benchmark_plotting, "_plot_triplet", tracked_plot_triplet)
    monitors = {
        f"plot_field_{axis}": {
            "axis": axis,
            "frequencies": (1.0, 2.0, 3.0),
            "fields": {component: field for component in ("Ex", "Ey", "Ez")},
            **({"y": np.linspace(-1.0, 1.0, 8), "z": np.linspace(-1.0, 1.0, 8)} if axis == "x" else {}),
            **({"x": np.linspace(-1.0, 1.0, 8), "z": np.linspace(-1.0, 1.0, 8)} if axis == "y" else {}),
            **({"x": np.linspace(-1.0, 1.0, 8), "y": np.linspace(-1.0, 1.0, 8)} if axis == "z" else {}),
        }
        for axis in ("x", "y", "z")
    }
    output_path = benchmark_plotting.save_field_comparison_plot(
        scenario_name="dipole_two_freq",
        maxwell_monitors=monitors,
        tidy3d_monitors=monitors,
        freq_index=2,
        frequency=3.0,
    )
    assert output_path.exists()
    assert output_path == tmp_path / "dipole" / "dipole_two_freq" / "field_comparison.png"
    assert selected_fields
    assert all(np.all(field_values == 3.0) for field_values in selected_fields)


def test_scenario_plot_dir_follows_scene_tree(tmp_path, monkeypatch):
    benchmark_paths.scenario_path_parts.cache_clear()
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    assert benchmark_paths.scenario_plot_dir("dipole_vacuum") == tmp_path / "dipole" / "dipole_vacuum"
    assert benchmark_paths.scenario_plot_dir("planewave_vacuum") == tmp_path / "planewave" / "planewave_vacuum"


def test_scenario_cache_path_follows_scene_tree(tmp_path, monkeypatch):
    benchmark_paths.scenario_path_parts.cache_clear()
    monkeypatch.setattr(benchmark_paths, "CACHE_DIR", tmp_path)

    assert benchmark_paths.scenario_cache_path("dipole_vacuum") == tmp_path / "dipole" / "dipole_vacuum.h5"
    assert benchmark_paths.scenario_cache_path("planewave_vacuum") == tmp_path / "planewave" / "planewave_vacuum.h5"
