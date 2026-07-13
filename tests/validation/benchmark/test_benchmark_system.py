from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark import cache as benchmark_cache
from benchmark import plotting as benchmark_plotting
from benchmark import report as benchmark_report
from benchmark import runner as benchmark_runner
from benchmark import paths as benchmark_paths
from benchmark.metrics import align_arrays, align_plane_fields, phase_align_field, significant_field_mask
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
        assert len(scene.sources) == 1
        assert len(scene.monitors) >= 5


def test_benchmark_cache_key_supports_geometry_and_source_objects():
    scene = build_scene("dielectric_slab")
    cache_key = benchmark_runner._benchmark_cache_key(scene, frequencies=(2.0e9,), run_time_factor=15.0)
    assert isinstance(cache_key, str)
    assert len(cache_key) == 64


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
        },
        cache_key="demo-cache-key",
    )
    assert output_path == tmp_path / "dipole" / "dipole_vacuum.h5"
    loaded = benchmark_cache.load_tidy3d_result("dipole_vacuum", expected_cache_key="demo-cache-key")
    np.testing.assert_allclose(loaded["field_xy"]["fields"]["Ez"], field)
    np.testing.assert_allclose(loaded["flux_pos_z"]["flux"], np.array([1.2]))


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
        field_linf=0.2,
        field_corr=0.99,
        flux_error=0.05,
        compared_monitor="field_xy",
        compared_component="Ez",
        material_source_plot=Path("plots/dipole/material_source.png"),
        field_plot=Path("plots/dipole/field_comparison.png"),
        updated_at="2026-03-16T12:00:00-07:00",
    )
    planewave_result = ScenarioMetrics(
        name="planewave_vacuum",
        description="demo 2",
        frequencies=(1.5e9,),
        maxwell_time_s=2.34,
        tidy3d_cache_hit=False,
        field_l2=0.2,
        field_linf=0.3,
        field_corr=0.98,
        flux_error=0.15,
        compared_monitor="field_xz",
        compared_component="Ex",
        material_source_plot=Path("plots/planewave/material_source.png"),
        field_plot=Path("plots/planewave/field_comparison.png"),
        updated_at="2026-03-16T12:10:00-07:00",
    )
    benchmark_report.write_results_markdown([dipole_result, planewave_result])
    content = output_path.read_text(encoding="utf-8")
    assert "## Metric Guide" in content
    assert "Field L2 [smaller, <1e-1]" in content
    assert "Field Corr [larger, >0.99]" in content
    assert "## dipole" in content
    assert "## planewave" in content
    assert "| dipole_vacuum | demo | field_xy | Ez |" in content
    assert "| planewave_vacuum | demo 2 | field_xz | Ex |" in content
    assert "### dipole" in content
    assert "### planewave" in content
    assert "[material+source]" in content


def test_material_source_plot_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "PLOTS_DIR", tmp_path)

    scene = build_scene("dipole_vacuum")
    tidy_scene = scene
    output_path = benchmark_plotting.save_material_source_plot(
        scene=scene,
        tidy_scene=tidy_scene,
        scenario_name="dipole_vacuum",
    )
    assert output_path.exists()
    assert output_path == tmp_path / "dipole" / "dipole_vacuum" / "material_source.png"


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

    field = np.ones((8, 8, 2))
    monitors = {
        f"plot_field_{axis}": {
            "axis": axis,
            "frequencies": (1.0, 2.0),
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
    )
    assert output_path.exists()
    assert output_path == tmp_path / "dipole" / "dipole_two_freq" / "field_comparison.png"


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
