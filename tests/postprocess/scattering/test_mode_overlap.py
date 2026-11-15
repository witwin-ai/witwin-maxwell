from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import sample_mode_source_component, solve_mode_source_profile
from witwin.maxwell.postprocess import compute_mode_overlap
from witwin.maxwell.postprocess.stratton_chu import build_plane_points
from witwin.maxwell.result import Result
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler import compile_materials


def _mode_scene():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.none(),
                structures=(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.24, 0.32)),
                material=mw.Material(eps_r=12.0),
                name='core',
            ),
            ),
                sources=(
            mw.ModeSource(
                position=(0.0, 0.0, 0.0),
                # Aperture edges land exactly on grid nodes (+-0.24); a 0.56 span
                # would place them exactly midway between nodes, where the nearest-
                # node resolution is an ill-conditioned floating-point tie.
                size=(0.0, 0.48, 0.48),
                polarization="Ez",
                source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
                name="port0",
            ),
            ),
            )
    return scene


def _mode_context(scene):
    prepared_scene = prepare_scene(scene, device="cpu")
    return SimpleNamespace(
        scene=prepared_scene,
        dx=prepared_scene.dx,
        dy=prepared_scene.dy,
        dz=prepared_scene.dz,
        Ex=torch.empty((1,), device=prepared_scene.device, dtype=torch.float32),
        c=299792458.0,
        boundary_kind=prepared_scene.scene.boundary.kind,
        _compiled_material_model=compile_materials(prepared_scene),
    )


def _synthetic_mode_monitor(*, reverse_magnetic: bool = False):
    scene = _mode_scene()
    source = scene.sources[0]
    context = _mode_context(scene)
    compiled_source = _compile_mode_source(source, default_frequency=1.0e9)
    mode_data = solve_mode_source_profile(context, compiled_source)
    points = build_plane_points(
        "x",
        0.0,
        mode_data["coords_u"],
        mode_data["coords_v"],
    )
    magnetic_sign = -1.0 if reverse_magnetic else 1.0
    ez_field = sample_mode_source_component(mode_data, points, "Ez").to(dtype=torch.complex64)
    hy_field = magnetic_sign * sample_mode_source_component(mode_data, points, "Hy").to(dtype=torch.complex64)
    mode_monitor = mw.ModeMonitor(
        "port",
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.48, 0.48),
        mode_index=0,
        direction="+",
        polarization="Ez",
        frequencies=(1.0e9,),
    )
    monitors = {
        "port": {
            "kind": "plane",
            "monitor_type": "mode",
            "fields": ("Ez", "Hy"),
            "frequency": 1.0e9,
            "frequencies": (1.0e9,),
            "axis": "x",
            "position": 0.0,
            "normal_direction": "+",
            "mode_spec": mode_monitor.mode_spec(),
            "y": mode_data["coords_u"],
            "z": mode_data["coords_v"],
            "Ez": ez_field,
            "Hy": hy_field,
            "data": ez_field,
            "component": "ez",
        }
    }
    return Result(
        method="fdtd",
        scene=scene,
        frequency=1.0e9,
        monitors=monitors,
    )


def _fdtd_mode_overlap_scene():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                structures=(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.24, 0.24)),
                material=mw.Material(eps_r=12.0),
                name='core',
            ),
            ),
                sources=(
            mw.ModeSource(
                position=(-0.32, 0.0, 0.0),
                size=(0.0, 0.48, 0.48),
                polarization="Ez",
                source_time=mw.CW(frequency=1.0e9, amplitude=50.0),
                name="port0",
            ),
            ),
                monitors=(
            mw.ModeMonitor(
                "port",
                position=(-0.24, 0.0, 0.0),
                size=(0.0, 0.48, 0.48),
                polarization="Ez",
                direction="+",
                frequencies=(1.0e9,),
            ),
            ),
            )
    return scene


def test_compute_mode_overlap_recovers_forward_mode_from_matching_plane_fields():
    result = _synthetic_mode_monitor(reverse_magnetic=False)

    overlap = compute_mode_overlap(result, "port")

    assert overlap["effective_index"] > 1.0
    assert torch.real(overlap["amplitude_forward"]) == pytest.approx(1.0, rel=5e-4, abs=5e-4)
    assert torch.imag(overlap["amplitude_forward"]) == pytest.approx(0.0, abs=1e-5)
    assert torch.abs(overlap["amplitude_backward"]) == pytest.approx(0.0, abs=1e-5)
    assert overlap["power_fraction_forward"] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert overlap["candidate_diagnostics"] is not None
    assert overlap["candidate_overlap_matrix"] is not None
    assert overlap["candidate_diagnostics"][overlap["selected_candidate_index"]]["selected"]


def test_compute_mode_overlap_separates_backward_mode_when_magnetic_field_is_reversed():
    result = _synthetic_mode_monitor(reverse_magnetic=True)

    overlap = compute_mode_overlap(result, "port")

    assert torch.abs(overlap["amplitude_forward"]) == pytest.approx(0.0, abs=1e-5)
    assert torch.real(overlap["amplitude_backward"]) == pytest.approx(1.0, rel=5e-4, abs=5e-4)
    assert torch.imag(overlap["amplitude_backward"]) == pytest.approx(0.0, abs=1e-5)
    assert overlap["power_fraction_backward"] == pytest.approx(1.0, rel=1e-3, abs=1e-3)


def test_result_mode_monitor_returns_mode_amplitudes_and_raw_plane():
    result = _synthetic_mode_monitor(reverse_magnetic=False)

    monitor = result.monitor("port", freq_index=0)
    raw_monitor = result.raw_monitor("port")

    assert isinstance(monitor, mw.ModeAmplitudeData) and monitor.dims == ("mode", "propagation")
    assert torch.real(monitor.isel(mode=0).sel(propagation="+").values) == pytest.approx(1.0, rel=5e-4, abs=5e-4)
    assert raw_monitor["kind"] == "plane"
    assert raw_monitor["monitor_type"] == "mode"
    assert raw_monitor["mode_spec"]["mode_index"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_compute_mode_overlap_fdtd_monitor_direction_switch_swaps_forward_and_backward_amplitudes():
    result = mw.Simulation.fdtd(
        _fdtd_mode_overlap_scene(),
        frequencies=[1.0e9],
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.fdtd.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ),
    ).run()

    plus = compute_mode_overlap(result, "port", direction="+")
    minus = compute_mode_overlap(result, "port", direction="-")
    plus_forward = complex(plus["amplitude_forward"].cpu().item())
    plus_backward = complex(plus["amplitude_backward"].cpu().item())
    minus_forward = complex(minus["amplitude_forward"].cpu().item())
    minus_backward = complex(minus["amplitude_backward"].cpu().item())

    assert plus["effective_index"] > 1.0
    assert bool(torch.isfinite(plus["amplitude_forward"]).item())
    assert bool(torch.isfinite(plus["amplitude_backward"]).item())
    assert abs(plus_forward) > 0.1 or abs(plus_backward) > 0.1
    assert plus_forward == pytest.approx(minus_backward, rel=1e-5, abs=1e-5)
    assert plus_backward == pytest.approx(minus_forward, rel=1e-5, abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_mode_port_materializes_first_class_source_and_monitor_results():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                structures=(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.24, 0.24)),
                material=mw.Material(eps_r=12.0),
                name='core',
            ),
            ),
                ports=(
            mw.ModePort(
                "port0",
                position=(-0.32, 0.0, 0.0),
                size=(0.0, 0.48, 0.48),
                polarization="Ez",
                source_time=mw.CW(frequency=1.0e9, amplitude=50.0),
                frequencies=(1.0e9,),
                monitor_offset=0.08,
            ),
            ),
            )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.fdtd.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ),
    ).run()
    monitor = result.monitor("port0", freq_index=0)
    raw_monitor = result.raw_monitor("port0")

    assert isinstance(monitor, mw.ModeAmplitudeData)
    assert raw_monitor["kind"] == "plane"
    assert raw_monitor["monitor_type"] == "mode"
    assert monitor.attrs["effective_index"] > 1.0
    assert bool(torch.isfinite(monitor.values).all())
