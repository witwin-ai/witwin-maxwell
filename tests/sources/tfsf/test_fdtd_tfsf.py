import numpy as np
import pytest
import torch

import witwin.maxwell as mw


TFSF_BOUNDS = ((-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32))


def _build_scene(source, *, structures=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.96, 0.96), (-0.96, 0.96), (-0.96, 0.96))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    for structure in structures or []:
        scene.add_structure(structure)
    scene.add_source(source)
    return scene


def _run_tfsf(scene, *, frequency=1.0e9, steady_cycles=6, transient_cycles=15):
    return mw.Simulation.fdtd(
        scene,
        frequency=frequency,
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()


def _get_raw_ez(result):
    return _get_raw_component(result, "Ez")


def _get_raw_component(result, component):
    solver = result.solver
    raw_field = result.raw_output[component]
    if isinstance(raw_field, torch.Tensor):
        field = torch.abs(raw_field).detach().cpu().numpy()
    else:
        field = np.abs(raw_field)
    if component == "Ex":
        x = np.linspace(
            solver.scene.domain_range[0] + 0.5 * solver.scene.dx,
            solver.scene.domain_range[1] - 0.5 * solver.scene.dx,
            field.shape[0],
        )
        y = np.linspace(solver.scene.domain_range[2], solver.scene.domain_range[3], field.shape[1])
        z = np.linspace(solver.scene.domain_range[4], solver.scene.domain_range[5], field.shape[2])
        return field, x, y, z
    if component == "Ey":
        x = np.linspace(solver.scene.domain_range[0], solver.scene.domain_range[1], field.shape[0])
        y = np.linspace(
            solver.scene.domain_range[2] + 0.5 * solver.scene.dy,
            solver.scene.domain_range[3] - 0.5 * solver.scene.dy,
            field.shape[1],
        )
        z = np.linspace(solver.scene.domain_range[4], solver.scene.domain_range[5], field.shape[2])
        return field, x, y, z
    x = np.linspace(solver.scene.domain_range[0], solver.scene.domain_range[1], field.shape[0])
    y = np.linspace(solver.scene.domain_range[2], solver.scene.domain_range[3], field.shape[1])
    z = np.linspace(
        solver.scene.domain_range[4] + 0.5 * solver.scene.dz,
        solver.scene.domain_range[5] - 0.5 * solver.scene.dz,
        field.shape[2],
    )
    return field, x, y, z


def _guarded_region_ratio(field, x_coords, y_coords, z_coords, bounds, *, dx, dy, dz):
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    inside = (
        (xx >= bounds[0][0])
        & (xx <= bounds[0][1])
        & (yy >= bounds[1][0])
        & (yy <= bounds[1][1])
        & (zz >= bounds[2][0])
        & (zz <= bounds[2][1])
    )
    outside = (
        (xx < bounds[0][0] - dx)
        | (xx > bounds[0][1] + dx)
        | (yy < bounds[1][0] - dy)
        | (yy > bounds[1][1] + dy)
        | (zz < bounds[2][0] - dz)
        | (zz > bounds[2][1] + dz)
    )
    inside_max = float(np.max(field[inside]))
    outside_max = float(np.max(field[outside]))
    return outside_max / max(inside_max, 1e-12), inside_max, outside_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
@pytest.mark.parametrize(
    ("direction", "polarization", "component"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "Ez"),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "Ey"),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "Ez"),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "Ex"),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "Ey"),
    ],
)
def test_tfsf_axis_aligned_plane_wave_null_leakage_is_small(direction, polarization, component):
    scene = _build_scene(
        mw.PlaneWave(
            direction=direction,
            polarization=polarization,
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=TFSF_BOUNDS),
            name="tfsf_pw",
        )
    )

    result = _run_tfsf(scene)
    field, x_coords, y_coords, z_coords = _get_raw_component(result, component)
    leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
        field,
        x_coords,
        y_coords,
        z_coords,
        TFSF_BOUNDS,
        dx=result.solver.scene.dx,
        dy=result.solver.scene.dy,
        dz=result.solver.scene.dz,
    )

    assert inside_max > 0.0
    assert outside_max < inside_max
    assert leakage_ratio < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
@pytest.mark.parametrize("dft_frequency", [0.7e9, 1.0e9, 1.3e9])
@pytest.mark.parametrize(
    ("direction", "polarization", "component"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "Ez"),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "Ez"),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "Ex"),
    ],
)
def test_tfsf_axis_aligned_gaussian_pulse_null_leakage_is_small_across_dft_frequencies(
    dft_frequency,
    direction,
    polarization,
    component,
):
    scene = _build_scene(
        mw.PlaneWave(
            direction=direction,
            polarization=polarization,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.35e9, amplitude=80.0),
            injection=mw.TFSF(bounds=TFSF_BOUNDS),
            name="tfsf_pw_pulse",
        )
    )

    result = _run_tfsf(
        scene,
        frequency=dft_frequency,
        steady_cycles=6,
        transient_cycles=20,
    )
    field, x_coords, y_coords, z_coords = _get_raw_component(result, component)
    leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
        field,
        x_coords,
        y_coords,
        z_coords,
        TFSF_BOUNDS,
        dx=result.solver.scene.dx,
        dy=result.solver.scene.dy,
        dz=result.solver.scene.dz,
    )

    assert inside_max > 0.0
    assert outside_max < inside_max
    assert leakage_ratio < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
def test_tfsf_oblique_plane_wave_keeps_leakage_bounded():
    scene = _build_scene(
        mw.PlaneWave(
            direction=(1.0, 0.25, 0.15),
            polarization=(0.0, 0.514495755, -0.857492925),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=TFSF_BOUNDS),
            name="tfsf_pw_oblique",
        )
    )

    result = _run_tfsf(scene, steady_cycles=12, transient_cycles=25)
    dominant_ratios = []
    for component in ("Ey", "Ez"):
        field, x_coords, y_coords, z_coords = _get_raw_component(result, component)
        leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
            field,
            x_coords,
            y_coords,
            z_coords,
            TFSF_BOUNDS,
            dx=result.solver.scene.dx,
            dy=result.solver.scene.dy,
            dz=result.solver.scene.dz,
        )
        assert inside_max > 0.0
        assert outside_max < inside_max
        dominant_ratios.append(leakage_ratio)

    assert max(dominant_ratios) < 1.2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
@pytest.mark.xfail(
    reason="Gaussian-beam TFSF still relies on the analytical patch provider until the discrete face engine is completed.",
    strict=False,
)
def test_tfsf_gaussian_beam_null_leakage_is_small():
    scene = _build_scene(
        mw.GaussianBeam(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            beam_waist=0.18,
            focus=(0.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=120.0),
            injection=mw.TFSF(bounds=TFSF_BOUNDS),
            name="tfsf_beam",
        )
    )

    result = _run_tfsf(scene)
    ez, x_coords, y_coords, z_coords = _get_raw_ez(result)
    leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
        ez,
        x_coords,
        y_coords,
        z_coords,
        TFSF_BOUNDS,
        dx=result.solver.scene.dx,
        dy=result.solver.scene.dy,
        dz=result.solver.scene.dz,
    )

    assert inside_max > 0.0
    assert outside_max < inside_max
    assert leakage_ratio < 8e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
def test_tfsf_scatterer_generates_scattered_field_outside_box():
    scatterer = mw.Structure(
        name="sphere",
        geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12),
        material=mw.Material(eps_r=4.0),
    )
    null_scene = _build_scene(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=TFSF_BOUNDS),
            name="tfsf_null",
        )
    )
    scatter_scene = _build_scene(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=TFSF_BOUNDS),
            name="tfsf_scatter",
        ),
        structures=[scatterer],
    )

    null_result = _run_tfsf(null_scene)
    scatter_result = _run_tfsf(scatter_scene)
    null_field, x_coords, y_coords, z_coords = _get_raw_ez(null_result)
    scatter_field, _, _, _ = _get_raw_ez(scatter_result)
    _, _, null_outside_max = _guarded_region_ratio(
        null_field,
        x_coords,
        y_coords,
        z_coords,
        TFSF_BOUNDS,
        dx=null_result.solver.scene.dx,
        dy=null_result.solver.scene.dy,
        dz=null_result.solver.scene.dz,
    )
    _, _, scatter_outside_max = _guarded_region_ratio(
        scatter_field,
        x_coords,
        y_coords,
        z_coords,
        TFSF_BOUNDS,
        dx=scatter_result.solver.scene.dx,
        dy=scatter_result.solver.scene.dy,
        dz=scatter_result.solver.scene.dz,
    )

    assert scatter_outside_max > null_outside_max * 2.0
