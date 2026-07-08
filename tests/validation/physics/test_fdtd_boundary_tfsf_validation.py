from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pytest
import torch

import witwin.maxwell as mw


_BOUNDARY_DOMAIN_BOUNDS = ((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))
_TFSF_DOMAIN_BOUNDS = ((-0.96, 0.96), (-0.96, 0.96), (-0.96, 0.96))
_TFSF_BOUNDS = ((-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32))
_OBLIQUE_TFSF_DIRECTION = np.asarray((1.0, 0.25, 0.15), dtype=np.float64)
_OBLIQUE_TFSF_DIRECTION /= np.linalg.norm(_OBLIQUE_TFSF_DIRECTION)
_OBLIQUE_TFSF_POLARIZATION = np.asarray((0.0, 0.514495755, -0.857492925), dtype=np.float64)
_OBLIQUE_TFSF_POLARIZATION /= np.linalg.norm(_OBLIQUE_TFSF_POLARIZATION)
_OBLIQUE_TFSF_PROBE_STEP = 0.12
_GRATING_TFSF_DOMAIN_BOUNDS = ((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))
_GRATING_TFSF_Z_BOUNDS = (-0.24, 0.24)


def _tests_root() -> Path:
    root = Path(__file__).resolve()
    while root.name != "tests" and root.parent != root:
        root = root.parent
    if root.name != "tests":
        raise RuntimeError("Unable to locate tests root directory.")
    return root


_OUTPUT_DIR = _tests_root() / "test_output" / "validation" / "boundary_tfsf"


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _complex_point_vector(point_monitor: dict[str, object], components: tuple[str, ...]) -> np.ndarray:
    return np.asarray([complex(_to_numpy(point_monitor[component]).item()) for component in components])


def _complex_vector_alignment_error(field: np.ndarray, target: np.ndarray) -> float:
    denominator = max(float(np.vdot(target, target).real), 1e-12)
    scale = np.vdot(target, field) / denominator
    return float(np.linalg.norm(field - scale * target) / max(np.linalg.norm(field), 1e-12))


def _component_volume(result, component: str):
    solver = result.solver
    field = _to_numpy(result.tensor(component))
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


def _save_plane_magnitude_plot(
    *,
    field,
    coord_a,
    coord_b,
    axis: str,
    output_name: str,
    title: str,
    overlay_bounds=None,
):
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coord_a = np.asarray(coord_a, dtype=np.float64)
    coord_b = np.asarray(coord_b, dtype=np.float64)
    array = np.abs(np.asarray(field))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        array.T,
        origin="lower",
        extent=[coord_a[0], coord_a[-1], coord_b[0], coord_b[-1]],
        cmap="hot",
        aspect="equal",
    )
    plt.colorbar(image, ax=ax, shrink=0.85, label="|E|")
    ax.set_title(title)
    plane_axes = {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }
    xlabel, ylabel = plane_axes[axis]
    ax.set_xlabel(f"{xlabel} [m]")
    ax.set_ylabel(f"{ylabel} [m]")

    if overlay_bounds is not None:
        (a0, a1), (b0, b1) = overlay_bounds
        rect = Rectangle(
            (a0, b0),
            a1 - a0,
            b1 - b0,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            edgecolor="cyan",
        )
        ax.add_patch(rect)

    output_path = _OUTPUT_DIR / output_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_line_plot(*, coords, values, output_name: str, title: str, xlabel: str, ylabel: str):
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(coords, values, linewidth=1.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    output_path = _OUTPUT_DIR / output_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _boundary_scene(boundary: mw.BoundarySpec):
    scene = mw.Scene(
        domain=mw.Domain(bounds=_BOUNDARY_DOMAIN_BOUNDS),
        grid=mw.GridSpec.uniform(0.15),
        boundary=boundary,
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=25.0),
            name="src",
        )
    )
    return scene


def _mixed_periodic_pml_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.9, 0.9), (-0.6, 0.6), (-0.9, 0.9))),
        grid=mw.GridSpec.uniform(0.15),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            y="periodic",
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, -0.55, 0.0),
            polarization="Ez",
            width=0.12,
            source_time=mw.CW(frequency=1.0e9, amplitude=25.0),
            name="src",
        )
    )
    return scene


def _pmc_symmetry_pair():
    half_scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.15),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        symmetry=("PMC", None, None),
        device="cuda",
    )
    half_scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=25.0),
            name="src",
        )
    )

    full_scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.45, 0.599999), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.15),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    full_scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=25.0),
            name="src",
        )
    )
    return half_scene, full_scene


def _cpml_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=100.0),
            name="src",
        )
    )
    for name, position in (
        ("center", (0.0, 0.0, 0.0)),
        ("inner_x", (0.24, 0.0, 0.0)),
        ("pml_mid_x", (0.40, 0.0, 0.0)),
        ("pml_outer_x", (0.56, 0.0, 0.0)),
    ):
        scene.add_monitor(mw.PointMonitor(name, position, fields=("Ez",)))
    return scene


def _tfsf_scene(*, with_scatterer: bool):
    scene = mw.Scene(
        domain=mw.Domain(bounds=_TFSF_DOMAIN_BOUNDS),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_pw",
        )
    )
    if with_scatterer:
        scene.add_structure(
            mw.Structure(
                name="sphere",
                geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12),
                material=mw.Material(eps_r=4.0),
            )
    )
    return scene


def _oblique_tfsf_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=_TFSF_DOMAIN_BOUNDS),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=tuple(float(value) for value in _OBLIQUE_TFSF_DIRECTION),
            polarization=tuple(float(value) for value in _OBLIQUE_TFSF_POLARIZATION),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_pw_oblique",
        )
    )
    for label, sign in (("upstream", -0.5), ("center", 0.0), ("downstream", 0.5)):
        position = tuple(float(value) for value in (sign * _OBLIQUE_TFSF_PROBE_STEP * _OBLIQUE_TFSF_DIRECTION))
        scene.add_monitor(
            mw.PointMonitor(
                name=f"probe_{label}",
                position=position,
                fields=("Ex", "Ey", "Ez"),
            )
        )
    return scene


def _grating_tfsf_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=_GRATING_TFSF_DOMAIN_BOUNDS),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector="auto",
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=_GRATING_TFSF_Z_BOUNDS),
            name="grating_tfsf",
        )
    )
    return scene


def _run_boundary_result(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
        absorber="cpml",
    ).run()


def _run_tfsf_result(scene, *, steady_cycles=6, transient_cycles=15):
    return mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()


def _run_grating_tfsf_result(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=64),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
        absorber="cpml",
    ).run()


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


@lru_cache(maxsize=None)
def _periodic_summary():
    scene = _boundary_scene(mw.BoundarySpec.periodic())
    result = _run_boundary_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="periodic_ez_xy.png",
        title="Periodic Boundary: |Ez| at z=0",
    )
    summary = {
        "ez": ez,
        "face_error": float(np.linalg.norm(ez[-1] - ez[0]) / max(np.linalg.norm(ez[0]), 1e-12)),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _bloch_summary():
    wavevector_x = np.pi / 1.2
    scene = _boundary_scene(mw.BoundarySpec.bloch((wavevector_x, 0.0, 0.0)))
    result = _run_boundary_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    domain_length_x = _BOUNDARY_DOMAIN_BOUNDS[0][1] - _BOUNDARY_DOMAIN_BOUNDS[0][0]
    phase = np.exp(1j * wavevector_x * domain_length_x)
    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="bloch_ez_xy.png",
        title="Bloch Boundary: |Ez| at z=0",
    )
    summary = {
        "ez": ez,
        "phase_error": float(np.linalg.norm(ez[-1] - phase * ez[0]) / max(np.linalg.norm(ez[0]), 1e-12)),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _pec_summary():
    scene = _boundary_scene(mw.BoundarySpec.pec())
    result = _run_boundary_result(scene)
    ex, x_ex, y_ex, z_ex = _component_volume(result, "Ex")
    ey, x_ey, y_ey, z_ey = _component_volume(result, "Ey")
    ez, x_ez, y_ez, z_ez = _component_volume(result, "Ez")
    boundary_max = max(
        np.max(np.abs(ex[:, 0, :])),
        np.max(np.abs(ex[:, -1, :])),
        np.max(np.abs(ey[0, :, :])),
        np.max(np.abs(ey[-1, :, :])),
        np.max(np.abs(ez[0, :, :])),
        np.max(np.abs(ez[-1, :, :])),
    )
    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, :, mid],
        coord_a=x_ez,
        coord_b=y_ez,
        axis="z",
        output_name="pec_ez_xy.png",
        title="PEC Boundary: |Ez| at z=0",
    )
    summary = {
        "boundary_max": float(boundary_max),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _pmc_symmetry_summary():
    half_scene, full_scene = _pmc_symmetry_pair()
    half_result = _run_boundary_result(half_scene)
    full_result = _run_boundary_result(full_scene)
    half_ez, x, y, z = _component_volume(half_result, "Ez")
    component_errors = {}
    for component in ("Ex", "Ey", "Ez"):
        half_field, *_ = _component_volume(half_result, component)
        full_field, *_ = _component_volume(full_result, component)
        full_positive_half = full_field[-half_field.shape[0] :]
        component_errors[component] = float(
            np.linalg.norm(half_field - full_positive_half) / max(np.linalg.norm(full_positive_half), 1e-12)
        )
    mid = half_ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=half_ez[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="pmc_symmetry_ez_xy.png",
        title="PMC Symmetry: |Ez| at z=0",
    )
    summary = {
        "component_errors": component_errors,
        "relative_error": component_errors["Ez"],
        "max_component_error": max(component_errors.values()),
        "plot_path": plot_path,
    }
    del half_result
    del full_result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _mixed_periodic_pml_summary():
    scene = _mixed_periodic_pml_scene()
    result = _run_boundary_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="mixed_periodic_pml_ez_xy.png",
        title="Mixed Boundary (Periodic + PML): |Ez| at z=0",
    )
    summary = {
        "face_error": float(
            np.linalg.norm(ez[:, -1, :] - ez[:, 0, :]) / max(np.linalg.norm(ez[:, 0, :]), 1e-12)
        ),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _cpml_summary():
    scene = _cpml_scene()
    result = _run_boundary_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    monitors = {}
    for name in ("center", "inner_x", "pml_mid_x", "pml_outer_x"):
        monitors[name] = abs(result.monitor(name)["data"])

    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z)))
    centerline = np.abs(ez[:, iy, iz])
    plot_path = _save_line_plot(
        coords=x,
        values=centerline,
        output_name="cpml_ez_centerline_x.png",
        title="CPML: |Ez(x,0,0)|",
        xlabel="x [m]",
        ylabel="|Ez|",
    )
    summary = {
        "center": float(monitors["center"]),
        "inner": float(monitors["inner_x"]),
        "mid": float(monitors["pml_mid_x"]),
        "outer": float(monitors["pml_outer_x"]),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _tfsf_null_summary():
    scene = _tfsf_scene(with_scatterer=False)
    result = _run_tfsf_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
        np.abs(ez),
        x,
        y,
        z,
        _TFSF_BOUNDS,
        dx=result.solver.scene.dx,
        dy=result.solver.scene.dy,
        dz=result.solver.scene.dz,
    )
    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="tfsf_null_ez_xy.png",
        title="TFSF Null Scene: |Ez| at z=0",
        overlay_bounds=(_TFSF_BOUNDS[0], _TFSF_BOUNDS[1]),
    )
    summary = {
        "ez": ez,
        "x": x,
        "y": y,
        "z": z,
        "leakage_ratio": float(leakage_ratio),
        "inside_max": float(inside_max),
        "outside_max": float(outside_max),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _tfsf_oblique_summary():
    scene = _oblique_tfsf_scene()
    result = _run_tfsf_result(scene, steady_cycles=12, transient_cycles=25)
    ez, x, y, z = _component_volume(result, "Ez")
    component_ratios = {}
    component_inside_max = {}
    component_outside_max = {}
    for component in ("Ey", "Ez"):
        field, x_coords, y_coords, z_coords = _component_volume(result, component)
        leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
            np.abs(field),
            x_coords,
            y_coords,
            z_coords,
            _TFSF_BOUNDS,
            dx=result.solver.scene.dx,
            dy=result.solver.scene.dy,
            dz=result.solver.scene.dz,
        )
        component_ratios[component] = float(leakage_ratio)
        component_inside_max[component] = float(inside_max)
        component_outside_max[component] = float(outside_max)

    polarization_residuals = {}
    transversality_errors = {}
    ex_fractions = {}
    for label in ("upstream", "center", "downstream"):
        point_monitor = result.monitor(f"probe_{label}")
        field_vector = _complex_point_vector(point_monitor, ("Ex", "Ey", "Ez"))
        polarization_residuals[label] = _complex_vector_alignment_error(field_vector, _OBLIQUE_TFSF_POLARIZATION)
        transversality_errors[label] = float(
            abs(np.vdot(_OBLIQUE_TFSF_DIRECTION, field_vector)) / max(np.linalg.norm(field_vector), 1e-12)
        )
        ex_fractions[label] = float(abs(field_vector[0]) / max(np.linalg.norm(field_vector[1:]), 1e-12))

    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="tfsf_oblique_ez_xy.png",
        title="Oblique TFSF: |Ez| at z=0",
        overlay_bounds=(_TFSF_BOUNDS[0], _TFSF_BOUNDS[1]),
    )
    summary = {
        "component_ratios": component_ratios,
        "component_inside_max": component_inside_max,
        "component_outside_max": component_outside_max,
        "polarization_residuals": polarization_residuals,
        "transversality_errors": transversality_errors,
        "ex_fractions": ex_fractions,
        "max_leakage_ratio": max(component_ratios.values()),
        "max_polarization_residual": max(polarization_residuals.values()),
        "max_transversality_error": max(transversality_errors.values()),
        "max_ex_fraction": max(ex_fractions.values()),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _tfsf_scatter_summary():
    scene = _tfsf_scene(with_scatterer=True)
    result = _run_tfsf_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    leakage_ratio, inside_max, outside_max = _guarded_region_ratio(
        np.abs(ez),
        x,
        y,
        z,
        _TFSF_BOUNDS,
        dx=result.solver.scene.dx,
        dy=result.solver.scene.dy,
        dz=result.solver.scene.dz,
    )
    null_summary = _tfsf_null_summary()
    difference = np.abs(ez - null_summary["ez"])
    mid = ez.shape[2] // 2
    plot_path = _save_plane_magnitude_plot(
        field=difference[:, :, mid],
        coord_a=x,
        coord_b=y,
        axis="z",
        output_name="tfsf_scatter_difference_ez_xy.png",
        title="TFSF Scatter Scene: |Ez_scatter - Ez_null| at z=0",
        overlay_bounds=(_TFSF_BOUNDS[0], _TFSF_BOUNDS[1]),
    )
    summary = {
        "outside_max": float(outside_max),
        "inside_max": float(inside_max),
        "leakage_ratio": float(leakage_ratio),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _grating_tfsf_summary():
    scene = _grating_tfsf_scene()
    result = _run_grating_tfsf_result(scene)
    ez, x, y, z = _component_volume(result, "Ez")
    mid_y = int(np.argmin(np.abs(y)))
    plot_path = _save_plane_magnitude_plot(
        field=ez[:, mid_y, :],
        coord_a=x,
        coord_b=z,
        axis="y",
        output_name="grating_tfsf_xy_bloch_z_pml.png",
        title="Grating TFSF (x/y Bloch + z PML): |Ez|",
        overlay_bounds=(_GRATING_TFSF_DOMAIN_BOUNDS[0], _GRATING_TFSF_Z_BOUNDS),
    )
    summary = {
        "ez_max": float(np.max(np.abs(ez))),
        "finite": bool(np.isfinite(np.abs(ez)).all()),
        "plot_path": plot_path,
    }
    del result
    torch.cuda.empty_cache()
    return summary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_periodic_boundary_validation_matches_opposite_faces_and_saves_field_plot():
    summary = _periodic_summary()
    assert summary["face_error"] < 1.0e-6
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_bloch_boundary_validation_matches_expected_phase_and_saves_field_plot():
    summary = _bloch_summary()
    assert summary["phase_error"] < 1.0e-6
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_pec_boundary_validation_enforces_tangential_electric_node_and_saves_field_plot():
    summary = _pec_summary()
    assert summary["boundary_max"] < 1.0e-7
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_pmc_symmetry_validation_matches_full_domain_reference_and_saves_field_plot():
    summary = _pmc_symmetry_summary()
    assert summary["relative_error"] < 5.0e-2
    assert summary["max_component_error"] < 7.0e-2
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_mixed_periodic_pml_boundary_validation_preserves_periodic_axis_and_saves_field_plot():
    summary = _mixed_periodic_pml_summary()
    assert summary["face_error"] < 1.0e-6
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_cpml_boundary_validation_shows_monotone_attenuation_into_absorber_and_saves_centerline_plot():
    summary = _cpml_summary()
    assert summary["center"] > 0.0
    assert summary["inner"] < summary["center"]
    assert summary["mid"] < summary["inner"]
    assert summary["outer"] < summary["mid"]
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_tfsf_null_scene_validation_keeps_leakage_small_and_saves_field_plot():
    summary = _tfsf_null_summary()
    assert summary["inside_max"] > 0.0
    assert summary["outside_max"] < summary["inside_max"]
    assert summary["leakage_ratio"] < 1.0e-3
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_tfsf_oblique_scene_validation_keeps_leakage_bounded_and_saves_field_plot():
    summary = _tfsf_oblique_summary()
    for component in ("Ey", "Ez"):
        assert summary["component_inside_max"][component] > 0.0
        assert summary["component_outside_max"][component] < summary["component_inside_max"][component]
    assert summary["max_leakage_ratio"] < 1.2e-3
    assert summary["max_polarization_residual"] < 5.0e-2
    assert summary["max_transversality_error"] < 1.5e-2
    assert summary["max_ex_fraction"] < 2.0e-3
    assert Path(summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_tfsf_scatter_scene_validation_produces_scattered_field_outside_box_and_saves_difference_plot():
    null_summary = _tfsf_null_summary()
    scatter_summary = _tfsf_scatter_summary()
    assert scatter_summary["outside_max"] > null_summary["outside_max"] * 2.0
    assert scatter_summary["outside_max"] < scatter_summary["inside_max"]
    assert Path(scatter_summary["plot_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for boundary/TFSF validation")
def test_grating_tfsf_validation_saves_xy_bloch_z_pml_field_plot():
    summary = _grating_tfsf_summary()
    assert summary["finite"] is True
    assert summary["ez_max"] > 0.0
    assert Path(summary["plot_path"]).exists()
