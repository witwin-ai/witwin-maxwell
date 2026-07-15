from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import witwin.maxwell as mw
from benchmark.metrics import (
    align_arrays,
    align_plane_fields,
    phase_align_field,
    plane_coord_keys,
    significant_field_mask,
)
from benchmark.paths import ensure_directories, scenario_plot_dir
from benchmark.tidy3d_scene import benchmark_physical_bounds
from witwin.maxwell.adapters.tidy3d import _M_TO_UM, _convert_geometry
from witwin.maxwell.fdtd.excitation.spatial import resolve_injection_axis, soft_plane_wave_index
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.sources import PlaneWave, PointDipole


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PLOT_AXES = ("x", "y", "z")
FIELD_COMPONENTS = ("Ex", "Ey", "Ez")
PLANE_COORD_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}


def plane_axes(axis: str) -> tuple[str, str]:
    if axis == "x":
        return "y", "z"
    if axis == "y":
        return "x", "z"
    return "x", "y"


def plane_slice_coords(scene: mw.Scene, axis: str) -> tuple[np.ndarray, np.ndarray]:
    scene = prepare_scene(scene)
    coord_names = plane_axes(axis)
    coord_map = {
        "x": scene.x.detach().cpu().numpy(),
        "y": scene.y.detach().cpu().numpy(),
        "z": scene.z.detach().cpu().numpy(),
    }
    return coord_map[coord_names[0]], coord_map[coord_names[1]]


def get_plane_slice_with_coords(
    scene: mw.Scene,
    *,
    axis: str,
    position: float,
    values,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    scene = prepare_scene(scene)
    axis_idx = AXIS_INDEX[axis]
    coords = {
        "x": scene.x.detach().cpu().numpy(),
        "y": scene.y.detach().cpu().numpy(),
        "z": scene.z.detach().cpu().numpy(),
    }
    plane_index = int(np.argmin(np.abs(coords[axis] - float(position))))
    array = np.asarray(values)
    coord_names = plane_axes(axis)
    if axis_idx == 0:
        return np.asarray(array[plane_index, :, :]), (coords[coord_names[0]], coords[coord_names[1]])
    if axis_idx == 1:
        return np.asarray(array[:, plane_index, :]), (coords[coord_names[0]], coords[coord_names[1]])
    return np.asarray(array[:, :, plane_index]), (coords[coord_names[0]], coords[coord_names[1]])


def get_plane_slice(scene: mw.Scene, *, axis: str, position: float, values) -> np.ndarray:
    plane_slice, _ = get_plane_slice_with_coords(scene, axis=axis, position=position, values=values)
    return plane_slice


def _first_resolved_source(scene: mw.Scene):
    sources = scene.resolved_sources() if hasattr(scene, "resolved_sources") else scene.sources
    return sources[0] if sources else None


def dominant_source_component(scene: mw.Scene) -> str:
    source = _first_resolved_source(scene)
    if isinstance(source, (PointDipole, PlaneWave)):
        component_index = max(range(3), key=lambda idx: abs(source.polarization[idx]))
        return FIELD_COMPONENTS[component_index]
    return "Ez"


def source_plane_map(
    scene: mw.Scene,
    *,
    axis: str,
    position: float,
    component: str,
    source=None,
) -> np.ndarray:
    source = _first_resolved_source(scene) if source is None else source
    scene = prepare_scene(scene)
    values = np.zeros((scene.Nx, scene.Ny, scene.Nz), dtype=np.float64)
    target_axis = component[-1].lower()
    if isinstance(source, PointDipole):
        width = max(float(source.width), min(scene.dx, scene.dy, scene.dz))
        xx = scene.xx.detach().cpu().numpy()
        yy = scene.yy.detach().cpu().numpy()
        zz = scene.zz.detach().cpu().numpy()
        profile = np.exp(
            -(
                (xx - source.position[0]) ** 2
                + (yy - source.position[1]) ** 2
                + (zz - source.position[2]) ** 2
            )
            / (2.0 * width * width)
        )
        component_index = max(range(3), key=lambda idx: abs(source.polarization[idx]))
        if target_axis == "xyz"[component_index]:
            values = profile
    elif isinstance(source, PlaneWave):
        inject_axis = resolve_injection_axis(source.direction, source.injection_axis)
        direction_component = float(source.direction[AXIS_INDEX[inject_axis]])
        plane_idx = soft_plane_wave_index(scene, inject_axis, direction_component)
        if inject_axis == "x":
            values[plane_idx, :, :] = 1.0
        elif inject_axis == "y":
            values[:, plane_idx, :] = 1.0
        else:
            values[:, :, plane_idx] = 1.0
        component_index = max(range(3), key=lambda idx: abs(source.polarization[idx]))
        if target_axis != "xyz"[component_index]:
            values.fill(0.0)
    return get_plane_slice(scene, axis=axis, position=position, values=values)


def _crop_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = align_arrays(np.asarray(a), np.asarray(b))
    return np.asarray(a), np.asarray(b)


def _align_plane_pair(
    source_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    source_coords: tuple[np.ndarray, np.ndarray],
    reference_coords: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        aligned_source, aligned_reference, _ = align_plane_fields(
            source_values,
            reference_values,
            source_coords=source_coords,
            reference_coords=reference_coords,
        )
        return np.asarray(aligned_source), np.asarray(aligned_reference)
    except ValueError:
        return _crop_pair(source_values, reference_values)


def _align_monitor_pair(
    maxwell_monitor: dict,
    tidy3d_monitor: dict,
    *,
    component: str,
    maxwell_field: np.ndarray,
    tidy3d_field: np.ndarray,
    maxwell_coords: tuple[np.ndarray, np.ndarray] | None = None,
    tidy3d_coords: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        if maxwell_coords is None:
            maxwell_coords = _component_plane_coords(maxwell_monitor, component)
        if tidy3d_coords is None:
            tidy3d_coords = _component_plane_coords(tidy3d_monitor, component)
        if maxwell_coords is None or tidy3d_coords is None:
            raise ValueError("Missing plane coordinates.")
        maxwell_coord_keys = PLANE_COORD_NAMES.get(maxwell_monitor.get("axis"), tuple())
        tidy3d_coord_keys = PLANE_COORD_NAMES.get(tidy3d_monitor.get("axis"), tuple())
        if not maxwell_coord_keys:
            maxwell_coord_keys = plane_coord_keys(maxwell_monitor)
        if not tidy3d_coord_keys:
            tidy3d_coord_keys = plane_coord_keys(tidy3d_monitor)
        if maxwell_coord_keys == tidy3d_coord_keys:
            aligned_maxwell, aligned_tidy3d, _ = align_plane_fields(
                maxwell_field,
                tidy3d_field,
                source_coords=maxwell_coords,
                reference_coords=tidy3d_coords,
            )
            return np.asarray(aligned_maxwell), np.asarray(aligned_tidy3d)
    except ValueError:
        pass
    return _crop_pair(maxwell_field, tidy3d_field)


def _component_plane_coords(monitor_data: dict, component: str) -> tuple[np.ndarray, np.ndarray] | None:
    field_coords = monitor_data.get("field_coords", {})
    component_coords = field_coords.get(component)
    if component_coords is not None:
        if "axis" in component_coords and component_coords["axis"] in PLANE_COORD_NAMES:
            coord_keys = PLANE_COORD_NAMES[component_coords["axis"]]
        else:
            coord_keys = plane_coord_keys(component_coords)
        return tuple(np.asarray(component_coords[key], dtype=np.float64) for key in coord_keys)

    axis = monitor_data.get("axis")
    if axis in PLANE_COORD_NAMES:
        coord_keys = PLANE_COORD_NAMES[axis]
        if all(key in monitor_data for key in coord_keys):
            return tuple(np.asarray(monitor_data[key], dtype=np.float64) for key in coord_keys)
        return None

    try:
        coord_keys = plane_coord_keys(monitor_data)
    except ValueError:
        return None
    return tuple(np.asarray(monitor_data[key], dtype=np.float64) for key in coord_keys)


def _select_plot_field(monitor_data: dict, component: str, field_values, *, freq_index: int = 0) -> np.ndarray:
    array = np.asarray(field_values).squeeze()
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise TypeError(f"Expected a 2D plane field or stacked multi-frequency planes, got shape {array.shape}.")

    index = int(freq_index)
    coords = _component_plane_coords(monitor_data, component)
    plane_shape = tuple(coord.size for coord in coords) if coords is not None else None
    frequencies = tuple(float(freq) for freq in monitor_data.get("frequencies", ()))

    if plane_shape is not None:
        if array.shape[-2:] == plane_shape:
            if index < 0 or index >= array.shape[0]:
                raise IndexError(f"freq_index {index} is out of range for field shape {array.shape}.")
            return np.asarray(array[index])
        if array.shape[:2] == plane_shape:
            if index < 0 or index >= array.shape[-1]:
                raise IndexError(f"freq_index {index} is out of range for field shape {array.shape}.")
            return np.asarray(array[..., index])

    if frequencies:
        if array.shape[0] == len(frequencies):
            return np.asarray(array[index])
        if array.shape[-1] == len(frequencies):
            return np.asarray(array[..., index])

    if 0 <= index < array.shape[0]:
        return np.asarray(array[index])
    if 0 <= index < array.shape[-1]:
        return np.asarray(array[..., index])

    raise TypeError(
        "Unable to identify the frequency axis for plane field "
        f"shape {array.shape} and component {component!r}."
    )


def _plot_triplet(
    ax_left,
    ax_mid,
    ax_right,
    *,
    left,
    mid,
    title_prefix: str,
    cmap: str,
    reference_label: str = "Tidy3D",
) -> None:
    left = np.asarray(left)
    mid = np.asarray(mid)
    diff = np.abs(np.abs(left) - np.abs(mid))
    left_peak = float(np.max(np.abs(left)))
    mid_peak = float(np.max(np.abs(mid)))
    vmax = max(left_peak, mid_peak, 1e-12)
    diff_max = max(float(np.max(diff)), 1e-12)

    image_left = ax_left.imshow(np.abs(left).T, origin="lower", cmap=cmap, aspect="equal", vmin=0.0, vmax=vmax)
    ax_left.set_title(f"{title_prefix} Maxwell\npeak={left_peak:.2e}", fontsize=8)
    plt.colorbar(image_left, ax=ax_left, shrink=0.72)

    image_mid = ax_mid.imshow(np.abs(mid).T, origin="lower", cmap=cmap, aspect="equal", vmin=0.0, vmax=vmax)
    ax_mid.set_title(f"{title_prefix} {reference_label}\npeak={mid_peak:.2e}", fontsize=8)
    plt.colorbar(image_mid, ax=ax_mid, shrink=0.72)

    image_diff = ax_right.imshow(diff.T, origin="lower", cmap="viridis", aspect="equal", vmin=0.0, vmax=diff_max)
    ax_right.set_title(
        f"{title_prefix} ||Maxwell|-|{reference_label}||\npeak={float(np.max(diff)):.2e}",
        fontsize=8,
    )
    plt.colorbar(image_diff, ax=ax_right, shrink=0.72)


def _plot_map(axis, values, *, title: str, cmap: str, vmin=None, vmax=None) -> None:
    image = axis.imshow(
        np.asarray(values).T,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title, fontsize=8)
    plt.colorbar(image, ax=axis, shrink=0.72)


def _geometry_export_masks(scene: mw.Scene) -> tuple[np.ndarray, np.ndarray]:
    """Sample public and Tidy3D-exported geometry envelopes on one Maxwell grid."""
    import tidy3d as td

    prepared = prepare_scene(scene.clone(device="cpu"))
    geometries = tuple(structure.geometry for structure in prepared.structures) + tuple(
        region.geometry for region in prepared.material_regions
    )
    maxwell_mask = np.zeros((prepared.Nx, prepared.Ny, prepared.Nz), dtype=bool)
    tidy3d_mask = np.zeros_like(maxwell_mask)
    x = prepared.x.detach().cpu().numpy() * _M_TO_UM
    y = prepared.y.detach().cpu().numpy() * _M_TO_UM
    z = prepared.z.detach().cpu().numpy() * _M_TO_UM

    for geometry in geometries:
        signed_distance = geometry.signed_distance(prepared.xx, prepared.yy, prepared.zz)
        maxwell_mask |= signed_distance.detach().cpu().numpy() <= 0.0
        exported_geometry = _convert_geometry(geometry, td, _M_TO_UM)
        tidy3d_mask |= np.asarray(
            exported_geometry.inside_meshgrid(x=x, y=y, z=z),
            dtype=bool,
        )
    return maxwell_mask, tidy3d_mask


def _take_plane_window(values, indices_a: np.ndarray, indices_b: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    windowed = np.take(array, indices_a, axis=-2)
    windowed = np.take(windowed, indices_b, axis=-1)
    return windowed


def _crop_monitor_field_to_physical_bounds(
    scene: mw.Scene,
    *,
    axis: str,
    values,
    coords: tuple[np.ndarray, np.ndarray] | None,
):
    if coords is None:
        return np.asarray(values), coords

    coord_names = PLANE_COORD_NAMES[axis]
    physical_bounds = benchmark_physical_bounds(scene)
    tangential_bounds = tuple(physical_bounds["xyz".index(coord_name)] for coord_name in coord_names)
    coords_a = np.asarray(coords[0], dtype=np.float64)
    coords_b = np.asarray(coords[1], dtype=np.float64)
    tolerance_a = 1e-7 * max(1.0, float(np.max(np.abs(coords_a))), *map(abs, tangential_bounds[0]))
    tolerance_b = 1e-7 * max(1.0, float(np.max(np.abs(coords_b))), *map(abs, tangential_bounds[1]))
    mask_a = (coords_a >= tangential_bounds[0][0] - tolerance_a) & (coords_a <= tangential_bounds[0][1] + tolerance_a)
    mask_b = (coords_b >= tangential_bounds[1][0] - tolerance_b) & (coords_b <= tangential_bounds[1][1] + tolerance_b)
    if not np.any(mask_a) or not np.any(mask_b):
        return np.asarray(values), (coords_a, coords_b)

    indices_a = np.flatnonzero(mask_a)
    indices_b = np.flatnonzero(mask_b)
    return _take_plane_window(values, indices_a, indices_b), (coords_a[indices_a], coords_b[indices_b])


def save_material_source_plot(
    *,
    scene: mw.Scene,
    scenario_name: str,
) -> Path:
    maxwell_source = _first_resolved_source(scene)
    source_component = dominant_source_component(scene)
    scene = prepare_scene(scene)
    ensure_directories()
    scenario_dir = scenario_plot_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / "material_source.png"

    maxwell_eps = scene.permittivity.detach().cpu().numpy()
    maxwell_geometry, tidy3d_geometry = _geometry_export_masks(scene)
    geometry_mismatch = np.logical_xor(maxwell_geometry, tidy3d_geometry)
    fig, axes = plt.subplots(len(PLOT_AXES), 5, figsize=(20, 12))
    fig.suptitle(
        f"{scenario_name}: compiled material, exported geometry, and source",
        fontsize=16,
    )

    for row, axis in enumerate(PLOT_AXES):
        maxwell_eps_slice = get_plane_slice(
            scene,
            axis=axis,
            position=0.0,
            values=maxwell_eps,
        )
        maxwell_geometry_slice = get_plane_slice(
            scene,
            axis=axis,
            position=0.0,
            values=maxwell_geometry,
        )
        tidy3d_geometry_slice = get_plane_slice(
            scene,
            axis=axis,
            position=0.0,
            values=tidy3d_geometry,
        )
        mismatch_slice = get_plane_slice(
            scene,
            axis=axis,
            position=0.0,
            values=geometry_mismatch,
        )

        maxwell_source_slice = source_plane_map(
            scene,
            axis=axis,
            position=0.0,
            component=source_component,
            source=maxwell_source,
        )

        _plot_map(
            axes[row, 0],
            maxwell_eps_slice,
            title=f"{axis}=0 Maxwell compiled eps_r",
            cmap="viridis",
        )
        _plot_map(
            axes[row, 1],
            maxwell_geometry_slice,
            title=f"{axis}=0 Maxwell geometry envelope",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
        )
        _plot_map(
            axes[row, 2],
            tidy3d_geometry_slice,
            title=f"{axis}=0 Tidy3D exported geometry envelope",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
        )
        _plot_map(
            axes[row, 3],
            mismatch_slice,
            title=(
                f"{axis}=0 geometry XOR\n"
                f"fraction={float(np.mean(mismatch_slice)):.3e}"
            ),
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
        )
        _plot_map(
            axes[row, 4],
            maxwell_source_slice,
            title=f"{axis}=0 Maxwell source stencil ({source_component})",
            cmap="magma",
        )

        axis_a, axis_b = plane_axes(axis)
        for col in range(5):
            axes[row, col].set_xlabel(axis_a)
            axes[row, col].set_ylabel(axis_b)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_field_comparison_plot(
    *,
    scenario_name: str,
    maxwell_monitors: dict[str, dict],
    tidy3d_monitors: dict[str, dict],
    scene: mw.Scene | None = None,
    reference_label: str = "Tidy3D",
) -> Path:
    ensure_directories()
    scenario_dir = scenario_plot_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / "field_comparison.png"

    fig, axes = plt.subplots(len(PLOT_AXES), len(FIELD_COMPONENTS) * 3, figsize=(30, 12))
    fig.suptitle(f"{scenario_name}: Ex/Ey/Ez on x/y/z planes", fontsize=16)

    for row, axis in enumerate(PLOT_AXES):
        monitor_name = f"plot_field_{axis}"
        axis_a, axis_b = plane_axes(axis)
        for component_index, component in enumerate(FIELD_COMPONENTS):
            col = component_index * 3
            maxwell_monitor = maxwell_monitors[monitor_name]
            tidy3d_monitor = tidy3d_monitors[monitor_name]
            maxwell_field = _select_plot_field(
                maxwell_monitor,
                component,
                maxwell_monitor["fields"][component],
                freq_index=0,
            )
            tidy3d_field = _select_plot_field(
                tidy3d_monitor,
                component,
                tidy3d_monitor["fields"][component],
                freq_index=0,
            )
            maxwell_coords = _component_plane_coords(maxwell_monitor, component)
            tidy3d_coords = _component_plane_coords(tidy3d_monitor, component)
            if scene is not None:
                maxwell_field, maxwell_coords = _crop_monitor_field_to_physical_bounds(
                    scene,
                    axis=axis,
                    values=maxwell_field,
                    coords=maxwell_coords,
                )
                tidy3d_field, tidy3d_coords = _crop_monitor_field_to_physical_bounds(
                    scene,
                    axis=axis,
                    values=tidy3d_field,
                    coords=tidy3d_coords,
                )
            maxwell_field, tidy3d_field = _align_monitor_pair(
                maxwell_monitor=maxwell_monitor,
                tidy3d_monitor=tidy3d_monitor,
                component=component,
                maxwell_field=maxwell_field,
                tidy3d_field=tidy3d_field,
                maxwell_coords=maxwell_coords,
                tidy3d_coords=tidy3d_coords,
            )
            _plot_triplet(
                axes[row, col],
                axes[row, col + 1],
                axes[row, col + 2],
                left=maxwell_field,
                mid=tidy3d_field,
                title_prefix=f"{axis}=0 {component}",
                cmap="hot",
                reference_label=reference_label,
            )
            for idx in range(3):
                axes[row, col + idx].set_xlabel(axis_a)
                axes[row, col + idx].set_ylabel(axis_b)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_vector_field_comparison_plot(
    *,
    scenario_name: str,
    components: tuple[str, ...],
    maxwell_vector: np.ndarray,
    reference_vector: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray],
    comparison: dict[str, object],
    reference_label: str = "Tidy3D",
) -> Path:
    """Plot one globally aligned electric-vector cross section and its residual."""
    ensure_directories()
    scenario_dir = scenario_plot_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / "vector_field_comparison.png"

    aligned = np.asarray(maxwell_vector) * complex(comparison["shape_scale"])
    reference = np.asarray(reference_vector)
    extent = (
        float(coords[0][0]),
        float(coords[0][-1]),
        float(coords[1][0]),
        float(coords[1][-1]),
    )
    row_labels = ("|E|",) + tuple(f"Re({name})" for name in components)
    left_rows = [np.sqrt(np.sum(np.abs(aligned) ** 2, axis=0))]
    reference_rows = [np.sqrt(np.sum(np.abs(reference) ** 2, axis=0))]
    left_rows.extend(np.real(aligned[index]) for index in range(len(components)))
    reference_rows.extend(np.real(reference[index]) for index in range(len(components)))

    fig, axes = plt.subplots(len(row_labels), 3, figsize=(13.5, 3.5 * len(row_labels)))
    for row, (label, left, right) in enumerate(zip(row_labels, left_rows, reference_rows)):
        difference = np.abs(left - right)
        if row == 0:
            vmax = max(float(np.max(left)), float(np.max(right)), 1.0e-30)
            left_limits = {"vmin": 0.0, "vmax": vmax, "cmap": "magma"}
        else:
            vmax = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0e-30)
            left_limits = {"vmin": -vmax, "vmax": vmax, "cmap": "RdBu_r"}
        images = (
            axes[row, 0].imshow(
                left.T, origin="lower", extent=extent, aspect="equal", **left_limits
            ),
            axes[row, 1].imshow(
                right.T, origin="lower", extent=extent, aspect="equal", **left_limits
            ),
            axes[row, 2].imshow(
                difference.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="viridis",
                vmin=0.0,
                vmax=max(float(np.max(difference)), 1.0e-30),
            ),
        )
        axes[row, 0].set_title(f"{label} Maxwell (one vector scale)")
        axes[row, 1].set_title(f"{label} {reference_label}")
        axes[row, 2].set_title(f"{label} absolute residual")
        for axis, image in zip(axes[row], images):
            axis.set_xlabel("transverse coordinate 1 (m)")
            axis.set_ylabel("transverse coordinate 2 (m)")
            plt.colorbar(image, ax=axis, shrink=0.78)

    fig.suptitle(
        f"{scenario_name}: electric-vector cross section | "
        f"overlap={float(comparison['overlap']):.4f}, "
        f"energy ratio={float(comparison['energy_ratio']):.4f}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_complex_field_diagnostic_plot(
    *,
    scenario_name: str,
    monitor_name: str,
    component: str,
    maxwell_monitor: dict,
    tidy3d_monitor: dict,
    reference_label: str = "Tidy3D",
    scene: mw.Scene,
    freq_index: int = 0,
) -> Path:
    """Plot complex-field slices and center lines on one shared coordinate grid."""
    ensure_directories()
    scenario_dir = scenario_plot_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / "complex_field_diagnostic.png"

    axis = maxwell_monitor.get("axis") or tidy3d_monitor.get("axis")
    if axis not in PLANE_COORD_NAMES:
        raise ValueError(f"Monitor {monitor_name!r} does not define a plane axis.")

    maxwell_field = _select_plot_field(
        maxwell_monitor,
        component,
        maxwell_monitor["fields"][component],
        freq_index=freq_index,
    )
    tidy3d_field = _select_plot_field(
        tidy3d_monitor,
        component,
        tidy3d_monitor["fields"][component],
        freq_index=freq_index,
    )
    maxwell_coords = _component_plane_coords(maxwell_monitor, component)
    tidy3d_coords = _component_plane_coords(tidy3d_monitor, component)
    maxwell_field, maxwell_coords = _crop_monitor_field_to_physical_bounds(
        scene, axis=axis, values=maxwell_field, coords=maxwell_coords
    )
    tidy3d_field, tidy3d_coords = _crop_monitor_field_to_physical_bounds(
        scene, axis=axis, values=tidy3d_field, coords=tidy3d_coords
    )
    if maxwell_coords is None or tidy3d_coords is None:
        raise ValueError(f"Monitor {monitor_name!r} is missing component coordinates.")
    maxwell_field, tidy3d_field, coords = align_plane_fields(
        maxwell_field,
        tidy3d_field,
        source_coords=maxwell_coords,
        reference_coords=tidy3d_coords,
    )
    support = significant_field_mask(tidy3d_field)
    phase_aligned_maxwell, phase_factor = phase_align_field(
        maxwell_field,
        tidy3d_field,
        mask=support,
    )

    coord_a, coord_b = coords
    extent = (float(coord_a[0]), float(coord_a[-1]), float(coord_b[0]), float(coord_b[-1]))
    difference = phase_aligned_maxwell - tidy3d_field
    magnitude_max = max(float(np.max(np.abs(maxwell_field))), float(np.max(np.abs(tidy3d_field))), 1e-12)
    real_max = max(float(np.max(np.abs(maxwell_field.real))), float(np.max(np.abs(tidy3d_field.real))), 1e-12)
    diff_max = max(float(np.max(np.abs(difference))), 1e-12)
    masked_maxwell_phase = np.where(support, np.angle(maxwell_field), np.nan)
    masked_tidy3d_phase = np.where(support, np.angle(tidy3d_field), np.nan)
    masked_phase_difference = np.where(
        support,
        np.angle(phase_aligned_maxwell * np.conj(tidy3d_field)),
        np.nan,
    )

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"{scenario_name}: {monitor_name}/{component} complex-field diagnostic", fontsize=15)

    maps = (
        (np.abs(maxwell_field), "Maxwell |E|", "viridis", 0.0, magnitude_max),
        (np.abs(tidy3d_field), f"{reference_label} |E|", "viridis", 0.0, magnitude_max),
        (np.abs(difference), f"|phase-aligned Maxwell - {reference_label}|", "magma", 0.0, diff_max),
        (phase_aligned_maxwell.real, "Phase-aligned Maxwell Re(E)", "RdBu_r", -real_max, real_max),
        (tidy3d_field.real, f"{reference_label} Re(E)", "RdBu_r", -real_max, real_max),
        (difference.real, "Aligned real-field difference", "RdBu_r", -diff_max, diff_max),
        (masked_maxwell_phase, "Maxwell phase (excited support)", "twilight", -np.pi, np.pi),
        (masked_tidy3d_phase, f"{reference_label} phase (excited support)", "twilight", -np.pi, np.pi),
        (masked_phase_difference, "Aligned phase difference (excited support)", "twilight", -np.pi, np.pi),
    )
    for plot_axis, (values, title, cmap, vmin, vmax) in zip(axes[:, :3].flat, maps):
        image = plot_axis.imshow(
            np.asarray(values).T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plot_axis.set_title(title)
        plot_axis.set_xlabel(PLANE_COORD_NAMES[axis][0])
        plot_axis.set_ylabel(PLANE_COORD_NAMES[axis][1])
        fig.colorbar(image, ax=plot_axis, shrink=0.78)

    center_index = maxwell_field.shape[0] // 2
    line_coord = coord_b
    maxwell_line = phase_aligned_maxwell[center_index, :]
    tidy3d_line = tidy3d_field[center_index, :]
    support_line = support[center_index, :]
    axes[0, 3].plot(line_coord, np.abs(maxwell_line), label="Maxwell")
    axes[0, 3].plot(line_coord, np.abs(tidy3d_line), label=reference_label)
    axes[0, 3].set_title("Center-line magnitude")
    axes[0, 3].set_ylabel(f"|{component}|")
    axes[0, 3].legend()

    axes[1, 3].plot(line_coord, maxwell_line.real, label="Maxwell")
    axes[1, 3].plot(line_coord, tidy3d_line.real, label=reference_label)
    axes[1, 3].set_title("Center-line real field")
    axes[1, 3].set_ylabel(f"Re({component})")

    maxwell_unwrapped = np.unwrap(np.angle(maxwell_line))
    tidy3d_unwrapped = np.unwrap(np.angle(tidy3d_line))
    if np.any(support_line):
        branch_offset = 2.0 * np.pi * np.round(
            np.median(maxwell_unwrapped[support_line] - tidy3d_unwrapped[support_line])
            / (2.0 * np.pi)
        )
        maxwell_unwrapped = maxwell_unwrapped - branch_offset
    maxwell_line_phase = np.where(support_line, maxwell_unwrapped, np.nan)
    tidy3d_line_phase = np.where(support_line, tidy3d_unwrapped, np.nan)
    axes[2, 3].plot(line_coord, maxwell_line_phase, label="Maxwell (aligned)")
    axes[2, 3].plot(line_coord, tidy3d_line_phase, label=reference_label)
    axes[2, 3].set_title("Center-line phase (excited support)")
    axes[2, 3].set_ylabel("phase (rad)")
    for plot_axis in axes[:, 3]:
        plot_axis.set_xlabel(PLANE_COORD_NAMES[axis][1])
        plot_axis.grid(alpha=0.25)
    axes[2, 3].legend()
    fig.text(
        0.5,
        0.01,
        f"Global phase factor applied to Maxwell: {np.angle(phase_factor):+.4f} rad",
        ha="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_scalar_comparison_plot(
    *,
    scenario_name: str,
    scalar_metrics: list[dict[str, object]],
) -> Path | None:
    """Plot complex spectra or paired real scalar observables from both solvers."""
    observable_names = list(
        dict.fromkeys(str(item["observable"]) for item in scalar_metrics)
    )
    if not observable_names:
        return None

    ensure_directories()
    scenario_dir = scenario_plot_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / "scalar_comparison.png"

    complex_names = [name for name in observable_names if name.upper().startswith("S")]
    if not complex_names:
        rows_by_name = {
            observable: sorted(
                (item for item in scalar_metrics if item["observable"] == observable),
                key=lambda item: float(item["frequency"]),
            )
            for observable in observable_names
        }
        series_names = [name for name, rows in rows_by_name.items() if len(rows) > 1]
        scalar_names = [name for name, rows in rows_by_name.items() if len(rows) == 1]
        panel_count = int(bool(series_names)) + int(bool(scalar_names))
        fig, panel_axes = plt.subplots(panel_count, 1, figsize=(11, 4.5 * panel_count))
        axes = np.atleast_1d(panel_axes)
        panel_index = 0

        if series_names:
            axis = axes[panel_index]
            panel_index += 1
            all_frequencies = [
                float(item["frequency"])
                for name in series_names
                for item in rows_by_name[name]
            ]
            frequency_scale = 1.0e6 if max(all_frequencies) < 1.0e9 else 1.0e9
            frequency_unit = "MHz" if frequency_scale == 1.0e6 else "GHz"
            for observable in series_names:
                rows = rows_by_name[observable]
                frequencies = np.asarray([float(item["frequency"]) for item in rows])
                maxwell = np.asarray([complex(item["maxwell"]).real for item in rows])
                tidy3d = np.asarray([complex(item["tidy3d"]).real for item in rows])
                label = observable.replace("_", " ")
                axis.plot(
                    frequencies / frequency_scale,
                    maxwell,
                    marker="o",
                    label=f"Maxwell {label}",
                )
                axis.plot(
                    frequencies / frequency_scale,
                    tidy3d,
                    marker="x",
                    linestyle="--",
                    label=f"Tidy3D {label}",
                )
            axis.set_xlabel(f"frequency ({frequency_unit})")
            axis.set_ylabel("value")
            axis.grid(alpha=0.25)
            axis.legend(ncol=2, fontsize=8)

        if scalar_names:
            axis = axes[panel_index]
            maxwell = np.asarray(
                [complex(rows_by_name[name][0]["maxwell"]).real for name in scalar_names]
            )
            tidy3d = np.asarray(
                [complex(rows_by_name[name][0]["tidy3d"]).real for name in scalar_names]
            )
            labels = [name.replace("_", " ") for name in scalar_names]
            ylabel = "value"
            if scalar_names == ["resonance_frequency"]:
                maxwell = maxwell / 1.0e6
                tidy3d = tidy3d / 1.0e6
                labels = ["resonance frequency"]
                ylabel = "frequency (MHz)"
            elif all(name.startswith("eta_") for name in scalar_names):
                ylabel = "diffraction efficiency"
            elif all(name.startswith("rcs_") for name in scalar_names):
                ylabel = "RCS (m^2)"
            elif set(scalar_names) == {
                "D_max",
                "D_peak_theta",
                "beam_width_E",
                "beam_width_H",
            }:
                physical_maxwell = maxwell.copy()
                physical_tidy3d = tidy3d.copy()
                scale = np.maximum(np.maximum(np.abs(maxwell), np.abs(tidy3d)), 1.0e-15)
                maxwell = maxwell / scale
                tidy3d = tidy3d / scale
                labels = [
                    f"{label}\nM={m:.4g}, T={t:.4g}"
                    for label, m, t in zip(labels, physical_maxwell, physical_tidy3d)
                ]
                ylabel = "per-observable normalized value"

            positions = np.arange(len(scalar_names), dtype=np.float64)
            width = 0.38
            axis.bar(positions - width / 2.0, maxwell, width, label="Maxwell")
            axis.bar(positions + width / 2.0, tidy3d, width, label="Tidy3D")
            axis.set_xticks(positions, labels, rotation=20, ha="right")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.legend()

        fig.suptitle(f"{scenario_name}: scalar observables", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output_path

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{scenario_name}: complex modal observables", fontsize=14)

    for observable in complex_names:
        rows = sorted(
            (item for item in scalar_metrics if item["observable"] == observable),
            key=lambda item: float(item["frequency"]),
        )
        frequencies_ghz = np.asarray([float(item["frequency"]) for item in rows]) / 1.0e9
        maxwell = np.asarray([complex(item["maxwell"]) for item in rows])
        tidy3d = np.asarray([complex(item["tidy3d"]) for item in rows])
        axes[0].plot(frequencies_ghz, np.abs(maxwell), marker="o", label=f"Maxwell {observable}")
        axes[0].plot(
            frequencies_ghz,
            np.abs(tidy3d),
            marker="x",
            linestyle="--",
            label=f"Tidy3D {observable}",
        )
        axes[1].plot(
            frequencies_ghz,
            np.unwrap(np.angle(maxwell)),
            marker="o",
            label=f"Maxwell {observable}",
        )
        axes[1].plot(
            frequencies_ghz,
            np.unwrap(np.angle(tidy3d)),
            marker="x",
            linestyle="--",
            label=f"Tidy3D {observable}",
        )

    axes[0].set_ylabel("magnitude")
    axes[1].set_ylabel("unwrapped phase (rad)")
    axes[1].set_xlabel("frequency (GHz)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_time_trace_comparison_plot(
    *,
    scenario_name: str,
    traces: list[dict[str, object]],
) -> Path:
    """Plot normalized field and flux traces on their shared physical time axes."""
    ensure_directories()
    scenario_dir = scenario_plot_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / "time_trace_comparison.png"

    fig, axes = plt.subplots(len(traces), 1, figsize=(11, 4.0 * len(traces)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, trace in zip(axes, traces):
        times_ns = np.asarray(trace["t"], dtype=np.float64) * 1.0e9
        axis.plot(times_ns, np.asarray(trace["maxwell"]), label="Maxwell", linewidth=1.5)
        axis.plot(
            times_ns,
            np.asarray(trace["tidy3d"]),
            label="Tidy3D",
            linewidth=1.3,
            linestyle="--",
        )
        axis.set_ylabel(str(trace["ylabel"]))
        axis.set_title(str(trace["label"]))
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("physical time (ns)")
    fig.suptitle(f"{scenario_name}: normalized time-monitor traces", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
