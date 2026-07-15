from __future__ import annotations

import math

import numpy as np
import torch

from ...sources import POINT_DIPOLE_REFERENCE_WIDTH
from .spatial import (
    beam_profile_from_source,
    plane_center,
    plane_wave_profile,
    resolve_injection_axis,
    soft_plane_wave_index,
    soft_plane_wave_region_spacing,
    source_plane_index,
)
from .modes import sample_mode_source_component, solve_mode_source_profile
from .temporal import append_source_term, apply_compiled_source_terms, apply_generic_source_terms
from .tfsf_common import build_term_from_profile, build_terms_from_specs, slice_coeff_patch, solve_numerical_wavenumber
from .tfsf_specs import E_CURL_ATTR, H_CURL_ATTR, build_discrete_tfsf_specs, magnetic_physical_vector


_FACE_SPEC_RANGES = {
    ("x", "low"): slice(0, 2),
    ("x", "high"): slice(2, 4),
    ("y", "low"): slice(4, 6),
    ("y", "high"): slice(6, 8),
    ("z", "low"): slice(8, 10),
    ("z", "high"): slice(10, 12),
}
_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_ETA0 = 376.730313668


def _normalized_point_dipole_profile(
    dist_sq: torch.Tensor, width: float, control_volumes: torch.Tensor
) -> torch.Tensor:
    width_sq = 2.0 * float(width) ** 2
    profile = torch.exp(-dist_sq / width_sq)
    integrated_mass = torch.sum(profile * control_volumes)
    return profile / torch.clamp(integrated_mass, min=torch.finfo(profile.dtype).eps)


def _sample_control_widths(coords: torch.Tensor) -> torch.Tensor:
    count = int(coords.numel())
    if count <= 1:
        return torch.ones_like(coords)
    diffs = coords[1:] - coords[:-1]
    widths = torch.empty_like(coords)
    widths[0] = 0.5 * diffs[0]
    widths[-1] = 0.5 * diffs[-1]
    if count > 2:
        widths[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return widths


def _yee_component_control_volumes(solver, field_name: str) -> torch.Tensor:
    coords = _yee_component_coords(solver, field_name)
    widths = tuple(_sample_control_widths(axis) for axis in coords)
    return widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]


def _yee_component_coords(solver, field_name: str) -> tuple[torch.Tensor, ...]:
    scene = solver.scene
    coords = {
        "Ex": (scene.x_half, scene.y, scene.z),
        "Ey": (scene.x, scene.y_half, scene.z),
        "Ez": (scene.x, scene.y, scene.z_half),
    }[field_name]
    return tuple(axis.to(device=solver.device, dtype=solver.Ex.dtype) for axis in coords)


def _symmetry_plane_source_scale(scene, field_name: str, position) -> float:
    """Scale a source that lies exactly on an image plane to its full-grid weight."""
    field_axis = {"Ex": 0, "Ey": 1, "Ez": 2}[field_name]
    scale = 1.0
    for axis, symmetry in enumerate(scene.symmetry):
        if symmetry is None:
            continue
        _mode, face = symmetry
        plane = float(scene.domain.bounds[axis][0 if face == "low" else 1])
        span = float(scene.domain.bounds[axis][1] - scene.domain.bounds[axis][0])
        if abs(float(position[axis]) - plane) > 1e-9 * max(span, 1.0):
            continue
        # A tangential E component is sampled on the image plane: the half-cell
        # control volume is half of its full-grid counterpart. A normal component
        # is staggered off the plane as well, so the full-grid source interpolation
        # also splits equally between the two mirrored edges.
        scale *= 0.25 if field_axis == axis else 0.5
    return scale


def _axis_control_overlap(coords: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Fraction of every sample's dual control interval covered by ``[lo, hi]``."""
    count = int(coords.numel())
    if count <= 1:
        return torch.ones_like(coords) if lo <= float(coords[0]) <= hi else torch.zeros_like(coords)
    midpoints = 0.5 * (coords[:-1] + coords[1:])
    left = torch.cat((coords[:1] - 0.5 * (coords[1:2] - coords[:1]), midpoints))
    right = torch.cat((midpoints, coords[-1:] + 0.5 * (coords[-1:] - coords[-2:-1])))
    overlap = torch.clamp(
        torch.minimum(right, torch.as_tensor(hi, device=coords.device, dtype=coords.dtype))
        - torch.maximum(left, torch.as_tensor(lo, device=coords.device, dtype=coords.dtype)),
        min=0.0,
    )
    return overlap / (right - left)


def _yee_box_overlap(solver, field_name: str, lo, hi):
    """Return a compact Yee patch containing volume-averaged box indicator weights."""
    axis_weights = tuple(
        _axis_control_overlap(coords, float(lo[axis]), float(hi[axis]))
        for axis, coords in enumerate(_yee_component_coords(solver, field_name))
    )
    nonzero = tuple(torch.nonzero(weights > 0.0, as_tuple=False).flatten() for weights in axis_weights)
    if any(int(indices.numel()) == 0 for indices in nonzero):
        return None
    start = tuple(int(indices[0].item()) for indices in nonzero)
    stop = tuple(int(indices[-1].item()) + 1 for indices in nonzero)
    slices = tuple(axis_weights[axis][start[axis] : stop[axis]] for axis in range(3))
    weights = slices[0][:, None, None] * slices[1][None, :, None] * slices[2][None, None, :]
    return start, stop, weights


def _ideal_axis_weights(coords: torch.Tensor, position: float) -> tuple[list[int], list[float]]:
    if coords.ndim != 1 or coords.numel() == 0:
        raise ValueError("coords must be a non-empty 1D tensor.")
    if coords.numel() == 1:
        return [0], [1.0]

    probe = coords.new_tensor(float(position))
    upper = int(torch.searchsorted(coords, probe, right=False).item())
    if upper <= 0:
        return [0], [1.0]
    if upper >= coords.numel():
        return [int(coords.numel() - 1)], [1.0]

    lower = upper - 1
    lower_coord = float(coords[lower].item())
    upper_coord = float(coords[upper].item())
    span = upper_coord - lower_coord
    if span <= torch.finfo(coords.dtype).eps:
        return [lower], [1.0]

    fraction = (float(position) - lower_coord) / span
    if fraction <= 1e-12:
        return [lower], [1.0]
    if fraction >= 1.0 - 1e-12:
        return [upper], [1.0]
    return [lower, upper], [1.0 - fraction, fraction]


def _ideal_point_dipole_term(
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
    source_position,
    *,
    eps_tensor: torch.Tensor,
    offsets,
    dt: float,
    polarization_component: float,
    control_volumes: torch.Tensor,
):
    x_indices, x_weights = _ideal_axis_weights(px, source_position[0])
    y_indices, y_weights = _ideal_axis_weights(py, source_position[1])
    z_indices, z_weights = _ideal_axis_weights(pz, source_position[2])

    local_start = (x_indices[0], y_indices[0], z_indices[0])
    local_stop = (x_indices[-1] + 1, y_indices[-1] + 1, z_indices[-1] + 1)
    global_offsets = (
        offsets[0] + local_start[0],
        offsets[1] + local_start[1],
        offsets[2] + local_start[2],
    )
    eps_slice = eps_tensor[
        global_offsets[0] : offsets[0] + local_stop[0],
        global_offsets[1] : offsets[1] + local_stop[1],
        global_offsets[2] : offsets[2] + local_stop[2],
    ]
    volume_slice = control_volumes[
        global_offsets[0] : offsets[0] + local_stop[0],
        global_offsets[1] : offsets[1] + local_stop[1],
        global_offsets[2] : offsets[2] + local_stop[2],
    ]
    source_patch = torch.zeros_like(eps_slice)
    source_scale = -float(dt) * float(polarization_component)

    for ix, wx in zip(x_indices, x_weights):
        for iy, wy in zip(y_indices, y_weights):
            for iz, wz in zip(z_indices, z_weights):
                local_index = (ix - local_start[0], iy - local_start[1], iz - local_start[2])
                source_patch[local_index] += (
                    source_scale
                    * float(wx)
                    * float(wy)
                    * float(wz)
                    / (eps_slice[local_index] * volume_slice[local_index])
                )

    return global_offsets, source_patch


def _axis_index_window(nodes64, lo_value, hi_value, size_cap):
    """Node-index window [lo, hi) covering [lo_value, hi_value] on an axis."""
    if float(hi_value) < float(nodes64[0]) or float(lo_value) > float(nodes64[-1]):
        return 0, 0
    lo_index = max(0, int(np.searchsorted(nodes64, float(lo_value), side="right")) - 1)
    hi_index = min(int(size_cap), int(np.searchsorted(nodes64, float(hi_value), side="left")) + 1)
    return lo_index, hi_index


def _prepare_point_dipole_source(solver, source, *, source_index):
    scene = solver.scene
    width = float(source["width"])
    polarization = source["polarization"]
    profile_kind = source.get("profile", "gaussian")
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    if profile_kind == "ideal":
        cutoff = 3.0 * POINT_DIPOLE_REFERENCE_WIDTH
    else:
        cutoff = 3.0 * max(width, 0.5 * POINT_DIPOLE_REFERENCE_WIDTH)
    coord_dtype = solver.Ex.dtype
    control_volumes = {
        field_name: _yee_component_control_volumes(solver, field_name)
        for field_name, component in zip(("Ex", "Ey", "Ez"), polarization)
        if not np.isclose(component, 0.0)
    }

    for image_position, phase_real, phase_imag in solver._iter_source_images(source["position"], cutoff):
        src_x, src_y, src_z = image_position
        ix_start, ix_end = _axis_index_window(scene.x_nodes64, src_x - cutoff, src_x + cutoff, solver.Nx)
        iy_start, iy_end = _axis_index_window(scene.y_nodes64, src_y - cutoff, src_y + cutoff, solver.Ny)
        iz_start, iz_end = _axis_index_window(scene.z_nodes64, src_z - cutoff, src_z + cutoff, solver.Nz)

        if ix_end <= ix_start or iy_end <= iy_start or iz_end <= iz_start:
            continue

        if polarization[2] != 0:
            iz_end_ez = min(iz_end, solver.Nz - 1)
            if iz_end_ez > iz_start:
                px = scene.x[ix_start:ix_end].to(dtype=coord_dtype)
                py = scene.y[iy_start:iy_end].to(dtype=coord_dtype)
                pz = scene.z_half[iz_start:iz_end_ez].to(dtype=coord_dtype)
                dist_sq = (
                    (px[:, None, None] - src_x) ** 2
                    + (py[None, :, None] - src_y) ** 2
                    + (pz[None, None, :] - src_z) ** 2
                )
                if profile_kind == "ideal":
                    patch_offsets, source_patch = _ideal_point_dipole_term(
                        px,
                        py,
                        pz,
                        image_position,
                        eps_tensor=solver.eps_Ez,
                        offsets=(ix_start, iy_start, iz_start),
                        dt=solver.dt,
                        polarization_component=polarization[2],
                        control_volumes=control_volumes["Ez"],
                    )
                else:
                    volume_slice = control_volumes["Ez"][
                        ix_start:ix_end, iy_start:iy_end, iz_start:iz_end_ez
                    ]
                    profile = _normalized_point_dipole_profile(dist_sq, width, volume_slice)
                    source_patch = (
                        (-solver.dt * polarization[2] / solver.eps_Ez[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end_ez])
                        * profile
                    )
                    patch_offsets = (ix_start, iy_start, iz_start)
                source_patch *= _symmetry_plane_source_scale(scene, "Ez", image_position)
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name="Ez",
                    offsets=patch_offsets,
                    patch=source_patch,
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_time=source_time,
                    omega=source_omega,
                )

        if polarization[0] != 0:
            ix_end_ex = min(ix_end, solver.Nx - 1)
            if ix_end_ex > ix_start:
                px = scene.x_half[ix_start:ix_end_ex].to(dtype=coord_dtype)
                py = scene.y[iy_start:iy_end].to(dtype=coord_dtype)
                pz = scene.z[iz_start:iz_end].to(dtype=coord_dtype)
                dist_sq = (
                    (px[:, None, None] - src_x) ** 2
                    + (py[None, :, None] - src_y) ** 2
                    + (pz[None, None, :] - src_z) ** 2
                )
                if profile_kind == "ideal":
                    patch_offsets, source_patch = _ideal_point_dipole_term(
                        px,
                        py,
                        pz,
                        image_position,
                        eps_tensor=solver.eps_Ex,
                        offsets=(ix_start, iy_start, iz_start),
                        dt=solver.dt,
                        polarization_component=polarization[0],
                        control_volumes=control_volumes["Ex"],
                    )
                else:
                    volume_slice = control_volumes["Ex"][
                        ix_start:ix_end_ex, iy_start:iy_end, iz_start:iz_end
                    ]
                    profile = _normalized_point_dipole_profile(dist_sq, width, volume_slice)
                    source_patch = (
                        (-solver.dt * polarization[0] / solver.eps_Ex[ix_start:ix_end_ex, iy_start:iy_end, iz_start:iz_end])
                        * profile
                    )
                    patch_offsets = (ix_start, iy_start, iz_start)
                source_patch *= _symmetry_plane_source_scale(scene, "Ex", image_position)
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name="Ex",
                    offsets=patch_offsets,
                    patch=source_patch,
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_time=source_time,
                    omega=source_omega,
                )

        if polarization[1] != 0:
            iy_end_ey = min(iy_end, solver.Ny - 1)
            if iy_end_ey > iy_start:
                px = scene.x[ix_start:ix_end].to(dtype=coord_dtype)
                py = scene.y_half[iy_start:iy_end_ey].to(dtype=coord_dtype)
                pz = scene.z[iz_start:iz_end].to(dtype=coord_dtype)
                dist_sq = (
                    (px[:, None, None] - src_x) ** 2
                    + (py[None, :, None] - src_y) ** 2
                    + (pz[None, None, :] - src_z) ** 2
                )
                if profile_kind == "ideal":
                    patch_offsets, source_patch = _ideal_point_dipole_term(
                        px,
                        py,
                        pz,
                        image_position,
                        eps_tensor=solver.eps_Ey,
                        offsets=(ix_start, iy_start, iz_start),
                        dt=solver.dt,
                        polarization_component=polarization[1],
                        control_volumes=control_volumes["Ey"],
                    )
                else:
                    volume_slice = control_volumes["Ey"][
                        ix_start:ix_end, iy_start:iy_end_ey, iz_start:iz_end
                    ]
                    profile = _normalized_point_dipole_profile(dist_sq, width, volume_slice)
                    source_patch = (
                        (-solver.dt * polarization[1] / solver.eps_Ey[ix_start:ix_end, iy_start:iy_end_ey, iz_start:iz_end])
                        * profile
                    )
                    patch_offsets = (ix_start, iy_start, iz_start)
                source_patch *= _symmetry_plane_source_scale(scene, "Ey", image_position)
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name="Ey",
                    offsets=patch_offsets,
                    patch=source_patch,
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_time=source_time,
                    omega=source_omega,
                )


_UNIFORM_CURRENT_COMPONENTS = (("Ex", 0), ("Ey", 1), ("Ez", 2))
_CURRENT_ELECTRIC_MAP = {"Jx": "Ex", "Jy": "Ey", "Jz": "Ez"}
_CURRENT_MAGNETIC_MAP = {"Mx": "Hx", "My": "Hy", "Mz": "Hz"}


def _region_index_range(solver, field_name, lo, hi):
    scene = solver.scene
    axis_nodes = (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64)
    axis_half = (scene.x_half64, scene.y_half64, scene.z_half64)
    sizes = tuple(int(dim) for dim in getattr(solver, field_name).shape)
    half_axes = solver._COMPONENT_HALF_OFFSET_AXES[field_name]
    start = []
    stop = []
    for axis in range(3):
        if float(lo[axis]) == float(hi[axis]):
            component_coords = axis_half[axis] if half_axes[axis] else axis_nodes[axis]
            component_coords = component_coords[: sizes[axis]]
            nearest = int(np.argmin(np.abs(component_coords - float(lo[axis]))))
            start.append(nearest)
            stop.append(nearest + 1)
            continue
        lo_index, hi_index = _axis_index_window(axis_nodes[axis], lo[axis], hi[axis], sizes[axis])
        if half_axes[axis]:
            # ``_axis_index_window`` returns the bounding-node window. A Yee
            # component centered between nodes has one fewer sample inside the
            # same discretized box along that axis.
            hi_index = max(lo_index, hi_index - 1)
        start.append(lo_index)
        stop.append(hi_index)
    return tuple(start), tuple(stop)


def _prepare_uniform_current_source(solver, source, *, source_index):
    center = source["center"]
    size = source["size"]
    lo = tuple(center[axis] - 0.5 * size[axis] for axis in range(3))
    hi = tuple(center[axis] + 0.5 * size[axis] for axis in range(3))
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    polarization = source["polarization"]

    for field_name, axis in _UNIFORM_CURRENT_COMPONENTS:
        pol_component = float(polarization[axis])
        if np.isclose(pol_component, 0.0):
            continue
        overlap = _yee_box_overlap(solver, field_name, lo, hi)
        if overlap is None:
            continue
        start, stop, weights = overlap
        field_eps = getattr(solver, f"eps_{field_name}")
        eps_slice = field_eps[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        source_patch = (-solver.dt * pol_component / eps_slice) * weights
        append_source_term(
            solver._source_terms,
            solver,
            field_name=field_name,
            offsets=start,
            patch=source_patch,
            source_index=source_index,
            source_time=source_time,
            omega=source_omega,
        )


def _axis_interp_weights(coords, query):
    count = int(coords.numel())
    if count == 1:
        index = torch.zeros_like(query, dtype=torch.long)
        return index, index, torch.zeros_like(query)
    upper = torch.searchsorted(coords, query.contiguous(), right=True).clamp(1, count - 1)
    lower = upper - 1
    lower_coord = coords[lower]
    upper_coord = coords[upper]
    fraction = (query - lower_coord) / (upper_coord - lower_coord)
    return lower, upper, fraction.clamp(0.0, 1.0)


def _trilinear_sample(values, axes, positions):
    x = positions[..., 0]
    y = positions[..., 1]
    z = positions[..., 2]
    ix0, ix1, wx = _axis_interp_weights(axes[0], x)
    iy0, iy1, wy = _axis_interp_weights(axes[1], y)
    iz0, iz1, wz = _axis_interp_weights(axes[2], z)

    def gather(i, j, k):
        return values[i, j, k]

    c00 = gather(ix0, iy0, iz0) * (1.0 - wz) + gather(ix0, iy0, iz1) * wz
    c01 = gather(ix0, iy1, iz0) * (1.0 - wz) + gather(ix0, iy1, iz1) * wz
    c10 = gather(ix1, iy0, iz0) * (1.0 - wz) + gather(ix1, iy0, iz1) * wz
    c11 = gather(ix1, iy1, iz0) * (1.0 - wz) + gather(ix1, iy1, iz1) * wz
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wx) + c1 * wx


def _dataset_axes_tensors(solver, dataset):
    return tuple(
        torch.tensor(axis, device=solver.device, dtype=solver.Ex.dtype)
        for axis in dataset.coords
    )


def _dataset_region_bounds(dataset):
    lo = tuple(float(axis[0]) for axis in dataset.coords)
    hi = tuple(float(axis[-1]) for axis in dataset.coords)
    return lo, hi


def _append_interpolated_current(
    solver,
    *,
    field_name,
    values,
    axes,
    denom_tensor,
    region_lo,
    region_hi,
    source_time,
    source_omega,
    source_index,
    term_list,
):
    start, stop = _region_index_range(solver, field_name, region_lo, region_hi)
    if any(stop[a] <= start[a] for a in range(3)):
        return
    shape = tuple(stop[a] - start[a] for a in range(3))
    positions = solver._component_positions(field_name, start, shape, dtype=solver.Ex.dtype)
    sampled = _trilinear_sample(values, axes, positions)
    if torch.max(torch.abs(sampled)).item() <= 1e-30:
        return
    denom_slice = denom_tensor[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    source_patch = (-solver.dt / denom_slice) * sampled
    append_source_term(
        term_list,
        solver,
        field_name=field_name,
        offsets=start,
        patch=source_patch,
        source_index=source_index,
        source_time=source_time,
        omega=source_omega,
    )


def _prepare_custom_current_source(solver, source, *, source_index):
    dataset = source["dataset"]
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    axes = _dataset_axes_tensors(solver, dataset)
    region_lo, region_hi = _dataset_region_bounds(dataset)

    for key, array in dataset.components.items():
        values = torch.tensor(array, device=solver.device, dtype=solver.Ex.dtype)
        if key in _CURRENT_ELECTRIC_MAP:
            field_name = _CURRENT_ELECTRIC_MAP[key]
            denom_tensor = getattr(solver, f"eps_{field_name}")
            term_list = solver._source_terms
        else:
            field_name = _CURRENT_MAGNETIC_MAP[key]
            denom_tensor = getattr(solver, f"mu_{field_name}")
            term_list = solver._magnetic_source_terms
        _append_interpolated_current(
            solver,
            field_name=field_name,
            values=values,
            axes=axes,
            denom_tensor=denom_tensor,
            region_lo=region_lo,
            region_hi=region_hi,
            source_time=source_time,
            source_omega=source_omega,
            source_index=source_index,
            term_list=term_list,
        )


def _prepare_custom_field_source(solver, source, *, source_index):
    dataset = source["dataset"]
    normal_axis = source["normal_axis"]
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    axes = _dataset_axes_tensors(solver, dataset)
    region_lo, region_hi = _dataset_region_bounds(dataset)
    axis_index = _AXIS_TO_INDEX[normal_axis]
    plane_coord = 0.5 * (float(region_lo[axis_index]) + float(region_hi[axis_index]))
    plane_coords = getattr(solver.scene, normal_axis)
    plane_index = int(torch.argmin(torch.abs(plane_coords - plane_coord)).item())

    lower = [0, 0, 0]
    upper = [0, 0, 0]
    for index, axis in enumerate("xyz"):
        if axis == normal_axis:
            lower[index] = plane_index
            upper[index] = plane_index + 1
            continue
        nodes64 = (solver.scene.x_nodes64, solver.scene.y_nodes64, solver.scene.z_nodes64)[index]
        size_cap = (solver.Nx, solver.Ny, solver.Nz)[index]
        start, stop = _axis_index_window(
            nodes64,
            region_lo[index],
            region_hi[index],
            size_cap,
        )
        lower[index] = start
        upper[index] = stop - 1

    electric_specs, magnetic_specs = build_discrete_tfsf_specs(tuple(lower), tuple(upper))
    face_side = "low"
    electric_specs = electric_specs[_FACE_SPEC_RANGES[(normal_axis, face_side)]]
    magnetic_specs = magnetic_specs[_FACE_SPEC_RANGES[(normal_axis, face_side)]]
    component_values = {
        name: torch.tensor(values, device=solver.device, dtype=solver.Ex.dtype)
        for name, values in dataset.components.items()
    }

    def build_profile_term(spec, coeff_patch, _component_scale):
        values = component_values[spec["incident_name"]]
        positions = _surface_plane_spec_positions(solver, spec)
        sampled = _trilinear_sample(values, axes, positions)
        if spec["incident_name"].startswith("H"):
            # The discrete face specification stores incident magnetic fields
            # with the update-equation sign (``-k x E``), while FieldDataset
            # exposes the physical ``H = k x E / eta`` convention.
            sampled = -sampled
        scale = spec["sign"] * sampled * coeff_patch
        return build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=scale,
            delay_patch=torch.zeros_like(scale),
            activation_delay_patch=None,
            source_time=source_time,
            omega=source_omega,
            source_index=source_index,
        )

    solver._electric_source_terms.extend(build_terms_from_specs(
        solver,
        electric_specs,
        {name: (1.0 if name in component_values else 0.0) for name in ("Hx", "Hy", "Hz")},
        E_CURL_ATTR,
        build_profile_term,
    ))
    solver._magnetic_source_terms.extend(build_terms_from_specs(
        solver,
        magnetic_specs,
        {name: (1.0 if name in component_values else 0.0) for name in ("Ex", "Ey", "Ez")},
        H_CURL_ATTR,
        build_profile_term,
    ))


def _prepare_surface_source(solver, source, *, source_index):
    if solver.scene.boundary.uses_kind("periodic") or solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError(
            f"{source['kind']} currently supports only none, pml, pec, or pmc boundaries."
        )

    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    direction = source["direction"]
    axis = resolve_injection_axis(direction, source.get("injection_axis"))
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    plane_index = source_plane_index(solver.scene, axis, float(direction[axis_index]))
    plane_coordinate = solver._plane_coordinate(axis, plane_index)
    reference_point = plane_center(
        solver.scene,
        axis,
        plane_coordinate,
        device=solver.device,
        dtype=solver.Ex.dtype,
    )

    for field_name, polarization_component in zip(("Ex", "Ey", "Ez"), source["polarization"]):
        if np.isclose(polarization_component, 0.0):
            continue
        offsets, shape = solver._component_plane_spec(field_name, axis, plane_index)
        positions = solver._component_positions(field_name, offsets, shape, dtype=solver.Ex.dtype)
        if source["kind"] == "plane_wave":
            spatial_amplitude, delay_patch = plane_wave_profile(
                positions,
                direction=direction,
                reference_point=reference_point,
                propagation_speed=solver.c,
            )
        else:
            spatial_amplitude, delay_patch = beam_profile_from_source(
                positions,
                source,
                frequency=source_frequency,
                propagation_speed=solver.c,
            )

        field_eps = getattr(solver, f"eps_{field_name}")
        eps_slice = field_eps[
            offsets[0] : offsets[0] + shape[0],
            offsets[1] : offsets[1] + shape[1],
            offsets[2] : offsets[2] + shape[2],
        ]
        source_patch = (-solver.dt * polarization_component / eps_slice) * spatial_amplitude
        if source_time["kind"] == "cw":
            phase_shift = source_omega * delay_patch
            append_source_term(
                solver._source_terms,
                solver,
                field_name=field_name,
                offsets=offsets,
                cw_cos_patch=source_patch * torch.cos(phase_shift),
                cw_sin_patch=source_patch * torch.sin(phase_shift),
                source_index=source_index,
                source_time=source_time,
                omega=source_omega,
            )
            continue
        append_source_term(
            solver._source_terms,
            solver,
            field_name=field_name,
            offsets=offsets,
            patch=source_patch,
            delay_patch=delay_patch,
            source_index=source_index,
            source_time=source_time,
            omega=source_omega,
        )


def _surface_plane_spec_positions(solver, spec):
    return solver._component_positions(
        spec["incident_name"],
        spec["sample_offsets"],
        spec["sample_shape"],
        dtype=solver.Ex.dtype,
    )


def _plane_wave_power_scale(source, aperture_bounds, injection_axis: str) -> float:
    """Absolute surface-current scale for unit time-averaged incident power.

    The soft ``PlaneWave`` face injects the surface-equivalent currents of the
    target incident wave (``J_s = n x H_inc`` on the electric face, ``M_s =
    -n x E_inc`` on the magnetic face). By the surface-equivalence principle
    these radiate the incident field forward with unit gain, so the injected
    incident amplitude equals the specified ``E`` amplitude scaled here. On the
    Yee grid the numerical wave impedance of a plane wave that satisfies the
    discrete dispersion relation is exactly ``eta0`` -- the leapfrog identity
    ``sin(omega*dt/2)/(c*dt) = sin(k~*d/2)/d`` makes ``H0/E0 = 1/eta0`` -- so the
    physical-impedance magnetic current the injector uses (``magnetic_physical_
    vector(..., eta0)``) is the correct numerical amplitude and the forward
    radiation gain is unity. No empirical calibration is therefore required.

    Normalizing to unit incident power over the illuminated aperture, the
    time-averaged forward power ``0.5*|E0|^2*A*cos(theta)/eta0`` equals one when
    ``|E0| = 1/sqrt(unit_power)`` with ``unit_power = A*cos(theta)/(2*eta0)``.
    The residual discrete-injection error is below the 2% absolute-power
    acceptance across frequencies and spacings (see
    ``tests/sources/incident/test_soft_planewave_absolute_power.py``).
    """
    axis_index = _AXIS_TO_INDEX[injection_axis]
    tangential_extents = [
        float(axis_bounds[1] - axis_bounds[0])
        for index, axis_bounds in enumerate(aperture_bounds)
        if index != axis_index
    ]
    aperture_area = tangential_extents[0] * tangential_extents[1]
    incidence_cosine = abs(float(source["direction"][axis_index]))
    unit_power = aperture_area * incidence_cosine / (2.0 * _ETA0)
    if unit_power <= 0.0:
        raise ValueError("PlaneWave source requires a positive aperture power for normalization.")
    return 1.0 / np.sqrt(unit_power)


def _beam_power_scale(source, injection_axis: str) -> float:
    """Peak electric-field scale for a unit-power Gaussian beam.

    The astigmatic profile uses ``exp(-u^2 / wu^2 - v^2 / wv^2)`` with the
    longitudinal prefactor that preserves its transverse L2 mass.  Therefore
    ``integral |E/E0|^2 dA = pi*wu*wv/2`` at every propagation plane.
    """
    if source["kind"] == "astigmatic_gaussian_beam":
        waist_u = float(source["beam_waist_u"])
        waist_v = float(source["beam_waist_v"])
    else:
        waist_u = waist_v = float(source["beam_waist"])
    axis_index = _AXIS_TO_INDEX[injection_axis]
    incidence_cosine = abs(float(source["direction"][axis_index]))
    unit_power = math.pi * waist_u * waist_v * incidence_cosine / (4.0 * _ETA0)
    if unit_power <= 0.0:
        raise ValueError("Gaussian beam requires a positive transverse power integral.")
    return 1.0 / math.sqrt(unit_power)


def _validate_soft_surface_source_boundary(boundary, direction, injection_axis: str) -> None:
    if boundary.uses_kind("bloch"):
        raise NotImplementedError(
            "PlaneWave soft injection does not support Bloch boundaries; use a TFSF source."
        )
    incompatible_periodic_axes = [
        axis
        for index, axis in enumerate("xyz")
        if boundary.axis_kind(axis) == "periodic"
        and (axis == injection_axis or abs(float(direction[index])) > 1.0e-12)
    ]
    if incompatible_periodic_axes:
        axes = ", ".join(incompatible_periodic_axes)
        raise NotImplementedError(
            "PlaneWave soft injection supports periodic boundaries only on zero-phase "
            f"transverse axes; incompatible axes: {axes}."
        )


def _prepare_power_normalized_surface_source(solver, source, *, source_index):
    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    direction = source["direction"]
    injection_axis = resolve_injection_axis(direction, source.get("injection_axis"))
    _validate_soft_surface_source_boundary(
        solver.scene.boundary,
        direction,
        injection_axis,
    )
    axis_index = _AXIS_TO_INDEX[injection_axis]
    direction_sign = 1 if float(direction[axis_index]) >= 0.0 else -1
    plane_index = soft_plane_wave_index(solver.scene, injection_axis, float(direction[axis_index]))

    lower = []
    upper = []
    aperture_bounds = []
    computational_range = solver.scene.domain_range
    for axis in "xyz":
        if axis == injection_axis:
            if direction_sign > 0:
                lo = plane_index
                hi = plane_index + 1
            else:
                lo = plane_index - 1
                hi = plane_index
            axis_coords = {"x": solver.scene.x, "y": solver.scene.y, "z": solver.scene.z}[axis]
            plane_coord = float(axis_coords[plane_index].item())
            aperture_bounds.append((plane_coord, plane_coord))
        else:
            lo = 0
            hi = {"x": solver.Nx, "y": solver.Ny, "z": solver.Nz}[axis] - 1
            range_offset = 2 * _AXIS_TO_INDEX[axis]
            aperture_bounds.append(
                (
                    float(computational_range[range_offset]),
                    float(computational_range[range_offset + 1]),
                )
            )
        lower.append(int(lo))
        upper.append(int(hi))

    lower = tuple(lower)
    upper = tuple(upper)
    face_side = "low" if direction_sign > 0 else "high"
    electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)
    electric_specs = electric_specs[_FACE_SPEC_RANGES[(injection_axis, face_side)]]
    magnetic_specs = magnetic_specs[_FACE_SPEC_RANGES[(injection_axis, face_side)]]

    phase_speed = solver.c
    # Numerical-dispersion phase correction with the grid spacing local to the
    # launch plane and aperture. On a uniform grid this is exactly the per-axis
    # spacing (bit-for-bit the previous global-minimum value); on a graded grid it
    # tracks the launch-region cells the wavefront actually crosses instead of the
    # global-minimum spacing, whose finest cell can lie far from the source and
    # mis-tune the injected phase velocity.
    deltas = soft_plane_wave_region_spacing(
        solver.scene,
        injection_axis=injection_axis,
        plane_index=plane_index,
        direction_sign=direction_sign,
    )
    k_numeric = solve_numerical_wavenumber(solver, direction, deltas)
    if k_numeric > 1e-12:
        phase_speed = source_omega / k_numeric

    if source["kind"] == "plane_wave":
        power_scale = _plane_wave_power_scale(source, tuple(aperture_bounds), injection_axis)
    else:
        power_scale = _beam_power_scale(source, injection_axis)
    # A native Yee-plane Poynting monitor co-locates the tangential E/H pair by
    # interpolating across their half-cell separation along the injection axis.
    # For the injected discrete plane wave this reduces the measured normal power
    # by cos(k_axis * delta_axis / 2).  Normalize by that derived factor so a unit
    # PlaneWave carries one watt through the full computational aperture, matching
    # Tidy3D's infinite-plane source convention without an empirical multiplier.
    k_axis = k_numeric * float(direction[axis_index])
    discrete_power_factor = math.cos(0.5 * k_axis * float(deltas[injection_axis]))
    if discrete_power_factor <= 0.0:
        raise ValueError(
            "PlaneWave grid is too coarse for positive discrete normal power; "
            "refine the injection-axis spacing."
        )
    power_scale /= math.sqrt(discrete_power_factor)
    electric_vector = {
        "Ex": power_scale * float(source["polarization"][0]),
        "Ey": power_scale * float(source["polarization"][1]),
        "Ez": power_scale * float(source["polarization"][2]),
    }
    magnetic_vector = {
        name: power_scale * value
        for name, value in magnetic_physical_vector(direction, source["polarization"], _ETA0).items()
    }

    plane_coordinate = solver._plane_coordinate(injection_axis, plane_index)
    reference_point = plane_center(
        solver.scene,
        injection_axis,
        plane_coordinate,
        device=solver.device,
        dtype=solver.Ex.dtype,
    )
    beam_reference_delay = None
    if source["kind"] != "plane_wave":
        _, beam_reference_delay = beam_profile_from_source(
            reference_point.reshape(1, 1, 1, 3),
            source,
            frequency=source_frequency,
            propagation_speed=phase_speed,
        )
        beam_reference_delay = beam_reference_delay.reshape(())

    def build_profile_term(spec, coeff_patch, component_scale):
        positions = _surface_plane_spec_positions(solver, spec)
        if source["kind"] == "plane_wave":
            spatial_amplitude, delay_patch = plane_wave_profile(
                positions,
                direction=direction,
                reference_point=reference_point,
                propagation_speed=phase_speed,
            )
        else:
            spatial_amplitude, delay_patch = beam_profile_from_source(
                positions,
                source,
                frequency=source_frequency,
                propagation_speed=phase_speed,
            )
            # The analytic beam phase is expressed relative to its waist.  A
            # soft source, however, starts on this launch plane: retaining the
            # large negative waist-to-plane delay would place most of a pulsed
            # waveform before t=0.  Remove only the common launch-plane delay;
            # transverse curvature and Gouy phase differences remain intact.
            delay_patch = delay_patch - beam_reference_delay
        scale = spec["sign"] * component_scale * spatial_amplitude * coeff_patch
        return build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=scale,
            delay_patch=delay_patch,
            activation_delay_patch=None,
            source_time=source_time,
            omega=source_omega,
            source_index=source_index,
        )

    solver._electric_source_terms.extend(build_terms_from_specs(
        solver,
        electric_specs,
        magnetic_vector,
        E_CURL_ATTR,
        build_profile_term,
    ))
    solver._magnetic_source_terms.extend(build_terms_from_specs(
        solver,
        magnetic_specs,
        electric_vector,
        H_CURL_ATTR,
        build_profile_term,
    ))


def _mode_normal_stagger_power_factor(solver, source, mode_data) -> float:
    """Return the resolved-grid correction for axial Yee interpolation.

    A non-positive cosine means that the requested mode is beyond the normal-axis
    Nyquist limit.  ModeSource historically permits such coarse grids for setup
    and adjoint smoke tests, so keep the transverse one-watt normalization there;
    physically resolved grids receive the exact half-cell correction.
    """
    normal_axis = source["normal_axis"]
    plane_index = int(mode_data["plane_index"])
    direction_sign = int(source["direction_sign"])
    plane_coordinate = solver._plane_coordinate(normal_axis, plane_index)
    half_axis = getattr(solver.scene, f"{normal_axis}_half")
    half_index = plane_index - 1 if direction_sign > 0 else plane_index
    normal_half_offset = abs(float(half_axis[half_index].item()) - float(plane_coordinate))
    factor = math.cos(abs(float(mode_data["beta"])) * normal_half_offset)
    if factor <= 0.0:
        return 1.0
    return float(factor)


def _prepare_mode_surface_source(solver, source, *, source_index):
    mode_data = solve_mode_source_profile(solver, source)
    source["effective_index"] = mode_data["effective_index"]
    source["beta"] = mode_data["beta"]
    source["effective_index_complex"] = mode_data.get("effective_index_complex")
    source["beta_complex"] = mode_data.get("beta_complex")
    source["mode_solver_kind"] = mode_data.get("mode_solver_kind")

    face_side = "low" if int(source["direction_sign"]) > 0 else "high"
    electric_specs, magnetic_specs = build_discrete_tfsf_specs(
        mode_data["box_lower"],
        mode_data["box_upper"],
    )
    electric_specs = electric_specs[_FACE_SPEC_RANGES[(source["normal_axis"], face_side)]]
    magnetic_specs = magnetic_specs[_FACE_SPEC_RANGES[(source["normal_axis"], face_side)]]
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    normal_axis_index = _AXIS_TO_INDEX[source["normal_axis"]]
    plane_coordinate = solver._plane_coordinate(source["normal_axis"], mode_data["plane_index"])
    direction_sign = int(source["direction_sign"])
    phase_speed = source_omega / max(abs(float(mode_data["beta"])), 1.0e-30)
    normal_stagger_power_factor = _mode_normal_stagger_power_factor(solver, source, mode_data)
    mode_injection_scale = 1.0 / math.sqrt(normal_stagger_power_factor)
    source["normal_stagger_power_factor"] = float(normal_stagger_power_factor)
    source["prepared_mode_power"] = 1.0

    def build_profile_term(spec, coeff_patch):
        positions = _surface_plane_spec_positions(solver, spec)
        component_profile = mode_injection_scale * sample_mode_source_component(
            mode_data,
            positions,
            spec["incident_name"],
        )
        if spec["incident_name"].startswith("H"):
            # TFSF face specifications store the magnetic incident component
            # with the update-equation ``-k x E`` sign. Mode profiles expose
            # the physical ``H = k x E / eta`` field, so convert conventions.
            component_profile = -component_profile
        if torch.max(torch.abs(component_profile)).item() <= 1e-12:
            return None
        delay_patch = (
            float(direction_sign)
            * (positions[..., normal_axis_index] - float(plane_coordinate))
            / float(phase_speed)
        )
        scale = spec["sign"] * component_profile * coeff_patch
        return build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=scale,
            delay_patch=delay_patch,
            activation_delay_patch=None,
            source_time=source_time,
            omega=source_omega,
            source_index=source_index,
        )

    for spec in electric_specs:
        coeff_patch = slice_coeff_patch(solver, spec, E_CURL_ATTR)
        term = build_profile_term(spec, coeff_patch)
        if term is not None:
            solver._electric_source_terms.append(term)
    for spec in magnetic_specs:
        coeff_patch = slice_coeff_patch(solver, spec, H_CURL_ATTR)
        term = build_profile_term(spec, coeff_patch)
        if term is not None:
            solver._magnetic_source_terms.append(term)


def inject_magnetic_surface_source_terms(solver, *, time_value):
    if not getattr(solver, "_magnetic_source_terms", None):
        return
    apply_generic_source_terms(
        solver,
        solver._magnetic_source_terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        clamp_pec=False,
    )


def inject_electric_surface_source_terms(solver, *, time_value):
    if not getattr(solver, "_electric_source_terms", None):
        return
    apply_generic_source_terms(
        solver,
        solver._electric_source_terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        clamp_pec=False,
    )


def initialize_source_terms(solver):
    solver._source_terms = []
    solver._magnetic_source_terms = []
    solver._electric_source_terms = []
    compiled_sources = tuple(getattr(solver, "_compiled_sources", ()) or ())
    if not compiled_sources and getattr(solver, "_compiled_source", None) is not None:
        compiled_sources = (solver._compiled_source,)
    if not compiled_sources:
        return

    for source_index, source in enumerate(compiled_sources):
        if source.get("injection", {}).get("kind") == "tfsf":
            continue
        if source["kind"] == "point_dipole":
            _prepare_point_dipole_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "mode_source":
            _prepare_mode_surface_source(solver, source, source_index=source_index)
            continue
        if source["kind"] in {"plane_wave", "gaussian_beam", "astigmatic_gaussian_beam"}:
            _prepare_power_normalized_surface_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "uniform_current":
            _prepare_uniform_current_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "custom_current":
            _prepare_custom_current_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "custom_field":
            _prepare_custom_field_source(solver, source, source_index=source_index)
            continue
        _prepare_surface_source(solver, source, source_index=source_index)


def inject_source_terms(solver, n=None, signal=None, time_value=None):
    if solver._source_terms is None:
        initialize_source_terms(solver)
    if not solver._source_terms:
        return

    if time_value is None:
        if n is None:
            raise ValueError("Either n or time_value must be provided for source injection.")
        time_value = n * solver.dt

    apply_compiled_source_terms(
        solver,
        solver._source_terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        signal=signal,
    )
