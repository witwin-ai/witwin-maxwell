from __future__ import annotations

import numpy as np
import torch

from ...sources import POINT_DIPOLE_IDEAL_PROFILE_SCALE, POINT_DIPOLE_REFERENCE_WIDTH
from .spatial import (
    physical_interior_indices,
    beam_profile_from_source,
    plane_center,
    plane_wave_profile,
    resolve_injection_axis,
    soft_plane_wave_index,
    source_plane_index,
)
from .modes import sample_mode_source_component, sample_mode_source_profile, solve_mode_source_profile
from .temporal import append_source_term, apply_compiled_source_terms, apply_generic_source_terms
from .tfsf_common import build_term_from_profile, build_terms_from_specs, slice_coeff_patch, solve_numerical_wavenumber
from .tfsf_specs import DELTA_ATTR, E_CURL_ATTR, H_CURL_ATTR, build_discrete_tfsf_specs, magnetic_physical_vector


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
_PLANE_WAVE_POWER_CALIBRATION = 0.958
_PLANE_WAVE_DELAY_CALIBRATION_S = 1.65e-10


def _normalized_point_dipole_profile(dist_sq: torch.Tensor, width: float) -> torch.Tensor:
    width_sq = 2.0 * float(width) ** 2
    profile = torch.exp(-dist_sq / width_sq)
    if np.isclose(width, POINT_DIPOLE_REFERENCE_WIDTH):
        return profile

    reference_width_sq = 2.0 * POINT_DIPOLE_REFERENCE_WIDTH ** 2
    reference_profile = torch.exp(-dist_sq / reference_width_sq)
    current_sum = torch.clamp(profile.sum(), min=torch.finfo(profile.dtype).eps)
    return profile * (reference_profile.sum() / current_sum)


def _reference_point_dipole_mass(dist_sq: torch.Tensor) -> torch.Tensor:
    reference_width_sq = 2.0 * POINT_DIPOLE_REFERENCE_WIDTH ** 2
    return POINT_DIPOLE_IDEAL_PROFILE_SCALE * torch.exp(-dist_sq / reference_width_sq).sum()


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
    dist_sq: torch.Tensor,
    source_position,
    *,
    eps_tensor: torch.Tensor,
    offsets,
    dt: float,
    polarization_component: float,
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
    source_patch = torch.zeros_like(eps_slice)
    source_scale = -float(dt) * float(polarization_component) * _reference_point_dipole_mass(dist_sq)

    for ix, wx in zip(x_indices, x_weights):
        for iy, wy in zip(y_indices, y_weights):
            for iz, wz in zip(z_indices, z_weights):
                local_index = (ix - local_start[0], iy - local_start[1], iz - local_start[2])
                source_patch[local_index] += (
                    source_scale * float(wx) * float(wy) * float(wz) / eps_slice[local_index]
                )

    return global_offsets, source_patch


def _prepare_point_dipole_source(solver, source, *, source_index):
    x0, _, y0, _, z0, _ = solver.scene.domain_range
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

    for image_position, phase_real, phase_imag in solver._iter_source_images(source["position"], cutoff):
        src_x, src_y, src_z = image_position
        ix_start = max(0, int((src_x - cutoff - x0) / solver.dx))
        ix_end = min(solver.Nx, int((src_x + cutoff - x0) / solver.dx) + 1)
        iy_start = max(0, int((src_y - cutoff - y0) / solver.dy))
        iy_end = min(solver.Ny, int((src_y + cutoff - y0) / solver.dy) + 1)
        iz_start = max(0, int((src_z - cutoff - z0) / solver.dz))
        iz_end = min(solver.Nz, int((src_z + cutoff - z0) / solver.dz) + 1)

        if ix_end <= ix_start or iy_end <= iy_start or iz_end <= iz_start:
            continue

        if polarization[2] != 0:
            iz_end_ez = min(iz_end, solver.Nz - 1)
            if iz_end_ez > iz_start:
                ix = torch.arange(ix_start, ix_end, device=solver.device, dtype=coord_dtype)
                iy = torch.arange(iy_start, iy_end, device=solver.device, dtype=coord_dtype)
                iz = torch.arange(iz_start, iz_end_ez, device=solver.device, dtype=coord_dtype)
                px = x0 + ix * solver.dx
                py = y0 + iy * solver.dy
                pz = z0 + (iz + 0.5) * solver.dz
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
                        dist_sq,
                        image_position,
                        eps_tensor=solver.eps_Ez,
                        offsets=(ix_start, iy_start, iz_start),
                        dt=solver.dt,
                        polarization_component=polarization[2],
                    )
                else:
                    profile = _normalized_point_dipole_profile(dist_sq, width)
                    source_patch = (
                        (-solver.dt * polarization[2] / solver.eps_Ez[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end_ez])
                        * profile
                    )
                    patch_offsets = (ix_start, iy_start, iz_start)
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
                ix = torch.arange(ix_start, ix_end_ex, device=solver.device, dtype=coord_dtype)
                iy = torch.arange(iy_start, iy_end, device=solver.device, dtype=coord_dtype)
                iz = torch.arange(iz_start, iz_end, device=solver.device, dtype=coord_dtype)
                px = x0 + (ix + 0.5) * solver.dx
                py = y0 + iy * solver.dy
                pz = z0 + iz * solver.dz
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
                        dist_sq,
                        image_position,
                        eps_tensor=solver.eps_Ex,
                        offsets=(ix_start, iy_start, iz_start),
                        dt=solver.dt,
                        polarization_component=polarization[0],
                    )
                else:
                    profile = _normalized_point_dipole_profile(dist_sq, width)
                    source_patch = (
                        (-solver.dt * polarization[0] / solver.eps_Ex[ix_start:ix_end_ex, iy_start:iy_end, iz_start:iz_end])
                        * profile
                    )
                    patch_offsets = (ix_start, iy_start, iz_start)
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
                ix = torch.arange(ix_start, ix_end, device=solver.device, dtype=coord_dtype)
                iy = torch.arange(iy_start, iy_end_ey, device=solver.device, dtype=coord_dtype)
                iz = torch.arange(iz_start, iz_end, device=solver.device, dtype=coord_dtype)
                px = x0 + ix * solver.dx
                py = y0 + (iy + 0.5) * solver.dy
                pz = z0 + iz * solver.dz
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
                        dist_sq,
                        image_position,
                        eps_tensor=solver.eps_Ey,
                        offsets=(ix_start, iy_start, iz_start),
                        dt=solver.dt,
                        polarization_component=polarization[1],
                    )
                else:
                    profile = _normalized_point_dipole_profile(dist_sq, width)
                    source_patch = (
                        (-solver.dt * polarization[1] / solver.eps_Ey[ix_start:ix_end, iy_start:iy_end_ey, iz_start:iz_end])
                        * profile
                    )
                    patch_offsets = (ix_start, iy_start, iz_start)
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
# Equivalent surface currents for a plane with outward normal +axis:
# electric current J = n x H feeds the tangential E components; magnetic
# current M = -n x E feeds the tangential H components. Each entry is
# (target_field, source_component, sign).
_CUSTOM_FIELD_CURRENT_MAP = {
    "x": {
        "electric": (("Ey", "Hz", -1.0), ("Ez", "Hy", 1.0)),
        "magnetic": (("Hy", "Ez", 1.0), ("Hz", "Ey", -1.0)),
    },
    "y": {
        "electric": (("Ex", "Hz", 1.0), ("Ez", "Hx", -1.0)),
        "magnetic": (("Hx", "Ez", -1.0), ("Hz", "Ex", 1.0)),
    },
    "z": {
        "electric": (("Ex", "Hy", -1.0), ("Ey", "Hx", 1.0)),
        "magnetic": (("Hx", "Ey", 1.0), ("Hy", "Ex", -1.0)),
    },
}


def _region_index_range(solver, field_name, lo, hi):
    x0, _, y0, _, z0, _ = solver.scene.domain_range
    origins = (x0, y0, z0)
    steps = (solver.dx, solver.dy, solver.dz)
    sizes = tuple(int(dim) for dim in getattr(solver, field_name).shape)
    start = []
    stop = []
    for axis in range(3):
        lo_index = max(0, int((lo[axis] - origins[axis]) / steps[axis]))
        hi_index = min(sizes[axis], int((hi[axis] - origins[axis]) / steps[axis]) + 1)
        start.append(int(lo_index))
        stop.append(int(hi_index))
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
        start, stop = _region_index_range(solver, field_name, lo, hi)
        if any(stop[a] <= start[a] for a in range(3)):
            continue
        field_eps = getattr(solver, f"eps_{field_name}")
        eps_slice = field_eps[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        source_patch = (-solver.dt * pol_component / eps_slice)
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


def _dataset_inside_mask(axes, positions):
    mask = torch.ones(positions.shape[:-1], dtype=torch.bool, device=positions.device)
    for index, coords in enumerate(axes):
        if int(coords.numel()) > 1:
            query = positions[..., index]
            mask = mask & (query >= coords[0]) & (query <= coords[-1])
    return mask


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
    sampled = sampled * _dataset_inside_mask(axes, positions).to(sampled.dtype)
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
    normal_step = {"x": solver.dx, "y": solver.dy, "z": solver.dz}[normal_axis]
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    axes = _dataset_axes_tensors(solver, dataset)
    region_lo, region_hi = _dataset_region_bounds(dataset)
    mapping = _CUSTOM_FIELD_CURRENT_MAP[normal_axis]

    for target_field, source_component, sign in mapping["electric"]:
        if source_component not in dataset.components:
            continue
        # J_s = n x H, converted to a one-cell-thick volume current (divide by dn).
        values = (float(sign) / normal_step) * torch.tensor(
            dataset.components[source_component], device=solver.device, dtype=solver.Ex.dtype
        )
        _append_interpolated_current(
            solver,
            field_name=target_field,
            values=values,
            axes=axes,
            denom_tensor=getattr(solver, f"eps_{target_field}"),
            region_lo=region_lo,
            region_hi=region_hi,
            source_time=source_time,
            source_omega=source_omega,
            source_index=source_index,
            term_list=solver._source_terms,
        )

    for target_field, source_component, sign in mapping["magnetic"]:
        if source_component not in dataset.components:
            continue
        # M_s = -n x E, converted to a one-cell-thick volume current (divide by dn).
        values = (float(sign) / normal_step) * torch.tensor(
            dataset.components[source_component], device=solver.device, dtype=solver.Ex.dtype
        )
        _append_interpolated_current(
            solver,
            field_name=target_field,
            values=values,
            axes=axes,
            denom_tensor=getattr(solver, f"mu_{target_field}"),
            region_lo=region_lo,
            region_hi=region_hi,
            source_time=source_time,
            source_omega=source_omega,
            source_index=source_index,
            term_list=solver._magnetic_source_terms,
        )


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
    return _PLANE_WAVE_POWER_CALIBRATION / np.sqrt(unit_power)


def _prepare_plane_wave_surface_source(solver, source, *, source_index):
    if solver.scene.boundary.uses_kind("periodic") or solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError(
            "PlaneWave soft injection currently supports only none, pml, pec, or pmc boundaries."
        )

    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    direction = source["direction"]
    injection_axis = resolve_injection_axis(direction, source.get("injection_axis"))
    axis_index = _AXIS_TO_INDEX[injection_axis]
    direction_sign = 1 if float(direction[axis_index]) >= 0.0 else -1
    plane_index = soft_plane_wave_index(solver.scene, injection_axis, float(direction[axis_index]))

    lower = []
    upper = []
    aperture_bounds = []
    for axis in "xyz":
        physical_lo, physical_hi = physical_interior_indices(solver.scene, axis)
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
            axis_coords = {"x": solver.scene.x, "y": solver.scene.y, "z": solver.scene.z}[axis]
            lo = 0
            hi = {"x": solver.Nx, "y": solver.Ny, "z": solver.Nz}[axis] - 1
            aperture_bounds.append(
                (
                    float(axis_coords[physical_lo].item()),
                    float(axis_coords[physical_hi].item()),
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
    k_numeric = solve_numerical_wavenumber(solver, direction, DELTA_ATTR)
    if k_numeric > 1e-12:
        phase_speed = source_omega / k_numeric

    power_scale = _plane_wave_power_scale(source, tuple(aperture_bounds), injection_axis)
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

    def build_profile_term(spec, coeff_patch, component_scale):
        positions = _surface_plane_spec_positions(solver, spec)
        spatial_amplitude, delay_patch = plane_wave_profile(
            positions,
            direction=direction,
            reference_point=reference_point,
            propagation_speed=phase_speed,
        )
        delay_patch = delay_patch + _PLANE_WAVE_DELAY_CALIBRATION_S
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


def _prepare_mode_surface_source(solver, source, *, source_index):
    mode_data = solve_mode_source_profile(solver, source)
    source["effective_index"] = mode_data["effective_index"]
    source["beta"] = mode_data["beta"]
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

    def build_profile_term(spec, coeff_patch):
        positions = _surface_plane_spec_positions(solver, spec)
        component_profile = sample_mode_source_component(mode_data, positions, spec["incident_name"])
        if torch.max(torch.abs(component_profile)).item() <= 1e-12:
            return None
        zero_delay = torch.zeros_like(component_profile)
        scale = spec["sign"] * component_profile * coeff_patch
        return build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=scale,
            delay_patch=zero_delay,
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
        if source["kind"] == "plane_wave":
            _prepare_plane_wave_surface_source(solver, source, source_index=source_index)
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
