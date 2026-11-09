from __future__ import annotations

import math

import numpy as np
import torch

from ...constants import ETA_0
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
from .temporal import append_source_term, apply_compiled_source_terms, apply_generic_source_terms, injection_time, prepare_source_gathers
from .tfsf_common import _component_roundoff_floor, build_term_from_profile, build_terms_from_specs, slice_coeff_patch, solve_numerical_wavenumber
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


def _normalized_point_dipole_profile(
    dist_sq: torch.Tensor, width: float, control_volumes: torch.Tensor
) -> torch.Tensor:
    width_sq = 2.0 * float(width) ** 2
    profile = torch.exp(-dist_sq / width_sq)
    integrated_mass = torch.sum(profile * control_volumes)
    if not integrated_mass > 0.0:
        raise ValueError("Gaussian point-dipole profile must have strictly positive integrated mass.")
    return profile / integrated_mass


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


def _yee_component_coords(solver, field_name: str, *, dtype=None) -> tuple[torch.Tensor, ...]:
    scene = solver.scene
    names = {
        "Ex": ("x_half", "y", "z"),
        "Ey": ("x", "y_half", "z"),
        "Ez": ("x", "y", "z_half"),
        "Hx": ("x", "y_half", "z_half"),
        "Hy": ("x_half", "y", "z_half"),
        "Hz": ("x_half", "y_half", "z"),
    }[field_name]
    if dtype is None:
        return tuple(getattr(scene, name).to(device=solver.device, dtype=solver.Ex.dtype) for name in names)
    return tuple(torch.tensor(
        getattr(scene, name + ("64" if name.endswith("_half") else "_nodes64")),
        device=solver.device, dtype=dtype,
    ) for name in names)


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


def _axis_control_overlap(coords: torch.Tensor, lo: float, hi: float, *, bounds=None) -> torch.Tensor:
    """Fraction of every sample's dual control interval covered by ``[lo, hi]``."""
    count = int(coords.numel())
    if count <= 1:
        if bounds is not None:
            return torch.full_like(coords, max(0.0, min(hi, bounds[1]) - max(lo, bounds[0])) / (bounds[1] - bounds[0]))
        return torch.ones_like(coords) if lo <= float(coords[0]) <= hi else torch.zeros_like(coords)
    midpoints = 0.5 * (coords[:-1] + coords[1:])
    left = torch.cat((coords[:1] - 0.5 * (coords[1:2] - coords[:1]), midpoints))
    right = torch.cat((midpoints, coords[-1:] + 0.5 * (coords[-1:] - coords[-2:-1])))
    if bounds is not None:
        left[0], right[-1] = bounds
    overlap = torch.clamp(
        torch.minimum(right, torch.as_tensor(hi, device=coords.device, dtype=coords.dtype))
        - torch.maximum(left, torch.as_tensor(lo, device=coords.device, dtype=coords.dtype)),
        min=0.0,
    )
    return overlap / (right - left)


def _yee_box_overlap(solver, field_name: str, lo, hi, *, current_dataset=False):
    """Return a compact Yee patch containing volume-averaged box indicator weights."""
    if current_dataset and any(float(lo[a]) == float(hi[a]) for a in range(3)):
        nearest, _ = _region_index_range(solver, field_name, lo, hi)
    axis_weights = []
    # Use declared grid coordinates for small fractional cells, whose coverage
    # loses relative precision when subtracting nearby FP32 coordinates.
    dtype = torch.float64 if current_dataset else None
    for axis, coords in enumerate(_yee_component_coords(solver, field_name, dtype=dtype)):
        if current_dataset and float(lo[axis]) == float(hi[axis]):
            # A singleton retains the historical nearest-plane density measure.
            weights = torch.zeros_like(coords)
            weights[nearest[axis]] = 1.0
        else:
            lower, upper = float(lo[axis]), float(hi[axis])
            if current_dataset:
                bounds = solver.scene.domain.bounds[axis]
                lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
            control_bounds = None
            if current_dataset and solver._COMPONENT_HALF_OFFSET_AXES[field_name][axis]:
                # A terminal half-step control cell ends at the grid boundary,
                # including when adjacent nonuniform spacings differ.
                nodes = getattr(solver.scene, "xyz"[axis] + "_nodes64")
                control_bounds = (float(nodes[0]), float(nodes[-1]))
            weights = _axis_control_overlap(coords, lower, upper, bounds=control_bounds)
        axis_weights.append(weights)
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


_WRAPPED_AXIS_KINDS = frozenset({"periodic", "bloch"})


def _point_dipole_has_single_image(solver, position, cutoff) -> bool:
    """Whether the unshifted source is the only periodic image inside the domain.

    A shifted image ``position + n * L`` contributes only when its cutoff window
    reaches the domain. When no ``n = +-1`` image does, compiling the unshifted
    source alone is already the periodic profile and stays bit-identical.
    """
    scene = solver.scene
    boundary = scene.boundary
    nodes = (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64)
    sizes = (solver.Nx, solver.Ny, solver.Nz)
    for axis, name in enumerate("xyz"):
        if boundary.axis_kind(name) not in _WRAPPED_AXIS_KINDS:
            continue
        length = scene.domain_range[2 * axis + 1] - scene.domain_range[2 * axis]
        for shift in (-1, 1):
            image = position[axis] + shift * length
            start, stop = _axis_index_window(nodes[axis], image - cutoff, image + cutoff, sizes[axis])
            if stop > start:
                return False
    return True


def _bracket_weights(coords64: np.ndarray, position: float) -> tuple[list[int], list[float]]:
    """Linear-interpolation weights of ``position`` between bracketing samples."""
    upper = int(np.searchsorted(coords64, float(position), side="left"))
    if upper <= 0:
        return [0], [1.0]
    if upper >= coords64.size:
        return [int(coords64.size - 1)], [1.0]
    lower = upper - 1
    span = float(coords64[upper] - coords64[lower])
    if span <= 0.0:
        return [lower], [1.0]
    fraction = (float(position) - float(coords64[lower])) / span
    if fraction <= 1e-12:
        return [lower], [1.0]
    if fraction >= 1.0 - 1e-12:
        return [upper], [1.0]
    return [lower, upper], [1.0 - fraction, fraction]


def _wrapped_axis_profile(nodes64, half64, staggered, position, cutoff, width, ideal, period_phase):
    """Per-axis point-dipole factor on the unique samples of a periodic axis.

    The source profile lives on the unbounded periodic lattice: every sample
    ``c_u + n L`` of the same cutoff window as the open-boundary compile carries
    its Gaussian (or linear ``ideal``) weight, and folds onto the unique in-domain
    sample ``u`` with the Floquet factor ``exp(-i k n L)``. Each in-domain sample
    therefore receives the summed weight once. The node at ``x_hi`` duplicates
    the node at ``x_lo``; only the latter is a unique sample and the source
    kernel copies it to its periodic partner.

    Returns ``(start, factors)`` with complex factors on ``[start, start + len)``,
    already divided by the axis mass (Gaussian) or the periodic control width
    (``ideal``), or ``None`` when the window holds no sample.
    """
    nodes64 = np.asarray(nodes64, dtype=np.float64)
    length = float(nodes64[-1] - nodes64[0])
    unique = np.asarray(half64, dtype=np.float64) if staggered else nodes64[:-1]
    count = int(unique.size)
    origin = float(nodes64[0])
    first_shift = int(math.floor((float(position) - cutoff - origin) / length)) - 1
    last_shift = int(math.floor((float(position) + cutoff - origin) / length)) + 1
    shifts = np.repeat(np.arange(first_shift, last_shift + 1, dtype=np.int64), count)
    lattice_nodes = np.tile(nodes64[:-1], last_shift - first_shift + 1) + shifts * length
    lattice_nodes = np.append(lattice_nodes, origin + (last_shift + 1) * length)
    samples = np.tile(unique, last_shift - first_shift + 1) + shifts * length
    unique_index = np.tile(np.arange(count, dtype=np.int64), last_shift - first_shift + 1)
    # Periodic control width of every unique sample (no open-boundary half cells).
    wrapped = np.concatenate((unique[-1:] - length, unique, unique[:1] + length))
    unique_widths = 0.5 * (wrapped[2:] - wrapped[:-2])

    if ideal:
        indices, weights = _bracket_weights(samples, position)
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(weights, dtype=np.float64) / unique_widths[unique_index[indices]]
    else:
        # Node window of the open-boundary compile, applied on the lattice.
        low = int(np.searchsorted(lattice_nodes, float(position) - cutoff, side="right")) - 1
        high = int(np.searchsorted(lattice_nodes, float(position) + cutoff, side="left"))
        indices = np.arange(low, high if staggered else high + 1, dtype=np.int64)
        if indices.size == 0:
            return None
        gaussian = np.exp(-((samples[indices] - float(position)) ** 2) / (2.0 * float(width) ** 2))
        mass = float(np.sum(gaussian * unique_widths[unique_index[indices]]))
        if not mass > 0.0:
            return None
        # Per-axis mass is a length; no absolute floor, so sub-mm widths keep unit moment.
        values = gaussian / mass

    phases = np.power(complex(period_phase).conjugate(), shifts[indices].astype(np.float64))
    factors = np.zeros(count, dtype=np.complex128)
    np.add.at(factors, unique_index[indices], values * phases)
    support = np.nonzero(factors != 0.0)[0]
    if support.size == 0:
        return None
    if factors[0] != 0.0 and factors[-1] != 0.0:
        return 0, factors
    return int(support[0]), factors[int(support[0]) : int(support[-1]) + 1]


def _open_axis_profile(nodes64, half64, staggered, size_cap, position, cutoff, width, ideal):
    """Per-axis factor on a non-periodic axis, matching the per-image compile."""
    start, stop = _axis_index_window(nodes64, position - cutoff, position + cutoff, size_cap)
    if staggered:
        stop = min(stop, size_cap - 1)
    if stop <= start:
        return None
    coords = np.asarray(half64 if staggered else nodes64, dtype=np.float64)
    count = int(coords.size)
    widths = np.empty(count, dtype=np.float64)
    if count <= 1:
        widths[:] = 1.0
    else:
        diffs = coords[1:] - coords[:-1]
        widths[0] = 0.5 * diffs[0]
        widths[-1] = 0.5 * diffs[-1]
        widths[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    window = coords[start:stop]
    window_widths = widths[start:stop]
    if ideal:
        indices, weights = _bracket_weights(window, position)
        factors = np.zeros(stop - start, dtype=np.float64)
        for index, weight in zip(indices, weights):
            factors[index] += weight / window_widths[index]
    else:
        gaussian = np.exp(-((window - float(position)) ** 2) / (2.0 * float(width) ** 2))
        mass = float(np.sum(gaussian * window_widths))
        if not mass > 0.0:
            return None
        factors = gaussian / mass
    return start, factors.astype(np.complex128)


def _prepare_wrapped_point_dipole_source(solver, source, *, source_index, cutoff):
    """Compile a point dipole whose cutoff window wraps across a periodic face.

    The profile is separable per axis, so periodic folding and Floquet phases are
    applied to 1D factors and the component receives one patch (two for a complex
    Bloch profile: real and imaginary parts) regardless of the period.
    """
    scene = solver.scene
    boundary = scene.boundary
    position = tuple(float(value) for value in source["position"])
    width = float(source["width"])
    ideal = source.get("profile", "gaussian") == "ideal"
    polarization = source["polarization"]
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    dtype = solver.Ex.dtype
    nodes = (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64)
    halves = (scene.x_half64, scene.y_half64, scene.z_half64)
    sizes = (solver.Nx, solver.Ny, solver.Nz)
    wavevector = getattr(solver, "resolved_bloch_wavevector", None)
    if wavevector is None:
        wavevector = scene.bloch_wavevector
    axis_kinds = tuple(boundary.axis_kind(name) for name in "xyz")
    period_phases = []
    for axis, kind in enumerate(axis_kinds):
        length = scene.domain_range[2 * axis + 1] - scene.domain_range[2 * axis]
        period_phases.append(np.exp(1j * float(wavevector[axis]) * length) if kind == "bloch" else 1.0 + 0.0j)

    for field_name, component_axis in (("Ez", 2), ("Ex", 0), ("Ey", 1)):
        component = polarization[component_axis]
        if component == 0:
            continue
        profiles = []
        for axis in range(3):
            staggered = axis == component_axis
            if axis_kinds[axis] in _WRAPPED_AXIS_KINDS:
                profile = _wrapped_axis_profile(
                    nodes[axis], halves[axis], staggered, position[axis], cutoff, width, ideal, period_phases[axis]
                )
            else:
                profile = _open_axis_profile(
                    nodes[axis], halves[axis], staggered, sizes[axis], position[axis], cutoff, width, ideal
                )
            if profile is None:
                break
            profiles.append(profile)
        else:
            offsets = tuple(start for start, _ in profiles)
            fx, fy, fz = (factors for _, factors in profiles)
            outer = fx[:, None, None] * fy[None, :, None] * fz[None, None, :]
            region = tuple(slice(start, start + factors.size) for start, factors in profiles)
            symmetry_scale = _symmetry_plane_source_scale(scene, field_name, position)
            coefficient = (-solver.dt * component * symmetry_scale) / getattr(solver, "eps_" + field_name)[region]
            for part, phase_real, phase_imag in ((outer.real, 1.0, 0.0), (outer.imag, 0.0, 1.0)):
                if not np.any(part):
                    continue
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name=field_name,
                    offsets=offsets,
                    patch=coefficient * torch.as_tensor(part, device=coefficient.device, dtype=dtype),
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_position=position,
                    source_symmetry_scale=symmetry_scale,
                    source_time=source_time,
                    omega=source_omega,
                )
                if not np.any(outer.imag):
                    # Real folded factors let shard-interface rebuilds keep the wrap.
                    solver._source_terms[-1]["axis_factors"] = (fx.real, fy.real, fz.real)


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
    if not _point_dipole_has_single_image(solver, source["position"], cutoff):
        _prepare_wrapped_point_dipole_source(solver, source, source_index=source_index, cutoff=cutoff)
        return
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
                symmetry_scale = _symmetry_plane_source_scale(scene, "Ez", image_position)
                source_patch *= symmetry_scale
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name="Ez",
                    offsets=patch_offsets,
                    patch=source_patch,
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_position=image_position,
                    source_symmetry_scale=symmetry_scale,
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
                symmetry_scale = _symmetry_plane_source_scale(scene, "Ex", image_position)
                source_patch *= symmetry_scale
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name="Ex",
                    offsets=patch_offsets,
                    patch=source_patch,
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_position=image_position,
                    source_symmetry_scale=symmetry_scale,
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
                symmetry_scale = _symmetry_plane_source_scale(scene, "Ey", image_position)
                source_patch *= symmetry_scale
                append_source_term(
                    solver._source_terms,
                    solver,
                    field_name="Ey",
                    offsets=patch_offsets,
                    patch=source_patch,
                    phase_real=phase_real,
                    phase_imag=phase_imag,
                    source_index=source_index,
                    source_position=image_position,
                    source_symmetry_scale=symmetry_scale,
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
    overlap = _yee_box_overlap(solver, field_name, region_lo, region_hi, current_dataset=True)
    if overlap is None:
        return
    start, stop, weights = overlap
    query_axes = tuple(
        coords[start[a]:stop[a]].clamp(region_lo[a], region_hi[a])
        for a, coords in enumerate(_yee_component_coords(solver, field_name, dtype=torch.float64))
    )
    positions = torch.stack(torch.meshgrid(*query_axes, indexing="ij"), dim=-1).to(solver.Ex.dtype)
    sampled = _trilinear_sample(values, axes, positions) * weights.to(solver.Ex.dtype)
    if torch.max(torch.abs(sampled)).item() == 0:
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


def _sheet_axis_quadrature(coords, bounds, lo, hi):
    """Clipped control cells and their fractional physical-aperture coverage."""
    mid = 0.5 * (coords[:-1] + coords[1:])
    left = torch.cat((coords.new_tensor([bounds[0]]), mid))
    right = torch.cat((mid, coords.new_tensor([bounds[1]])))
    covered = (right.clamp(max=hi) - left.clamp(min=lo)).clamp(min=0.0)
    indices = torch.nonzero(covered > 0.0).flatten()
    start, stop = int(indices[0]), int(indices[-1]) + 1
    return start, stop, covered[start:stop], (covered / (right - left))[start:stop]


def _sheet_axis_modes(coords, widths, *, bounds, odd_faces=(False, False), period_phase=None):
    """Device-native weighted modes on one aperture's control cells.

    Reflecting the sampled profile at the aperture edge keeps continuation
    inside its physical support. Full-span axes use their actual wall parity
    or periodic phase. Uniform Yee axes recover the exact sine/cosine/Floquet
    symbols, including nodal half cells and duplicated periodic endpoints.
    """
    root = widths.sqrt()
    if period_phase is not None:
        duplicate = bool(coords[0] == bounds[0] and coords[-1] == bounds[1])
        unique = coords[:-1] if duplicate else coords
        mass = widths[:-1].clone() if duplicate else widths
        if duplicate:
            mass[0] += widths[-1]
        conductance = 1.0 / (unique[1:] - unique[:-1])
        diagonal = torch.nn.functional.pad(conductance, (0, 1)) + torch.nn.functional.pad(conductance, (1, 0))
        stiffness = (torch.diag(diagonal) - torch.diag(conductance, 1) - torch.diag(conductance, -1)).to(torch.complex128)
        wrap = 1.0 / (bounds[1] - bounds[0] - unique[-1] + unique[0])
        stiffness[0, 0] += wrap
        stiffness[-1, -1] += wrap
        stiffness[0, -1] -= complex(period_phase).conjugate() * wrap
        stiffness[-1, 0] -= period_phase * wrap
        eigenvalues, vectors = torch.linalg.eigh(stiffness / mass.sqrt()[:, None] / mass.sqrt()[None, :])
        physical = vectors / mass.sqrt()[:, None]
        if duplicate:
            physical = torch.cat((physical, period_phase * physical[:1]), dim=0)
        return eigenvalues.clamp(min=0.0), root[:, None] * physical, root

    conductance = 1.0 / (coords[1:] - coords[:-1])
    diagonal = torch.nn.functional.pad(conductance, (0, 1)) + torch.nn.functional.pad(conductance, (1, 0))
    stiffness = torch.diag(diagonal) - torch.diag(conductance, 1) - torch.diag(conductance, -1)
    start, stop = 0, len(coords)
    for side, odd in enumerate(odd_faces):
        if not odd:
            continue
        index = 0 if side == 0 else len(coords) - 1
        distance = abs(float(coords[index]) - bounds[side])
        if distance == 0.0:
            if side == 0:
                start = 1
            else:
                stop -= 1
        else:
            stiffness[index, index] += 1.0 / distance
    active_root = root[start:stop]
    eigenvalues, active = torch.linalg.eigh(stiffness[start:stop, start:stop] / active_root[:, None] / active_root[None, :])
    vectors = torch.nn.functional.pad(active, (0, 0, start, len(coords) - stop))
    return eigenvalues.clamp(min=0.0), vectors, root


def _common_plane_sheet_weights(kh, fraction):
    """Two-way Yee radiation weights for a bracketing sheet.

    A unit Yee sheet radiates with the factor sec(k_z h/2). For bracketing
    sheets a,b, w_a = sin(k_z (b-z0))/(2 sin(k_z h/2)) and its partner cancel
    that factor and translate BOTH outgoing waves to z0. The sinc form also
    includes grazing incidence without a singular quotient.
    """
    return fraction * torch.sinc(kh * fraction / math.pi) / torch.sinc(kh / (2.0 * math.pi))


def _prepare_custom_field_source(solver, source, *, source_index):
    dataset = source["dataset"]
    normal_axis = source["normal_axis"]
    source_time = source["source_time"]
    source_omega = 2.0 * np.pi * float(source_time["frequency"])
    axes = _dataset_axes_tensors(solver, dataset)
    region_lo, region_hi = _dataset_region_bounds(dataset)
    axis_index = _AXIS_TO_INDEX[normal_axis]
    plane_coord = 0.5 * (float(region_lo[axis_index]) + float(region_hi[axis_index]))
    transverse = [index for index in range(3) if index != axis_index]
    scene = solver.scene
    nodes = (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64)
    halves = (scene.x_half64, scene.y_half64, scene.z_half64)
    # The low-face specification fixes the physical J=n x H, M=-n x E signs.
    electric, magnetic = build_discrete_tfsf_specs((1, 1, 1), (2, 2, 2))
    specs = electric[_FACE_SPEC_RANGES[(normal_axis, "low")]] + magnetic[_FACE_SPEC_RANGES[(normal_axis, "low")]]
    K = 2.0 * math.sin(source_omega * solver.dt / 2.0) / (solver.c * solver.dt)

    for template in specs:
        incident = template["incident_name"]
        if incident not in dataset.components:
            continue
        values = torch.as_tensor(dataset.components[incident], device=solver.device, dtype=solver.Ex.dtype)
        field = template["field_name"]
        field_axis = "xyz".index(field[-1].lower())
        coordinates = tuple(torch.as_tensor(
            halves[a] if (a == field_axis) == field.startswith("E") else nodes[a],
            device=solver.device, dtype=torch.float64,
        ) for a in range(3))
        normal = coordinates[axis_index]
        upper = int(torch.searchsorted(normal, normal.new_tensor(plane_coord)).clamp(1, len(normal) - 1))
        lower = upper - 1
        h = float(normal[upper] - normal[lower])
        starts, stops, overlaps, modes = {}, {}, {}, []
        for a in transverse:
            start, stop, widths, overlap = _sheet_axis_quadrature(
                coordinates[a], (nodes[a][0], nodes[a][-1]), region_lo[a], region_hi[a],
            )
            starts[a], stops[a], overlaps[a] = start, stop, overlap
            bounds = (float(nodes[a][0]), float(nodes[a][-1]))
            touches = (region_lo[a] <= bounds[0], region_hi[a] >= bounds[1])
            kinds = scene.boundary.axis_face_kinds("xyz"[a])
            even_pec = field.startswith("E") == (a == field_axis)
            odd_faces = tuple(touch and ((kind == "pec" and not even_pec) or (kind == "pmc" and even_pec))
                              for touch, kind in zip(touches, kinds))
            phase = None
            if all(touches) and scene.boundary.axis_kind("xyz"[a]) in _WRAPPED_AXIS_KINDS:
                phase = (complex(float(solver.boundary_phase_cos[a]), float(solver.boundary_phase_sin[a]))
                         if kinds[0] == "bloch" else 1.0 + 0.0j)
            modes.append(_sheet_axis_modes(coordinates[a][start:stop], widths,
                                          bounds=bounds, odd_faces=odd_faces, period_phase=phase))
        query_axes = [None] * 3
        for a in transverse:
            # Boundary-cell quadrature uses the nearest point INSIDE the sample
            # span. No dataset value is extrapolated outside the aperture.
            query_axes[a] = coordinates[a][starts[a]:stops[a]].clamp(region_lo[a], region_hi[a]).to(solver.Ex.dtype)
        query_axes[axis_index] = solver.Ex.new_tensor([plane_coord])
        positions = torch.stack(torch.meshgrid(*query_axes, indexing="ij"), dim=-1)
        sampled = _trilinear_sample(values, axes, positions).squeeze(axis_index).to(torch.float64)
        if sampled.abs().max().item() == 0:
            continue
        la, va, ra = modes[0]
        lb, vb, rb = modes[1]
        va, vb = va.to(torch.complex128), vb.to(torch.complex128)
        spectrum = va.mH @ (sampled * ra[:, None] * rb[None, :]).to(torch.complex128) @ vb.conj()
        kh = 2.0 * torch.asin(torch.sqrt(((K * K - la[:, None] - lb[None, :]) * (h * h / 4.0)).to(torch.complex128)))
        coverage = overlaps[transverse[0]][:, None] * overlaps[transverse[1]][None, :]
        sign = template["sign"] * (-1.0 if incident.startswith("H") else 1.0)
        curl_map = E_CURL_ATTR if field.startswith("E") else H_CURL_ATTR
        term_list = solver._electric_source_terms if field.startswith("E") else solver._magnetic_source_terms
        for index, distance in ((lower, float(normal[upper]) - plane_coord), (upper, plane_coord - float(normal[lower]))):
            weights = _common_plane_sheet_weights(kh, distance / h)
            profile = (va @ (spectrum * weights) @ vb.T) / ra[:, None] / rb[None, :]
            profile = (profile * coverage).unsqueeze(axis_index)
            offsets = tuple(index if a == axis_index else starts[a] for a in range(3))
            spec = dict(template, offsets=offsets, shape=profile.shape)
            coefficient = sign * slice_coeff_patch(solver, spec, curl_map)
            patches = (profile.real.to(solver.Ex.dtype) * coefficient,
                       profile.imag.to(solver.Ex.dtype) * coefficient)
            # Real and quadrature channels use the existing delayed-waveform
            # path, including table waveforms and the explicit adjoint replay.
            for patch, delay in zip(patches, (0.0, math.pi / (2.0 * source_omega))):
                term = build_term_from_profile(
                    solver, field_name=field, offsets=offsets,
                    scale=patch,
                    delay_patch=torch.full_like(coefficient, delay), activation_delay_patch=None,
                    source_time=source_time, omega=source_omega, source_index=source_index,
                    profile_floor=0.0,
                )
                if term is not None:
                    term_list.append(term)


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
    unit_power = aperture_area * incidence_cosine / (2.0 * ETA_0)
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
    unit_power = math.pi * waist_u * waist_v * incidence_cosine / (4.0 * ETA_0)
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
    from ..runtime.stepping import electric_update_support

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

    # Keep the full launch aperture and magnetic sheet. Only electric rows
    # outside the curl update support would accumulate a frozen sheet current.
    supported_specs = []
    for spec in electric_specs:
        support = electric_update_support(solver, spec["field_name"])
        offsets = tuple(max(start, lo) for start, (lo, _) in zip(spec["offsets"], support))
        stops = tuple(min(start + size, hi) for start, size, (_, hi)
                      in zip(spec["offsets"], spec["shape"], support))
        shape = tuple(stop - start for start, stop in zip(offsets, stops))
        if all(size > 0 for size in shape):
            sample_offsets = tuple(sample + start - original for sample, start, original
                                   in zip(spec["sample_offsets"], offsets, spec["offsets"]))
            supported_specs.append({**spec, "offsets": offsets, "shape": shape,
                                    "sample_offsets": sample_offsets, "sample_shape": shape})
    electric_specs = supported_specs

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
        for name, value in magnetic_physical_vector(direction, source["polarization"], ETA_0).items()
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
        # Shared face templates encode the enclosure correction orientation;
        # an open launch sheet needs the opposite sign for +E_inc, as in the
        # TFSF counterpart in _set_tfsf_state.
        scale = -spec["sign"] * component_scale * spatial_amplitude * coeff_patch
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


def _mode_normal_stagger_power_factor(solver, source, mode_data) -> float | torch.Tensor:
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
    beta_complex = mode_data.get("beta_complex")
    if mode_data.get("mode_solver_kind") == "vector_dense_torch":
        factor = torch.cos(mode_data["beta_tensor"].abs() * normal_half_offset)
        return factor if float(factor.detach()) > 0.0 else 1.0
    if beta_complex is not None:
        from .modes.common import _discrete_mode_profile_power
        profiles = mode_data["component_profiles"]
        import cmath
        multiplier = cmath.cos(beta_complex * normal_half_offset)
        interpolated = {name: value * multiplier if name.startswith("H") else value
                        for name, value in profiles.items()}
        power_args = dict(coords_u=mode_data["coords_u"],coords_v=mode_data["coords_v"],
                          normal_axis=normal_axis,staggered=mode_data.get("staggered_profiles",False))
        factor = float((_discrete_mode_profile_power(interpolated,**power_args) /
                        _discrete_mode_profile_power(profiles,**power_args)).item())
    else:
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
    phase_speed = (source_omega / mode_data["beta_tensor"].abs()
                   if mode_data.get("mode_solver_kind") == "vector_dense_torch"
                   else source_omega / max(abs(float(mode_data["beta"])), 1.0e-30))
    normal_stagger_power_factor = _mode_normal_stagger_power_factor(solver, source, mode_data)
    mode_injection_scale = (torch.rsqrt(normal_stagger_power_factor)
                            if torch.is_tensor(normal_stagger_power_factor)
                            else 1.0 / math.sqrt(normal_stagger_power_factor))
    source["normal_stagger_power_factor"] = float(
        normal_stagger_power_factor.detach() if torch.is_tensor(normal_stagger_power_factor)
        else normal_stagger_power_factor)
    source["prepared_mode_power"] = 1.0
    profiles = mode_data["component_profiles"]
    # Mode builders already cast their float64/complex128 outputs to field dtype.
    # Prune only roundoff from that original solve, using float64 epsilon.
    profile_floors = {
        family: mode_injection_scale * _component_roundoff_floor(
            [profile for name, profile in profiles.items() if name.startswith(family)],
            dtype=torch.float64,
        )
        for family in ("E", "H")
    }

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
        profile_floor = profile_floors[spec["incident_name"][0]]
        if torch.max(torch.abs(component_profile)).item() <= profile_floor:
            return None
        delay_patch = (
            float(direction_sign)
            * (positions[..., normal_axis_index] - float(plane_coordinate))
            / phase_speed
        )
        scale = spec["sign"] * component_profile * coeff_patch
        beta_complex = mode_data.get("beta_complex")
        if beta_complex is not None:
            displacement = float(direction_sign) * (positions[..., normal_axis_index] - float(plane_coordinate))
            scale = scale * torch.exp(-float(beta_complex.imag) * displacement)
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


_ELECTRIC_CURRENT_SOURCE_KINDS = frozenset({"point_dipole", "uniform_current", "custom_current"})


def _append_full_aniso_current_terms(solver, compiled_sources):
    """Compile transverse responses to impressed J through the Yee material map.

    Source patches already contain the diagonal response -dt J / eps_eff.
    Recover their material-independent current and contract the two remaining
    inverse-tensor rows once, retaining the original temporal envelope. Field
    corrections and magnetic currents are not electric constitutive drives.
    """
    if not getattr(solver, "full_aniso_enabled", False):
        return
    import torch.nn.functional as functional

    for term in tuple(solver._source_terms):
        source_index = term.get("source_index")
        if source_index is None or compiled_sources[source_index]["kind"] not in _ELECTRIC_CURRENT_SOURCE_KINDS:
            continue
        source_name = term["field_name"]
        source_axis = "xyz".index(source_name[-1].lower())
        patch = term["patch"]
        region = tuple(slice(start, start + size) for start, size in zip(term["offsets"], patch.shape))
        current_patch = -patch * getattr(solver, "eps_" + source_name)[region] / solver.dt
        shape = getattr(solver, source_name).shape
        periodic = tuple(solver.scene.boundary.axis_kind(axis) == "periodic" for axis in "xyz")
        if not any(periodic):
            # A local current couples only to adjacent transverse edges. Compile
            # bounded patches without allocating full-domain source arrays.
            for target_axis, target_name in enumerate(("Ex", "Ey", "Ez")):
                if target_axis == source_axis:
                    continue
                coefficients = getattr(solver, f"ce{'xyz'[target_axis]}_aniso_{'xyz'[source_axis]}")
                starts = list(term["offsets"])
                ends = [start + size for start, size in zip(starts, patch.shape)]
                starts[target_axis] = max(0, starts[target_axis] - 1)
                ends[target_axis] = min(coefficients.shape[target_axis], ends[target_axis])
                ends[source_axis] = min(coefficients.shape[source_axis], ends[source_axis] + 1)
                target_region = tuple(slice(start, end) for start, end in zip(starts, ends))
                local_coefficients = coefficients[target_region]
                response = torch.zeros_like(local_coefficients[..., 0])
                for target_side in range(2):
                    for source_side in range(2):
                        indices, masks = [], []
                        for axis in range(3):
                            offset = target_side if axis == target_axis else source_side - 1 if axis == source_axis else 0
                            index = torch.arange(starts[axis], ends[axis], device=patch.device) + offset - term["offsets"][axis]
                            view = [1, 1, 1]
                            view[axis] = -1
                            masks.append(((index >= 0) & (index < patch.shape[axis])).reshape(view))
                            indices.append(index.clamp(0, patch.shape[axis] - 1).reshape(view))
                        sample = current_patch[tuple(indices)] * (masks[0] & masks[1] & masks[2])
                        slot = 2 * target_side + source_side if target_axis < source_axis else 2 * source_side + target_side
                        response = response - local_coefficients[..., slot] * sample
                transverse = dict(term)
                transverse.update(field_name=target_name, offsets=tuple(starts), patch=response.contiguous(), periodic_compiled=True)
                solver._source_terms.append(transverse)
            continue
        padding = tuple(value for axis in (2, 1, 0) for value in (
            term["offsets"][axis], shape[axis] - term["offsets"][axis] - patch.shape[axis]))
        current = functional.pad(current_patch, padding)
        # The source kernel copies tangential node endpoints to their periodic
        # partners. Compile those same copies before the cross-component gather.
        for axis in range(3):
            if periodic[axis] and axis != source_axis:
                boundary = current.narrow(axis, 0, 1) + current.narrow(axis, shape[axis] - 1, 1)
                current = torch.cat((boundary, current.narrow(axis, 1, shape[axis] - 2), boundary), dim=axis)
        zero_shape = list(current.shape)
        zero_shape[source_axis] = 1
        zero = current.new_zeros(zero_shape)
        previous = current.narrow(source_axis, shape[source_axis] - 1, 1) if periodic[source_axis] else zero
        following = current.narrow(source_axis, 0, 1) if periodic[source_axis] else zero
        neighbors = (torch.cat((previous, current), dim=source_axis),
                     torch.cat((current, following), dim=source_axis))
        for target_axis, target_name in enumerate(("Ex", "Ey", "Ez")):
            if target_axis == source_axis:
                continue
            coefficients = getattr(solver, f"ce{'xyz'[target_axis]}_aniso_{'xyz'[source_axis]}")
            response = torch.zeros_like(coefficients[..., 0])
            for target_side in range(2):
                selector = [slice(None)] * 3
                selector[target_axis] = slice(target_side, shape[target_axis] - 1 + target_side)
                for source_side in range(2):
                    slot = 2 * target_side + source_side if target_axis < source_axis else 2 * source_side + target_side
                    response = response - coefficients[..., slot] * neighbors[source_side][tuple(selector)]
            transverse = dict(term)
            transverse.update(field_name=target_name, offsets=(0, 0, 0),
                              patch=response.contiguous(), periodic_compiled=True)
            solver._source_terms.append(transverse)


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
        source["reference_waveform_kind"] = source["source_time"]["kind"]
        if source["kind"] == "point_dipole":
            source["reference_provider_kind"] = "additive_J"
            _prepare_point_dipole_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "mode_source":
            source["reference_provider_kind"] = "mode_surface"
            _prepare_mode_surface_source(solver, source, source_index=source_index)
            continue
        if source["kind"] in {"plane_wave", "gaussian_beam", "astigmatic_gaussian_beam"}:
            source["reference_provider_kind"] = "soft_equivalent_sheet"
            _prepare_power_normalized_surface_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "uniform_current":
            source["reference_provider_kind"] = "additive_J"
            _prepare_uniform_current_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "custom_current":
            components = source["dataset"].components
            source["reference_provider_kind"] = ("additive_J" if all(k.startswith("J") for k in components)
                else "magnetic_M" if all(k.startswith("M") for k in components) else "mixed_JM")
            _prepare_custom_current_source(solver, source, source_index=source_index)
            continue
        if source["kind"] == "custom_field":
            source["reference_provider_kind"] = "custom_field_surface"
            _prepare_custom_field_source(solver, source, source_index=source_index)
            continue
        source["reference_provider_kind"] = "soft_equivalent_sheet"
        _prepare_surface_source(solver, source, source_index=source_index)
    _append_full_aniso_current_terms(solver, compiled_sources)
    prepare_source_gathers(solver)


def inject_source_terms(solver, n=None, signal=None, time_value=None):
    if solver._source_terms is None:
        initialize_source_terms(solver)
    if not solver._source_terms:
        return

    if time_value is None:
        if n is None:
            raise ValueError("Either n or time_value must be provided for source injection.")
        time_value = injection_time("electric_current", n * solver.dt, solver.dt)

    apply_compiled_source_terms(
        solver,
        solver._source_terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        signal=signal,
    )
