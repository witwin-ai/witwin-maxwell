from __future__ import annotations

import math

import torch

from .spatial import gaussian_beam_profile, plane_wave_profile
from .tfsf_specs import reference_sample_axis_code
from .temporal import build_source_term


def nearest_index(axis_coords, value: float) -> int:
    return int(torch.argmin(torch.abs(axis_coords - float(value))).item())


def resolve_bounds_indices(scene, bounds):
    lower = []
    upper = []
    for axis_coords, axis_bounds in zip((scene.x, scene.y, scene.z), bounds):
        lower.append(nearest_index(axis_coords, axis_bounds[0]))
        upper.append(nearest_index(axis_coords, axis_bounds[1]))
    return tuple(lower), tuple(upper)


def validate_bounds(solver, lower, upper):
    shape = (solver.Nx, solver.Ny, solver.Nz)
    for axis, lo, hi, size in zip("xyz", lower, upper, shape):
        low_margin = solver.scene.pml_thickness_for_face(axis, "low") + 2
        high_margin = solver.scene.pml_thickness_for_face(axis, "high") + 2
        if hi <= lo + 1:
            raise ValueError(f"TFSF bounds must span at least two cells along {axis}.")
        if lo < low_margin or hi > size - high_margin - 1:
            raise ValueError("TFSF bounds must lie strictly inside the non-PML simulation region.")


def validate_background_is_vacuum(solver, lower, upper, tol=1e-5):
    eps = solver.epsilon_r
    mu = solver.mu_r
    slices = (
        (slice(lower[0], lower[0] + 1), slice(lower[1], upper[1] + 1), slice(lower[2], upper[2] + 1)),
        (slice(upper[0], upper[0] + 1), slice(lower[1], upper[1] + 1), slice(lower[2], upper[2] + 1)),
        (slice(lower[0], upper[0] + 1), slice(lower[1], lower[1] + 1), slice(lower[2], upper[2] + 1)),
        (slice(lower[0], upper[0] + 1), slice(upper[1], upper[1] + 1), slice(lower[2], upper[2] + 1)),
        (slice(lower[0], upper[0] + 1), slice(lower[1], upper[1] + 1), slice(lower[2], lower[2] + 1)),
        (slice(lower[0], upper[0] + 1), slice(lower[1], upper[1] + 1), slice(upper[2], upper[2] + 1)),
    )
    for face_slice in slices:
        eps_face = eps[face_slice]
        mu_face = mu[face_slice]
        if not torch.allclose(eps_face, torch.ones_like(eps_face), atol=tol, rtol=0.0):
            raise ValueError("TFSF boundary must remain in vacuum on all faces.")
        if not torch.allclose(mu_face, torch.ones_like(mu_face), atol=tol, rtol=0.0):
            raise ValueError("TFSF boundary must remain in vacuum on all faces.")


def solve_numerical_wavenumber(solver, direction, delta_attr_map):
    omega = float(solver.source_omega)
    if omega <= 0.0:
        return 0.0

    target = math.sin(0.5 * omega * float(solver.dt)) ** 2 / (float(solver.c) ** 2 * float(solver.dt) ** 2)
    if target <= 0.0:
        return omega / float(solver.c)

    axis_limits = []
    for axis, component in zip("xyz", direction):
        abs_component = abs(float(component))
        if abs_component <= 1e-12:
            continue
        delta = float(getattr(solver, delta_attr_map[axis]))
        axis_limits.append(2.0 * math.pi / max(abs_component * delta, 1e-30))
    if not axis_limits:
        return omega / float(solver.c)

    def dispersion_residual(k_mag):
        total = 0.0
        for axis, component in zip("xyz", direction):
            abs_component = abs(float(component))
            if abs_component <= 1e-12:
                continue
            delta = float(getattr(solver, delta_attr_map[axis]))
            total += math.sin(0.5 * k_mag * abs_component * delta) ** 2 / (delta * delta)
        return total - target

    low = 0.0
    high = min(axis_limits)
    for _ in range(80):
        mid = 0.5 * (low + high)
        if dispersion_residual(mid) >= 0.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def solve_auxiliary_step(solver, k_numeric: float) -> float:
    omega = float(solver.source_omega)
    if omega <= 0.0 or k_numeric <= 1e-12:
        return float(solver.c) * float(solver.dt)

    rhs = math.sin(0.5 * omega * float(solver.dt)) / (float(solver.c) * float(solver.dt))
    low = float(solver.c) * float(solver.dt) * (1.0 + 1e-6)
    high = 0.999 * math.pi / k_numeric
    if high <= low:
        high = low * 1.5

    def residual(ds):
        return math.sin(0.5 * k_numeric * ds) / ds - rhs

    low_residual = residual(low)
    high_residual = residual(high)
    if low_residual < 0.0:
        return low
    if high_residual > 0.0:
        return high

    for _ in range(80):
        mid = 0.5 * (low + high)
        if residual(mid) >= 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def entry_projection(bounds, direction):
    x_bounds, y_bounds, z_bounds = bounds
    min_projection = None
    for x in x_bounds:
        for y in y_bounds:
            for z in z_bounds:
                projection = x * float(direction[0]) + y * float(direction[1]) + z * float(direction[2])
                if min_projection is None or projection < min_projection:
                    min_projection = projection
    return float(min_projection)


def projection_extrema(bounds, direction):
    x_bounds, y_bounds, z_bounds = bounds
    min_projection = None
    max_projection = None
    for x in x_bounds:
        for y in y_bounds:
            for z in z_bounds:
                projection = x * float(direction[0]) + y * float(direction[1]) + z * float(direction[2])
                if min_projection is None or projection < min_projection:
                    min_projection = projection
                if max_projection is None or projection > max_projection:
                    max_projection = projection
    return float(min_projection), float(max_projection)


def box_center_tensor(solver, bounds):
    return torch.tensor(
        [
            0.5 * (bounds[0][0] + bounds[0][1]),
            0.5 * (bounds[1][0] + bounds[1][1]),
            0.5 * (bounds[2][0] + bounds[2][1]),
        ],
        device=solver.device,
        dtype=solver.Ex.dtype,
    )


def make_aux_term(solver, *, field_name, offsets, coeff_patch, sample_positions, component_scale):
    return {
        "field_name": field_name,
        "offsets": offsets,
        "grid": solver._compute_linear_launch_shape(int(coeff_patch.numel())),
        "coeff_patch": coeff_patch.contiguous(),
        "sample_positions": sample_positions.contiguous(),
        "component_scale": float(component_scale),
    }


def _field_component_code(field_name: str) -> int:
    component = field_name[-1].lower()
    if component == "x":
        return 0
    if component == "y":
        return 1
    return 2


def _pack_batched_term_metadata(solver, terms):
    coeff_cursor = 0
    term_starts = []
    term_shapes = []
    term_offsets = []
    field_codes = []
    coeff_chunks = []
    field_code_chunks = []
    field_offset_chunks = []
    field_offset_dtype = torch.int32
    for term in terms:
        field_shape = tuple(int(length) for length in getattr(solver, term["field_name"]).shape)
        if math.prod(field_shape) > torch.iinfo(torch.int32).max:
            field_offset_dtype = torch.int64
            break

    for term_index, term in enumerate(terms):
        coeff_flat = (term["coeff_patch"] * float(term["component_scale"])).reshape(-1)
        shape = tuple(int(length) for length in term["coeff_patch"].shape)
        offsets = tuple(int(offset) for offset in term["offsets"])
        local_linear = torch.arange(int(coeff_flat.numel()), device=solver.device, dtype=torch.int32)
        stride_i = shape[1] * shape[2]
        local_i = local_linear // stride_i
        remainder = local_linear - local_i * stride_i
        local_j = remainder // shape[2]
        local_k = remainder - local_j * shape[2]
        field_code = _field_component_code(term["field_name"])
        field_shape = tuple(int(length) for length in getattr(solver, term["field_name"]).shape)
        field_i = local_i + offsets[0]
        field_j = local_j + offsets[1]
        field_k = local_k + offsets[2]
        coeff_chunks.append(coeff_flat)
        field_code_chunks.append(
            torch.full(
                (int(coeff_flat.numel()),),
                field_code,
                device=solver.device,
                dtype=torch.int32,
            )
        )
        field_offset_chunks.append(
            (
                field_i.to(dtype=torch.int64) * (field_shape[1] * field_shape[2])
                + field_j.to(dtype=torch.int64) * field_shape[2]
                + field_k.to(dtype=torch.int64)
            ).to(dtype=field_offset_dtype).contiguous()
        )
        term_starts.append(coeff_cursor)
        term_shapes.append(list(shape))
        term_offsets.append(list(offsets))
        field_codes.append(field_code)
        coeff_cursor += int(coeff_flat.numel())

    if not coeff_chunks:
        return None

    coeff_data = torch.cat(coeff_chunks).contiguous()
    return {
        "coeff_data": coeff_data,
        "term_starts": torch.tensor(term_starts, device=solver.device, dtype=torch.int32),
        "term_shapes": torch.tensor(term_shapes, device=solver.device, dtype=torch.int32),
        "term_offsets": torch.tensor(term_offsets, device=solver.device, dtype=torch.int32),
        "field_codes": torch.tensor(field_codes, device=solver.device, dtype=torch.int32),
        "field_codes_per_coeff": torch.cat(field_code_chunks).contiguous(),
        "field_offsets": torch.cat(field_offset_chunks).contiguous(),
        "grid": solver._compute_linear_launch_shape(int(coeff_data.numel())),
    }


def build_batched_reference_terms(solver, terms):
    metadata = _pack_batched_term_metadata(solver, terms)
    if metadata is None:
        return None

    sample_cursor = 0
    sample_axis_codes = []
    sample_index_starts = []
    sample_chunks = []
    sample_index_per_coeff_chunks = []
    for term in terms:
        sample_indices = term["sample_indices"].to(device=solver.device, dtype=torch.int32).reshape(-1)
        shape = tuple(int(length) for length in term["coeff_patch"].shape)
        local_linear = torch.arange(int(term["coeff_patch"].numel()), device=solver.device, dtype=torch.int32)
        stride_i = shape[1] * shape[2]
        local_i = local_linear // stride_i
        remainder = local_linear - local_i * stride_i
        local_j = remainder // shape[2]
        local_k = remainder - local_j * shape[2]
        axis = int(reference_sample_axis_code(term))
        sample_linear = local_i if axis == 0 else (local_j if axis == 1 else local_k)
        sample_chunks.append(sample_indices)
        sample_index_per_coeff_chunks.append(sample_indices[sample_linear.to(dtype=torch.long)])
        sample_axis_codes.append(axis)
        sample_index_starts.append(sample_cursor)
        sample_cursor += int(sample_indices.numel())

    metadata["sample_axis_codes"] = torch.tensor(sample_axis_codes, device=solver.device, dtype=torch.int32)
    metadata["sample_index_starts"] = torch.tensor(sample_index_starts, device=solver.device, dtype=torch.int32)
    metadata["sample_indices"] = torch.cat(sample_chunks).contiguous()
    metadata["sample_indices_per_coeff"] = torch.cat(sample_index_per_coeff_chunks).contiguous()
    return metadata


def build_batched_aux_terms(solver, terms):
    metadata = _pack_batched_term_metadata(solver, terms)
    if metadata is None:
        return None

    metadata["sample_positions"] = torch.cat(
        [term["sample_positions"].reshape(-1) for term in terms]
    ).contiguous()
    return metadata


def slice_coeff_patch(solver, spec, curl_attr_map):
    offsets = spec["offsets"]
    shape = spec["shape"]
    return getattr(solver, curl_attr_map[spec["field_name"]])[
        offsets[0] : offsets[0] + shape[0],
        offsets[1] : offsets[1] + shape[1],
        offsets[2] : offsets[2] + shape[2],
    ] / float(getattr(solver, spec["delta_attr"]))


def build_terms_from_specs(solver, specs, vector, curl_attr_map, term_factory):
    terms = []
    for spec in specs:
        component_scale = vector[spec["incident_name"]]
        coeff = float(spec["sign"]) * component_scale
        if isinstance(coeff, torch.Tensor):
            if torch.max(torch.abs(coeff)).item() <= 1e-12:
                continue
        elif abs(float(coeff)) <= 1e-12:
            continue
        coeff_patch = slice_coeff_patch(solver, spec, curl_attr_map)
        term = term_factory(spec, coeff_patch, component_scale)
        if term is not None:
            terms.append(term)
    return terms


def build_term_from_profile(
    solver,
    *,
    field_name,
    offsets,
    scale,
    delay_patch,
    activation_delay_patch,
    source_time,
    omega,
    source_index=None,
):
    if torch.max(torch.abs(scale)).item() <= 1e-12:
        return None
    if source_time["kind"] == "cw" and activation_delay_patch is None:
        phase_shift = float(omega) * delay_patch
        return build_source_term(
            solver,
            field_name=field_name,
            offsets=offsets,
            cw_cos_patch=scale * torch.cos(phase_shift),
            cw_sin_patch=scale * torch.sin(phase_shift),
            source_index=source_index,
            source_time=source_time,
            omega=omega,
        )
    return build_source_term(
        solver,
        field_name=field_name,
        offsets=offsets,
        patch=scale,
        delay_patch=delay_patch,
        activation_delay_patch=activation_delay_patch,
        source_index=source_index,
        source_time=source_time,
        omega=omega,
    )


def incident_profile(
    solver,
    source,
    positions,
    reference_point,
    *,
    phase_speed,
    source_frequency=None,
    polarization_override=None,
):
    if source["kind"] == "plane_wave":
        return plane_wave_profile(
            positions,
            direction=source["direction"],
            reference_point=reference_point,
            propagation_speed=phase_speed,
        )
    return gaussian_beam_profile(
        positions,
        direction=source["direction"],
        polarization=source["polarization"] if polarization_override is None else polarization_override,
        beam_waist=source["beam_waist"],
        focus=source["focus"],
        frequency=solver.source_frequency if source_frequency is None else float(source_frequency),
        propagation_speed=phase_speed,
    )


def project_positions(positions, direction):
    direction_tensor = positions.new_tensor(direction)
    return torch.sum(positions * direction_tensor, dim=-1)
