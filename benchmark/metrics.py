from __future__ import annotations

import numpy as np


def significant_field_mask(reference: np.ndarray, *, relative_floor: float = 0.1) -> np.ndarray:
    """Select the physically excited support of a reference complex field."""
    values = np.abs(np.asarray(reference, dtype=np.complex128))
    peak = float(np.max(values)) if values.size else 0.0
    if peak == 0.0:
        return np.ones(values.shape, dtype=bool)
    return values >= float(relative_floor) * peak


def phase_align_field(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, complex]:
    """Remove one global phasor-reference offset without changing amplitude."""
    actual_array = np.asarray(actual, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    if actual_array.shape != reference_array.shape:
        raise ValueError("Global phase alignment requires equal field shapes.")
    selected = np.ones(actual_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    overlap = np.vdot(actual_array[selected], reference_array[selected])
    phase = 1.0 + 0.0j if abs(overlap) == 0.0 else overlap / abs(overlap)
    return actual_array * phase, complex(phase)


def best_fit_field_scale(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, complex]:
    """Fit one global complex scale so the remaining error measures field shape."""
    actual_array = np.asarray(actual, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    if actual_array.shape != reference_array.shape:
        raise ValueError("Global field scaling requires equal field shapes.")
    selected = np.ones(actual_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    denominator = np.vdot(actual_array[selected], actual_array[selected])
    scale = (
        1.0 + 0.0j
        if abs(denominator) == 0.0
        else np.vdot(actual_array[selected], reference_array[selected]) / denominator
    )
    return actual_array * scale, complex(scale)


def field_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    diff_norm = np.linalg.norm(a.ravel() - b.ravel())
    ref_norm = np.linalg.norm(b.ravel())
    if ref_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return float(diff_norm / ref_norm)


def field_max_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    diff_max = np.max(np.abs(a - b))
    ref_max = np.max(np.abs(b))
    if ref_max == 0.0:
        return 0.0 if diff_max == 0.0 else float("inf")
    return float(diff_max / ref_max)


def flux_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.abs(b) > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))


def flux_incident_normalized_error(
    a: np.ndarray,
    b: np.ndarray,
    *,
    incident_power: float,
) -> float:
    """Return the largest absolute flux difference on one incident-power scale."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    common = min(a.size, b.size)
    if common == 0:
        raise ValueError("Flux arrays must not be empty.")
    scale = abs(float(incident_power))
    if scale == 0.0:
        difference = float(np.max(np.abs(a[:common] - b[:common])))
        return 0.0 if difference == 0.0 else float("inf")
    return float(np.max(np.abs(a[:common] - b[:common])) / scale)


def field_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == 0.0 and norm_b == 0.0 else 0.0
    return float(np.abs(np.vdot(a, b)) / (norm_a * norm_b))


def align_arrays(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    if a.shape == b.shape:
        return a, b

    ndim = min(a.ndim, b.ndim)
    slices_a = []
    slices_b = []
    for dim in range(ndim):
        size_a = a.shape[dim]
        size_b = b.shape[dim]
        common = min(size_a, size_b)
        offset_a = (size_a - common) // 2
        offset_b = (size_b - common) // 2
        slices_a.append(slice(offset_a, offset_a + common))
        slices_b.append(slice(offset_b, offset_b + common))
    return a[tuple(slices_a)], b[tuple(slices_b)]


def plane_coord_keys(monitor_data: dict) -> tuple[str, str]:
    keys = tuple(key for key in ("x", "y", "z") if key in monitor_data)
    if len(keys) != 2:
        raise ValueError("Plane monitor data must expose exactly two spatial coordinate arrays.")
    return keys


def _normalize_axis_coords(coords) -> np.ndarray:
    axis = np.asarray(coords, dtype=np.float64).reshape(-1)
    if axis.size == 0:
        raise ValueError("Coordinate arrays must not be empty.")
    deltas = np.diff(axis)
    if np.any(deltas <= 0.0):
        raise ValueError("Coordinate arrays must be strictly increasing.")
    return axis


def _target_axis_coords(source_coords, reference_coords) -> np.ndarray:
    source = _normalize_axis_coords(source_coords)
    reference = _normalize_axis_coords(reference_coords)
    tolerance = 1e-9 * max(
        1.0,
        float(np.max(np.abs(source))),
        float(np.max(np.abs(reference))),
    )
    lo = max(float(source[0]), float(reference[0]))
    hi = min(float(source[-1]), float(reference[-1]))
    target = reference[(reference >= lo - tolerance) & (reference <= hi + tolerance)]
    if target.size == 0:
        raise ValueError(
            "Plane monitor coordinates do not overlap: "
            f"source=({source[0]}, {source[-1]}), reference=({reference[0]}, {reference[-1]})"
        )
    return target


def _interpolate_axis(data: np.ndarray, source_coords, target_coords, axis: int) -> np.ndarray:
    array = np.asarray(data)
    source = _normalize_axis_coords(source_coords)
    target = _normalize_axis_coords(target_coords)
    axis_index = axis if axis >= 0 else array.ndim + axis
    if axis_index < 0 or axis_index >= array.ndim:
        raise ValueError(f"axis={axis} is out of range for array shape {array.shape}.")
    if array.shape[axis_index] != source.size:
        raise ValueError(
            f"Axis length mismatch for interpolation: shape={array.shape}, "
            f"axis={axis_index}, coords={source.size}"
        )
    if source.size == target.size and np.allclose(source, target, rtol=1e-9, atol=1e-12):
        return array

    moved = np.moveaxis(np.asarray(array, dtype=np.complex128), axis_index, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty((flat.shape[0], target.size), dtype=np.complex128)
    for idx, row in enumerate(flat):
        out[idx] = np.interp(target, source, row.real) + 1j * np.interp(target, source, row.imag)
    reshaped = out.reshape(moved.shape[:-1] + (target.size,))
    return np.moveaxis(reshaped, -1, axis_index)


def align_plane_fields(
    source_field: np.ndarray,
    reference_field: np.ndarray,
    *,
    source_coords: tuple[np.ndarray, np.ndarray],
    reference_coords: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    source_array = np.squeeze(np.asarray(source_field))
    reference_array = np.squeeze(np.asarray(reference_field))
    if source_array.ndim < 2 or reference_array.ndim < 2:
        raise ValueError("Plane field alignment requires arrays with at least two dimensions.")

    target_x = _target_axis_coords(source_coords[0], reference_coords[0])
    target_y = _target_axis_coords(source_coords[1], reference_coords[1])

    aligned_source = _interpolate_axis(source_array, source_coords[0], target_x, axis=-2)
    aligned_source = _interpolate_axis(aligned_source, source_coords[1], target_y, axis=-1)
    aligned_reference = _interpolate_axis(reference_array, reference_coords[0], target_x, axis=-2)
    aligned_reference = _interpolate_axis(aligned_reference, reference_coords[1], target_y, axis=-1)
    return aligned_source, aligned_reference, (target_x, target_y)


__all__ = [
    "align_arrays",
    "align_plane_fields",
    "best_fit_field_scale",
    "field_correlation",
    "field_l2_error",
    "field_max_error",
    "flux_incident_normalized_error",
    "flux_relative_error",
    "phase_align_field",
    "plane_coord_keys",
    "significant_field_mask",
]
