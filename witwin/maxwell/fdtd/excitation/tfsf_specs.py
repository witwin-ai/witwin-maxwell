from __future__ import annotations

import math

import torch


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
E_CURL_ATTR = {"Ex": "cex_curl", "Ey": "cey_curl", "Ez": "cez_curl"}
H_CURL_ATTR = {"Hx": "chx_curl", "Hy": "chy_curl", "Hz": "chz_curl"}
_FACE_SPEC_RANGES = {
    ("x", "low"): slice(0, 2),
    ("x", "high"): slice(2, 4),
    ("y", "low"): slice(4, 6),
    ("y", "high"): slice(6, 8),
    ("z", "low"): slice(8, 10),
    ("z", "high"): slice(10, 12),
}


def make_discrete_spec(
    *,
    field_name,
    incident_name,
    offsets,
    shape,
    sample_offsets,
    sample_shape,
    delta_axis,
    sign,
    sample_kind,
):
    return {
        "field_name": field_name,
        "incident_name": incident_name,
        "offsets": offsets,
        "shape": shape,
        "sample_offsets": sample_offsets,
        "sample_shape": sample_shape,
        "delta_axis": delta_axis,
        "sign": float(sign),
        "sample_kind": sample_kind,
    }


def build_discrete_tfsf_specs(lower, upper):
    ix0, iy0, iz0 = lower
    ix1, iy1, iz1 = upper

    magnetic_specs = [
        make_discrete_spec(
            field_name="Hy",
            incident_name="Ez",
            offsets=(ix0 - 1, iy0, iz0),
            shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            sample_offsets=(ix0, iy0, iz0),
            sample_shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            delta_axis="x",
            sign=+1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hz",
            incident_name="Ey",
            offsets=(ix0 - 1, iy0, iz0),
            shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            sample_offsets=(ix0, iy0, iz0),
            sample_shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            delta_axis="x",
            sign=-1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hy",
            incident_name="Ez",
            offsets=(ix1, iy0, iz0),
            shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            sample_offsets=(ix1, iy0, iz0),
            sample_shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            delta_axis="x",
            sign=-1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hz",
            incident_name="Ey",
            offsets=(ix1, iy0, iz0),
            shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            sample_offsets=(ix1, iy0, iz0),
            sample_shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            delta_axis="x",
            sign=+1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hx",
            incident_name="Ez",
            offsets=(ix0, iy0 - 1, iz0),
            shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            sample_offsets=(ix0, iy0, iz0),
            sample_shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            delta_axis="y",
            sign=-1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hz",
            incident_name="Ex",
            offsets=(ix0, iy0 - 1, iz0),
            shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            sample_offsets=(ix0, iy0, iz0),
            sample_shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            delta_axis="y",
            sign=+1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hx",
            incident_name="Ez",
            offsets=(ix0, iy1, iz0),
            shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            sample_offsets=(ix0, iy1, iz0),
            sample_shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            delta_axis="y",
            sign=+1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hz",
            incident_name="Ex",
            offsets=(ix0, iy1, iz0),
            shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            sample_offsets=(ix0, iy1, iz0),
            sample_shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            delta_axis="y",
            sign=-1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hx",
            incident_name="Ey",
            offsets=(ix0, iy0, iz0 - 1),
            shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            sample_offsets=(ix0, iy0, iz0),
            sample_shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            delta_axis="z",
            sign=+1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hy",
            incident_name="Ex",
            offsets=(ix0, iy0, iz0 - 1),
            shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            sample_offsets=(ix0, iy0, iz0),
            sample_shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            delta_axis="z",
            sign=-1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hx",
            incident_name="Ey",
            offsets=(ix0, iy0, iz1),
            shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            sample_offsets=(ix0, iy0, iz1),
            sample_shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            delta_axis="z",
            sign=-1.0,
            sample_kind="electric",
        ),
        make_discrete_spec(
            field_name="Hy",
            incident_name="Ex",
            offsets=(ix0, iy0, iz1),
            shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            sample_offsets=(ix0, iy0, iz1),
            sample_shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            delta_axis="z",
            sign=+1.0,
            sample_kind="electric",
        ),
    ]

    electric_specs = [
        make_discrete_spec(
            field_name="Ey",
            incident_name="Hz",
            offsets=(ix0, iy0, iz0),
            shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            sample_offsets=(ix0 - 1, iy0, iz0),
            sample_shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            delta_axis="x",
            sign=+1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ez",
            incident_name="Hy",
            offsets=(ix0, iy0, iz0),
            shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            sample_offsets=(ix0 - 1, iy0, iz0),
            sample_shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            delta_axis="x",
            sign=-1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ey",
            incident_name="Hz",
            offsets=(ix1, iy0, iz0),
            shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            sample_offsets=(ix1, iy0, iz0),
            sample_shape=(1, iy1 - iy0, iz1 - iz0 + 1),
            delta_axis="x",
            sign=-1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ez",
            incident_name="Hy",
            offsets=(ix1, iy0, iz0),
            shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            sample_offsets=(ix1, iy0, iz0),
            sample_shape=(1, iy1 - iy0 + 1, iz1 - iz0),
            delta_axis="x",
            sign=+1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ex",
            incident_name="Hz",
            offsets=(ix0, iy0, iz0),
            shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            sample_offsets=(ix0, iy0 - 1, iz0),
            sample_shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            delta_axis="y",
            sign=-1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ez",
            incident_name="Hx",
            offsets=(ix0, iy0, iz0),
            shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            sample_offsets=(ix0, iy0 - 1, iz0),
            sample_shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            delta_axis="y",
            sign=+1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ex",
            incident_name="Hz",
            offsets=(ix0, iy1, iz0),
            shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            sample_offsets=(ix0, iy1, iz0),
            sample_shape=(ix1 - ix0, 1, iz1 - iz0 + 1),
            delta_axis="y",
            sign=+1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ez",
            incident_name="Hx",
            offsets=(ix0, iy1, iz0),
            shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            sample_offsets=(ix0, iy1, iz0),
            sample_shape=(ix1 - ix0 + 1, 1, iz1 - iz0),
            delta_axis="y",
            sign=-1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ex",
            incident_name="Hy",
            offsets=(ix0, iy0, iz0),
            shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            sample_offsets=(ix0, iy0, iz0 - 1),
            sample_shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            delta_axis="z",
            sign=+1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ey",
            incident_name="Hx",
            offsets=(ix0, iy0, iz0),
            shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            sample_offsets=(ix0, iy0, iz0 - 1),
            sample_shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            delta_axis="z",
            sign=-1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ex",
            incident_name="Hy",
            offsets=(ix0, iy0, iz1),
            shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            sample_offsets=(ix0, iy0, iz1),
            sample_shape=(ix1 - ix0, iy1 - iy0 + 1, 1),
            delta_axis="z",
            sign=-1.0,
            sample_kind="magnetic",
        ),
        make_discrete_spec(
            field_name="Ey",
            incident_name="Hx",
            offsets=(ix0, iy0, iz1),
            shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            sample_offsets=(ix0, iy0, iz1),
            sample_shape=(ix1 - ix0 + 1, iy1 - iy0, 1),
            delta_axis="z",
            sign=+1.0,
            sample_kind="magnetic",
        ),
    ]
    return electric_specs, magnetic_specs


def build_slab_tfsf_specs(lower, upper, *, axis: str):
    axis_name = str(axis).lower()
    if axis_name not in AXIS_INDEX:
        raise ValueError("TFSF slab axis must be 'x', 'y', or 'z'.")
    electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)
    low_slice = _FACE_SPEC_RANGES[(axis_name, "low")]
    high_slice = _FACE_SPEC_RANGES[(axis_name, "high")]
    return (
        electric_specs[low_slice] + electric_specs[high_slice],
        magnetic_specs[low_slice] + magnetic_specs[high_slice],
    )


def magnetic_unit_vector(direction, polarization):
    mx = -(float(direction[1]) * float(polarization[2]) - float(direction[2]) * float(polarization[1]))
    my = -(float(direction[2]) * float(polarization[0]) - float(direction[0]) * float(polarization[2]))
    mz = -(float(direction[0]) * float(polarization[1]) - float(direction[1]) * float(polarization[0]))
    magnitude = math.sqrt(mx * mx + my * my + mz * mz)
    if magnitude <= 1e-12:
        raise ValueError("Plane-wave magnetic direction is ill-defined.")
    return {
        "Hx": mx / magnitude,
        "Hy": my / magnitude,
        "Hz": mz / magnitude,
    }


def magnetic_physical_vector(direction, polarization, impedance):
    unit = magnetic_unit_vector(direction, polarization)
    return {name: value / float(impedance) for name, value in unit.items()}


def normalize_vector_components(vector, *, name):
    magnitude = math.sqrt(sum(float(component) * float(component) for component in vector))
    if magnitude <= 1e-12:
        raise ValueError(f"{name} must have non-zero magnitude.")
    return tuple(float(component) / magnitude for component in vector)


def cross_vector(a, b):
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def discrete_plane_wave_vectors(direction, polarization, k_numeric: float, deltas):
    effective_wavevector = []
    for axis, component in zip("xyz", direction):
        delta = float(deltas[axis])
        effective_wavevector.append(math.sin(0.5 * k_numeric * float(component) * delta) / delta)
    kappa_hat = normalize_vector_components(effective_wavevector, name="effective_wavevector")

    projection = sum(float(p) * float(k) for p, k in zip(polarization, kappa_hat))
    electric = tuple(float(p) - projection * float(k) for p, k in zip(polarization, kappa_hat))
    electric = normalize_vector_components(electric, name="discrete_polarization")
    magnetic = normalize_vector_components(
        tuple(-float(component) for component in cross_vector(kappa_hat, electric)),
        name="discrete_magnetic",
    )
    return (
        {"Ex": electric[0], "Ey": electric[1], "Ez": electric[2]},
        {"Hx": magnetic[0], "Hy": magnetic[1], "Hz": magnetic[2]},
    )


def is_reference_plane_wave_x_ez(source) -> bool:
    direction = source["direction"]
    polarization = source["polarization"]
    return (
        abs(float(direction[0]) - 1.0) <= 1e-9
        and abs(float(direction[1])) <= 1e-9
        and abs(float(direction[2])) <= 1e-9
        and abs(float(polarization[0])) <= 1e-9
        and abs(float(polarization[1])) <= 1e-9
        and abs(float(polarization[2]) - 1.0) <= 1e-9
    )


def axis_aligned_direction(direction):
    dominant_axis = None
    dominant_sign = 0
    for axis, component in zip("xyz", direction):
        value = float(component)
        if abs(value) > 1e-9:
            if dominant_axis is not None:
                return None
            dominant_axis = axis
            dominant_sign = 1 if value > 0.0 else -1
    if dominant_axis is None:
        return None
    return dominant_axis, dominant_sign


def line_index_tensor(solver, start: int, length: int):
    return torch.arange(start, start + length, device=solver.device, dtype=torch.int64)


def constant_line_index_tensor(solver, index: int):
    return torch.tensor([index], device=solver.device, dtype=torch.int64)


def make_reference_term(
    solver,
    *,
    field_name,
    offsets,
    coeff_patch,
    sample_kind,
    sample_indices,
    component_scale,
):
    scalar_sample_index = None
    if int(sample_indices.numel()) == 1:
        scalar_sample_index = int(sample_indices.reshape(-1)[0].item())
    return {
        "field_name": field_name,
        "offsets": offsets,
        "coeff_patch": coeff_patch.contiguous(),
        "sample_kind": sample_kind,
        "sample_indices": sample_indices.contiguous(),
        "scalar_sample_index": scalar_sample_index,
        "sample_view": (int(sample_indices.numel()), 1, 1),
        "component_scale": float(component_scale),
    }


def axis_aligned_sample_view(axis: str, length: int):
    if axis == "x":
        return (int(length), 1, 1)
    if axis == "y":
        return (1, int(length), 1)
    return (1, 1, int(length))


def sample_axis_code(sample_view):
    if int(sample_view[0]) > 1:
        return 0
    if int(sample_view[1]) > 1:
        return 1
    return 2


def reference_sample_axis_code(term):
    if int(term["sample_indices"].numel()) > 1:
        return sample_axis_code(term["sample_view"])

    shape = tuple(int(length) for length in term["coeff_patch"].shape)
    if shape[0] == 1:
        return 0
    if shape[1] == 1:
        return 1
    return 2


def axis_aligned_sample_indices(solver, axis: str, direction_sign: int, sample_kind: str, offsets, shape):
    axis_index = AXIS_INDEX[axis]
    start = int(offsets[axis_index])
    length = int(shape[axis_index])
    indices = torch.arange(start, start + length, device=solver.device, dtype=torch.int64)
    if direction_sign > 0:
        return indices

    if sample_kind == "electric":
        max_index = {"x": solver.Nx - 1, "y": solver.Ny - 1, "z": solver.Nz - 1}[axis]
    else:
        max_index = {"x": solver.Nx - 2, "y": solver.Ny - 2, "z": solver.Nz - 2}[axis]
    return max_index - indices
