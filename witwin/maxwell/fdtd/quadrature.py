"""Plane cell-quadrature and Poynting-flux integration for FDTD monitors.

Holds the single source of truth for cell-center quadrature weights, exact
primal-cell widths, and the time-averaged plane-normal Poynting reduction used
by flux monitors, incident-power density, adjoint flux objectives, and the
distributed monitor merge.
"""

import numpy as np
import torch

from ..monitors import normalize_axis

_AXIS_CODES = {"x": 0, "y": 1, "z": 2}

_PLANE_COORD_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}


def _plane_coord_names(axis):
    return _PLANE_COORD_NAMES[normalize_axis(axis)]


def _contains_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_torch_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_torch_tensor(item) for item in value)
    return False


def _infer_tensor_device_and_dtype(*values):
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device, value.real.dtype
    return None, torch.float32


def _to_torch_scalar_or_tensor(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _cell_center_weights_1d(points):
    """Control-volume widths for samples located at Yee cell centers."""
    if isinstance(points, torch.Tensor):
        coords = points.to(dtype=points.real.dtype)
        count = int(coords.numel())
        if count <= 1:
            return torch.ones((count,), device=coords.device, dtype=coords.dtype)
        diffs = coords[1:] - coords[:-1]
        weights = torch.empty((count,), device=coords.device, dtype=coords.dtype)
        weights[0] = diffs[0]
        weights[-1] = diffs[-1]
        if count > 2:
            weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
        return weights

    coords = np.asarray(points, dtype=float)
    count = coords.size
    if count <= 1:
        return np.ones((count,), dtype=float)
    diffs = np.diff(coords)
    weights = np.empty((count,), dtype=float)
    weights[0] = diffs[0]
    weights[-1] = diffs[-1]
    if count > 2:
        weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    return weights


def _exact_cell_center_widths(solver, axis, points):
    """Resolve aligned Yee center samples to their exact primal cell widths."""

    axis_name = normalize_axis(axis)
    centers = np.asarray(
        {
            "x": solver.scene.x_half64,
            "y": solver.scene.y_half64,
            "z": solver.scene.z_half64,
        }[axis_name],
        dtype=np.float64,
    )
    widths = np.asarray(
        {
            "x": solver.scene.dx_primal64,
            "y": solver.scene.dy_primal64,
            "z": solver.scene.dz_primal64,
        }[axis_name],
        dtype=np.float64,
    )
    requested = np.asarray(points, dtype=np.float64).reshape(-1)
    indices = np.asarray(
        [int(np.argmin(np.abs(centers - value))) for value in requested],
        dtype=np.int64,
    )
    matched = centers[indices]
    tolerance = 1.0e-7 * np.maximum.reduce(
        (np.ones_like(requested), np.abs(requested), np.abs(matched))
    )
    if not np.all(np.abs(requested - matched) <= tolerance):
        raise ValueError("Aligned plane coordinates do not lie on Yee cell centers.")
    return widths[indices]


def plane_normal_poynting(result):
    """Per-cell time-averaged normal Poynting ``S.n`` and cell-area weights.

    ``S.n = 0.5 * Re((E x conj(H)) . n_hat)`` in W/m^2, oriented by the payload's
    ``normal_direction`` (``"+"`` -> ``+axis``, ``"-"`` -> ``-axis``). The returned
    ``poynting`` carries a leading frequency axis only when the payload is
    multi-frequency; ``weights`` is the ``(nu, nv)`` cell-area map. Returns torch
    tensors when the payload holds tensors, else NumPy arrays. This is the single
    source of truth shared by plane-flux integration and incident power density so
    both stay exactly consistent (``flux == sum(poynting * weights)``).
    """

    axis = normalize_axis(result["axis"])
    axis_index = _AXIS_CODES[axis]
    coord_names = _plane_coord_names(axis)
    frequencies = tuple(float(freq) for freq in result.get("frequencies", (result["frequency"],)))
    has_multi_frequency = len(frequencies) > 1

    if _contains_torch_tensor(result):
        device, coord_dtype = _infer_tensor_device_and_dtype(
            result.get(coord_names[0]),
            result.get(coord_names[1]),
            result.get("Ex"),
            result.get("Ey"),
            result.get("Ez"),
            result.get("Hx"),
            result.get("Hy"),
            result.get("Hz"),
        )
        coord_a = _to_torch_scalar_or_tensor(result[coord_names[0]], device=device, dtype=coord_dtype)
        coord_b = _to_torch_scalar_or_tensor(result[coord_names[1]], device=device, dtype=coord_dtype)
        exact_widths = result.get("cell_widths")
        if exact_widths is None:
            weights = _cell_center_weights_1d(coord_a)[:, None] * _cell_center_weights_1d(coord_b)[None, :]
        else:
            width_a = _to_torch_scalar_or_tensor(
                exact_widths[coord_names[0]], device=device, dtype=coord_dtype
            )
            width_b = _to_torch_scalar_or_tensor(
                exact_widths[coord_names[1]], device=device, dtype=coord_dtype
            )
            weights = width_a[:, None] * width_b[None, :]
        field_shape = ((len(frequencies),) if has_multi_frequency else ()) + (coord_a.numel(), coord_b.numel())
        complex_dtype = None
        for component_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            value = result.get(component_name)
            if isinstance(value, torch.Tensor):
                complex_dtype = value.dtype
                break
        if complex_dtype is None:
            complex_dtype = torch.complex64
        e_field = torch.zeros(field_shape + (3,), device=device, dtype=complex_dtype)
        h_field = torch.zeros(field_shape + (3,), device=device, dtype=complex_dtype)
        for component_name in ("Ex", "Ey", "Ez"):
            if component_name not in result:
                continue
            component_index = "xyz".index(component_name[1].lower())
            e_field[..., component_index] = _to_torch_scalar_or_tensor(
                result[component_name],
                device=device,
                dtype=coord_dtype,
            )
        for component_name in ("Hx", "Hy", "Hz"):
            if component_name not in result:
                continue
            component_index = "xyz".index(component_name[1].lower())
            h_field[..., component_index] = _to_torch_scalar_or_tensor(
                result[component_name],
                device=device,
                dtype=coord_dtype,
            )

        direction = 1.0 if result.get("normal_direction", "+") == "+" else -1.0
        poynting = 0.5 * torch.real(torch.cross(e_field, torch.conj(h_field), dim=-1)[..., axis_index]) * direction
        return poynting, weights

    coord_a = np.asarray(result[coord_names[0]], dtype=float)
    coord_b = np.asarray(result[coord_names[1]], dtype=float)
    exact_widths = result.get("cell_widths")
    if exact_widths is None:
        weights = _cell_center_weights_1d(coord_a)[:, None] * _cell_center_weights_1d(coord_b)[None, :]
    else:
        weights = (
            np.asarray(exact_widths[coord_names[0]], dtype=float)[:, None]
            * np.asarray(exact_widths[coord_names[1]], dtype=float)[None, :]
        )
    leading_shape = (len(frequencies),) if has_multi_frequency else ()
    field_shape = leading_shape + (coord_a.size, coord_b.size)

    e_field = np.zeros(field_shape + (3,), dtype=np.complex128)
    h_field = np.zeros(field_shape + (3,), dtype=np.complex128)
    for component_name in ("Ex", "Ey", "Ez"):
        if component_name not in result:
            continue
        component_index = "xyz".index(component_name[1].lower())
        values = np.asarray(result[component_name], dtype=np.complex128)
        e_field[..., component_index] = values
    for component_name in ("Hx", "Hy", "Hz"):
        if component_name not in result:
            continue
        component_index = "xyz".index(component_name[1].lower())
        values = np.asarray(result[component_name], dtype=np.complex128)
        h_field[..., component_index] = values

    direction = 1.0 if result.get("normal_direction", "+") == "+" else -1.0
    poynting = 0.5 * np.real(np.cross(e_field, np.conj(h_field), axis=-1)[..., axis_index]) * direction
    return poynting, weights


def _compute_plane_flux(result):
    frequencies = tuple(float(freq) for freq in result.get("frequencies", (result["frequency"],)))
    has_multi_frequency = len(frequencies) > 1
    poynting, weights = plane_normal_poynting(result)
    if isinstance(poynting, torch.Tensor):
        flux = torch.sum(poynting * weights, dim=(-2, -1))
        if not has_multi_frequency:
            return flux.reshape(())
        return flux
    flux = np.sum(poynting * weights, axis=(-2, -1))
    if not has_multi_frequency:
        return float(flux)
    return flux
