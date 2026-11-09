"""Gaussian-beam overlap decomposition of a recorded plane monitor.

The reference profile is the paraxial (astigmatic) Gaussian beam used by the
``GaussianBeam`` / ``AstigmaticGaussianBeam`` sources, evaluated on the
monitor's aligned sample points as a running-DFT phasor (``exp(+i w t)``
accumulation, so a delay ``tau`` maps to ``exp(+i w tau)``). The forward and
backward amplitudes use the same reciprocity decomposition as
:class:`~witwin.maxwell.ModeMonitor`: the backward reference is the
tangentially mirrored profile ``(E_t, -H_t)``.
"""

from __future__ import annotations

import math

import torch

from ..constants import C_0, EPSILON_0, MU_0, resolve_complex_dtype, resolve_real_dtype
from ..fdtd.excitation.spatial import _beam_profile_core
from .modal import (
    _FIELD_NAMES,
    _integrated_power,
    _monitor_coords,
    _monitor_vector_fields,
    _surface_normal,
)
from .stratton_chu import _as_1d_coords, _resolve_tensor_device, _trapz_weights_1d, build_plane_points

_ETA_0 = math.sqrt(MU_0 / EPSILON_0)
# Pre-scale-guard absolute power floor, restored verbatim (scale-relative guard is post-release backlog).
_POWER_FLOOR = 1e-30


def _plane_weights(payload, tangential_axes, coords_a, coords_b, *, device, dtype):
    cell_widths = payload.get("cell_widths")
    if cell_widths is None:
        return _trapz_weights_1d(coords_a)[:, None] * _trapz_weights_1d(coords_b)[None, :]
    widths = []
    for axis_label, coordinates in zip(tangential_axes, (coords_a, coords_b)):
        width = torch.as_tensor(cell_widths[axis_label], device=device, dtype=dtype)
        if width.shape != coordinates.shape or not bool(torch.all(torch.isfinite(width) & (width > 0))):
            raise ValueError("Gaussian-overlap monitor cell widths must be finite, positive and match its coordinates.")
        widths.append(width)
    return widths[0][:, None] * widths[1][None, :]


def _background_medium(result, monitor, frequency: float) -> tuple[float, float]:
    """Relative (eps_r, mu_r) of the compiled medium at the monitor center."""

    prepared = getattr(result, "prepared_scene", None)
    if prepared is None:
        return 1.0, 1.0
    eps_r, mu_r = prepared.compile_relative_materials(frequency=frequency)
    index = []
    for axis_index, axis in enumerate("xyz"):
        coords = torch.as_tensor(getattr(prepared, axis), device=eps_r.device)
        index.append(int(torch.argmin(torch.abs(coords - float(monitor.position[axis_index]))).item()))
    eps_value = complex(eps_r[tuple(index)].item())
    mu_value = complex(mu_r[tuple(index)].item())
    return eps_value.real, mu_value.real


def _overlap_one_frequency(result, monitor, payload, frequency: float):
    axis, tangential_axes, coord_a, coord_b = _monitor_coords(payload)
    device = _resolve_tensor_device(coord_a, coord_b, *[payload.get(name) for name in _FIELD_NAMES])
    real_dtype = resolve_real_dtype(coord_a, coord_b)
    complex_dtype = resolve_complex_dtype(*[payload.get(name) for name in _FIELD_NAMES])
    coords_a = _as_1d_coords(coord_a, tangential_axes[0], device=device, dtype=real_dtype)
    coords_b = _as_1d_coords(coord_b, tangential_axes[1], device=device, dtype=real_dtype)
    weights = _plane_weights(payload, tangential_axes, coords_a, coords_b, device=device, dtype=real_dtype)
    normal = _surface_normal(axis, monitor.normal_direction, device=device, dtype=real_dtype).to(dtype=complex_dtype)
    electric, magnetic = _monitor_vector_fields(
        payload, (coords_a.numel(), coords_b.numel()), device=device, dtype=complex_dtype
    )

    eps_r, mu_r = _background_medium(result, monitor, frequency)
    if eps_r <= 0.0 or mu_r <= 0.0:
        raise ValueError("GaussianOverlapMonitor requires a propagating (positive eps_r, mu_r) medium at its center.")
    index = math.sqrt(eps_r * mu_r)
    impedance = _ETA_0 * math.sqrt(mu_r / eps_r)
    points = build_plane_points(axis, float(payload["position"]), coords_a, coords_b).to(device=device, dtype=torch.float64)
    amplitude, delay = _beam_profile_core(
        points,
        direction=monitor.direction,
        polarization=monitor.polarization,
        beam_waist_u=monitor.beam_waists[0],
        beam_waist_v=monitor.beam_waists[1],
        focus=monitor.focus,
        focus_offset_u=monitor.focus_offsets[0],
        focus_offset_v=monitor.focus_offsets[1],
        frequency=frequency,
        propagation_speed=C_0 / index,
    )
    phasor = (amplitude * torch.exp(1j * (2.0 * math.pi * frequency) * delay)).to(complex_dtype)
    polarization = torch.as_tensor(monitor.polarization, device=device, dtype=torch.float64)
    binormal = torch.cross(torch.as_tensor(monitor.direction, device=device, dtype=torch.float64), polarization, dim=0)
    electric_ref = phasor[..., None] * polarization.to(complex_dtype)
    magnetic_ref = phasor[..., None] * (binormal / impedance).to(complex_dtype)

    reference_power = _integrated_power(electric_ref, magnetic_ref, normal, weights)
    reference_power_abs = torch.abs(torch.real(reference_power))
    if float(reference_power_abs.item()) <= _POWER_FLOOR:
        raise ValueError("Gaussian reference power is numerically zero on the monitor plane.")
    electric_overlap = torch.sum(
        torch.sum(torch.cross(electric, torch.conj(magnetic_ref), dim=-1) * normal, dim=-1) * weights
    ) / (2.0 * reference_power)
    magnetic_overlap = torch.sum(
        torch.sum(torch.cross(torch.conj(electric_ref), magnetic, dim=-1) * normal, dim=-1) * weights
    ) / (2.0 * torch.conj(reference_power))
    scale = torch.sqrt(reference_power_abs).to(dtype=electric_overlap.dtype)
    forward = 0.5 * (electric_overlap + magnetic_overlap) * scale
    backward = 0.5 * (electric_overlap - magnetic_overlap) * scale
    total_power = torch.real(_integrated_power(electric, magnetic, normal, weights))
    return forward, backward, total_power, reference_power_abs, index


def compute_gaussian_overlap(result, monitor, *, frequency: float | None = None, freq_index: int | None = None):
    """Forward/backward Gaussian-beam amplitudes of a Gaussian-overlap monitor.

    Amplitudes are in ``sqrt(W)`` against the unit-power reference profile, so
    ``|amplitude|^2`` is the power carried by the matching beam; the power
    fractions divide by ``|net plane flux|``. Every tensor keeps an explicit
    leading frequency axis and stays differentiable with respect to the
    recorded plane fields.
    """
    from ..result import _resolve_frequency_index

    available = tuple(float(value) for value in (monitor.frequencies or result.frequencies))
    selected_index = _resolve_frequency_index(available, frequency=frequency, freq_index=freq_index)
    selected = available if selected_index is None else (available[selected_index],)
    forward, backward, total, background_index = [], [], [], []
    for value in selected:
        payload = result.raw_monitor(monitor.name, frequency=value)
        amplitude_forward, amplitude_backward, total_power, _, index = _overlap_one_frequency(
            result, monitor, payload, value
        )
        forward.append(amplitude_forward)
        backward.append(amplitude_backward)
        total.append(total_power)
        background_index.append(index)
    amplitude_forward = torch.stack(forward)
    amplitude_backward = torch.stack(backward)
    total_power = torch.stack(total)
    power_forward = torch.abs(amplitude_forward).square()
    power_backward = torch.abs(amplitude_backward).square()
    denominator = torch.clamp(torch.abs(total_power), min=_POWER_FLOOR)
    return {
        "kind": monitor.kind,
        "monitor": monitor.name,
        "frequencies": selected,
        "axis": monitor.axis,
        "position": float(monitor.plane_position),
        "normal_direction": monitor.normal_direction,
        "direction": monitor.direction,
        "polarization": monitor.polarization,
        "beam_waist": monitor.beam_waists,
        "focus": monitor.focus,
        "focus_offsets": monitor.focus_offsets,
        "background_index": tuple(background_index),
        "amplitude_forward": amplitude_forward,
        "amplitude_backward": amplitude_backward,
        "power_forward": power_forward,
        "power_backward": power_backward,
        "total_power": total_power,
        "power_fraction_forward": power_forward / denominator,
        "power_fraction_backward": power_backward / denominator,
    }


__all__ = ["compute_gaussian_overlap"]
