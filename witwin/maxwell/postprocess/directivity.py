from __future__ import annotations

import math

import torch

from ..constants import resolve_real_dtype
from .stratton_chu import _resolve_tensor_device, _to_real_tensor, _trapz_weights_1d


def _extract_angular_grid(far_field) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _resolve_tensor_device(far_field.get("theta"), far_field.get("phi"))
    dtype = resolve_real_dtype(far_field.get("theta"), far_field.get("phi"))
    theta = _to_real_tensor(far_field["theta"], device=device, dtype=dtype)
    phi = _to_real_tensor(far_field["phi"], device=device, dtype=dtype)
    theta, phi = torch.broadcast_tensors(theta, phi)
    if theta.ndim != 2:
        raise ValueError("far_field theta/phi must define a 2D angular grid.")

    theta_vector = theta[:, 0]
    phi_vector = phi[0, :]
    if not torch.allclose(theta, theta_vector[:, None]):
        raise ValueError("theta must be constant along the phi axis.")
    if not torch.allclose(phi, phi_vector[None, :]):
        raise ValueError("phi must be constant along the theta axis.")
    return theta, phi, theta_vector, phi_vector


def _half_power_beam_width(theta_vector: torch.Tensor, pattern: torch.Tensor) -> float | None:
    if pattern.ndim != 1 or theta_vector.ndim != 1 or tuple(pattern.shape) != tuple(theta_vector.shape):
        raise ValueError("theta_vector and pattern must be matching 1D arrays.")

    peak_index = int(torch.argmax(pattern).item())
    peak_value = float(pattern[peak_index].item())
    if peak_value <= 0.0 or peak_index == 0 or peak_index == pattern.numel() - 1:
        return None

    half_power = peak_value / 2.0

    left_index = None
    for index in range(peak_index - 1, -1, -1):
        if float(pattern[index].item()) < half_power <= float(pattern[index + 1].item()):
            left_index = index
            break
    if left_index is None:
        return None

    right_index = None
    for index in range(peak_index, pattern.numel() - 1):
        if float(pattern[index].item()) >= half_power > float(pattern[index + 1].item()):
            right_index = index
            break
    if right_index is None:
        return None

    def interpolate(theta_a, theta_b, value_a, value_b):
        if math.isclose(value_a, value_b, rel_tol=1e-9, abs_tol=1e-12):
            return theta_a
        weight = (half_power - value_a) / (value_b - value_a)
        return theta_a + weight * (theta_b - theta_a)

    theta_left = interpolate(
        float(theta_vector[left_index].item()),
        float(theta_vector[left_index + 1].item()),
        float(pattern[left_index].item()),
        float(pattern[left_index + 1].item()),
    )
    theta_right = interpolate(
        float(theta_vector[right_index].item()),
        float(theta_vector[right_index + 1].item()),
        float(pattern[right_index].item()),
        float(pattern[right_index + 1].item()),
    )
    return math.degrees(theta_right - theta_left)


def _nearest_phi_index(phi_vector: torch.Tensor, target: float) -> int:
    wrapped = torch.remainder(phi_vector, 2.0 * math.pi)
    delta = torch.abs(torch.remainder(wrapped - target + math.pi, 2.0 * math.pi) - math.pi)
    return int(torch.argmin(delta).item())


def compute_directivity(
    far_field: dict,
    *,
    input_power=None,
) -> dict:
    theta_grid, phi_grid, theta_vector, phi_vector = _extract_angular_grid(far_field)
    device = theta_grid.device
    dtype = theta_grid.dtype
    radius = torch.broadcast_to(
        _to_real_tensor(far_field["radius"], device=device, dtype=dtype),
        theta_grid.shape,
    )
    power_density = _to_real_tensor(far_field["power_density"], device=device, dtype=dtype)
    if tuple(power_density.shape) != tuple(theta_grid.shape):
        raise ValueError("power_density must match the theta/phi grid shape.")

    radiation_intensity = torch.clamp_min(radius.square() * power_density, 0.0)
    weights = _trapz_weights_1d(theta_vector)[:, None] * _trapz_weights_1d(phi_vector)[None, :]
    p_rad = torch.sum(radiation_intensity * torch.sin(theta_grid) * weights)
    if float(p_rad.item()) <= 0.0:
        raise ValueError("Total radiated power must be positive.")

    directivity = 4.0 * math.pi * radiation_intensity / p_rad
    directivity_db = 10.0 * torch.log10(torch.clamp_min(directivity, 1e-30))
    flat_index = int(torch.argmax(directivity.reshape(-1)).item())
    phi_count = directivity.shape[1]
    peak_coords = (flat_index // phi_count, flat_index % phi_count)
    d_max = directivity[peak_coords]
    d_max_theta = float(theta_grid[peak_coords].item())
    d_max_phi = float(phi_grid[peak_coords].item())

    gain = None
    gain_db = None
    g_max = None
    g_max_db = None
    radiation_efficiency = None
    if input_power is not None:
        input_power_tensor = _to_real_tensor(input_power, device=device, dtype=dtype)
        if float(input_power_tensor.item()) <= 0.0:
            raise ValueError("input_power must be positive.")
        radiation_efficiency = p_rad / input_power_tensor
        gain = radiation_efficiency * directivity
        gain_db = 10.0 * torch.log10(torch.clamp_min(gain, 1e-30))
        g_max = gain.reshape(-1)[flat_index]
        g_max_db = 10.0 * torch.log10(torch.clamp_min(g_max, 1e-30))

    e_plane_index = _nearest_phi_index(phi_vector, 0.0)
    h_plane_index = _nearest_phi_index(phi_vector, 0.5 * math.pi)
    beam_width_e_plane = _half_power_beam_width(theta_vector, directivity[:, e_plane_index])
    beam_width_h_plane = _half_power_beam_width(theta_vector, directivity[:, h_plane_index])

    return {
        "theta": theta_grid,
        "phi": phi_grid,
        "frequency": far_field["frequency"],
        "P_rad": p_rad,
        "U": radiation_intensity,
        "directivity": directivity,
        "directivity_db": directivity_db,
        "D_max": d_max,
        "D_max_db": 10.0 * torch.log10(torch.clamp_min(d_max, 1e-30)),
        "D_max_theta": d_max_theta,
        "D_max_phi": d_max_phi,
        "gain": gain,
        "gain_db": gain_db,
        "G_max": g_max,
        "G_max_db": g_max_db,
        "radiation_efficiency": radiation_efficiency,
        "beam_width_e_plane": beam_width_e_plane,
        "beam_width_h_plane": beam_width_h_plane,
    }
