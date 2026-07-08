from __future__ import annotations

import math

import torch

from .modal import (
    _AXIS_TO_INDEX,
    _monitor_coords,
    _monitor_vector_fields,
    _surface_normal,
)
from .stratton_chu import (
    _as_1d_coords,
    _resolve_complex_dtype,
    _resolve_real_dtype,
    _resolve_tensor_device,
)

_C = 299792458.0
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_POWER_FLOOR = 1e-30


def enumerate_diffraction_orders(
    *,
    periods: tuple[float, float],
    k_bloch: tuple[float, float],
    k0: float,
    background_index: float = 1.0,
    max_order: int | None = None,
) -> tuple[list[dict[str, float]], tuple[float, float]]:
    """Enumerate grating diffraction orders for a periodic unit cell.

    ``periods`` are the transverse unit-cell lengths ``(La, Lb)`` and ``k_bloch``
    is the transverse Bloch wavevector ``(ka, kb)``. An order ``(m, n)`` has
    transverse wavevector ``k_bloch + (m Ga, n Gb)`` with reciprocal vectors
    ``Ga = 2*pi/La`` and ``Gb = 2*pi/Lb``; it propagates when its transverse
    wavenumber is below ``k0 * background_index``.
    """

    period_a, period_b = float(periods[0]), float(periods[1])
    reciprocal_a = 2.0 * math.pi / period_a
    reciprocal_b = 2.0 * math.pi / period_b
    k_cutoff = float(k0) * float(background_index)

    if max_order is None:
        bound_a = int(math.floor((k_cutoff + abs(k_bloch[0])) / reciprocal_a)) + 1
        bound_b = int(math.floor((k_cutoff + abs(k_bloch[1])) / reciprocal_b)) + 1
    else:
        bound_a = int(max_order)
        bound_b = int(max_order)

    orders: list[dict[str, float]] = []
    for m in range(-bound_a, bound_a + 1):
        for n in range(-bound_b, bound_b + 1):
            kx = float(k_bloch[0]) + m * reciprocal_a
            ky = float(k_bloch[1]) + n * reciprocal_b
            kz_squared = k_cutoff * k_cutoff - kx * kx - ky * ky
            orders.append(
                {
                    "m": m,
                    "n": n,
                    "kx": kx,
                    "ky": ky,
                    "kz_squared": kz_squared,
                    "propagating": kz_squared > 0.0,
                }
            )
    return orders, (reciprocal_a, reciprocal_b)


def _transverse_periods(result, tangential_axes: tuple[str, str]) -> dict[str, float]:
    domain_range = result.solver.scene.domain_range
    periods: dict[str, float] = {}
    for axis in tangential_axes:
        index = _AXIS_TO_INDEX[axis]
        periods[axis] = float(domain_range[2 * index + 1] - domain_range[2 * index])
    return periods


def _payload_is_diffraction(payload: dict[str, object]) -> bool:
    return payload.get("monitor_type") == "diffraction"


def compute_diffraction_from_payload(
    result,
    monitor_name: str,
    payload: dict[str, object],
    *,
    incident_power: float | None = None,
    background_index: float = 1.0,
) -> dict[str, object]:
    spec = payload.get("diffraction_spec") or {}
    max_order = spec.get("orders")
    normal_direction = str(payload.get("normal_direction", spec.get("normal_direction", "+")))
    axis, tangential_axes, coord_a, coord_b = _monitor_coords(payload)
    frequency = float(payload.get("frequency"))

    field_values = [payload.get(name) for name in _FIELD_NAMES]
    device = _resolve_tensor_device(coord_a, coord_b, *field_values)
    real_dtype = _resolve_real_dtype(coord_a, coord_b)
    complex_dtype = _resolve_complex_dtype(*field_values)
    coords_a = _as_1d_coords(coord_a, tangential_axes[0], device=device, dtype=real_dtype)
    coords_b = _as_1d_coords(coord_b, tangential_axes[1], device=device, dtype=real_dtype)
    sample_count_a = int(coords_a.numel())
    sample_count_b = int(coords_b.numel())

    electric, magnetic = _monitor_vector_fields(
        payload,
        (sample_count_a, sample_count_b),
        device=device,
        dtype=complex_dtype,
    )

    periods = _transverse_periods(result, tangential_axes)
    period_a = periods[tangential_axes[0]]
    period_b = periods[tangential_axes[1]]
    area = period_a * period_b

    bloch = getattr(result.solver, "resolved_bloch_wavevector", (0.0, 0.0, 0.0))
    k_bloch = (
        float(bloch[_AXIS_TO_INDEX[tangential_axes[0]]]),
        float(bloch[_AXIS_TO_INDEX[tangential_axes[1]]]),
    )
    k0 = 2.0 * math.pi * frequency / _C
    order_specs, reciprocal = enumerate_diffraction_orders(
        periods=(period_a, period_b),
        k_bloch=k_bloch,
        k0=k0,
        background_index=background_index,
        max_order=max_order,
    )

    normal = _surface_normal(axis, normal_direction, device=device, dtype=real_dtype).to(complex_dtype)
    grid_a, grid_b = torch.meshgrid(coords_a, coords_b, indexing="ij")
    projection = 1.0 / float(sample_count_a * sample_count_b)
    k_cutoff = k0 * float(background_index)

    order_records: list[dict[str, object]] = []
    total_power = 0.0
    for order in order_specs:
        phase = torch.exp(-1j * (order["kx"] * grid_a + order["ky"] * grid_b)).to(complex_dtype)
        amplitude_e = torch.stack([torch.sum(electric[..., c] * phase) for c in range(3)]) * projection
        amplitude_h = torch.stack([torch.sum(magnetic[..., c] * phase) for c in range(3)]) * projection
        poynting = 0.5 * torch.cross(amplitude_e, torch.conj(amplitude_h), dim=-1)
        power = float(torch.real(torch.sum(poynting * normal)).item()) * area
        total_power += power

        propagating = bool(order["propagating"])
        if propagating:
            kz = math.sqrt(max(order["kz_squared"], 0.0))
            transverse = math.hypot(order["kx"], order["ky"])
            theta = math.asin(min(transverse / k_cutoff, 1.0)) if k_cutoff > 0.0 else None
        else:
            kz = None
            theta = None
        order_records.append(
            {
                "m": int(order["m"]),
                "n": int(order["n"]),
                "kx": float(order["kx"]),
                "ky": float(order["ky"]),
                "kz": kz,
                "propagating": propagating,
                "power": power,
                "amplitude_E": amplitude_e,
                "amplitude_H": amplitude_h,
                "theta": theta,
                "phi": math.atan2(order["ky"], order["kx"]),
            }
        )

    propagating_power = sum(record["power"] for record in order_records if record["propagating"])
    denominator = incident_power if incident_power is not None else propagating_power
    for record in order_records:
        if denominator is not None and abs(denominator) > _POWER_FLOOR:
            record["efficiency"] = record["power"] / denominator
        else:
            record["efficiency"] = None

    return {
        "monitor": monitor_name,
        "kind": "diffraction",
        "frequency": frequency,
        "axis": axis,
        "normal_direction": normal_direction,
        "position": float(payload.get("position")),
        "tangential_axes": tangential_axes,
        "periods": periods,
        "reciprocal_vectors": {tangential_axes[0]: reciprocal[0], tangential_axes[1]: reciprocal[1]},
        "background_index": float(background_index),
        "incident_power": incident_power,
        "total_power": total_power,
        "propagating_power": propagating_power,
        "num_propagating": sum(1 for record in order_records if record["propagating"]),
        "orders": order_records,
    }


def compute_diffraction_orders(
    result,
    monitor_name: str,
    *,
    incident_power: float | None = None,
    background_index: float = 1.0,
    frequency: float | None = None,
    freq_index: int | None = None,
) -> dict[str, object]:
    payload = result.raw_monitor(monitor_name, frequency=frequency, freq_index=freq_index)
    if not _payload_is_diffraction(payload):
        raise ValueError(f"Monitor {monitor_name!r} is not a DiffractionMonitor.")
    return compute_diffraction_from_payload(
        result,
        monitor_name,
        payload,
        incident_power=incident_power,
        background_index=background_index,
    )
