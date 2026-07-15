from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.excitation.modes import _normalize_mode_profiles_to_unit_power
from witwin.maxwell.fdtd.excitation.injection import _mode_normal_stagger_power_factor


def _frozen_ez_mode(
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
    *,
    impedance: float = 200.0,
) -> dict[str, torch.Tensor]:
    u_phase = (coords_u - coords_u[0]) / (coords_u[-1] - coords_u[0])
    v_phase = (coords_v - coords_v[0]) / (coords_v[-1] - coords_v[0])
    profile = torch.sin(math.pi * u_phase)[:, None] * torch.sin(math.pi * v_phase)[None, :]
    zeros = torch.zeros_like(profile)
    return {
        "Ex": zeros.clone(),
        "Ey": zeros.clone(),
        "Ez": profile,
        "Hx": zeros.clone(),
        "Hy": -profile / float(impedance),
        "Hz": zeros.clone(),
    }


def _trapz2(values: torch.Tensor, coords_u: torch.Tensor, coords_v: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(
        torch.trapezoid(values, x=coords_v, dim=1),
        x=coords_u,
        dim=0,
    )


def _x_directed_yee_power(
    profiles: dict[str, torch.Tensor],
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
) -> torch.Tensor:
    coords_u_half = 0.5 * (coords_u[:-1] + coords_u[1:])
    coords_v_half = 0.5 * (coords_v[:-1] + coords_v[1:])
    ey = 0.5 * (profiles["Ey"][:-1, :] + profiles["Ey"][1:, :])
    hz = 0.5 * (profiles["Hz"][:-1, :] + profiles["Hz"][1:, :])
    ez = 0.5 * (profiles["Ez"][:, :-1] + profiles["Ez"][:, 1:])
    hy = 0.5 * (profiles["Hy"][:, :-1] + profiles["Hy"][:, 1:])
    return _trapz2(0.5 * torch.real(ey * torch.conj(hz)), coords_u_half, coords_v) - _trapz2(
        0.5 * torch.real(ez * torch.conj(hy)),
        coords_u,
        coords_v_half,
    )


def _x_directed_modal_amplitudes(
    total: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords_v_half = 0.5 * (coords_v[:-1] + coords_v[1:])
    ez_total = 0.5 * (total["Ez"][:, :-1] + total["Ez"][:, 1:])
    hy_total = 0.5 * (total["Hy"][:, :-1] + total["Hy"][:, 1:])
    ez_ref = 0.5 * (reference["Ez"][:, :-1] + reference["Ez"][:, 1:])
    hy_ref = 0.5 * (reference["Hy"][:, :-1] + reference["Hy"][:, 1:])
    electric_pairing = _trapz2(
        -ez_total * torch.conj(hy_ref),
        coords_u,
        coords_v_half,
    )
    magnetic_pairing = _trapz2(
        -torch.conj(ez_ref) * hy_total,
        coords_u,
        coords_v_half,
    )
    reference_power = _x_directed_yee_power(reference, coords_u, coords_v)
    return (
        (electric_pairing + magnetic_pairing) / (4.0 * reference_power),
        (electric_pairing - magnetic_pairing) / (4.0 * reference_power),
    )


def test_frozen_mode_profile_is_normalized_on_the_yee_planes() -> None:
    coords_u = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)
    coords_v = torch.linspace(-0.4, 0.4, 9, dtype=torch.float64)
    raw_profiles = _frozen_ez_mode(coords_u, coords_v)
    raw_power = _x_directed_yee_power(raw_profiles, coords_u, coords_v)
    profiles = _normalize_mode_profiles_to_unit_power(
        raw_profiles,
        coords_u=coords_u,
        coords_v=coords_v,
        normal_axis="x",
    )

    power = _x_directed_yee_power(profiles, coords_u, coords_v)
    cross_polarized_energy = torch.sum(torch.abs(profiles["Ey"]) ** 2) + torch.sum(
        torch.abs(profiles["Hz"]) ** 2
    )

    assert raw_power.item() > 0.0
    assert raw_power.item() != pytest.approx(1.0, rel=1e-3)
    assert power.item() == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert cross_polarized_energy.item() == pytest.approx(0.0, abs=1e-12)


def test_mode_profile_normalization_rejects_mismatched_coordinate_contract() -> None:
    coords_u = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)
    coords_v = torch.linspace(-0.4, 0.4, 9, dtype=torch.float64)
    profiles = _frozen_ez_mode(coords_u[1:-1], coords_v[1:-1])

    with pytest.raises(ValueError, match="profile shape"):
        _normalize_mode_profiles_to_unit_power(
            profiles,
            coords_u=coords_u,
            coords_v=coords_v,
            normal_axis="x",
        )


def test_frozen_mode_profile_has_zero_backward_leakage() -> None:
    coords_u = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)
    coords_v = torch.linspace(-0.4, 0.4, 9, dtype=torch.float64)
    forward = _normalize_mode_profiles_to_unit_power(
        _frozen_ez_mode(coords_u, coords_v),
        coords_u=coords_u,
        coords_v=coords_v,
        normal_axis="x",
    )
    backward = {
        name: (-profile if name.startswith("H") else profile)
        for name, profile in forward.items()
    }

    forward_amplitude, backward_leakage = _x_directed_modal_amplitudes(
        forward,
        forward,
        coords_u,
        coords_v,
    )
    reverse_forward_leakage, reverse_amplitude = _x_directed_modal_amplitudes(
        backward,
        forward,
        coords_u,
        coords_v,
    )

    assert forward_amplitude.item() == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert backward_leakage.item() == pytest.approx(0.0, abs=1e-12)
    assert reverse_forward_leakage.item() == pytest.approx(0.0, abs=1e-12)
    assert reverse_amplitude.item() == pytest.approx(1.0, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("direction_sign", "half_index"), ((1, 1), (-1, 2)))
def test_normal_yee_half_cell_power_factor_is_direction_symmetric(
    direction_sign: int,
    half_index: int,
) -> None:
    nodes = torch.tensor((-0.2, -0.1, 0.0, 0.1, 0.2), dtype=torch.float64)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    solver = SimpleNamespace(
        scene=SimpleNamespace(x_half=half),
        _plane_coordinate=lambda axis, index: float(nodes[index]),
    )
    source = {"normal_axis": "x", "direction_sign": direction_sign}
    mode_data = {"plane_index": 2, "beta": 4.0}

    factor = _mode_normal_stagger_power_factor(solver, source, mode_data)
    expected = math.cos(4.0 * abs(float(half[half_index]) - float(nodes[2])))
    injection_scale = 1.0 / math.sqrt(factor)

    assert factor == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert factor * injection_scale**2 == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_normal_yee_half_cell_correction_preserves_coarse_grid_setup() -> None:
    nodes = torch.tensor((-0.2, -0.1, 0.0, 0.1, 0.2), dtype=torch.float64)
    solver = SimpleNamespace(
        scene=SimpleNamespace(x_half=0.5 * (nodes[:-1] + nodes[1:])),
        _plane_coordinate=lambda axis, index: float(nodes[index]),
    )
    source = {"normal_axis": "x", "direction_sign": 1}
    mode_data = {"plane_index": 2, "beta": 40.0}

    assert _mode_normal_stagger_power_factor(solver, source, mode_data) == 1.0
