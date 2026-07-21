"""Operator-level gates for interior-PEC masking on the Yee-staggered mode operator (F2a).

Gate taxonomy: **modal-eigensolve / quasi-static** solver-correctness regressions on
the interior-PEC-masked transverse operator and the companion quasi-static electrostatic
line-mode engine, not wave-level exit gates.

Two engines are pinned here:

1. ``_solve_yee_transverse_pec_mode`` -- the staggered curl-curl ``beta**2`` operator with
   conductor-interior transverse-component unknowns eliminated (Dirichlet 0) and the
   conductor-interior longitudinal nodes dropped from the ``eps_ww`` divergence coupling.
   This serves the **guided** (non-TEM, hybrid) interior-PEC modes; it structurally does
   not carry the TEM branch (its spectral maximum is the lowest-cutoff guided mode, not
   the ``beta**2 = eps k0**2`` gradient TEM), so a ``wave_family="tem"`` request fails
   closed. The masking is validated against a PEC-septum half-guide whose cutoff is
   analytic.

2. ``_solve_quasistatic_line_modes`` -- the quasi-static electrostatic line-mode engine
   that serves the TEM/quasi-TEM interior-PEC lines (coax, microstrip, differential pair)
   via the capacitance ratio ``eps_eff = C / C0`` and ``beta = k0 sqrt(eps_eff)``.

All gates run in float64 as CPU oracles (permitted for tests).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from witwin.maxwell.fdtd.excitation.modes import (
    _build_yee_transverse_operator_sparse,
    _label_connected_components,
    _solve_pec_tem_mode_torch,
    _solve_quasistatic_line_modes,
    _solve_yee_transverse_pec_mode,
    _yee_pec_connectivity_check,
    _yee_stagger_pec_from_nodes,
    _yee_stagger_eps_from_nodes,
)

_FIELD_NAMES = ("Eu", "Ev", "Hu", "Hv")


def _hammerstad_jensen_eps_eff(w_over_h: float, eps_r: float) -> float:
    """Hammerstad-Jensen (1980) quasi-static microstrip effective permittivity.

    Reference: E. Hammerstad and O. Jensen, "Accurate Models for Microstrip
    Computer-Aided Design", IEEE MTT-S 1980. Valid for W/h >= ~0.05.
    """
    u = float(w_over_h)
    a = (
        1.0
        + (1.0 / 49.0) * math.log((u ** 4 + (u / 52.0) ** 2) / (u ** 4 + 0.432))
        + (1.0 / 18.7) * math.log(1.0 + (u / 18.1) ** 3)
    )
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    return (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b)


def _uniform_nodes(nu_nodes: int, nv_nodes: int, eps_r: float):
    plane = np.full((nu_nodes, nv_nodes), float(eps_r), dtype=np.float64)
    return (plane.copy(), plane.copy(), plane.copy())


def _square_annulus_occupancy(nu_nodes: int, nv_nodes: int, inner_fraction: float) -> np.ndarray:
    occ = np.zeros((nu_nodes, nv_nodes), dtype=np.float64)
    u0 = int(nu_nodes * (0.5 - inner_fraction / 2.0))
    u1 = int(nu_nodes * (0.5 + inner_fraction / 2.0))
    v0 = int(nv_nodes * (0.5 - inner_fraction / 2.0))
    v1 = int(nv_nodes * (0.5 + inner_fraction / 2.0))
    occ[u0:u1, v0:v1] = 1.0
    return occ


def _microstrip_planes(*, eps_r, h, w, box_width, box_height, nc_u, nc_v):
    du = box_height / nc_u
    dv = box_width / nc_v
    u = np.arange(nc_u + 1) * du
    v = np.arange(nc_v + 1) * dv
    grid_u, _ = np.meshgrid(u, v, indexing="ij")
    eps = np.where(grid_u <= h + 1e-9, float(eps_r), 1.0)
    return eps, u, v, du, dv


# --------------------------------------------------------------------------------------
# 1. Masking mechanics
# --------------------------------------------------------------------------------------


def test_interior_pec_masking_eliminates_conductor_unknowns_and_keeps_symmetry():
    nc = 40
    du = dv = 1.0 / nc
    eps_planes = _uniform_nodes(nc + 1, nc + 1, 2.25)
    occ = _square_annulus_occupancy(nc + 1, nc + 1, 0.3)
    eps_uu, eps_vv, eps_ww = _yee_stagger_eps_from_nodes(eps_planes, nu_cells=nc, nv_cells=nc)
    pec_eu, pec_ev, pec_ww = _yee_stagger_pec_from_nodes(occ, nu_cells=nc, nv_cells=nc, threshold=0.5)

    operator, meta = _build_yee_transverse_operator_sparse(
        nu_cells=nc, nv_cells=nc, du=du, dv=dv, k0=5.0,
        eps_uu=eps_uu, eps_vv=eps_vv, eps_ww=eps_ww,
        pec_eu=pec_eu, pec_ev=pec_ev, pec_ww=pec_ww,
    )
    active = np.asarray(meta["pec_active"], dtype=bool)
    expected_active = int((~pec_eu).sum() + (~pec_ev).sum())
    assert int(active.sum()) == expected_active
    assert int((~active).sum()) == int(pec_eu.sum() + pec_ev.sum()) > 0

    # Conductor nodes drop from the eps_ww coupling: the stored interior-node mask matches.
    assert np.array_equal(np.asarray(meta["pec_ww_mask"], dtype=bool), pec_ww)

    # A uniform-fill masked operator stays symmetric (same symmetric Dirichlet elimination
    # as the outer walls -- no penalty terms, no asymmetry).
    reduced = operator[active][:, active].toarray()
    asymmetry = np.max(np.abs(reduced - reduced.T))
    scale = np.max(np.abs(reduced))
    assert asymmetry <= 1e-9 * scale


def test_interior_pec_masked_profile_is_zero_inside_conductor():
    # A full-span PEC septum: the transverse fields must vanish on the conductor row.
    au, bv = 2.0, 1.0
    nc_u, nc_v = 48, 24
    du, dv = au / nc_u, bv / nc_v
    eps = np.ones((nc_u + 1, nc_v + 1), dtype=np.float64)
    occ = np.zeros((nc_u + 1, nc_v + 1), dtype=np.float64)
    septum_row = int(round(1.0 / du))
    occ[septum_row, :] = 1.0

    beta, profiles, _ = _solve_yee_transverse_pec_mode(
        (eps.copy(), eps.copy(), eps.copy()), occ, k0=4.0, du=du, dv=dv, mode_index=0,
        field_names=_FIELD_NAMES, preferred_field_name="Ev", wave_family=None,
        uniform=True, use_dense=True,
    )
    peak = max(float(np.max(np.abs(profiles[name]))) for name in _FIELD_NAMES)
    for name in _FIELD_NAMES:
        septum_amplitude = float(np.max(np.abs(np.asarray(profiles[name])[septum_row, :])))
        assert septum_amplitude <= 5e-2 * peak


# --------------------------------------------------------------------------------------
# 2. Connectivity check
# --------------------------------------------------------------------------------------


def test_connectivity_reports_coax_regions():
    nc = 40
    occ = _square_annulus_occupancy(nc + 1, nc + 1, 0.3)
    _, _, pec_ww = _yee_stagger_pec_from_nodes(occ, nu_cells=nc, nv_cells=nc, threshold=0.5)
    report = _yee_pec_connectivity_check(pec_ww)
    # One dielectric annulus around one interior conductor block.
    assert report["dielectric_regions"] == 1
    assert report["conductor_regions"] == 1
    assert report["active_interior_nodes"] > 0


def test_connectivity_pinch_point_raises():
    # A conductor-free node fully surrounded by conductor is a degenerate pinch, even when
    # a large conductor-free region exists elsewhere on the plane.
    pec_ww = np.zeros((7, 7), dtype=bool)
    pec_ww[2:5, 2:5] = True   # conductor block
    pec_ww[3, 3] = False      # one dielectric node fully surrounded by that block
    with pytest.raises(ValueError, match="fully surrounded by conductor"):
        _yee_pec_connectivity_check(pec_ww)


def test_label_connected_components_counts_two_conductors():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4:6, 5:7] = True
    _, count = _label_connected_components(mask)
    assert count == 2


# --------------------------------------------------------------------------------------
# 3. Coax quasi-static TEM (cross-implementation vs legacy + analytic)
# --------------------------------------------------------------------------------------


def test_coax_quasistatic_tem_matches_analytic_and_legacy():
    nc = 40
    du = dv = 1.0 / nc
    eps_r = 2.25
    k0 = 5.0
    eps_planes_np = _uniform_nodes(nc + 1, nc + 1, eps_r)
    occ = _square_annulus_occupancy(nc + 1, nc + 1, 0.3)

    result = _solve_quasistatic_line_modes(
        tuple(torch.tensor(plane) for plane in eps_planes_np), torch.tensor(occ),
        k0=k0, du=du, dv=dv, field_names=("Ey", "Ez", "Hy", "Hz"), signal_potentials=[1.0],
    )
    # Uniform fill: eps_eff is exactly eps_r (capacitance ratio), beta = k0 sqrt(eps_r).
    assert result["eps_eff"] == pytest.approx(eps_r, rel=1e-6)
    beta_analytic = k0 * math.sqrt(eps_r)
    assert float(result["beta"]) == pytest.approx(beta_analytic, rel=1e-6)
    assert result["isolated_conductor_count"] == 1

    # Cross-implementation: the legacy electrostatic TEM path returns the same beta.
    eps_torch = tuple(torch.tensor(plane) for plane in eps_planes_np)
    mu_torch = tuple(torch.ones((nc + 1, nc + 1), dtype=torch.float64) for _ in range(3))
    beta_legacy, _, _ = _solve_pec_tem_mode_torch(
        eps_torch, mu_torch, torch.tensor(occ), k0=k0, du=du, dv=dv, mode_index=0,
        field_names=("Ey", "Ez", "Hy", "Hz"),
    )
    assert float(result["beta"]) == pytest.approx(float(beta_legacy), rel=1e-6)


def test_parallel_plate_limit_eps_eff_equals_fill():
    # Interior PEC plate spanning the full width, uniform dielectric fill: the region
    # between the plate and the ground wall is a parallel-plate capacitor whose TEM
    # effective permittivity is exactly the fill permittivity (analytic parallel-plate).
    eps_r = 3.0
    nc_u, nc_v = 40, 40
    du = dv = 1.0 / nc_u
    eps = np.full((nc_u + 1, nc_v + 1), eps_r, dtype=np.float64)
    occ = np.zeros((nc_u + 1, nc_v + 1), dtype=np.float64)
    plate_row = nc_u // 2
    occ[plate_row, 1:-1] = 1.0  # isolated plate spanning the interior width
    result = _solve_quasistatic_line_modes(
        (torch.tensor(eps), torch.tensor(eps), torch.tensor(eps)), torch.tensor(occ),
        k0=5.0, du=du, dv=dv, field_names=_FIELD_NAMES, signal_potentials=[1.0],
    )
    assert result["eps_eff"] == pytest.approx(eps_r, rel=1e-6)


# --------------------------------------------------------------------------------------
# 4. Microstrip quasi-static eps_eff vs Hammerstad-Jensen
# --------------------------------------------------------------------------------------

# Pre-registered tolerance: the quasi-static energy method reproduces the Hammerstad-Jensen
# closed form to within 3% on this grid (measured 0.7% at nc=(60,100)); the residual is the
# finite-box truncation and the staircased strip edge.
_MICROSTRIP_EPS_EFF_TOL = 0.03


def test_microstrip_quasistatic_eps_eff_matches_hammerstad_jensen():
    eps_r = 4.0
    h = 0.2
    w = 0.4          # W/h = 2
    box_width, box_height = 2.0, 1.2
    nc_u, nc_v = 60, 100
    eps, u, v, du, dv = _microstrip_planes(
        eps_r=eps_r, h=h, w=w, box_width=box_width, box_height=box_height, nc_u=nc_u, nc_v=nc_v
    )
    occ = np.zeros_like(eps)
    strip_row = int(round(h / du))
    occ[strip_row, np.abs(v - box_width / 2.0) <= w / 2.0 + 1e-9] = 1.0

    result = _solve_quasistatic_line_modes(
        (torch.tensor(eps), torch.tensor(eps), torch.tensor(eps)), torch.tensor(occ),
        k0=5.0, du=du, dv=dv, field_names=_FIELD_NAMES, signal_potentials=[1.0],
    )
    hj = _hammerstad_jensen_eps_eff(w / h, eps_r)
    assert 1.0 < result["eps_eff"] < eps_r
    assert abs(result["eps_eff"] - hj) / hj <= _MICROSTRIP_EPS_EFF_TOL


def test_microstrip_without_dielectric_has_unit_eps_eff():
    # Falsification companion: replacing the substrate with vacuum drives eps_eff -> 1.
    box_width, box_height = 2.0, 1.2
    nc_u, nc_v = 60, 100
    eps = np.ones((nc_u + 1, nc_v + 1), dtype=np.float64)
    du, dv = box_height / nc_u, box_width / nc_v
    v = np.arange(nc_v + 1) * dv
    occ = np.zeros_like(eps)
    strip_row = int(round(0.2 / du))
    occ[strip_row, np.abs(v - box_width / 2.0) <= 0.2 + 1e-9] = 1.0
    result = _solve_quasistatic_line_modes(
        (torch.tensor(eps), torch.tensor(eps), torch.tensor(eps)), torch.tensor(occ),
        k0=5.0, du=du, dv=dv, field_names=_FIELD_NAMES, signal_potentials=[1.0],
    )
    assert result["eps_eff"] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------------------
# 5. Differential pair even/odd modes
# --------------------------------------------------------------------------------------


def _parity_errors(potential, center_index):
    field = np.asarray(potential.detach().cpu().numpy())
    left = field[:, :center_index]
    right = np.flip(field[:, center_index + 1:], axis=1)
    n = min(left.shape[1], right.shape[1])
    symmetric_error = float(np.mean((left[:, -n:] - right[:, :n]) ** 2))
    antisymmetric_error = float(np.mean((left[:, -n:] + right[:, :n]) ** 2))
    return symmetric_error, antisymmetric_error


def test_differential_pair_even_odd_modes_are_distinct_and_parity_classified():
    eps_r = 4.0
    h = 0.2
    w = 0.4
    box_width, box_height = 2.0, 1.2
    nc_u, nc_v = 60, 100
    eps, u, v, du, dv = _microstrip_planes(
        eps_r=eps_r, h=h, w=w, box_width=box_width, box_height=box_height, nc_u=nc_u, nc_v=nc_v
    )
    occ = np.zeros_like(eps)
    strip_row = int(round(h / du))
    center = box_width / 2.0
    occ[strip_row, np.abs(v - (center - 0.3)) <= w / 2.0 + 1e-9] = 1.0
    occ[strip_row, np.abs(v - (center + 0.3)) <= w / 2.0 + 1e-9] = 1.0
    eps_planes = (torch.tensor(eps), torch.tensor(eps), torch.tensor(eps))

    even = _solve_quasistatic_line_modes(
        eps_planes, torch.tensor(occ), k0=5.0, du=du, dv=dv,
        field_names=_FIELD_NAMES, signal_potentials=[1.0, 1.0],
    )
    odd = _solve_quasistatic_line_modes(
        eps_planes, torch.tensor(occ), k0=5.0, du=du, dv=dv,
        field_names=_FIELD_NAMES, signal_potentials=[1.0, -1.0],
    )

    assert even["isolated_conductor_count"] == 2
    # Two distinct modes, both physical; the even (common) mode traps more field in the
    # substrate so its effective permittivity is the larger one.
    assert 1.0 < odd["eps_eff"] < even["eps_eff"] < eps_r
    assert abs(even["eps_eff"] - odd["eps_eff"]) / even["eps_eff"] > 0.01
    assert float(even["beta"]) > float(odd["beta"]) > 0.0

    center_index = nc_v // 2
    even_sym, even_anti = _parity_errors(even["potential"], center_index)
    odd_sym, odd_anti = _parity_errors(odd["potential"], center_index)
    # Even mode potential is mirror-symmetric about the pair centreline; odd is antisymmetric.
    assert even_sym < 1e-6 * even_anti
    assert odd_anti < 1e-6 * odd_sym


# --------------------------------------------------------------------------------------
# 6. Masked operator guided mode (analytic septum) + falsification
# --------------------------------------------------------------------------------------


def test_masked_operator_guided_septum_matches_half_guide_analytic():
    # Rectangular guide a=2 (u) x b=1 (v). A full-span PEC septum at u=1 splits it into two
    # 1x1 half-guides whose lowest TE cutoff rises from k~c=pi/2 (full) to k~c=pi (half).
    au, bv = 2.0, 1.0
    nc_u, nc_v = 48, 24
    du, dv = au / nc_u, bv / nc_v
    k0 = 4.0
    eps = np.ones((nc_u + 1, nc_v + 1), dtype=np.float64)
    occ = np.zeros((nc_u + 1, nc_v + 1), dtype=np.float64)
    septum_row = int(round(1.0 / du))
    occ[septum_row, :] = 1.0

    beta_masked, _, diagnostics = _solve_yee_transverse_pec_mode(
        (eps.copy(), eps.copy(), eps.copy()), occ, k0=k0, du=du, dv=dv, mode_index=0,
        field_names=_FIELD_NAMES, preferred_field_name="Ev", wave_family=None,
        uniform=True, use_dense=True,
    )
    beta_half_guide = math.sqrt(k0 ** 2 - (math.pi / 1.0) ** 2)
    beta_full_guide = math.sqrt(k0 ** 2 - (math.pi / 2.0) ** 2)
    assert float(np.real(beta_masked)) == pytest.approx(beta_half_guide, rel=5e-3)
    # The masking is load-bearing: the septum answer is far from the un-split guide.
    assert abs(float(np.real(beta_masked)) - beta_full_guide) > 0.2 * beta_full_guide
    # Two conductor-free half-guides separated by one conductor septum.
    assert diagnostics["connectivity"]["dielectric_regions"] == 2
    assert diagnostics["connectivity"]["conductor_regions"] == 1


def test_masked_operator_rejects_tem_request():
    # The curl-curl beta**2 operator has no TEM branch; a TEM request fails closed.
    nc = 40
    du = dv = 1.0 / nc
    eps = np.ones((nc + 1, nc + 1), dtype=np.float64)
    occ = _square_annulus_occupancy(nc + 1, nc + 1, 0.3)
    with pytest.raises(ValueError, match="does not support TEM"):
        _solve_yee_transverse_pec_mode(
            (eps.copy(), eps.copy(), eps.copy()), occ, k0=5.0, du=du, dv=dv, mode_index=0,
            field_names=_FIELD_NAMES, preferred_field_name="Ev", wave_family="tem",
            uniform=True, use_dense=True,
        )


def test_masked_operator_under_resolved_conductor_raises():
    # Occupancy that never reaches the staggered threshold is fail-closed, not silently
    # treated as a hollow guide.
    nc = 40
    du = dv = 1.0 / nc
    eps = np.ones((nc + 1, nc + 1), dtype=np.float64)
    occ = np.zeros((nc + 1, nc + 1), dtype=np.float64)
    occ[20, 20] = 0.2  # below threshold everywhere after staggering
    with pytest.raises(ValueError, match="under-resolved|no staggered component"):
        _solve_yee_transverse_pec_mode(
            (eps.copy(), eps.copy(), eps.copy()), occ, k0=5.0, du=du, dv=dv, mode_index=0,
            field_names=_FIELD_NAMES, preferred_field_name="Ev", wave_family=None,
            uniform=True, use_dense=True,
        )
