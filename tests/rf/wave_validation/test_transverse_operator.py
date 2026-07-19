"""Golden gates for the Yee-staggered transverse full-vector mode operator (E1a).

Gate taxonomy (S0.3): **modal-eigensolve** (a solver-correctness regression on the
transverse operator itself, not a wave-level exit gate).

These tests pin the genuinely Yee-staggered transverse operator
``_build_yee_transverse_operator_sparse``, which keeps each transverse electric
component on its own Yee location (``Eu`` on the ``u`` half / ``v`` node grid,
``Ev`` on the ``u`` node / ``v`` half grid). Unlike the legacy centered
uniform-isotropic branch -- whose stride-two stencil decoupled the odd/even
transverse sublattices and capped the full-grid ``sin(pi u/a)`` correlation of the
TE10 mode at 0.51-0.59 with ``checkerboard_fraction > 0.35`` (see
``docs/reference/rf-wave-validation-2026-07-18.md`` sections 1.2 / 5) -- this
operator reproduces the closed-form discrete eigenpairs of a hollow rectangular
guide to machine precision.

The discretization has exact eigenpairs: for a homogeneous (vacuum) cross-section
the operator decouples into two scalar transverse Helmholtz problems with the
physically correct mixed walls built in by the staggering, and

    beta**2 = k0**2 - k~(m, a)**2 - k~(n, b)**2 ,
    k~(m, a) = (2 / dx) * sin(m * pi * dx / (2 a)) ,

with the transverse profile a separable product of ``sin`` (Dirichlet, tangential
wall) and ``cos`` (natural / Neumann, normal wall) factors. All gates run in
float64 on the CPU as an oracle (permitted for tests); no selector is touched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as scipy_sparse_linalg
from scipy.sparse.csgraph import connected_components

from witwin.maxwell.fdtd.excitation.modes import (
    _build_yee_transverse_operator_sparse,
    _split_yee_transverse_eigenvector,
    _yee_stagger_eps_from_nodes,
    _yee_transverse_discrete_transverse_wavenumber as _ktilde,
)

# Wall-to-wall aperture: a along u (larger) and b along v; a != 2 b so the
# TE20 and TE01 branches stay non-degenerate and TE20 is unambiguously the
# sin(2 pi u / a) profile. k0 is chosen above every cutoff exercised here so the
# reported modes are propagating (beta**2 > 0).
_APERTURE_A = 1.0
_APERTURE_B = 0.6
_NU_CELLS = 24
_NV_CELLS = 12
_DU = _APERTURE_A / _NU_CELLS
_DV = _APERTURE_B / _NV_CELLS
_K0 = 10.0


def _homogeneous_operator(*, nu=_NU_CELLS, nv=_NV_CELLS, du=_DU, dv=_DV, k0=_K0):
    return _build_yee_transverse_operator_sparse(
        nu_cells=nu, nv_cells=nv, du=du, dv=dv, k0=k0
    )


def _analytic_beta_sq(m: int, n: int, *, a=_APERTURE_A, b=_APERTURE_B, du=_DU, dv=_DV, k0=_K0) -> float:
    return k0 * k0 - _ktilde(m, a, du) ** 2 - _ktilde(n, b, dv) ** 2


def _full_analytic_spectrum() -> np.ndarray:
    """Closed-form discrete beta**2 for the homogeneous guide, both E-blocks.

    ``Ev`` (u Dirichlet, v Neumann): m = 1 .. nu-1, n = 0 .. nv-1.
    ``Eu`` (u Neumann, v Dirichlet): m = 0 .. nu-1, n = 1 .. nv-1.
    """
    ev_block = [_analytic_beta_sq(m, n) for m in range(1, _NU_CELLS) for n in range(0, _NV_CELLS)]
    eu_block = [_analytic_beta_sq(m, n) for m in range(0, _NU_CELLS) for n in range(1, _NV_CELLS)]
    return np.sort(np.array(ev_block + eu_block, dtype=np.float64))[::-1]


def _checkerboard_fraction(eu: np.ndarray, ev: np.ndarray) -> float:
    """Normalized one-cell electric-field variation (Nyquist content near one)."""
    profiles = (eu, ev)
    energy = sum(float(np.vdot(p, p).real) for p in profiles)
    if energy <= 0.0:
        return 1.0
    variation = 0.0
    for p in profiles:
        variation += float(np.sum(np.abs(np.diff(p, axis=0)) ** 2))
        variation += float(np.sum(np.abs(np.diff(p, axis=1)) ** 2))
    return variation / (8.0 * energy)


def _sorted_eigpairs(operator):
    dense = operator.toarray()
    values, vectors = scipy_linalg.eigh(dense)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order], dense


def _select_by_beta_sq(values, vectors, target: float):
    index = int(np.argmin(np.abs(values - target)))
    return values[index], vectors[:, index]


def _profile_correlation(profile: np.ndarray, reference: np.ndarray) -> float:
    flat = profile.reshape(-1).astype(np.float64)
    ref = reference.reshape(-1).astype(np.float64)
    if np.dot(flat, ref) < 0:
        flat = -flat
    denom = np.linalg.norm(flat) * np.linalg.norm(ref)
    return float(np.dot(flat, ref) / denom) if denom > 0 else 0.0


def test_operator_is_symmetric_for_homogeneous_cross_section() -> None:
    """Vacuum cross-section: the operator is real-symmetric so beta**2 is real."""
    operator, _ = _homogeneous_operator()
    dense = operator.toarray()
    assert np.allclose(dense, dense.T, atol=1e-9)


def _uniform_filled_operator(eps_r: float, *, nu=_NU_CELLS, nv=_NV_CELLS, du=_DU, dv=_DV, k0=_K0):
    """Yee operator for a homogeneous cross-section uniformly filled with ``eps_r``."""
    node = np.full((nu + 1, nv + 1), float(eps_r), dtype=np.float64)
    eps_uu, eps_vv, eps_ww = _yee_stagger_eps_from_nodes((node, node, node), nu_cells=nu, nv_cells=nv)
    return _build_yee_transverse_operator_sparse(
        nu_cells=nu, nv_cells=nv, du=du, dv=dv, k0=k0, eps_uu=eps_uu, eps_vv=eps_vv, eps_ww=eps_ww
    )


def test_uniform_dielectric_fill_shifts_spectrum_and_stays_symmetric() -> None:
    """A uniformly dielectric-filled guide carries the filled propagation constant.

    Regression for the routing defect where a uniform eps_r != 1 aperture was solved
    with the vacuum operator and returned the vacuum beta. For a homogeneous
    cross-section the Yee operator equals the vacuum operator plus a scalar
    ``(eps_r - 1) * k0**2`` identity shift, so (a) it stays exactly symmetric and
    (b) every discrete eigenvalue is exactly its vacuum counterpart shifted by
    ``(eps_r - 1) * k0**2``. The fundamental therefore sits at
    ``beta**2 = eps_r * k0**2 - k~x**2 - k~y**2``, NOT the vacuum value.
    """
    eps_r = 4.0
    vac_operator, _ = _homogeneous_operator()
    filled_operator, _ = _uniform_filled_operator(eps_r)
    dense = filled_operator.toarray()
    assert np.allclose(dense, dense.T, atol=1e-9)

    shift = (eps_r - 1.0) * _K0 * _K0
    expected = vac_operator.toarray() + shift * np.eye(dense.shape[0])
    assert np.allclose(dense, expected, atol=1e-9)

    vac_beta_sq = float(np.linalg.eigvalsh(vac_operator.toarray()).max())
    filled_beta_sq = float(np.linalg.eigvalsh(dense).max())
    analytic = eps_r * _K0 * _K0 - _ktilde(1, _APERTURE_A, _DU) ** 2 - _ktilde(0, _APERTURE_B, _DV) ** 2
    assert abs(filled_beta_sq - analytic) <= 1e-9 * abs(analytic)
    # The filled beta must not collapse onto the vacuum value (the defect).
    assert filled_beta_sq == pytest.approx(vac_beta_sq + shift, rel=1e-12)
    assert abs(filled_beta_sq - vac_beta_sq) > 0.5 * shift


def test_full_discrete_spectrum_matches_closed_form() -> None:
    """Every eigenvalue equals the closed-form discrete beta**2 (rtol <= 1e-10)."""
    operator, _ = _homogeneous_operator()
    values = np.sort(np.linalg.eigvalsh(operator.toarray()))[::-1]
    analytic = _full_analytic_spectrum()
    assert values.shape == analytic.shape
    rel = np.abs(values - analytic) / np.maximum(np.abs(analytic), 1.0)
    assert float(np.max(rel)) <= 1e-10, f"max spectrum rtol {float(np.max(rel)):.2e}"


def test_te10_eigenpair_is_exact_sin() -> None:
    """TE10: eigenvalue matches (rtol <= 1e-10), Ev is sin(pi u/a), clean (cb < 0.05)."""
    operator, meta = _homogeneous_operator()
    values, vectors, _ = _sorted_eigpairs(operator)
    target = _analytic_beta_sq(1, 0)
    # TE10 is the fundamental (largest beta**2) for a > b.
    beta_sq = values[0]
    assert abs(beta_sq - target) / abs(target) <= 1e-10

    eu, ev = _split_yee_transverse_eigenvector(vectors[:, 0], meta)
    # A pure Ez (=Ev) TE mode: the Eu block carries no energy.
    assert np.sum(eu ** 2) <= 1e-18 * np.sum(ev ** 2) + 1e-24
    reference = np.outer(np.sin(math.pi * meta["ev_u"] / meta["extent_u"]), np.ones(ev.shape[1]))
    corr = _profile_correlation(ev, reference)
    assert corr >= 0.9999, f"TE10 Ev sin-correlation {corr:.6f} < 0.9999"
    cb = _checkerboard_fraction(eu, ev)
    assert cb < 0.05, f"TE10 checkerboard_fraction {cb:.4f} >= 0.05"


def test_te20_eigenpair_is_exact_sin() -> None:
    """TE20: eigenvalue matches and Ev is sin(2 pi u/a), uniform in v."""
    operator, meta = _homogeneous_operator()
    values, vectors, _ = _sorted_eigpairs(operator)
    target = _analytic_beta_sq(2, 0)
    beta_sq, vector = _select_by_beta_sq(values, vectors, target)
    assert abs(beta_sq - target) / abs(target) <= 1e-10

    eu, ev = _split_yee_transverse_eigenvector(vector, meta)
    assert np.sum(eu ** 2) <= 1e-18 * np.sum(ev ** 2) + 1e-24
    reference = np.outer(np.sin(2.0 * math.pi * meta["ev_u"] / meta["extent_u"]), np.ones(ev.shape[1]))
    corr = _profile_correlation(ev, reference)
    assert corr >= 0.9999, f"TE20 Ev sin(2 pi)-correlation {corr:.6f} < 0.9999"


def test_tm11_analytic_field_is_an_exact_eigenvector() -> None:
    """TM11: sin(pi u/a) cos(pi v/b) Ev is an eigenvector at the closed-form beta**2.

    TM11 is degenerate with TE11 in the homogeneous guide, so the eigensolver may
    return any rotation of the 2D subspace. This gate is rotation-independent: it
    verifies that the analytic TM11 Ev profile satisfies ``P v = beta**2 v`` to
    machine precision, which is only possible if the discrete operator carries the
    exact TM11 eigenpair.
    """
    operator, meta = _homogeneous_operator()
    dense = operator.toarray()
    target = _analytic_beta_sq(1, 1)
    ev_analytic = np.outer(
        np.sin(math.pi * meta["ev_u"] / meta["extent_u"]),
        np.cos(math.pi * meta["ev_v"] / meta["extent_v"]),
    )
    vector = np.concatenate([np.zeros(meta["n_eu"]), ev_analytic.reshape(-1)])
    applied = dense @ vector
    residual = np.linalg.norm(applied - target * vector) / np.linalg.norm(applied)
    assert residual <= 1e-10, f"TM11 exact-eigenvector residual {residual:.2e}"
    # The eigenvalue is present in the solved spectrum (multiplicity from TE11+TM11).
    values = np.linalg.eigvalsh(dense)
    assert float(np.min(np.abs(values - target)) / abs(target)) <= 1e-10


def test_homogeneous_blocks_are_connected_no_sublattice_decoupling() -> None:
    """Each E-component stencil graph is a single connected component.

    The legacy centered operator decoupled the odd/even transverse sublattices
    (stride-two stencil), producing checkerboard mode copies. The staggered
    operator's per-component blocks are ordinary 5-point graphs: connected across
    the whole component grid. (The Eu and Ev blocks decouple from EACH OTHER in a
    homogeneous medium -- that is the physically correct TE/TM separation, not a
    sublattice defect -- so connectivity is asserted per block.)
    """
    _, meta = _homogeneous_operator()
    for name in ("block_uu", "block_vv"):
        block = meta[name]
        adjacency = np.abs(block) + np.abs(block.transpose())
        n_components, _ = connected_components(adjacency, directed=False)
        assert n_components == 1, f"{name} decouples into {n_components} components"


def test_te10_converges_second_order_to_continuum() -> None:
    """Solved TE10 beta -> continuum sqrt(k0^2 - (pi/a)^2) at second order in dx."""
    continuum = math.sqrt(_K0 * _K0 - (math.pi / _APERTURE_A) ** 2)
    errors = []
    for nu in (16, 32, 64):
        nv = max(2, nu // 2)
        operator, _ = _homogeneous_operator(
            nu=nu, nv=nv, du=_APERTURE_A / nu, dv=_APERTURE_B / nv
        )
        beta = math.sqrt(float(np.max(np.linalg.eigvalsh(operator.toarray()))))
        errors.append(abs(beta - continuum))
    order_lo = math.log2(errors[0] / errors[1])
    order_hi = math.log2(errors[1] / errors[2])
    assert order_lo >= 1.9, f"coarse->mid convergence order {order_lo:.3f} < 1.9"
    assert order_hi >= 1.9, f"mid->fine convergence order {order_hi:.3f} < 1.9"


def test_operator_rejects_degenerate_grid() -> None:
    """A single-cell axis has no interior structure and is rejected fail-closed."""
    import pytest

    with pytest.raises(ValueError):
        _build_yee_transverse_operator_sparse(nu_cells=1, nv_cells=8, du=_DU, dv=_DV, k0=_K0)


# --- Inhomogeneous hybrid capability (E1b): half-filled parallel-plate guide ---
#
# The Yee-staggered operator carries per-component permittivity at each field's own
# Yee location, so it is the full-vectorial hybrid-mode solver for an inhomogeneous
# cross-section. A half-filled guide -- eps = EPS1 for u < D, EPS2 for u > D, uniform
# in v -- hosts an LSE mode uniform in v whose transverse-electric field Ev(u) obeys
# the 1D Sturm-Liouville problem  Ev'' + (k0^2 eps(u) - beta^2) Ev = 0  with Dirichlet
# walls, i.e. the classic slab-loaded transverse resonance
#   k1 cot(k1 D) + k2 cot(k2 (a - D)) = 0 ,  k_i^2 = k0^2 eps_i - beta^2 .
# The operator's inhomogeneous spectrum is validated to machine precision against an
# independently assembled 1D discrete operator and, in the continuum limit, against
# the analytic transcendental root.

_HF_A = 1.0
_HF_B = 0.6
_HF_K0 = 8.0
_HF_EPS1 = 4.0
_HF_EPS2 = 1.0
_HF_D = 0.5  # dielectric interface position along u


def _hf_eps_of_u(u: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(u) < _HF_D, _HF_EPS1, _HF_EPS2)


def _half_filled_operator(nu: int, nv: int):
    du = _HF_A / nu
    dv = _HF_B / nv
    u_half = (np.arange(nu) + 0.5) * du
    u_int = np.arange(1, nu) * du
    eps_uu = np.repeat(_hf_eps_of_u(u_half)[:, None], nv - 1, axis=1)
    eps_vv = np.repeat(_hf_eps_of_u(u_int)[:, None], nv, axis=1)
    eps_ww = np.repeat(_hf_eps_of_u(u_int)[:, None], nv - 1, axis=1)
    operator, meta = _build_yee_transverse_operator_sparse(
        nu_cells=nu, nv_cells=nv, du=du, dv=dv, k0=_HF_K0,
        eps_uu=eps_uu, eps_vv=eps_vv, eps_ww=eps_ww,
    )
    return operator, meta, u_int


def _one_dimensional_reference_beta_sq(nu: int) -> float:
    """Largest eigenvalue of the independently built 1D LSE Sturm-Liouville operator."""
    du = _HF_A / nu
    u_int = np.arange(1, nu) * du
    n = nu - 1
    main = np.full(n, -2.0 / (du * du))
    off = np.full(n - 1, 1.0 / (du * du))
    laplacian = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    operator = laplacian + _HF_K0 * _HF_K0 * np.diag(_hf_eps_of_u(u_int))
    return float(np.max(np.linalg.eigvalsh(operator)))


def _matched_inhomogeneous_eigenpair(nu: int, nv: int):
    operator, meta, u_int = _half_filled_operator(nu, nv)
    reference = _one_dimensional_reference_beta_sq(nu)
    # The inhomogeneous operator is real but non-symmetric and (as with every
    # full-vector transverse operator) carries spurious high-|beta| eigenvalues, so
    # the physical LSE mode is selected via a shift-invert eigensolve centred on the
    # 1D reference (fast and robust; the spectral maximum is spurious).
    eigenvalues, eigenvectors = scipy_sparse_linalg.eigs(
        operator, k=4, sigma=reference, which="LM"
    )
    real_mask = np.abs(eigenvalues.imag) < 1.0e-6 * np.maximum(np.abs(eigenvalues.real), 1.0)
    real_indices = np.where(real_mask)[0]
    best = real_indices[np.argmin(np.abs(eigenvalues.real[real_indices] - reference))]
    beta_sq = float(eigenvalues.real[best])
    eu, ev = _split_yee_transverse_eigenvector(eigenvectors[:, best].real, meta)
    return beta_sq, reference, eu, ev, u_int


def _analytic_lse_fundamental_beta() -> float:
    """Largest propagating root of the half-filled transverse-resonance condition."""
    from scipy import optimize

    def dispersion(beta: float) -> float:
        k1 = np.sqrt(complex(_HF_K0 ** 2 * _HF_EPS1 - beta * beta))
        k2 = np.sqrt(complex(_HF_K0 ** 2 * _HF_EPS2 - beta * beta))
        return float(np.real(k1 / np.tan(k1 * _HF_D) + k2 / np.tan(k2 * (_HF_A - _HF_D))))

    grid = np.linspace(0.01, _HF_K0 * math.sqrt(_HF_EPS1) - 1.0e-3, 40000)
    values = [dispersion(b) for b in grid]
    roots = []
    for i in range(len(grid) - 1):
        if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] < 0.0:
            try:
                roots.append(optimize.brentq(dispersion, grid[i], grid[i + 1]))
            except ValueError:
                pass
    return max(roots)


def test_inhomogeneous_operator_spectrum_matches_one_dimensional_lse_reference() -> None:
    """The half-filled operator's LSE mode equals the 1D discrete SL eigenvalue."""
    beta_sq, reference, _eu, _ev, _u_int = _matched_inhomogeneous_eigenpair(48, 24)
    assert abs(beta_sq - reference) <= 1.0e-10 * abs(reference), (
        f"inhomogeneous LSE beta^2 {beta_sq:.8f} != 1D reference {reference:.8f}"
    )


def test_inhomogeneous_lse_mode_is_v_uniform_ev_polarized_and_field_concentrates() -> None:
    """The LSE mode is uniform in v, Ev-dominant, and concentrated in the high-eps half."""
    _beta_sq, _reference, eu, ev, u_int = _matched_inhomogeneous_eigenpair(48, 24)
    ev_energy = float(np.sum(ev ** 2))
    eu_energy = float(np.sum(eu ** 2))
    assert eu_energy < 1.0e-12 * ev_energy, "LSE mode should be Ev-polarized (Eu ~ 0)"
    peak = float(np.max(np.abs(ev)))
    v_nonuniformity = float(np.max(np.std(np.abs(ev), axis=1))) / peak
    assert v_nonuniformity < 1.0e-9, f"LSE mode is not v-uniform ({v_nonuniformity:.2e})"
    high_eps = u_int < _HF_D
    high_energy = float(np.sum(ev[high_eps, :] ** 2))
    low_energy = float(np.sum(ev[~high_eps, :] ** 2))
    assert high_energy > 3.0 * low_energy, "LSE field should concentrate in the high-eps half"


def test_inhomogeneous_lse_beta_converges_to_analytic_transverse_resonance() -> None:
    """Operator LSE beta approaches the analytic slab-loaded transcendental root."""
    analytic = _analytic_lse_fundamental_beta()
    errors = []
    for nu in (24, 48, 96):
        beta_sq, _reference, _eu, _ev, _u_int = _matched_inhomogeneous_eigenpair(nu, max(2, nu // 2))
        errors.append(abs(math.sqrt(beta_sq) - analytic))
    # A material discontinuity landing on a node gives first-order (staircase)
    # convergence; the error must decrease monotonically and the finest grid must
    # agree with the analytic LSE root to better than half a percent.
    assert errors[0] > errors[1] > errors[2], f"non-monotonic convergence: {errors}"
    assert errors[2] / analytic < 5.0e-3, (
        f"finest-grid LSE beta error {errors[2] / analytic:.2e} exceeds 0.5%"
    )
