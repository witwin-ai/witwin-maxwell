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
from scipy import linalg as scipy_linalg
from scipy.sparse.csgraph import connected_components

from witwin.maxwell.fdtd.excitation.modes import (
    _build_yee_transverse_operator_sparse,
    _split_yee_transverse_eigenvector,
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
