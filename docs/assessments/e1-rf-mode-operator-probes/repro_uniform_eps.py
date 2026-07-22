"""Reproduce the uniform dielectric-fill routing evidence (selector path).

Runs the production selector entry ``_solve_yee_transverse_vector_mode`` (NOT the
raw operator builder) on a hollow rectangular guide uniformly filled with several
``eps_r`` values and prints, for each fill:

  * the solved ``beta`` and ``beta**2``,
  * the exact discrete analytic ``beta**2 = eps_r*k0**2 - k~x(1)**2 - k~y(0)**2``
    for the TE10 branch (``k~ = (2/dx) sin(m pi dx / (2 a))``),
  * the vacuum ``beta**2`` the operator would return if the routing dropped the real
    eps (the ``operator_eps = (None, None, None)`` defect),
  * the TE10 ``Ev`` (here ``Ey``) full-grid ``sin(pi u / a)`` correlation.

The selector-path unit test
``tests/rf/wave_validation/test_te10_mode_selection.py::
test_uniform_dielectric_fill_selector_path_carries_filled_beta`` pins the eps_r=2.25
row of this probe.

Run (from the worktree root, PYTHONPATH set to it):
    conda run -n maxwell --no-capture-output python \
        docs/assessments/e1-rf-mode-operator-probes/repro_uniform_eps.py
"""

from __future__ import annotations

import math

import numpy as np

from witwin.maxwell.fdtd.excitation.modes import (
    _solve_yee_transverse_vector_mode,
    _yee_transverse_discrete_transverse_wavenumber as ktilde,
)

# Small float64 hollow-guide cross-section: a along u (larger), b along v.
A, B = 1.0, 0.6
NU, NV = 24, 12
DU, DV = A / NU, B / NV
K0 = 10.0
FIELD_NAMES = ("Ex", "Ey", "Hx", "Hy")  # u = x, v = y


def _te10_sin_correlation(ey_node: np.ndarray) -> float:
    u_node = np.arange(NU + 1) * DU
    reference = np.outer(np.sin(math.pi * u_node / A), np.ones(NV + 1))
    flat = np.asarray(ey_node).real.reshape(-1)
    ref = reference.reshape(-1)
    if np.dot(flat, ref) < 0.0:
        flat = -flat
    denom = np.linalg.norm(flat) * np.linalg.norm(ref)
    return float(np.dot(flat, ref) / denom) if denom > 0 else 0.0


def main() -> None:
    ktx = ktilde(1, A, DU)
    kty = ktilde(0, B, DV)
    vacuum_beta_sq = K0 * K0 - ktx * ktx - kty * kty
    print(f"# hollow guide a={A} b={B} nu={NU} nv={NV} k0={K0}")
    print(f"# TE10 discrete k~x(1)={ktx:.6f} k~y(0)={kty:.6f}")
    print(f"# vacuum (defect) beta**2 = {vacuum_beta_sq:.6f}  beta = {math.sqrt(vacuum_beta_sq):.6f}")
    print(f"# {'eps_r':>6} {'beta':>10} {'beta**2':>12} {'analytic':>12} {'rel_err':>10} {'sin_corr':>9}")
    for eps_r in (1.0, 2.25, 4.0):
        node = np.full((NU + 1, NV + 1), float(eps_r), dtype=np.float64)
        beta, components, _diag = _solve_yee_transverse_vector_mode(
            (node, node, node),
            k0=K0,
            du=DU,
            dv=DV,
            mode_index=0,
            field_names=FIELD_NAMES,
            preferred_field_name="Ey",
            wave_family="te",
            uniform=True,
            use_dense=True,
        )
        beta_sq = float(beta) ** 2
        analytic = eps_r * K0 * K0 - ktx * ktx - kty * kty
        rel = abs(beta_sq - analytic) / abs(analytic)
        corr = _te10_sin_correlation(components["Ey"])
        print(f"  {eps_r:6.3f} {beta:10.6f} {beta_sq:12.6f} {analytic:12.6f} {rel:10.2e} {corr:9.6f}")


if __name__ == "__main__":
    main()
