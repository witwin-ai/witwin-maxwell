"""Verify the inhomogeneous transverse operator's spectrum is real and bounded.

The half-filled parallel-plate cross-section (``eps = EPS1`` for ``u < D``, ``EPS2``
otherwise, uniform in ``v``) produces a real-but-non-symmetric transverse operator.
This probe regenerates, at three grids:

  * the true spectral maximum ``max Re(beta**2)`` and ``max|Im(beta**2)|`` of the
    unmodified operator (dense ``numpy.linalg.eigvals``) -- the physical LSE mode IS
    the spectral maximum and the spectrum is entirely real, so no spurious
    high-|beta| filter is required;
  * for comparison, ``max`` eigenvalue of the naively SYMMETRIZED operator
    ``0.5 (P + P^T)`` (``numpy.linalg.eigvalsh``). This symmetrization is an artifact:
    its maximum grows with ``1/dx`` and does NOT represent the true spectrum. It is
    recorded here only to document that the growing-maximum number quoted in an
    earlier draft of the acceptance doc came from this symmetrized surrogate, not the
    real operator.

Run (from the worktree root, PYTHONPATH set to it):
    conda run -n maxwell --no-capture-output python \
        docs/assessments/e1-rf-mode-operator-probes/verify_spurious_spectrum.py
"""

from __future__ import annotations

import numpy as np

from witwin.maxwell.fdtd.excitation.modes import _build_yee_transverse_operator_sparse

HF_A, HF_B = 1.0, 0.6
HF_K0 = 8.0
HF_EPS1, HF_EPS2 = 4.0, 1.0
HF_D = 0.5  # dielectric interface position along u


def _eps_of_u(u: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(u) < HF_D, HF_EPS1, HF_EPS2)


def _half_filled_operator(nu: int, nv: int):
    du = HF_A / nu
    dv = HF_B / nv
    u_half = (np.arange(nu) + 0.5) * du
    u_int = np.arange(1, nu) * du
    eps_uu = np.repeat(_eps_of_u(u_half)[:, None], nv - 1, axis=1)
    eps_vv = np.repeat(_eps_of_u(u_int)[:, None], nv, axis=1)
    eps_ww = np.repeat(_eps_of_u(u_int)[:, None], nv - 1, axis=1)
    operator, _meta = _build_yee_transverse_operator_sparse(
        nu_cells=nu, nv_cells=nv, du=du, dv=dv, k0=HF_K0,
        eps_uu=eps_uu, eps_vv=eps_vv, eps_ww=eps_ww,
    )
    return operator


def main() -> None:
    print(f"# half-filled guide a={HF_A} b={HF_B} k0={HF_K0} eps1={HF_EPS1} eps2={HF_EPS2} D={HF_D}")
    print(f"# {'nu':>4} {'true_max_Re':>12} {'true_max_Im':>12} {'symmetrized_max':>16}")
    for nu in (24, 48, 96):
        nv = max(2, nu // 2)
        dense = _half_filled_operator(nu, nv).toarray()
        eigenvalues = np.linalg.eigvals(dense)
        true_max_re = float(np.max(eigenvalues.real))
        true_max_im = float(np.max(np.abs(eigenvalues.imag)))
        symmetrized_max = float(np.max(np.linalg.eigvalsh(0.5 * (dense + dense.T))))
        print(f"  {nu:4d} {true_max_re:12.4f} {true_max_im:12.2e} {symmetrized_max:16.4f}")


if __name__ == "__main__":
    main()
