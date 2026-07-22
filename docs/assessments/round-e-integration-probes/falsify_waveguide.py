"""Executed falsification for the waveguide wave-level gate.

1. Mode corruption: patch the runner's mode-shape correlation to emulate the legacy
   checkerboard operator (sub-0.9) -> the runner must return status='blocked'.
2. Analytic-reference corruption: correlate the SECOND transverse eigenmode (TE20,
   sin(2 pi y/a)) against the TE10 sin(pi y/a) reference on the REAL operator ->
   the guard's correlation collapses, proving a mode-index slip is caught.
3. Restore: the unpatched runner is a genuine PASS.
"""
import math
import numpy as np

import benchmark.rf_validation as rv

# --- (1) emulate the legacy operator regression via the fail-closed guard ---
orig = rv._waveguide_te10_sin_correlation
rv._waveguide_te10_sin_correlation = lambda *a, **k: 0.55  # checkerboard-class value
report_red = rv.run_rectangular_waveguide()
print("FALSIFY(1) forced sin-corr=0.55 -> status:", report_red.status,
      "| class:", report_red.gate_class)
assert report_red.status == "blocked", "guard did not fire!"
rv._waveguide_te10_sin_correlation = orig  # restore

# --- (2) real operator: TE20 profile vs the TE10 reference ---
from witwin.maxwell.fdtd.excitation.modes import (
    _build_yee_transverse_operator_sparse,
    _split_yee_transverse_eigenvector,
)
import scipy.linalg as sla

a, b, nu, nv = 0.6, 0.3, 24, 12
op, meta = _build_yee_transverse_operator_sparse(
    nu_cells=nu, nv_cells=nv, du=a / nu, dv=b / nv, k0=10.0
)
vals, vecs = sla.eigh(op.toarray())
order = np.argsort(vals)[::-1]
ref_te10 = np.outer(np.sin(math.pi * meta["ev_u"] / meta["extent_u"]),
                    np.ones(len(meta["ev_v"])))


def corr(vec):
    _eu, ev = _split_yee_transverse_eigenvector(vec, meta)
    f, r = ev.reshape(-1), ref_te10.reshape(-1)
    if np.dot(f, r) < 0:
        f = -f
    return abs(np.dot(f, r) / (np.linalg.norm(f) * np.linalg.norm(r)))


c_te10 = corr(vecs[:, order[0]])   # fundamental
c_te20 = corr(vecs[:, order[1]])   # next mode (TE20-class)
print(f"FALSIFY(2) TE10 corr={c_te10:.4f} (clean), mode-1 corr={c_te20:.4f} (< 0.9 guard)")
assert c_te10 >= 0.99 and c_te20 < 0.9

# --- (3) restored runner is a genuine PASS ---
report_green = rv.run_rectangular_waveguide()
print("FALSIFY(3) restored -> status:", report_green.status)
assert report_green.status == "pass"
print("ALL FALSIFICATIONS EXECUTED OK")
