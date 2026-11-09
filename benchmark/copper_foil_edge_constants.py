"""Regenerate the discrete Yee half-plane constants of the CopperFoil edge model.

A zero-thickness conducting half-line (x <= 0, y = 0) on a 5-point grid with
tangential step 1 and normal step r is solved with the continuum far field
phi = Re sqrt(z) on a square box of N nodes per side. The rim node's two-face
current c0 and the native two-face edge loss offset g,

    sum_{m=0..M} (q_top,m^2 + q_bottom,m^2) = (ln(M + 1/2) + g) / 2,

are the constants tabulated in ``witwin.maxwell.compiler.copper_foil``. The
split of the rim's lateral flux between the faces follows the foil update
(half of each adjacent normal cell). Richardson extrapolation in 1/N uses
N and 2N with M = N/4. numpy/scipy only.

    python -m benchmark.copper_foil_edge_constants 0.125 0.25 0.5 1 2 4 8
"""
from __future__ import annotations

import math
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def half_plane_charges(n, aspect):
    """Per-face rim-first charges (q_top, q_bottom) of the discrete half-plane."""
    size = 2 * n + 1
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    exact = np.sqrt((i + 1j * j * aspect).astype(complex)).real
    conductor = (j == 0) & (i <= 0)
    exact[conductor] = 0.0
    free = ~(conductor | (np.abs(i) == n) | (np.abs(j) == n))
    index = -np.ones((size, size), int)
    index[free] = np.arange(free.sum())
    fi, fj = np.nonzero(free)
    k = index[fi, fj]
    wy, wz = 1.0, 1.0 / aspect ** 2
    rows, cols, vals = [k], [k], [np.full(len(k), 2 * wy + 2 * wz)]
    rhs = np.zeros(free.sum())
    for di, dj, weight in ((1, 0, wy), (-1, 0, wy), (0, 1, wz), (0, -1, wz)):
        ni, nj = fi + di, fj + dj
        neighbor = free[ni, nj]
        rows.append(k[neighbor]); cols.append(index[ni[neighbor], nj[neighbor]])
        vals.append(np.full(neighbor.sum(), -weight))
        np.add.at(rhs, k[~neighbor], weight * exact[ni[~neighbor], nj[~neighbor]])
    matrix = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))))
    phi = exact.copy()
    phi[free] = spl.spsolve(matrix.tocsc(), rhs)
    m = np.arange(n)
    top = phi[n - m, n + 1] / aspect
    bottom = phi[n - m, n - 1] / aspect
    top[0] += phi[n + 1, n] * aspect / 2
    bottom[0] += phi[n + 1, n] * aspect / 2
    return top, bottom


def edge_constants(aspect, n=400):
    """Richardson (N/2, N) estimate of (c0, g)."""
    values = []
    for size in (n // 2, n):
        top, bottom = half_plane_charges(size, aspect)
        loss = np.cumsum(top ** 2 + bottom ** 2)
        last = size // 4
        values.append((top[0] + bottom[0], 2 * loss[last] - math.log(last + .5)))
    return tuple(2 * b - a for a, b in zip(*values))


if __name__ == "__main__":
    for value in map(float, sys.argv[1:] or ("1",)):
        c0, g = edge_constants(value)
        print(f"aspect={value:.8g} c0={c0:.5f} g={g:.5f}")
