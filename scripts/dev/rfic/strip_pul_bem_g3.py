"""Independent 2D magnetostatic PEC BEM: per-unit-length inductance of two coplanar rectangular bars (go/return), side by side along the width.

Unknown: piecewise-constant longitudinal surface current K on perimeter panels (graded geometrically to every
corner). A_z(r) = -(mu0/2pi) sum_p K_p int_p ln|r-r'| dl'. PEC: A_z equal on each bar (A1 on bar 1, A2 on bar 2),
total current +1 / -1. L' = A1 - A2. Collocation at panel midpoints; the panel integral of ln is closed form
(exact for collinear self and neighbour panels, Gauss-Legendre 16 elsewhere with analytic near treatment by
subdivision). Units: metres. Convergence: rerun with finer grading.
"""
import argparse, math
import numpy as np

MU0 = 4e-7 * math.pi


def edge_points(a, b, n, ratio):
    """Points on [0, L] graded geometrically toward both ends (n panels per half)."""
    L = math.dist(a, b)
    if ratio == 1:
        s = np.linspace(0, 1, 2 * n + 1)
    else:
        g = ratio ** np.arange(n)
        h = np.concatenate(([0.], np.cumsum(g)))
        h = h / h[-1] * .5
        s = np.concatenate((h, 1 - h[-2::-1]))
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a[None] + s[:, None] * (b - a)[None]


def rectangle(cx, cy, w, t, n, ratio):
    corners = [(cx - w / 2, cy - t / 2), (cx + w / 2, cy - t / 2), (cx + w / 2, cy + t / 2), (cx - w / 2, cy + t / 2)]
    panels = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        nn = max(2, int(round(n * math.dist(a, b) / max(w, t)))) if args.scaled else n
        pts = edge_points(a, b, nn, ratio)
        panels += [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return panels


def log_integral(p, a, b):
    """Exact int_a^b ln|p - r(s)| ds over the straight panel a-b (s arc length)."""
    d = b - a
    L = np.linalg.norm(d, axis=-1)
    u = d / L[..., None]
    q = p - a
    s0 = (q * u).sum(-1)                       # projection
    h = np.abs(q[..., 0] * u[..., 1] - q[..., 1] * u[..., 0])
    def F(s):  # int ln sqrt((s - s0)^2 + h^2) ds
        x = s - s0
        r2 = x * x + h * h
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(r2 > 0, .5 * np.log(np.where(r2 > 0, r2, 1.)), 0.)
            at = np.where(h > 0, h * np.arctan2(x, np.where(h > 0, h, 1.)), 0.)
        return x * lg - x + at
    return F(L) - F(np.zeros_like(L))


p = argparse.ArgumentParser()
p.add_argument("--w", type=float, default=10.)
p.add_argument("--t", type=float, default=2.)
p.add_argument("--sep", type=float, default=110.)
p.add_argument("--n", type=int, default=20)
p.add_argument("--ratio", type=float, default=1.2)
p.add_argument("--scaled", action="store_true")
args = p.parse_args()
um = 1e-6
bars = [rectangle(0., 0., args.w * um, args.t * um, args.n, args.ratio),
        rectangle(args.sep * um, 0., args.w * um, args.t * um, args.n, args.ratio)]
panels = [(a, b, k) for k, bar in enumerate(bars) for a, b in bar]
A = np.array([pp[0] for pp in panels]); B = np.array([pp[1] for pp in panels]); owner = np.array([pp[2] for pp in panels])
mid = (A + B) / 2
length = np.linalg.norm(B - A, axis=1)
N = len(panels)
# M[i, j] = -(mu0/2pi) int_j ln|mid_i - r'| dl'
M = -(MU0 / (2 * math.pi)) * log_integral(mid[:, None, :], A[None], B[None])
# Unknowns K (N), A1, A2: M K - A_owner = 0; sum_bar1 K l = 1; sum_bar2 K l = -1.
S = np.zeros((N + 2, N + 2))
S[:N, :N] = M
S[np.arange(N), N + owner] = -1.
S[N, :N] = length * (owner == 0)
S[N + 1, :N] = length * (owner == 1)
rhs = np.zeros(N + 2); rhs[N] = 1.; rhs[N + 1] = -1.
x = np.linalg.solve(S, rhs)
Lp = x[N] - x[N + 1]
print(f"w={args.w} t={args.t} sep={args.sep} panels={N} ratio={args.ratio}  L' = {Lp*1e6:.7f} uH/m = {Lp*1e9*1e-3:.7f} nH/um*1e3")
