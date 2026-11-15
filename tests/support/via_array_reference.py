"""Independent 2D magnetostatic PEC-limit BEM for parallel via arrays between two PEC planes (R7-1 / B2 reference).

Vertical vias that join two parallel PEC planes carry z-invariant currents: the plane images make the
quasi-static field exactly two-dimensional, so the via array is a 2D multiconductor problem per unit height.
Every via of one net touches both planes, so with perfectly conducting planes the loop between any two vias of a
net links no flux: all conductors of a net share one vector potential A_n (the PEC limit, omega L >> R).

Unknowns: piecewise-constant longitudinal surface current K on straight panels of each via outline (graded
geometrically toward every vertex), plus A_+ and A_-. A_z(r) = -(mu0/2pi) sum_p K_p int_p ln|r-r'| dl', with the
exact closed-form panel integral of ln (any observer, any straight panel). Collocation at panel midpoints:
A_z = A_net on each panel, and net currents are +1 and -1.

Per unit height: L' = A_+ - A_- (external, PEC limit); R' = Rs sum_p K_p^2 l_p for that current. With omega,
the Leontovich wall impedance Zs = Rs (1 - i) (exp(-i omega t)) enters the wall condition, which is the model
the layered MoM via walls use; at Rs/(omega L') ~ 0.1 it relaxes proximity crowding and R' drops by ~20-25%
from the PEC-limit value (2x2 array: 197.4 -> 152.9 ohm/m at 1 GHz, copper). The return-net current split gives the per-via currents.
Units: metres. Convergence: refine the per-edge panel count (``panels``) and the grading ratio.
"""

import math

import numpy as np

MU0 = 4e-7 * math.pi


def _edge_points(a, b, n, ratio):
    """2n+1 points on segment a-b, graded geometrically toward both ends by ``ratio``."""
    g = ratio ** np.arange(n)
    h = np.concatenate(([0.], np.cumsum(g)))
    h = h / h[-1] * .5
    s = np.concatenate((h, 1 - h[-2::-1]))
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a[None] + s[:, None] * (b - a)[None]


def outline_panels(outline, panels, ratio):
    """Straight panels (P, 2, 2) of a closed outline, ``panels`` per half edge, graded toward each vertex."""
    loop = np.asarray(outline, float)
    out = []
    for k in range(len(loop)):
        points = _edge_points(loop[k], loop[(k + 1) % len(loop)], panels, ratio)
        out.append(np.stack((points[:-1], points[1:]), 1))
    return np.concatenate(out)


def log_integral(p, a, b):
    """Exact int_a^b ln|p - r(s)| ds over straight panels a-b for observers p (broadcast)."""
    d = b - a
    length = np.linalg.norm(d, axis=-1)
    u = d / length[..., None]
    q = p - a
    s0 = (q * u).sum(-1)
    h = np.abs(q[..., 0] * u[..., 1] - q[..., 1] * u[..., 0])

    def antiderivative(s):
        x = s - s0
        r2 = x * x + h * h
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(r2 > 0, .5 * np.log(np.where(r2 > 0, r2, 1.)), 0.)
            at = np.where(h > 0, h * np.arctan2(x, np.where(h > 0, h, 1.)), 0.)
        return x * lg - x + at
    return antiderivative(length) - antiderivative(np.zeros_like(length))


def solve(outlines, nets, *, panels=24, ratio=1.15, surface_resistance=1.0, omega=None):
    """Per-unit-height solve for via outlines (each ((x, y), ...)) with net signs +1/-1.

    omega=None: PEC limit (A_z = A_net on every panel). Returns L_per_m (external) and
    R_per_m = Rs sum K^2 l for that current. With omega: the Leontovich surface impedance
    Zs = Rs (1 - i) (exp(-i omega t)) enters the wall condition E_z = Zs K, i.e.
    A_z - Zs K / (i omega) = c_net, and Z' = -i omega (c_+ - c_-) = R' - i omega L' is the exact
    2D impedance of that model (L' includes the internal Rs/omega part).
    Returns dict(L_per_m, R_per_m, via_currents (signed, per outline), panels).
    """
    blocks = [outline_panels(o, panels, ratio) for o in outlines]
    owner = np.concatenate([np.full(len(b), i) for i, b in enumerate(blocks)])
    allp = np.concatenate(blocks)
    a, b = allp[:, 0], allp[:, 1]
    lengths = np.linalg.norm(b - a, axis=-1)
    mid = .5 * (a + b)
    count = len(allp)
    dtype = float if omega is None else complex
    matrix = np.zeros((count + 2, count + 2), dtype=dtype)
    matrix[:count, :count] = -MU0 / (2 * math.pi) * log_integral(mid[:, None], a[None], b[None])
    if omega is not None:
        matrix[np.arange(count), np.arange(count)] -= surface_resistance * (1 - 1j) / (1j * omega)
    sign = np.asarray(nets)[owner]
    matrix[:count, count] = -(sign > 0).astype(float)
    matrix[:count, count + 1] = -(sign < 0).astype(float)
    matrix[count, :count] = lengths * (sign > 0)
    matrix[count + 1, :count] = lengths * (sign < 0)
    rhs = np.zeros(count + 2, dtype=dtype)
    rhs[count], rhs[count + 1] = 1., -1.
    x = np.linalg.solve(matrix, rhs)
    k = x[:count]
    currents = np.array([np.sum(k[owner == i] * lengths[owner == i]) for i in range(len(outlines))])
    if omega is None:
        return dict(L_per_m=float(x[count] - x[count + 1]), R_per_m=float(surface_resistance * np.sum(k * k * lengths)),
                    via_currents=currents, panels=count)
    impedance = -1j * omega * (x[count] - x[count + 1])
    return dict(L_per_m=float(-impedance.imag / omega), R_per_m=float(impedance.real), via_currents=currents.real,
                panels=count)


def converged(outlines, nets, *, surface_resistance=1.0, omega=None, levels=(12, 24, 48), ratio=1.15):
    """Solve on a panel-refinement sequence; report the finest values and the last relative changes."""
    rows = [solve(outlines, nets, panels=n, ratio=ratio, surface_resistance=surface_resistance, omega=omega)
            for n in levels]
    last, previous = rows[-1], rows[-2]
    return dict(L_per_m=last["L_per_m"], R_per_m=last["R_per_m"], via_currents=last["via_currents"],
                L_change=abs(last["L_per_m"] / previous["L_per_m"] - 1),
                R_change=abs(last["R_per_m"] / previous["R_per_m"] - 1), rows=rows)


def regular_outline(centre, circumradius, sides, phase=0.0):
    return tuple((centre[0] + circumradius * math.cos(phase + 2 * math.pi * k / sides),
                  centre[1] + circumradius * math.sin(phase + 2 * math.pi * k / sides)) for k in range(sides))


def closed_form_checks(sides=96, omega=None, surface_resistance=1.0):
    """Relative errors against closed forms: coaxial L' = mu0/2pi ln(b/a) (uniform K, R' = Rs/2pi (1/a + 1/b)),
    and the two-wire line L' = mu0/pi acosh(x), R' = Rs/(pi a) x/sqrt(x^2-1), x = D/2a (proximity effect).
    Polygons with ``sides`` sides stand in for circles (perimeter/area-radius corrections are O(sides^-2))."""
    a, b, distance = 1.0, 3.0, 2.6
    coax = solve([regular_outline((0, 0), a, sides), regular_outline((0, 0), b, sides)], [1, -1], panels=4, ratio=1.,
                 omega=omega, surface_resistance=surface_resistance)
    internal = 0.0 if omega is None else surface_resistance * (1 / a + 1 / b) / (2 * math.pi) / omega
    x = distance / (2 * a)
    pair = solve([regular_outline((-distance / 2, 0), a, sides), regular_outline((distance / 2, 0), a, sides)],
                 [1, -1], panels=4, ratio=1.)  # the proximity factor is the PEC-limit (omega L >> R) closed form
    return dict(coax_L=coax["L_per_m"] / (MU0 / (2 * math.pi) * math.log(b / a) + internal) - 1,
                coax_R=coax["R_per_m"] / (surface_resistance * (1 / a + 1 / b) / (2 * math.pi)) - 1,
                pair_L=pair["L_per_m"] / (MU0 / math.pi * math.acosh(x)) - 1,
                pair_R=pair["R_per_m"] / (2 / (math.pi * a) * x / math.sqrt(x * x - 1) / 2) - 1)


if __name__ == "__main__":
    for sides in (48, 96, 192):
        print(sides, closed_form_checks(sides), closed_form_checks(sides, omega=2 * math.pi * 1e9, surface_resistance=8e-3))
