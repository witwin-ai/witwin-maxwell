"""Independent 2D quasi-static reference: per-unit-length capacitance of zero-thickness strips in a layered stack.

Spectral-domain Galerkin (no code shared with the layered MoM). Each strip (centre c, half-width a, plane z) carries
the charge sum_n c_n T_n(u)/sqrt(1-u^2), u = (x-c)/a: the Chebyshev edge-weighted basis that represents the
zero-thickness edge singularity exactly, so the capacitance converges exponentially in the order count N.
Spectral Green function: the stack is a ladder of layer two-ports between the interface nodes (and the strip planes):
a layer of thickness t, permittivity eps carries D = eps k [[coth kt, -csch kt], [-csch kt, coth kt]] phi; half-spaces add
eps k at the end nodes; a PEC ground removes its node. G(k) = Y(k)^-1 at the strip-plane nodes.
Kernel split: G = S + R with the static part S = 1/((eps- + eps+) sqrt(k^2 + kappa^2)) on same-plane pairs, integrated in
real space (the Chebyshev log identity for a strip with itself, Gauss-Chebyshev product rules between separated
strips; K0(kappa r) is the exact transform of S, so no gauge constant is introduced), and the remainder R, which decays
like exp(-2 k t) (t = distance to the nearest other interface), integrated in k with composite Gauss-Legendre.
Ungrounded stacks: G ~ 1/(k (eps_top + eps_bottom)) for k -> 0 on every pair; the divergent part is the rank-one total-
charge mode, removed by an exp(-k s)/k subtraction and a neutrality constraint (Lagrange multiplier = common potential).
Outputs the Maxwell capacitance matrix per unit length [F/m] of the declared conductors (rows of an ungrounded system
sum to zero). Validity: quasi-static (TEM limit), zero-thickness PEC, lossless or complex eps.

CLI: python quasistatic_sda_r7.py selftest   (CPW vs Ghione-Naldi conformal map, microstrip vs Hammerstad-Jensen)
"""
import math
import sys

import numpy as np
from scipy.special import ellipk, jv, k0

EPS0 = 8.8541878128e-12


class Stack:
    """Layers bottom to top as (thickness_m, eps_r); bottom = 'pec' or eps_r (half-space); top = eps_r (half-space)."""

    def __init__(self, layers, bottom, top):
        self.layers, self.bottom, self.top = [tuple(map(float, l)) for l in layers], bottom, float(top)
        self.interfaces = np.concatenate(([0.], np.cumsum([t for t, _ in self.layers])))

    def eps_at(self, z, side):
        """eps_r just below (side=-1) or above (side=+1) z."""
        probe = z + side * 1e-15
        if probe < 0:
            return math.inf if self.bottom == "pec" else float(self.bottom)
        if probe > self.interfaces[-1]:
            return self.top
        index = int(np.searchsorted(self.interfaces, probe) - 1)
        return self.layers[index][1]

    def nodes(self, planes):
        """Node z values (interfaces plus strip planes) and per-node layer list between them."""
        z = np.unique(np.concatenate((self.interfaces, np.asarray(planes, float))))
        segments = []
        for lo, hi in zip(z[:-1], z[1:]):
            segments.append((hi - lo, self.eps_at((lo + hi) / 2, 0)))
        return z, segments

    def green(self, k, planes):
        """G(k) [K, P, P] at the strip planes (k > 0, units 1/(eps0 k) folded in: returns eps0*k*G = Yhat^-1)."""
        z, segments = self.nodes(planes)
        n = len(z)
        k = np.asarray(k, float)
        y = np.zeros((len(k), n, n), dtype=complex)
        for i, (t, eps) in enumerate(segments):
            e = np.exp(-k * t)
            e2 = e * e
            coth = (1 + e2) / (1 - e2)
            csch = 2 * e / (1 - e2)
            y[:, i, i] += eps * coth
            y[:, i + 1, i + 1] += eps * coth
            y[:, i, i + 1] -= eps * csch
            y[:, i + 1, i] -= eps * csch
        y[:, -1, -1] += self.top
        keep = np.arange(n)
        if self.bottom == "pec":
            keep = keep[1:]
        else:
            y[:, 0, 0] += float(self.bottom)
        y = y[:, keep][:, :, keep]
        index = [int(np.flatnonzero(np.isclose(z[keep], p, rtol=0, atol=1e-15))[0]) for p in planes]
        inverse = np.linalg.inv(y)
        return inverse[:, index][:, :, index]


def _gauss_chebyshev(m):
    theta = (2 * np.arange(1, m + 1) - 1) * math.pi / (2 * m)
    return np.cos(theta), math.pi / m


def capacitance(stack, strips, conductors, orders=24, kappa_scale=1e-3, quad=96, panels_per_width=1.0,
                kmax_factor=48.0, gl=24):
    """Per-length Maxwell capacitance matrix [F/m] of ``conductors`` (list of names); strips = [(name, c, a, z)]."""
    planes = sorted({s[3] for s in strips})
    plane_of = [planes.index(s[3]) for s in strips]
    n_s, nb = len(strips), orders
    size = n_s * nb
    span = max(s[1] + s[2] for s in strips) - min(s[1] - s[2] for s in strips)
    kappa = kappa_scale / span
    grounded = stack.bottom == "pec"
    eps_s = [EPS0 * (stack.eps_at(s[3], -1) + stack.eps_at(s[3], +1)) for s in strips]
    # Static part (same-plane pairs), real space.
    p = np.zeros((size, size), dtype=complex)
    u, w = _gauss_chebyshev(quad)
    tn = np.stack([np.cos(n * np.arccos(u)) for n in range(nb)])  # [nb, quad]
    for i, (_, ci, ai, zi) in enumerate(strips):
        for j, (_, cj, aj, zj) in enumerate(strips):
            if zi != zj:
                continue
            if i == j:
                block = np.zeros((nb, nb))
                block[0, 0] = math.pi ** 2 * math.log(2)
                for n in range(1, nb):
                    block[n, n] = math.pi ** 2 / (2 * n)
                # K0(kappa a r) = -ln r + [K0(kappa a r) + ln(kappa a r)] - ln(kappa a); smooth bracket by product rule.
                r = np.abs(u[:, None] - u[None, :])
                zz = kappa * ai * np.where(r > 0, r, 1.)
                bracket = np.where(r > 0, k0(zz) + np.log(zz), math.log(2) - np.euler_gamma) - math.log(kappa * ai)
                block = block + w * w * tn @ bracket @ tn.T
                block *= ai * ai
            else:
                x = ci + ai * u
                xp = cj + aj * u
                kernel = k0(kappa * np.abs(x[:, None] - xp[None, :]))
                block = ai * aj * w * w * tn @ kernel @ tn.T
            p[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb] += block / (math.pi * eps_s[i])
    # Remainder, spectral: (1/pi) int_0^inf Re[F_i^* R_ij F_j] dk, F = pi a (-i)^n J_n(k a) exp(-i k c).
    gaps = [abs(a - b) for a in stack.interfaces for b in planes if abs(a - b) > 0] + \
           [abs(a - b) for a in planes for b in planes if a != b]
    t_min = min(gaps) if gaps else span
    kmax = kmax_factor / t_min
    a_max = max(s[2] for s in strips)
    width = math.pi / (panels_per_width * (2 * a_max + span))
    # Log-graded panels from kappa-scale to the first oscillation panel, then uniform panels to kmax.
    edges = [0.] + list(np.geomspace(width * 1e-6, width, 25)) + list(np.arange(2, int(kmax / width) + 2) * width)
    x, wx = np.polynomial.legendre.leggauss(gl)
    ks, wk = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ks.append((hi - lo) / 2 * x + (hi + lo) / 2)
        wk.append((hi - lo) / 2 * wx)
    ks, wk = np.concatenate(ks), np.concatenate(wk)
    shift = 1. / span
    for start in range(0, len(ks), 4096):
        k, wkk = ks[start:start + 4096], wk[start:start + 4096]
        g = stack.green(k, planes) / (EPS0 * k[:, None, None])  # [K, P, P]
        f = np.zeros((len(k), size), dtype=complex)
        for i, (_, ci, ai, _) in enumerate(strips):
            for n in range(nb):
                f[:, i * nb + n] = math.pi * ai * (-1j) ** n * jv(n, k * ai) * np.exp(-1j * k * ci)
        r = g[:, plane_of][:, :, plane_of]  # [K, S, S]
        same = np.array([[plane_of[i] == plane_of[j] for j in range(n_s)] for i in range(n_s)])
        static = 1. / np.sqrt(k * k + kappa * kappa)
        r = r - same[None] * static[:, None, None] / np.array(eps_s)[None, :, None]
        if not grounded:
            r = r - np.exp(-k * shift)[:, None, None] / (EPS0 * k[:, None, None] * (stack.top + float(stack.bottom)))
        rb = np.repeat(np.repeat(r, nb, axis=1), nb, axis=2)
        p += np.einsum("k,ki,kij,kj->ij", wkk, f.conj(), rb, f).real / math.pi
    # Ungrounded: the subtracted exp(-k s)/k part acts only on the total-charge mode (rank one) and drops out
    # under the neutrality constraint below.
    ub = np.zeros((size, len(conductors)))
    for i, (name, _, ai, _) in enumerate(strips):
        ub[i * nb, conductors.index(name)] = math.pi * ai
    total = ub.sum(1)
    if grounded:
        coefficients = np.linalg.solve(p, ub)
    else:
        system = np.block([[p, -total[:, None]], [total[None, :], np.zeros((1, 1))]])
        rhs = np.vstack((ub, np.zeros((1, len(conductors)))))
        coefficients = np.linalg.solve(system, rhs)[:size]
    return (ub.T @ coefficients).real


def line(stack, strips, signal, ground, **options):
    """C' [F/m], eps_eff and Z0 [ohm] of a two-conductor quasi-TEM line (signal vs ground conductor or PEC plane)."""
    names = [signal] + ([ground] if ground else [])
    c = capacitance(stack, strips, names, **options)[0, 0]
    air = Stack([(t, 1.) for t, _ in stack.layers], "pec" if stack.bottom == "pec" else 1., 1.)
    c0 = capacitance(air, strips, names, **options)[0, 0]
    light = 299792458.0
    return dict(C_F_per_m=c, C_air_F_per_m=c0, eps_eff=c / c0, Z0_ohm=1 / (light * math.sqrt(c * c0)))


def ghione_naldi(s, slot, ground, eps_r):
    """Finite-ground CPW of zero thickness on the interface of two half-spaces (eps_r below, air above): exact."""
    a, b, c = s / 2, s / 2 + slot, s / 2 + slot + ground
    k = (a / b) * math.sqrt((1 - b * b / (c * c)) / (1 - a * a / (c * c)))
    eps_eff = (eps_r + 1) / 2
    z0 = 376.730313668 / (4 * math.sqrt(eps_eff)) * ellipk(1 - k * k) / ellipk(k * k)
    return eps_eff, z0


def hammerstad_jensen(u, eps_r):
    a = 1 + math.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + .432)) / 49 + math.log(1 + (u / 18.1) ** 3) / 18.7
    b = .564 * ((eps_r - .9) / (eps_r + 3)) ** .053
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 10 / u) ** (-a * b)
    f = 6 + (2 * math.pi - 6) * math.exp(-(30.666 / u) ** .7528)
    z01 = 60 * math.log(f / u + math.sqrt(1 + 4 / u ** 2))
    return eps_eff, z01 / math.sqrt(eps_eff)


def cpw_strips(s, slot, ground, z):
    b = s / 2 + slot
    return [("sig", 0., s / 2, z), ("gnd", -(b + ground / 2), ground / 2, z), ("gnd", b + ground / 2, ground / 2, z)]


def selftest():
    um = 1e-6
    stack = Stack([(100 * um, 11.9)], 11.9, 1.)
    for orders in (8, 16, 24):
        r = line(stack, cpw_strips(10 * um, 10 * um, 40 * um, 100 * um), "sig", "gnd", orders=orders)
        print("cpw N", orders, r)
    print("ghione-naldi", ghione_naldi(10 * um, 10 * um, 40 * um, 11.9))
    h = 254 * um
    ms = Stack([(h, 9.8)], "pec", 1.)
    for orders in (8, 16, 24):
        print("microstrip N", orders, line(ms, [("sig", 0., h / 2, h)], "sig", None, orders=orders))
    print("hammerstad-jensen", hammerstad_jensen(1., 9.8))


def references(orders=32):
    """The F-D and F-E reference values (tests/support/rfic_a4_fixtures geometry) with their convergence spread."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
    from tests.support import rfic_a4_fixtures as fx
    out = {}
    stack = Stack([(fx.MIM_BELOW, fx.OX_EPS), (fx.MIM_D, fx.MIM_EPS), (fx.MIM_ABOVE, fx.OX_EPS)], fx.OX_EPS, 1.)
    area = EPS0 * fx.MIM_EPS / fx.MIM_D
    mim = {}
    # Plate widths of the cases plus half of each square side (3D edge bound of the square plates).
    for width in sorted({w for w, _ in fx.MIM_CASES} | {w / 2 for w, l in fx.MIM_CASES if w == l}):
        strips = [("b", 0., width / 2, fx.MIM_Z[0]), ("t", 0., width / 2, fx.MIM_Z[1])]
        values = [capacitance(stack, strips, ["b", "t"], orders=n)[0, 0] for n in (orders // 2, orders)]
        mim[f"{width * 1e6:g}um"] = dict(C_per_length_F_per_m=values[1], order_spread_rel=abs(values[1] / values[0] - 1),
                                        parallel_plate_F_per_m=area * width,
                                        fringe_per_edge_F_per_m=(values[1] - area * width) / 2)
    out["fd"] = dict(area_capacitance_F_per_m2=area, widths=mim)
    ms = Stack([(fx.MS_H, fx.MS_EPS)], "pec", 1.)
    out["fe_microstrip"] = line(ms, [("sig", 0., fx.MS_W / 2, fx.MS_H)], "sig", None, orders=orders)
    out["fe_microstrip"]["hammerstad_jensen"] = hammerstad_jensen(fx.MS_W / fx.MS_H, fx.MS_EPS)
    cpw_stack = Stack([(fx.CPW_Z, fx.CPW_EPS)], fx.CPW_EPS, 1.)
    out["fe_cpw"] = line(cpw_stack, cpw_strips(fx.CPW_S, fx.CPW_SLOT, fx.CPW_G, fx.CPW_Z), "sig", "gnd", orders=orders)
    out["fe_cpw"]["ghione_naldi"] = ghione_naldi(fx.CPW_S, fx.CPW_SLOT, fx.CPW_G, fx.CPW_EPS)
    # F-F lattice calibration: the trace pair (go/return) per-length L' = mu0 eps0 / C'_air (zero thickness, PEC).
    gap = fx.FF_R[0] - fx.FF_R[1] - fx.FF_W
    centre = (fx.FF_W + gap) / 2 * 1e-6
    c_air = capacitance(Stack([(1e-5, 1.)], 1., 1.), [("a", centre, fx.FF_W / 2 * 1e-6, 1e-5),
                                                     ("b", -centre, fx.FF_W / 2 * 1e-6, 1e-5)], ["a", "b"], orders=orders)[0, 0]
    out["ff_pair"] = dict(C_air_F_per_m=c_air, L_per_length_H_per_m=4e-7 * math.pi * EPS0 / c_air)
    return {k: {q: (float(v) if isinstance(v, (float, np.floating)) else v) for q, v in d.items()} for k, d in out.items()}


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["references"]:
        import json
        print(json.dumps(references(), indent=1, default=float))
