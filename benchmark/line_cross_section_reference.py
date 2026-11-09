"""Independent 2D cross-section reference for printed transmission lines.

Validation-only CPU implementation in numpy/scipy. It imports nothing from the
FDTD runtime: the conductor/dielectric geometry is declared here from SI
dimensions and solved on its own graded tensor mesh.

Two solvers share one mesh:

* ``QuasiStatic``: finite-volume Laplace solve (node potentials, cell
  permittivity). It returns the capacitance matrices with/without dielectric,
  quasi-TEM modes and, from the air-filled charge distribution, the surface
  current density used by the conductor-loss perturbation
  ``alpha_c = Rs * sum|Js|^2 dl / (2 Re V.I*)``.
* ``FullWave``: Yee finite-difference transverse-E eigenproblem
  ``(w^2 mu eps - curl curl + grad eps_l^-1 div eps) E_t = beta^2 E_t`` with
  complex permittivity, so dielectric loss and dispersion enter exactly.

Conductors are ideal (Dirichlet / tangential-E zero) rectangles; a rectangle
with ``z0 == z1`` is a zero-thickness sheet. Internally the time convention is
exp(+j w t), exp(-j beta x); public impedances are conjugated to the
repository's exp(-i omega t) convention (negative imaginary part when lossy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ETA0 = MU0 * C0

# Printed-line fixture dimensions (declared independently; a test pins them
# against benchmark/scenes/rf/printed_lines.py so drift fails loudly).
WIDTH = 0.010
HEIGHT = 0.005
GAP = 0.005
PAIR_WIDTH = 0.005
PAIR_OFFSET = (PAIR_WIDTH + GAP) / 2
COPPER_THICKNESS = 35e-6
COPPER_CONDUCTIVITY = 5.8e7


@dataclass(frozen=True)
class Rect:
    """Ideal conductor rectangle; ``z0 == z1`` (or ``y0 == y1``) is a sheet."""
    y0: float
    y1: float
    z0: float
    z1: float
    net: int
    lossy: bool = True
    name: str = ""


@dataclass(frozen=True)
class Slab:
    """Full-width dielectric layer between z0 and z1."""
    z0: float
    z1: float
    eps_r: float
    sigma: float = 0.0


@dataclass(frozen=True)
class CrossSection:
    box: tuple[float, float, float, float]
    rects: tuple[Rect, ...]
    slabs: tuple[Slab, ...] = ()
    background: tuple[float, float] = (1.0, 0.0)
    walls: tuple[str, str, str, str] = ("pec", "pec", "pec", "pec")  # y-, y+, z-, z+
    extra_y: tuple[float, ...] = ()
    extra_z: tuple[float, ...] = ()

    @property
    def nets(self):
        return sorted({r.net for r in self.rects if r.net > 0})


@dataclass(frozen=True)
class MeshSpec:
    """Graded spacing h(d) = min(h_max, h_min + growth*d), d = distance to the
    nearest conductor edge coordinate on that axis. ``refined(L)`` halves every
    length scale and the growth L times (uniform refinement in the mapped
    coordinate), so successive levels support Richardson extrapolation."""
    h_min: float
    h_max: float
    growth: float

    def refined(self, level):
        s = 2.0 ** level
        return MeshSpec(self.h_min / s, self.h_max / s, self.growth / s)


def graded_axis(lo, hi, breaks, singular, spec):
    points = np.unique(np.r_[lo, hi, [b for b in breaks if lo < b < hi]])
    singular = np.asarray(sorted(set(singular)), dtype=np.float64)
    nodes = [lo]
    for a, b in zip(points[:-1], points[1:]):
        sample = [np.linspace(a, b, 2001)]
        for s in singular:
            offsets = np.geomspace(spec.h_min / 20, max(b - a, spec.h_min), 600)
            sample.append(np.clip(np.r_[s - offsets, s + offsets], a, b))
        x = np.unique(np.concatenate(sample))
        if singular.size:
            d = np.min(np.abs(x[:, None] - singular[None, :]), axis=1)
            h = np.minimum(spec.h_max, spec.h_min + spec.growth * d)
        else:
            h = np.full_like(x, spec.h_max)
        inv = 1.0 / h
        cumulative = np.r_[0.0, np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(x))]
        n = max(1, int(math.ceil(cumulative[-1] - 1e-9)))
        target = np.linspace(0.0, cumulative[-1], n + 1)[1:-1]
        nodes.extend(np.interp(target, cumulative, x))
        nodes.append(b)
    return np.asarray(nodes, dtype=np.float64)


class Mesh:
    def __init__(self, section: CrossSection, spec: MeshSpec | None = None, *, nodes=None):
        """Graded mesh from ``spec``, or explicit ``nodes=(y, z)`` arrays (e.g. a
        replica of another solver's transverse grid). Every conductor, slab and
        path coordinate must then already be a node."""
        self.section, self.spec = section, spec
        ymin, ymax, zmin, zmax = section.box
        ybreak = [v for r in section.rects for v in (r.y0, r.y1)] + list(section.extra_y)
        zbreak = [v for r in section.rects for v in (r.z0, r.z1)] + [v for s in section.slabs for v in (s.z0, s.z1)] + list(section.extra_z)
        tol = 1e-12
        ysing = [v for r in section.rects for v in (r.y0, r.y1) if ymin + tol < v < ymax - tol]
        zsing = [v for r in section.rects for v in (r.z0, r.z1) if zmin + tol < v < zmax - tol]
        if (spec is None) == (nodes is None):
            raise ValueError("Give exactly one of spec or nodes.")
        if spec is not None:
            self.y = graded_axis(ymin, ymax, ybreak, ysing, spec)
            self.z = graded_axis(zmin, zmax, zbreak, zsing, spec)
        else:
            self.y, self.z = (np.asarray(a, dtype=np.float64) for a in nodes)
            for axis, lo, hi, breaks in ((self.y, ymin, ymax, ybreak), (self.z, zmin, zmax, zbreak)):
                if abs(axis[0] - lo) > 1e-12 or abs(axis[-1] - hi) > 1e-12 or np.any(np.diff(axis) <= 0):
                    raise ValueError("Explicit nodes must be increasing and span the box.")
                for b in breaks:
                    if lo < b < hi and np.min(np.abs(axis - b)) > 1e-12:
                        raise ValueError(f"Explicit nodes miss the geometric coordinate {b}.")
        self.ny, self.nz = len(self.y), len(self.z)
        self.dy, self.dz = np.diff(self.y), np.diff(self.z)
        yc, zc = 0.5 * (self.y[1:] + self.y[:-1]), 0.5 * (self.z[1:] + self.z[:-1])
        # Cell material (relative permittivity, conductivity).
        eps = np.full((self.ny - 1, self.nz - 1), float(section.background[0]))
        sig = np.full_like(eps, float(section.background[1]))
        for s in section.slabs:
            inside = (zc > min(s.z0, s.z1)) & (zc < max(s.z0, s.z1))
            eps[:, inside] = s.eps_r
            sig[:, inside] = s.sigma
        self.cell_eps_r, self.cell_sigma = eps, sig
        self.cell_metal = np.zeros_like(eps, dtype=bool)
        Y, Z = np.meshgrid(self.y, self.z, indexing="ij")
        self.net = np.full((self.ny, self.nz), -1, dtype=np.int64)
        self.lossy_node = np.zeros((self.ny, self.nz), dtype=bool)
        for r in section.rects:
            inside = (Y >= r.y0 - tol) & (Y <= r.y1 + tol) & (Z >= r.z0 - tol) & (Z <= r.z1 + tol)
            if np.any((self.net[inside] >= 0) & (self.net[inside] != r.net)):
                raise ValueError("Distinct conductor nets touch.")
            self.net[inside] = r.net
            if r.lossy:
                self.lossy_node |= inside
            cin = (yc[:, None] > r.y0) & (yc[:, None] < r.y1) & (zc[None, :] > r.z0) & (zc[None, :] < r.z1)
            self.cell_metal |= cin
        for side, kind in zip(("y-", "y+", "z-", "z+"), section.walls):
            if kind not in ("pec", "pmc"):
                raise ValueError("Walls are 'pec' or 'pmc'.")
            if kind == "pec":
                sl = {"y-": (0, slice(None)), "y+": (-1, slice(None)), "z-": (slice(None), 0), "z+": (slice(None), -1)}[side]
                wall = self.net[sl]
                if np.any((wall > 0)):
                    raise ValueError("A signal conductor touches a PEC wall.")
                wall[wall < 0] = 0
                self.net[sl] = wall
        if not np.any(self.net == 0):
            raise ValueError("A reference (net 0) conductor or PEC wall is required.")

    @property
    def nodes(self):
        return self.ny * self.nz

    def complex_eps(self, omega):
        """Relative complex cell permittivity, exp(+jwt): eps_r - j sigma/(w eps0)."""
        if omega is None:
            return self.cell_eps_r.astype(np.float64)
        return self.cell_eps_r - 1j * self.cell_sigma / (omega * EPS0)


def nested_dissection(ny, nz, leaf=64):
    """Fill-reducing elimination order of the ny x nz tensor node grid."""
    out = []

    def recurse(i0, i1, j0, j1):
        ni, nj = i1 - i0, j1 - j0
        if ni <= 0 or nj <= 0:
            return
        if ni * nj <= leaf or (ni < 3 and nj < 3):
            out.append((np.arange(i0, i1)[:, None] * nz + np.arange(j0, j1)[None, :]).ravel())
            return
        if ni >= nj:
            s = (i0 + i1) // 2
            recurse(i0, s, j0, j1)
            recurse(s + 1, i1, j0, j1)
            out.append(s * nz + np.arange(j0, j1))
        else:
            s = (j0 + j1) // 2
            recurse(i0, i1, j0, s)
            recurse(i0, i1, s + 1, j1)
            out.append(np.arange(i0, i1) * nz + s)

    recurse(0, ny, 0, nz)
    return np.concatenate(out)


def _factor(matrix, order):
    """Sparse LU in a precomputed nested-dissection order."""
    permuted = matrix[order][:, order].tocsc()
    lu = spla.splu(permuted, permc_spec="NATURAL", diag_pivot_thresh=0.0,
                   options=dict(SymmetricMode=True))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    def solve(rhs):
        return lu.solve(rhs[order])[inverse]
    return solve


def surface_groups(mesh, edge_fields, *, return_nodes=False):
    """Integrate an edge vector field over each conductor surface piece.

    ``edge_fields`` [edges, k] holds values on y-edges then z-edges (Ey/Ez
    layout). For every lossy conductor node the outward flux through its dual
    cell is split into the four quadrant cells; quadrants connected without
    crossing metal form one surface piece (so the two faces of a sheet stay
    separate while its edge wraps around). With eps0*E this is surface charge;
    with hhat = (Hz, -Hy) it is the surface current (Ampere circulation).
    Returns (values [pieces, k], conductor-surface length of each piece).
    """
    m = mesh
    ney = (m.ny - 1) * m.nz
    k = edge_fields.shape[1]
    fy = edge_fields[:ney].reshape(m.ny - 1, m.nz, k)
    fz = edge_fields[ney:].reshape(m.ny, m.nz - 1, k)
    net, free_cell = m.net, ~m.cell_metal
    values, lengths, owners = [], [], []
    for i, j in np.argwhere(m.lossy_node):
        # Quadrant cells LL, LR, UR, UL; half-edge e (down, right, up, left)
        # separates quadrants e and (e+1)%4.
        quad = [(i - 1, j - 1), (i, j - 1), (i, j), (i - 1, j)]
        qfree = [0 <= a < m.ny - 1 and 0 <= b < m.nz - 1 and free_cell[a, b] for a, b in quad]
        if not any(qfree):
            continue
        neighbors = [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)]
        exists = [0 <= a < m.ny and 0 <= b < m.nz for a, b in neighbors]
        metal = [e and net[a, b] >= 0 for e, (a, b) in zip(exists, neighbors)]
        lossy_edge = [e and m.lossy_node[a, b] for e, (a, b) in zip(exists, neighbors)]
        half = [m.dz[j - 1] / 2 if j > 0 else 0.0, m.dy[i] / 2 if i < m.ny - 1 else 0.0,
                m.dz[j] / 2 if j < m.nz - 1 else 0.0, m.dy[i - 1] / 2 if i > 0 else 0.0]
        parts = np.zeros((4, k), dtype=edge_fields.dtype)
        if exists[1] and not metal[1]:
            f = fy[i, j]
            if qfree[1]:
                parts[1] += f * m.dz[j - 1] / 2
            if qfree[2]:
                parts[2] += f * m.dz[j] / 2
        if exists[3] and not metal[3]:
            f = -fy[i - 1, j]
            if qfree[0]:
                parts[0] += f * m.dz[j - 1] / 2
            if qfree[3]:
                parts[3] += f * m.dz[j] / 2
        if exists[2] and not metal[2]:
            f = fz[i, j]
            if qfree[3]:
                parts[3] += f * m.dy[i - 1] / 2
            if qfree[2]:
                parts[2] += f * m.dy[i] / 2
        if exists[0] and not metal[0]:
            f = -fz[i, j - 1]
            if qfree[0]:
                parts[0] += f * m.dy[i - 1] / 2
            if qfree[1]:
                parts[1] += f * m.dy[i] / 2
        group = [-1] * 4
        count = 0
        for q in range(4):
            if not qfree[q] or group[q] >= 0:
                continue
            group[q] = count
            stack = [q]
            while stack:
                a = stack.pop()
                for b, between in (((a + 1) % 4, a), ((a - 1) % 4, (a - 1) % 4)):
                    if qfree[b] and group[b] < 0 and not metal[between]:
                        group[b] = count
                        stack.append(b)
            count += 1
        glen = np.zeros(count)
        gv = np.zeros((count, k), dtype=edge_fields.dtype)
        for q in range(4):
            if group[q] >= 0:
                gv[group[q]] += parts[q]
        for e in range(4):
            if metal[e] and lossy_edge[e]:
                for q in (e, (e + 1) % 4):
                    if qfree[q]:
                        glen[group[q]] += half[e]
        for g in range(count):
            if glen[g] > 0:
                values.append(gv[g])
                lengths.append(glen[g])
                owners.append((i, j))
    if return_nodes:
        return np.asarray(values), np.asarray(lengths), np.asarray(owners)
    return np.asarray(values), np.asarray(lengths)


class QuasiStatic:
    """Finite-volume Laplace solve; charges are exact discrete Gauss fluxes."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        m = mesh
        self.free = m.net.ravel() < 0
        self.nets = sorted({int(v) for v in np.unique(m.net) if v > 0})
        order = nested_dissection(m.ny, m.nz)
        rank = np.empty(m.nodes, dtype=np.int64)
        rank[order] = np.arange(m.nodes)
        free_index = np.flatnonzero(self.free)
        self._free_order = np.argsort(rank[free_index], kind="stable")
        self._cache = {}

    def edge_weights(self, eps_cell):
        m = self.mesh
        lower = np.zeros((m.ny - 1, m.nz), dtype=eps_cell.dtype)
        upper = np.zeros_like(lower)
        lower[:, 1:] = eps_cell * m.dz[None, :] / 2
        upper[:, :-1] = eps_cell * m.dz[None, :] / 2
        wy_lower, wy_upper = lower / m.dy[:, None], upper / m.dy[:, None]
        left = np.zeros((m.ny, m.nz - 1), dtype=eps_cell.dtype)
        right = np.zeros_like(left)
        left[1:, :] = eps_cell * m.dy[:, None] / 2
        right[:-1, :] = eps_cell * m.dy[:, None] / 2
        wz_left, wz_right = left / m.dz[None, :], right / m.dz[None, :]
        return wy_lower, wy_upper, wz_left, wz_right

    def laplacian(self, eps_cell):
        m = self.mesh
        wyl, wyu, wzl, wzr = self.edge_weights(eps_cell)
        wy, wz = (wyl + wyu).ravel(), (wzl + wzr).ravel()
        idx = np.arange(m.nodes).reshape(m.ny, m.nz)
        a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
        c, d = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        rows = np.r_[a, b, a, b, c, d, c, d]
        cols = np.r_[a, b, b, a, c, d, d, c]
        vals = np.r_[wy, wy, -wy, -wy, wz, wz, -wz, -wz]
        return sp.csr_matrix((vals, (rows, cols)), shape=(m.nodes, m.nodes))

    def unit_potentials(self, eps_cell, key):
        """Potential solutions for 1 V on each signal net (others grounded)."""
        if key in self._cache:
            return self._cache[key]
        m = self.mesh
        A = self.laplacian(eps_cell)
        free = self.free
        Aff = A[free][:, free]
        Afd = A[free][:, ~free]
        solve = _factor(Aff, self._free_order)
        net = m.net.ravel()
        solutions = []
        for k in self.nets:
            phi = np.zeros(m.nodes, dtype=np.result_type(eps_cell.dtype, np.float64))
            phi[net == k] = 1.0
            phi[free] = solve(-(Afd @ phi[~free]))
            solutions.append(phi)
        flux = [A @ phi for phi in solutions]
        C = np.array([[EPS0 * np.sum(f[net == l]) for f in flux] for l in self.nets])
        ground = np.array([EPS0 * np.sum(f[net == 0]) for f in flux])
        result = dict(A=A, phi=solutions, C=C, ground_charge=ground)
        self._cache[key] = result
        return result

    def capacitance(self, omega=None, *, air=False):
        m = self.mesh
        if air:
            eps = np.ones_like(m.cell_eps_r)
            return self.unit_potentials(eps, "air")["C"]
        eps = m.complex_eps(omega)
        key = ("diel", None if omega is None else float(omega))
        return self.unit_potentials(eps, key)["C"]

    def surface_groups(self):
        """Air-filled surface charge groups [groups, nets] and their lengths."""
        if "groups" in self._cache:
            return self._cache["groups"]
        m = self.mesh
        sol = self.unit_potentials(np.ones_like(m.cell_eps_r), "air")
        fields = []
        for phi in sol["phi"]:
            phi = phi.reshape(m.ny, m.nz)
            ey = (phi[:-1, :] - phi[1:, :]) / m.dy[:, None]
            ez = (phi[:, :-1] - phi[:, 1:]) / m.dz[None, :]
            fields.append(EPS0 * np.r_[ey.ravel(), ez.ravel()])
        result = surface_groups(m, np.stack(fields, axis=1))
        self._cache["groups"] = result
        return result

    def modes(self, frequency, surface_resistance=None):
        """Quasi-TEM modes: gamma^2 eigenvalues of (jwL)(jwC_c), L = mu0 eps0 C0^-1.

        Conductor loss is a first-order perturbation from the air-filled surface
        current density Js = c0 * rho_s for the modal current vector.
        """
        omega = 2 * math.pi * frequency
        C0m = self.capacitance(air=True)
        Cc = self.capacitance(omega)
        L = MU0 * EPS0 * np.linalg.inv(C0m)
        Zs, Ys = 1j * omega * L, 1j * omega * Cc
        lam, V = np.linalg.eig(Zs @ Ys)
        gamma = np.sqrt(lam)
        gamma = np.where(gamma.real < 0, -gamma, gamma)
        out = []
        for n in np.argsort(gamma.imag):
            g = gamma[n]
            v = V[:, n]
            i = Ys @ v / g  # I = (Y/gamma) V for a forward wave
            power = 0.5 * np.real(np.vdot(i, v))
            if power < 0:
                v, i, power = -v, -i, -power
            record = dict(alpha_d_np_per_m=float(g.real), beta_rad_per_m=float(g.imag),
                          voltage=v, current=i, power_W=power,
                          eps_eff=float((g.imag / (omega / C0)) ** 2))
            if surface_resistance is not None:
                record["alpha_c_np_per_m"] = self.conductor_alpha(i, power, surface_resistance)
            out.append(record)
        return out

    def conductor_alpha(self, current, power, surface_resistance):
        charges, lengths = self.surface_groups()
        C0m = self.capacitance(air=True)
        # air-problem charge q = I/c0 per net -> unit-potential coefficients
        coeff = np.linalg.solve(C0m, np.asarray(current) / C0)
        q = charges @ coeff
        js2 = (C0 ** 2) * np.abs(q) ** 2 / lengths
        loss = 0.5 * surface_resistance * np.sum(js2)
        return float(loss / (2 * power))


class FullWave:
    """Yee transverse-E eigen solver on the shared tensor mesh.

    Unknowns: Ey on y-edges, Ez on z-edges. Ex lives on nodes (Dirichlet zero on
    conductors), Hx on cells. hhat = (Hz, -Hy) is colocated with (Ey, Ez).
    """

    def __init__(self, mesh: Mesh, frequency):
        self.mesh, self.frequency = mesh, float(frequency)
        m = mesh
        self.omega = omega = 2 * math.pi * self.frequency
        ny, nz = m.ny, m.nz
        ney, nez = (ny - 1) * nz, ny * (nz - 1)
        self.ney, self.nez = ney, nez
        nedge, ncell = ney + nez, (ny - 1) * (nz - 1)
        node = np.arange(m.nodes).reshape(ny, nz)
        # G: nodes -> edges gradient
        rows_y = np.arange(ney)
        a, b = node[:-1, :].ravel(), node[1:, :].ravel()
        invdy = np.repeat(1 / m.dy, nz)
        rows_z = ney + np.arange(nez)
        c, d = node[:, :-1].ravel(), node[:, 1:].ravel()
        invdz = np.tile(1 / m.dz, ny)
        G = sp.csr_matrix((np.r_[-invdy, invdy, -invdz, invdz],
                           (np.r_[rows_y, rows_y, rows_z, rows_z], np.r_[a, b, c, d])),
                          shape=(nedge, m.nodes))
        # C: edges -> cells curl (dEz/dy - dEy/dz)
        cell = np.arange(ncell).reshape(ny - 1, nz - 1)
        yedge = np.arange(ney).reshape(ny - 1, nz)
        zedge = ney + np.arange(nez).reshape(ny, nz - 1)
        cdy = np.repeat(1 / m.dy, nz - 1)
        cdz = np.tile(1 / m.dz, ny - 1)
        cr = cell.ravel()
        Cm = sp.csr_matrix((np.r_[cdy, -cdy, -cdz, cdz],
                            (np.r_[cr, cr, cr, cr],
                             np.r_[zedge[1:, :].ravel(), zedge[:-1, :].ravel(),
                                   yedge[:, 1:].ravel(), yedge[:, :-1].ravel()])),
                           shape=(ncell, nedge))
        dyd = np.r_[m.dy[0] / 2, (m.dy[1:] + m.dy[:-1]) / 2, m.dy[-1] / 2]
        dzd = np.r_[m.dz[0] / 2, (m.dz[1:] + m.dz[:-1]) / 2, m.dz[-1] / 2]
        Me = np.r_[(m.dy[:, None] * dzd[None, :]).ravel(), (dyd[:, None] * m.dz[None, :]).ravel()]
        Mn = (dyd[:, None] * dzd[None, :]).ravel()
        Mc = (m.dy[:, None] * m.dz[None, :]).ravel()
        self.Me, self.Mn, self.Mc = Me, Mn, Mc
        eps = m.complex_eps(omega) * EPS0
        eps = np.where(m.cell_metal, EPS0, eps)
        # Edge permittivity: dual-length weighted average of the adjacent cells.
        ey = np.zeros((ny - 1, nz), dtype=complex)
        ey[:, 1:] += eps * m.dz[None, :] / 2
        ey[:, :-1] += eps * m.dz[None, :] / 2
        ey /= dzd[None, :]
        ez = np.zeros((ny, nz - 1), dtype=complex)
        ez[1:, :] += eps * m.dy[:, None] / 2
        ez[:-1, :] += eps * m.dy[:, None] / 2
        ez /= dyd[:, None]
        eps_edge = np.r_[ey.ravel(), ez.ravel()]
        # Longitudinal (node) permittivity: area-weighted quadrant average.
        area = m.dy[:, None] * m.dz[None, :] / 4
        num = np.zeros((ny, nz), dtype=complex)
        den = np.zeros((ny, nz))
        for si, sj in ((slice(1, None), slice(1, None)), (slice(None, -1), slice(1, None)),
                       (slice(1, None), slice(None, -1)), (slice(None, -1), slice(None, -1))):
            num[si, sj] += eps * area
            den[si, sj] += area
        eps_node = (num / den).ravel()
        pec_node = m.net.ravel() >= 0
        Lam = np.where(pec_node, 0.0, 1.0 / eps_node)
        D = -sp.diags(1 / Mn) @ G.T @ sp.diags(Me)
        R = sp.diags(1 / Me) @ Cm.T @ sp.diags(Mc)
        self.G, self.Cm, self.D, self.R = G, Cm, D, R
        self.eps_edge, self.Lam = eps_edge, Lam
        operator = (omega ** 2 * MU0) * sp.diags(eps_edge) - R @ Cm + G @ sp.diags(Lam) @ D @ sp.diags(eps_edge)
        # Tangential E vanishes on edges whose two nodes are conductor nodes.
        netf = m.net.ravel()
        tail = np.r_[a, c]
        head = np.r_[b, d]
        pec_edge = (netf[tail] >= 0) & (netf[head] >= 0)
        self.free = ~pec_edge
        self.operator = operator.tocsr()[self.free][:, self.free].tocsc()
        # Edge ordering keyed by the edge's tail node for nested dissection.
        order = nested_dissection(ny, nz)
        rank = np.empty(m.nodes, dtype=np.int64)
        rank[order] = np.arange(m.nodes)
        key = 2 * rank[tail] + np.r_[np.zeros(ney, int), np.ones(nez, int)]
        self._edge_order = np.argsort(key[self.free], kind="stable")

    def solve(self, beta_guess, count=4):
        A = self.operator
        n = A.shape[0]
        shift = complex(beta_guess) ** 2
        solve = _factor((A - shift * sp.identity(n, format="csc")).tocsc(), self._edge_order)
        op = spla.LinearOperator((n, n), matvec=solve, dtype=complex)
        values, vectors = spla.eigs(op, k=count, which="LM", tol=1e-12)
        lam = shift + 1 / values
        modes = []
        for value, vector in zip(lam, vectors.T):
            residual = np.linalg.norm(A @ vector - value * vector) / np.linalg.norm(value * vector)
            kx = np.sqrt(value)
            if kx.real < 0:
                kx = -kx
            modes.append(self._mode(kx, vector, residual))
        modes.sort(key=lambda r: -r["beta_rad_per_m"])
        return modes

    def _mode(self, kx, vector, residual):
        e = np.zeros(self.ney + self.nez, dtype=complex)
        e[self.free] = vector
        omega = self.omega
        # j kx hhat = j w eps e + R C e / (j w mu0)
        hhat = (1j * omega * self.eps_edge * e + self.R @ (self.Cm @ e) / (1j * omega * MU0)) / (1j * kx)
        power = 0.5 * np.real(np.sum(self.Me * e * np.conj(hhat)))
        if power < 0:
            hhat, power = -hhat, -power
        scale = 1 / math.sqrt(power)
        e, hhat = e * scale, hhat * scale
        ex = self.Lam * (self.D @ hhat) / (1j * omega)
        return dict(beta_rad_per_m=float(kx.real), alpha_np_per_m=float(-kx.imag), kx=kx,
                    eps_eff=float((kx.real / (omega / C0)) ** 2), relative_residual=float(residual),
                    e=e, hhat=hhat, ex=ex, power_W=0.5 * float(np.real(np.sum(self.Me * e * np.conj(hhat)))))

    def conductor_alpha(self, mode, surface_resistance):
        """First-order surface-impedance loss from the full-wave surface current."""
        values, lengths = surface_groups(self.mesh, mode["hhat"][:, None])
        loss = 0.5 * surface_resistance * np.sum(np.abs(values[:, 0]) ** 2 / lengths)
        return float(loss / (2 * mode["power_W"]))

    # ---- field access -------------------------------------------------
    def fields(self, mode):
        m = self.mesh
        e, h = mode["e"], mode["hhat"]
        Ey = e[:self.ney].reshape(m.ny - 1, m.nz)
        Ez = e[self.ney:].reshape(m.ny, m.nz - 1)
        Hz = h[:self.ney].reshape(m.ny - 1, m.nz)
        Hy = -h[self.ney:].reshape(m.ny, m.nz - 1)
        return Ey, Ez, Hy, Hz

    def conductor_current(self, mode, net):
        """Discrete Ampere: I = sum over the net's node dual cells of (curl H)_x."""
        m = self.mesh
        curl = self.D @ mode["hhat"]
        mask = (m.net.ravel() == net)
        return complex(np.sum(self.Mn[mask] * curl[mask]))

    def line_voltage(self, mode, start, stop):
        """V = phi(start)-phi(stop) = integral of E.dl from start to stop along a
        mesh line (both endpoints on mesh nodes)."""
        m = self.mesh
        Ey, Ez, _, _ = self.fields(mode)
        (y0, z0), (y1, z1) = start, stop
        if abs(z0 - z1) < 1e-15:
            j = _node_index(m.z, z0)
            i0, i1 = _node_index(m.y, y0), _node_index(m.y, y1)
            lo, hi = sorted((i0, i1))
            value = np.sum(Ey[lo:hi, j] * m.dy[lo:hi])
            return complex(value if i1 > i0 else -value)
        if abs(y0 - y1) < 1e-15:
            i = _node_index(m.y, y0)
            j0, j1 = _node_index(m.z, z0), _node_index(m.z, z1)
            lo, hi = sorted((j0, j1))
            value = np.sum(Ez[i, lo:hi] * m.dz[lo:hi])
            return complex(value if j1 > j0 else -value)
        raise ValueError("Voltage paths must follow a mesh line.")

    def sample(self, mode, component, y, z):
        """Bilinear interpolation of a Yee transverse component at (y, z)."""
        m = self.mesh
        Ey, Ez, Hy, Hz = self.fields(mode)
        yh, zh = 0.5 * (m.y[1:] + m.y[:-1]), 0.5 * (m.z[1:] + m.z[:-1])
        grids = {"Ey": (yh, m.z, Ey), "Hz": (yh, m.z, Hz), "Ez": (m.y, zh, Ez), "Hy": (m.y, zh, Hy)}
        gy, gz, values = grids[component]
        return complex(_bilinear(gy, gz, values, y, z))


def _node_index(axis, value):
    i = int(np.argmin(np.abs(axis - value)))
    if abs(axis[i] - value) > 1e-12:
        raise ValueError(f"{value} is not a mesh node.")
    return i


def _bilinear(gy, gz, values, y, z):
    i = int(np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2))
    j = int(np.clip(np.searchsorted(gz, z) - 1, 0, len(gz) - 2))
    ty = (y - gy[i]) / (gy[i + 1] - gy[i])
    tz = (z - gz[j]) / (gz[j + 1] - gz[j])
    return ((1 - ty) * (1 - tz) * values[i, j] + ty * (1 - tz) * values[i + 1, j]
            + (1 - ty) * tz * values[i, j + 1] + ty * tz * values[i + 1, j + 1])


# ---- material helpers ----------------------------------------------------
def homogeneous_gamma(frequency, eps_r, sigma):
    """Exact TEM propagation constant in a homogeneous lossy dielectric."""
    omega = 2 * math.pi * np.asarray(frequency, dtype=np.float64)
    gamma = np.sqrt(1j * omega * MU0 * (sigma + 1j * omega * EPS0 * eps_r))
    return gamma.real, gamma.imag


def foil_surface_impedance(frequency, thickness=COPPER_THICKNESS, conductivity=COPPER_CONDUCTIVITY):
    """Per-face self impedance of a two-sided foil (exp(+jwt)): Zc*coth(k t),
    Zc = (1+j)/(sigma delta). Thick limit -> (1+j) Rs."""
    omega = 2 * math.pi * frequency
    delta = math.sqrt(2 / (omega * MU0 * conductivity))
    k = (1 + 1j) / delta
    zc = (1 + 1j) / (conductivity * delta)
    self_term = zc / np.tanh(k * thickness)
    mutual = zc / np.sinh(k * thickness)
    return dict(skin_depth_m=delta, thickness_over_skin_depth=thickness / delta,
                surface_resistance_ohm=1 / (conductivity * delta), self_impedance_ohm=complex(self_term),
                mutual_impedance_ohm=complex(mutual))


# ---- printed-line cross sections -----------------------------------------
def printed_line_section(kind, *, thickness=0.0, half_y=0.14, half_z=0.13, walls="pec",
                         eps_r=2.25, sigma=0.0):
    """Cross sections matching benchmark/scenes/rf/printed_lines.py.

    ``thickness`` > 0 grows each foil upward from its declared plane (bottom
    face on the FDTD sheet coordinate; the pair ground grows downward into the
    shielded region). The default box is the FDTD physical domain (its PML lies
    outside). ``walls`` = 'pec'|'pmc' sets the two side walls and top wall;
    CPW/stripline bottom walls follow the same kind.
    """
    t = float(thickness)
    w = (walls,) * 4
    if kind == "cpw":
        rects = (Rect(-WIDTH / 2, WIDTH / 2, 0.0, t, 1, name="signal"),
                 Rect(-half_y, -WIDTH / 2 - GAP, 0.0, t, 0, name="ground_left"),
                 Rect(WIDTH / 2 + GAP, half_y, 0.0, t, 0, name="ground_right"))
        if walls == "pmc":
            # Grounds end on PMC walls; join them through the far bottom wall.
            w = ("pmc", "pmc", "pec", "pec")
        return CrossSection(box=(-half_y, half_y, -half_z, half_z), rects=rects,
                            background=(eps_r, sigma), walls=w, extra_y=(0.0,))
    if kind == "stripline":
        rects = (Rect(-WIDTH / 2, WIDTH / 2, -t / 2, t / 2, 1, name="signal"),
                 Rect(-half_y, half_y, -HEIGHT - t, -HEIGHT, 0, name="ground"),
                 Rect(-half_y, half_y, HEIGHT, HEIGHT + t, 0, name="top_ground"))
        box = (-half_y, half_y, -HEIGHT - t, HEIGHT + t)
        return CrossSection(box=box, rects=rects, background=(eps_r, sigma),
                            walls=(walls, walls, "pec", "pec"), extra_y=(0.0,))
    if kind == "differential_pair":
        rects = (Rect(PAIR_OFFSET - PAIR_WIDTH / 2, PAIR_OFFSET + PAIR_WIDTH / 2, 0.0, t, 1, name="positive"),
                 Rect(-PAIR_OFFSET - PAIR_WIDTH / 2, -PAIR_OFFSET + PAIR_WIDTH / 2, 0.0, t, 2, name="negative"),
                 Rect(-half_y, half_y, -HEIGHT - t, -HEIGHT, 0, name="ground"))
        box = (-half_y, half_y, -HEIGHT - t, half_z)
        return CrossSection(box=box, rects=rects, slabs=(Slab(-HEIGHT, 0.0, eps_r, sigma),),
                            walls=(walls, walls, "pec", walls), extra_y=(0.0, PAIR_OFFSET, -PAIR_OFFSET))
    raise ValueError(kind)


# ---- closed-form sanity layer (not references) ---------------------------
def cpw_conformal(eps_r=2.25, width=WIDTH, gap=GAP):
    from scipy.special import ellipk
    k = width / (width + 2 * gap)
    ratio = float(ellipk(1 - k * k) / ellipk(k * k))
    return dict(k=k, z0_ohm=ETA0 / (4 * math.sqrt(eps_r)) * ratio, eps_eff=eps_r)


def cpw_conductor_loss_owyang_wu(frequency, thickness=COPPER_THICKNESS, eps_r=2.25,
                                 width=WIDTH, gap=GAP, conductivity=COPPER_CONDUCTIVITY):
    """Closed-form CPW conductor attenuation (Owyang-Wu / Gupta), Np/m."""
    from scipy.special import ellipk
    a, b = width / 2, width / 2 + gap
    k = a / b
    rs = foil_surface_impedance(frequency, thickness, conductivity)["surface_resistance_ohm"]
    K, Kp = float(ellipk(k * k)), float(ellipk(1 - k * k))
    pre = rs * math.sqrt(eps_r) / (480 * math.pi * K * Kp * (1 - k * k))
    term = lambda s: (math.pi + math.log(8 * math.pi * s * (1 - k) / (thickness * (1 + k)))) / s
    return pre * (term(a) + term(b))


def stripline_cohn(eps_r=2.25, width=WIDTH, spacing=2 * HEIGHT):
    from scipy.special import ellipk
    kp = math.tanh(math.pi * width / (2 * spacing))
    ratio = float(ellipk(1 - kp * kp) / ellipk(kp * kp))
    return ETA0 / (4 * math.sqrt(eps_r)) * ratio


def stripline_conductor_loss_wheeler(frequency, thickness=COPPER_THICKNESS, eps_r=2.25,
                                     width=WIDTH, spacing=2 * HEIGHT, conductivity=COPPER_CONDUCTIVITY):
    """Wheeler incremental-inductance stripline conductor loss (textbook form,
    valid for sqrt(eps_r) Z0 < 120 ohm), Np/m. Spacing b is ground-to-ground."""
    rs = foil_surface_impedance(frequency, thickness, conductivity)["surface_resistance_ohm"]
    b, t, w = spacing + thickness, thickness, width
    z0 = stripline_cohn(eps_r, width, spacing)
    A = 1 + 2 * w / (b - t) + (b + t) / (math.pi * (b - t)) * math.log((2 * b - t) / t)
    return 2.7e-3 * rs * eps_r * z0 / (30 * math.pi * (b - t)) * A
