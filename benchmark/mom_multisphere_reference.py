"""Independent multi-sphere T-matrix (generalized multiparticle Mie) reference.

Validation-only host code: special functions and dense linear algebra in
NumPy/SciPy, independent of meshing, surface quadrature and CUDA kernels.
Convention exp(-i omega t), outgoing h^(1).

Each sphere j (centre c_j, radius a_j, relative permittivity eps_j) scatters
an outgoing field sum_nm s^M_nm M^(3)_nm + s^N_nm N^(3)_nm about c_j with

    M_nm = z_n(kr) X_nm,   N_nm = curl(M_nm)/k,   X_nm = L Y_nm / sqrt(n(n+1)),

and the Mie relation s^M = -b_n e^M, s^N = -a_n e^N to the regular
coefficients e of its exciting field (incident plus every other sphere's
scattered field). Regular coefficients are obtained by projecting the
tangential field on a sphere about c_j onto the orthonormal X_nm and
r x X_nm (exact Gauss-Legendre by uniform-azimuth quadrature), which gives the
outgoing-to-regular translation A of every basis function without closed-form
translation coefficients. With T = diag(-b_n, -a_n), the coupled system
s = T (e_inc + A s) is solved directly, and the far field is

    F(s) = sum_j exp(-ik s.c_j) sum_nm [s^M (-i)^(n+1) X_nm(s) + s^N (-i)^n s x X_nm(s)] / k,

so E_scattered ~ exp(ikr)/r F. A single sphere reproduces the Mie series of
``sie_reference.sphere_far_field``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn, sph_harm_y

from benchmark.sie_reference import sphere_coefficients


def _indices(order):
    return [(n, m) for n in range(1, order+1) for m in range(-n, n+1)]


def _frame(points):
    r = np.linalg.norm(points, axis=-1)
    theta = np.arccos(np.clip(points[:, 2]/r, -1, 1))
    phi = np.arctan2(points[:, 1], points[:, 0])
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    er = np.stack((st*cp, st*sp, ct), -1)
    et = np.stack((ct*cp, ct*sp, -st), -1)
    ep = np.stack((-sp, cp, np.zeros_like(sp)), -1)
    return r, theta, phi, er, et, ep


def vector_harmonics(order, points):
    """Y [P, B] and X [P, B, 3] at the directions of ``points`` (no point on the polar axis)."""
    r, theta, phi, er, et, ep = _frame(points)
    if np.any(np.sin(theta) < 1e-9):
        raise ValueError('Vector harmonics are evaluated off the polar axis')
    index = np.array(_indices(order))
    n, m = index[None, :, 0], index[None, :, 1]
    y, jacobian = sph_harm_y(n, m, theta[:, None], phi[:, None], diff_n=1)
    dtheta, dphi = jacobian[..., 0], jacobian[..., 1]
    # L Y = i theta_hat (1/sin) dY/dphi - i phi_hat dY/dtheta
    lx = 1j*(dphi/np.sin(theta)[:, None])[..., None]*et[:, None]-1j*dtheta[..., None]*ep[:, None]
    return y, lx/np.sqrt(n*(n+1))[..., None], r, er


def wave_functions(order, k, points, *, outgoing):
    """M, N [P, B, 3] about the origin; regular (j_n) or outgoing (h_n^(1))."""
    y, x, r, er = vector_harmonics(order, points)
    rho = k*r
    ns = np.array([n for n, _ in _indices(order)])
    orders = np.arange(1, order+1)[None]
    z = spherical_jn(orders, rho[:, None])
    dz = spherical_jn(orders, rho[:, None], derivative=True)
    if outgoing:
        z = z+1j*spherical_yn(orders, rho[:, None])
        dz = dz+1j*spherical_yn(orders, rho[:, None], derivative=True)
    z, dz = z[:, ns-1], (z/rho[:, None]+dz)[:, ns-1]  # (rho z)'/rho
    m_wave = z[..., None]*x
    n_wave = (1j*np.sqrt(ns*(ns+1))*z/rho[:, None]*y)[..., None]*er[:, None, :] \
        + dz[..., None]*np.cross(er[:, None, :], x)
    return m_wave, n_wave


def projection_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(2*order+2)
    azimuths = 2*np.pi*(np.arange(4*order+4)+0.5)/(4*order+4)
    ct, ph = np.meshgrid(nodes, azimuths, indexing='ij')
    st = np.sqrt(1-ct**2)
    directions = np.stack((st*np.cos(ph), st*np.sin(ph), ct), -1).reshape(-1, 3)
    return directions, np.repeat(weights, len(azimuths))*2*np.pi/len(azimuths)


class Projector:
    """Regular-wave coefficients (e^M, e^N) of a field from its tangential trace on |r| = radius."""

    def __init__(self, order, k, radius):
        self.order, self.k, self.radius = order, k, radius
        self.directions, self.weights = projection_rule(order)
        self.points = radius*self.directions
        _, x, _, er = vector_harmonics(order, self.points)
        self.x, self.rx = x, np.cross(er[:, None, :], x)
        ns = np.array([n for n, _ in _indices(order)])
        rho = k*radius
        self.jn = spherical_jn(ns, rho)
        self.djn = spherical_jn(ns, rho)/rho+spherical_jn(ns, rho, derivative=True)

    def __call__(self, field):
        """field [P, ..., 3] at ``points`` -> (e^M, e^N) [B, ...]."""
        w = self.weights[:, None, None]
        m = np.tensordot(self.x.conj()*w, field, axes=([0, 2], [0, field.ndim-1]))
        n = np.tensordot(self.rx.conj()*w, field, axes=([0, 2], [0, field.ndim-1]))
        shape = (-1,)+(1,)*(field.ndim-2)
        return m/self.jn.reshape(shape), n/self.djn.reshape(shape)


def multisphere_far_field(spheres, k, direction, polarization, directions, *, order=None):
    """Scattered far-field amplitude F [D, 3] for E_inc = polarization exp(ik direction.r).

    ``spheres`` is a sequence of (centre [3], radius, eps_r). Returns F and a
    record with the truncation order and the GMT residual.
    """
    direction = np.asarray(direction, float)/np.linalg.norm(direction)
    polarization = np.asarray(polarization, complex)
    if abs(np.dot(direction, polarization)) > 1e-12:
        raise ValueError('Plane-wave polarization must be transverse')
    radii = [float(a) for _, a, _ in spheres]
    order = order or max(int(np.ceil(k*a+4*(k*a)**(1/3)+8)) for a in radii)
    basis = len(_indices(order))
    ns = np.array([n for n, _ in _indices(order)])
    tm = []
    for _, a, eps in spheres:
        an, bn = sphere_coefficients(k*a, eps, order)
        tm.append(np.concatenate((-bn[ns-1], -an[ns-1])))
    projectors = [Projector(order, k, a) for a in radii]
    centres = [np.asarray(c, float) for c, _, _ in spheres]
    count = len(spheres)
    excitation = np.zeros(2*basis*count, complex)
    coupling = np.zeros((2*basis*count,)*2, complex)
    for j, (cj, pj) in enumerate(zip(centres, projectors)):
        rows = slice(2*basis*j, 2*basis*(j+1))
        incident = polarization[None]*np.exp(1j*k*(pj.points+cj)@direction)[:, None]
        em, en = pj(incident)
        excitation[rows] = np.concatenate((em, en))
        for l, cl in enumerate(centres):
            if l == j:
                continue
            m_wave, n_wave = wave_functions(order, k, pj.points+cj-cl, outgoing=True)
            mm, mn = pj(m_wave)
            nm, nn = pj(n_wave)
            cols_m = slice(2*basis*l, 2*basis*l+basis)
            cols_n = slice(2*basis*l+basis, 2*basis*(l+1))
            coupling[rows.start:rows.start+basis, cols_m] = mm
            coupling[rows.start+basis:rows.stop, cols_m] = mn
            coupling[rows.start:rows.start+basis, cols_n] = nm
            coupling[rows.start+basis:rows.stop, cols_n] = nn
    t = np.concatenate(tm)
    system = np.eye(len(t))-t[:, None]*coupling
    scattered = np.linalg.solve(system, t*excitation)
    residual = np.linalg.norm(system@scattered-t*excitation)/np.linalg.norm(t*excitation)
    directions = np.asarray(directions, float)
    _, x, _, er = vector_harmonics(order, directions)
    rx = np.cross(er[:, None, :], x)
    phase_m, phase_n = (-1j)**(ns+1), (-1j)**ns
    field = np.zeros((len(directions), 3), complex)
    for j, cj in enumerate(centres):
        sm, sn = scattered[2*basis*j:2*basis*j+basis], scattered[2*basis*j+basis:2*basis*(j+1)]
        local = np.einsum('b,dbc->dc', sm*phase_m, x)+np.einsum('b,dbc->dc', sn*phase_n, rx)
        field += np.exp(-1j*k*directions@cj)[:, None]*local/k
    return field, {'order': order, 'unknowns': len(t), 'relative_residual': float(residual)}
