"""Analytic references for the FDTD subpixel-averaging convergence study.

Validation-only numpy/scipy code; nothing here is used by the solver.

* Sphere Mie series (dielectric and PEC): coefficients a_n, b_n, efficiencies
  and amplitude functions S1/S2 in the Bohren-Huffman convention.
* Infinite-cylinder series at normal incidence for both polarizations (TM:
  E along the axis; TE: H along the axis).
* Two-layer parallel-plate TM0/TE1 dispersion relations, solved as a
  continuum problem in the layered (z) direction with an optional
  semi-discrete Yee substitution for the uniform (x, t) directions.

Time convention exp(-i omega t), outgoing Hankel functions of the first kind.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import h1vp, hankel1, jv, jvp, spherical_jn, spherical_yn

C0 = 299792458.0


# --------------------------------------------------------------------------
# Sphere
# --------------------------------------------------------------------------
def _sphere_order(x, m=1.0):
    return int(math.ceil(abs(x) + 4.05 * abs(x) ** (1.0 / 3.0) + 2.0 + abs(m) * 0 + 10))


def _riccati(n, z):
    """psi_n(z) = z j_n(z) and its derivative, for real or complex z."""
    j = spherical_jn(n, z)
    dj = spherical_jn(n, z, derivative=True)
    return z * j, j + z * dj


def _riccati_xi(n, x):
    """xi_n(x) = x h1_n(x) and its derivative (real x)."""
    j, y = spherical_jn(n, x), spherical_yn(n, x)
    dj, dy = spherical_jn(n, x, derivative=True), spherical_yn(n, x, derivative=True)
    h, dh = j + 1j * y, dj + 1j * dy
    return x * h, h + x * dh


def mie_sphere_coefficients(x, m=None, *, pec=False, nmax=None):
    """Bohren-Huffman a_n, b_n (n = 1..nmax) for size parameter x = k a.

    ``m`` is the relative refractive index (real for the lossless media used
    here; spherical_jn of complex arguments is not needed). ``pec=True`` gives
    the perfectly conducting limit a_n = psi_n'/xi_n', b_n = psi_n/xi_n.
    """
    if nmax is None:
        nmax = _sphere_order(x, 1.0 if m is None else m)
    n = np.arange(1, nmax + 1)
    psi_x, dpsi_x = _riccati(n, x)
    xi_x, dxi_x = _riccati_xi(n, x)
    if pec:
        return n, dpsi_x / dxi_x, psi_x / xi_x
    m = float(m)
    psi_mx, dpsi_mx = _riccati(n, m * x)
    a = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / (m * psi_mx * dxi_x - xi_x * dpsi_mx)
    b = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / (psi_mx * dxi_x - m * xi_x * dpsi_mx)
    return n, a, b


def mie_sphere_efficiencies(x, m=None, *, pec=False):
    n, a, b = mie_sphere_coefficients(x, m, pec=pec)
    weight = 2 * n + 1
    q_sca = 2.0 / x**2 * float(np.sum(weight * (np.abs(a) ** 2 + np.abs(b) ** 2)))
    q_ext = 2.0 / x**2 * float(np.sum(weight * np.real(a + b)))
    back = np.sum(weight * (-1.0) ** n * (a - b))
    q_back = float(np.abs(back) ** 2 / x**2)
    return {"q_sca": q_sca, "q_ext": q_ext, "q_back": q_back}


def _pi_tau(nmax, mu):
    mu = np.asarray(mu, dtype=float)
    pi = np.zeros((nmax + 1,) + mu.shape)
    tau = np.zeros_like(pi)
    pi[1] = 1.0
    for n in range(1, nmax + 1):
        if n >= 2:
            pi[n] = (2 * n - 1) / (n - 1) * mu * pi[n - 1] - n / (n - 1) * pi[n - 2]
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi[1:], tau[1:]


def mie_sphere_amplitudes(x, theta, m=None, *, pec=False):
    """Amplitude functions S1(theta), S2(theta) (Bohren-Huffman 4.74)."""
    n, a, b = mie_sphere_coefficients(x, m, pec=pec)
    theta = np.asarray(theta, dtype=float)
    pi, tau = _pi_tau(len(n), np.cos(theta))
    e = ((2 * n + 1) / (n * (n + 1))).reshape((-1,) + (1,) * theta.ndim)
    s1 = np.sum(e * (a.reshape(e.shape) * pi + b.reshape(e.shape) * tau), axis=0)
    s2 = np.sum(e * (a.reshape(e.shape) * tau + b.reshape(e.shape) * pi), axis=0)
    return s1, s2


def mie_sphere_bistatic_rcs(radius, wavelength, theta, phi, m=None, *, pec=False):
    """Bistatic cross section sigma(theta, phi) = 4 pi |E_s|^2 r^2 / |E_0|^2.

    Incidence along +z with x polarization; phi is measured from +x, so
    phi = 0 is the E-plane (|S2|^2) and phi = 90 deg the H-plane (|S1|^2).
    """
    k = 2 * math.pi / wavelength
    s1, s2 = mie_sphere_amplitudes(k * radius, theta, m, pec=pec)
    phi = np.asarray(phi, dtype=float)
    return 4 * math.pi / k**2 * (np.abs(s2) ** 2 * np.cos(phi) ** 2 + np.abs(s1) ** 2 * np.sin(phi) ** 2)


# --------------------------------------------------------------------------
# Infinite cylinder, normal incidence
# --------------------------------------------------------------------------
def _cylinder_order(x, m):
    return int(math.ceil(abs(m * x) + 4.05 * abs(m * x) ** (1.0 / 3.0) + 12))


def mie_cylinder_coefficients(x, m=None, *, polarization, pec=False):
    """Coefficients c_n, n = 0..nmax (scattered field sum_n c_n i^n H_n e^{i n phi}
    up to the BH sign convention). ``polarization`` is ``"TM"`` (E parallel to
    the axis, BH case I, b_n) or ``"TE"`` (H parallel to the axis, BH case II,
    a_n)."""
    nmax = _cylinder_order(x, 1.0 if m is None else m)
    n = np.arange(0, nmax + 1)
    jx, djx = jv(n, x), jvp(n, x)
    hx, dhx = hankel1(n, x), h1vp(n, x)
    pol = polarization.upper()
    if pec:
        if pol == "TM":
            return n, jx / hx
        if pol == "TE":
            return n, djx / dhx
        raise ValueError(polarization)
    m = float(m)
    jmx, djmx = jv(n, m * x), jvp(n, m * x)
    if pol == "TM":
        return n, (jmx * djx - m * djmx * jx) / (jmx * dhx - m * djmx * hx)
    if pol == "TE":
        return n, (m * djx * jmx - jx * djmx) / (m * dhx * jmx - hx * djmx)
    raise ValueError(polarization)


def mie_cylinder_efficiencies(x, m=None, *, polarization, pec=False):
    """Per-unit-length efficiencies Q = C / (2 a)."""
    n, c = mie_cylinder_coefficients(x, m, polarization=polarization, pec=pec)
    weight = np.where(n == 0, 1.0, 2.0)
    q_sca = 2.0 / x * float(np.sum(weight * np.abs(c) ** 2))
    q_ext = 2.0 / x * float(np.sum(weight * np.real(c)))
    return {"q_sca": q_sca, "q_ext": q_ext}


# --------------------------------------------------------------------------
# Two-layer parallel plate: dielectric in 0 < z < d, air in d < z < h
# --------------------------------------------------------------------------
def _quasi_tem_residual(omega, kx, eps1, d, h, normal):
    k0 = omega / C0
    kz1 = math.sqrt(eps1 * k0 * k0 - kx * kx)
    a2 = math.sqrt(kx * kx - k0 * k0)
    weight = 1.0 / eps1 if normal else 1.0
    return weight * kz1 * math.tan(kz1 * d) - a2 * math.tanh(a2 * (h - d))


def _quasi_tem_omega(kx, eps1, d, h, normal):
    lo = C0 * kx / math.sqrt(eps1) * (1 + 1e-13)
    hi = C0 * kx * (1 - 1e-13)
    return brentq(_quasi_tem_residual, lo, hi, args=(kx, eps1, d, h, normal), xtol=1e-6, rtol=1e-15, maxiter=500)


def parallel_plate_tm0_omega(kx, eps1, d, h):
    """TM0 between PEC plates (E normal to the interface, Ez/Ex with Hy).

    Hy ~ cos(kz1 z) in the dielectric and cosh(a2 (h - z)) in the air; the
    interface condition is (kz1/eps1) tan(kz1 d) = a2 tanh(a2 (h - d)).
    Bound below the air light line: k0 < kx < sqrt(eps1) k0.
    """
    return _quasi_tem_omega(kx, eps1, d, h, normal=True)


def parallel_plate_te0_omega(kx, eps1, d, h):
    """TE0 between PMC plates (E tangential to the interface, Ey with Hx/Hz).

    Ey ~ cos(kz1 z) and cosh(a2 (h - z)); continuity of Ey and dEy/dz gives
    kz1 tan(kz1 d) = a2 tanh(a2 (h - d)). Its quasi-static eps_eff is the
    parallel (arithmetic) mean, the dual of the TM0 series (harmonic) mean.
    """
    return _quasi_tem_omega(kx, eps1, d, h, normal=False)


def semi_discrete_frequency(continuum_omega_of_kx, kx, dx, dt):
    """Yee x/t substitution: kx -> (2/dx) sin(kx dx/2), omega -> (2/dt) sin(omega dt/2).

    For a structure homogeneous along x the semi-discrete Yee operator is the
    continuum operator with these substitutions, so the returned frequency is
    the exact FDTD answer for a continuum-resolved z profile. The residual
    error against it is the z discretization (interface + z dispersion).
    """
    kx_tilde = 2.0 / dx * math.sin(0.5 * kx * dx)
    omega_tilde = continuum_omega_of_kx(kx_tilde)
    return 2.0 / dt * math.asin(0.5 * omega_tilde * dt)


def rayleigh_sphere_q_sca(x, m):
    return 8.0 / 3.0 * x**4 * abs((m * m - 1) / (m * m + 2)) ** 2


def rayleigh_cylinder_q_sca(x, m, polarization):
    if polarization.upper() == "TM":
        return math.pi**2 * x**3 / 8.0 * abs(m * m - 1) ** 2
    return math.pi**2 * x**3 / 4.0 * abs((m * m - 1) / (m * m + 1)) ** 2
