"""Mie series oracle for a homogeneous sphere in vacuum under a unit plane wave.

Validation-only analytic reference: special functions only, independent of the
solver grid, materials compiler and kernels. Convention exp(-i omega t); the
incident field is E = x_hat exp(i k z), H = y_hat exp(i k z) / eta0, and the
sphere is centred at the origin. Materials are PEC (``PEC``) or a passive,
nonmagnetic complex relative permittivity (lossless, lossy, or a dispersive
model evaluated at the drive frequency with the helpers below).

Dielectric partial-wave coefficients and dielectric near fields reuse
``benchmark.sie_reference`` (the bench-pinned SIE oracle, left byte-identical);
this module adds the PEC limit, coefficient-generic far fields, cross sections,
radar cross sections and exterior near fields, and an independent
logarithmic-derivative coefficient path used to validate the series.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from benchmark.sie_reference import sphere_coefficients as _dielectric_coefficients
from benchmark.sie_reference import sphere_fields as _dielectric_fields

PEC = "pec"
ETA0 = 376.730313668
C0 = 299_792_458.0


def wiscombe_cutoff(size_parameter: float) -> int:
    """Wiscombe (1980) series length; the default adds a safety margin of 8 terms."""
    x = float(size_parameter)
    return int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2.0))


def default_cutoff(size_parameter: float) -> int:
    return wiscombe_cutoff(size_parameter) + 8


def _riccati(n, x):
    psi = x * spherical_jn(n, x)
    dpsi = spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
    chi = x * spherical_yn(n, x)
    dchi = spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True)
    return psi, dpsi, psi + 1j * chi, dpsi + 1j * dchi


def coefficients(size_parameter, material, cutoff=None):
    """External partial-wave coefficients (a_n, b_n), n = 1..cutoff (Bohren-Huffman)."""
    x = float(size_parameter)
    cutoff = default_cutoff(x) if cutoff is None else int(cutoff)
    if isinstance(material, str):
        if material != PEC:
            raise ValueError(f"Unknown sphere material {material!r}.")
        n = np.arange(1, cutoff + 1)
        psi, dpsi, xi, dxi = _riccati(n, x)
        return dpsi / dxi, psi / xi
    return _dielectric_coefficients(x, complex(material), cutoff)


def coefficients_log_derivative(size_parameter, epsilon_r, cutoff=None):
    """Independent dielectric coefficients through the downward D_n(mx) recurrence.

    Bohren and Huffman (1983) section 4.8: a_n = ((D_n/m + n/x) psi_n - psi_{n-1}) /
    ((D_n/m + n/x) xi_n - xi_{n-1}), b_n with m D_n. Used to validate the
    Bessel-ratio path above, which evaluates j_n(mx) directly.
    """
    x = float(size_parameter)
    m = np.sqrt(complex(epsilon_r))
    cutoff = default_cutoff(x) if cutoff is None else int(cutoff)
    mx = m * x
    # Start well above max(n, |mx|): for real mx the backward recurrence converges
    # slowly near n ~ |mx| (a +16 start leaves ~1e-7 errors at |mx| = 60).
    start = int(1.5 * max(cutoff, abs(mx))) + 50
    d = np.zeros(start + 1, dtype=complex)
    for n in range(start, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)
    n = np.arange(0, cutoff + 1)
    psi = x * spherical_jn(n, x)
    xi = psi + 1j * x * spherical_yn(n, x)
    nn = n[1:]
    da = d[1:cutoff + 1] / m + nn / x
    db = d[1:cutoff + 1] * m + nn / x
    a = (da * psi[1:] - psi[:-1]) / (da * xi[1:] - xi[:-1])
    b = (db * psi[1:] - psi[:-1]) / (db * xi[1:] - xi[:-1])
    return a, b


def cross_sections(radius, k, material, cutoff=None):
    """Extinction, scattering, absorption and monostatic (back) cross sections [m^2]."""
    a, b = coefficients(k * radius, material, cutoff)
    n = np.arange(1, len(a) + 1)
    weight = 2 * n + 1
    scattering = 2 * np.pi / k**2 * float(np.sum(weight * (np.abs(a) ** 2 + np.abs(b) ** 2)))
    extinction = 2 * np.pi / k**2 * float(np.sum(weight * (a + b).real))
    back = np.pi / k**2 * float(abs(np.sum(weight * (-1.0) ** n * (a - b))) ** 2)
    return {
        "extinction": extinction,
        "scattering": scattering,
        "absorption": extinction - scattering,
        "backscatter": back,
        "cutoff": len(a),
    }


def amplitude_functions(a, b, mu):
    """Bohren-Huffman S1(mu), S2(mu) for mu = cos(theta)."""
    mu = np.asarray(mu, dtype=float)
    pi_prev, pi_now = np.zeros_like(mu), np.ones_like(mu)
    s1 = np.zeros_like(mu, dtype=complex)
    s2 = np.zeros_like(mu, dtype=complex)
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tau = n * mu * pi_now - (n + 1) * pi_prev
        factor = (2 * n + 1) / (n * (n + 1))
        s1 += factor * (an * pi_now + bn * tau)
        s2 += factor * (an * tau + bn * pi_now)
        pi_prev, pi_now = pi_now, ((2 * n + 1) * mu * pi_now - (n + 1) * pi_prev) / n
    return s1, s2


def far_field(radius, k, material, directions, cutoff=None):
    """Cartesian far-field amplitude F with E_s(r d) = exp(i k r) / r F(d) + O(r^-2)."""
    d = np.asarray(directions, dtype=float)
    if d.ndim != 2 or d.shape[1] != 3 or not np.allclose(np.linalg.norm(d, axis=1), 1):
        raise ValueError("directions must be unit vectors [N,3]")
    a, b = coefficients(k * radius, material, cutoff)
    mu = d[:, 2]
    phi = np.arctan2(d[:, 1], d[:, 0])
    sine = np.sqrt(np.maximum(0.0, 1 - mu**2))
    s1, s2 = amplitude_functions(a, b, mu)
    etheta = np.column_stack((mu * np.cos(phi), mu * np.sin(phi), -sine))
    ephi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(mu)))
    return 1j / k * (
        (s2 * np.cos(phi))[:, None] * etheta - (s1 * np.sin(phi))[:, None] * ephi
    )


def bistatic_rcs(radius, k, material, theta, phi, cutoff=None):
    """sigma(theta, phi) = 4 pi |F|^2 for a unit incident field [m^2]."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    directions = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1
    ).reshape(-1, 3)
    amplitude = far_field(radius, k, material, directions, cutoff)
    return (4 * np.pi * np.sum(np.abs(amplitude) ** 2, axis=-1)).reshape(theta.shape)


def _exterior_scattered(radius, k, a, b, points, eta):
    """Scattered E, H outside the sphere from (a_n, b_n): Bohren-Huffman eq. 4.45."""
    r = np.linalg.norm(points, axis=-1)
    mu = points[:, 2] / r
    sine = np.sqrt(np.maximum(0.0, 1 - mu * mu))
    phi = np.arctan2(points[:, 1], points[:, 0])
    cp, sp = np.cos(phi), np.sin(phi)
    er = np.column_stack((sine * cp, sine * sp, mu))
    et = np.column_stack((mu * cp, mu * sp, -sine))
    ep = np.column_stack((-sp, cp, np.zeros_like(r)))
    rho = k * r
    pi_prev, pi_now = np.zeros_like(r), np.ones_like(r)
    e = np.zeros(points.shape, dtype=complex)
    h = np.zeros(points.shape, dtype=complex)
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tau = n * mu * pi_now - (n + 1) * pi_prev
        zn = spherical_jn(n, rho) + 1j * spherical_yn(n, rho)
        dz = spherical_jn(n, rho, derivative=True) + 1j * spherical_yn(n, rho, derivative=True)
        zr = zn / rho
        dr = zr + dz
        radial = n * (n + 1) * sine * pi_now * zr
        mo = (cp * pi_now * zn)[:, None] * et - (sp * tau * zn)[:, None] * ep
        me = -(sp * pi_now * zn)[:, None] * et - (cp * tau * zn)[:, None] * ep
        ne = (cp * radial)[:, None] * er + (cp * tau * dr)[:, None] * et - (sp * pi_now * dr)[:, None] * ep
        no = (sp * radial)[:, None] * er + (sp * tau * dr)[:, None] * et + (cp * pi_now * dr)[:, None] * ep
        en = 1j**n * (2 * n + 1) / (n * (n + 1))
        e += en * (1j * an * ne - bn * mo)
        h += en * (an * me + 1j * bn * no) / eta
        pi_prev, pi_now = pi_now, ((2 * n + 1) * mu * pi_now - (n + 1) * pi_prev) / n
    return e, h


def near_fields(radius, k, material, points, *, field_kind="total", eta=ETA0, cutoff=None):
    """Cartesian E [V/m], H [A/m] at points (sphere centred at the origin).

    field_kind is total, scattered or incident. PEC interior points return a zero
    total field. Dielectric spheres delegate to the pinned SIE near-field oracle.
    """
    points = np.asarray(points, dtype=float)
    if not (isinstance(material, str) and material == PEC):
        return _dielectric_fields(radius, k, complex(material), points, eta=eta,
                                  field_kind=field_kind, cutoff=cutoff)
    if field_kind not in ("total", "scattered", "incident"):
        raise ValueError("field_kind must be total, scattered or incident")
    incident_e = np.zeros(points.shape, dtype=complex)
    incident_e[:, 0] = np.exp(1j * k * points[:, 2])
    incident_h = np.zeros(points.shape, dtype=complex)
    incident_h[:, 1] = incident_e[:, 0] / eta
    if field_kind == "incident":
        return incident_e, incident_h
    a, b = coefficients(k * radius, PEC, cutoff)
    r = np.linalg.norm(points, axis=-1)
    outside = r > radius
    scattered_e = np.zeros_like(incident_e)
    scattered_h = np.zeros_like(incident_h)
    if outside.any():
        scattered_e[outside], scattered_h[outside] = _exterior_scattered(
            radius, k, a, b, points[outside], eta)
    scattered_e[~outside] = -incident_e[~outside]
    scattered_h[~outside] = -incident_h[~outside]
    if field_kind == "scattered":
        return scattered_e, scattered_h
    return incident_e + scattered_e, incident_h + scattered_h


# Dispersive permittivity models evaluated at a real drive frequency f [Hz]
# (exp(-i omega t), Im eps >= 0 for loss). Closed forms written independently of
# the library pole classes; tests check them against Material.relative_permittivity.

def lorentz_permittivity(frequency, *, eps_inf, delta_eps, resonance_frequency, gamma):
    w, w0, g = 2 * np.pi * frequency, 2 * np.pi * resonance_frequency, 2 * np.pi * gamma
    return eps_inf + delta_eps * w0**2 / (w0**2 - w**2 - 1j * g * w)


def lorentz_permittivity_discrete(frequency, dt, *, eps_inf, delta_eps, resonance_frequency, gamma):
    """Susceptibility of the FDTD J-P Lorentz ADE at a CW drive (not a physical model).

    The update J^{n+1/2} = decay J^{n-1/2} - restoring P^n + drive E^n,
    P^{n+1} = P^n + dt J^{n+1/2} is the central difference of
    P'' + gamma P' + w0^2 P = eps0 delta_eps w0^2 E. For P, E ~ exp(-i w n dt) it gives
    chi = delta_eps w0^2 / (w0^2 - W^2 - i gamma W cos(w dt / 2)), W = 2 sin(w dt / 2) / dt,
    whose resonance sits at w with W = w0, i.e. above w0 by about (w0 dt)^2 / 24.
    """
    w, w0, g = 2 * np.pi * frequency, 2 * np.pi * resonance_frequency, 2 * np.pi * gamma
    big_w = 2 * np.sin(0.5 * w * dt) / dt
    return eps_inf + delta_eps * w0**2 / (w0**2 - big_w**2 - 1j * g * big_w * np.cos(0.5 * w * dt))


def debye_permittivity(frequency, *, eps_inf, delta_eps, tau):
    return eps_inf + delta_eps / (1 - 1j * 2 * np.pi * frequency * tau)


def sellmeier_permittivity(frequency, *, eps_inf, b_coefficients, c_coefficients):
    wavelength_sq = (C0 / frequency) ** 2
    return eps_inf + sum(b * wavelength_sq / (wavelength_sq - c)
                         for b, c in zip(b_coefficients, c_coefficients))


def conductive_permittivity(frequency, *, eps_r, sigma_e):
    return eps_r + 1j * sigma_e / (2 * np.pi * frequency * 8.8541878128e-12)
