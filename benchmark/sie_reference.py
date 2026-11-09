"""Independent analytic sphere oracle for SIE acceptance.

This validation-only implementation uses special functions, independently of
surface meshing, quadrature and CUDA solver kernels. It is not a solver fallback.
Convention: exp(-i omega t), outgoing spherical Hankel function of the first kind.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def sphere_coefficients(size_parameter, epsilon_r, cutoff=None):
    x = float(size_parameter)
    if not np.isfinite(x) or x <= 0:
        raise ValueError("size_parameter must be finite and positive")
    eps = complex(epsilon_r)
    if not np.isfinite(eps) or eps.real <= 0 or eps.imag < 0:
        raise ValueError("The oracle requires a passive positive dielectric")
    cutoff = int(np.ceil(x+4*x**(1/3)+8)) if cutoff is None else cutoff
    if not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("cutoff must be positive")
    n = np.arange(1, cutoff+1)
    m = np.sqrt(eps)
    y = m*x
    psi = x*spherical_jn(n, x)
    dpsi = spherical_jn(n, x)+x*spherical_jn(n, x, derivative=True)
    psim = y*spherical_jn(n, y)
    dpsim = spherical_jn(n, y)+y*spherical_jn(n, y, derivative=True)
    xi = x*(spherical_jn(n, x)+1j*spherical_yn(n, x))
    dxi = dpsi+1j*(spherical_yn(n, x)+x*spherical_yn(n, x, derivative=True))
    a = (m*psim*dpsi-psi*dpsim)/(m*psim*dxi-xi*dpsim)
    b = (psim*dpsi-m*psi*dpsim)/(psim*dxi-m*xi*dpsim)
    return a, b


def sphere_cross_sections(radius, k, epsilon_r, cutoff=None):
    """Scattering, extinction and absorption cross sections in square metres."""
    a, b = sphere_coefficients(k*radius, epsilon_r, cutoff)
    n = np.arange(1, len(a)+1)
    scattering = 2*np.pi/k**2*np.sum((2*n+1)*(np.abs(a)**2+np.abs(b)**2))
    extinction = 2*np.pi/k**2*np.sum((2*n+1)*(a+b).real)
    return {"scattering": float(scattering), "extinction": float(extinction),
            "absorption": float(extinction-scattering), "cutoff": len(a)}


def sphere_far_field(radius, k, epsilon_r, directions, cutoff=None):
    """Complex E amplitude for a unit x-polarized wave propagating along +z.

    E_scattered(r*d) = exp(i*k*r)/r * returned_amplitude(d) + O(r^-2).
    The scatterer is centred at the origin. This is a Cartesian vector,
    retaining phase and angular polarization, rather than only intensity.
    """
    d = np.asarray(directions, dtype=float)
    if d.ndim != 2 or d.shape[1] != 3 or not np.allclose(np.linalg.norm(d, axis=1), 1):
        raise ValueError("directions must be unit vectors [N,3]")
    a, b = sphere_coefficients(k*radius, epsilon_r, cutoff)
    mu = d[:, 2]
    phi = np.arctan2(d[:, 1], d[:, 0])
    sine = np.sqrt(np.maximum(0, 1-mu**2))
    pi_prev, pi_now = np.zeros_like(mu), np.ones_like(mu)
    s1, s2 = np.zeros_like(mu, dtype=complex), np.zeros_like(mu, dtype=complex)
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tau = n*mu*pi_now-(n+1)*pi_prev
        factor = (2*n+1)/(n*(n+1))
        s1 += factor*(an*pi_now+bn*tau)
        s2 += factor*(an*tau+bn*pi_now)
        pi_prev, pi_now = pi_now, ((2*n+1)*mu*pi_now-(n+1)*pi_prev)/n
    etheta = np.column_stack((mu*np.cos(phi), mu*np.sin(phi), -sine))
    ephi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(mu)))
    return 1j/k*(s2[:, None]*np.cos(phi)[:, None]*etheta-s1[:, None]*np.sin(phi)[:, None]*ephi)


def sphere_fields(radius, k, epsilon_r, points, *, eta=376.730313668, field_kind="total", cutoff=None):
    """Independent Cartesian E [V/m], H [A/m] for a unit +z/x plane wave.

    Nonmagnetic passive sphere, real lossless exterior k and impedance eta.
    Points are measured from the sphere centre; exact surface queries are
    rejected because normal E is discontinuous. Scattered means total minus
    the analytic exterior incident wave, including at internal points.

    Electric partial waves follow Wallen et al. (2018), equations (1)-(5),
    https://doi.org/10.23919/URSI-AT-RASC.2018.8471401. Magnetic waves follow
    curl(E)/(i omega mu). Internal c,d follow tangential continuity, with the
    Riccati Wronskian psi*xi'-xi*psi'=i avoiding subtractive cancellation.
    """
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(k) or k <= 0:
        raise ValueError("radius and exterior k must be finite and positive")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Exterior eta must be finite, real and positive")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite [N,3]")
    if field_kind not in ("total", "scattered", "incident"):
        raise ValueError("field_kind must be total, scattered or incident")
    r = np.linalg.norm(points, axis=-1)
    if np.any(np.abs(r-radius) <= 16*np.finfo(float).eps*radius):
        raise ValueError("Exact interface queries require a specified one-sided limit")
    incident_e = np.zeros(points.shape, dtype=complex)
    incident_h = np.zeros(points.shape, dtype=complex)
    incident_e[:, 0] = np.exp(1j*k*points[:, 2])
    incident_h[:, 1] = incident_e[:, 0]/eta
    if field_kind == "incident":
        return incident_e, incident_h
    a, b = sphere_coefficients(k*radius, epsilon_r, cutoff)
    nvec = np.arange(1, len(a)+1)
    m, size = np.sqrt(complex(epsilon_r)), k*radius
    psi_m = m*size*spherical_jn(nvec, m*size)
    dpsi_m = spherical_jn(nvec, m*size)+m*size*spherical_jn(nvec, m*size, derivative=True)
    xi = size*(spherical_jn(nvec, size)+1j*spherical_yn(nvec, size))
    dxi = (spherical_jn(nvec, size)+size*spherical_jn(nvec, size, derivative=True)
           +1j*(spherical_yn(nvec, size)+size*spherical_yn(nvec, size, derivative=True)))
    c = 1j*m/(psi_m*dxi-m*xi*dpsi_m)
    d = 1j*m/(m*psi_m*dxi-xi*dpsi_m)
    electric, magnetic = np.zeros_like(incident_e), np.zeros_like(incident_h)
    for inside in (False, True):
        mask = r < radius if inside else r > radius
        if not mask.any():
            continue
        rr, xyz = r[mask], points[mask]
        mu = np.divide(xyz[:, 2], rr, out=np.ones_like(rr), where=rr > 0)
        sine = np.sqrt(np.maximum(0., 1-mu*mu))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        cp, sp = np.cos(phi), np.sin(phi)
        er = np.column_stack((sine*cp, sine*sp, mu))
        et = np.column_stack((mu*cp, mu*sp, -sine))
        ep = np.column_stack((-sp, cp, np.zeros_like(rr)))
        rho = k*(m if inside else 1)*rr
        pi_prev, pi_now = np.zeros_like(rr), np.ones_like(rr)
        ee, hh = np.zeros_like(xyz, dtype=complex), np.zeros_like(xyz, dtype=complex)
        for n, (an, bn, cn, dn) in enumerate(zip(a, b, c, d), 1):
            tau = n*mu*pi_now-(n+1)*pi_prev
            zn = spherical_jn(n, rho)
            dz = spherical_jn(n, rho, derivative=True)
            if not inside:
                zn = zn+1j*spherical_yn(n, rho)
                dz = dz+1j*spherical_yn(n, rho, derivative=True)
            zr = np.divide(zn, rho, out=np.full_like(zn, 1/3 if n == 1 else 0), where=rho != 0)
            dr = zr+dz
            radial = n*(n+1)*sine*pi_now*zr
            mo = (cp*pi_now*zn)[:, None]*et-(sp*tau*zn)[:, None]*ep
            me = -(sp*pi_now*zn)[:, None]*et-(cp*tau*zn)[:, None]*ep
            ne = (cp*radial)[:, None]*er+(cp*tau*dr)[:, None]*et-(sp*pi_now*dr)[:, None]*ep
            no = (sp*radial)[:, None]*er+(sp*tau*dr)[:, None]*et+(cp*pi_now*dr)[:, None]*ep
            en = 1j**n*(2*n+1)/(n*(n+1))
            if inside:
                ee += en*(cn*mo-1j*dn*ne)
                hh += en*(-dn*me-1j*cn*no)*m/eta
            else:
                ee += en*(1j*an*ne-bn*mo)
                hh += en*(an*me+1j*bn*no)/eta
            pi_prev, pi_now = pi_now, ((2*n+1)*mu*pi_now-(n+1)*pi_prev)/n
        electric[mask], magnetic[mask] = ee, hh
    outside = r > radius
    if field_kind == "total":
        electric[outside] += incident_e[outside]
        magnetic[outside] += incident_h[outside]
    else:
        electric[~outside] -= incident_e[~outside]
        magnetic[~outside] -= incident_h[~outside]
    return electric, magnetic
