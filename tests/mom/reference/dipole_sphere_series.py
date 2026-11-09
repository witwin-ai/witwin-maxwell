"""Independent series reference: electric dipole outside a homogeneous sphere.

Validation-only NumPy/SciPy code, independent of MoM meshing, quadrature and
CUDA kernels. Convention: exp(-i omega t), outgoing spherical Hankel h_n^(1),
vacuum exterior, nonmagnetic (mu_r = 1) sphere of relative permittivity
``epsilon_r`` (complex with Im >= 0 for loss) centred at ``center``.

Method (vector spherical wave expansion, translation-free): in each region of
wavenumber kappa the fields are

    E = sum_nm A_nm z_n(kappa r) X_nm + B_nm N_nm
    H = kappa/(i omega mu0) * sum_nm A_nm N_nm + B_nm z_n(kappa r) X_nm

with X_nm = L Y_nm / sqrt(n(n+1)) and
N_nm = i sqrt(n(n+1)) z_n/rho Y_nm r_hat + (rho z_n)'/rho (r_hat x X_nm).
The exact analytic dipole incident tangential E/H traces on the sphere are
projected onto the orthonormal tangential harmonics X_nm and r_hat x X_nm with
a Gauss-Legendre (cos theta) x trapezoid (phi) product rule; the traces are
smooth on the sphere because the dipole is exterior, so the rule converges
spectrally. Tangential E/H continuity then gives, per (n, m), two 2x2 systems
for the scattered (h_n) and interior (j_n(m k r)) coefficients. Any dipole
position and orientation is handled directly without rotations. Truncation
is explicit (``n_max``) so callers can check convergence in the number of
terms.

Source normalisation: ``moment`` is the ideal current moment Il [A*m], i.e.
J = Il delta(r - r0), matching the public PointDipole contract.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import sph_harm_y_all, spherical_jn, spherical_yn

C0 = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 1/(MU0*C0**2)


def dipole_fields(points, position, moment, k, omega, mu=MU0):
    """Exact homogeneous-medium fields of an ideal electric current moment."""
    points = np.asarray(points, dtype=float)
    delta = points-np.asarray(position, dtype=float)
    radius = np.linalg.norm(delta, axis=-1)
    if np.any(radius == 0):
        raise ValueError('dipole field is singular at the source')
    unit = delta/radius[..., None]
    p = np.asarray(moment, dtype=complex)
    kr = k*radius
    green = np.exp(1j*kr)/(4*np.pi*radius)
    first = 1+1j/kr-1/kr**2
    second = 1+3j/kr-3/kr**2
    projection = unit@p
    electric = 1j*omega*mu*green[..., None]*(first[..., None]*p-second[..., None]*unit*projection[..., None])
    magnetic = (green*(1j*k-1/radius))[..., None]*np.cross(unit, p)
    return electric, magnetic


def _frame(points):
    r = np.linalg.norm(points, axis=-1)
    theta = np.arccos(np.clip(points[..., 2]/r, -1, 1))
    phi = np.arctan2(points[..., 1], points[..., 0])
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    r_hat = np.stack((st*cp, st*sp, ct), -1)
    t_hat = np.stack((ct*cp, ct*sp, -st), -1)
    p_hat = np.stack((-sp, cp, np.zeros_like(st)), -1)
    return r, theta, phi, r_hat, t_hat, p_hat


def _harmonics(n_max, theta, phi):
    """All Y_nm and the theta/phi components of X_nm (index [n, m], negative m wraps)."""
    value, gradient = sph_harm_y_all(n_max, n_max, theta, phi, diff_n=1)
    n = np.arange(n_max+1)
    m = np.concatenate((np.arange(n_max+1), np.arange(-n_max, 0)))
    norm = np.sqrt(np.maximum(n*(n+1), 1))[:, None]
    shape = (slice(None), slice(None))+(None,)*theta.ndim
    x_theta = -(m[None, :]/norm)[shape]*value/np.sin(theta)
    x_phi = -1j*gradient[..., 0]/norm[shape]
    return value, x_theta, x_phi


def _riccati(n, rho, outgoing):
    z = spherical_jn(n, rho)
    dz = spherical_jn(n, rho, derivative=True)
    if outgoing:
        z = z+1j*spherical_yn(n, rho)
        dz = dz+1j*spherical_yn(n, rho, derivative=True)
    return z, (z+rho*dz)/rho


class DipoleSphereSeries:
    """Series solution truncated at degree ``n_max``."""

    def __init__(self, *, radius, frequency, epsilon_r, position, moment, n_max,
                 center=(0., 0., 0.), quadrature=None):
        self.radius, self.frequency = float(radius), float(frequency)
        self.omega = 2*np.pi*self.frequency
        self.k = self.omega/C0
        self.m = np.sqrt(complex(epsilon_r))
        if self.m.imag < 0:
            self.m = -self.m
        self.center = np.asarray(center, dtype=float)
        self.position = np.asarray(position, dtype=float)
        self.moment = np.asarray(moment, dtype=complex)
        self.n_max = int(n_max)
        if np.linalg.norm(self.position-self.center) <= self.radius:
            raise ValueError('the reference requires an exterior dipole')
        n_theta = self.n_max+40 if quadrature is None else quadrature[0]
        n_phi = 2*self.n_max+40 if quadrature is None else quadrature[1]
        x, w = np.polynomial.legendre.leggauss(n_theta)
        theta = np.arccos(x)
        phi = 2*np.pi*np.arange(n_phi)/n_phi
        tt, pp = np.meshgrid(theta, phi, indexing='ij')
        st, ct, sp, cp = np.sin(tt), np.cos(tt), np.sin(pp), np.cos(pp)
        r_hat = np.stack((st*cp, st*sp, ct), -1)
        t_hat = np.stack((ct*cp, ct*sp, -st), -1)
        p_hat = np.stack((-sp, cp, np.zeros_like(st)), -1)
        e, h = dipole_fields(self.center+self.radius*r_hat, self.position, self.moment, self.k, self.omega)
        # Phi integrals by FFT: entry m holds (2 pi/n_phi) * sum_l q(phi_l) exp(-i m phi_l).
        spectra = [np.fft.fft((v*u).sum(-1), axis=1)*(2*np.pi/n_phi)
                   for v in (e, h) for u in (t_hat, p_hat)]
        e_t, e_p, h_t, h_p = spectra
        _, xt, xp = _harmonics(self.n_max, theta, np.zeros_like(theta))
        x_out = self.k*self.radius
        x_in = self.m*x_out
        c_out = self.k/(1j*self.omega*MU0)
        c_in = self.m*self.k/(1j*self.omega*MU0)
        self.coefficients = {}
        for n in range(1, self.n_max+1):
            jo, djo = _riccati(n, x_out, False)
            ho, dho = _riccati(n, x_out, True)
            ji, dji = _riccati(n, x_in, False)
            for m in range(-n, n+1):
                a_t, a_p = np.conj(xt[n, m]), np.conj(xp[n, m])
                # X = xt t_hat + xp p_hat; r_hat x X = xt p_hat - xp t_hat.
                e1 = (w*(e_t[:, m]*a_t+e_p[:, m]*a_p)).sum()
                e2 = (w*(-e_t[:, m]*a_p+e_p[:, m]*a_t)).sum()
                h1 = (w*(h_t[:, m]*a_t+h_p[:, m]*a_p)).sum()
                h2 = (w*(-h_t[:, m]*a_p+h_p[:, m]*a_t)).sum()
                # A-type: E_X and H_(r x X) continuity.
                s_a, t_a = np.linalg.solve(np.array([[ho, -ji], [c_out*dho, -c_in*dji]]), -np.array([e1, h2]))
                # B-type: E_(r x X) and H_X continuity.
                s_b, t_b = np.linalg.solve(np.array([[dho, -dji], [c_out*ho, -c_in*ji]]), -np.array([e2, h1]))
                self.coefficients[n, m] = (s_a, s_b, t_a, t_b, e1/jo, e2/djo)

    def _expand(self, points, region):
        """region: 'scattered' (outgoing exterior), 'interior' or 'regular_incident'."""
        local = np.asarray(points, dtype=float)-self.center
        r, theta, phi, r_hat, t_hat, p_hat = _frame(local)
        if np.any(np.sin(theta) < 1e-9):
            raise ValueError('reference probes must avoid the sphere-frame polar axis')
        kappa = self.m*self.k if region == 'interior' else self.k
        rho = kappa*r
        electric = np.zeros(local.shape, dtype=complex)
        magnetic = np.zeros(local.shape, dtype=complex)
        index = {'scattered': (0, 1), 'interior': (2, 3), 'regular_incident': (4, 5)}[region]
        y, xt, xp = _harmonics(self.n_max, theta, phi)
        for n in range(1, self.n_max+1):
            z, dz = _riccati(n, rho, region == 'scattered')
            root = math.sqrt(n*(n+1))
            for m in range(-n, n+1):
                a, b = (self.coefficients[n, m][i] for i in index)
                x_vec = xt[n, m][:, None]*t_hat+xp[n, m][:, None]*p_hat
                rx_vec = xt[n, m][:, None]*p_hat-xp[n, m][:, None]*t_hat
                m_vec = z[:, None]*x_vec
                n_vec = (1j*root*z/rho*y[n, m])[:, None]*r_hat+dz[:, None]*rx_vec
                electric += a*m_vec+b*n_vec
                magnetic += a*n_vec+b*m_vec
        return electric, magnetic*(kappa/(1j*self.omega*MU0))

    def scattered(self, points):
        """Exterior scattered E/H at points outside the sphere."""
        points = np.asarray(points, dtype=float)
        if np.any(np.linalg.norm(points-self.center, axis=-1) <= self.radius):
            raise ValueError('scattered probes must be exterior')
        return self._expand(points, 'scattered')

    def incident(self, points):
        return dipole_fields(points, self.position, self.moment, self.k, self.omega)

    def total(self, points):
        """Physical total field: incident plus scattered outside, transmitted inside."""
        points = np.asarray(points, dtype=float)
        inside = np.linalg.norm(points-self.center, axis=-1) < self.radius
        electric = np.zeros(points.shape, dtype=complex)
        magnetic = np.zeros(points.shape, dtype=complex)
        if inside.any():
            electric[inside], magnetic[inside] = self._expand(points[inside], 'interior')
        if (~inside).any():
            es, hs = self._expand(points[~inside], 'scattered')
            ei, hi = self.incident(points[~inside])
            electric[~inside], magnetic[~inside] = es+ei, hs+hi
        return electric, magnetic

    def regular_incident(self, points):
        """Regular-wave reconstruction of the incident field, valid for |r - c| < |r0 - c|."""
        return self._expand(points, 'regular_incident')
