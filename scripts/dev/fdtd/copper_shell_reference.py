"""Independent full-Maxwell conductive spherical-shell validation oracle.

Validation only: NumPy/SciPy are never used in the time marcher. Each TE/TM
partial wave is obtained by continuity of tangential E and H at BOTH physical
radii, rather than from a surface impedance or the production copper model.
Convention is exp(-i omega t), exterior and core vacuum, shell mu=mu0.

The boundary construction follows Aden and Kerker, J. Appl. Phys. 22, 1242
(1951), https://doi.org/10.1063/1.1699834 . Exponentially scaled integer-order
Hankel polynomials avoid overflow when the copper radius spans many skin depths.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import spherical_jn, spherical_yn


EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
C_0 = 1/math.sqrt(EPS_0*MU_0)


def scaled_hankel(order, z, anchor, *, outgoing):
    """Return h_l(z) and (z h_l)'/z, with exp(+-i anchor) removed.

    Anchors are arguments at the inner face for outgoing waves and at the
    outer face for incoming waves, so the retained exponent never grows in
    a passive shell. Both returned values have the SAME column scaling.
    """
    z = np.asarray(z, dtype=np.complex128)
    sign = 1 if outgoing else -1
    polynomial = np.ones_like(z)
    derivative = np.zeros_like(z)
    term = np.ones_like(z)
    for m in range(1, order+1):
        term = term*(sign*1j)*(order+m)*(order-m+1)/(2*m*z)
        polynomial = polynomial+term
        derivative = derivative-m*term/z
    prefactor = (-sign*1j)**(order+1)*np.exp(sign*1j*(z-anchor))/z
    return prefactor*polynomial, prefactor*(sign*1j*polynomial+derivative)


def _regular(order, z):
    value = spherical_jn(order, z)
    return value, value/z+spherical_jn(order, z, derivative=True)


def _outgoing(order, z):
    value = spherical_jn(order, z)+1j*spherical_yn(order, z)
    derivative = spherical_jn(order, z, derivative=True)+1j*spherical_yn(order, z, derivative=True)
    return value, value/z+derivative


@dataclass(frozen=True)
class ConductiveShellSolution:
    radius: float
    thickness: float
    frequency: float
    conductivity: float
    a: np.ndarray
    b: np.ndarray
    interior_te: np.ndarray
    interior_tm: np.ndarray
    shell_te: np.ndarray
    shell_tm: np.ndarray
    boundary_residual: float

    @property
    def cross_sections(self):
        k = 2*math.pi*self.frequency/C_0
        weight = 2*np.arange(1, len(self.a)+1)+1
        scale = 2*math.pi/k**2
        scattering = scale*np.sum(weight*(abs(self.a)**2+abs(self.b)**2))
        extinction = scale*np.sum(weight*(self.a.real+self.b.real))
        return dict(scattering=float(scattering), extinction=float(extinction),
            absorption=float(extinction-scattering))


def conductive_shell(radius, thickness, frequency, conductivity, *, cutoff=None):
    """Solve the exact vacuum / lossy-shell / vacuum boundary problem.

    The input radius is the center surface, not the outer physical radius.
    Coefficients a,b use the standard exterior expansion incident-scattered;
    interior coefficients multiply the regular incident-wave TE/TM basis.
    """
    for name, value in (("radius", radius), ("thickness", thickness), ("frequency", frequency)):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"Shell {name} must be finite and positive.")
    if not math.isfinite(float(conductivity)) or conductivity < 0 or thickness >= 2*radius:
        raise ValueError("Shell conductivity must be nonnegative and inner radius positive.")
    inner, outer = radius-thickness/2, radius+thickness/2
    omega = 2*math.pi*frequency
    k = omega/C_0
    kc = k*np.sqrt(1+1j*conductivity/(omega*EPS_0))
    cutoff = int(math.ceil(k*outer+4*(k*outer)**(1/3)+8)) if cutoff is None else int(cutoff)
    if cutoff < 1:
        raise ValueError("Shell partial-wave cutoff must be positive.")
    a, b, te, tm, shell_te, shell_tm = [], [], [], [], [], []
    residual = 0.
    for order in range(1, cutoff+1):
        jo, djo = _regular(order, k*outer)
        ho, dho = _outgoing(order, k*outer)
        ji, dji = _regular(order, k*inner)
        if abs((kc*outer).imag) < 10:
            # For a weakly absorbing small sphere, outgoing/incoming columns
            # nearly cancel in the regular solution at high order. The j/y
            # fundamental pair avoids that unrelated cancellation. At large
            # imaginary arguments the anchored Hankel pair avoids overflow.
            f1o, d1o = _regular(order, kc*outer)
            f1i, d1i = _regular(order, kc*inner)
            f2o = spherical_yn(order, kc*outer)
            f2i = spherical_yn(order, kc*inner)
            d2o = f2o/(kc*outer)+spherical_yn(order, kc*outer, derivative=True)
            d2i = f2i/(kc*inner)+spherical_yn(order, kc*inner, derivative=True)
        else:
            f1o, d1o = scaled_hankel(order, kc*outer, kc*inner, outgoing=True)
            f2o, d2o = scaled_hankel(order, kc*outer, kc*outer, outgoing=False)
            f1i, d1i = scaled_hankel(order, kc*inner, kc*inner, outgoing=True)
            f2i, d2i = scaled_hankel(order, kc*inner, kc*outer, outgoing=False)
        for transverse_electric in (True, False):
            if transverse_electric:
                matrix = np.array(((ho, -f1o, -f2o, 0), (k*dho, -kc*d1o, -kc*d2o, 0),
                    (0, f1i, f2i, -ji), (0, kc*d1i, kc*d2i, -k*dji)), dtype=complex)
                rhs = np.array((-jo, -k*djo, 0, 0), dtype=complex)
            else:
                matrix = np.array(((dho, -d1o, -d2o, 0), (k*ho, -kc*f1o, -kc*f2o, 0),
                    (0, d1i, d2i, -dji), (0, kc*f1i, kc*f2i, -k*ji)), dtype=complex)
                rhs = np.array((-djo, -k*jo, 0, 0), dtype=complex)
            # Column/row equilibration changes no equation and introduces no
            # regularizer. The shell columns otherwise have unlike SI scales.
            column = np.max(abs(matrix), axis=0)
            balanced = matrix/column
            row = np.max(abs(balanced), axis=1)
            equilibrated = balanced/row[:, None]
            unknown = np.linalg.solve(equilibrated, rhs/row)/column
            # Extended-precision residual refinement improves thin-shell
            # cancellation without changing equations or adding a tolerance.
            for _ in range(2):
                remainder = rhs.astype(np.clongdouble)-matrix.astype(np.clongdouble)@unknown.astype(np.clongdouble)
                unknown += np.linalg.solve(equilibrated, np.asarray(remainder/row, dtype=complex))/column
            error = abs(matrix@unknown-rhs)
            denominator = abs(matrix)@abs(unknown)+abs(rhs)
            residual = max(residual, float(np.max(error/np.maximum(denominator, np.finfo(float).tiny))))
            if transverse_electric:
                b.append(-unknown[0]); te.append(unknown[3]); shell_te.append(unknown[1:3])
            else:
                a.append(-unknown[0]); tm.append(unknown[3]); shell_tm.append(unknown[1:3])
    return ConductiveShellSolution(radius, thickness, frequency, conductivity,
        np.array(a), np.array(b), np.array(te), np.array(tm),
        np.array(shell_te), np.array(shell_tm), residual)


def sheet_sphere_coefficients(radius, frequency, electric_admittance, magnetic_impedance, *, cutoff):
    """Independent spherical GSTC boundary solve, not a finite-shell oracle.

    This intermediate control measures the geometrical thin-layer error of
    collapsing the exact physical shell onto its center surface. The two
    equations are jump(n x H)=Y*average(E) and jump(E)=Z*average(n x H).
    """
    eta = math.sqrt(MU_0/EPS_0)
    x = 2*math.pi*frequency/C_0*radius
    coefficients = [[], [], [], []]
    for order in range(1, cutoff+1):
        j, dj = _regular(order, x)
        h, dh = _outgoing(order, x)
        for te in (True, False):
            ej, eh = (j, h) if te else (dj, dh)
            qj, qh = (1j*dj/eta, 1j*dh/eta) if te else (-1j*j/eta, -1j*h/eta)
            y, z = electric_admittance, magnetic_impedance
            matrix = np.array(((qh-y*eh/2, -qj-y*ej/2),
                               (eh-z*qh/2, -ej-z*qj/2)))
            rhs = -np.array((qj-y*ej/2, ej-z*qj/2))
            scattering, transmission = np.linalg.solve(matrix, rhs)
            coefficients[1 if te else 0].append(-scattering)
            coefficients[2 if te else 3].append(transmission)
    return tuple(np.array(value) for value in coefficients)


def vacuum_region_fields(solution, points, *, region="exterior"):
    """Full Cartesian E,H for a unit +z x-polarized plane wave.

    Exterior returns scattered fields; vacuum core returns total fields.
    Shell-interior fields are deliberately excluded from this near-field API.
    """
    points = np.asarray(points, dtype=float)
    r = np.linalg.norm(points, axis=-1)
    if region not in ("exterior", "interior") or np.any(r == 0):
        raise ValueError("Shell field query requires nonzero points in exterior or interior vacuum.")
    outer, inner = solution.radius+solution.thickness/2, solution.radius-solution.thickness/2
    if np.any(r < outer) if region == "exterior" else np.any(r > inner):
        raise ValueError("The shell field query crosses a physical material boundary.")
    k, eta = 2*math.pi*solution.frequency/C_0, math.sqrt(MU_0/EPS_0)
    mu = points[:, 2]/r
    sine = np.sqrt(np.maximum(0., 1-mu*mu))
    phi = np.arctan2(points[:, 1], points[:, 0])
    cp, sp = np.cos(phi), np.sin(phi)
    er = np.column_stack((sine*cp, sine*sp, mu))
    et = np.column_stack((mu*cp, mu*sp, -sine))
    ep = np.column_stack((-sp, cp, np.zeros_like(r)))
    previous, current = np.zeros_like(r), np.ones_like(r)
    electric, magnetic = np.zeros_like(points, dtype=complex), np.zeros_like(points, dtype=complex)
    rho = k*r
    for n in range(1, len(solution.a)+1):
        tau = n*mu*current-(n+1)*previous
        radial_function, dr = _outgoing(n, rho) if region == "exterior" else _regular(n, rho)
        radial = n*(n+1)*sine*current*radial_function/rho
        mo = (cp*current*radial_function)[:, None]*et-(sp*tau*radial_function)[:, None]*ep
        me = -(sp*current*radial_function)[:, None]*et-(cp*tau*radial_function)[:, None]*ep
        ne = (cp*radial)[:, None]*er+(cp*tau*dr)[:, None]*et-(sp*current*dr)[:, None]*ep
        no = (sp*radial)[:, None]*er+(sp*tau*dr)[:, None]*et+(cp*current*dr)[:, None]*ep
        en = 1j**n*(2*n+1)/(n*(n+1))
        if region == "exterior":
            an, bn = solution.a[n-1], solution.b[n-1]
            electric += en*(1j*an*ne-bn*mo)
            magnetic += en*(an*me+1j*bn*no)/eta
        else:
            cn, dn = solution.interior_te[n-1], solution.interior_tm[n-1]
            electric += en*(cn*mo-1j*dn*ne)
            magnetic += en*(-dn*me-1j*cn*no)/eta
        previous, current = current, ((2*n+1)*mu*current-(n+1)*previous)/n
    return electric, magnetic
