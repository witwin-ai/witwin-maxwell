"""Berreman 4x4 transfer-matrix oracle for anisotropic planar multilayers.

Validation-only analytic reference for a stack of homogeneous layers with full
3x3 relative permittivity tensors (mu = 1) between isotropic half spaces.
Convention exp(-i omega t); fields vary as exp(i k0 xi x) in the plane of
incidence x-z with the stack normal along +z and the wave incident from z < z0.

State vector psi = (Ex, Hy~, Ey, Hx~) with H~ = eta0 H. Maxwell's curl equations
with d/dx = i k0 xi, d/dy = 0 give d psi / d(k0 z) = i M psi, where Ez is
eliminated through (eps E)_z = -xi Hy~. Each layer propagates by expm(i M k0 d).

Half-space modes (refractive index n, q = sqrt(n^2 - xi^2), Im q >= 0), unit
electric-field amplitude A:
  p forward  : Ex =  q/n A, Hy~ =  n A        (E = (q, 0, -xi)/n A)
  p backward : Ex =  q/n A, Hy~ = -n A        (E = (q, 0,  xi)/n A)
  s forward  : Ey = A, Hx~ = -q A
  s backward : Ey = A, Hx~ =  q A
For a real index the z power flux of every mode is proportional to Re(q) |A|^2,
so power fractions are |amplitude|^2 Re(q_out) / Re(q_in).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

C0 = 299_792_458.0


@dataclass(frozen=True)
class Layer:
    thickness: float
    epsilon: np.ndarray  # 3x3 complex relative permittivity


def tensor(epsilon):
    """3x3 complex tensor from a scalar, a diagonal triple or a 3x3 array."""
    e = np.asarray(epsilon, dtype=complex)
    if e.shape == ():
        return e * np.eye(3)
    if e.shape == (3,):
        return np.diag(e)
    return e


def system_matrix(epsilon, xi):
    """M with d psi / d(k0 z) = i M psi for psi = (Ex, Hy~, Ey, Hx~)."""
    e = tensor(epsilon)
    # Ez = (-xi Hy - ezx Ex - ezy Ey) / ezz, as a row over psi.
    ez = np.array([-e[2, 0], -xi, -e[2, 1], 0.0], dtype=complex) / e[2, 2]
    ex_row = np.array([1, 0, 0, 0], dtype=complex)
    ey_row = np.array([0, 0, 1, 0], dtype=complex)
    m = np.zeros((4, 4), dtype=complex)
    m[0] = np.array([0, 1, 0, 0]) + xi * ez                       # Ex' = i(Hy + xi Ez)
    m[1] = e[0, 0] * ex_row + e[0, 1] * ey_row + e[0, 2] * ez     # Hy' = i (eps E)_x
    m[2] = np.array([0, 0, 0, -1])                                # Ey' = -i Hx
    m[3] = -(e[1, 0] * ex_row + e[1, 1] * ey_row + e[1, 2] * ez) + xi**2 * ey_row
    return m


def _half_space_modes(index, xi):
    n = complex(index)
    q = np.sqrt(n * n - xi * xi + 0j)
    if q.imag < 0 or (q.imag == 0 and q.real < 0):
        q = -q
    modes = np.array([
        [q / n, n, 0, 0],    # p forward
        [0, 0, 1, -q],       # s forward
        [q / n, -n, 0, 0],   # p backward
        [0, 0, 1, q],        # s backward
    ], dtype=complex).T
    return modes, q


def _stack_matrix(layers, k0, xi):
    p = np.eye(4, dtype=complex)
    for layer in layers:
        p = expm(1j * k0 * layer.thickness * system_matrix(layer.epsilon, xi)) @ p
    return p


@dataclass(frozen=True)
class StackResponse:
    r: np.ndarray            # 2x2 Jones reflection, columns = incident (p, s), rows = (p, s)
    t: np.ndarray            # 2x2 Jones transmission
    reflectance: np.ndarray  # 2x2 power fractions |r|^2 (incident column -> outgoing row)
    transmittance: np.ndarray


def solve(layers, frequency, *, theta=0.0, n_in=1.0, n_out=1.0):
    """Jones r, t and power fractions for p/s incidence at angle theta [rad]."""
    k0 = 2 * np.pi * frequency / C0
    xi = complex(n_in) * np.sin(theta)
    f_in, q_in = _half_space_modes(n_in, xi)
    f_out, q_out = _half_space_modes(n_out, xi)
    p = _stack_matrix(layers, k0, xi)
    # f_out @ (t_p, t_s, 0, 0) = p @ f_in @ (inc_p, inc_s, r_p, r_s)
    lhs = np.zeros((4, 4), dtype=complex)
    lhs[:, :2] = f_out[:, :2]
    lhs[:, 2:] = -(p @ f_in[:, 2:])
    rhs = p @ f_in[:, :2]
    solution = np.linalg.solve(lhs, rhs)
    t, r = solution[:2], solution[2:]
    reflectance = np.abs(r) ** 2
    transmittance = np.abs(t) ** 2 * q_out.real / q_in.real
    return StackResponse(r=r, t=t, reflectance=reflectance, transmittance=transmittance)


def fields(layers, frequency, z, *, z0=0.0, jones=(1.0, 0.0), theta=0.0, n_in=1.0, n_out=1.0):
    """Total E (x, y, z) and H~ = eta0 H at x = 0 and positions z.

    The stack occupies [z0, z0 + sum(thickness)]. ``jones`` is the incident (p, s)
    electric amplitude at z0; at normal incidence p = x and s = y.
    """
    k0 = 2 * np.pi * frequency / C0
    xi = complex(n_in) * np.sin(theta)
    response = solve(layers, frequency, theta=theta, n_in=n_in, n_out=n_out)
    jones = np.asarray(jones, dtype=complex)
    f_in, q_in = _half_space_modes(n_in, xi)
    f_out, q_out = _half_space_modes(n_out, xi)
    reflected = response.r @ jones
    transmitted = response.t @ jones
    psi0 = f_in @ np.concatenate((jones, reflected))
    bounds = [z0]
    for layer in layers:
        bounds.append(bounds[-1] + layer.thickness)
    z = np.asarray(z, dtype=float)
    psi = np.zeros((z.size, 4), dtype=complex)
    ez = np.zeros(z.size, dtype=complex)
    for i, zi in enumerate(z.ravel()):
        if zi <= z0:
            dz = zi - z0
            psi[i] = f_in @ np.concatenate((jones * np.exp(1j * k0 * q_in * dz),
                                            reflected * np.exp(-1j * k0 * q_in * dz)))
            eps = tensor(complex(n_in) ** 2)
        elif zi >= bounds[-1]:
            psi[i] = f_out[:, :2] @ (transmitted * np.exp(1j * k0 * q_out * (zi - bounds[-1])))
            eps = tensor(complex(n_out) ** 2)
        else:
            state = psi0
            for layer, low, high in zip(layers, bounds[:-1], bounds[1:]):
                step = min(zi, high) - low
                state = expm(1j * k0 * step * system_matrix(layer.epsilon, xi)) @ state
                if zi <= high:
                    eps = tensor(layer.epsilon)
                    break
            psi[i] = state
        ez[i] = (-xi * psi[i, 1] - eps[2, 0] * psi[i, 0] - eps[2, 1] * psi[i, 2]) / eps[2, 2]
    electric = np.stack((psi[:, 0], psi[:, 2], ez), axis=-1)
    magnetic = np.stack((psi[:, 3], psi[:, 1], xi * psi[:, 2]), axis=-1)
    return electric.reshape(z.shape + (3,)), magnetic.reshape(z.shape + (3,))
