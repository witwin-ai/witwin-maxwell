"""Closed-form coupled-line references for the TerminalWavePort validation.

* Hammerstad-Jensen zero-thickness microstrip (static ``eps_eff`` and ``Z0``).
* Kirschning-Jansen static even/odd coupled microstrip (zero thickness); quoted
  accuracy about 1 % for 0.1 <= W/h <= 10, 0.1 <= s/h <= 10, eps_r <= 18.
* Cohn's exact conformal-mapping even/odd impedances of a zero-thickness
  edge-coupled stripline centred between infinite grounds.
"""
from __future__ import annotations

import math

from scipy.special import ellipk

ETA_0 = 376.730313412


def _a(v):
    return 1.0 + math.log((v**4 + (v / 52.0) ** 2) / (v**4 + 0.432)) / 49.0 + math.log(1.0 + (v / 18.1) ** 3) / 18.7


def _b(eps_r):
    return 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053


def hammerstad_jensen_microstrip(u, eps_r):
    """Return ``(eps_eff, Z0)`` of a zero-thickness microstrip with ``u = W/h``."""
    eps_eff = 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * (1.0 + 10.0 / u) ** (-_a(u) * _b(eps_r))
    f = 6.0 + (2.0 * math.pi - 6.0) * math.exp(-((30.666 / u) ** 0.7528))
    z_air = ETA_0 / (2.0 * math.pi) * math.log(f / u + math.sqrt(1.0 + (2.0 / u) ** 2))
    return eps_eff, z_air / math.sqrt(eps_eff)


def kirschning_jansen_coupled_microstrip(u, g, eps_r):
    """Return ``(eps_even, eps_odd, Z_even, Z_odd)`` for ``u = W/h`` and ``g = s/h``."""
    eps_single, z_single = hammerstad_jensen_microstrip(u, eps_r)
    v = u * (20.0 + g**2) / (10.0 + g**2) + g * math.exp(-g)
    eps_even = 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * (1.0 + 10.0 / v) ** (-_a(v) * _b(eps_r))
    a_o = 0.7287 * (eps_single - 0.5 * (eps_r + 1.0)) * (1.0 - math.exp(-0.179 * u))
    b_o = 0.747 * eps_r / (0.15 + eps_r)
    c_o = b_o - (b_o - 0.207) * math.exp(-0.414 * u)
    d_o = 0.593 + 0.694 * math.exp(-0.562 * u)
    eps_odd = (0.5 * (eps_r + 1.0) + a_o - eps_single) * math.exp(-c_o * g**d_o) + eps_single
    q1 = 0.8695 * u**0.194
    q2 = 1.0 + 0.7519 * g + 0.189 * g**2.31
    q3 = 0.1975 + (16.6 + (8.4 / g) ** 6) ** -0.387 + math.log(g**10 / (1.0 + (g / 3.4) ** 10)) / 241.0
    q4 = (2.0 * q1 / q2) / (math.exp(-g) * u**q3 + (2.0 - math.exp(-g)) * u**-q3)
    q5 = 1.794 + 1.14 * math.log(1.0 + 0.638 / (g + 0.517 * g**2.43))
    q6 = 0.2305 + math.log(g**10 / (1.0 + (g / 5.8) ** 10)) / 281.3 + math.log(1.0 + 0.598 * g**1.154) / 5.1
    q7 = (10.0 + 190.0 * g**2) / (1.0 + 82.3 * g**3)
    q8 = math.exp(-6.5 - 0.95 * math.log(g) - (g / 0.15) ** 5)
    q9 = math.log(q7) * (q8 + 1.0 / 16.5)
    q10 = (q2 * q4 - q5 * math.exp(math.log(u) * q6 * u**-q9)) / q2
    scale = math.sqrt(eps_single) * z_single / ETA_0
    z_even = z_single * math.sqrt(eps_single / eps_even) / (1.0 - q4 * scale)
    z_odd = z_single * math.sqrt(eps_single / eps_odd) / (1.0 - q10 * scale)
    return eps_even, eps_odd, z_even, z_odd


def cohn_coupled_stripline(width, gap, ground_spacing, eps_r):
    """Return exact ``(Z_even, Z_odd)`` of zero-thickness centred coupled striplines."""
    first = math.tanh(math.pi * width / (2.0 * ground_spacing))
    second = math.pi * (width + gap) / (2.0 * ground_spacing)
    k_even = first * math.tanh(second)
    k_odd = first / math.tanh(second)

    def ratio(k):
        return float(ellipk(1.0 - k * k) / ellipk(k * k))

    scale = 30.0 * math.pi / math.sqrt(eps_r)
    return scale * ratio(k_even), scale * ratio(k_odd)
