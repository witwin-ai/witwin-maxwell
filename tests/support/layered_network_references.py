"""Closed-form circuit references for the layered MPIE network tests."""

import math

from scipy.special import ellipk


MU0 = 4e-7 * math.pi
ETA0 = 376.730313668


def square_loop_inductance(side, strip_width):
    """High-frequency inductance of a square loop of thin PEC strip (centreline side ``side``).

    Partial inductances of round wire with surface current: self term
    (mu0 a / 2 pi)(ln(2a/r) - 1), opposite-side mutual (mu0 a / 2 pi)(ln(1 + sqrt 2) - sqrt 2 + 1),
    perpendicular sides do not couple, L = 4 (L_self - M) = (2 mu0 a / pi)(ln(a/r) - 0.774)
    (Grover, Inductance Calculations, square of round wire). A thin PEC strip of width w has
    the equivalent radius w/4 (conformal map of the strip onto a circle).
    """
    radius = strip_width / 4
    constant = 1 + math.log(1 + math.sqrt(2)) - math.sqrt(2) + 1 - math.log(2)
    return 2 * MU0 * side / math.pi * (math.log(side / radius) - constant)


def square_loop_inductance_uniform(side, strip_width, thickness):
    """Square-loop inductance for a current spread uniformly over a thin strip cross-section.

    Same partial-inductance form as square_loop_inductance, with the strip's geometric mean
    distance 0.2235 (w + t) as the radius (Grover). This is the limit when the sheet resistance
    dominates the reactance, so the current no longer crowds at the strip edges.
    """
    radius = 0.2235 * (strip_width + thickness)
    constant = 1 + math.log(1 + math.sqrt(2)) - math.sqrt(2) + 1 - math.log(2)
    return 2 * MU0 * side / math.pi * (math.log(side / radius) - constant)


def small_loop_radiation_resistance(area, wavelength):
    """R = 320 pi^4 (A / lambda^2)^2 for an electrically small loop of area A."""
    return 320 * math.pi**4 * (area / wavelength**2) ** 2


def coplanar_strips_quasi_static(width, gap, eps_r):
    """Zero-thickness coplanar strips on the interface of a dielectric half-space and air.

    eps_eff = (eps_r + 1) / 2 and Z0 = (eta0 / sqrt(eps_eff)) K(k) / K(k'), k = s / (s + 2 w)
    (Gupta, Garg, Bahl and Bhartia, Microstrip Lines and Slotlines, coplanar strips).
    scipy's ellipk takes the parameter m = k^2.
    """
    eps_eff = (eps_r + 1) / 2
    k = gap / (gap + 2 * width)
    return eps_eff, ETA0 / math.sqrt(eps_eff) * ellipk(k * k) / ellipk(1 - k * k)
