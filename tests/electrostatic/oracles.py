"""Test-only analytic oracles and complement geometries for electrostatics.

These helpers exist purely to build closed-form validation cases (an outer
grounded shell is the complement of a sphere/cylinder, which the shipped
geometry primitives do not express directly). They are never imported by the
production package.
"""

from __future__ import annotations

import math

import torch

from witwin.core.material import VACUUM_PERMITTIVITY as EPS0


class OutsideSphere:
    """Conductor occupying ``r >= radius`` (the complement of a solid sphere)."""

    def __init__(self, radius: float):
        self.radius = float(radius)

    def signed_distance(self, x, y, z):
        return self.radius - torch.sqrt(x * x + y * y + z * z)


class OutsideCylinder:
    """Conductor occupying ``sqrt(x^2 + y^2) >= radius`` (z-axis cylinder complement)."""

    def __init__(self, radius: float):
        self.radius = float(radius)

    def signed_distance(self, x, y, z):
        return self.radius - torch.sqrt(x * x + y * y)


def concentric_sphere_capacitance(a: float, b: float, eps_r: float = 1.0) -> float:
    return 4.0 * math.pi * EPS0 * eps_r / (1.0 / a - 1.0 / b)


def coaxial_capacitance_per_length(a: float, b: float, eps_r: float = 1.0) -> float:
    return 2.0 * math.pi * EPS0 * eps_r / math.log(b / a)


def parallel_plate_capacitance(area: float, gap: float, eps_r: float = 1.0) -> float:
    return EPS0 * eps_r * area / gap
