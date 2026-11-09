"""Stretched-coordinate PML for the FDFD Yee operator.

The FDFD absorber is the frequency-domain form of the FDTD CPML in
``fdtd/boundary/cpml.py``: one ``BoundarySpec`` describes the same graded
absorber in both solvers. For each PML face of an axis, the layer occupies
the ``num_layers`` cells that scene preparation appends outside the physical
domain. With ``u`` the normalized physical depth (0 at the physical-domain
interface, 1 at the outer wall, so graded and nonuniform cells grade by
distance rather than by index):

    sigma(u) = sigma_max u^m
    kappa(u) = 1 + (kappa_max - 1) u^m
    alpha(u) = alpha_max (1 - u)            (only inside the layer)
    sigma_max = -(m + 1) ln(R) / (2 eta0 d) * scale

where ``d`` is the physical layer thickness of that face and ``m``, ``R``,
``kappa_max`` and ``alpha_max`` are the FDTD defaults
(``DEFAULT_CPML_CONFIG``: cubic grading, R = 1e-6, kappa = 1, alpha = 0).
``scale`` is the FDTD reading of ``BoundarySpec.strength``: a value in
(0, 100] multiplies ``sigma_max``; 0 and any value above 100 (including the
``BoundarySpec.pml`` default 1e6) leave the nominal ``sigma_max`` unchanged.

Under the solver's exp(-i omega t) convention the stretch factor is

    s(u) = kappa + sigma / (alpha - i omega eps0)
         = kappa + i sigma / (omega eps0)    for alpha = 0,

so the absorption per wavelength is frequency independent and the
continuous-layer normal-incidence reflection is ``R ** scale``.

Yee placement: a derivative producing an E component along an axis is
evaluated at that axis' primal nodes, a derivative producing an H component
at the half nodes. Both samples come from the same continuous profile at
their own physical coordinates, matching the FDTD E/H profiles.
"""
from __future__ import annotations

import math

import numpy as np

from ..fdtd.boundary.cpml import DEFAULT_CPML_CONFIG, _legacy_strength_scale


def _face_depth(positions, nodes, layers, *, low):
    """Normalized depth into one face's layer (0 outside, 1 at the wall)."""
    if low:
        interface = float(nodes[layers])
        return np.clip((interface - positions) / (interface - float(nodes[0])), 0.0, 1.0), interface - float(nodes[0])
    interface = float(nodes[-1 - layers])
    return np.clip((positions - interface) / (float(nodes[-1]) - interface), 0.0, 1.0), float(nodes[-1]) - interface


def stretch_profile_1d(positions, nodes, low_layers, high_layers, *, omega, eps0, eta0, strength,
                       config=DEFAULT_CPML_CONFIG):
    """Complex stretch factors s at ``positions`` (float64 host coordinates).

    ``nodes`` are the axis' primal nodes including the appended PML cells;
    ``low_layers`` / ``high_layers`` are the PML cell counts on each face
    (0 for a non-PML face). Returns a complex128 host array.
    """
    positions = np.asarray(positions, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    order = float(config["grading_order"])
    kappa_max = float(config["kappa_max"])
    alpha_max = float(config["alpha_max"])
    numerator = -(order + 1.0) * math.log(float(config["reflection"])) / (2.0 * float(eta0))
    numerator *= _legacy_strength_scale(strength)

    depth = np.zeros_like(positions)
    sigma_peak = np.zeros_like(positions)
    for layers, low in ((int(low_layers), True), (int(high_layers), False)):
        if layers <= 0:
            continue
        face_depth, thickness = _face_depth(positions, nodes, layers, low=low)
        deeper = face_depth > depth
        depth = np.where(deeper, face_depth, depth)
        sigma_peak = np.where(deeper, numerator / thickness, sigma_peak)

    graded = depth ** order
    sigma = sigma_peak * graded
    kappa = 1.0 + (kappa_max - 1.0) * graded
    alpha = np.where(depth > 0.0, alpha_max * (1.0 - depth), 0.0)
    return kappa + sigma / (alpha - 1j * float(omega) * float(eps0))
