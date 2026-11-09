"""Exact exterior electric point-dipole incident fields for scalar PMCHWT.

A ``PointDipole`` is an ideal electric current moment ``Il`` [A*m]: the source
current density is ``J(r) = Il * delta(r - r0)`` in
``curl H = -i*omega*epsilon*E + J`` under ``exp(-i omega t)``. With
``g = exp(i k R)/(4 pi R)``, ``R = |r - r0|`` and ``n = (r - r0)/R``:

    E = i*omega*mu*g*[(1 + i/(kR) - 1/(kR)^2) Il - (1 + 3i/(kR) - 3/(kR)^2) n (n.Il)]
    H = g*(i*k - 1/R) * (n x Il)

including every near-field term. The field is a source-free Maxwell solution
inside every body when the dipole lies outside all of them, which is the
property the analytic incident-trace lifting requires.
"""
from __future__ import annotations

import math

import torch

from ..constants import complex_dtype_for

# A dipole must stay at least this many panel radii (maximum centroid-to-vertex
# distance) away from every surface triangle. With the lowest source-panel
# rule order admitted below, the measured worst-panel relative error of the
# incident-trace quadrature at this limit is below 2e-4 (order 4) and 1e-7
# (order 6) for both normal and tangential dipoles over panel centroids, edges
# and vertices. Closer dipoles need local mesh refinement, not silent
# under-integration.
DIPOLE_CLEARANCE_PANEL_RADII = 3.0
DIPOLE_MINIMUM_SOURCE_ORDER = 4
CLEARANCE_RULE = 'point_dipole_panel_clearance_v1'


def point_dipole_fields(points, position, moment, medium):
    """Return exact E [V/m] and H [A/m] of one electric current moment.

    ``points`` is a real tensor [..., 3] in metres; ``position`` three
    coordinates; ``moment`` three complex current-moment components [A*m].
    The medium supplies ``k``, ``omega`` and ``mu``. Points at the dipole
    position are rejected because the field is singular there.
    """
    if (not isinstance(points, torch.Tensor) or points.dtype not in (torch.float32, torch.float64)
            or points.shape[-1:] != (3,)):
        raise ValueError('Point-dipole query points must be real float32/float64 with shape (..., 3).')
    dtype = complex_dtype_for(points.dtype)
    origin = points.new_tensor(tuple(float(x) for x in position))
    current = torch.tensor(tuple(complex(x) for x in moment), dtype=dtype, device=points.device)
    delta = points-origin
    radius = torch.linalg.vector_norm(delta, dim=-1)
    if bool((radius == 0).any()):
        raise ValueError('The point-dipole field is singular at the dipole position; move the query point.')
    k = complex(medium.k)
    unit = (delta/radius[..., None]).to(dtype)
    kr = k*radius.to(dtype)
    green = torch.exp(1j*kr)/(4*math.pi*radius)
    first = 1+1j/kr-1/kr**2
    second = 1+3j/kr-3/kr**2
    projection = (unit*current).sum(-1, keepdim=True)
    electric = 1j*medium.omega*complex(medium.mu)*green[..., None]*(
        first[..., None]*current-second[..., None]*unit*projection)
    magnetic = (green*(1j*k-1/radius))[..., None]*torch.linalg.cross(unit, current.expand_as(unit))
    return electric, magnetic


def validate_point_dipole_clearance(position, triangles, *, body_name=None):
    """Reject a dipole inside, on, or too close to one closed triangulated body.

    ``triangles`` is [T, 3, 3] in metres. Clearance is local: each panel only
    constrains the dipole by its own radius, so a fine local mesh admits a
    closer dipole. Returns the smallest distance-to-radius ratio.
    """
    from .fields import _solid_angle_inside, _triangle_distance
    label = 'a body' if body_name is None else f'body {body_name!r}'
    point = triangles.new_tensor(tuple(float(x) for x in position))[None]
    if not bool(torch.isfinite(point).all()):
        raise ValueError('PointDipole position must be finite.')
    distance = _triangle_distance(point, triangles)[0]
    radius = torch.linalg.vector_norm(triangles-triangles.mean(1)[:, None], dim=-1).amax(-1)
    ratio = distance/radius
    worst = int(torch.argmin(ratio))
    smallest = float(ratio[worst])
    vertices = triangles.reshape(-1, 3)
    tolerance = 128*torch.finfo(triangles.dtype).eps*float(torch.linalg.vector_norm(vertices.amax(0)-vertices.amin(0)))
    if float(distance.min()) <= tolerance:
        raise ValueError(f'PointDipole at {tuple(position)} lies on the surface of {label}; core MoM supports '
                         'exterior dipoles with panel clearance only.')
    if bool(_solid_angle_inside(point, triangles)[0]):
        raise ValueError(f'PointDipole at {tuple(position)} lies inside {label}; core MoM supports '
                         'exterior dipoles only, because the incident field must be source-free inside '
                         'every homogeneous body.')
    if smallest < DIPOLE_CLEARANCE_PANEL_RADII:
        raise ValueError(
            f'PointDipole at {tuple(position)} is {float(distance[worst]):.6g} m from a surface panel of '
            f'{label} with panel radius {float(radius[worst]):.6g} m (ratio {smallest:.3g}); the exterior '
            f'source needs at least {DIPOLE_CLEARANCE_PANEL_RADII:g} panel radii of clearance from every '
            'panel so its singular incident trace is integrated accurately. Refine the mesh locally '
            'or move the dipole.')
    return smallest
