"""Polarized (Kottke) subpixel-averaging math for the material compiler.

This module holds the pure formulas behind polarized subpixel averaging:
interface-normal estimation from signed-distance / field gradients, the
normal-projection per-axis blend, and the reconstruction of node components
from separately accumulated arithmetic and reciprocal subcell means. The
material compiler (:mod:`witwin.maxwell.compiler.materials`) keeps grid and
structure orchestration and calls into this module.
"""

from __future__ import annotations

import torch

_AXES = ("x", "y", "z")

_NORMAL_EPS = 1e-24
_HARMONIC_EPS = 1e-12


def _field_gradients(scene, field, coords=None):
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    x = xx[:, 0, 0]
    y = yy[0, :, 0]
    z = zz[0, 0, :]
    grads = []
    for dim, coord in enumerate((x, y, z)):
        if field.shape[dim] < 2:
            grads.append(torch.zeros_like(field))
        else:
            grads.append(torch.gradient(field, spacing=(coord,), dim=dim)[0])
    return dict(zip(_AXES, grads, strict=True))


def _normalized_field_gradients(scene, field, coords=None):
    grads = _field_gradients(scene, field, coords=coords)
    gx, gy, gz = (grads[axis] for axis in _AXES)
    # Floor is applied inside the sqrt (squared scale ``_NORMAL_EPS``) so the
    # sqrt's own backward stays finite at degenerate nodes where the summed
    # squared gradient is exactly zero (medial-axis / symmetric-center nodes).
    # A floor added after the sqrt would leave d(sqrt)/d(.) = inf there, and the
    # vanishing upstream gradient would evaluate 0*inf = NaN, poisoning the whole
    # geometry-gradient tensor through the sum reduction.
    mag = torch.sqrt(gx * gx + gy * gy + gz * gz + _NORMAL_EPS)
    inv = 1.0 / mag
    return {"x": gx * inv, "y": gy * inv, "z": gz * inv}


def _interface_normals(scene, geometry, coords=None):
    """Unit outward interface normals per node from the signed-distance field.

    The normal is the gradient of the per-structure signed-distance field on the
    node grid, evaluated with true (possibly graded) spacings via central finite
    differences. SDFs are eikonal so ``|grad| ~ 1`` on the interface band; deep
    interior / medial-axis nodes have ``|grad| ~ 0`` and yield ``n ~ 0`` after the
    floored normalization, which is harmless because the polarized correction there
    is weighted by the vanishing ``(eps_arith - eps_harm)`` term. Differentiable in
    geometry parameters through ``signed_distance``.
    """
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    return _normalized_field_gradients(
        scene,
        geometry.signed_distance(xx, yy, zz),
        coords=(xx, yy, zz),
    )


def _blend_material_polarized(tensor, occupancy, normal_axis, *, value):
    """Normal-projection (Kottke) per-axis blend of a background field with ``value``.

    ``tensor`` is the running per-axis accumulated background permittivity/permeability,
    ``occupancy`` this structure's soft fill, and ``normal_axis`` the interface-normal
    component for this axis. The harmonic (series) mean is weighted by ``n_a^2`` along
    the interface normal and the arithmetic (parallel) mean by ``1 - n_a^2`` tangentially.
    Reduces exactly to the arithmetic blend when ``n_a = 0``.
    """
    value_tensor = torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)
    arithmetic = (1.0 - occupancy) * tensor + occupancy * value_tensor
    harmonic = 1.0 / (
        (1.0 - occupancy) / (tensor + _HARMONIC_EPS)
        + occupancy / (value_tensor + _HARMONIC_EPS)
    )
    weight = normal_axis * normal_axis
    return (1.0 - weight) * arithmetic + weight * harmonic


def _reconstruct_sampled_polarized_components(scene, arithmetic, inverse):
    """Combine subcell arithmetic and reciprocal means along the material normal.

    Averaging an already blended harmonic value at every subcell sample collapses
    toward the arithmetic mean when those samples are nearly binary. Integrating
    the material and its reciprocal separately preserves the finite-volume series
    mean before the normal projection is applied.
    """
    squared_gradients = {
        axis: torch.zeros_like(arithmetic[axis]) for axis in _AXES
    }
    for component in arithmetic.values():
        for axis, gradient in _field_gradients(scene, component).items():
            squared_gradients[axis] += gradient * gradient
    gradient_norm_squared = sum(squared_gradients.values()) + _NORMAL_EPS
    components = {}
    for axis in _AXES:
        weight = squared_gradients[axis] / gradient_norm_squared
        harmonic = 1.0 / (inverse[axis] + _HARMONIC_EPS)
        components[axis] = (1.0 - weight) * arithmetic[axis] + weight * harmonic
    return components
