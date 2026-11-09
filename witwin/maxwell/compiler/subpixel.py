"""Polarized (Kottke) subpixel-averaging math for the material compiler.

This module holds the pure formulas behind polarized subpixel averaging:
interface-normal estimation from signed-distance / field gradients, the
normal-projection per-axis blend, the reconstruction of node components
from separately accumulated arithmetic and reciprocal subcell means (inverse
Kottke tensor diagonal), and the sub-sample coverage fill fraction. The
material compiler (:mod:`witwin.maxwell.compiler.materials`) keeps grid and
structure orchestration and calls into this module.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from witwin.core import Mesh

_AXES = ("x", "y", "z")

_NORMAL_EPS = 1e-24
_HARMONIC_EPS = 1e-12
_MESH_FILLS = ContextVar("maxwell_mesh_subsample_fills", default=None)


@contextmanager
def _mesh_fill_cache(scene, samples, include_edges):
    """Reuse fixed-mesh coverage at bit-identical uniform-grid query points."""
    from .materials import _axis_sample_offsets, _edge_axis_base_1d, _edge_axis_offsets

    axes = None
    if (samples != (1, 1, 1) and not (scene.grid.is_custom or scene.grid.is_auto)
            and all(scene.boundary.axis_kind(a) not in ("periodic", "bloch") for a in _AXES)):
        axes = []
        for axis, count in zip(_AXES, samples):
            values = [getattr(scene, axis) + offset
                      for offset in _axis_sample_offsets(scene, axis, count)]
            if all(n % 2 for n in samples):
                # The all-zero node sample bypasses addition, including at -0.
                values.append(getattr(scene, axis))
            if include_edges:
                base = _edge_axis_base_1d(scene, axis, True)
                values.extend(base + offset for offset in _edge_axis_offsets(scene, axis, count, True))
            # Integer views distinguish signed zero and all rounding differences.
            coord = torch.cat(values) - 0.0
            bits = torch.int32 if coord.dtype == torch.float32 else torch.int64
            axes.append(torch.unique(coord.view(bits), sorted=True))
        # Bound retained coverage to 256 MiB per compile; larger charts retain the
        # existing packed-query path rather than allocating a dense point cache.
        if math.prod(axis.numel() for axis in axes) * scene.x.element_size() > 256 * 1024 * 1024:
            axes = None
    token = _MESH_FILLS.set(None if axes is None else (axes, {}))
    try:
        yield
    finally:
        _MESH_FILLS.reset(token)


def _cached_mesh_subsample_fill(geometry, coords, spacing, delta):
    context = _MESH_FILLS.get()
    if context is None or any(torch.is_tensor(h) for h in spacing):
        return _mesh_subsample_fill(geometry, *coords, spacing, delta)
    axes, fields = context
    key = (id(geometry), spacing, coords[0].dtype, coords[0].device)
    if key not in fields:
        size = math.prod(a.numel() for a in axes) * coords[0].element_size()
        used = fields.get("bytes", 0)
        fields[key] = (torch.full(tuple(a.numel() for a in axes), -1.0,
                                  device=coords[0].device, dtype=coords[0].dtype)
                       if used + size <= 256 * 1024 * 1024 else None)
        if fields[key] is not None:
            fields["bytes"] = used + size
    field = fields[key]
    if field is None:
        return _mesh_subsample_fill(geometry, *coords, spacing, delta)
    lines = (coords[0][:, 0, 0], coords[1][0, :, 0], coords[2][0, 0, :])
    keys = [line.contiguous().view(a.dtype) for a, line in zip(axes, lines)]
    indices = [torch.searchsorted(a, key) for a, key in zip(axes, keys)]
    # Only exact bit-pattern hits on a tensor-product query grid may reuse the chart;
    # anything else is queried directly.
    shape = torch.broadcast_shapes(*(coord.shape for coord in coords))
    if not (all(torch.equal(coord.expand(shape), line.reshape([-1 if d == axis else 1 for d in range(3)]).expand(shape))
                for axis, (coord, line) in enumerate(zip(coords, lines)))
            and all(bool((i < a.numel()).all()) and torch.equal(a[i.clamp(max=a.numel() - 1)], key)
                    for a, i, key in zip(axes, indices, keys))):
        return _mesh_subsample_fill(geometry, *coords, spacing, delta)
    index = (indices[0][:, None, None], indices[1][None, :, None], indices[2][None, None, :])
    fill = field[index]
    missing = fill < 0
    if missing.any():
        # Packing does not change the scalar uniform spacing or the per-point
        # coverage/SDF reductions. Existing AABB and band pruning still apply.
        packed = tuple(coord[missing].reshape(-1, 1, 1) for coord in coords)
        fill[missing] = _mesh_subsample_fill(geometry, *packed, spacing, delta).reshape(-1)
        field[index] = fill
    return fill


def _field_gradients(scene, field, coords=None, *, periodic_nodes=False):
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    x = xx[:, 0, 0]
    y = yy[0, :, 0]
    z = zz[0, 0, :]
    grads = []
    for dim, coord in enumerate((x, y, z)):
        if field.shape[dim] < 2:
            grads.append(torch.zeros_like(field))
        else:
            gradient = torch.gradient(field, spacing=(coord,), dim=dim)[0]
            if periodic_nodes and scene.boundary.axis_kind(_AXES[dim]) in ("periodic", "bloch"):
                # Endpoint nodes denote the same physical point. One-sided
                # normals would give that shared material tensor two values and
                # break reciprocity of the volume-weighted Yee couplings.
                if field.shape[dim] == 2:
                    seam = torch.zeros_like(field.narrow(dim, 0, 1))
                else:
                    previous = field.narrow(dim, field.shape[dim] - 2, 1)
                    center = field.narrow(dim, 0, 1)
                    following = field.narrow(dim, 1, 1)
                    left, right = coord[-1] - coord[-2], coord[1] - coord[0]
                    seam = (-right / (left * (left + right)) * previous
                            + (right - left) / (left * right) * center
                            + left / (right * (left + right)) * following)
                gradient = torch.cat((seam, gradient.narrow(dim, 1, field.shape[dim] - 2), seam), dim=dim)
            grads.append(gradient)
    return dict(zip(_AXES, grads, strict=True))


def _normalized_field_gradients(scene, field, coords=None, *, periodic_nodes=False):
    grads = _field_gradients(scene, field, coords=coords, periodic_nodes=periodic_nodes)
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


# --- Sub-sample coverage fill and inverse-tensor polarized blend --------------
#
# A binary (or nearly binary) occupancy per sub-sample quantizes the cell fill
# fraction to multiples of 1/s, which caps every averaged method at first-order
# convergence, and the sharp dielectric tanh still leaks into sub-samples that
# sit half a sub-sample spacing from a grid-aligned face. The coverage fill below
# replaces the per-sub-sample occupancy with a linear ramp of the signed distance
# across one sub-sample spacing measured along the interface normal. The
# midpoint sum of that ramp over the sub-sample lattice equals the exact
# volume fraction of an axis-aligned planar interface for any interface
# position, and is exactly 0 or 1 for a face lying on a grid plane.

_FILL_SNAP = 1.0e-3
_HOMOGENEOUS_CONTRAST = 1.0e-5


def _broadcast_spacing(value, axis_index, shape):
    """Sub-sample spacing along one axis, broadcast to a sample-coordinate grid.

    ``value`` is a float (uniform axis) or a 1D per-sample-line tensor (graded
    axis). A tensor whose length does not match the grid along its axis (a
    cropped or periodic-plane bridge evaluation) falls back to its minimum.
    """
    if not torch.is_tensor(value):
        return float(value)
    if value.numel() == shape[axis_index]:
        view = [1, 1, 1]
        view[axis_index] = value.numel()
        return value.reshape(view)
    return value.min()


def _subsample_fill_fraction(geometry, xx, yy, zz, spacing):
    """Linear coverage ramp ``clamp(1/2 - (d + c) / w, 0, 1)`` at sub-sample points.

    ``spacing`` holds the sub-sample spacing per axis (float, or a per-line
    tensor on graded axes). The ramp width ``w`` is that spacing projected on
    the local interface normal (``sum_a |n_a| h_a``). Normal and Laplacian come
    from central differences of the signed distance at a quarter of the
    smallest spacing. On a curved surface the ramp volume exceeds the enclosed
    volume by ``w^2 (k1 + k2) / 24`` per unit area; the shift
    ``c = w^2 lap(d) / 24`` (``lap(d) = k1 + k2`` for a signed distance,
    clamped to a quarter ramp so SDF kinks cannot dominate) removes it, and is
    zero for planar faces. Values within ``_FILL_SNAP`` of 0 or 1 snap to the
    exact bound so float32 coordinate rounding cannot dilute a cell bounded by
    grid-aligned faces.
    """
    shape = xx.shape
    hx, hy, hz = (_broadcast_spacing(value, index, shape) for index, value in enumerate(spacing))
    smallest = min(float(h.min()) if torch.is_tensor(h) else h for h in (hx, hy, hz))
    delta = 0.25 * smallest
    if (isinstance(geometry, Mesh)
            and (geometry.fill_mode == "solid" or geometry.is_watertight)
            and geometry.fill_mode != "surface" and not geometry.has_trainable_geometry()
            and not any(coord.requires_grad for coord in (xx, yy, zz))):
        return _cached_mesh_subsample_fill(geometry, (xx, yy, zz), (hx, hy, hz), delta)
    distance = geometry.signed_distance(xx, yy, zz)
    plus = (
        geometry.signed_distance(xx + delta, yy, zz),
        geometry.signed_distance(xx, yy + delta, zz),
        geometry.signed_distance(xx, yy, zz + delta),
    )
    minus = (
        geometry.signed_distance(xx - delta, yy, zz),
        geometry.signed_distance(xx, yy - delta, zz),
        geometry.signed_distance(xx, yy, zz - delta),
    )
    return _coverage_fill(distance, plus, minus, (hx, hy, hz), delta)


def _coverage_fill(distance, plus, minus, spacing, delta):
    hx, hy, hz = spacing
    gx, gy, gz = ((p - m) for p, m in zip(plus, minus))
    norm = torch.sqrt(gx * gx + gy * gy + gz * gz + (_NORMAL_EPS * delta * delta))
    width = (gx.abs() * hx + gy.abs() * hy + gz.abs() * hz) / norm
    local_min = torch.minimum(torch.minimum(torch.as_tensor(hx, dtype=width.dtype, device=width.device),
                                            torch.as_tensor(hy, dtype=width.dtype, device=width.device)),
                              torch.as_tensor(hz, dtype=width.dtype, device=width.device))
    width = torch.maximum(width, local_min)
    laplacian = sum(p + m - 2.0 * distance for p, m in zip(plus, minus)) / (delta * delta)
    shift = torch.clamp(width * width * laplacian / 24.0, min=-0.25 * width, max=0.25 * width)
    fill = torch.clamp(0.5 - (distance + shift) / width, 0.0, 1.0)
    fill = torch.where(fill > 1.0 - _FILL_SNAP, torch.ones_like(fill), fill)
    return torch.where(fill < _FILL_SNAP, torch.zeros_like(fill), fill)


def _mesh_subsample_fill(geometry, xx, yy, zz, spacing, delta):
    """Pack fixed solid-mesh queries; preserve the coverage/reduction formulas."""
    maxima = [float(h.max()) if torch.is_tensor(h) else h for h in spacing]
    ramp = math.sqrt(sum(h * h for h in maxima))
    vertices, faces, _, _ = geometry._sdf_query_data(device=xx.device, dtype=xx.dtype)
    lower, upper = vertices.amin(0), vertices.amax(0)
    # Include the full ramp, curvature cap, stencil and absolute SDF smoothing.
    # Coordinates already include the actual rounded Yee/sub-sample/image offset.
    margin = 64 * torch.finfo(xx.dtype).eps * (vertices.abs().max() + ramp)
    padding = 2 * ramp + delta + 2e-6 + margin
    active = torch.ones_like(xx, dtype=torch.bool)
    closed = geometry.is_watertight
    context = _MESH_FILLS.get()
    if closed is None and context is not None:
        metadata = context[1]
        key = ("closed", id(geometry))
        if key not in metadata:
            # An oriented triangle cycle has zero winding outside its AABB.
            # Prove closure even when optional core diagnostics were disabled:
            # every directed boundary edge must cancel its reverse, exactly.
            edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).to(torch.int64)
            forward = edges[:, 0] * vertices.shape[0] + edges[:, 1]
            reverse = edges[:, 1] * vertices.shape[0] + edges[:, 0]
            metadata[key] = torch.equal(forward.sort().values, reverse.sort().values)
        closed = metadata[key]
    # Open/uncertified topology retains every center query.
    if closed:
        for axis, coord in enumerate((xx, yy, zz)):
            active &= (coord >= lower[axis] - padding) & (coord <= upper[axis] + padding)
    fill = torch.zeros_like(xx)
    coords = [coord[active].reshape(-1, 1, 1) for coord in (xx, yy, zz)]
    if coords[0].numel() == 0:
        return fill
    distance = geometry.signed_distance(*coords)
    # w <= ||h||_2 and |shift| <= w/4, so |d| > 3||h||_2/4
    # saturates exactly. Retain a rounding margin around this bound.
    band = distance.abs() <= .75 * ramp + margin
    local_fill = (distance < 0).to(distance.dtype)
    band_coords = [coord[band].reshape(-1, 1, 1) for coord in coords]
    if band_coords[0].numel():
        shifted = [[], [], []]
        for sign in (1, -1):
            for axis in range(3):
                for index, coord in enumerate(band_coords):
                    shifted[index].append(coord + sign * delta if index == axis else coord)
        # One dispatch for six point sets, with core's bounded point chunks and
        # unchanged per-point triangle reduction order. No dense all-pairs cache.
        queries = [torch.cat(parts) for parts in shifted]
        distances = geometry.signed_distance(*queries).chunk(6)
        local_spacing = tuple(torch.broadcast_to(h, xx.shape)[active].reshape(-1, 1, 1)[band].reshape(-1, 1, 1)
            if torch.is_tensor(h) else h for h in spacing)
        local_fill[band] = _coverage_fill(distance[band].reshape(-1, 1, 1),
            distances[:3], distances[3:], local_spacing, delta).reshape(-1)
    fill[active] = local_fill.reshape(-1)
    return fill


class SubsampleFillGeometry:
    """Geometry adapter whose ``to_mask`` is the sub-sample coverage ramp.

    Lets the shared occupancy path (periodic image union, periodic boundary
    plane wrap) evaluate the coverage fill without duplicating that logic.
    """

    def __init__(self, geometry, spacing):
        self.geometry = geometry
        self.spacing = tuple(spacing)

    def signed_distance(self, xx, yy, zz):
        return self.geometry.signed_distance(xx, yy, zz)

    def to_mask(self, xx, yy, zz, offset=0.0, beta=None):
        return _subsample_fill_fraction(self.geometry, xx, yy, zz, self.spacing)


def _uniform_axis_spacing(scene, axis):
    """Resolved Yee spacing, including for cropped material evaluations."""
    lo, hi = scene.domain.bounds[_AXES.index(axis)]
    span = float(hi) - float(lo)
    count = max(1, math.ceil(span / float(getattr(scene.grid, f"d{axis}"))))
    return span / count


def _subsample_spacing(scene, samples, staggered_axes=None):
    """Per-axis sub-sample spacing (``None`` for one sample per cell).

    Uniform axes give the Python float ``resolved_spacing / count``. Graded axes
    give the per-line averaging window divided by ``count``: the node dual width,
    or the primal width on an axis listed in ``staggered_axes`` (edge-native Yee
    components), matching the windows the sub-sample offsets span.
    """
    counts = tuple(int(value) for value in samples)
    if counts == (1, 1, 1):
        return None
    grid = scene.grid
    spacing = []
    for index, (axis, count) in enumerate(zip(_AXES, counts)):
        if not (grid.is_custom or grid.is_auto):
            spacing.append(_uniform_axis_spacing(scene, axis) / count)
            continue
        staggered = staggered_axes is not None and index in staggered_axes
        width64 = getattr(scene, f"d{axis}_primal64" if staggered else f"d{axis}_dual64")
        width64 = torch.as_tensor(width64, dtype=torch.float64)
        if float(width64.max() - width64.min()) <= 1.0e-12 * float(width64.max()):
            # A uniformly spaced custom axis keeps the scalar path bit for bit.
            spacing.append(float(scene.grid.min_spacing[index]) / count)
            continue
        if staggered:
            coord = getattr(scene, axis)
            width = coord[1:] - coord[:-1]
        else:
            width = width64.to(device=scene.device, dtype=scene.x.dtype)
        spacing.append(width / count)
    return tuple(spacing)


# TODO: conductivity in partially filled cells is still averaged arithmetically, which
# over-predicts absorption at lossy curved interfaces; a consistent treatment (for example
# averaging the complex admittance per Yee component) is the open follow-up.
def _inverse_diagonal_blend(arithmetic, inverse, weight):
    """Diagonal of the inverse Kottke tensor, returned as an effective permittivity.

    The Kottke tensor is ``P <1/eps>^-1 + (1 - P) <eps>`` with ``P = n n^T``; the
    Yee update needs ``E_a = (eps^-1)_aa D_a``, whose diagonal is
    ``n_a^2 <1/eps> + (1 - n_a^2) / <eps>``. ``weight`` is ``n_a^2``. For an
    axis-aligned normal this equals the diagonal of the tensor itself.
    """
    # Written as <eps> / (1 + w (<eps><1/eps> - 1)). <eps><1/eps> >= 1 with
    # equality only for a homogeneous window, so contrasts below the float32
    # accumulation noise of the reciprocal mean return <eps> exactly; a cell
    # bounded by grid-aligned faces keeps its bulk value bit for bit.
    contrast = arithmetic * inverse - 1.0
    contrast = torch.where(contrast < _HOMOGENEOUS_CONTRAST, torch.zeros_like(contrast), contrast)
    return arithmetic / (1.0 + weight * contrast)


def _reconstruct_sampled_polarized_inverse(scene, arithmetic, inverse):
    """Node reconstruction with the inverse-tensor diagonal (see ``_inverse_diagonal_blend``)."""
    squared_gradients = {axis: torch.zeros_like(arithmetic[axis]) for axis in _AXES}
    for component in arithmetic.values():
        for axis, gradient in _field_gradients(scene, component, periodic_nodes=True).items():
            squared_gradients[axis] += gradient * gradient
    gradient_norm_squared = sum(squared_gradients.values()) + _NORMAL_EPS
    return {
        axis: _inverse_diagonal_blend(arithmetic[axis], inverse[axis], squared_gradients[axis] / gradient_norm_squared)
        for axis in _AXES
    }


def _kottke_edge_diagonal(sample, offsets, axis):
    """Physical diagonal from window samples and their local first moment."""
    arithmetic = sample.mean(1)
    harmonic = 1.0 / (1.0 / sample).mean(1)
    real = sample.real if sample.is_complex() else sample
    moment = ((real - real.mean(1, keepdim=True))[..., None] * offsets).sum(1)
    norm = torch.linalg.vector_norm(moment, dim=-1)
    denominator = torch.where(norm > 0, norm.clamp_min(1e-30), torch.ones_like(norm)).square()
    weight = torch.where(norm > 0, moment[:, axis].square() / denominator,
                         torch.zeros_like(norm))
    return weight * harmonic + (1 - weight) * arithmetic


def _kottke_normal_correction(arithmetic, inverse):
    """Difference between normal and tangential eigenvalues, zero in bulk."""
    return torch.where(
        arithmetic * inverse - 1.0 < _HOMOGENEOUS_CONTRAST,
        torch.zeros_like(arithmetic), inverse.reciprocal() - arithmetic,
    )


def _reconstruct_sampled_kottke_tensor(scene, arithmetic, inverse):
    """Symmetric effective epsilon for scalar constituents: A I + (H-A) n n^T."""
    normal = _normalized_field_gradients(scene, arithmetic["x"], periodic_nodes=True)
    correction = _kottke_normal_correction(arithmetic["x"], inverse["x"])
    diagonal = {axis: arithmetic[axis] + correction * normal[axis].square() for axis in _AXES}
    offdiagonal = {a + b: correction * normal[a] * normal[b]
                   for a, b in (("x", "y"), ("x", "z"), ("y", "z"))}
    return diagonal, offdiagonal
