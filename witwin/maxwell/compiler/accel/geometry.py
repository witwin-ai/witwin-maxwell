"""Ordered descriptors and conservative coarse-bin candidate lists.

Unqualified coverage primitives keep torch and pack their occupied samples into
bounded slabs for the native cell loop.
"""
from itertools import product

import torch
from witwin.core import Box, Sphere, Cylinder, Pyramid, Mesh, Union, Intersection, Difference
from ...geometry.polyslab import PolySlab


def qualified_geometry(geometry):
    if type(geometry) in (Box, Sphere, Cylinder, Pyramid, PolySlab, Mesh):
        return True
    if type(geometry) in (Union, Intersection, Difference):
        return all(qualified_geometry(child) for child in geometry.geometries)
    return False


def native_geometry(geometry):
    return type(geometry) in (Box, Sphere, Cylinder, Union, Intersection, Difference) and all(
        native_geometry(child) for child in getattr(geometry, "geometries", ()))


MAX_TREE_NODES = 16


def tree_size(geometry):
    return 1 + sum(tree_size(child) for child in getattr(geometry, "geometries", ()))


def descriptors(structures, *, device, native_coverage):
    from witwin.core.geometry.base import _rotation_quaternion, quat_to_rotation_matrix

    rows = []
    for structure in structures:
        nodes = []
        def pack(geometry, parent):
            index = len(nodes)
            row = torch.zeros(24, device=device, dtype=torch.float32)
            row[0] = {Sphere: 1, Box: 2, Union: 3, Intersection: 4, Difference: 5, Cylinder: 6}[type(geometry)]
            row[1] = parent
            row[2] = float(getattr(geometry, "smooth", False))
            if hasattr(geometry, "smoothness"):
                row[3] = geometry.smoothness.to(device=device, dtype=row.dtype).clamp_min(torch.finfo(row.dtype).eps)
            row[4:7] = geometry.position.to(device)
            row[7:16] = quat_to_rotation_matrix(_rotation_quaternion(geometry.rotation)).T.to(device).flatten()
            if type(geometry) is Sphere:
                row[16] = geometry.radius.to(device)
            elif type(geometry) is Box:
                # Match core SDF: convert BEFORE halving, including subnormals.
                # Pole ownership separately packs fp64 query dimensions.
                row[16:19] = geometry.size.to(device=device, dtype=row.dtype)/2
            elif type(geometry) is Cylinder:
                row[16] = geometry.radius.to(device)
                row[17] = geometry.height.to(device=device, dtype=row.dtype)/2
                row[18] = "xyz".index(geometry.axis)
            nodes.append(row)
            for child in getattr(geometry, "geometries", ()):
                pack(child, index)
        if native_coverage and native_geometry(structure.geometry):
            pack(structure.geometry, -1)
            nodes[0][23] = len(nodes)
        rows.append(torch.stack(nodes + [torch.zeros(24, device=device, dtype=torch.float32)]*(MAX_TREE_NODES-len(nodes))))
    return torch.stack(rows) if rows else torch.empty((0, MAX_TREE_NODES, 24), device=device, dtype=torch.float32)


def coarse_bins(axes, widths, geometry, native):
    device, count = geometry.device, len(native)
    sizes = [(axis.shape[1]+7)//8 for axis in axes]
    bins = sizes[0]*sizes[1]*sizes[2]
    candidates = torch.ones((bins, count), device=device, dtype=torch.bool)
    if any(native):
        lows, highs = [], []
        for axis in axes:
            lows.append(torch.stack([axis[:, i:i+8].amin() for i in range(0, axis.shape[1], 8)]))
            highs.append(torch.stack([axis[:, i:i+8].amax() for i in range(0, axis.shape[1], 8)]))
        lower = torch.stack(torch.meshgrid(*lows, indexing="ij"), -1).reshape(-1, 3)
        upper = torch.stack(torch.meshgrid(*highs, indexing="ij"), -1).reshape(-1, 3)
        # Coverage shift is clamped to a quarter width; the projected width is
        # bounded by the sum of the axis spacings. Grow by four such sums plus
        # an outward rounding margin before excluding a whole rounded bin.
        reach = 4*sum(width.max() for width in widths)
        scale = torch.maximum(lower.abs().amax(), upper.abs().amax())
        reach = reach + 64*torch.finfo(torch.float32).eps*(scale+reach+1)
        for j, fused in enumerate(native):
            if fused:
                if geometry[j, 0, 0] == 1:
                    center, radius = geometry[j, 0, 4:7], geometry[j, 0, 16]
                    candidates[:, j] = ((upper >= center-radius-reach) &
                                         (lower <= center+radius+reach)).all(-1)
    offsets = torch.cat((torch.zeros(1, device=device, dtype=torch.int32),
                         candidates.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32)))
    ids = torch.arange(count, device=device, dtype=torch.int32).expand(bins, -1)[candidates]
    return offsets, ids


def torch_fills(scene, structures, native, axes, spacing, delta, shifts, selected, stagger):
    from .. import materials as m
    from ..subpixel import _broadcast_spacing, _cached_mesh_subsample_fill, _coverage_fill
    from .primitives import sampling_geometry

    periodic = any(scene.boundary.axis_kind(a) in ("periodic", "bloch") for a in "xyz")
    geometries = [s.geometry if fused else sampling_geometry(s.geometry, spacing=spacing if periodic else None)
                  for s, fused in zip(structures, native)]
    rows = []
    for sample in product(*(range(axis.shape[0]) for axis in axes)):
        full_coords = tuple(axes[a][sample[a]] for a in range(3))
        coords = torch.meshgrid(full_coords[0][selected], *full_coords[1:], indexing="ij")
        fills = []
        for structure, geometry, fused in zip(structures, geometries, native):
            if fused:
                fills.append(torch.zeros_like(coords[0]))
            elif periodic:
                full = torch.meshgrid(*full_coords, indexing="ij")
                fill = m._geometry_occupancy(
                    scene, geometry, coords=full,
                    periodic_shift_options=shifts[id(structure.geometry)],
                    wrap_skip_axes=stagger,
                    fill_spacing=None if native_geometry(structure.geometry) else spacing,
                )
                fills.append(fill[selected])
            elif spacing is None:
                fills.append(m._geometry_occupancy(scene, geometry, coords=coords,
                             periodic_shift_options=shifts[id(structure.geometry)]))
            else:
                local_spacing = tuple(value[selected] if a == 0 and torch.is_tensor(value)
                                      else value for a, value in enumerate(spacing))
                widths = tuple(_broadcast_spacing(v, a, coords[0].shape)
                               for a, v in enumerate(local_spacing))
                if (type(geometry) is Mesh and geometry.fill_mode != "surface"
                        and (geometry.fill_mode == "solid" or geometry.is_watertight)):
                    fills.append(_cached_mesh_subsample_fill(geometry, coords, widths, delta))
                    continue
                distance = geometry.signed_distance(*coords)
                plus, minus = [], []
                for a in range(3):
                    query = list(coords)
                    query[a] = coords[a] + delta
                    plus.append(geometry.signed_distance(*query))
                    query[a] = coords[a] - delta
                    minus.append(geometry.signed_distance(*query))
                fills.append(_coverage_fill(distance, plus, minus, widths, delta))
        rows.append(torch.stack(fills).flatten(1))
    return torch.stack(rows).contiguous()
