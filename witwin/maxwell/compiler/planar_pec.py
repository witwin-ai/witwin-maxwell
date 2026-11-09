"""Hard, tangential Yee ownership for explicitly declared planar PEC polygons."""
from __future__ import annotations

from itertools import product

import numpy as np
import torch

from witwin.core import Box
from ..geometry.planar_pec import PlanarPEC
from ..geometry.surface_mesh import SurfaceMesh
from .structures import pec_structures

_AXES = ("x", "y", "z")


def volume_pec_structures(scene):
    return tuple(s for s in pec_structures(scene) if not isinstance(s.geometry, (PlanarPEC, SurfaceMesh)))


def validate_legacy_thin_pec(scene):
    """Reject axis-aligned subcell volumes with no normal grid support.

    This check leaves resolved volumes untouched. Explicit PlanarPEC is required
    when a physical foil is intentionally replaced by an ideal sheet.
    """
    for structure in volume_pec_structures(scene):
        geometry = structure.geometry
        if not isinstance(geometry, Box):
            continue
        q = geometry.rotation.detach().cpu().numpy()
        if not np.allclose(np.abs(q), (1, 0, 0, 0), atol=1e-7):
            continue
        sizes = geometry.size.detach().cpu().numpy().astype(np.float64)
        positions = geometry.position.detach().cpu().numpy().astype(np.float64)
        for dim, axis in enumerate(_AXES):
            nodes = getattr(scene, f"{axis}_nodes64")
            center, width = positions[dim], sizes[dim]
            if center + width / 2 < nodes[0] or center - width / 2 > nodes[-1]:
                continue
            index = int(np.argmin(abs(nodes - center)))
            local = min(np.diff(nodes)[max(0, index - 1):min(len(nodes) - 1, index + 1)])
            tol = min(local * 1e-5, max(local * 1e-10, abs(center) * np.finfo(np.float32).eps * 2))
            if width < local and not np.any(abs(nodes - center) <= width / 2 + tol):
                raise ValueError(
                    f"PEC structure {structure.name!r} is unresolved along {axis}: its "
                    f"{width:g} m thickness contains no Yee node plane and would disappear. "
                    "Insert the physical boundaries in GridSpec.custom, or explicitly use "
                    "PlanarPEC with its coordinate included in the grid. No snapping is performed."
                )


def _wholly_supported(geometry, u0, v0, u1, v1, direction, tol):
    """Whole-segment inclusion, including holes smaller than an edge interval.

    Endpoint and midpoint tests alone miss a small slot. Every polygon crossing
    strictly inside the segment therefore opens that edge conservatively. An edge
    collinear with an outer/hole boundary remains a legitimate conductor edge.
    """
    inside = (geometry.footprint_distance(u0, v0) <= tol) & (geometry.footprint_distance(u1, v1) <= tol)
    inside &= geometry.footprint_distance((u0 + u1) / 2, (v0 + v1) / 2) <= tol
    varying_start, varying_end = (u0, u1) if direction == 0 else (v0, v1)
    fixed = v0 if direction == 0 else u0
    for loop in (geometry.vertices, *geometry.holes):
        loop = loop.to(device=u0.device, dtype=u0.dtype)
        for first, second in zip(loop, loop.roll(-1, 0)):
            delta = second[1 - direction] - first[1 - direction]
            safe = torch.where(delta.abs() > tol, delta, torch.ones_like(delta))
            alpha = (fixed - first[1 - direction]) / safe
            crossing = first[direction] + alpha * (second[direction] - first[direction])
            intersects = ((delta.abs() > tol) & (alpha >= 0) & (alpha <= 1)
                          & (crossing > varying_start + tol) & (crossing < varying_end - tol))
            inside &= ~intersects
    return inside


def _compile_planar_sheet(scene, structure, nodes, normal, index):
    """Owned tangential edge masks, retained nodes and audit of one validated sheet.

    The result depends only on the sheet declaration, its node plane and the
    grid/boundary, never on another structure; :func:`compile_planar_pec`
    composes owners across sheets.
    """
    geometry = structure.geometry
    transverse = [dim for dim in range(3) if dim != normal]
    coordinates = [torch.tensor(nodes[d], dtype=torch.float64, device=scene.device) for d in transverse]
    u, v = torch.meshgrid(*coordinates, indexing="ij")
    lateral_tol = min(float(np.diff(nodes[d]).min()) for d in transverse) * 1e-10
    shifts = []
    physical = scene.physical_domain_range
    for dim in transverse:
        span = float(physical[2*dim+1] - physical[2*dim])
        shifts.append((-span, 0., span) if scene.boundary.axis_kind(_AXES[dim]) in ("periodic", "bloch") else (0.,))
    images = tuple(product(*shifts))
    support = torch.zeros_like(u, dtype=torch.bool)
    for du, dv in images:
        support |= geometry.footprint_distance(u-du, v-dv) <= lateral_tol
    retained_nodes = torch.zeros_like(support)
    counts, local_masks, masks = {}, {}, {}
    for direction, dim in enumerate(transverse):
        lo, hi = [slice(None)] * 2, [slice(None)] * 2
        lo[direction], hi[direction] = slice(None, -1), slice(1, None)
        mask = torch.zeros_like(u[tuple(lo)], dtype=torch.bool)
        for du, dv in images:
            mask |= _wholly_supported(geometry, u[tuple(lo)]-du, v[tuple(lo)]-dv,
                                      u[tuple(hi)]-du, v[tuple(hi)]-dv, direction, lateral_tol)
        name = f"E{_AXES[dim]}"
        masks[name] = mask
        counts[name] = int(mask.sum().item())
        local_masks[direction] = mask
        retained_nodes[tuple(lo)] |= mask
        retained_nodes[tuple(hi)] |= mask
    if not all(counts.values()):
        raise ValueError(f"PlanarPEC structure {structure.name!r} has unresolved lateral support "
                         f"({counts}); refine the transverse grid to retain both tangential edge families.")
    cells = (local_masks[0][:, :-1] & local_masks[0][:, 1:]
             & local_masks[1][:-1, :] & local_masks[1][1:, :])
    cell_areas = torch.diff(coordinates[0])[:, None] * torch.diff(coordinates[1])[None, :]
    represented_area = float((cells * cell_areas).sum().item())
    def loop_area(loop):
        return abs(float((loop[:, 0]*loop.roll(-1, 0)[:, 1]-loop[:, 1]*loop.roll(-1, 0)[:, 0]).sum().item())) / 2
    physical_area = loop_area(geometry.vertices) - sum(loop_area(hole) for hole in geometry.holes)
    audit = {"name": structure.name, "normal_axis": geometry.normal_axis,
             "coordinate_m": geometry.coordinate, "node_index": index,
             "physical_thickness_m": geometry.physical_thickness,
             "constraint_edges": counts, "retained": True,
             "physical_polygon_area_m2": physical_area,
             "fully_supported_cell_area_m2": represented_area,
             "lateral_discretization": "conservative hard staircase; cut lateral edges open"}
    return {"masks": masks, "retained_nodes": retained_nodes, "audit": audit}


def compile_planar_pec(scene, *, sheets=None):
    """Compose hard tangential ownership of every enabled PlanarPEC sheet.

    ``sheets`` optionally maps a structure index to its
    :func:`_compile_planar_sheet` result. Present entries are reused and missing
    ones are compiled and inserted, so a caller that has proven a sheet and the
    grid unchanged composes the identical layout without resampling the sheet.
    """
    from dataclasses import replace
    indexed = []
    for i, structure in enumerate(scene.structures):
        if not getattr(structure, "enabled", True):
            continue
        if isinstance(structure.geometry, SurfaceMesh):
            from .thin_sheet import owns_thin_sheet
            if owns_thin_sheet(structure):
                continue
            if not bool(getattr(structure.material, "is_pec", False)):
                raise NotImplementedError("SurfaceMesh requires PEC; finite copper needs independent two-sided cut traces.")
            planar = structure.geometry.planar_pec(scene)
            if planar is not None:
                structure = replace(structure, geometry=planar)
        if isinstance(structure.geometry, PlanarPEC):
            indexed.append((i, structure))
    if not indexed:
        return None
    for _, structure in indexed:
        if not bool(getattr(structure.material, "is_pec", False)):
            raise ValueError("PlanarPEC geometry requires Material.pec(); use a surface material geometry for finite loss.")
    indexed.sort(key=lambda item: (int(getattr(item[1], "priority", 0)), item[0]))
    nodes = [getattr(scene, f"{a}_nodes64") for a in _AXES]
    shape = tuple(len(n) for n in nodes)
    device = scene.device
    occupancy = torch.zeros(shape, device=device)
    owners = {}
    for dim, axis in enumerate(_AXES):
        edge_shape = list(shape)
        edge_shape[dim] -= 1
        owners[f"E{axis}"] = torch.full(edge_shape, -1, dtype=torch.int32, device=device)
    audit, node_sets = [], []
    for owner, structure in indexed:
        geometry = structure.geometry
        if any(loop.requires_grad for loop in (geometry.vertices, *geometry.holes, geometry.position, geometry.rotation)):
            raise NotImplementedError("PlanarPEC has discrete edge ownership; shape gradients are unsupported.")
        rotation = geometry.rotation.detach().cpu().numpy()
        if not np.array_equal(np.abs(rotation), (1., 0., 0., 0.)):
            raise NotImplementedError("PlanarPEC supports axis-aligned planes; rotated sheets need a cut-surface discretization.")
        normal = _AXES.index(geometry.normal_axis)
        declared_position = torch.zeros_like(geometry.position)
        declared_position[normal] = geometry.coordinate
        if not torch.equal(geometry.position, declared_position):
            raise ValueError("PlanarPEC.position is derived from its plane coordinate; construct a new polygon with translated vertices/coordinate rather than mutating position.")
        index = int(np.argmin(abs(nodes[normal] - geometry.coordinate)))
        if (scene.boundary.axis_kind(geometry.normal_axis) in ("periodic", "bloch")
                and index in (0, len(nodes[normal])-1)):
            raise NotImplementedError("A PlanarPEC plane coincident with a periodic normal seam requires duplicate-plane ownership; place the plane inside the periodic domain.")
        spacing = np.diff(nodes[normal])
        local = float(min(spacing[max(0, index - 1):min(len(spacing), index + 1)]))
        tol = max(local * 1e-10, abs(geometry.coordinate) * np.finfo(np.float64).eps * 8)
        if abs(nodes[normal][index] - geometry.coordinate) > tol:
            raise ValueError(
                f"PlanarPEC structure {structure.name!r} coordinate {geometry.coordinate:.17g} "
                f"is off-grid along {geometry.normal_axis}. Insert that coordinate into "
                "GridSpec.custom (or provide an auto-grid plane override) and prepare again. "
                "No snapping or artificial thickness is permitted."
            )
        sheet = None if sheets is None else sheets.get(owner)
        if sheet is None:
            sheet = _compile_planar_sheet(scene, structure, nodes, normal, index)
            if sheets is not None:
                sheets[owner] = sheet
        retained_nodes = sheet["retained_nodes"]
        node_slice = [slice(None)] * 3
        node_slice[normal] = index
        for name, mask in sheet["masks"].items():
            selected = owners[name][tuple(node_slice)]
            selected[mask] = owner
        occupancy[tuple(node_slice)] = torch.maximum(occupancy[tuple(node_slice)], retained_nodes.float())
        node_sets.append((owner, normal, index, retained_nodes))
        entry = {"owner": owner, **sheet["audit"]}
        entry["constraint_edges"] = dict(entry["constraint_edges"])
        audit.append(entry)
    # Contact graph is a compile-time diagnostic, evaluated from exact native node
    # ownership. It does not run in the field update or alter any material masks.
    contacts = []
    for i, (first, normal_a, index_a, support_a) in enumerate(node_sets):
        for second, normal_b, index_b, support_b in node_sets[i + 1:]:
            if normal_a == normal_b:
                touching = index_a == index_b and bool((support_a & support_b).any().item())
            else:
                a_dims = [d for d in range(3) if d != normal_a]
                b_dims = [d for d in range(3) if d != normal_b]
                sa, sb = [slice(None)] * 2, [slice(None)] * 2
                sa[a_dims.index(normal_b)] = index_b
                sb[b_dims.index(normal_a)] = index_a
                touching = bool((support_a[tuple(sa)] & support_b[tuple(sb)]).any().item())
            if touching:
                contacts.append([first, second])
    return {"edges": {name: (owner >= 0).float() for name, owner in owners.items()},
            "owners": owners, "occupancy": occupancy,
            "audit": {"sheets": audit, "contacts": contacts, "retention_fraction": 1.0,
                      "retention_definition": "fraction of declarations retaining both tangential edge families",
                      "edge_rule": "whole tangential edge; hard PEC; last priority/index owner"}}


def attach_planar_pec(scene, model, *, sheets=None):
    layout = compile_planar_pec(scene, sheets=sheets)
    if layout is None:
        return
    model["pec_volume_occupancy"] = model["pec_occupancy"]
    occupancy = model["pec_occupancy"]
    model["pec_occupancy"] = layout["occupancy"] if occupancy is None else torch.maximum(occupancy, layout["occupancy"])
    model["pec_sheet_edges"] = layout["edges"]
    model["pec_sheet_owners"] = layout["owners"]
    model["pec_sheet_audit"] = layout["audit"]
