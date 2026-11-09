"""Conforming Manhattan sheets, via walls and bridge strips on the device."""

from dataclasses import dataclass

import torch

from ..open_surface import OpenRWGBasis, OpenSurfaceMesh


@dataclass
class LayeredMesh:
    mesh: OpenSurfaceMesh
    basis: OpenRWGBasis
    triangle_planes: torch.Tensor
    support_slots: torch.Tensor | None = None
    grounded: bool = False

    @property
    def quality(self):
        """Minimum angle (degrees), longest-edge/altitude ratio and edge (metres)."""
        triangle = self.mesh.vertices[self.mesh.faces]
        edges = triangle.roll(-1, 1) - triangle
        lengths = torch.linalg.vector_norm(edges, dim=-1)
        twice_area = torch.linalg.vector_norm(torch.linalg.cross(edges[:, 0], -edges[:, 2]), dim=-1)
        cosine = -(edges * edges.roll(1, 1)).sum(-1) / (lengths * lengths.roll(1, 1))
        values = torch.stack((torch.acos(cosine.clamp(-1, 1)).min() * (180 / torch.pi),
                              (lengths.max(-1).values.square() / twice_area).max(), lengths.min())).tolist()
        return dict(min_angle_degrees=values[0], max_aspect_ratio=values[1], smallest_edge_m=values[2])


def clustered_coordinates(values, tolerance):
    """Host layout metadata: complete-span clusters on an explicit SI lattice.

    Each cluster spans <= tolerance (no transitive chaining); its mean rounds to
    the nearest tolerance lattice coordinate, ties to even. Zero disables repair.
    Apply one map to sheets, via rims and ports together, and never to z.
    Features at or below the tolerance can collapse and are rejected by meshing.
    """
    values = sorted(set(values))
    if tolerance == 0:
        return {v: v for v in values}
    groups = []
    for value in values:
        if not groups or value - groups[-1][0] > tolerance:
            groups.append([])
        groups[-1].append(value)
    return {v: round((sum(group) / len(group)) / tolerance) * tolerance
            for group in groups for v in group}


def _axis_points(edges, h, h_edge, growth_ratio):
    edges = torch.unique(edges, sorted=True)
    length = edges.diff()
    if h_edge is None or growth_ratio == 1:
        spacing = h if h_edge is None else h_edge
        count = torch.ceil(length / spacing).to(torch.int64)
        interval = torch.repeat_interleave(torch.arange(len(length), device=edges.device), count)
        local = torch.arange(len(interval), device=edges.device) - (count.cumsum(0) - count)[interval]
        return torch.cat((edges[interval] + length[interval] * local / count[interval], edges[-1:]))

    # Mirror a capped geometric sequence about each interval midpoint. Include
    # an odd central cell when it fits, then solve for the growth ratio.
    half = length / 2
    cap = torch.ceil(torch.log(edges.new_tensor(h / h_edge)) / edges.new_tensor(growth_ratio).log())
    capacity = h_edge * (growth_ratio ** cap - 1) / (growth_ratio - 1)
    count = torch.where(half <= capacity,
                        torch.ceil(torch.log1p(half * (growth_ratio - 1) / h_edge)
                                   / edges.new_tensor(growth_ratio).log()),
                        cap + torch.ceil((half - capacity) / h)).to(torch.int64).clamp_min(1)
    geometric = torch.minimum(count, cap)
    total = h_edge * (growth_ratio ** geometric - 1) / (growth_ratio - 1) + (count - geometric) * h
    last = (h_edge * growth_ratio ** (count - 1)).clamp_max(h)
    count = 2 * count - (2 * total - last >= length).to(torch.int64)
    interval = torch.repeat_interleave(torch.arange(len(length), device=edges.device), count)
    start = count.cumsum(0) - count
    local = torch.arange(len(interval), device=edges.device) - start[interval]
    power = torch.minimum(local, count[interval] - 1 - local)
    first = torch.minimum(length / count, half.new_full(half.shape, h_edge))
    low, high = torch.ones_like(half), torch.full_like(half, growth_ratio)
    for _ in range(48):
        ratio = (low + high) / 2
        widths = (first[interval] * ratio[interval] ** power).clamp_max(h)
        total = torch.segment_reduce(widths, "sum", lengths=count)
        low, high = torch.where(total < length, ratio, low), torch.where(total < length, high, ratio)
    widths = (first[interval] * ((low + high) / 2)[interval] ** power).clamp_max(h)
    prefix = torch.cat((widths.new_zeros(1), widths.cumsum(0)))
    distance = prefix[:-1] - prefix[start[interval]]
    return torch.cat((edges[interval] + distance, edges[-1:]))


def mesh_rectangles(rectangles, planes, h, *, cut_lines=None, h_edge=None, growth_ratio=1.5):
    """Mesh (x0,x1,y0,y1,z) rectangles on the sorted CUDA table ``planes``.

    ``cut_lines`` is a per-plane sequence of ``(x_cuts, y_cuts)``. Every edge/cut
    extends across its plane before union masking. All faces have +z normal;
    cell (i,j) uses the same alternating diagonal rule as the open plate mesher.
    Grading starts at h_edge on both ends of every breakpoint interval, grows
    by at most growth_ratio, and caps at h. Short intervals reduce the first cell.
    """
    rectangles = torch.as_tensor(rectangles, dtype=torch.float64, device=planes.device)
    if rectangles.ndim != 2 or rectangles.shape[1] != 5:
        raise ValueError("Axis-aligned rectangles require (x0,x1,y0,y1,z)")
    if not bool(torch.isfinite(rectangles).all() & (rectangles[:, 1] > rectangles[:, 0]).all()
                & (rectangles[:, 3] > rectangles[:, 2]).all()):
        raise ValueError("Axis-aligned rectangles require finite, increasing bounds")
    if not bool((rectangles[:, 4, None] == planes[None]).any(1).all()):
        raise ValueError("Rectangle is off every metal plane")
    if h <= 0 or (h_edge is not None and not 0 < h_edge <= h) or not 1 <= growth_ratio <= 1.5:
        raise ValueError("Require h>0, 0<h_edge<=h and 1<=growth_ratio<=1.5")
    cut_lines = [((), ())] * len(planes) if cut_lines is None else cut_lines
    vertices, faces, plane_ids = [], [], []
    offset = 0
    # Only small plane metadata is iterated on the host, never cells or edges.
    for plane in range(len(planes)):
        rect = rectangles[rectangles[:, 4] == planes[plane]]
        if not len(rect):
            continue
        cuts = cut_lines[plane]
        xedges, yedges = rect[:, :2].reshape(-1), rect[:, 2:4].reshape(-1)
        xc, yc = (torch.as_tensor(cut, dtype=planes.dtype, device=planes.device) for cut in cuts)
        xedges = torch.cat((xedges, xc[(xc > xedges.min()) & (xc < xedges.max())]))
        yedges = torch.cat((yedges, yc[(yc > yedges.min()) & (yc < yedges.max())]))
        x = _axis_points(xedges, h, h_edge, growth_ratio)
        y = _axis_points(yedges, h, h_edge, growth_ratio)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack((xx, yy, planes[plane].expand_as(xx)), -1).reshape(-1, 3)
        i, j = torch.meshgrid(torch.arange(len(x) - 1, device=planes.device),
                              torch.arange(len(y) - 1, device=planes.device), indexing="ij")
        cx, cy = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
        inside = ((cx[:, None, None] > rect[:, 0]) & (cx[:, None, None] < rect[:, 1])
                  & (cy[None, :, None] > rect[:, 2]) & (cy[None, :, None] < rect[:, 3])).any(-1)
        a = i * len(y) + j
        b, c, d = a + len(y), a + len(y) + 1, a + 1
        flip = ((i + j) % 2).bool()
        first = torch.where(flip[..., None], torch.stack((a, b, d), -1), torch.stack((a, b, c), -1))
        second = torch.where(flip[..., None], torch.stack((b, c, d), -1), torch.stack((a, c, d), -1))
        selected = torch.stack((first, second), 2)[inside].reshape(-1, 3)
        used, inverse = torch.unique(selected, sorted=True, return_inverse=True)
        vertices.append(points[used])
        faces.append(inverse.reshape(-1, 3) + offset)
        plane_ids.append(torch.full((len(selected),), plane, dtype=torch.int64, device=planes.device))
        offset += len(used)
    # Construction guarantees disjoint planar cells; avoid the general mesh's
    # quadratic geometric intersection validator. Its device topology is reused.
    mesh = OpenSurfaceMesh(torch.cat(vertices), torch.cat(faces), validate=False)
    return LayeredMesh(mesh, OpenRWGBasis(mesh), torch.cat(plane_ids))


def validate_vias(rectangles, vias):
    """Validate host Box records (x0,x1,y0,y1,z0,z1), including union coverage."""
    import math
    planes = sorted({r[4] for r in rectangles})
    for i, via in enumerate(vias):
        if len(via) != 6 or not all(math.isfinite(v) for v in via):
            raise ValueError("Via bounds must be six finite coordinates")
        x0, x1, y0, y1, z0, z1 = via
        if not (x0 < x1 and y0 < y1 and z0 < z1):
            raise ValueError("Via bounds must be increasing")
        if z0 not in planes or z1 not in planes:
            raise ValueError("Via endpoints must coincide with declared metal planes")
        if planes.index(z1) != planes.index(z0) + 1:
            raise NotImplementedError("Vias must join consecutive metal planes")
        for z in (z0, z1):
            rect = [r for r in rectangles if r[4] == z]
            xs = sorted({x0, x1, *(x for r in rect for x in r[:2] if x0 < x < x1)})
            ys = sorted({y0, y1, *(y for r in rect for y in r[2:4] if y0 < y < y1)})
            if any(not any(r[0] <= (a+b)/2 <= r[1] and r[2] <= (c+d)/2 <= r[3] for r in rect)
                   for a, b in zip(xs[:-1], xs[1:]) for c, d in zip(ys[:-1], ys[1:])):
                raise ValueError("Via footprint must be inside metal on both planes")
            # Every rim edge needs sheet metal on the outside of the hole.
            # A footprint on a sheet boundary would leave an unattached tube rim.
            for x, left in ((x0, True), (x1, False)):
                if any(not any((r[0] < x <= r[1] if left else r[0] <= x < r[1])
                               and r[2] <= (a+b)/2 <= r[3] for r in rect)
                       for a,b in zip(ys[:-1],ys[1:])):
                    raise ValueError("Via footprint rim must be surrounded by metal on both planes")
            for y, lower in ((y0, True), (y1, False)):
                if any(not any((r[2] < y <= r[3] if lower else r[2] <= y < r[3])
                               and r[0] <= (a+b)/2 <= r[1] for r in rect)
                       for a,b in zip(xs[:-1],xs[1:])):
                    raise ValueError("Via footprint rim must be surrounded by metal on both planes")
        for other in vias[:i]:
            if (max(z0, other[4]) < min(z1, other[5]) and
                    max(x0, other[0]) <= min(x1, other[1]) and
                    max(y0, other[2]) <= min(y1, other[3])):
                raise NotImplementedError("Touching or overlapping vias require junction geometry")


def _ground_basis(mesh, grounds):
    """Keep ordinary RWGs and only PEC-attached boundary half functions.

    Their image carries the opposite surface charge: physical plus image charge
    is neutral. The divergence graph has a single PEC reservoir vertex; it is
    used only to build cycles, never as a charged Galerkin face.
    """
    from ..topology import INTERIOR
    basis = OpenRWGBasis(mesh, boundary_half_functions=True)
    topology = mesh.topology
    endpoints = mesh.vertices[topology.edges, 2]
    attached = torch.zeros(len(endpoints), dtype=torch.bool, device=mesh.device)
    for z in grounds:
        attached |= (endpoints == z).all(-1)
    keep = (topology.kind == INTERIOR) | attached
    remap = keep.to(torch.int64).cumsum(0) - 1
    valid = keep[topology.face_edges]
    basis.face_edges = torch.where(valid, remap[topology.face_edges], 0)
    basis.face_signs = torch.where(valid, basis.face_signs, 0)
    for name in ('edges', 'plus_faces', 'minus_faces', 'plus_vertices', 'minus_vertices', 'edge_lengths'):
        setattr(basis, name, getattr(basis, name)[keep].contiguous())
    return basis, topology.incidence_slots[keep].clamp_min(0).contiguous()


def mesh_conductors(rectangles, planes, h, *, vias=(), bridges=(), cut_lines=None,
                    h_edge=None, growth_ratio=1.5, grounds=(), interfaces=()):
    """Conforming sheets, four-wall tubes and vertical bridge strips.

    One breakpoint grid is shared by all surfaces. Footprints are cut out before
    walls are welded by exact coordinate identity. A native integer BFS orients
    bent sheets; true junctions raise. Bridge records are
    (normal_axis, fixed, low, high, z0, z1, cut, centre, sign).
    """
    from ..topology import classify_surface
    from .build import build_extension
    validate_vias(rectangles, vias)
    if not vias and not bridges:
        return mesh_rectangles(rectangles, planes, h, cut_lines=cut_lines,
                               h_edge=h_edge, growth_ratio=growth_ratio)
    if h <= 0 or (h_edge is not None and not 0 < h_edge <= h) or not 1 <= growth_ratio <= 1.5:
        raise ValueError("Require h>0, 0<h_edge<=h and 1<=growth_ratio<=1.5")
    device = planes.device
    def tensor(values):
        return torch.tensor(values, device=device, dtype=torch.float64)
    xedges = [v for r in rectangles for v in r[:2]] + [v for r in vias for v in r[:2]]
    yedges = [v for r in rectangles for v in r[2:4]] + [v for r in vias for v in r[2:4]]
    zedges = planes.cpu().tolist()
    for normal, fixed, lo, hi, za, zb, cut, _, _ in bridges:
        (xedges if normal == 0 else yedges).append(fixed)
        (yedges if normal == 0 else xedges).extend((lo, hi))
        zedges.extend((za, cut, zb))
    if cut_lines is not None:
        for xc, yc in cut_lines:
            xedges.extend(xc)
            yedges.extend(yc)
    zedges.extend(z for z in interfaces if min(zedges) < z < max(zedges))
    axes = [_axis_points(tensor(e), h, h_edge, growth_ratio) for e in (xedges, yedges, zedges)]
    vertices, faces, ids = [], [], []
    offset = 0
    def surface(u, v, fixed_axis, fixed, mask, plane):
        nonlocal offset
        a0, a1 = [a for a in range(3) if a != fixed_axis]
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        points = torch.empty((uu.numel(), 3), device=device, dtype=torch.float64)
        points[:, a0], points[:, a1], points[:, fixed_axis] = uu.flatten(), vv.flatten(), fixed
        i, j = torch.meshgrid(torch.arange(len(u)-1, device=device),
                              torch.arange(len(v)-1, device=device), indexing='ij')
        a = i * len(v) + j
        b, c, d = a + len(v), a + len(v) + 1, a + 1
        flip = ((i+j) % 2).bool()
        first = torch.where(flip[..., None], torch.stack((a,b,d),-1), torch.stack((a,b,c),-1))
        second = torch.where(flip[..., None], torch.stack((b,c,d),-1), torch.stack((a,c,d),-1))
        selected = torch.stack((first, second), 2)[mask].reshape(-1,3)
        used, inverse = torch.unique(selected, sorted=True, return_inverse=True)
        vertices.append(points[used])
        faces.append(inverse.reshape(-1,3)+offset)
        ids.append(torch.full((len(selected),), plane, device=device, dtype=torch.int64))
        offset += len(used)
    x, y, z = axes
    cx, cy = (x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2
    for plane, height in enumerate(planes.cpu().tolist()):
        mask = torch.zeros((len(cx),len(cy)), device=device, dtype=torch.bool)
        for x0,x1,y0,y1,zr in rectangles:
            if zr == height:
                mask |= (cx[:,None]>x0)&(cx[:,None]<x1)&(cy[None,:]>y0)&(cy[None,:]<y1)
        for x0,x1,y0,y1,za,zb in vias:
            if height in (za,zb):
                mask &= ~((cx[:,None]>x0)&(cx[:,None]<x1)&(cy[None,:]>y0)&(cy[None,:]<y1))
        if bool(mask.any()):
            surface(x,y,2,height,mask,plane)
    for x0,x1,y0,y1,za,zb in vias:
        uz = z[(z>=za)&(z<=zb)]
        for normal, fixed, lo, hi in ((0,x0,y0,y1),(0,x1,y0,y1),(1,y0,x0,x1),(1,y1,x0,x1)):
            u = axes[1-normal]
            u = u[(u>=lo)&(u<=hi)]
            surface(u,uz,normal,fixed,torch.ones((len(u)-1,len(uz)-1),device=device,dtype=torch.bool),-1)
    for normal,fixed,lo,hi,za,zb,_,_,_ in bridges:
        u = axes[1-normal]
        u, uz = u[(u>=lo)&(u<=hi)], z[(z>=za)&(z<=zb)]
        surface(u,uz,normal,fixed,torch.ones((len(u)-1,len(uz)-1),device=device,dtype=torch.bool),-1)
    points, inverse = torch.unique(torch.cat(vertices), dim=0, sorted=True, return_inverse=True)
    triangles = inverse[torch.cat(faces)]
    topology = classify_surface(triangles, len(points))
    if topology.junction_edges:
        raise NotImplementedError("Geometry needs junction RWGs (three or more faces on an edge)")
    oriented = build_extension().orient(triangles, topology.incidence_faces, topology.face_edges,
                                         topology.face_edge_signs)
    mesh = OpenSurfaceMesh(points, oriented, validate=False)
    basis, slots = _ground_basis(mesh, grounds) if grounds else (OpenRWGBasis(mesh), None)
    return LayeredMesh(mesh, basis, torch.cat(ids), slots, bool(grounds))
