"""True triangle/cell quadrature and sparse integrated Whitney electric traces."""
from __future__ import annotations

from itertools import product
import math

import torch


def clip_triangles(triangles, lows, highs):
    """Clip independent triangles against Cartesian boxes on their torch device."""
    batch, capacity = len(triangles), 12
    polygon = triangles.new_zeros((batch, capacity, 3))
    polygon[:, :3] = triangles
    counts = torch.full((batch,), 3, dtype=torch.int64, device=triangles.device)
    row = torch.arange(batch, device=triangles.device)[:, None]
    edge = torch.arange(capacity, device=triangles.device)[None]
    for axis in range(3):
        for boundary, direction in ((lows[:, axis], 1.), (highs[:, axis], -1.)):
            valid = edge < counts[:, None]
            next_edge = torch.remainder(edge+1, counts.clamp_min(1)[:, None])
            following = polygon.gather(1, next_edge[..., None].expand(-1, -1, 3))
            distance = direction*(polygon[:, :, axis]-boundary[:, None])
            next_distance = direction*(following[:, :, axis]-boundary[:, None])
            inside = (distance >= 0) & valid
            next_inside = (next_distance >= 0) & valid
            crossing = (inside != next_inside) & valid
            emitted = inside.long()+crossing.long()
            start = emitted.cumsum(1)-emitted
            total = emitted.sum(1)
            if bool(torch.any(total > capacity)):
                raise RuntimeError("Triangle/box clipping exceeded its exact convex polygon capacity.")
            output = torch.zeros_like(polygon)
            flat = output.reshape(-1, 3)
            flat[(row*capacity+start)[inside]] = polygon[inside]
            denominator = distance-next_distance
            fraction = distance/torch.where(crossing, denominator, torch.ones_like(denominator))
            point = polygon+fraction[..., None]*(following-polygon)
            flat[(row*capacity+start+inside.long())[crossing]] = point[crossing]
            polygon, counts = output, total
    return polygon, counts


def clipped_surface_quadrature(vertices, triangles, nodes, *, include_face_ids=False):
    """Return physical triangle quadrature and Cartesian cells, without voxels."""
    device = vertices.device
    tri = vertices[triangles]
    minimum, maximum = tri.amin(1), tri.amax(1)
    lower, upper = [], []
    for axis, coordinates in enumerate(nodes):
        if bool(torch.any(minimum[:, axis] < coordinates[0])) or bool(torch.any(maximum[:, axis] > coordinates[-1])):
            raise ValueError("SurfaceMesh must lie inside the physical Yee domain.")
        lo = (torch.searchsorted(coordinates, minimum[:, axis].contiguous(), right=True)-1).clamp(0, len(coordinates)-2)
        hi = (torch.searchsorted(coordinates, maximum[:, axis].contiguous(), right=True)-1).clamp(0, len(coordinates)-2)
        lower.append(lo.cpu().tolist())
        upper.append(hi.cpu().tolist())
    candidate_faces, candidate_cells = [], []
    for index in range(len(triangles)):
        for cell in product(*(range(lower[axis][index], upper[axis][index]+1) for axis in range(3))):
            candidate_faces.append(index)
            candidate_cells.append(cell)
    cells = torch.tensor(candidate_cells, device=device, dtype=torch.int64).reshape(-1, 3)
    face_ids = torch.tensor(candidate_faces, device=device, dtype=torch.int64)
    lows = torch.stack([coordinates[cells[:, axis]] for axis, coordinates in enumerate(nodes)], -1)
    highs = torch.stack([coordinates[cells[:, axis]+1] for axis, coordinates in enumerate(nodes)], -1)
    polygon, counts = clip_triangles(tri[face_ids], lows, highs)
    fan = torch.arange(1, polygon.shape[1]-1, device=device)[None].expand(len(cells), -1)
    valid = fan+1 < counts[:, None]
    rows = torch.arange(len(cells), device=device)[:, None].expand_as(fan)[valid]
    fan = fan[valid]
    pieces = torch.stack((polygon[rows, 0], polygon[rows, fan], polygon[rows, fan+1]), 1)
    twice_area = torch.linalg.vector_norm(torch.linalg.cross(pieces[:, 1]-pieces[:, 0], pieces[:, 2]-pieces[:, 0]), dim=-1)
    nonzero = twice_area > 0
    pieces, rows, area = pieces[nonzero], rows[nonzero], twice_area[nonzero]/2
    if not len(pieces):
        raise ValueError("SurfaceMesh has no resolved physical surface area.")
    normals = torch.linalg.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals = normals/torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    a, b = (6+math.sqrt(15))/21, (6-math.sqrt(15))/21
    barycentric = vertices.new_tensor(((1/3, 1/3, 1/3), (a, a, 1-2*a), (a, 1-2*a, a),
        (1-2*a, a, a), (b, b, 1-2*b), (b, 1-2*b, b), (1-2*b, b, b)))
    normalized_weights = vertices.new_tensor((9/40, *((155+math.sqrt(15))/1200,)*3,
                                                    *((155-math.sqrt(15))/1200,)*3))
    points = torch.einsum("qv,pvc->pqc", barycentric, pieces).reshape(-1, 3)
    weights = (area[:, None]*normalized_weights[None]).reshape(-1)
    result = (points, weights, cells[rows].repeat_interleave(7, 0),
              normals[face_ids[rows]].repeat_interleave(7, 0))
    return (*result, face_ids[rows].repeat_interleave(7)) if include_face_ids else result


def _refine_current_mesh(vertices, triangles, spacing):
    """Conforming combinatorial refinement; all metric arithmetic stays on GPU."""
    for _ in range(16):
        triangle = vertices[triangles]
        longest = max(float(torch.linalg.vector_norm(triangle[:, i]-triangle[:, (i+1)%3], dim=-1).max()) for i in range(3))
        if longest <= 2*spacing:
            return vertices, triangles
        host = triangles.cpu().tolist()
        edges = sorted({tuple(sorted((face[i], face[(i+1)%3]))) for face in host for i in range(3)})
        edge_ids = {edge: len(vertices)+i for i, edge in enumerate(edges)}
        midpoint_ids = torch.tensor(edges, device=vertices.device)
        vertices = torch.cat((vertices, vertices[midpoint_ids].mean(1)))
        refined = []
        for a, b, c in host:
            ab, bc, ca = (edge_ids[tuple(sorted(edge))] for edge in ((a,b), (b,c), (c,a)))
            refined.extend(((a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca)))
        triangles = torch.tensor(refined, device=vertices.device)
    raise NotImplementedError("Surface-current mesh refinement exceeded its explicit depth cap.")


def compile_integrated_traces(vertices, triangles, nodes, *, maximum_constraints=4096, include_magnetic=False):
    """Conforming RWG surface-current tests of actual tangential Yee traces."""
    spacing = min(float((axis[1:]-axis[:-1]).min()) for axis in nodes)
    vertices, triangles = _refine_current_mesh(vertices, triangles, spacing)
    incidence = {}
    for face, (a,b,c) in enumerate(triangles.cpu().tolist()):
        for i,j,opposite in ((a,b,c), (b,c,a), (c,a,b)):
            incidence.setdefault(tuple(sorted((i,j))), []).append((face, opposite))
    edges = [(edge, entries) for edge, entries in incidence.items() if len(entries) == 2]
    if not edges:
        raise ValueError("Surface-current trace space has no internal edge; refine the triangle mesh.")
    if len(edges) > maximum_constraints:
        raise NotImplementedError(f"Cut-surface constraint count {len(edges)} exceeds the explicit {maximum_constraints} row cap.")
    points, weights, cells, normals, face_ids = clipped_surface_quadrature(vertices, triangles, nodes, include_face_ids=True)
    cell_shape = tuple(len(axis)-1 for axis in nodes)
    flat_cell = (cells[:, 0]*cell_shape[1]+cells[:, 1])*cell_shape[2]+cells[:, 2]
    unique, inverse = torch.unique(flat_cell, sorted=True, return_inverse=True)
    area = weights.new_zeros(len(unique)).index_add_(0, inverse, weights)
    mean = normals.new_zeros((len(unique),3)).index_add_(0,inverse,weights[:,None]*normals)
    mean /= torch.linalg.vector_norm(mean,dim=-1,keepdim=True).clamp_min(torch.finfo(mean.dtype).tiny)
    if bool(torch.any((mean[inverse]*normals).sum(-1) < 2**-.5)):
        raise ValueError("SurfaceMesh normal variation exceeds 45 degrees inside a Yee cell; refine the geometry/grid.")
    # Each oriented facet has at most three supported surface-current functions.
    host_rows = [[-1]*3 for _ in range(len(triangles))]
    host_opposite = [[0]*3 for _ in range(len(triangles))]
    host_sign = [[0.]*3 for _ in range(len(triangles))]
    slots = [0]*len(triangles)
    for row, (edge, entries) in enumerate(edges):
        for sign, (face, opposite) in zip((1.,-1.),entries):
            slot = slots[face]
            host_rows[face][slot], host_opposite[face][slot], host_sign[face][slot] = row, opposite, sign
            slots[face] += 1
    face_rows = torch.tensor(host_rows, device=vertices.device)
    face_opposite = torch.tensor(host_opposite, device=vertices.device)
    face_sign = vertices.new_tensor(host_sign)
    edge_indices = torch.tensor([edge for edge, _entries in edges], device=vertices.device)
    lengths = torch.linalg.vector_norm(vertices[edge_indices[:,1]]-vertices[edge_indices[:,0]],dim=-1)
    tri = vertices[triangles]
    twice_area = torch.linalg.vector_norm(torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),dim=-1)
    row_ids = face_rows[face_ids]
    current = (points[:,None]-vertices[face_opposite[face_ids]])
    current *= (face_sign[face_ids]*lengths[row_ids.clamp_min(0)]/twice_area[face_ids,None])[...,None]
    valid = row_ids >= 0
    row_norm = weights.new_zeros(len(edges)).index_add_(0,row_ids[valid],
        (weights[:,None]*current.square().sum(-1))[valid]).sqrt()
    current /= row_norm[row_ids.clamp_min(0),None]
    local = torch.stack([(points[:,axis]-coordinates[cells[:,axis]])/
        (coordinates[cells[:,axis]+1]-coordinates[cells[:,axis]]) for axis,coordinates in enumerate(nodes)],-1)
    shapes = ((cell_shape[0],cell_shape[1]+1,cell_shape[2]+1),
              (cell_shape[0]+1,cell_shape[1],cell_shape[2]+1),
              (cell_shape[0]+1,cell_shape[1]+1,cell_shape[2]))
    rows, columns, values, offset = [], [], [], 0
    for component, shape in enumerate(shapes):
        transverse = [axis for axis in range(3) if axis != component]
        for a,b in product((0,1),repeat=2):
            index=cells.clone()
            index[:,transverse[0]] += a
            index[:,transverse[1]] += b
            column=(index[:,0]*shape[1]+index[:,1])*shape[2]+index[:,2]+offset
            interpolation=(local[:,transverse[0]] if a else 1-local[:,transverse[0]])
            interpolation = interpolation*(local[:,transverse[1]] if b else 1-local[:,transverse[1]])
            rows.append(row_ids[valid])
            columns.append(column[:,None].expand(-1,3)[valid])
            values.append((weights[:,None]*interpolation[:,None]*current[:,:,component])[valid])
        offset += shape[0]*shape[1]*shape[2]
    matrix=torch.sparse_coo_tensor(torch.stack((torch.cat(rows),torch.cat(columns))),torch.cat(values),(len(edges),offset)).coalesce()
    keep=matrix.values()!=0
    indices,value=matrix.indices()[:,keep],matrix.values()[keep]
    support,local_column=torch.unique(indices[1],sorted=True,return_inverse=True)
    if len(edges)*len(support)*8 > 512*1024**2:
        raise NotImplementedError("Cut-surface float64 rank-reveal matrix exceeds the explicit 512 MiB cap.")
    matrix=torch.sparse_coo_tensor(torch.stack((indices[0],local_column)),value,(len(edges),len(support))).coalesce()
    result = dict(matrix=matrix,support=support,shapes=shapes,cell_count=len(unique),constraint_count=len(edges),
        cells=unique,area=area.sum(),minimum_cell_area=area.min(),maximum_cell_area=area.max())
    if not include_magnetic:
        return result
    # Separate H-face Raviart--Thomas and E-edge Whitney trace spaces share the
    # physical current basis and its full L2 mass, not a diagonal substitute.
    mass_valid = valid[:,:,None] & valid[:,None,:]
    mass_rows = row_ids[:,:,None].expand(-1,3,3)[mass_valid]
    mass_columns = row_ids[:,None,:].expand(-1,3,3)[mass_valid]
    mass_values = (weights[:,None,None]*torch.einsum("nsi,nti->nst",current,current))[mass_valid]
    mass = torch.sparse_coo_tensor(torch.stack((mass_rows,mass_columns)),mass_values,
                                   (len(edges),len(edges))).coalesce().to_dense()
    magnetic_shapes = ((cell_shape[0]+1,cell_shape[1],cell_shape[2]),
                       (cell_shape[0],cell_shape[1]+1,cell_shape[2]),
                       (cell_shape[0],cell_shape[1],cell_shape[2]+1))
    rows, columns, values, offset = [], [], [], 0
    for component, shape in enumerate(magnetic_shapes):
        for endpoint in (0,1):
            index = cells.clone()
            index[:,component] += endpoint
            column = (index[:,0]*shape[1]+index[:,1])*shape[2]+index[:,2]+offset
            interpolation = local[:,component] if endpoint else 1-local[:,component]
            rows.append(row_ids[valid])
            columns.append(column[:,None].expand(-1,3)[valid])
            values.append((weights[:,None]*interpolation[:,None]*current[:,:,component])[valid])
        offset += shape[0]*shape[1]*shape[2]
    magnetic = torch.sparse_coo_tensor(torch.stack((torch.cat(rows),torch.cat(columns))),torch.cat(values),
                                       (len(edges),offset)).coalesce()
    keep = magnetic.values()!=0
    indices, value = magnetic.indices()[:,keep],magnetic.values()[keep]
    magnetic_support, local_columns = torch.unique(indices[1],sorted=True,return_inverse=True)
    if len(edges)*(len(support)+len(magnetic_support))*8 > 512*1024**2:
        raise NotImplementedError("Combined copper mortar rank-reveal matrices exceed the explicit 512 MiB cap.")
    magnetic = torch.sparse_coo_tensor(torch.stack((indices[0],local_columns)),value,
                                       (len(edges),len(magnetic_support))).coalesce()
    point_rows = (3*torch.arange(len(points),device=points.device)[:,None,None]+
                  torch.arange(3,device=points.device)[None,None,:]).expand(-1,3,-1)
    point_columns = row_ids[:,:,None].expand(-1,-1,3)
    point_valid = valid[:,:,None].expand(-1,-1,3)
    evaluation = torch.sparse_coo_tensor(torch.stack((point_rows[point_valid],point_columns[point_valid])),
        current[point_valid],(3*len(points),len(edges))).coalesce()
    result.update(surface_mass=mass,magnetic_matrix=magnetic,magnetic_support=magnetic_support,
        magnetic_shapes=magnetic_shapes,surface_basis=evaluation,surface_points=points,
        surface_weights=weights,surface_normals=normals,quadrature_cells=cells)
    return result
