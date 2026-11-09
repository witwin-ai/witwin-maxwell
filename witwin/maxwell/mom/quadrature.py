"""CUDA triangle quadrature and source-centred singularity removal.

The inner Duffy fan has signed area, so the same formula also integrates a
triangle when the projected observation point is outside it. Orders are
explicit numerical parameters; convergence must be measured, not inferred.
"""
from __future__ import annotations

import math
from functools import lru_cache
import torch


def gauss_rule(order, *, device, dtype=torch.float64):
    """Gauss--Legendre nodes on [0, 1], computed on the target GPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("MoM numerical quadrature requires CUDA.")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("Quadrature order must be a positive integer.")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return _cached_gauss_rule(order, device=resolved, dtype=dtype)


@lru_cache(maxsize=64)
def _cached_gauss_rule(order, *, device, dtype):
    """Cache by concrete CUDA index, never by the mutable current device."""
    j = torch.arange(1, order, device=device, dtype=dtype)
    off = j / torch.sqrt(4 * j * j - 1)
    jacobi = torch.diag(off, 1) + torch.diag(off, -1)
    nodes, vectors = torch.linalg.eigh(jacobi)
    return (nodes + 1) / 2, vectors[0].square()


def triangle_rule(triangles, order):
    """Return points [..., Q, 3] and physical weights [..., Q]."""
    u, w = gauss_rule(order, device=triangles.device, dtype=triangles.dtype)
    a, b = torch.meshgrid(u, u, indexing="ij")
    weights = (w[:, None] * w[None, :] * a).flatten()
    a, b = a.flatten(), b.flatten()
    v0, v1, v2 = triangles.unbind(-2)
    points = v0[..., None, :] + a[:, None] * (
        (1 - b[:, None]) * (v1 - v0)[..., None, :]
        + b[:, None] * (v2 - v0)[..., None, :]
    )
    area2 = torch.linalg.vector_norm(torch.linalg.cross(v1-v0, v2-v0), dim=-1)
    return points, area2[..., None] * weights


def duffy_fan(observers, triangles, order):
    """Pairwise projected fan for observers [B,3], triangles [B,3,3].

    Returns inner nodes [B,3*order**2,3] and signed area weights. For
    coplanar observers inside a source triangle, the radial Jacobian
    cancels the Green-function 1/r singularity without an epsilon cutoff.
    """
    u, w = gauss_rule(order, device=triangles.device, dtype=triangles.dtype)
    a, b = torch.meshgrid(u, u, indexing="ij")
    weight = (w[:, None] * w[None, :] * a).flatten()
    a, b = a.flatten(), b.flatten()
    normal = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    height = ((observers-triangles[:, 0])*normal).sum(-1, keepdim=True)
    centre = observers-height*normal
    edge0 = triangles-centre[:, None, :]
    edge1 = torch.roll(edge0, -1, 1)
    signed_area2 = (torch.linalg.cross(edge0, edge1)*normal[:, None, :]).sum(-1)
    radial = (1-b[None, None, :, None])*edge0[:, :, None, :] + b[None, None, :, None]*edge1[:, :, None, :]
    points = centre[:, None, None, :] + a[None, None, :, None]*radial
    weights = signed_area2[:, :, None]*weight
    area2 = torch.linalg.vector_norm(torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), dim=-1)
    degenerate = signed_area2.abs() <= 32*torch.finfo(triangles.dtype).eps*area2[:, None]
    # An observer projected onto an edge gives a zero-area fan. Its odd-order
    # midpoint may coincide with the observer; omit that fan before kernels.
    points = torch.where(degenerate[:, :, None, None], triangles.mean(1)[:, None, None], points)
    weights = torch.where(degenerate[:, :, None], torch.zeros_like(weights), weights)
    return points.flatten(1, 2), weights.flatten(1, 2)


def distance_scaled_line_rule(along,distance,length,order):
    """Physical edge arclength and weights clustered by observer-line distance.

    For arclength s and projection p, use s=p+d*sinh(u) with exact
    endpoints and Jacobian. Separated edges and exact zero distance retain
    the ordinary full-segment rule; callers must separately reject an
    observer on the integration segment. No part of the edge is removed.
    """
    u,w=gauss_rule(order,device=along.device,dtype=along.dtype)
    if not bool(torch.isfinite(along).all() & torch.isfinite(distance).all()
                & torch.isfinite(length).all() & (distance>=0).all() & (length>0).all()):
        raise ValueError('Graded edge geometry is not representable')
    # Far, same-side endpoints can round to the same asinh value. Such an
    # edge is smooth and needs no grading. Keep inactive transform inputs
    # benign as well, so native backward cannot encounter inf*0 there.
    ordinary=(distance==0)|(distance>=length)|(along<=-length)|(along>=2*length)
    safe=torch.where(ordinary,torch.ones_like(distance),distance)
    projection=torch.where(ordinary,torch.zeros_like(along),along)
    extent=torch.where(ordinary,torch.ones_like(length),length)
    left=-projection/safe;right=(extent-projection)/safe
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError('Graded edge distance ratio is not representable')
    lower=torch.asinh(left);span=torch.asinh(right)-lower
    transformed=lower[...,None]+span[...,None]*u
    position=projection[...,None]+safe[...,None]*torch.sinh(transformed)
    weight=safe[...,None]*torch.cosh(transformed)*span[...,None]*w
    position=torch.where(ordinary[...,None],length[...,None]*u,position)
    weight=torch.where(ordinary[...,None],length[...,None]*w,weight)
    if not bool(torch.isfinite(position).all() & torch.isfinite(weight).all()
                & (weight>0).all() & (position>=0).all() & (position<=length[...,None]).all()
                & (position.diff(dim=-1)>0).all()):
        raise ValueError('Graded edge rule is not representable')
    return position,weight


def distance_scaled_duffy_fan(observers,triangles,order,*,edge_grading=False,coplanar_tolerance=None):
    """Signed projected fans with a distance-scaled radial coordinate.

    Pairwise observers [B,3] and triangles [B,3,3] produce 3*order**2
    nodes per face. For face distance h and directional radius R, map
    a=sinh(u*asinh(R/h))/(R/h), retaining its exact radial Jacobian.
    Exact coplanar observers use the ordinary fan; no distance cutoff is
    added. Optional edge grading also clusters the angular edge parameter by
    the physical observer-line distance. Well-separated pairs use the full
    ordinary triangle rule to avoid cancellation of large signed fans.
    With ``coplanar_tolerance``, heights below that fraction of the face
    diameter count as exactly coplanar: roundoff heights of rotated coplanar
    panels would otherwise select an extreme sinh map. Unrepresentable
    geometry raises.
    """
    if (observers.ndim!=2 or observers.shape[-1]!=3 or triangles.shape!=(len(observers),3,3)
            or observers.device!=triangles.device or observers.dtype!=triangles.dtype
            or not bool(torch.isfinite(observers).all() & torch.isfinite(triangles).all())):
        raise ValueError('Graded fan needs finite matching point/triangle pairs')
    if type(edge_grading) is not bool:
        raise ValueError('Fan edge grading must be boolean')
    u,w=gauss_rule(order,device=triangles.device,dtype=triangles.dtype)
    normal=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area2=normal.norm(dim=-1)
    if not bool(torch.isfinite(area2).all() and (area2>0).all()):
        raise ValueError('Graded fan requires finite positive face area')
    local=triangles-triangles[:,:1]
    diameter=(local-local.roll(-1,1)).norm(dim=-1).amax(-1)
    distance=(observers-triangles[:,0]-local.mean(1)).norm(dim=-1)
    separated=distance>=2*diameter
    if bool(separated.any()):
        # Preserve the 3*order**2 layout. Only the first ordinary copy has
        # nonzero weights; no source area is removed or approximated away.
        nodes,weights=triangle_rule(triangles[separated],order)
        far_nodes=nodes.repeat(1,3,1)
        far_weights=torch.cat((weights,torch.zeros_like(weights),torch.zeros_like(weights)),1)
        points=triangles.new_zeros(len(observers),3*order**2,3)
        measure=triangles.new_zeros(len(observers),3*order**2)
        ids=separated.nonzero().flatten()
        points=points.index_copy(0,ids,far_nodes);measure=measure.index_copy(0,ids,far_weights)
        if bool((~separated).any()):
            near_nodes,near_weights=distance_scaled_duffy_fan(observers[~separated],triangles[~separated],
                order,edge_grading=edge_grading,coplanar_tolerance=coplanar_tolerance)
            ids=(~separated).nonzero().flatten()
            points=points.index_copy(0,ids,near_nodes);measure=measure.index_copy(0,ids,near_weights)
        return points,measure
    normal=normal/area2[:,None]
    signed_height=((observers-triangles[:,0])*normal).sum(-1)
    if coplanar_tolerance is not None:
        signed_height=torch.where(signed_height.abs()<=coplanar_tolerance*diameter,torch.zeros_like(signed_height),signed_height)
    centre=observers-signed_height[:,None]*normal
    left=triangles-centre[:,None];right=left.roll(-1,1)
    edge=triangles.roll(-1,1)-triangles
    signed_area=(torch.linalg.cross(left,edge)*normal[:,None]).sum(-1)
    angular_weights=w[None,None,:]
    if edge_grading:
        length=edge.norm(dim=-1);unit=edge/length[:,:,None]
        along=-(left*unit).sum(-1)
        perpendicular=left+along[:,:,None]*unit
        distance=torch.hypot(perpendicular.norm(dim=-1),signed_height[:,None])
        position,edge_weights=distance_scaled_line_rule(along,distance,length,order)
        parameter=position/length[:,:,None]
        direction=left[:,:,None]+parameter[:,:,:,None]*edge[:,:,None]
        angular_weights=edge_weights/length[:,:,None]
    else:
        direction=(1-u[None,None,:,None])*left[:,:,None]+u[None,None,:,None]*right[:,:,None]
    radius=direction.norm(dim=-1)
    if not bool(torch.isfinite(signed_height).all() and torch.isfinite(signed_area).all()
                and torch.isfinite(centre).all() and torch.isfinite(radius).all()):
        raise ValueError('Graded fan intermediate geometry is not finite')
    height=signed_height.abs()[:,None,None]
    # Exact zero height uses the ordinary fan; its radial volume cone is
    # degenerate. Exact zero directional length is a zero-area fan limit.
    safe_height=torch.where(height==0,torch.ones_like(height),height)
    ratio=radius/safe_height
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError('Graded fan radial scale is not representable')
    safe_ratio=torch.where(ratio==0,torch.ones_like(ratio),ratio)
    stretch=torch.asinh(ratio)
    mapped=torch.sinh(u[None,None,:,None]*stretch[:,:,None,:])/safe_ratio[:,:,None,:]
    jacobian=stretch[:,:,None,:]*torch.cosh(u[None,None,:,None]*stretch[:,:,None,:])/safe_ratio[:,:,None,:]
    linear=(height==0)|(ratio==0)
    mapped=torch.where(linear[:,:,None,:],u[None,None,:,None],mapped)
    jacobian=torch.where(linear[:,:,None,:],torch.ones_like(jacobian),jacobian)
    points=centre[:,None,None,None,:]+mapped[...,None]*direction[:,:,None,:,:]
    weights=signed_area[:,:,None,None]*mapped*jacobian*w[None,None,:,None]*angular_weights[:,:,None,:]
    degenerate=signed_area.abs()<=32*torch.finfo(triangles.dtype).eps*area2[:,None]
    points=torch.where(degenerate[:,:,None,None,None],triangles.mean(1)[:,None,None,None,:],points)
    weights=torch.where(degenerate[:,:,None,None],torch.zeros_like(weights),weights)
    if not bool(torch.isfinite(points).all() and torch.isfinite(weights).all()):
        raise ValueError('Graded fan geometry is not representable at this precision')
    return points.flatten(1,3),weights.flatten(1,3)


def _cube_rule(order, dimensions, *, device, dtype):
    nodes, weights = gauss_rule(order, device=device, dtype=dtype)
    coordinates = torch.meshgrid(*([nodes] * dimensions), indexing="ij")
    factors = torch.meshgrid(*([weights] * dimensions), indexing="ij")
    product = torch.ones_like(factors[0])
    for factor in factors:
        product = product * factor
    return [coordinate.flatten() for coordinate in coordinates], product.flatten()


def touching_pair_rule(left, right, left_ids, right_ids, category, order):
    """Joint four-dimensional Duffy rules for self, edge and vertex pairs.

    Return paired observation/source nodes [B,Q,3] and product-area weights
    [B,Q]. These are paired nodes, not a Cartesian product of triangle rules.
    Shared physical vertices are located by the supplied global indices.

    Self: six sectors of the barycentric difference, with radial Jacobian
    rho*(1-rho)^2. Edge: two orderings of edge coordinates times three
    maximal-coordinate sectors of (normal distance, normal distance, edge
    separation), with radial Jacobian rho^2. Vertex: two radial orderings,
    with Jacobian rho^3*tau. Thus the actual singularity is transformed in
    the joint domain rather than approached with a higher ordinary order.
    """
    if category not in (0, 1, 2):
        raise ValueError("Joint touching rule requires self, edge or vertex category")
    area_left = torch.linalg.vector_norm(torch.linalg.cross(left[:, 1]-left[:, 0], left[:, 2]-left[:, 0]), dim=-1)
    area_right = torch.linalg.vector_norm(torch.linalg.cross(right[:, 1]-right[:, 0], right[:, 2]-right[:, 0]), dim=-1)
    observations, sources, weights = [], [], []
    if category == 0:
        z, wz = triangle_rule(left, order)
        (rho, t), wc = _cube_rule(order, 2, device=left.device, dtype=left.dtype)
        rho, t = rho[None, :, None, None], t[None, :, None, None]
        wz = wz[:, None] * wc[None, :, None]
        weight = area_left[:, None, None] * rho[..., 0] * (1-rho[..., 0]).square() * wz
        for i in range(3):
            vi = left[:, i, None, None]
            vj, vk = left[:, (i+1) % 3, None, None], left[:, (i+2) % 3, None, None]
            x = (1-rho)*z[:, None] + rho*vi
            y = (1-rho)*z[:, None] + rho*((1-t)*vj+t*vk)
            for obs, src in ((x, y), (y, x)):
                observations.append(obs.flatten(1, 2)); sources.append(src.flatten(1, 2))
                weights.append(weight.flatten(1, 2))
    else:
        shared = (left_ids[:, :, None] == right_ids[:, None, :]).any(-1)
        shared_right = (right_ids[:, :, None] == left_ids[:, None, :]).any(-1)
        batch_ids = torch.arange(len(left), device=left.device)
        (rho, alpha, beta, t), wc = _cube_rule(order, 4, device=left.device, dtype=left.dtype)
        if category == 2:
            # Move the common vertex first, preserving either triangle's shape.
            li = shared.long().argmax(-1)
            ri = shared_right.long().argmax(-1)
            common = left[batch_ids, li]
            la, lb = left[batch_ids, (li+1) % 3]-common, left[batch_ids, (li+2) % 3]-common
            ra, rb = right[batch_ids, (ri+1) % 3]-common, right[batch_ids, (ri+2) % 3]-common
            ld = (1-beta[None, :, None])*la[:, None] + beta[None, :, None]*lb[:, None]
            rd = (1-t[None, :, None])*ra[:, None] + t[None, :, None]*rb[:, None]
            weight = area_left[:, None]*area_right[:, None]*(wc*rho**3*alpha)[None]
            for r, s in ((rho, rho*alpha), (rho*alpha, rho)):
                observations.append(common[:, None]+r[None, :, None]*ld)
                sources.append(common[:, None]+s[None, :, None]*rd)
                weights.append(weight)
        else:
            # Common edge is ordered by global vertex id, independently of faces.
            maximum = torch.iinfo(left_ids.dtype).max
            ordered = torch.where(shared, left_ids, maximum).argsort(-1)
            a, b = left[batch_ids, ordered[:, 0]], left[batch_ids, ordered[:, 1]]
            c = left[batch_ids, (~shared).long().argmax(-1)]
            d = right[batch_ids, (~shared_right).long().argmax(-1)]
            for r, s, separation in ((rho, rho*alpha, rho*beta), (rho*alpha, rho, rho*beta), (rho*alpha, rho*beta, rho)):
                weight = area_left[:, None]*area_right[:, None]*(wc*rho.square()*(1-separation)*(1-r)*(1-s))[None]
                low, high = t*(1-separation), t*(1-separation)+separation
                for tx, ty in ((high, low), (low, high)):
                    edge_x = a[:, None]+tx[None, :, None]*(b-a)[:, None]
                    edge_y = a[:, None]+ty[None, :, None]*(b-a)[:, None]
                    observations.append((1-r)[None, :, None]*edge_x+r[None, :, None]*c[:, None])
                    sources.append((1-s)[None, :, None]*edge_y+s[None, :, None]*d[:, None])
                    weights.append(weight)
    return torch.cat(observations, 1), torch.cat(sources, 1), torch.cat(weights, 1)


def green(displacement, k):
    """Outgoing scalar Green function and its observation gradient."""
    from .kernels import FREE_SPACE
    return FREE_SPACE.evaluate(displacement, k)


def _pair_geometry(mesh):
    triangles = mesh.vertices[mesh.faces]
    centre = triangles.mean(1)
    radius = torch.linalg.vector_norm(triangles-centre[:, None], dim=-1).amax(-1)
    return centre, radius


def _classify_pairs(mesh, rows, cols, centre, radius, near_ratio):
    left, right = mesh.faces[rows], mesh.faces[cols]
    shared = (left[:, :, None] == right[:, None, :]).sum((1, 2))
    near = torch.linalg.vector_norm(centre[rows]-centre[cols], dim=-1) < near_ratio*(radius[rows]+radius[cols])
    result = torch.where(near, 3, 4)
    return torch.where(shared == 3, 0, torch.where(shared == 2, 1, torch.where(shared == 1, 2, result)))


def classify_pairs(mesh, rows, cols, near_ratio=2.0):
    """0=self, 1=edge, 2=vertex, 3=near-disjoint, 4=far."""
    centre, radius = _pair_geometry(mesh)
    return _classify_pairs(mesh, rows, cols, centre, radius, near_ratio)


def classify_all_pairs(mesh, *, batch_size=65536):
    """Classify ordered face pairs with bounded CUDA geometry temporaries.

    The compact category vector remains quadratic, as required by the dense
    assembly traversal. Only classification workspace is bounded here.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Pair classification batch size must be a positive integer.")
    if mesh.device.type != "cuda":
        raise ValueError("MoM pair classification requires CUDA.")
    count = len(mesh.faces)
    centre, radius = _pair_geometry(mesh)
    result = torch.empty(count*count, dtype=torch.uint8, device=mesh.device)
    for start in range(0, count*count, batch_size):
        stop = min(start+batch_size, count*count)
        ids = torch.arange(start, stop, device=mesh.device)
        result[start:stop] = _classify_pairs(mesh, ids//count, ids % count, centre, radius, 2.0)
    return result


def triangle_outside_inscribed_disk(observers, triangles, order):
    """Integrate a face minus its largest observer-centered interior disk.

    Observers must be strict face-interior points. The removed disk's integral
    of any odd displacement kernel is exactly zero in the Cauchy principal
    value sense. Each oriented edge defines an angular sector and its exact
    radial boundary. No geometric cutoff or normal displacement is introduced.
    """
    if (observers.ndim != 2 or observers.shape[-1] != 3 or triangles.shape != (len(observers), 3, 3)
            or observers.device.type != 'cuda' or triangles.device != observers.device
            or triangles.dtype != observers.dtype or not bool(torch.isfinite(observers).all() & torch.isfinite(triangles).all())):
        raise ValueError('Disk-complement quadrature requires matching finite CUDA point/triangle pairs')
    nodes, weights = gauss_rule(order, device=observers.device, dtype=observers.dtype)
    edge = torch.roll(triangles, -1, 1)-triangles
    normal = torch.linalg.cross(edge[:, 0], -edge[:, 2])
    normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    conormal = torch.linalg.cross(edge, normal[:, None].expand_as(edge))
    conormal = conormal/torch.linalg.vector_norm(conormal, dim=-1, keepdim=True)
    radius_vector = triangles-observers[:, None]
    distance = (radius_vector*conormal).sum(-1)
    scale = torch.linalg.vector_norm(edge, dim=-1).amax(-1)
    tolerance = 128*torch.finfo(observers.dtype).eps*scale
    height = ((observers-triangles[:, 0])*normal).sum(-1).abs()
    if bool((height > tolerance).any() | (distance <= tolerance[:, None]).any()):
        raise ValueError('Disk-complement quadrature requires a strict face-interior observer')
    inner = distance.amin(-1)
    tangent = radius_vector[:, 0]/torch.linalg.vector_norm(radius_vector[:, 0], dim=-1, keepdim=True)
    bitangent = torch.linalg.cross(normal, tangent)
    next_vector = torch.roll(radius_vector, -1, 1)
    angle = torch.atan2((radius_vector*bitangent[:, None]).sum(-1), (radius_vector*tangent[:, None]).sum(-1))
    span = torch.atan2((torch.linalg.cross(radius_vector, next_vector)*normal[:, None]).sum(-1),
                       (radius_vector*next_vector).sum(-1))
    theta = angle[:, :, None]+span[:, :, None]*nodes
    direction = torch.cos(theta)[..., None]*tangent[:, None, None]+torch.sin(theta)[..., None]*bitangent[:, None, None]
    outer = distance[:, :, None]/(direction*conormal[:, :, None]).sum(-1)
    extent = outer-inner[:, None, None]
    radius = inner[:, None, None, None]+extent[..., None]*nodes
    source = observers[:, None, None, None]+radius[..., None]*direction[:, :, :, None]
    area_weights = radius*extent[..., None]*span[:, :, None, None]*weights[None, None, :, None]*weights
    return source.flatten(1, 3), area_weights.flatten(1, 3), inner


def _segment_distance(p, u, q, v):
    """Paired distances between segments p + s u and q + t v, s, t in [0, 1]."""
    r = p-q
    a, b, c = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
    d, e = (u*r).sum(-1), (v*r).sum(-1)
    denominator = a*c-b*b
    valid = denominator > torch.finfo(p.dtype).tiny
    s = torch.where(valid, (b*e-c*d)/torch.where(valid, denominator, torch.ones_like(denominator)), 0.).clamp(0, 1)
    t = (b*s+e)/c
    s = torch.where(t < 0, (-d/a).clamp(0, 1), s)
    s = torch.where(t > 1, ((b-d)/a).clamp(0, 1), s)
    return torch.linalg.vector_norm(r+s[..., None]*u-t.clamp(0, 1)[..., None]*v, dim=-1)


def source_feature_distances(triangles, sources):
    """Paired distances from triangles [N,3,3] to source vertices [N] and edges [N,3].

    Edge ``j`` joins source vertices ``j`` and ``j+1``. Near-singular source
    integrals vary on the scale of these distances; an observer projected onto
    a source interior far from its edges does not.
    """
    from .mesh import _point_triangle_distance_squared
    vertex = torch.full(triangles.shape[:1], torch.inf, device=triangles.device, dtype=triangles.dtype)
    edges = []
    normal = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    scale = torch.linalg.vector_norm(triangles-triangles.roll(1, 1), dim=-1).amax(-1)
    tolerance = (64*torch.finfo(triangles.dtype).eps*scale).square()
    for j in range(3):
        vertex = torch.minimum(vertex, _point_triangle_distance_squared(sources[:, j], triangles).sqrt())
    for j in range(3):
        a, b = sources[:, j], sources[:, (j+1) % 3]
        best = torch.minimum(_point_triangle_distance_squared(a, triangles), _point_triangle_distance_squared(b, triangles)).sqrt()
        for i in range(3):
            best = torch.minimum(best, _segment_distance(triangles[:, i], triangles[:, (i+1) % 3]-triangles[:, i], a, b-a))
        height_a, height_b = ((a-triangles[:, 0])*normal).sum(-1), ((b-triangles[:, 0])*normal).sum(-1)
        crossing = height_a*height_b < 0
        fraction = height_a/torch.where(crossing, height_a-height_b, torch.ones_like(height_a))
        # A transversal crossing through the triangle means zero distance; the
        # projected crossing carries only roundoff relative to the triangle.
        inside = crossing & (_point_triangle_distance_squared(a+fraction[:, None]*(b-a), triangles) <= tolerance)
        edges.append(torch.where(inside, torch.zeros_like(best), best))
    return vertex, torch.stack(edges, -1)


def source_boundary_distance(triangles, sources):
    """Paired distance [N] from triangles [N,3,3] to the boundary of sources [N,3,3]."""
    vertex, edges = source_feature_distances(triangles, sources)
    return torch.minimum(vertex, edges.amin(-1))


def refine_toward_source_edges(observers, sources, *, ratio, max_level, isotropic=False):
    """Anisotropic bisection of observer triangles toward source edges and vertices.

    Near a source vertex the integrated potential varies on the scale of the
    distance in every direction; near a source edge only across that edge. A
    leaf is therefore split while its distance to a source vertex is below
    ``ratio`` times its diameter (bisecting its longest edge), or its distance
    to a source edge is below ``ratio`` times its extent perpendicular to that
    edge (bisecting the leaf edge with the largest perpendicular component).
    Long slivers parallel to a nearby edge stay long, so the leaf count grows
    with the logarithm of the gap rather than with the aspect ratio. With
    ``isotropic=True`` every leaf is bisected along its longest edge while any
    source feature is closer than ``ratio`` diameters; that costlier variant
    is kept for independent reference integrals. Returns leaf triangles
    [L,3,3], their pair indices [L] and the depth reached.
    """
    triangles = observers
    owner = torch.arange(len(observers), device=observers.device)
    leaves, owners, level = [], [], 0
    while len(triangles):
        source = sources[owner]
        sides = triangles.roll(-1, 1)-triangles.roll(1, 1)  # side i is opposite vertex i
        lengths = torch.linalg.vector_norm(sides, dim=-1)
        vertex, edge = source_feature_distances(triangles, source)
        direction = source.roll(-1, 1)-source
        direction = direction/torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        # Leaf sides projected perpendicular to each source edge: [N, edge, side].
        along = torch.einsum('nec,nsc->nes', direction, sides)
        across = (lengths[:, None, :].square()-along.square()).clamp_min(0).sqrt()
        extent = across.amax(-1)
        near_vertex = vertex < ratio*lengths.amax(-1)
        near_edge = edge < ratio*(lengths.amax(-1, keepdim=True) if isotropic else extent)
        if isotropic:
            near_vertex = near_vertex | near_edge.any(-1)
        split = near_vertex | near_edge.any(-1)
        if level == max_level:
            split = torch.zeros_like(split)
        leaves.append(triangles[~split])
        owners.append(owner[~split])
        tightest = torch.where(near_edge, edge/extent.clamp_min(torch.finfo(edge.dtype).tiny), torch.inf).argmin(-1)
        rows = torch.arange(len(triangles), device=triangles.device)
        chosen = torch.where(near_vertex, lengths.argmax(-1), across[rows, tightest].argmax(-1))[split]
        parent = triangles[split]
        # Roll each parent so the chosen side joins local vertices 1 and 2.
        index = (torch.arange(3, device=parent.device)[None]+chosen[:, None]) % 3
        parent = parent.gather(1, index[..., None].expand(-1, -1, 3))
        apex, first, second = parent.unbind(1)
        middle = (first+second)/2
        triangles = torch.stack((torch.stack((apex, first, middle), 1),
                                 torch.stack((apex, middle, second), 1)), 1).reshape(-1, 3, 3)
        owner = owner[split].repeat_interleave(2)
        level += 1
    return torch.cat(leaves), torch.cat(owners), level-1


# Touching (self, shared-edge, shared-vertex) pairs whose larger normalized
# aspect ratio exceeds this value leave the fixed-order joint Duffy rules.
ANISOTROPIC_TOUCHING_ASPECT = 3.0
COPLANAR_TOLERANCE = 1e-10


def triangle_aspect(triangles):
    """Normalized aspect ratio [..., T]: longest edge times perimeter over 4 sqrt(3) area; 1 is equilateral."""
    lengths = torch.linalg.vector_norm(triangles.roll(-1, -2)-triangles, dim=-1)
    area = torch.linalg.vector_norm(torch.linalg.cross(triangles[..., 1, :]-triangles[..., 0, :],
                                                       triangles[..., 2, :]-triangles[..., 0, :]), dim=-1)/2
    return lengths.amax(-1)*lengths.sum(-1)/(4*math.sqrt(3)*area)


def anisotropic_touching_pairs(left, right):
    """Mask of touching pairs [B] that need the anisotropy-aware rule."""
    return torch.maximum(triangle_aspect(left), triangle_aspect(right)) > ANISOTROPIC_TOUCHING_ASPECT


def static_potential_integrals(points, triangles):
    """Closed-form flat-triangle potentials for paired points [N,3] and triangles [N,3,3].

    Returns, with R = |x - y| over y in the triangle,
    ``I0 = int 1/R``, the in-plane vector ``I1 = int (y - x_p)/R`` where x_p is
    the projection of x onto the triangle plane, and ``grad = grad_x int 1/R``.
    These are the edge-sum formulas for uniform and linear sources on planar
    polygons; they are exact for any triangle shape, which makes singularity
    subtraction robust for slivers. Each logarithm uses a cancellation-free
    form of ln((R+ + l+)/(R- + l-)); points on a source edge are excluded.
    """
    v0, v1, v2 = triangles.unbind(1)
    normal = torch.linalg.cross(v1-v0, v2-v0)
    normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    height = ((points-v0)*normal).sum(-1)
    projected = points-height[:, None]*normal
    absolute = height.abs()
    i0 = torch.zeros_like(height)
    i1 = torch.zeros_like(points)
    solid = torch.zeros_like(height)
    in_plane = torch.zeros_like(points)
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        length = torch.linalg.vector_norm(b-a, dim=-1, keepdim=True)
        tangent = (b-a)/length
        outward = torch.linalg.cross(tangent, normal)
        plus, minus = ((b-projected)*tangent).sum(-1), ((a-projected)*tangent).sum(-1)
        offset = ((a-projected)*outward).sum(-1)
        r0_squared = offset.square()+height.square()
        r_plus, r_minus = (plus.square()+r0_squared).sqrt(), (minus.square()+r0_squared).sqrt()
        # (R+l)(R-l) = R0^2 gives three cancellation-free forms by the signs of l-/l+.
        safe = r0_squared.clamp_min(torch.finfo(points.dtype).tiny)
        logarithm = torch.where(minus >= 0, torch.log((r_plus+plus)/(r_minus+minus)),
                    torch.where(plus <= 0, torch.log((r_minus-minus)/(r_plus-plus)),
                                torch.log(r_plus+plus)+torch.log(r_minus-minus)-torch.log(safe)))
        beta = (torch.atan2(offset*plus, r0_squared+absolute*r_plus)-torch.atan2(offset*minus, r0_squared+absolute*r_minus))
        i0 = i0+offset*logarithm-absolute*beta
        i1 = i1+.5*outward*(r0_squared*logarithm+plus*r_plus-minus*r_minus)[:, None]
        solid = solid+beta
        in_plane = in_plane+outward*logarithm[:, None]
    gradient = -normal*(torch.sign(height)*solid)[:, None]-in_plane
    return i0, i1, gradient



def touching_remainder_rule(left, right, left_ids, right_ids, category, order):
    """Paired nodes for the remainder of a touching pair after static subtraction.

    The remainder vanishes like R at coincidence, so the joint Duffy rules
    converge on it even for slivers, where they fail on 1/R itself.
    """
    return touching_pair_rule(left, right, left_ids, right_ids, category, order)


def touching_remainder_nodes(category, order):
    """Nodes per pair of ``touching_remainder_rule``, for memory chunking."""
    return (2 if category == 2 else 6)*order**4
