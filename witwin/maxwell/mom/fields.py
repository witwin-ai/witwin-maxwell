"""CUDA RWG field potentials, geometric query semantics and observables.

Surface currents are J=n×H and M=-n×E, with normal pointing into the
exterior. The exterior scattered field is F(J,M); the interior total field
is -F(J,M) evaluated with interior material parameters. Face-interior traces
use principal values plus explicit jumps. Edges and vertices fail explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import time

import torch

from ..constants import complex_dtype_for
from .operator import local_geometry
from .physics import incident_fields
from .quadrature import duffy_fan, green, triangle_rule


# Field-query workspace: queries per distance/selection chunk and quadrature
# samples per kernel batch.
QUERY_CHUNK = 64
SAMPLE_CAP = 262144


def _sum_observation_pairs(values, indices, count):
    """Reduce bounded pair batches without unordered CUDA atomic additions."""
    rows = torch.arange(count, device=indices.device)
    selection = (rows[:, None] == indices[None]).to(values.dtype)
    return selection @ values


@dataclass(frozen=True)
class FieldQueryConfig:
    relative_tolerance: float = 1e-5
    absolute_tolerance: float = 1e-9
    max_order: int = 64
    adaptive: bool = True

    def __post_init__(self):
        for name in ("relative_tolerance", "absolute_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.relative_tolerance == self.absolute_tolerance == 0:
            raise ValueError("A field query tolerance must be positive.")
        if isinstance(self.max_order, bool) or not isinstance(self.max_order, int) or self.max_order < 2:
            raise ValueError("max_order must be an integer >= 2.")
        if not isinstance(self.adaptive, bool):
            raise TypeError("adaptive must be boolean.")


def _validate_arrays(handle, coefficients, points):
    mesh = handle.basis.mesh
    if not isinstance(coefficients, torch.Tensor) or coefficients.device != mesh.device:
        raise ValueError("MoM coefficients and mesh must share a CUDA device.")
    if coefficients.shape != (2 * handle.basis.n_edges,) or coefficients.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("MoM coefficients must be one complex vector ordered [J, M].")
    if coefficients.dtype != complex_dtype_for(mesh.dtype):
        raise ValueError("MoM coefficient complex precision must match mesh real precision.")
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("MoM coefficients must be finite.")
    if not isinstance(points, torch.Tensor) or points.device != mesh.device or points.device.type != "cuda":
        raise ValueError("MoM query points and mesh must share a CUDA device.")
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype != mesh.dtype:
        raise ValueError("MoM queries must have shape (N, 3) and match mesh real precision.")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("MoM query coordinates must be finite.")


def _triangle_distance(points, triangles):
    """Distances for points [P,3] to triangles [T,3,3], including their edges."""
    return _pair_triangle_distance(points[:, None], triangles)


def _pair_triangle_distance(points, triangles):
    """Triangle distances for broadcast-compatible point and panel batches."""
    a, ab, ac = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    delta = points - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (delta * ab).sum(-1), (delta * ac).sum(-1)
    den = d00 * d11 - d01.square()
    v, w = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    projected = a + v[..., None] * ab + w[..., None] * ac
    distance2 = (points - projected).square().sum(-1)
    distance2 = torch.where(inside, distance2, torch.full_like(distance2, torch.inf))
    for edge in range(3):
        start, end = triangles[:, edge], triangles[:, (edge + 1) % 3]
        direction = end - start
        fraction = (((points - start) * direction).sum(-1) / direction.square().sum(-1)).clamp(0, 1)
        candidate = start + fraction[..., None] * direction
        distance2 = torch.minimum(distance2, (points - candidate).square().sum(-1))
    return distance2.clamp_min(0).sqrt()


def _solid_angle_inside(points, triangles):
    """Closed-surface interior test from the summed signed solid angle."""
    a, b, c = (triangles[None] - points[:, None, None]).unbind(-2)
    la, lb, lc = (torch.linalg.vector_norm(item, dim=-1) for item in (a, b, c))
    numerator = (a * torch.linalg.cross(b, c)).sum(-1)
    denominator = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
    solid_angle = (2 * torch.atan2(numerator, denominator)).sum(-1)
    return solid_angle.abs() > 2 * math.pi


def classify_queries(mesh, points, *, allow_surface=False):
    """Return interior mask and distances, plus face IDs when allowing traces."""
    triangles = mesh.triangles
    scale = torch.linalg.vector_norm(mesh.vertices.amax(0) - mesh.vertices.amin(0))
    tolerance = 128 * torch.finfo(mesh.dtype).eps * scale
    interiors, distances, surface_faces = [], [], []
    for batch in points.split(64):
        all_distances = _triangle_distance(batch, triangles)
        distance, face = all_distances.min(-1)
        on_surface = distance <= tolerance
        if bool(on_surface.any()):
            if not allow_surface:
                raise ValueError("MoM on-surface queries require explicit trace_side.")
            if bool(((all_distances <= tolerance).sum(-1)[on_surface] != 1).any()):
                raise ValueError("MoM edge and vertex traces are undefined; use a triangle-interior sensor.")
            # Also reject edge proximity without relying on neighbouring panels.
            tri = triangles[face[on_surface]]
            p = batch[on_surface]
            edge_distance = []
            for edge in range(3):
                a, b = tri[:, edge], tri[:, (edge + 1) % 3]
                edge_distance.append(torch.linalg.vector_norm(torch.linalg.cross(p-a, b-a), dim=-1) / torch.linalg.vector_norm(b-a, dim=-1))
            if bool((torch.stack(edge_distance).amin(0) <= tolerance).any()):
                raise ValueError("MoM edge and vertex traces are undefined; use a triangle-interior sensor.")
        surface_faces.append(torch.where(on_surface, face, -1))
        interiors.append(_solid_angle_inside(batch, triangles))
        distances.append(distance)
    if not len(points):
        result = (torch.empty(0, dtype=torch.bool, device=points.device), points.new_empty(0))
        return (*result, torch.empty(0, dtype=torch.long, device=points.device)) if allow_surface else result
    result = (torch.cat(interiors), torch.cat(distances))
    return (*result, torch.cat(surface_faces)) if allow_surface else result


def _static_edge_log(observers, first, last):
    """Integral of 1/r along a segment without near-edge norm subtraction."""
    edge=last-first;length=edge.norm(dim=-1);tangent=edge/length[:,None]
    delta=first-observers
    t0=(delta*tangent).sum(-1);t1=t0+length
    radius0=delta.norm(dim=-1);radius1=(last-observers).norm(dim=-1)
    perpendicular2=torch.linalg.cross(delta,tangent).square().sum(-1)
    # r0+r1-length equals the extension distance plus two rationalized
    # r-|t| terms. The direct difference rounds to zero near the segment.
    # amax splits ties at projected endpoints, cancelling the |t| derivative.
    extension=2*torch.stack((t0,-t1,torch.zeros_like(t0)),-1).amax(-1)
    gap=extension+perpendicular2/(radius0+t0.abs())+perpendicular2/(radius1+t1.abs())
    return torch.log1p(2*length/gap)


def _static_panel_gradient(observers, triangles, *, principal_value=None):
    """Exact static triangle gradient: edge logs plus oriented solid angle.

    The explicit principal-value mask removes the normal half jump only for
    a face-interior surface query. Off-plane signs follow source-minus-query
    vectors, so the normal derivative is negative above the oriented panel.
    """
    normal = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    gradient = torch.zeros_like(observers)
    for edge in range(3):
        a, b = triangles[:, edge], triangles[:, (edge+1) % 3]
        tangent = b-a
        length = torch.linalg.vector_norm(tangent, dim=-1)
        conormal = torch.linalg.cross(tangent / length[:, None], normal)
        line_integral = _static_edge_log(observers,a,b)
        gradient = gradient-conormal*line_integral[:, None]/(4*math.pi)
    a, b, c = (triangles-observers[:, None]).unbind(-2)
    la, lb, lc = (torch.linalg.vector_norm(vector, dim=-1) for vector in (a, b, c))
    numerator = (a*torch.linalg.cross(b, c)).sum(-1)
    denominator = la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
    solid_angle = 2*torch.atan2(numerator, denominator)
    if principal_value is not None:
        solid_angle = torch.where(principal_value, 0, solid_angle)
    return gradient+normal*solid_angle[:, None]/(4*math.pi)


def _static_panel_potential(observers, triangles, normals, gradients):
    """Exact Coulomb triangle potential from edge logs and normal derivative."""
    potential = observers.new_zeros(len(observers))
    for edge in range(3):
        a, b = triangles[:, edge], triangles[:, (edge+1)%3]
        tangent = b-a
        length = torch.linalg.vector_norm(tangent, dim=-1)
        conormal = torch.linalg.cross(tangent/length[:, None], normals)
        line_integral = _static_edge_log(observers,a,b)
        potential += ((a-observers)*conormal).sum(-1)*line_integral/(4*math.pi)
    height = ((triangles[:,0]-observers)*normals).sum(-1)
    return potential-height*(gradients*normals).sum(-1)

def _static_panel_vector_moment(observers, triangles, normals, potential):
    """Integral of (source-observer)/(4*pi*r) from planar divergence theorem."""
    moment = torch.zeros_like(observers)
    for edge in range(3):
        a, b = triangles[:, edge], triangles[:, (edge+1)%3]
        direction = b-a
        length = torch.linalg.vector_norm(direction, dim=-1)
        tangent = direction/length[:, None]
        conormal = torch.linalg.cross(tangent, normals)
        delta = a-observers
        t0 = (delta*tangent).sum(-1)
        t1 = t0+length
        r0 = torch.linalg.vector_norm(delta, dim=-1)
        r1 = torch.linalg.vector_norm(b-observers, dim=-1)
        distance2 = torch.linalg.cross(delta, tangent).square().sum(-1)
        line = _static_edge_log(observers,a,b)
        radial_integral = .5*(t1*r1-t0*r0+distance2*line)
        moment = moment+conormal*radial_integral[:, None]/(4*math.pi)
    height = ((triangles[:, 0]-observers)*normals).sum(-1)
    return moment+normals*(height*potential)[:, None]


def _potentials(handle, coefficients, queries, medium, *, order=None, singular_order=None, surface_faces=None, edge_indices=None):
    basis = handle.basis
    mesh, config = basis.mesh, handle.integration
    order = config.order if order is None else order
    singular_order = config.singular_order if singular_order is None else singular_order
    edges, free, slopes = local_geometry(basis)
    ne, triangles = basis.n_edges, mesh.triangles
    vector_input = coefficients.ndim == 1
    coefficients = coefficients[:, None] if vector_input else coefficients
    columns = coefficients.shape[1]
    panel_ids = torch.arange(len(triangles),device=mesh.device)
    if edge_indices is None:
        local_j = coefficients[:ne][edges] * slopes[..., None]
        local_m = coefficients[ne:][edges] * slopes[..., None]
    else:
        # Keep complete RWG support through face incidence, independent of
        # separate plus/minus lookup caches. Unselected local edges are zero.
        panel_ids = torch.isin(edges,edge_indices).any(-1).nonzero().flatten()
        edges,free,slopes = edges[panel_ids],free[panel_ids],slopes[panel_ids]
        triangles = triangles[panel_ids]
        lookup = torch.full((ne,),len(edge_indices),device=mesh.device,dtype=torch.int64)
        lookup[edge_indices] = torch.arange(len(edge_indices),device=mesh.device)
        ne = len(edge_indices)
        zero = coefficients.new_zeros(1,columns)
        local_j = torch.cat((coefficients[:ne],zero))[lookup[edges]]*slopes[...,None]
        local_m = torch.cat((coefficients[ne:],zero))[lookup[edges]]*slopes[...,None]
    div_j, div_m = 2 * local_j.sum(1), 2 * local_m.sum(1)
    panel_radius = torch.linalg.vector_norm(triangles - triangles.mean(1)[:, None], dim=-1).amax(-1)
    result_e, result_h = [], []
    for start in range(0, len(queries), QUERY_CHUNK):
        query = queries[start:start+QUERY_CHUNK]
        distances = _triangle_distance(query, triangles)
        obs_ids = torch.arange(len(query), device=query.device)[:, None].expand(-1, len(triangles)).flatten()
        tri_ids = torch.arange(len(triangles), device=query.device)[None].expand(len(query), -1).flatten()
        near = (distances < 2 * panel_radius).flatten()
        electric = torch.zeros((len(query), 3, columns), device=query.device, dtype=coefficients.dtype)
        magnetic = torch.zeros_like(electric)
        for is_near in (False, True):
            selected = (near == is_near).nonzero().flatten()
            # Near fans and far panels are both bounded by the quadrature
            # sample count, independently of the operator pair batches.
            pair_batch = max(1, SAMPLE_CAP // (3*singular_order**2 if is_near else order**2))
            for indices in selected.split(pair_batch):
                oi, ti = obs_ids[indices], tri_ids[indices]
                if is_near:
                    source, weights = duffy_fan(query[oi], triangles[ti], singular_order)
                else:
                    source, weights = triangle_rule(triangles[ti], order)
                delta = query[oi, None] - source
                g, grad = handle.kernel.evaluate(delta, medium.k)
                local_displacement = (source[:, :, None] - free[ti, None]).to(coefficients.dtype)
                # Integrate three geometric panel basis functions once, then
                # contract every source column without a quadrature-column axis.
                potential_basis = torch.einsum('pq,pqlc,pq->plc',g,local_displacement,weights.to(g.dtype))
                grad_integral = (grad * weights[..., None]).sum(1)
                curl_basis = (torch.linalg.cross(grad[:, :, None],local_displacement,dim=-1)
                              *weights[:, :, None, None]).sum(1)
                if is_near:
                    radius = torch.linalg.vector_norm(delta, dim=-1)
                    z = 1j * medium.k * radius
                    factor = z + (z-1) * torch.expm1(z)
                    # Stable series for the smooth gradient remainder near zero.
                    series = z*z/2 + z**3/3 + z**4/8 + z**5/30
                    factor = torch.where(z.abs() < 1e-3, series, factor)
                    remainder = factor[..., None] * delta / (4*math.pi*radius[..., None]**3)
                    self_panel = None if surface_faces is None else panel_ids[ti] == surface_faces[start+oi]
                    exact_gradient = _static_panel_gradient(query[oi], triangles[ti], principal_value=self_panel)
                    grad_integral = exact_gradient + (remainder * weights[..., None]).sum(1)
                    # Every panel current is affine with scalar slope. The term
                    # proportional to (source-query) vanishes in the cross product.
                    at_query = (query[oi, None]-free[ti]).to(coefficients.dtype)
                    normals = mesh.normals[panel_ids[ti]]
                    static_potential = _static_panel_potential(query[oi], triangles[ti], normals, exact_gradient)
                    static_moment = _static_panel_vector_moment(query[oi], triangles[ti], normals, static_potential)
                    smooth_g = torch.expm1(z)/(4*math.pi*radius)
                    potential_basis = (torch.einsum('pq,pqlc,pq->plc',smooth_g,local_displacement,weights.to(smooth_g.dtype))
                        + static_potential[:, None, None]*at_query+static_moment[:, None])
                    curl_basis = torch.linalg.cross(grad_integral[:, None],at_query,dim=-1)
                sj = torch.einsum('plr,plc->pcr',local_j[ti],potential_basis)
                sm = torch.einsum('plr,plc->pcr',local_m[ti],potential_basis)
                curl_j = torch.einsum('plr,plc->pcr',local_j[ti],curl_basis)
                curl_m = torch.einsum('plr,plc->pcr',local_m[ti],curl_basis)
                e = 1j * medium.omega * medium.mu * (sj + grad_integral[:, :, None] * div_j[ti, None] / medium.k**2) - curl_m
                h = curl_j + 1j * medium.omega * medium.epsilon * (sm + grad_integral[:, :, None] * div_m[ti, None] / medium.k**2)
                electric = electric + _sum_observation_pairs(e.to(coefficients.dtype).flatten(1), oi, len(query)).reshape(len(query),3,columns)
                magnetic = magnetic + _sum_observation_pairs(h.to(coefficients.dtype).flatten(1), oi, len(query)).reshape(len(query),3,columns)
        result_e.append(electric)
        result_h.append(magnetic)
    if not len(queries):
        empty = coefficients.new_empty((0, 3) if vector_input else (0,3,columns))
        return empty, empty.clone()
    results = (torch.cat(result_e), torch.cat(result_h))
    return tuple(value[...,0] for value in results) if vector_input else results


def decode_interior_columns(handle, coefficients, queries):
    """Fixed-order interior response columns, excluding boundary traces.

    The preparation caller owns successive-order convergence checks for each
    query/column. Geometry quadrature is shared; this function certifies none.
    """
    if (not isinstance(coefficients,torch.Tensor) or coefficients.ndim != 2
            or coefficients.shape[0] != 2*handle.basis.n_edges or coefficients.shape[1] < 1
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Interior field columns require finite [2*edges,columns] coefficients')
    _,_,faces,_ = _query_context(handle,coefficients[:,0],queries,1,'total',None,
                                  FieldQueryConfig(adaptive=False))
    order = max(handle.integration.order,handle.integration.singular_order)
    from .tensor_medium import TensorMedium
    if isinstance(handle.interior,TensorMedium):
        from .dyadic_green import DyadicGreenKernel
        from .anisotropic_fields import tensor_current_potential_columns
        kernel=getattr(handle,'tensor_kernel',None)
        if not isinstance(kernel,DyadicGreenKernel) or kernel.medium is not handle.interior:
            raise ValueError('Tensor interior columns require the owning regional kernel')
        e,h,receipt=tensor_current_potential_columns(handle.basis,coefficients,queries,kernel,order=order)
        handle.last_column_field_receipt=receipt
        return -e,-h
    e,h = _potentials(handle,coefficients,queries,handle.interior,order=order,
                      singular_order=order,surface_faces=faces)
    return -e,-h


def selected_interior_field_blocks(handle, queries, edge_indices):
    """Fixed-order [query,xyz,selected J then M] interior field blocks.

    Only full supports of requested local RWG edges are integrated. Repeated
    edge columns retain their requested order. Traces and exterior queries are
    rejected; callers own quadrature refinement and coupled error checks.
    """
    from .physics import Medium
    basis=handle.basis;mesh=basis.mesh
    if (not isinstance(edge_indices,torch.Tensor) or edge_indices.device!=mesh.device
            or edge_indices.dtype!=torch.int64 or edge_indices.ndim!=1
            or bool(((edge_indices<0)|(edge_indices>=basis.n_edges)).any())):
        raise ValueError('Selected field edges require matching CUDA int64 indices in range')
    dtype=complex_dtype_for(mesh.dtype)
    zero=torch.zeros(2*basis.n_edges,device=mesh.device,dtype=dtype)
    _,_,faces,_=_query_context(handle,zero,queries,1,'total',None,FieldQueryConfig(adaptive=False))
    from .tensor_medium import TensorMedium
    if isinstance(handle.interior,TensorMedium):
        from .dyadic_green import DyadicGreenKernel
        from .anisotropic_fields import tensor_selected_current_blocks
        kernel=getattr(handle,'tensor_kernel',None)
        if type(kernel) is not DyadicGreenKernel or kernel.medium is not handle.interior:
            raise ValueError('Selected tensor fields require the owning regional kernel')
        order=max(handle.integration.order,handle.integration.singular_order)
        e,h,receipt=tensor_selected_current_blocks(basis,queries,edge_indices,kernel,order=order)
        handle.last_column_field_receipt=receipt
        return -e,-h
    if not isinstance(handle.interior,Medium):
        raise TypeError('Selected interior blocks require a supported regional medium')
    selected,inverse=torch.unique(edge_indices,sorted=True,return_inverse=True)
    coefficients=torch.eye(2*len(selected),device=mesh.device,dtype=dtype)
    order=max(handle.integration.order,handle.integration.singular_order)
    e,h=_potentials(handle,coefficients,queries,handle.interior,order=order,
        singular_order=order,surface_faces=faces,edge_indices=selected)
    columns=torch.cat((inverse,inverse+len(selected)))
    return -e[...,columns],-h[...,columns]


def _record_query(handle, log, start):
    torch.cuda.synchronize(handle.basis.mesh.device)
    log["seconds"] = time.perf_counter()-start
    log["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(handle.basis.mesh.device)
    handle.last_field_query_log = log
    if hasattr(handle, "metadata"):
        handle.metadata["last_field_query"] = log


def adapt_columns(evaluate, order, config, eta, logs, *, require_finite):
    """Successive-order doubling with an independent acceptance per column.

    ``evaluate(order)`` returns E/H [P,3,S]. Column s is frozen at the first
    order whose difference from the previous order meets the tolerance for
    all of its queries, exactly as if it were adapted alone; ``logs[s]``
    receives its own checks. Returns the frozen fields and the failed columns.
    """
    old_e, old_h = evaluate(order)
    count = old_e.shape[-1]
    if not config.adaptive or not old_e.shape[0]:
        for log in logs:
            log.update({"converged": not old_e.shape[0], "final_order": order,
                        "failure_reason": None if not old_e.shape[0] else "fixed_order_error_unestimated"})
        return old_e, old_h, []
    result_e, result_h = old_e.clone(), old_h.clone()
    pending = list(range(count))
    while pending and order < config.max_order:
        order = min(2*order, config.max_order)
        e, h = evaluate(order)
        error = ((e-old_e).abs().square().sum(1) + abs(eta)**2*(h-old_h).abs().square().sum(1)).sqrt()
        norm = (e.abs().square().sum(1) + abs(eta)**2*h.abs().square().sum(1)).sqrt()
        threshold = config.absolute_tolerance + config.relative_tolerance*norm
        accepted = error <= threshold
        if require_finite:
            accepted = accepted & torch.isfinite(error) & torch.isfinite(norm) & torch.isfinite(threshold)
        relative = error/norm.clamp_min(torch.finfo(norm.dtype).tiny)
        columns = torch.tensor(pending, device=e.device)
        summary = torch.stack((error[:, columns].amax(0), relative[:, columns].amax(0),
                               accepted[:, columns].all(0).to(error.dtype))).detach().cpu().tolist()
        remaining = []
        for index, column in enumerate(pending):
            logs[column]["checks"].append({"order": order, "max_absolute_error_V_per_m": summary[0][index],
                                           "max_relative_error": summary[1][index]})
            if summary[2][index]:
                result_e[..., column], result_h[..., column] = e[..., column], h[..., column]
                logs[column].update({"converged": True, "final_order": order, "failure_reason": None})
            else:
                remaining.append(column)
        pending = remaining
        old_e, old_h = e, h
    for column in pending:
        logs[column].update({"final_order": order, "failure_reason": "quadrature_order_budget_exhausted"})
    return result_e, result_h, pending


def _adaptive_log(handle, x, queries, surface_faces, config):
    return {"query_count": len(queries), "surface_count": int((surface_faces >= 0).sum()),
            "adaptive": config.adaptive, "converged": False, "checks": [],
            "query_chunk": QUERY_CHUNK, "far_quadrature_sample_cap": SAMPLE_CAP,
            "relative_tolerance": config.relative_tolerance, "absolute_tolerance_V_per_m": config.absolute_tolerance,
            "norm": "sqrt(sum(abs(E)^2)+abs(eta)^2*sum(abs(H)^2)) per query",
            "estimator": "successive_quadrature_order_difference", "device": str(x.device),
            "peak_memory_scope": "device_process_high_water_mark"}


def _evaluate_adaptive(handle, x, queries, medium, surface_faces, query_config):
    """Adapt the potentials of one coefficient vector or of [2*edges,S] columns.

    Columns return [P,3,S] fields and one log each, retained as a list.
    """
    torch.cuda.synchronize(handle.basis.mesh.device)
    start = time.perf_counter()
    config = query_config
    order = max(handle.integration.order, handle.integration.singular_order)
    columns = x.ndim == 2
    logs = [_adaptive_log(handle, x, queries, surface_faces, config) for _ in range(x.shape[1] if columns else 1)]

    def record():
        _record_query(handle, logs[0], start)
        for log in logs[1:]:
            log.update(seconds=logs[0]["seconds"], peak_allocated_bytes=logs[0]["peak_allocated_bytes"])
        if columns:
            handle.last_field_query_logs = logs

    if config.adaptive and order >= config.max_order:
        for log in logs:
            log["failure_reason"] = "initial_order_leaves_no_refinement"
        record()
        raise ValueError("Field max_order must exceed the initial integration order for error estimation.")

    def evaluate(current):
        e, h = _potentials(handle, x, queries, medium, order=current, singular_order=current, surface_faces=surface_faces)
        return (e, h) if columns else (e[..., None], h[..., None])

    e, h, failed = adapt_columns(evaluate, order, config, medium.eta, logs, require_finite=True)
    record()
    if failed:
        raise RuntimeError("MoM field quadrature did not converge within max_order; inspect last_field_query_log.")
    return (e, h) if columns else (e[..., 0], h[..., 0])


def _trace_jump(handle, x, queries, faces, medium):
    edges, free, slopes = local_geometry(handle.basis)
    n = handle.basis.mesh.normals[faces].to(x.dtype)
    jlocal = x[:handle.basis.n_edges][edges[faces]] * slopes[faces]
    mlocal = x[handle.basis.n_edges:][edges[faces]] * slopes[faces]
    displacement = (queries[:, None]-free[faces]).to(x.dtype)
    j = torch.einsum("ql,qlc->qc", jlocal, displacement)
    m = torch.einsum("ql,qlc->qc", mlocal, displacement)
    divj, divm = 2*jlocal.sum(-1), 2*mlocal.sum(-1)
    jump_e = torch.linalg.cross(n, m) - 1j*n*divj[:, None]/(medium.omega*medium.epsilon)
    jump_h = -torch.linalg.cross(n, j) - 1j*n*divm[:, None]/(medium.omega*medium.mu)
    return jump_e, jump_h


def _query_context(handle, x, queries, region_id, field_kind, trace_side, query_config):
    """Validate shared query geometry and return its resolved region context."""
    _validate_arrays(handle, x, queries)
    if region_id not in (0, 1, "exterior", "interior") or isinstance(region_id, bool):
        raise ValueError("MoM region_id must be 0/'exterior' or 1/'interior'.")
    interior = region_id in (1, "interior")
    if field_kind not in ("total", "scattered", "incident"):
        raise ValueError("MoM field_kind must be total, scattered, or incident.")
    if interior and field_kind != "total":
        raise ValueError("MoM interior queries require field_kind='total'.")
    config = FieldQueryConfig() if query_config is None else query_config
    if not isinstance(config, FieldQueryConfig):
        raise TypeError("query_config must be FieldQueryConfig.")
    if trace_side not in (None, "interior", "exterior"):
        raise ValueError("trace_side must be interior or exterior.")
    if trace_side is not None and (trace_side == "interior") != interior:
        raise ValueError("trace_side must match region_id.")
    is_inside, _, faces = classify_queries(handle.basis.mesh, queries, allow_surface=True)
    on_surface = faces >= 0
    if bool(on_surface.any()) and trace_side is None:
        raise ValueError("MoM on-surface queries require explicit trace_side.")
    if trace_side is not None and not bool(on_surface.all()):
        raise ValueError("trace_side is only defined for on-surface queries.")
    if bool(((is_inside != interior) & ~on_surface).any()):
        raise ValueError("MoM query geometry does not match the requested region_id.")
    return interior, config, faces, on_surface


def decode(handle, x, queries, region_id, field_kind, trace_side=None, *, query_config=None):
    """Evaluate fields with explicit region and total/scattered semantics.

    Region 0 (``exterior``) supports total, scattered and incident fields;
    region 1 (``interior``) supports total fields. Queries must belong to the
    requested geometric region. At face interiors, trace_side='exterior' or
    'interior' selects the analytic one-sided jump and must match region_id.
    Quadrature doubles to the supplied tolerance and fails if the order budget
    is exhausted. The last convergence estimate and cost are retained on the
    handle; an order-difference estimate is not a certified error bound.
    """
    interior, config, faces, on_surface = _query_context(
        handle, x, queries, region_id, field_kind, trace_side, query_config)
    if field_kind == "incident":
        e = torch.zeros_like(queries, dtype=x.dtype)
        h = torch.zeros_like(e)
    else:
        medium = handle.interior if interior else handle.exterior
        e, h = _evaluate_adaptive(handle, x, queries, medium, faces, config)
        if bool(on_surface.any()):
            je, jh = _trace_jump(handle, x, queries[on_surface], faces[on_surface], medium)
            sign = -0.5 if interior else 0.5
            e, h = e.clone(), h.clone()
            e[on_surface] = e[on_surface]+sign*je
            h[on_surface] = h[on_surface]+sign*jh
    if interior:
        return -e, -h
    if field_kind in ("total", "incident"):
        for source in handle.sources:
            incident_e, incident_h = incident_fields(source, queries, handle.exterior)
            e, h = e + incident_e, h + incident_h
    return e, h


def far_field(handle, x, directions):
    """Complex outgoing amplitudes F_E [V], F_H [A], with E~exp(ikr)F_E/r.

    Directions are unit Cartesian vectors; phase is referenced to the global
    coordinate origin. Only the radiating scattered field is included.
    """
    _validate_arrays(handle, x, directions)
    norm = torch.linalg.vector_norm(directions, dim=-1)
    if not bool(torch.isclose(norm, torch.ones_like(norm), rtol=1e-6, atol=1e-7).all()):
        raise ValueError("MoM far-field directions must be unit vectors.")
    basis, medium = handle.basis, handle.exterior
    points, weights = triangle_rule(basis.mesh.triangles, handle.integration.order)
    j = basis.current(x[:basis.n_edges], points)
    m = basis.current(x[basis.n_edges:], points)
    outputs_e, outputs_h = [], []
    for direction in directions.split(128):
        phase = torch.exp(-1j * medium.k * torch.einsum("dc,tqc->dtq", direction, points))
        integral_j = torch.einsum("dtq,tqc,tq->dc", phase, j, weights.to(x.dtype))
        integral_m = torch.einsum("dtq,tqc,tq->dc", phase, m, weights.to(x.dtype))
        n = direction.to(x.dtype)
        transverse_j = integral_j - n * (n * integral_j).sum(-1, keepdim=True)
        transverse_m = integral_m - n * (n * integral_m).sum(-1, keepdim=True)
        prefactor = 1j * medium.k / (4 * math.pi)
        outputs_e.append(prefactor * (medium.eta * transverse_j - torch.linalg.cross(n, integral_m)))
        outputs_h.append(prefactor * (torch.linalg.cross(n, integral_j) + transverse_m / medium.eta))
    if not len(directions):
        empty = x.new_empty((0, 3))
        return empty, empty.clone()
    return torch.cat(outputs_e), torch.cat(outputs_h)


def radar_cross_section(electric_far_field, incident_electric_amplitude):
    """RCS [m²] for the explicitly supplied incident E amplitude [V/m]."""
    if not isinstance(electric_far_field, torch.Tensor) or electric_far_field.device.type != "cuda":
        raise ValueError("MoM RCS requires CUDA far-field samples.")
    if electric_far_field.shape[-1:] != (3,) or not bool(torch.isfinite(electric_far_field).all()):
        raise ValueError("MoM RCS requires finite far-field Cartesian vectors.")
    amplitude = torch.as_tensor(incident_electric_amplitude, device=electric_far_field.device,
                                dtype=electric_far_field.dtype)
    if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude)) or bool(amplitude.abs() == 0):
        raise ValueError("RCS requires a finite nonzero scalar incident electric amplitude.")
    return 4 * math.pi * electric_far_field.abs().square().sum(-1) / amplitude.abs().square()


def poynting_flux(electric, magnetic, normals, weights):
    """Independent sensor integral of outward time-average power [W]."""
    if electric.shape != magnetic.shape or electric.shape[-1:] != (3,):
        raise ValueError("E and H must have matching (..., 3) shapes.")
    if normals.shape != electric.shape or weights.shape != electric.shape[:-1]:
        raise ValueError("Sensor normals and area weights must match field samples.")
    if any(value.device.type != "cuda" or value.device != electric.device for value in (electric, magnetic, normals, weights)):
        raise ValueError("MoM flux sensor arrays must share a CUDA device.")
    if not all(bool(torch.isfinite(value).all()) for value in (electric, magnetic, normals, weights)):
        raise ValueError("Flux sensor arrays must be finite.")
    if bool((weights < 0).any()) or not bool(torch.isclose(torch.linalg.vector_norm(normals, dim=-1), torch.ones_like(weights), rtol=1e-6, atol=1e-7).all()):
        raise ValueError("Flux sensors require unit normals and nonnegative area weights.")
    density = 0.5 * (torch.linalg.cross(electric, magnetic.conj()).real * normals).sum(-1)
    return (density * weights).sum()
