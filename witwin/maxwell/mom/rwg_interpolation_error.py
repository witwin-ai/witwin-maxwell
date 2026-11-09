"""Kernel-to-RWG interpolation error bounds without expanded far matrices.

These bounds concern the ideal exactly integrated interpolant. They exclude
quadrature, numerical moments/transfers, floating-point effects and solve error.
"""
import math
import torch
from .nearfield import ScalarSystemBlocks


def _edge_logsumexp(values, indices, size):
    maxima = values.new_full((size,), -torch.inf)
    maxima.scatter_reduce_(0, indices, values, reduce='amax', include_self=True)
    sums = values.new_zeros(size)
    sums.index_add_(0, indices, torch.exp(values-maxima[indices]))
    return maxima+torch.log(sums)


def rwg_integral_logs(basis, geometry):
    """Return log upper bounds for integral |f| and integral |div f| per edge."""
    mesh = basis.mesh
    if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
        raise ValueError('Basis geometry changed before integral bounds')
    geometry.assert_current(basis)
    triangles, slopes = geometry.triangles.double(), geometry.slopes.double().abs()
    a = triangles.roll(-1, 1)-triangles
    b = triangles.roll(-2, 1)-triangles
    scale = torch.maximum(a.abs().amax(-1), b.abs().amax(-1))
    aa, bb = a/scale[..., None], b/scale[..., None]
    mean_square = (aa.square().sum(-1)+bb.square().sum(-1)+(aa*bb).sum(-1))/6
    cross = torch.linalg.cross(aa[:, 0], bb[:, 0], dim=-1)
    # Compute area in log space, after scaling the two triangle edge vectors.
    log_area = math.log(.5)+2*torch.log(scale[:, 0])+torch.log(torch.linalg.vector_norm(cross, dim=-1))
    log_vector = torch.log(slopes)+log_area[:, None]+torch.log(scale)+.5*torch.log(mean_square)
    log_divergence = math.log(2)+torch.log(slopes)+log_area[:, None]
    if not bool(torch.isfinite(log_vector).all() & torch.isfinite(log_divergence).all()):
        raise ValueError('RWG integral bound exceeds valid native geometry range')
    indices = basis.face_edges.flatten()
    result = torch.stack((_edge_logsumexp(log_vector.flatten(), indices, basis.n_edges),
                          _edge_logsumexp(log_divergence.flatten(), indices, basis.n_edges)), dim=-1)
    geometry.assert_current(basis)
    return result


def tree_integral_statistics(projection):
    """Node log sums of u^2, q^2 and u*q; retained workspace is O(nodes+edges)."""
    projection.assert_current()
    from .fmm_directional_moments import DirectionalRWGDAGProjection
    if isinstance(projection, DirectionalRWGDAGProjection) and projection.plan_guard is not None:
        # Actual basis/tree/support were bound before maps. The planner owns
        # these fixed statistics and validates their identity, version and mode.
        return projection.plan_guard._statistics
    if isinstance(projection.basis, ScalarSystemBlocks):
        provider = projection.basis
        logs = torch.cat([rwg_integral_logs(body.basis, geometry)
                          for body, geometry in zip(provider.system.bodies, provider.geometry)])
    else:
        logs = rwg_integral_logs(projection.basis, projection.geometry)
    statistics = integral_statistics_from_logs(projection.tree, logs)
    projection.assert_current()
    return statistics


def integral_statistics_from_logs(tree, logs):
    """Aggregate precomputed per-edge bounds before interpolation allocation."""
    tree.assert_current()
    if (not isinstance(logs, torch.Tensor) or logs.shape != (tree.size, 2)
            or logs.device != tree.device or logs.dtype != torch.float64 or logs.requires_grad
            or not bool(torch.isfinite(logs).all())):
        raise ValueError('Integral statistics require finite fixed CUDA float64 edge logs')
    starts, stops = tree.starts.cpu().tolist(), tree.stops.cpu().tolist()
    children_by_node = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
    statistics = logs.new_empty((tree.node_count, 3))
    for node in reversed(range(tree.node_count)):
        children = children_by_node[node]
        if children:
            statistics[node] = torch.logsumexp(statistics[list(children)], dim=0)
        else:
            indices = tree.permutation[starts[node]:stops[node]]
            u, q = logs[indices].unbind(-1)
            statistics[node] = torch.logsumexp(torch.stack((2*u, 2*q, u+q), dim=-1), dim=0)
    tree.assert_current()
    return statistics


def _log_magnitude(value):
    value = complex(value)
    scale = max(abs(value.real), abs(value.imag))
    if not math.isfinite(scale):
        raise ValueError('Error bound requires finite material factors')
    return math.log(scale)+.5*math.log((value.real/scale)**2+(value.imag/scale)**2) if scale else -math.inf


def block_error_log(test, source, medium, kernel_bound, impedance):
    """Log balanced Frobenius bound, using three integral statistics per side."""
    if type(impedance) not in (float, int) or not math.isfinite(impedance) or impedance <= 0:
        raise ValueError('Error balancing impedance must be positive and finite')
    g, gradient = kernel_bound['green_absolute_bound'], kernel_bound['gradient_component_absolute_bound']
    if any(not math.isfinite(v) or v < 0 for v in (g, gradient)):
        raise ValueError('Kernel bounds must be finite and nonnegative')
    log_g, log_gradient = math.log(g) if g else -math.inf, math.log(gradient) if gradient else -math.inf
    log_k = _log_magnitude(medium.k)
    if not math.isfinite(log_k):
        raise ValueError('RWG interpolation bound requires nonzero wave number')
    uu, qq, uq = test+source
    d_squared = 2*log_g+torch.logsumexp(torch.stack((uu, math.log(2)+uq-2*log_k, qq-4*log_k)), dim=0)
    k_squared = math.log(3)+2*log_gradient+uu
    log_eta = math.log(impedance)
    log_e, log_m = _log_magnitude(1j*medium.omega*medium.epsilon), _log_magnitude(-1j*medium.omega*medium.mu)
    off_diagonal = torch.logaddexp(test.new_tensor(4*log_eta+2*log_e), test.new_tensor(2*log_m))
    return .5*torch.logaddexp(math.log(2)+2*log_eta+k_squared, off_diagonal+d_squared)


def _receipt(log_bound, impedance):
    if type(impedance) not in (float, int) or not math.isfinite(impedance) or impedance <= 0:
        raise ValueError('Error balancing impedance must be positive and finite')
    if bool(torch.isnan(log_bound) | torch.isposinf(log_bound)):
        raise ValueError('RWG interpolation bound exceeds finite native representation')
    if bool(torch.isneginf(log_bound)):
        absolute, logged = 0., None
    else:
        value = torch.exp(log_bound.clamp_min(math.log(torch.finfo(torch.float64).tiny)))
        if not bool(torch.isfinite(value)):
            raise ValueError('RWG interpolation bound exceeds finite native representation')
        absolute, logged = float(value), float(log_bound)
    return {'version': 'rwg_ideal_interpolation_frobenius_v1', 'absolute_bound': absolute,
            'log_absolute_bound': logged, 'balancing_impedance': impedance,
            'norm': 'Frobenius(diag(eta_J,1_M) error diag(1_J,eta_M))',
            'scope': 'ideal_exactly_integrated_kernel_interpolant_only',
            'quadrature_included': False, 'floating_point_certified': False,
            'implemented_operator_error_certified': False, 'physical_qualification': False}


def regional_interpolation_bound(operator, impedance):
    operator.assert_current()
    test = tree_integral_statistics(operator.test)
    source = test if operator.source is operator.test else tree_integral_statistics(operator.source)
    pairs = {}
    for a, b, kernel in operator.far:
        bound = block_error_log(test[a], source[b], kernel.medium, kernel.truncation_bound(), impedance)
        # Regional media overlap on one block: use triangle inequality first.
        pairs[a, b] = torch.logaddexp(pairs[a, b], bound) if (a, b) in pairs else bound
    logged = (.5*torch.logsumexp(2*torch.stack(list(pairs.values())), dim=0)
              if pairs else test.new_tensor(-torch.inf))
    operator.assert_current()
    result = _receipt(logged, impedance)
    result['moment_quadrature'] = {'test': operator.test.moment_quadrature_receipt(),
                                   'source': operator.source.moment_quadrature_receipt()}
    return result


def system_interpolation_bound(operator):
    operator.assert_current()
    eta = abs(operator.provider.exterior.eta)
    exterior = regional_interpolation_bound(operator.exterior, eta)
    interiors = [regional_interpolation_bound(region, eta) for region in operator.interiors]
    logs = torch.tensor([value['log_absolute_bound'] if value['log_absolute_bound'] is not None else -math.inf
                         for value in [exterior, *interiors]], device=operator.device, dtype=torch.float64)
    interior_log = .5*torch.logsumexp(2*logs[1:], dim=0)
    result = _receipt(torch.logaddexp(logs[0], interior_log), eta)
    result['exterior'] = exterior
    result['interiors'] = interiors
    operator.assert_current()
    return result


def directional_system_ideal_bound(exterior,interiors,*,device,balancing_impedance):
    """Triangle across overlapping exterior; RSS over disjoint body interiors.

    Inputs are already admitted common-impedance ideal component receipts. This
    numerical combination does not establish their physical/body provenance.
    """
    device = torch.device(device)
    if device.type!='cuda' or not isinstance(interiors,(tuple,list)):
        raise ValueError('Directional ideal system bounds require CUDA and explicit interior receipts')
    if (type(balancing_impedance) not in (float,int)
            or not math.isfinite(balancing_impedance) or balancing_impedance<=0):
        raise ValueError('Directional ideal system balancing impedance must be positive and finite')
    logs = []
    for bound in (exterior,*interiors):
        if not isinstance(bound,dict) or bound.get('balancing_impedance')!=balancing_impedance:
            raise ValueError('Directional ideal system receipts must share the common impedance')
        value,logged = bound.get('absolute_bound'),bound.get('log_absolute_bound')
        if (type(value) not in (float,int) or not math.isfinite(value) or value<0
                or (logged is None and value!=0)
                or (logged is not None and (type(logged) not in (float,int)
                    or not math.isfinite(logged) or value<=0))):
            raise ValueError('Directional ideal system component receipt has invalid native range')
        logs.append(-math.inf if logged is None else logged)
    native = torch.tensor(logs,device=device,dtype=torch.float64)
    interior_log = .5*torch.logsumexp(2*native[1:],dim=0)
    result = _receipt(torch.logaddexp(native[0],interior_log),balancing_impedance)
    result.update(version='directional_exterior_triangle_disjoint_interiors_ideal_v1',
        combination='exterior_bound + RSS(disjoint_interior_bounds)',
        overlapping_media_triangle_included=True,production_qualification=False)
    return result
