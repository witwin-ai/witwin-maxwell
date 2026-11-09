"""Incident-field work on physical scattered equivalent surface currents."""
import torch
from .operator import local_geometry
from .quadrature import triangle_rule
from .lifting_fields import system_local_coefficients


def incident_surface_work(handle, coefficients, *, quadrature_order=6):
    """Integrate Re(E_inc* . J_s + H_inc* . M_s)/2 over all bodies.

    This extinction diagnostic uses local incident fields and surface quadrature,
    independently of outgoing modal normalization. It is not absorbed material
    work, and agreement with a flux ledger does not qualify a current solution.
    """
    handle.system.assert_current()
    system = handle.system
    dtype = torch.complex128 if system.dtype == torch.float64 else torch.complex64
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != (2*system.total_edges,)
            or coefficients.device != system.device or coefficients.dtype != dtype
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Incident surface work requires finite matching global RWG currents')
    if type(quadrature_order) is not int or quadrature_order < 2:
        raise ValueError('Incident surface work quadrature must be an integer >=2')
    values = []
    for index, body in enumerate(system.bodies):
        local = system_local_coefficients(handle, coefficients, index)
        edges, free, slopes = local_geometry(body.basis)
        j = local[:body.basis.n_edges][edges]*slopes
        m = local[body.basis.n_edges:][edges]*slopes
        power = coefficients.real.new_zeros(())
        for start in range(0, len(body.mesh.faces), handle.integration.pair_batch):
            stop = min(start+handle.integration.pair_batch, len(body.mesh.faces))
            points, weights = triangle_rule(body.mesh.triangles[start:stop], quadrature_order)
            displacement = (points[:, :, None]-free[start:stop, None]).to(dtype)
            electric_current = torch.einsum('fl,fqlc->fqc', j[start:stop], displacement)
            magnetic_current = torch.einsum('fl,fqlc->fqc', m[start:stop], displacement)
            electric, magnetic = handle.incident_carrier.evaluate(points.reshape(-1, 3), handle.exterior)
            density = (electric.reshape_as(points).conj()*electric_current+
                       magnetic.reshape_as(points).conj()*magnetic_current).sum(-1).real
            power = power+.5*(weights*density).sum()
        values.append(power)
    per_body = torch.stack(values)
    if not bool(torch.isfinite(per_body).all()):
        raise ValueError('Nonfinite incident surface work')
    return {'extinction_work_W': per_body.sum(), 'per_body_work_W': per_body,
        'quadrature_order': quadrature_order, 'quadrature_converged': False,
        'independent_material_work_verified': False, 'physical_qualification': False,
        'definition': 'half_real_integral_conjugate_incident_E_dot_Js_plus_H_dot_Ms'}


def exterior_surface_work(provider, coefficients, *, edge_batch=32):
    """Measure emitted power from original exterior weak blocks in a lossless medium.

    PMCHWT exterior rows test (H, -E), hence source work is
    Re(J* . row_E - M* . row_H)/2. The real work of the two jump traces cancels.
    Every selected block is evaluated, at quadratic cost. Forward block storage
    is bounded; current autograd can retain those blocks for its backward pass.
    This diagnostic is independent of Rayleigh channel normalization, but still
    depends on the configured weak quadrature and Ewald truncation.
    """
    from .nearfield import ScalarSystemBlocks
    from .ewald_nearfield import EwaldSystemBlocks
    if type(provider) not in (ScalarSystemBlocks, EwaldSystemBlocks):
        raise TypeError('Exterior work requires free-space or Ewald original scalar blocks')
    provider.assert_current()
    system, medium = provider.system, provider.exterior
    if medium.epsilon_r.imag != 0 or medium.mu_r.imag != 0 or medium.k.imag != 0:
        raise ValueError('Radiated surface work requires a lossless exterior')
    if type(edge_batch) is not int or edge_batch < 1:
        raise ValueError('edge_batch must be a positive integer')
    n = system.total_edges
    dtype = torch.complex128 if system.dtype == torch.float64 else torch.complex64
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != (2*n,)
            or coefficients.device != system.device or coefficients.dtype != dtype
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Exterior work requires finite matching global RWG currents')
    power = coefficients.real.new_zeros(())
    for start in range(0, n, edge_batch):
        rows = torch.arange(start, min(start+edge_batch, n), device=system.device)
        action = coefficients.new_zeros(2*len(rows))
        for begin in range(0, n, edge_batch):
            cols = torch.arange(begin, min(begin+edge_batch, n), device=system.device)
            block = provider.exterior_block(rows, cols)
            if isinstance(provider, EwaldSystemBlocks):
                block = block+provider.regular_block(rows, cols)
            action = action+block@coefficients[torch.cat((cols, cols+n))]
        power = power+.5*(coefficients[rows].conj()*action[len(rows):]-
                          coefficients[rows+n].conj()*action[:len(rows)]).sum().real
    provider.assert_current()
    if not bool(torch.isfinite(power)):
        raise ValueError('Nonfinite exterior surface work')
    return {'emitted_power_W': power, 'edge_batch': edge_batch,
        'all_exterior_coefficients_measured': True, 'physical_qualification': False,
        'quadrature_converged': False, 'scope': 'configured_original_exterior_weak_work',
        'interior_material_terms_included': False}
