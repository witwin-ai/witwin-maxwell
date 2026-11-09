"""Ideal multipole kernel truncation mapped to balanced RWG Frobenius bounds."""
import math
import torch
from .rwg_interpolation_error import tree_integral_statistics, block_error_log, _receipt


def multipole_block_error_log(test, source, medium, kernel_bound, impedance):
    return block_error_log(test, source, medium,
        {'green_absolute_bound': kernel_bound['green_absolute'],
         'gradient_component_absolute_bound': kernel_bound['gradient_component_absolute']}, impedance)


def multipole_error_receipt(logged, impedance):
    result = _receipt(logged, impedance)
    result.update(version='rwg_ideal_multipole_frobenius_v1',
                  scope='ideal_exactly_integrated_cartesian_multipole_only')
    return result


def regional_multipole_bound(operator, impedance):
    operator.assert_current()
    test = tree_integral_statistics(operator.test)
    source = test if operator.source is operator.test else tree_integral_statistics(operator.source)
    pairs = {}
    for a, b, coupling in operator.far:
        logged = multipole_block_error_log(test[a], source[b], coupling.medium, coupling.truncation_bound(), impedance)
        pairs[a, b] = torch.logaddexp(pairs[a, b], logged) if (a, b) in pairs else logged
    total = .5*torch.logsumexp(2*torch.stack(list(pairs.values())), dim=0) if pairs else test.new_tensor(-torch.inf)
    result = multipole_error_receipt(total, impedance)
    result['polynomial_exact_in_exact_arithmetic'] = all(p.quadrature_order >= (p.order+4)//2
                                                       for p in (operator.test, operator.source))
    operator.assert_current()
    return result


def system_multipole_bound(operator):
    operator.assert_current()
    eta = abs(operator.provider.exterior.eta)
    exterior = regional_multipole_bound(operator.exterior, eta)
    interiors = [regional_multipole_bound(region, eta) for region in operator.interiors]
    logs = torch.tensor([r['log_absolute_bound'] if r['log_absolute_bound'] is not None else -math.inf
                         for r in (exterior, *interiors)], device=operator.device, dtype=torch.float64)
    result = multipole_error_receipt(torch.logaddexp(logs[0], .5*torch.logsumexp(2*logs[1:], dim=0)), eta)
    result.update(exterior=exterior, interiors=interiors)
    operator.assert_current()
    return result
