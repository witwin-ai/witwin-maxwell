"""Exhaustive streamed refinement of the smooth Ewald weak correction."""
from dataclasses import replace
import math
import torch
from .ewald_nearfield import EwaldSystemBlocks
from .operator_error import _scaled_norm


@torch.no_grad()
def measure_ewald_refinement(provider, *, edge_batch=32, relative_tolerance=1e-7,
                             absolute_tolerance=1e-12):
    """Independently double integration, image, spectral and contour resolution.

    Every correction coefficient is measured in impedance-balanced coordinates,
    with separate cross-body reporting. This quadratic diagnostic is a refinement
    estimate, not a tail bound or physical certification. Primary singular and
    interior blocks are intentionally excluded, preventing their much larger
    norms from hiding a periodic correction error.
    """
    if not isinstance(provider, EwaldSystemBlocks):
        raise TypeError('Ewald refinement requires original Ewald blocks')
    if type(edge_batch) is not int or edge_batch < 1:
        raise ValueError('edge_batch must be a positive integer')
    if (any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in (relative_tolerance, absolute_tolerance))
            or relative_tolerance+absolute_tolerance == 0):
        raise ValueError('Refinement tolerances must be finite nonnegative and nonzero')
    provider.assert_current()
    n, device = provider.system.total_edges, provider.system.device
    eta = abs(provider.exterior.eta)
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('Balancing impedance must be finite and positive')
    checks = []
    for axis in ('regular_testing_order', 'real_shells', 'reciprocal_shells', 'quadrature_order'):
        initial = provider.config.order if axis == 'regular_testing_order' else getattr(provider.kernel, axis)
        refined = 2*initial
        config = replace(provider.config, order=refined) if axis == 'regular_testing_order' else provider.config
        kernel = provider.kernel if axis == 'regular_testing_order' else replace(provider.kernel, **{axis: refined})
        other = EwaldSystemBlocks(provider.system, provider.exterior, config, kernel)
        sums = torch.zeros((2, 2), device=device, dtype=torch.float64)
        counts = [0, 0]
        for rs in range(0, n, edge_batch):
            rows = torch.arange(rs, min(rs+edge_batch, n), device=device)
            row_body = torch.bucketize(rows, provider.edge_stops, right=True).repeat(2)
            left = torch.ones(2*len(rows), device=device, dtype=torch.float64)
            left[:len(rows)] = eta
            for cs in range(0, n, edge_batch):
                cols = torch.arange(cs, min(cs+edge_batch, n), device=device)
                right = torch.ones(2*len(cols), device=device, dtype=torch.float64)
                right[len(cols):] = eta
                baseline = provider.regular_block(rows, cols)
                reference = other.regular_block(rows, cols)
                difference = left[:, None]*(reference-baseline)*right
                weighted = left[:, None]*reference*right
                if not bool(torch.isfinite(difference).all() & torch.isfinite(weighted).all()):
                    raise ValueError('Nonfinite Ewald refinement block')
                cross = row_body[:, None] != torch.bucketize(cols, provider.edge_stops, right=True).repeat(2)
                for index, mask in enumerate((None, cross)):
                    d = difference if mask is None else difference[mask]
                    r = weighted if mask is None else weighted[mask]
                    sums[index, 0] = torch.hypot(sums[index, 0], _scaled_norm(d))
                    sums[index, 1] = torch.hypot(sums[index, 1], _scaled_norm(r))
                    counts[index] += d.numel()
        metrics = {}
        for name, (error, norm), count in zip(('global', 'cross_body'), sums.cpu().tolist(), counts):
            if not math.isfinite(error) or not math.isfinite(norm):
                raise ValueError('Nonfinite accumulated Ewald refinement norm')
            threshold = absolute_tolerance+relative_tolerance*norm
            if not math.isfinite(threshold):
                raise ValueError('Nonfinite Ewald refinement threshold')
            metrics[name] = dict(entries=count, absolute_difference=error, reference_norm=norm,
                threshold=threshold, passed=count > 0 and error <= threshold)
        checks.append(dict(axis=axis, initial=initial, refined=refined, metrics=metrics,
            passed=metrics['global']['passed'] and
                (counts[1] == 0 or metrics['cross_body']['passed'])))
        other.assert_current()
    provider.assert_current()
    return dict(version='ewald_regular_exhaustive_refinement_v1', checks=checks,
        converged=all(check['passed'] for check in checks), edge_batch=edge_batch,
        relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
        all_correction_coefficients_measured=True, primary_image_reassembled=False,
        scope='impedance_balanced_regular_correction_frobenius',
        certified_error_bound=False, physical_qualification=False)
