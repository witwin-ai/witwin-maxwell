"""Exhaustive, streamed comparison with the implemented original weak operator."""
import math
import torch


def _scaled_norm(value):
    if not value.numel():
        return value.real.new_zeros((), dtype=torch.float64)
    scale = value.abs().max()
    if not bool(scale):
        return scale
    return scale*torch.linalg.vector_norm(value/scale)


@torch.no_grad()
def measure_original_operator_error(actions, *, edge_batch=32, tolerance=1e-8):
    """Measure balanced Frobenius error over every coefficient, without a dense matrix.

    This diagnostic costs quadratic reference work. It measures the configured
    original quadrature, not continuum error or a directed-rounding bound.
    """
    if type(edge_batch) is not int or edge_batch < 1:
        raise ValueError('edge_batch must be a positive integer')
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    actions.assert_current()
    provider = actions.provider
    n = provider.system.total_edges
    eta = abs(provider.exterior.eta)
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('Balancing impedance must be finite and positive')
    sums = torch.zeros((2, 2), dtype=torch.float64, device=actions.device)
    entries = [0, 0]
    for start in range(0, n, edge_batch):
        columns = torch.arange(start, min(start+edge_batch, n), device=actions.device)
        cc = torch.cat((columns, columns+n))
        probes = torch.zeros((len(cc), 2*n), device=actions.device, dtype=actions.dtype)
        probes[torch.arange(len(cc), device=actions.device), cc] = 1
        actual = actions.apply(probes).mT
        del probes
        for row_start in range(0, n, edge_batch):
            rows = torch.arange(row_start, min(row_start+edge_batch, n), device=actions.device)
            rr = torch.cat((rows, rows+n))
            reference = provider(rows, columns)
            left = torch.ones(len(rr), device=actions.device, dtype=torch.float64)
            right = torch.ones(len(cc), device=actions.device, dtype=torch.float64)
            left[rr < n] = eta
            right[cc >= n] = eta
            weighted = left[:, None]*reference*right
            difference = left[:, None]*(actual[rr]-reference)*right
            if not bool(torch.isfinite(weighted).all() & torch.isfinite(difference).all()):
                raise ValueError('Nonfinite original operator comparison')
            # Cross-body reporting prevents large self blocks hiding coupling errors.
            row_body = torch.bucketize(rows, provider.edge_stops, right=True).repeat(2)
            col_body = torch.bucketize(columns, provider.edge_stops, right=True).repeat(2)
            cross = row_body[:, None] != col_body
            for index, mask in enumerate((None, cross)):
                d = difference if mask is None else difference[mask]
                r = weighted if mask is None else weighted[mask]
                sums[index, 0] = torch.hypot(sums[index, 0], _scaled_norm(d))
                sums[index, 1] = torch.hypot(sums[index, 1], _scaled_norm(r))
                entries[index] += d.numel()
        del actual
    actions.assert_current()
    metrics = {}
    for name, (error, reference), count in zip(('global', 'cross_body'), sums.cpu().tolist(), entries):
        if not math.isfinite(error) or not math.isfinite(reference):
            raise ValueError('Nonfinite accumulated operator norm')
        relative = error/reference if reference else (0. if error == 0 else None)
        if relative is not None and not math.isfinite(relative):
            relative = None
        metrics[name] = dict(entries=count, absolute_frobenius=error,
            reference_frobenius=reference, relative_frobenius=relative,
            passed=count > 0 and relative is not None and relative <= tolerance)
    return dict(version='original_operator_exhaustive_measurement_v1', metrics=metrics,
        tolerance=tolerance, edge_batch=edge_batch, all_coefficients_measured=True,
        scope='implemented_original_quadrature_balanced_frobenius',
        directed_rounding_certified=False, physical_qualification=False,
        complexity_qualification=False)


@torch.no_grad()
def measure_original_global_residual(provider, value, rhs, *, edge_batch=32,
                                     source_batch=64, tolerance=1e-10,
                                     absolute_tolerance=0.):
    """Stream every original weak row against each supplied current/RHS pair.

    Both target and source selections are bounded; no dense global matrix or
    compressed action is constructed. This costs quadratic original-quadrature
    work. It measures a complete vector residual, not all operator coefficients,
    quadrature convergence, a continuum error or a rounding certificate.
    """
    from .nearfield import ScalarSystemBlocks
    if type(provider) is not ScalarSystemBlocks:
        raise TypeError('Original global residual requires scalar system blocks')
    if any(type(batch) is not int or batch < 1 for batch in (edge_batch, source_batch)):
        raise ValueError('Original residual edge/source batches must be positive integers')
    if (type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0
            or type(absolute_tolerance) not in (int, float)
            or not math.isfinite(absolute_tolerance) or absolute_tolerance < 0):
        raise ValueError('Original residual tolerances must be finite with positive relative and nonnegative absolute values')
    provider.assert_current()
    n = provider.system.total_edges
    dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64
    if (not isinstance(value, torch.Tensor) or not isinstance(rhs, torch.Tensor)
            or value.ndim < 1 or value.shape != rhs.shape or value.shape[-1] != 2*n
            or not value.numel() or value.device != provider.system.device
            or rhs.device != value.device or value.device.type != 'cuda'
            or value.dtype != dtype or rhs.dtype != dtype
            or not bool(torch.isfinite(value).all() & torch.isfinite(rhs).all())):
        raise ValueError('Original residual requires matching finite global complex CUDA currents and RHS')
    def state():
        return tuple((id(t), t._version, t.requires_grad, t.shape, t.device, t.dtype)
                     for t in (value, rhs))
    binding = state()
    def operator_state():
        from dataclasses import fields
        def medium_state(medium):
            return (id(medium), medium.frequency, medium.epsilon_r, medium.mu_r,
                    medium.k, medium.eta)
        return (id(provider.system), provider.system.total_edges, provider.system.total_faces,
                id(provider.config), tuple(getattr(provider.config, field.name)
                    for field in fields(provider.config)), medium_state(provider.exterior),
                tuple((id(body), body.structure_index, body.name, id(body.basis),
                       body.basis.n_edges, body.edge_offset, body.face_offset, body.region_id,
                       medium_state(body.interior)) for body in provider.system.bodies))
    operator_binding = operator_state()
    def assert_bound():
        provider.assert_current()
        if operator_state() != operator_binding:
            raise ValueError('Original residual operator configuration or material changed during measurement')
        if state() != binding:
            raise ValueError('Original residual currents or RHS changed during measurement')
    count = math.prod(value.shape[:-1])
    sums = torch.zeros((count, 2), device=value.device, dtype=torch.float64)
    rows_measured = 0
    for start in range(0, n, edge_batch):
        rows = torch.arange(start, min(start+edge_batch, n), device=value.device)
        indices = torch.cat((rows, rows+n))
        original = provider.sampled_action(rows, value, source_batch=source_batch)
        assert_bound()
        selected_rhs = rhs[..., indices]
        difference = original-selected_rhs
        if not bool(torch.isfinite(original).all() & torch.isfinite(difference).all()):
            raise ValueError('Original residual action or subtraction exceeds native precision')
        differences = difference.reshape(count, len(indices))
        references = selected_rhs.reshape(count, len(indices))
        # Separate every RHS so a large accurate channel cannot hide another
        # channel's relative error. Reuse the primary scaled norm primitive.
        for index in range(count):
            sums[index, 0] = torch.hypot(sums[index, 0], _scaled_norm(differences[index]))
            sums[index, 1] = torch.hypot(sums[index, 1], _scaled_norm(references[index]))
        rows_measured += len(indices)
    assert_bound()
    if not bool(torch.isfinite(sums).all()):
        raise ValueError('Original residual accumulated norms exceed native precision')
    error, reference = sums.unbind(-1)
    threshold = absolute_tolerance+tolerance*reference
    if not bool(torch.isfinite(threshold).all()):
        raise ValueError('Original residual threshold exceeds native precision')
    passed = error <= threshold
    relative = torch.where(reference > 0, error/reference,
        torch.where(error == 0, torch.zeros_like(error), torch.full_like(error, torch.inf)))
    def shaped(tensor):
        return tensor.reshape(value.shape[:-1]).cpu().tolist()
    # Null is the JSON-safe representation of an undefined/infinite relative
    # error. Acceptance always uses the native absolute/relative threshold.
    relative_values = [float(v) if math.isfinite(v) else None for v in relative.cpu().tolist()]
    def reshape_metadata(values, shape):
        if not shape:
            return values[0]
        width = math.prod(shape[1:])
        return [reshape_metadata(values[i*width:(i+1)*width], shape[1:])
                for i in range(shape[0])]
    return dict(version='original_global_weak_residual_measurement_v1',
        rows_measured=rows_measured, total_rows=2*n, rhs_count=count,
        all_original_rows_measured=rows_measured == 2*n,
        absolute_residual=shaped(error), rhs_norm=shaped(reference),
        relative_residual=reshape_metadata(relative_values, value.shape[:-1]),
        threshold=shaped(threshold), passed_per_rhs=shaped(passed),
        all_rhs_passed=bool(passed.all()), relative_tolerance=tolerance,
        absolute_tolerance=absolute_tolerance, edge_batch=edge_batch,
        source_batch=source_batch, scope='complete_configured_original_weak_vector_residual',
        compressed_action_used=False, dense_global_matrix_constructed=False,
        all_operator_coefficients_error_measured=False, quadrature_convergence_certified=False,
        directed_rounding_certified=False, physical_qualification=False,
        complexity_qualification=False, peak_memory_qualified=False)
