"""Independent coefficient, physical-kernel and ownership controls for modulation."""
import json
import math
import pytest
import torch
from witwin.maxwell.mom.fmm_directional_fft import CartesianDirectionalFFTTranslation, directional_helmholtz_taylor
from witwin.maxwell.mom.fmm_expansion import multi_indices, helmholtz_taylor, multipole_moments, monomials
from witwin.maxwell.mom.fmm_fft_coefficients import CartesianCoefficientFFTTranslation
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _modulated_coefficients(original, direction, wave, degree):
    indices = multi_indices(degree)
    origins, destinations, powers, factorials = [], [], [], []
    for i, alpha in enumerate(indices):
        for j, beta in enumerate(indices):
            if all(b <= a for a, b in zip(alpha, beta)):
                gamma = tuple(a-b for a, b in zip(alpha, beta))
                origins.append(j); destinations.append(i); powers.append(gamma)
                factorials.append(math.prod(math.factorial(g) for g in gamma))
    origin = torch.tensor(origins, device=original.device)
    destination = torch.tensor(destinations, device=original.device)
    power = torch.tensor(powers, device=original.device)
    denominator = original.real.new_tensor(factorials)
    terms = original[origin]*(-1j*wave*direction).pow(power).prod(-1)/denominator
    result = torch.zeros_like(original)
    # Independent oracle reduction is not the production shift recurrence.
    result.index_add_(0, destination, terms)
    return result


def _dense_tables(coefficients, order, direction, wave):
    indices = multi_indices(order)
    lookup = {a: i for i, a in enumerate(multi_indices(2*order+1))}
    gather, factors = [], []
    for field in range(4):
        ids, weights = [], []
        for beta in indices:
            row, factor_row = [], []
            for alpha in indices:
                gamma = tuple(a+b for a, b in zip(alpha, beta))
                factor = (-1)**sum(alpha)*math.prod(math.comb(a+b, a) for a, b in zip(alpha, beta))
                derivative = tuple(g+(axis == field-1) for axis, g in enumerate(gamma)) if field else gamma
                row.append(lookup[derivative]); factor_row.append(factor*(gamma[field-1]+1) if field else factor)
            ids.append(row); weights.append(factor_row)
        gather.append(ids); factors.append(weights)
    ids = torch.tensor(gather, device=coefficients.device)
    weights = coefficients.real.new_tensor(factors)
    tables = coefficients[ids]*weights
    tables[1:] += (1j*wave*direction)[:, None, None]*tables[0]
    return tables


@pytest.mark.parametrize('order,wave', [(0, 5.2), (3, 5.2), (3, 5.2+.04j)])
def test_directional_coefficients_all_fields_adjoint_and_current_gradient(order, wave, tmp_path):
    center = torch.tensor([1., .2, -.1], device='cuda', dtype=torch.float64)
    direction = center/center.norm()
    direction = direction + center.new_tensor([.001, -.002, .003]); direction = direction/direction.norm()
    scale = center.new_tensor(.05)
    original = helmholtz_taylor(center, wave, 2*order+1)
    independent = _modulated_coefficients(original, direction, wave, 2*order+1)
    op = CartesianDirectionalFFTTranslation(center, direction, wave, scale, order)
    torch.testing.assert_close(op.coefficients, independent, rtol=3e-10, atol=3e-12)
    tables = _dense_tables(independent, order, direction, wave)
    generator = torch.Generator(device='cuda').manual_seed(954)
    points = torch.randn((9, 3), device='cuda', dtype=torch.float64, generator=generator)*.01
    charges = torch.randn((3, 7, 9), device='cuda', dtype=op.dtype, generator=generator)
    value = multipole_moments(points, charges, order).transpose(0, 1).detach().requires_grad_(True)
    weights = .01**op.indices.sum(-1)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        actual = op.apply(value)
    assert not saved
    expected = torch.einsum('fba,...a->...fb', tables, value)
    error = float((((actual-expected)*weights).norm()/(expected*weights).norm()).detach())
    dual = torch.randn(actual.shape, device='cuda', dtype=op.dtype, generator=generator)*weights
    adjoint = op.adjoint(dual)
    reference = torch.einsum('fba,...fb->...a', tables.conj(), dual)
    adjoint_error = float((((adjoint-reference)*weights).norm()/(reference*weights).norm()).detach())
    derivative, = torch.autograd.grad(actual, value, dual)
    gradient_error = float((derivative-adjoint).norm()/adjoint.norm())
    lhs, rhs = (actual.conj()*dual).sum(), (value.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach())
    record = dict(order=order, wave=str(wave), forward=error, adjoint=adjoint_error,
        gradient=gradient_error, dot=dot, receipt=op.receipt())
    (tmp_path/'algebra.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert error < 1e-11 and adjoint_error < 1e-11 and gradient_error < 1e-11 and dot < 1e-11


@pytest.mark.parametrize('kind', ('axial_high_ka', 'oblique_high_ka', 'lossy_off_axis', 'native_order12'))
def test_directional_actual_green_and_physical_gradient_with_unmodulated_comparator(kind, tmp_path):
    order = 12 if kind == 'native_order12' else 6
    wave = .5 if kind == 'native_order12' else 100.+.02j if kind == 'lossy_off_axis' else 100.
    center = torch.tensor([1., .2, -.1] if kind == 'native_order12' else
        [2000., 0., 0.] if kind == 'axial_high_ka' else [2000., 400., -200.], device='cuda', dtype=torch.float64)
    direction = center/center.norm()
    if kind == 'lossy_off_axis':
        direction = direction + center.new_tensor([.0001, -.0002, .0003]); direction = direction/direction.norm()
    radius = .01 if kind == 'native_order12' else .5
    generator = torch.Generator(device='cuda').manual_seed(956)
    source = torch.randn((7, 3), device='cuda', dtype=torch.float64, generator=generator)
    target = torch.randn((9, 3), device='cuda', dtype=torch.float64, generator=generator)
    source = source/source.norm(dim=-1, keepdim=True)*radius
    target = target/target.norm(dim=-1, keepdim=True)*radius
    scale = center.new_tensor(radius)
    op = CartesianDirectionalFFTTranslation(center, direction, wave, scale, order)
    source_phase = torch.exp(-1j*wave*(source@direction))
    target_phase = torch.exp(1j*wave*(target@direction))
    charges = torch.eye(len(source), device='cuda', dtype=torch.complex128)
    moments = multipole_moments(source, charges*source_phase, order)
    locals = op.apply(moments)
    receiver = monomials(target, order).to(torch.complex128)
    actual = torch.einsum('tb,sfb->tsf', receiver, locals)*target_phase[:, None, None]
    delta = center+target[:, None]-source[None]
    r = delta.norm(dim=-1)
    green = torch.exp(1j*wave*r)/(4*math.pi*r)
    gradient = green[..., None]*(1j*wave-1/r)[..., None]*delta/r[..., None]
    expected = torch.cat((green[..., None], gradient), dim=-1)
    original = CartesianCoefficientFFTTranslation(helmholtz_taylor(center, wave, 2*order+1), center, scale, order)
    unmodulated = torch.einsum('tb,sfb->tsf', receiver, original.apply(multipole_moments(source, charges, order)))
    errors = [float((actual[..., field]-expected[..., field]).norm()/expected[..., field].norm())
        for field in (0,)] + [float((actual[..., 1:]-expected[..., 1:]).norm()/expected[..., 1:].norm())]
    old_errors = [float((unmodulated[..., 0]-expected[..., 0]).norm()/expected[..., 0].norm()),
                  float((unmodulated[..., 1:]-expected[..., 1:]).norm()/expected[..., 1:].norm())]
    record = dict(kind=kind, order=order, wave=str(wave), support_radius=radius, electrical_radius=abs(wave)*radius,
        parabolic_phase_scale=abs(wave)*radius**2/float(center.norm()), green=errors[0], gradient=errors[1],
        same_order_unmodulated_green=old_errors[0], same_order_unmodulated_gradient=old_errors[1],
        input_source_phase_once=True, output_receiver_phase_once=True, receipt=op.receipt(),
        physical_qualification=False, full_directional_fmm=False)
    (tmp_path/'physical.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert max(errors) < 1e-8
    if kind != 'native_order12': assert max(errors) < min(old_errors)


def test_directional_batch_independent_ownership_lifetime_and_native_preallocation(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_directional_fft as module
    centers = torch.tensor([[1.+i*.01, .2, -.1] for i in range(33)], device='cuda', dtype=torch.float64)
    directions = centers/centers.norm(dim=-1, keepdim=True)
    scales = centers.new_full((33,), .05)
    calls = []; native = module.directional_helmholtz_taylor
    def observed(c, *args, **kwargs): calls.append(tuple(c.shape)); return native(c, *args, **kwargs)
    monkeypatch.setattr(module, 'directional_helmholtz_taylor', observed)
    expected = m2l_storage_bytes(3, 8, 'fft_v1')+24
    objects = CartesianDirectionalFFTTranslation.create_many(centers, directions, 5.2, scales, 3,
        retained_budget_bytes=33*expected)
    assert calls == [(32, 3), (1, 3)]
    assert storage_bytes(t for obj in objects for t in obj.tensors()) == 33*expected
    for channel in range(9):
        tensors = [obj.tensors()[channel] for obj in objects]
        assert len({t.untyped_storage().data_ptr() for t in tensors}) == 33
        assert all(t._base is None and t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in tensors)
    centers.add_(1); directions.add_(1); scales.add_(1)
    for obj in objects: obj.assert_current()
    for index in range(9):
        objects[index].tensors()[index].add_(1)
        with pytest.raises(ValueError, match='changed'): objects[index].assert_current()
    (tmp_path/'ownership.json').write_text(json.dumps(dict(native_batches=[32, 1], owned_bytes=33*expected,
        independent_owned_tensors=9, qualification=False), indent=2)+'\n')


def test_directional_native_budget_and_phase_contracts_before_coefficients(monkeypatch):
    import witwin.maxwell.mom.fmm_directional_fft as module
    center = torch.tensor([1., .2, -.1], device='cuda', dtype=torch.float64)
    direction = center/center.norm(); scale = center.new_tensor(.05)
    def forbidden(*args, **kwargs): raise AssertionError('Rejected directional object reached coefficients')
    monkeypatch.setattr(module, 'directional_helmholtz_taylor', forbidden)
    for c, d, wave, length, order, exc, kw in (
            (center.float(), direction.float(), 5., scale, 3, ValueError, {}),
            (center, direction*2, 5., scale, 3, ValueError, {}),
            (center, direction.clone().requires_grad_(), 5., scale, 3, ValueError, {}),
            (center, direction, 5.-1j, scale, 3, ValueError, {}),
            (center, direction, 5., None, 3, ValueError, {}),
            (center, direction, 5., scale, 13, ValueError, {}),
            (center, direction, 5., scale, 3, MemoryError, {'retained_budget_bytes':1}),
            (center, direction, 5., scale, 3, MemoryError, {'tensor_workspace_budget_bytes':1}),
            (center, direction, 5., scale, 3, ValueError, {'rhs_batch':False})):
        with pytest.raises(exc): CartesianDirectionalFFTTranslation(c, d, wave, length, order, **kw)
    for kwargs in ({'rhs_batch':False}, {'tensor_workspace_budget_bytes':0}, {'retained_budget_bytes':0}):
        with pytest.raises(ValueError): CartesianDirectionalFFTTranslation.create_many(center[None][:0], direction[None][:0],
            5., scale[None][:0], 3, **kwargs)
