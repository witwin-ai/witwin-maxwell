"""Independent fixed-carrier point moments, receiver polynomials and native guards."""
import json
import pytest
import torch
from witwin.maxwell.mom.fmm_directional_shift import DirectionalFixedCarrierShift
from witwin.maxwell.mom.fmm_expansion import multi_indices
from witwin.maxwell.mom.fmm_shift import SeparableMultipoleShift

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _poly(points, order):
    powers = torch.tensor(multi_indices(order), device=points.device)
    return points[:, None].pow(powers).prod(-1).to(torch.complex128)


def _relative(actual, expected):return float(((actual-expected).norm()/expected.norm()).detach())


@pytest.mark.parametrize('order,wave', ((0, 5.), (3, 5.+.7j), (6, 5.+.7j), (12, 5.+.7j)))
def test_fixed_carrier_actual_source_moments_receiver_points_full_adjoints_and_native_current(order, wave, tmp_path):
    d = torch.tensor([-.13, .27, .07], device='cuda', dtype=torch.float64)
    direction = d.new_tensor([.7, -.2, .3]); direction = direction/direction.norm()
    op = DirectionalFixedCarrierShift(d, direction, wave, order)
    generator = torch.Generator(device='cuda').manual_seed(1281)
    child_points = torch.randn((9, 3), device='cuda', dtype=torch.float64, generator=generator)*.02
    parent_points = child_points+d
    currents = torch.randn((3, 5, 4, 9), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1)
    child_basis = _poly(child_points, order)*torch.exp(-1j*wave*(child_points@direction))[:, None]
    parent_basis = _poly(parent_points, order)*torch.exp(-1j*wave*(parent_points@direction))[:, None]
    child_moments = (currents@child_basis).movedim(-1, -2).detach().requires_grad_()
    expected_parent = (currents@parent_basis).movedim(-1, -2)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        actual_parent = op.source(child_moments)
    assert not saved
    source_error = _relative(actual_parent, expected_parent)
    parent_coefficients = torch.randn((3, 5, op.count, 4), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    actual_child = op.receiver(parent_coefficients)
    receiver_child_basis = _poly(child_points, order)*torch.exp(1j*wave*(child_points@direction))[:, None]
    receiver_parent_basis = _poly(parent_points, order)*torch.exp(1j*wave*(parent_points@direction))[:, None]
    actual_at_points = torch.einsum('pa,...ac->...pc', receiver_child_basis, actual_child)
    expected_at_points = torch.einsum('pa,...ac->...pc', receiver_parent_basis, parent_coefficients)
    receiver_error = _relative(actual_at_points, expected_at_points)
    records = {}
    for name, value in [('source', child_moments), ('receiver', parent_coefficients)]:
        fn = getattr(op, name)
        applied = fn(value); dual = torch.randn_like(applied)
        adjoint = fn(dual, adjoint=True)
        gradient, = torch.autograd.grad(applied, value, dual)
        dot_lhs, dot_rhs = (applied.conj()*dual).sum(), (value.conj()*adjoint).sum()
        dot = float(((dot_lhs-dot_rhs).abs()/torch.maximum(dot_lhs.abs(), dot_rhs.abs())).detach())
        gradient_error = _relative(gradient, adjoint)
        # Identity columns give an explicit matrix reference independent of H action.
        eye = torch.eye(op.count, device='cuda', dtype=torch.complex128)[..., None].expand(-1, -1, 4)
        matrix_rows = fn(eye)[..., 0]
        reference_adjoint = torch.einsum('ji,...ic->...jc', matrix_rows.conj(), dual)
        adjoint_error = _relative(adjoint, reference_adjoint)
        records[name] = dict(adjoint=adjoint_error, gradient=gradient_error, dot=dot)
        assert adjoint_error < 1e-11 and gradient_error < 1e-11 and dot < 1e-11
    record = dict(order=order, wave=str(wave), source_actual_point_error=source_error,
        receiver_actual_point_error=receiver_error, checks=records, receipt=op.receipt(),
        source_receiver_hermitian_relation_assumed=False, physical_qualification=False)
    (tmp_path/'shift.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert source_error < 1e-11 and receiver_error < 1e-11


def test_fixed_carrier_owned_and_shared_layout_exact_storage_preflight_mutation_and_native_guards(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_directional_shift as module
    d = torch.tensor([-.13, .27, .07], device='cuda', dtype=torch.float64)
    u = d.new_tensor([1., 0., 0.]); order = 3
    layout = SeparableMultipoleShift(order, device='cuda', dtype=torch.float64)
    layout_bytes = sum(t.untyped_storage().nbytes() for t in layout.tensors())
    owned = DirectionalFixedCarrierShift(d, u, 5.+.7j, order, retained_budget_bytes=104+layout_bytes)
    shared = DirectionalFixedCarrierShift(d, u, 5.+.7j, order, layout=layout, retained_budget_bytes=104)
    for op, expected in [(owned, 104+layout_bytes), (shared, 104)]:
        tensors = op.tensors()
        assert len({t.untyped_storage().data_ptr() for t in tensors}) == len(tensors)
        assert all(t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in tensors)
        assert sum(t.untyped_storage().nbytes() for t in tensors) == op.retained_bytes == op.planned_retained_bytes == expected
    value = torch.randn((3, 5, owned.count, 4), device='cuda', dtype=torch.complex128).transpose(0, 1)
    for name in ('source', 'receiver'):
        torch.testing.assert_close(getattr(owned, name)(value), getattr(shared, name)(value), rtol=0, atol=0)
    def forbidden(*args, **kwargs):raise AssertionError('Rejected input reached layout construction')
    monkeypatch.setattr(module, 'SeparableMultipoleShift', forbidden)
    # All these default-owned failures must occur before constructing a layout.
    opposite_zero = u.clone(); opposite_zero[1] = -0.
    for kw, exc in [({'target_direction': u+d.new_tensor([0., 1e-15, 0.])}, ValueError),
            ({'target_direction':opposite_zero}, ValueError), ({'direction':u*2}, ValueError),
            ({'order':13}, ValueError), ({'wave_number':5.-.1j}, ValueError),
            ({'rhs_batch':False}, ValueError), ({'retained_budget_bytes':104+layout_bytes-1}, MemoryError),
            ({'tensor_workspace_budget_bytes':1}, MemoryError),
            ({'displacement':u*1000., 'wave_number':1j}, ValueError)]:
        args = dict(displacement=d, direction=u, wave_number=5.+.7j, order=order)
        args.update(kw)
        with pytest.raises(exc):DirectionalFixedCarrierShift(**args)
    for method in (owned.source, owned.receiver):
        with pytest.raises(ValueError):method(value, adjoint=1)
        with pytest.raises(ValueError):method(value*float('nan'))
    d.add_(1); u.add_(1)
    owned.assert_current(); shared.assert_current()
    owned.layout.tensors()[1].add_(1)
    with pytest.raises(ValueError, match='changed'):owned.source(value)
    layout.tensors()[0].add_(1)
    with pytest.raises(ValueError, match='changed'):shared.receiver(value)
    record = dict(layout_bytes=layout_bytes, owned_retained_bytes=owned.retained_bytes,
        shared_metadata_retained_bytes=shared.retained_bytes, owned_tensor_count=len(owned.tensors()),
        shared_tensor_count=len(shared.tensors()), direction_change_rejected_bit_exactly=True,
        native_phase_overflow_rejected_before_layout=True, qualification=False)
    (tmp_path/'ownership.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
