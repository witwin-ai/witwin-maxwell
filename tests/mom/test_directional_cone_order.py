"""Independent cubed-sphere, high-precision admission and actual FFT controls."""
import json
import math
import mpmath as mp
import pytest
import torch
from witwin.maxwell.mom.fmm_order import DirectionalConeOrderPlan, DirectionalConeOrderError
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_expansion import monomials, multipole_moments

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _cube_oracle(center, grid):
    axis = max(range(3), key=lambda j: abs(center[j]))
    other = [j for j in range(3) if j != axis]
    indices = [min(grid, max(0, math.floor((center[j]/abs(center[axis])+1)*grid/2+.5))) for j in other]
    cube = [0., 0., 0.]
    cube[axis] = -1. if center[axis] < 0 else 1.
    for j, index in zip(other, indices):
        cube[j] = 2*index/grid-1
    norm = math.sqrt(sum(x*x for x in cube))
    return [x/norm for x in cube], [grid, 2*axis+int(center[axis] < 0), *indices]


def _mp_bounds(center, carrier, rt, rs, wave, order):
    # Independent positive contour arithmetic, not the CUDA log recurrence.
    with mp.workdps(100):
        R, u = [list(map(mp.mpf, values)) for values in (center, carrier)]
        k = mp.mpc(wave); D = mp.sqrt(sum(x*x for x in R))
        rt, rs = mp.mpf(rt), mp.mpf(rs)
        if rt == rs == 0:
            return 0., 0.
        ratio = (rt+rs)/D
        mismatch = mp.sqrt(sum((x/D-v)**2 for x, v in zip(R, u)))
        unorm = mp.sqrt(sum(v*v for v in u))
        contours = list(map(mp.mpf, ('.05', '.1', '.15', '.2', '.25', '.3', '.35', '.4')))
        contours += [ratio*mp.mpf(v) for v in ('1.25', '1.5', '2', '3', '4', '8', '12', '16', '32')]
        green, gradient = [], []
        for rho in contours:
            if not ratio < rho <= mp.mpf('.4'):
                continue
            alpha = mp.sqrt(1-2*rho-rho*rho)
            maximum = mp.exp(-k.imag*(D-unorm*(rt+rs))
                +abs(k)*D*(rho*mismatch+rho*rho/(alpha+1-rho)))/(4*mp.pi*D*alpha)
            q = ratio/rho; power = q**(order+1)
            tail = power/(1-q) if rt == 0 or rs == 0 else power*(2-power)/(1-q)**2
            green.append(maximum*tail)
            gradient.append(maximum*tail*(abs(k)+1/(D*alpha))*(1+rho)/alpha)
        floor = mp.mpf(float(torch.finfo(torch.float64).tiny))
        return tuple(float(max(floor, min(values))) if values else math.inf for values in (green, gradient))


def _relative(a, b):
    return float(((a-b).norm()/b.norm()).detach())


def test_exhaustive_independent_cone_order_selection_and_storage(tmp_path):
    # Dominant-axis/sign ties, exact/one-sided zero support and genuine finer-grid
    # admission share one noncontiguous batch; no CUDA bound is used by the oracle.
    centers = [[2000., 250., -125.], [-2000., -250., 125.], [1., 1., -1.], [.2, -.4, .3]]
    base = torch.tensor(centers, device='cuda', dtype=torch.float64)
    R = torch.stack((base, torch.zeros_like(base)), -1)[..., 0]
    radii = R.new_tensor([.5, .5, 0., .001])
    source = R.new_tensor([.5, .5, 0., 0.])
    grids = (1, 2, 4, 8, 16, 32, 64)
    budget = KernelErrorBudget(1e-8, 1e-8)
    plan = DirectionalConeOrderPlan(R, radii, source, 100., budget, cone_subdivisions=grids)
    expected, errors = [], []
    for row, center in enumerate(centers):
        candidates = []
        for grid in grids:
            carrier, key = _cube_oracle(center, grid)
            for order in range(13):
                g, h = _mp_bounds(center, carrier, float(radii[row]), float(source[row]), 100., order)
                if g <= budget.green_absolute and h <= budget.gradient_component_absolute:
                    candidates.append((order, grid, key, carrier, g, h))
        best = min(candidates, key=lambda v: v[:2]); expected.append(best[:3])
        assert int(plan.orders[row]) == best[0]
        assert plan.cone_keys[row].tolist() == best[2]
        torch.testing.assert_close(plan.directions[row], R.new_tensor(best[3]), rtol=1e-14, atol=0)
        for actual, wanted in ((float(plan.green_bounds[row]), best[4]), (float(plan.gradient_bounds[row]), best[5])):
            if wanted == 0:
                assert actual == 0
            else:
                error = abs(actual/wanted-1); errors.append(error); assert error < 1e-11
    assert any(key[0] > 1 for key in plan.cone_keys.tolist())
    assert int(plan.orders[2]) == 0
    distance_scale = R.norm(dim=-1)/(2*plan.orders+2)
    support_scale = torch.maximum(radii, source)
    expected_scale = torch.where(support_scale > 0, torch.minimum(distance_scale, support_scale), distance_scale)
    torch.testing.assert_close(plan.normalization_lengths, expected_scale, rtol=0, atol=0)
    names = ('displacements', 'test_radii', 'source_radii', 'directions', 'cone_keys', 'orders',
        'normalization_lengths', 'green_bounds', 'gradient_bounds', 'green_log_bounds', 'gradient_log_bounds')
    arrays = [getattr(plan, name) for name in names]
    owned = sum(t.untyped_storage().nbytes() for t in arrays)
    assert len({t.untyped_storage().data_ptr() for t in arrays}) == len(arrays)
    assert owned == plan.retained_bytes == 144*len(R)
    receipt = plan.receipt(); receipt['orders'][0] = 999
    assert plan.receipt()['orders'][0] != 999
    record = dict(expected=expected, maximum_mp_relative_error=max(errors), owned_bytes=owned,
        receipt=plan.receipt(), mp_precision_digits=100, noncontiguous_input=True, qualification=False)
    (tmp_path/'selection.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


@pytest.mark.parametrize('wave', (100., 100.+.002j))
def test_actual_auto_selected_four_fields_full_h_current_and_higher(wave, tmp_path):
    R = torch.tensor([[2000., 250., -125.]], device='cuda', dtype=torch.float64)
    r = R.new_tensor([.5])
    plan = DirectionalConeOrderPlan(R, r, r, wave, KernelErrorBudget(1e-8, 1e-8))
    op = plan.create_translation(0)
    p = int(plan.orders[0]); u = plan.directions[0]
    x = R.new_tensor([[.15, -.2, .1], [-.1, .2, -.15]])
    y = R.new_tensor([[.1, -.05, .2], [-.2, -.1, .15], [.05, .1, -.2]])
    charges = R.new_tensor([[1., -.4, .7], [.3, .5, -.2]]).to(torch.complex128)
    source_phase = torch.exp(-1j*wave*(y@u))
    receiver = monomials(x, p).to(torch.complex128)*torch.exp(1j*wave*(x@u))[:, None]
    def action(q):
        moments = multipole_moments(y, q*source_phase, p)
        return torch.einsum('tb,sfb->stf', receiver, op.apply(moments))
    delta = R[0]+x[:, None]-y[None]
    distance = delta.norm(dim=-1)
    G = torch.exp(1j*wave*distance)/(4*math.pi*distance)
    grad = G[..., None]*(1j*wave-1/distance)[..., None]*delta/distance[..., None]
    exact = torch.cat((G[..., None], grad), -1)
    actual_columns = action(torch.eye(3, device='cuda', dtype=torch.complex128))
    fields = [_relative(actual_columns[..., j].T, exact[..., j]) for j in range(4)]
    assert max(fields) < 1e-8
    q = charges.detach().requires_grad_(True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        # The entire point wrapper multiplies charges by phases and is allowed
        # to save caller data; the native primitive itself saves no tensors.
        moments = multipole_moments(y, q*source_phase, p)
    native_saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: native_saved.append(t) or t, lambda t: t):
        value = op.apply(moments)
    assert not native_saved
    dual = (torch.ones_like(value)*(1+.4j)).requires_grad_(True)
    derivative, = torch.autograd.grad(value, moments, dual, create_graph=True)
    native_error = _relative(derivative, op.adjoint(dual))
    assert native_error < 1e-11
    tangent = torch.ones_like(derivative)*(.2-.3j)
    higher, = torch.autograd.grad(derivative, dual, tangent)
    higher_error = _relative(higher, op.apply(tangent))
    assert higher_error < 1e-11
    lhs = (value.conj()*dual).sum(); rhs = (moments.conj()*op.adjoint(dual)).sum()
    dot = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()))
    assert dot < 1e-11
    record = dict(wave=str(wave), four_physical_fields=fields, native_current=native_error,
        native_higher=higher_error, dot=dot, native_saved_tensors=len(native_saved),
        plan=plan.receipt(), primitive=op.receipt(), qualification=False)
    (tmp_path/'actual.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_unavailable_late_row_capacity_empty_positive_floor_and_mutation(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_order as module
    import witwin.maxwell.mom.fmm_directional_fft as fft
    R = torch.tensor([[1., .2, -.1], [1., 0., 0.]], device='cuda', dtype=torch.float64)
    r = R.new_tensor([0., 1.])
    budget = KernelErrorBudget(1e-8, 1e-8)
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected plan reached coefficient or contour allocation')
    with monkeypatch.context() as patch:
        patch.setattr(fft, 'directional_helmholtz_taylor', forbidden)
        with pytest.raises(DirectionalConeOrderError) as exc:
            DirectionalConeOrderPlan(R, r, r, 5., budget)
        assert exc.value.receipt['unadmitted_rows'] == [1]
        patch.setattr(module, 'directional_multipole_truncation_bounds_many', forbidden)
        for kwargs in ({'retained_budget_bytes': 287}, {'tensor_workspace_budget_bytes': 1}):
            with pytest.raises(MemoryError):
                DirectionalConeOrderPlan(R, r, r, 5., budget, **kwargs)
        empty = DirectionalConeOrderPlan(R[:0], r[:0], r[:0], 5., budget)
        assert empty.retained_bytes == 0 and empty.receipt()['orders'] == []
        with pytest.raises(ValueError):
            empty.create_translation(0)
    decay_r = R.new_tensor([.01])
    decay = DirectionalConeOrderPlan(R[:1], decay_r, decay_r, 1000j, budget)
    assert bool((decay.green_bounds > 0).all())
    zero = R.new_zeros(1)
    tripwires = []
    for kind in ('caller', 'owned', 'replace', 'config', 'budget'):
        local_R = R[:1].clone(); local_budget = KernelErrorBudget(1e-8, 1e-8)
        plan = DirectionalConeOrderPlan(local_R, zero, zero, 5., local_budget)
        if kind == 'caller':
            local_R.add_(.01)
        elif kind == 'owned':
            plan.directions.add_(.01)
        elif kind == 'replace':
            plan.orders = plan.orders.clone()
        elif kind == 'config':
            plan.min_order = 1
        else:
            object.__setattr__(local_budget, 'green_absolute', 1.)
        with monkeypatch.context() as patch:
            patch.setattr(fft, 'directional_helmholtz_taylor', forbidden)
            with pytest.raises(ValueError):
                plan.create_translation(0)
            with pytest.raises(ValueError):
                plan.receipt()
        tripwires.append(kind)
    record = dict(late_unavailable_receipt=exc.value.receipt, empty=empty.receipt(),
        positive_floor=decay.green_bounds.tolist(), mutation_tripwires=tripwires, qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_support_capped_normalization_changed_path_only(tmp_path):
    R = torch.tensor([[2000., 250., -125.], [1., .2, -.1], [1., .2, -.1]],
        device='cuda', dtype=torch.float64)
    rt = R.new_tensor([.5, .1, 0.]); rs = R.new_tensor([.5, .1, 0.])
    plan = DirectionalConeOrderPlan(R, rt, rs, 0., KernelErrorBudget(100., 1e5), min_order=12)
    expected = []
    for center, a, b in zip(R.tolist(), rt.tolist(), rs.tolist()):
        distance_scale = math.sqrt(sum(v*v for v in center))/26
        expected.append(min(distance_scale, max(a, b)) if max(a, b) else distance_scale)
    torch.testing.assert_close(plan.normalization_lengths, R.new_tensor(expected), rtol=1e-14, atol=0)
    assert expected[0] == .5 and expected[1] < .1 and expected[2] > 0
    record = dict(independent_expected_scale=expected, actual=plan.normalization_lengths.tolist(),
        unchanged_selection_and_guard_controls_not_repeated=True, receipt=plan.receipt(), qualification=False)
    (tmp_path/'normalization.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
