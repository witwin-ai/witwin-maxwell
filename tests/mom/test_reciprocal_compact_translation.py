"""Opposite-displacement compact tables share fixed native coefficient storage."""
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
from witwin.maxwell.mom.fmm_expansion import multipole_to_local, multi_indices
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action
from witwin.maxwell.mom.direct_capacity import storage_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _problem(order=3, batch=7):
    medium = compile_medium(mw.Material(eps_r=1.5, mu_r=1.3, sigma_e=.02, sigma_m=.03), 3e8)
    displacement = torch.tensor([1., .3, -.2], device='cuda', dtype=torch.float64)
    owner = CompactElectromagneticM2L(displacement, medium, order, row_batch=batch)
    return displacement, owner


def _relative(actual, reference):
    return float(((actual-reference).norm()/reference.norm()).detach())


@pytest.mark.parametrize('order,batch', [(0, 1), (3, 7), (6, 16)])
def test_reversed_compact_dense_H_current_higher_and_shared_storage(order, batch, monkeypatch, tmp_path):
    displacement, owner = _problem(order, batch)
    import witwin.maxwell.mom.fmm_compact as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Reverse construction reached numerical coefficient/layout generation')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'helmholtz_taylor', forbidden)
        patch.setattr(module, '_coefficient_layout', forbidden)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reverse = owner.reversed_translation()
        assert torch.cuda.memory_allocated() == allocated
    assert all(a is b for a, b in zip(owner.tensors(), reverse.tensors(), strict=True))
    assert storage_bytes((*owner.tensors(), *reverse.tensors())) == owner.retained_bytes
    assert owner.receipt()['version'] == 'compact_cartesian_m2l_coefficients_v1'
    assert reverse.receipt()['additional_owned_tensor_bytes'] == 0
    # The independent original dense factory evaluates the opposite center,
    # rather than reusing or transposing the new compact tables.
    green, gradient = multipole_to_local(-displacement, owner.medium.k, order)
    gradient = gradient.movedim(0, -1)
    tables = [reverse._tables(first, min(first+batch, reverse.count))
              for first in range(0, reverse.count, batch)]
    table_errors = dict(green=_relative(torch.cat([g for g, _ in tables]), green),
                        gradient=_relative(torch.cat([h for _, h in tables]), gradient))
    assert max(table_errors.values()) < 1e-11
    generator = torch.Generator(device='cuda').manual_seed(104729+order)
    current = torch.randn((3, 5, 2, reverse.count, 4), device='cuda', dtype=reverse.dtype,
                          generator=generator).transpose(0, 1).requires_grad_()
    dual = torch.randn(current.shape, device='cuda', dtype=reverse.dtype,
                       generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        forward = reverse.apply(current)
    assert not saved and not current.is_contiguous()
    H = reverse.adjoint(dual)
    derivative, = torch.autograd.grad(forward, current, dual, retain_graph=True)
    H_derivative, = torch.autograd.grad(H, dual, current, retain_graph=True)
    first, = torch.autograd.grad(forward.abs().square().sum(), current, create_graph=True)
    tangent = torch.randn_like(current)
    second, = torch.autograd.grad(first, current, tangent)
    lhs, rhs = (forward.conj()*dual).sum(), (current.conj()*H).sum()
    errors = dict(forward=_relative(forward, scalar_moment_action(green, gradient, owner.medium, current)),
                  H=_relative(H, scalar_moment_action(green, gradient, owner.medium, dual, adjoint=True)),
                  current=_relative(derivative, H), adjoint_current=_relative(H_derivative, forward),
                  higher=_relative(second, 2*reverse.adjoint(reverse.apply(tangent))),
                  dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach()))
    assert max(errors.values()) < 1e-11
    record = dict(order=order, row_batch=batch, native_RHS=15, magnetic_lossy_scalar=True,
                  opposite_displacement_dense_table_errors=table_errors, native_errors=errors,
                  saved_internal_tensors=len(saved), two_direction_unique_retained_bytes=owner.retained_bytes,
                  ordered_independent_two_direction_bytes=2*owner.retained_bytes,
                  reverse_constructor_numerical_allocations=0, receipt=reverse.receipt(),
                  full_FMM_integration=False, production_qualification=False,
                  whole_peak_or_crossover_qualified=False)
    (tmp_path/'translation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(record, flush=True)


def test_reversed_compact_independent_MP100_derivative_parity(tmp_path):
    displacement, owner = _problem()
    reverse = owner.reversed_translation()
    # Independent derivatives of the radial analytic kernel at both centers.
    # Host arbitrary precision is an oracle, never a production fallback.
    samples = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
               (2, 0, 0), (1, 1, 0), (0, 1, 1), (3, 0, 0), (1, 1, 1)]
    lookup = {index: i for i, index in enumerate(multi_indices(2*owner.order+1))}
    errors = []
    with mp.workdps(100):
        wave = mp.mpc(owner.medium.k.real, owner.medium.k.imag)
        def kernel(x, y, z):
            radius = mp.sqrt(x*x+y*y+z*z)
            return mp.exp(mp.j*wave*radius)/(4*mp.pi*radius)
        center = tuple(mp.mpf(float(v)) for v in displacement)
        for alpha in samples:
            original = complex(mp.diff(kernel, center, alpha)/math.prod(math.factorial(a) for a in alpha))
            opposite = complex(mp.diff(kernel, tuple(-v for v in center), alpha)
                               /math.prod(math.factorial(a) for a in alpha))
            native = complex(owner.coefficients[lookup[alpha]])
            parity_native = native*(-1)**sum(alpha)
            errors.extend((abs(native-original)/abs(original), abs(parity_native-opposite)/abs(opposite)))
    assert max(errors) < 1e-11
    assert reverse.coefficients is owner.coefficients
    record = dict(precision_digits=100, derivative_samples=samples, max_relative_error=max(errors),
                  independent_radial_kernel_reference=True, ordinary_degree_parity=True,
                  physical_quadrature_or_full_FMM_qualified=False, production_qualification=False)
    (tmp_path/'mp100.json').write_text(json.dumps(record, indent=2)+'\n')
    print(record, flush=True)


def test_reversed_compact_saved_owner_identity_mode_and_version_guards(tmp_path):
    _, owner = _problem()
    reverse = owner.reversed_translation()
    current = torch.ones((2, reverse.count, 4), device='cuda', dtype=reverse.dtype, requires_grad=True)
    with pytest.raises(ValueError, match='original owner'):
        reverse.reversed_translation()
    checked = []
    def checks(output):
        for call in (reverse.receipt, lambda: reverse.apply(current), lambda: reverse.adjoint(current),
                     lambda: torch.autograd.grad(output, current, torch.ones_like(output))):
            with pytest.raises(ValueError, match='changed'):
                call()
    for label, target, field, replacement in (
            ('original_row_batch', owner, 'row_batch', 8),
            ('reverse_row_batch', reverse, 'row_batch', 8),
            ('coefficient_identity', reverse, 'coefficients', reverse.coefficients.clone()),
            ('original_coefficient_identity', owner, 'coefficients', owner.coefficients.clone()),
            ('owner_identity', reverse, '_reverse_owner', _problem()[1])):
        output = reverse.apply(current)
        original = getattr(target, field)
        setattr(target, field, replacement)
        checks(output)
        setattr(target, field, original)
        reverse.assert_current()
        checked.append(label)
    output = reverse.apply(current)
    owner.coefficients.requires_grad_(True)
    checks(output)
    with pytest.raises(ValueError, match='fixed native'):
        owner.reversed_translation()
    owner.coefficients.requires_grad_(False)
    reverse.assert_current()
    checked.append('borrowed_requires_grad_mode')
    output = reverse.apply(current)
    owner.coefficients.add_(0)
    checks(output)
    checked.append('borrowed_coefficient_version')
    record = dict(saved_owner_guards=checked, chained_reverse_rejected=True,
                  differentiable_coefficients_rejected=True, production_qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record, indent=2)+'\n')
