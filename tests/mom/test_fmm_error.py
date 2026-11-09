"""Cauchy truncation bounds for the implemented double-total-degree M2L."""
import pytest
import torch
from witwin.maxwell.mom.fmm_error import multipole_truncation_bound, require_multipole_budget, MultipoleTruncationError
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget, KernelBoundUnavailable
from witwin.maxwell.mom.fmm_expansion import monomials, multipole_to_local
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('wave', [0., 6.3, -6.3+.4j])
def test_actual_truncation_is_below_cauchy_bound(wave):
    generator = torch.Generator(device='cuda').manual_seed(316)
    displacement = torch.tensor([1., .3, -.2], device='cuda', dtype=torch.float64)
    radius = displacement.new_tensor(.025)
    points = torch.randn((16, 3), device='cuda', dtype=torch.float64, generator=generator)
    points = points/torch.linalg.vector_norm(points, dim=-1, keepdim=True)*radius
    source, target = points[:8], points[8:]
    exact, gradient = FREE_SPACE.evaluate(displacement+target[:, None]-source[None], wave)
    previous = float('inf')
    for order in (2, 4, 6):
        bound = multipole_truncation_bound(displacement, radius, radius, wave, order)
        green, grad = multipole_to_local(displacement, wave, order)
        bt, bs = (monomials(p, order).to(green.dtype) for p in (target, source))
        error = float((bt@green@bs.mT-exact).abs().max())
        gradient_error = float((torch.einsum('ib,abc,jc->ija', bt, grad, bs)-gradient).abs().max())
        print('order/error/bound', wave, order, error, bound['green_absolute'], gradient_error, bound['gradient_component_absolute'])
        assert error <= bound['green_absolute']
        assert gradient_error <= bound['gradient_component_absolute']
        assert bound['green_absolute'] < previous
        previous = bound['green_absolute']
        assert not bound['implemented_operator_certified']
        require_multipole_budget(bound, KernelErrorBudget(bound['green_absolute']*2, bound['gradient_component_absolute']*2))
        with pytest.raises(MultipoleTruncationError):
            require_multipole_budget(bound, KernelErrorBudget(bound['green_absolute']/2, bound['gradient_component_absolute']/2))


def test_degenerate_supports_unavailable_contour_and_log_underflow():
    r = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    zero, radius = r.new_tensor(0.), r.new_tensor(.02)
    exact = multipole_truncation_bound(r, zero, zero, 3., 0)
    assert exact['green_absolute'] == exact['gradient_component_absolute'] == 0
    one = multipole_truncation_bound(r, radius, zero, 3., 4)
    two = multipole_truncation_bound(r, radius, radius, 3., 4)
    assert one['green_absolute'] < two['green_absolute']
    with pytest.raises(KernelBoundUnavailable, match='contour'):
        multipole_truncation_bound(r, r.new_tensor(.21), r.new_tensor(.21), 3., 4)
    tiny_r = r*1e-100
    tiny = multipole_truncation_bound(tiny_r, r.new_tensor(1e-103), r.new_tensor(1e-103), 8e102j, 0)
    assert tiny['green_absolute'] > 1e-300 and tiny['gradient_component_absolute'] > tiny['green_absolute']
    floor = multipole_truncation_bound(r, radius, radius, 1e6j, 6)
    assert floor['green_absolute'] > 0 and floor['gradient_component_absolute'] > 0
