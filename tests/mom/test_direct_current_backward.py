"""Fixed direct actions use guarded adjoints without saved projection graphs."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.direct_system import DirectSystemActions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_direct_backward_no_saved_graph_second_derivative_and_stale_guard():
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.1), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-.5, .5))])
    actions = DirectSystemActions(ScalarSystemBlocks(system, exterior,
        IntegrationConfig(order=2, singular_order=3)), order=3, leaf_size=30)
    n = actions.shape[0]
    for adjoint in (False, True):
        x = torch.randn((2, 3, 2*n), device='cuda', dtype=torch.complex128)[..., ::2].requires_grad_()
        assert not x.is_contiguous()
        saved = []
        def pack(value):
            saved.append(value)
            return value
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            result = actions.adjoint(x) if adjoint else actions.apply(x)
        assert not saved
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            reference = actions._action(x, adjoint=adjoint)
        assert saved
        print('raw saved tensor entries', adjoint, len(saved))
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        probe = torch.randn_like(result)
        gradient, = torch.autograd.grad(result, x, probe, retain_graph=True)
        expected, = torch.autograd.grad(reference, x, probe)
        torch.testing.assert_close(gradient, expected, rtol=2e-12, atol=1e-12)
        loss = result.abs().square().sum()/2
        first, = torch.autograd.grad(loss, x, create_graph=True)
        direction = torch.randn_like(x)
        second, = torch.autograd.grad(first, x, direction)
        with torch.no_grad():
            hessian_action = actions.apply(actions.adjoint(direction)) if adjoint else actions.adjoint(actions.apply(direction))
        torch.testing.assert_close(second, hessian_action, rtol=2e-12, atol=1e-12)
        del reference, saved
    pending = actions.apply(x)
    actions.exterior.far[0][2].green.add_(1)
    with pytest.raises(ValueError, match='changed'):
        torch.autograd.grad(pending, x, torch.ones_like(pending))
