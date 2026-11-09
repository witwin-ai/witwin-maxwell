"""Independent closed-form kernel equivalence and derivative controls."""
import math
import pytest
import torch
from witwin.maxwell.mom.kernels import FreeSpaceHelmholtzKernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_free_space_matches_closed_form_and_gradient():
    kernel = FreeSpaceHelmholtzKernel()
    displacement = torch.tensor([[.2, -.3, .4], [1., .2, -.1]], device='cuda', dtype=torch.float64, requires_grad=True)
    k = 3.1+.17j
    value, gradient = kernel.evaluate(displacement, k)
    r = displacement.square().sum(-1).sqrt()
    expected = torch.exp(1j*k*r)/(4*math.pi*r)
    assert torch.allclose(value, expected, rtol=2e-15, atol=0)
    real = torch.autograd.grad(value.real.sum(), displacement, retain_graph=True)[0]
    imag = torch.autograd.grad(value.imag.sum(), displacement)[0]
    assert torch.allclose(gradient, torch.complex(real, imag), rtol=2e-15, atol=1e-15)
    assert kernel.receipt() == FreeSpaceHelmholtzKernel().receipt()


def test_free_space_rejects_unhandled_singularity():
    with pytest.raises(ValueError, match='singular quadrature'):
        FreeSpaceHelmholtzKernel().evaluate(torch.zeros((1, 3), device='cuda'), 1.)


def test_kernel_is_injected_into_assembly_and_field_potentials():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import sphere_mesh, RWGBasis
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.operator import assemble_weak, IntegrationConfig
    from witwin.maxwell.mom.runtime import DecodeHandle
    from witwin.maxwell.mom.fields import _potentials

    class ObservedKernel:
        kernel_id = 'free_space_helmholtz'
        version = 'outgoing_helmholtz_v1'
        local_singularity = 'coulomb_1_over_4pi_r'
        calls = 0
        def receipt(self): return FreeSpaceHelmholtzKernel().receipt()
        def evaluate(self, displacement, k):
            self.calls += 1
            return FreeSpaceHelmholtzKernel().evaluate(displacement, k)

    kernel = ObservedKernel()
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    integration = IntegrationConfig(order=2, singular_order=4)
    actual, _ = assemble_weak(basis, medium, medium, integration, kernel=kernel)
    expected, _ = assemble_weak(basis, medium, medium, integration)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    assert kernel.calls > 0
    handle = DecodeHandle(basis, medium, medium, (), integration, {})
    x = torch.arange(2*basis.n_edges, device='cuda', dtype=torch.float64).to(torch.complex128)
    points = torch.tensor([[.3, .2, .4]], device='cuda', dtype=torch.float64)
    reference = _potentials(handle, x, points, medium)
    handle.kernel = kernel
    count = kernel.calls
    observed = _potentials(handle, x, points, medium)
    assert kernel.calls > count
    for a, b in zip(reference, observed):
        torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)
