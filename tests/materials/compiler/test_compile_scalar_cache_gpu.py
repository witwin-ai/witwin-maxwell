"""Real CUDA stream and tensor-gradient isolation; no historical fixtures."""
import pytest
import torch
from witwin.maxwell.compiler import _scalar_cache as cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA execution.")
def test_compile_scalar_cache_respects_actual_CUDA_streams_and_tensor_gradients():
    assert torch.cuda.is_available(), 'Explicit GPU qualification requires CUDA.'
    first, second = torch.cuda.Stream(), torch.cuda.Stream()

    @cache.with_compile_scalar_cache
    def construct():
        with torch.cuda.stream(first):
            left = cache.scalar_tensor(2., device='cuda', dtype=torch.float64)
            assert left is cache.scalar_tensor(2., device='cuda', dtype=torch.float64)
        with torch.cuda.stream(second):
            right = cache.scalar_tensor(2., device='cuda', dtype=torch.float64)
        assert left is not right
        return left, right

    left, right = construct()
    first.synchronize()
    second.synchronize()
    assert left.is_cuda and right.is_cuda and left.item() == right.item() == 2.
    assert cache._SCALARS.get() is None
    parameter = torch.tensor(3., device='cuda', dtype=torch.float64, requires_grad=True)

    @cache.with_compile_scalar_cache
    def differentiable():
        value = cache.scalar_tensor(parameter, device=parameter.device, dtype=parameter.dtype)
        assert value is parameter
        return value.square()

    value = differentiable()
    assert cache._SCALARS.get() is None
    assert torch.autograd.grad(value, parameter)[0].item() == 6.
