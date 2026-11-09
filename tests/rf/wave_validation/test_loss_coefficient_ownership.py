"""Coherent diagnostic coefficients own storage without cutting tensor gradients."""
import warnings
import numpy as np
import pytest
import torch
from scripts.dev.fdtd.printed_line_loss_execution import CoherentObservations
from tests.rf.wave_validation.test_printed_line_loss_execution_contract import prescribed_observation


@pytest.mark.parametrize('kind', ['readonly_broadcast', 'writable_numpy', 'nested_sequence'])
def test_coherent_view_owns_caller_values_and_does_not_warn_for_readonly_broadcast(kind):
    result, basis = prescribed_observation()
    original = np.array([1., .5], dtype=np.complex128)
    if kind == 'readonly_broadcast':
        supplied = np.broadcast_to(original, (1, 2))
        assert not supplied.flags.writeable
    elif kind == 'writable_numpy':
        supplied = original[None]
    else:
        supplied = [[1., .5]]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        view = CoherentObservations([result, result], [basis, basis], supplied)
    before = view.monitor('plane', frequency=1e9)['components']['Ex']['data'].clone()
    original[:] = 9.
    if kind == 'nested_sequence':
        supplied[0][:] = [9., 9.]
    torch.testing.assert_close(view.coefficients, torch.tensor([[1., .5]], dtype=torch.complex128), rtol=0, atol=0)
    torch.testing.assert_close(view.monitor('plane', frequency=1e9)['components']['Ex']['data'], before, rtol=0, atol=0)
    if kind != 'nested_sequence':
        assert not np.shares_memory(view.coefficients.numpy(), supplied)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_owned_tensor_coefficients_retain_gradient_and_do_not_alias_mutated_caller(dtype):
    result, basis = prescribed_observation()
    supplied = torch.tensor([[1., .5]], dtype=dtype, requires_grad=True)
    view = CoherentObservations([result, result], [basis, basis], supplied)
    assert view.coefficients.data_ptr() != supplied.data_ptr()
    assert view.coefficients.requires_grad
    with torch.no_grad():
        supplied.fill_(9.)
    actual = view.monitor('plane', frequency=1e9)['components']['Ex']['data']
    torch.testing.assert_close(actual, torch.full((2, 2), 1.5, dtype=torch.complex128), rtol=0, atol=0)
    actual.abs().square().sum().backward()
    torch.testing.assert_close(supplied.grad, torch.full((1, 2), 12., dtype=dtype), rtol=1e-14, atol=0)


@pytest.fixture(autouse=True)
def _cpu_preparation_keeps_CUDA_initialization_state():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before
