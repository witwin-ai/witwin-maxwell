"""The fresh-interpreter no-CUDA helper passes host work and rejects CUDA use."""
import pytest
import torch

from tests._fresh_interpreter import run_without_cuda


def _host_only(scale):
    assert torch.ones(2).mul(scale).sum().item() == 2 * scale


def _initializes_cuda():
    torch.zeros(1, device='cuda')


def _raises():
    raise ValueError('child failure is surfaced')


def test_host_only_check_passes_with_literal_arguments():
    run_without_cuda(__file__, '_host_only', 1.5)


def test_cuda_use_in_the_check_fails_even_after_the_parent_initialized_cuda():
    torch.zeros(1, device='cuda')
    with pytest.raises(AssertionError, match='host-only check initialized CUDA'):
        run_without_cuda(__file__, '_initializes_cuda')


def test_child_traceback_is_the_failure_message():
    with pytest.raises(AssertionError, match='ValueError: child failure is surfaced'):
        run_without_cuda(__file__, '_raises')
