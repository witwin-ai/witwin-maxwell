"""Repeated cell contributions retain exact CUDA replay and native gradients."""
import pytest
import torch
from witwin.maxwell.mom.ordered_reduction import add_ordered_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_ordered_rows_replay_and_gradient():
    rows = torch.tensor([0]*108+[2]*5+[4]*129, device='cuda')
    source = torch.randn(len(rows),6,6,device='cuda',dtype=torch.complex128,requires_grad=True)
    base = torch.randn(5,6,6,device='cuda',dtype=source.dtype,requires_grad=True)
    actual = add_ordered_rows(base,rows,source)
    expected = torch.stack([base[i]+source[rows==i].sum(0) for i in range(5)])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for _ in range(16):
        torch.testing.assert_close(add_ordered_rows(base,rows,source),actual,rtol=0,atol=0)
    gradient = torch.randn_like(actual)
    got = torch.autograd.grad(actual,(base,source),gradient,retain_graph=True)
    reference = torch.autograd.grad(expected,(base,source),gradient)
    for a,b in zip(got,reference):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert add_ordered_rows(base,rows[:0],source[:0]) is base
