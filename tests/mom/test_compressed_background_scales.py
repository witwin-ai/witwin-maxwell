"""Real source scales and batched unpumped harmonics need no repeated solves."""
import runpy
import pytest
import torch
from witwin.maxwell.sources import PlaneWave
from witwin.maxwell.mom.compressed_background import prepare_compressed_background

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_real_scale_gradient_and_batched_empty_without_new_solves(monkeypatch):
    _,bindings,inverse=runpy.run_path('tests/mom/test_compressed_volume_preparation.py')['setup']()
    bank=prepare_compressed_background(inverse,bindings,(PlaneWave(),))
    empty=prepare_compressed_background(inverse,bindings,())
    def forbidden(*args,**kwargs):raise AssertionError('Combining prepared scales must not solve')
    monkeypatch.setattr(inverse,'apply',forbidden)
    scale=torch.tensor([.7],device='cuda',dtype=torch.float64,requires_grad=True)
    def objective(s):return bank.combine(s)['background_electric'].abs().square().sum()
    gradient,=torch.autograd.grad(objective(scale),scale)
    epsilon=1e-5
    finite_difference=(objective(scale.detach()+epsilon)-objective(scale.detach()-epsilon))/(2*epsilon)
    torch.testing.assert_close(gradient[0],finite_difference,rtol=1e-8,atol=1e-10)
    result=empty.combine(torch.empty(2,3,0,device='cuda',dtype=torch.float64))
    assert result['background_electric'].shape==(2,3,2,3)
    assert result['background_surface'].shape==(2,3,inverse.shape[0])
    assert all(not bool(v.any()) for v in result.values())
