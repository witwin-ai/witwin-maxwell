"""Streamed tensor volume geometry retains values and native query gradients."""
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom import tensor_volume

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('rule',['triangle','projected_duffy','distance_scaled_duffy'])
def test_streamed_geometry_values_gradients_and_fan_reuse(monkeypatch,rule):
    op,binding=setup();kernel=op.body_handles[0].tensor_kernel
    points=binding.volume.vertices.new_tensor([[.13,.17,.11]],requires_grad=True)
    calls=[]
    if rule!='triangle':
        name='duffy_fan' if rule=='projected_duffy' else 'distance_scaled_duffy_fan'
        original=getattr(tensor_volume,name)
        def counted(*args,**kwargs):
            calls.append(len(args[0]));return original(*args,**kwargs)
        monkeypatch.setattr(tensor_volume,name,counted)
    streamed,receipt=tensor_volume.cell_tensor_current_blocks(binding.volume,points,kernel,
        order=2,radial_order=3,kernel_batch=7,face_rule=rule)
    if rule!='triangle':
        # A fan is reused across multiple displacement batches. This assertion
        # measures eliminated geometry evaluations, not wall-time speedup.
        assert len(calls)<len(receipt['spectral_evaluations'])
    whole,other=tensor_volume.cell_tensor_current_blocks(binding.volume,points,kernel,
        order=2,radial_order=3,kernel_batch=8192,face_rule=rule)
    torch.testing.assert_close(streamed,whole,rtol=2e-12,atol=1e-10)
    left=torch.autograd.grad(streamed.abs().square().sum(),points)[0]
    right=torch.autograd.grad(whole.abs().square().sum(),points)[0]
    torch.testing.assert_close(left,right,rtol=2e-11,atol=1e-9)
    assert receipt['kernel_nodes']==other['kernel_nodes']
    assert not receipt['physical_qualification']
