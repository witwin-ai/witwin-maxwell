"""Near and far physical edge rules must retain nonzero line measure."""
import pytest
import torch
from witwin.maxwell.mom.quadrature import distance_scaled_line_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_far_projection_small_edges_and_zero_distance_extensions():
    along=torch.tensor([1e100,-1e100,1.,-2.,2.,.5],device='cuda',dtype=torch.float64)
    distance=along.new_tensor([1.,1.,1.,0.,0.,1e100])
    length=along.new_tensor([1.,1.,1e-100,1.,1.,1.])
    position,weight=distance_scaled_line_rule(along,distance,length,16)
    assert bool((weight>0).all() & (position.diff(dim=-1)>0).all())
    torch.testing.assert_close(weight.sum(-1),length,rtol=2e-15,atol=0)
    torch.testing.assert_close(((position/length[:,None])**2*weight).sum(-1),length/3,rtol=3e-15,atol=0)
    p=along[:1].clone().requires_grad_()
    s,w=distance_scaled_line_rule(p,distance[:1],length[:1],16)
    value=(w/torch.hypot(p[:,None]-s,distance[:1,None])).sum()
    derivative=torch.autograd.grad(value,p)[0]
    torch.testing.assert_close(value,along.new_tensor(1e-100),rtol=2e-15,atol=0)
    torch.testing.assert_close(derivative,along.new_tensor([-1e-200]),rtol=2e-15,atol=0)


def test_near_edge_line_potential_and_ordinary_transition():
    along=torch.tensor([.37,.37,.37,.37],device='cuda',dtype=torch.float64)
    distance=along.new_tensor([1e-7,.1,.999,1.001]);length=torch.ones_like(along)
    position,weight=distance_scaled_line_rule(along,distance,length,64)
    observed=(weight/torch.hypot(position-along[:,None],distance[:,None])).sum(-1)
    expected=torch.asinh((length-along)/distance)-torch.asinh(-along/distance)
    torch.testing.assert_close(observed,expected,rtol=2e-10,atol=1e-12)
    torch.testing.assert_close(weight.sum(-1),length,rtol=1e-13,atol=1e-14)


def test_far_slant_fan_preserves_area_and_first_moment():
    from witwin.maxwell.mom.quadrature import distance_scaled_duffy_fan
    panel=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],device='cuda',dtype=torch.float64)
    points=panel.new_tensor([[1e8,1e8,1.],[1e100,1e100,1.]])
    panels=panel.expand(2,-1,-1)
    for grading in (False,True):
        for order in (8,16):
            nodes,weights=distance_scaled_duffy_fan(points,panels,order,edge_grading=grading)
            assert nodes.shape==(2,3*order**2,3)
            torch.testing.assert_close(weights.sum(-1),points.new_full((2,),.5),rtol=2e-15,atol=0)
            torch.testing.assert_close((nodes*weights[:,:,None]).sum(1),
                points.new_tensor([[1/6,1/6,0.]]).expand(2,-1),rtol=3e-15,atol=0)
