"""P2 single-layer values and analytic gradients against independent cubature."""
import pytest
import torch
from test_static_tensor_green import coefficient
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.quadrature import triangle_rule


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

def test_quadratic_near_values_gradients_native_derivatives():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2;kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]]).expand(2,-1,-1)
    points=a.new_tensor([[.24,-.12,.31],[-.21,.23,-.17]],requires_grad=True)
    actual,gradient=kernel.panel_quadratic_integrals(points,tri)
    nodes,weights=triangle_rule(tri,32)
    edges=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('pij,pni->pjn',edges,nodes-tri[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    basis=torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,:,0]*bary[:,:,1],bary[:,:,1]*bary[:,:,2],bary[:,:,2]*bary[:,:,0]),-1)),-1)
    values,grads=kernel.evaluate((points[:,None]-nodes).reshape(-1,3))
    expected=torch.einsum('pq,pq,pqa->pa',weights,values.reshape_as(weights),basis)
    expected_grad=torch.einsum('pq,pqi,pqa->pai',weights,grads.reshape_as(nodes),basis)
    torch.testing.assert_close(actual,expected,rtol=2e-9,atol=1e-13)
    torch.testing.assert_close(gradient,expected_grad,rtol=2e-9,atol=1e-13)
    for i in range(6):
        native=torch.autograd.grad(actual[:,i].sum(),points,retain_graph=True)[0]
        torch.testing.assert_close(gradient[:,i],native,rtol=2e-9,atol=1e-13)
    for actual_loss,expected_loss in ((actual.square().sum(),expected.square().sum()),(gradient.square().sum(),expected_grad.square().sum())):
        native=torch.autograd.grad(actual_loss,(points,parameter),retain_graph=True)
        oracle=torch.autograd.grad(expected_loss,(points,parameter),retain_graph=True)
        for x,y in zip(native,oracle):torch.testing.assert_close(x,y,rtol=2e-7,atol=1e-13)
    constant,cgradient=kernel.panel_integrals(points,tri)
    torch.testing.assert_close(actual.sum(1),constant,rtol=2e-10,atol=1e-13)
    torch.testing.assert_close(gradient.sum(1),cgradient,rtol=2e-10,atol=1e-13)


def test_quadratic_anisotropic_jump_and_pv():
    a=coefficient();kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]])
    bary=a.new_tensor([[.2,.3,.5]]);center=(tri*bary[:,:,None]).sum(1)
    n=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n=n/n.norm(dim=-1,keepdim=True)
    basis=torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,0]*bary[:,1],bary[:,1]*bary[:,2],bary[:,2]*bary[:,0]),-1)),-1)
    sides=[]
    for sign in (-1,1):
        sides.append(kernel.panel_quadratic_integrals(center+sign*1e-8*n,tri)[1])
    jump=((sides[1]-sides[0])@a.T*n[:,None]).sum(-1)
    torch.testing.assert_close(jump,-basis,rtol=2e-5,atol=1e-7)
    pv=kernel.panel_quadratic_integrals(center,tri,principal_value=torch.ones(1,device='cuda',dtype=torch.bool))[1]
    torch.testing.assert_close(pv,(sides[1]+sides[0])/2,rtol=2e-5,atol=1e-7)


def test_quadratic_single_layer_orientation_scaling_translation_and_empty():
    a=coefficient();kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]])
    point=a.new_tensor([[.04,.03,.08]])
    value,gradient=kernel.panel_quadratic_integrals(point,tri)
    permutation=[0,2,1,5,4,3]
    reverse=kernel.panel_quadratic_integrals(point,tri[:,[0,2,1]])
    for actual,expected in zip(reverse,(value[:,permutation],gradient[:,permutation])):
        torch.testing.assert_close(actual,expected,rtol=2e-10,atol=1e-13)
    shifted=kernel.panel_quadratic_integrals(point+a.new_tensor([10.,-20.,30.]),tri+a.new_tensor([10.,-20.,30.]))
    scaled=StaticTensorGreen(17*a).panel_quadratic_integrals(point,tri)
    for actual,expected in zip(shifted,(value,gradient)):
        torch.testing.assert_close(actual,expected,rtol=2e-9,atol=1e-12)
    for actual,expected in zip(scaled,(value,gradient)):
        torch.testing.assert_close(17*actual,expected,rtol=2e-10,atol=1e-13)
    with torch.inference_mode():
        inferred=kernel.panel_quadratic_integrals(point,tri)
    for actual,expected in zip(inferred,(value,gradient)):
        torch.testing.assert_close(actual,expected)
    empty=kernel.panel_quadratic_integrals(a.new_empty(0,3),a.new_empty(0,3,3))
    assert empty[0].shape==(0,6) and empty[1].shape==(0,6,3)
    with pytest.raises(ValueError,match='principal values'):
        kernel.panel_quadratic_integrals(tri.mean(1),tri)
    with pytest.raises(ValueError,match='edges'):
        kernel.panel_quadratic_integrals(tri[:,0],tri,principal_value=torch.ones(1,device='cuda',dtype=torch.bool))


def test_quadratic_far_single_layers_and_gradients():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2;kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[0.,0.,0.],[.1,0.,0.],[.02,.13,.03]]]).expand(4,-1,-1)
    points=(tri.mean(1)+a.new_tensor([1.,10.,100.,1000.])[:,None]*a.new_tensor([[.3,-.2,.7]])).requires_grad_()
    actual,gradient=kernel.panel_quadratic_integrals(points,tri)
    nodes,weights=triangle_rule(tri,24)
    edges=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('pij,pni->pjn',edges,nodes-tri[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    basis=torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,:,0]*bary[:,:,1],bary[:,:,1]*bary[:,:,2],bary[:,:,2]*bary[:,:,0]),-1)),-1)
    values,grads=kernel.evaluate((points[:,None]-nodes).reshape(-1,3))
    expected=torch.einsum('pq,pq,pqa->pa',weights,values.reshape_as(weights),basis)
    expected_grad=torch.einsum('pq,pqi,pqa->pai',weights,grads.reshape_as(nodes),basis)
    assert bool(((actual-expected).norm(dim=1)/expected.norm(dim=1)<2e-9).all())
    assert bool(((gradient-expected_grad).flatten(1).norm(dim=1)/expected_grad.flatten(1).norm(dim=1)<2e-9).all())
    native=torch.autograd.grad(actual.square().sum()+gradient.square().sum(),(points,parameter),retain_graph=True)
    oracle=torch.autograd.grad(expected.square().sum()+expected_grad.square().sum(),(points,parameter))
    for x,y in zip(native,oracle):torch.testing.assert_close(x,y,rtol=2e-7,atol=1e-14)


def test_quadratic_reused_gradient_conormal_identity():
    a=coefficient();kernel=StaticTensorGreen(a)
    base=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]])
    normal=torch.linalg.cross(base[:,1]-base[:,0],base[:,2]-base[:,0])
    normal=normal/normal.norm(dim=-1,keepdim=True)
    center=(base*a.new_tensor([.17,.31,.52])[None,:,None]).sum(1)
    points=torch.cat((center+.03*normal,center-.03*normal,center+1e-8*normal,
                      center-1e-8*normal,center),0)
    triangles=base.expand(len(points),-1,-1)
    mask=torch.tensor([False,False,False,False,True],device='cuda')
    _,gradient=kernel.panel_quadratic_integrals(points,triangles,principal_value=mask)
    expected=kernel.panel_quadratic_double_layer(points,triangles,principal_value=mask)
    torch.testing.assert_close(-(gradient@a.T*normal[:,None]).sum(-1),expected,rtol=2e-9,atol=1e-12)
