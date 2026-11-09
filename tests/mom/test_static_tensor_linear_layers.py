"""Analytic P1 conormal moments against independent physical point cubature."""
import pytest
import torch
from test_static_tensor_green import coefficient
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_linear_double_layer_point_cubature_native_derivatives_and_orientation():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2;kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]]).expand(2,-1,-1)
    points=a.new_tensor([[.24,-.12,.31],[-.21,.23,-.17]],requires_grad=True)
    actual=kernel.panel_linear_double_layer(points,tri)
    nodes,weights=triangle_rule(tri,16)
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    cross=torch.linalg.cross(e1,e2);normal=cross/cross.norm(dim=-1,keepdim=True)
    # Recover physical quadrature barycentrics by an independent Gram solve.
    edges=torch.stack((e1,e2),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('qij,qni->qjn',edges,nodes-tri[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    _,gradient=kernel.evaluate((points[:,None]-nodes).reshape(-1,3))
    density=-(gradient.reshape_as(nodes)@a.T*normal[:,None]).sum(-1)
    expected=(weights[:,:,None]*density[:,:,None]*bary).sum(1)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-14)
    native=torch.autograd.grad(actual.square().sum(),(points,parameter),retain_graph=True)
    oracle=torch.autograd.grad(expected.square().sum(),(points,parameter),retain_graph=True)
    for x,y in zip(native,oracle):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-14)
    _,grad=kernel.panel_integrals(points,tri)
    torch.testing.assert_close(actual.sum(-1),-(grad@a.T*normal).sum(-1),rtol=1e-12,atol=1e-14)
    reverse=kernel.panel_linear_double_layer(points,tri[:,[0,2,1]])
    torch.testing.assert_close(reverse,-actual[:,[0,2,1]],rtol=1e-12,atol=1e-14)
    scaled=StaticTensorGreen(17*a).panel_linear_double_layer(points,tri)
    torch.testing.assert_close(scaled,actual,rtol=1e-12,atol=1e-14)


def test_linear_double_layer_pv_and_translation_admission():
    a=coefficient();kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]]])
    center=tri.mean(1)
    with pytest.raises(ValueError,match='principal values'):kernel.panel_linear_double_layer(center,tri)
    pv=kernel.panel_linear_double_layer(center,tri,principal_value=torch.ones(1,device='cuda',dtype=torch.bool))
    torch.testing.assert_close(pv,torch.zeros_like(pv),rtol=0,atol=1e-15)
    with pytest.raises(ValueError,match='edges'):
        kernel.panel_linear_double_layer(tri[:,0],tri,principal_value=torch.ones(1,device='cuda',dtype=torch.bool))
    point=a.new_tensor([[.03,.02,.2]])
    shift=a.new_tensor([10.,-20.,30.])
    torch.testing.assert_close(kernel.panel_linear_double_layer(point+shift,tri+shift),
        kernel.panel_linear_double_layer(point,tri),rtol=1e-11,atol=1e-13)


def test_linear_single_layer_point_cubature_native_derivatives_and_orientation():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2;kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]]).expand(2,-1,-1)
    points=a.new_tensor([[.24,-.12,.31],[-.21,.23,-.17]],requires_grad=True)
    actual=kernel.panel_linear_single_layer(points,tri)
    nodes,weights=triangle_rule(tri,16)
    edges=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('qij,qni->qjn',edges,nodes-tri[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    density=kernel.evaluate((points[:,None]-nodes).reshape(-1,3))[0].reshape_as(weights)
    expected=(weights[:,:,None]*density[:,:,None]*bary).sum(1)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-14)
    native=torch.autograd.grad(actual.square().sum(),(points,parameter),retain_graph=True)
    oracle=torch.autograd.grad(expected.square().sum(),(points,parameter),retain_graph=True)
    for x,y in zip(native,oracle):torch.testing.assert_close(x,y,rtol=1e-8,atol=1e-14)
    single,_=kernel.panel_integrals(points,tri)
    torch.testing.assert_close(actual.sum(-1),single,rtol=1e-12,atol=1e-14)
    reverse=kernel.panel_linear_single_layer(points,tri[:,[0,2,1]])
    torch.testing.assert_close(reverse,actual[:,[0,2,1]],rtol=1e-11,atol=1e-14)
    scaled=StaticTensorGreen(17*a).panel_linear_single_layer(points,tri)
    torch.testing.assert_close(17*scaled,actual,rtol=1e-11,atol=1e-14)


def test_linear_single_layer_face_symmetry_jump_and_translation():
    eye=torch.eye(3,device='cuda',dtype=torch.float64);kernel=StaticTensorGreen(eye)
    tri=eye.new_tensor([[[0.,0.,0.],[.1,0.,0.],[.05,3**.5*.05,0.]]])
    center=tri.mean(1);mask=torch.ones(1,device='cuda',dtype=torch.bool)
    with pytest.raises(ValueError,match='principal values'):kernel.panel_linear_single_layer(center,tri)
    value=kernel.panel_linear_single_layer(center,tri,principal_value=mask)
    constant=kernel.panel_integrals(center,tri,principal_value=mask)[0]
    torch.testing.assert_close(value,constant[:,None].expand(-1,3)/3,rtol=1e-12,atol=1e-14)
    with pytest.raises(ValueError,match='edges'):
        kernel.panel_linear_single_layer(tri[:,0],tri,principal_value=mask)
    for sign in (-1,1):
        point=(center+eye.new_tensor([[0.,0.,sign*1e-7]])).requires_grad_()
        moments=kernel.panel_linear_single_layer(point,tri)
        for i in range(3):
            gradient,=torch.autograd.grad(moments[0,i],point,retain_graph=True)
            torch.testing.assert_close(gradient[0,2],eye.new_tensor(-sign/6),rtol=1e-5,atol=1e-8)
    point=eye.new_tensor([[.03,.02,.2]]);shift=eye.new_tensor([10.,-20.,30.])
    torch.testing.assert_close(kernel.panel_linear_single_layer(point+shift,tri+shift),
        kernel.panel_linear_single_layer(point,tri),rtol=1e-10,atol=1e-13)


def test_quadratic_double_layer_point_cubature_native_derivatives_and_basis():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2;kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]]).expand(2,-1,-1)
    points=a.new_tensor([[.24,-.12,.31],[-.21,.23,-.17]],requires_grad=True)
    actual=kernel.panel_quadratic_double_layer(points,tri)
    nodes,weights=triangle_rule(tri,16)
    edges=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('qij,qni->qjn',edges,nodes-tri[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    basis=torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,:,0]*bary[:,:,1],
        bary[:,:,1]*bary[:,:,2],bary[:,:,2]*bary[:,:,0]),-1)),-1)
    cross=torch.linalg.cross(edges[:,:,0],edges[:,:,1]);normal=cross/cross.norm(dim=-1,keepdim=True)
    _,gradient=kernel.evaluate((points[:,None]-nodes).reshape(-1,3))
    density=-(gradient.reshape_as(nodes)@a.T*normal[:,None]).sum(-1)
    expected=(weights[:,:,None]*density[:,:,None]*basis).sum(1)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-13)
    native=torch.autograd.grad(actual.square().sum(),(points,parameter),retain_graph=True)
    oracle=torch.autograd.grad(expected.square().sum(),(points,parameter),retain_graph=True)
    for x,y in zip(native,oracle):torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-13)
    constant=-(kernel.panel_integrals(points,tri)[1]@a.T*normal).sum(-1)
    torch.testing.assert_close(actual.sum(-1),constant,rtol=1e-11,atol=1e-13)
    reversed=kernel.panel_quadratic_double_layer(points,tri[:,[0,2,1]])
    torch.testing.assert_close(reversed,-actual[:,[0,2,1,5,4,3]],rtol=1e-10,atol=1e-13)
    with torch.no_grad():
        detached=kernel.panel_quadratic_double_layer(points.detach(),tri)
    assert not detached.requires_grad
    torch.testing.assert_close(detached,actual,rtol=1e-12,atol=1e-13)
    center=tri.mean(1)
    with pytest.raises(ValueError,match='principal values'):kernel.panel_quadratic_double_layer(center,tri)
    value=kernel.panel_quadratic_double_layer(center,tri,principal_value=torch.ones(2,device='cuda',dtype=torch.bool))
    torch.testing.assert_close(value,torch.zeros_like(value),rtol=0,atol=1e-13)


def test_higher_layer_anisotropic_face_jumps_and_empty_queries():
    a=coefficient();kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]])
    bary=a.new_tensor([[.2,.3,.5]]);center=(bary[:,:,None]*tri).sum(1)
    normal=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal=normal/normal.norm(dim=-1,keepdim=True)
    derivatives=[];potentials=[]
    for sign in (-1,1):
        point=(center+sign*1e-8*normal).requires_grad_()
        single=kernel.panel_linear_single_layer(point,tri)
        gradient=torch.stack([torch.autograd.grad(single[0,i],point,retain_graph=True)[0][0] for i in range(3)])
        derivatives.append((gradient@a.T*normal).sum(-1))
        potentials.append(kernel.panel_quadratic_double_layer(point,tri))
    torch.testing.assert_close(derivatives[1]-derivatives[0],-bary[0],rtol=1e-5,atol=1e-7)
    basis=torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,0]*bary[:,1],bary[:,1]*bary[:,2],bary[:,2]*bary[:,0]),-1)),-1)
    torch.testing.assert_close(potentials[1]-potentials[0],basis,rtol=1e-5,atol=1e-7)
    points=a.new_empty(0,3,requires_grad=True);panels=a.new_empty(0,3,3)
    assert kernel.panel_linear_single_layer(points,panels).shape==(0,3)
    assert kernel.panel_quadratic_double_layer(points,panels).shape==(0,6)
    with pytest.raises(ValueError,match='CUDA coordinates'):
        kernel.panel_quadratic_double_layer(None,tri)


def test_higher_layer_far_moments_and_gradients_match_point_cubature():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2;kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[0.,0.,0.],[.1,0.,0.],[.02,.13,.03]]]).expand(4,-1,-1)
    points=(tri.mean(1)+a.new_tensor([1.,10.,100.,1000.])[:,None]*a.new_tensor([[.3,-.2,.7]])).requires_grad_()
    nodes,weights=triangle_rule(tri,20)
    edges=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('qij,qni->qjn',edges,nodes-tri[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    basis=torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,:,0]*bary[:,:,1],bary[:,:,1]*bary[:,:,2],bary[:,:,2]*bary[:,:,0]),-1)),-1)
    normal=torch.linalg.cross(edges[:,:,0],edges[:,:,1]);normal=normal/normal.norm(dim=-1,keepdim=True)
    green,gradient=kernel.evaluate((points[:,None]-nodes).reshape(-1,3))
    oracle_s=(weights[:,:,None]*green.reshape_as(weights)[:,:,None]*bary).sum(1)
    oracle_d=(weights[:,:,None]*(-(gradient.reshape_as(nodes)@a.T*normal[:,None]).sum(-1))[:,:,None]*basis).sum(1)
    single=kernel.panel_linear_single_layer(points,tri)
    double=kernel.panel_quadratic_double_layer(points,tri)
    for actual,expected in ((single,oracle_s),(double,oracle_d)):
        errors=(actual-expected).norm(dim=-1)/expected.norm(dim=-1)
        assert bool((errors<1e-9).all())
        # Per-pair normalization prevents the nearest point from masking a
        # far-field error in the derivative comparison.
        scale=expected.detach().norm(dim=-1)
        own=torch.autograd.grad((actual/scale[:,None]).square().sum(),(points,parameter),retain_graph=True)
        want=torch.autograd.grad((expected/scale[:,None]).square().sum(),(points,parameter),retain_graph=True)
        for x,y in zip(own,want):torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-9)


def test_linear_single_layer_pv_gradient_matches_two_sided_mean():
    a=coefficient();kernel=StaticTensorGreen(a)
    tri=a.new_tensor([[[.01,.02,.03],[.13,.01,.02],[.02,.14,.05]]])
    normal=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal=normal/normal.norm(dim=-1,keepdim=True)
    mask=torch.ones(1,device=a.device,dtype=torch.bool)
    def gradient(point,pv=None):
        point=point.detach().requires_grad_()
        value=kernel.panel_linear_single_layer(point,tri,principal_value=pv)
        return torch.stack([torch.autograd.grad(value[0,i],point,retain_graph=True)[0][0]
                            for i in range(3)])
    for bary in ([1/3,1/3,1/3],[.2,.3,.5],[2/3,1/6,1/6]):
        center=(a.new_tensor(bary)[None,:,None]*tri).sum(1)
        pv=gradient(center,mask)
        mean=[]
        for height in (1e-6,5e-7):
            mean.append((gradient(center-height*normal)+gradient(center+height*normal))/2)
        limit=2*mean[1]-mean[0]
        torch.testing.assert_close(pv,limit,rtol=1e-8,atol=1e-10)
        # Source and observation lie in the same physical plane: n.A.grad G=0
        # away from the singularity, hence every P1 self PV conormal is zero.
        conormal=(pv@a.T*normal).sum(-1)
        torch.testing.assert_close(conormal,torch.zeros_like(conormal),rtol=0,atol=1e-12)
