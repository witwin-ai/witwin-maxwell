"""Disjoint weak blocks with independent double quadrature and rejection gates."""
import pytest
import torch
from test_static_tensor_green import coefficient
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def panels(a,gap=.2):
    left=a.new_tensor([[[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]]])
    return left,left+a.new_tensor([.02,.01,gap])


def test_disjoint_weak_independent_product_rule_and_native_material_gradient():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2
    kernel=StaticTensorGreen(a);left,right=panels(a)
    actual=kernel.disjoint_linear_weak_layers(left,right,order=12)
    x,wx=triangle_rule(left,16);y,wy=triangle_rule(right,16)
    def bary(points,triangle):
        edges=torch.stack((triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),-1)
        uv=torch.linalg.solve(edges.mT@edges,torch.einsum('pij,pqi->pjq',edges,points-triangle[:,:1])).mT
        return torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    lb,rb=bary(x,left),bary(y,right)
    value,gradient=kernel.evaluate((x[:,:,None]-y[:,None]).reshape(-1,3))
    weight=wx[:,:,None]*wy[:,None]
    normal=torch.linalg.cross(left[:,1]-left[:,0],left[:,2]-left[:,0]);normal=normal/normal.norm(dim=-1,keepdim=True)
    density=(gradient.reshape(1,len(x[0]),len(y[0]),3)@a.T*normal[:,None,None]).sum(-1)
    oracle=(torch.einsum('pqr,pqa,prb->pab',weight*value.reshape_as(weight),lb,rb),
            torch.einsum('pqr,pqa,prb->pab',weight*density,lb,rb))
    for got,want in zip(actual,oracle):torch.testing.assert_close(got,want,rtol=1e-9,atol=1e-13)
    own,=torch.autograd.grad(sum(v.square().sum() for v in actual),parameter,retain_graph=True)
    expected,=torch.autograd.grad(sum(v.square().sum() for v in oracle),parameter)
    torch.testing.assert_close(own,expected,rtol=1e-7,atol=1e-14)
    reverse=kernel.disjoint_linear_weak_layers(right,left,order=12)[0]
    torch.testing.assert_close(reverse.mT,actual[0],rtol=1e-9,atol=1e-13)


def test_disjoint_adaptive_mixed_separations_and_failure_admission():
    a=coefficient();kernel=StaticTensorGreen(a)
    pairs=[panels(a,gap) for gap in (.2,.015)]
    left=torch.cat([p[0] for p in pairs]);right=torch.cat([p[1] for p in pairs])
    single,conormal,receipt=kernel.adaptive_disjoint_linear_weak_layers(left,right,rtol=1e-6)
    reference=kernel.disjoint_linear_weak_layers(left,right,order=48,max_points=4096)
    for actual,want in zip((single,conormal),reference):torch.testing.assert_close(actual,want,rtol=2e-6,atol=1e-11)
    assert bool((receipt['relative_change']<=1e-6).all())
    assert not receipt['certified_error_bound'] and not receipt['physical_qualification']
    assert receipt['orders'][0]<=receipt['orders'][1]
    print(dict(orders=receipt['orders'].tolist(),changes=receipt['relative_change'].tolist()),flush=True)
    with pytest.raises(ValueError,match='did not converge'):
        kernel.adaptive_disjoint_linear_weak_layers(left[1:],right[1:],orders=(2,4),rtol=1e-12)
    with pytest.raises(ValueError,match='positive gap'):kernel.disjoint_linear_weak_layers(left,left)
    with pytest.raises(ValueError,match='budget'):kernel.disjoint_linear_weak_layers(left,right,order=8,max_points=1)
    with pytest.raises(ValueError,match='strictly increasing'):
        kernel.adaptive_disjoint_linear_weak_layers(left,right,orders=(8,4))


def test_adaptive_weak_rejects_overflowing_flux_metric_before_integration():
    a=torch.diag(torch.tensor([1e308,1.,1.],device='cuda',dtype=torch.float64))
    kernel=StaticTensorGreen(a)
    left,right=panels(a)
    # Finite SPD with representable Cholesky/determinant; the comparison
    # coefficient/length can still overflow and must not erase K' from the gate.
    with pytest.raises(ValueError,match='comparison flux scale'):
        kernel.adaptive_disjoint_linear_weak_layers(left,right)
