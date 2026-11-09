"""P2 weak blocks: independent point rules, singular convergence and P1 embedding."""
import pytest
import torch
from test_singular_convergence import pair
from test_static_tensor_green import coefficient
from test_static_tensor_disjoint_weak import panels
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def physical_basis(points,triangles):
    edges=torch.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('pij,pqi->pjq',edges,points-triangles[:,:1])).mT
    a,b=uv.unbind(-1);c=1-a-b
    return torch.stack((c*(2*c-1),a*(2*a-1),b*(2*b-1),4*c*a,4*a*b,4*b*c),-1)


def test_quadratic_disjoint_independent_product_and_native_derivative():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2
    kernel=StaticTensorGreen(a);left,right=panels(a)
    actual=kernel.disjoint_quadratic_weak_layers(left,right,order=12)
    x,wx=triangle_rule(left,16);y,wy=triangle_rule(right,16)
    lb,rb=physical_basis(x,left),physical_basis(y,right)
    value,gradient=kernel.evaluate((x[:,:,None]-y[:,None]).reshape(-1,3))
    weight=wx[:,:,None]*wy[:,None]
    normal=torch.linalg.cross(left[:,1]-left[:,0],left[:,2]-left[:,0]);normal=normal/normal.norm(dim=-1,keepdim=True)
    density=(gradient.reshape(1,len(x[0]),len(y[0]),3)@a.T*normal[:,None,None]).sum(-1)
    expected=(torch.einsum('pqr,pqa,prb->pab',weight*value.reshape_as(weight),lb,rb),
              torch.einsum('pqr,pqa,prb->pab',weight*density,lb,rb))
    for got,want in zip(actual,expected):torch.testing.assert_close(got,want,rtol=2e-8,atol=1e-13)
    own,=torch.autograd.grad(sum(v.square().sum() for v in actual),parameter,retain_graph=True)
    oracle,=torch.autograd.grad(sum(v.square().sum() for v in expected),parameter)
    torch.testing.assert_close(own,oracle,rtol=2e-7,atol=1e-14)
    reverse=kernel.disjoint_quadratic_weak_layers(right,left,order=12)[0]
    torch.testing.assert_close(reverse.mT,actual[0],rtol=2e-8,atol=1e-13)


@pytest.mark.parametrize('category',[0,1,2])
@torch.no_grad()
def test_quadratic_touching_embedding_convergence_and_reciprocity(category):
    a=coefficient();kernel=StaticTensorGreen(a);geometry=pair(category)
    # Values of the linear vertex bases at the six quadratic interpolation nodes.
    injection=a.new_tensor([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5]])
    low=kernel.touching_quadratic_weak_layers(*geometry,category,order=8)
    linear=kernel.touching_linear_weak_layers(*geometry,category,order=8)
    for quadratic,p1 in zip(low,linear):
        torch.testing.assert_close(injection.T@quadratic@injection,p1,rtol=1e-11,atol=1e-13)
    high=kernel.touching_quadratic_weak_layers(*geometry,category,order=16,max_nodes=400000)
    for got,want in zip(low,high):
        assert float((got-want).norm()/want.norm().clamp_min(1e-30))<2e-5
    left,right,li,ri=geometry
    reverse=kernel.touching_quadratic_weak_layers(right,left,ri,li,category,order=16,max_nodes=400000)[0]
    torch.testing.assert_close(reverse.mT,high[0],rtol=1e-8,atol=1e-12)
    if category==0:
        assert bool((torch.linalg.eigvalsh(high[0])>0).all())
        assert torch.count_nonzero(high[1])==0


@torch.no_grad()
def test_quadratic_touching_independent_analytic_source_integration():
    a=coefficient();kernel=StaticTensorGreen(a)
    left,right,li,ri=pair(1)
    reference=kernel.touching_quadratic_weak_layers(left,right,li,ri,1,order=16,max_nodes=400000)
    normal=torch.linalg.cross(left[:,1]-left[:,0],left[:,2]-left[:,0]);normal=normal/normal.norm(dim=-1,keepdim=True)
    errors=[]
    for order in (128,512):
        x,weights=triangle_rule(left,order);basis=physical_basis(x,left)[0]
        single=left.new_zeros(6,6);conormal=torch.zeros_like(single)
        for start in range(0,len(x[0]),4096):
            points=x[0,start:start+4096]
            v,g=kernel.panel_quadratic_integrals(points,right.expand(len(points),-1,-1))
            density=(g@a.T*normal[:,None]).sum(-1)
            lb=basis[start:start+4096];w=weights[0,start:start+4096]
            single+=torch.einsum('q,qa,qb->ab',w,lb,v)
            conormal+=torch.einsum('q,qa,qb->ab',w,lb,density)
        errors.append(max(float((got[None]-want).norm()/want.norm()) for got,want in zip((single,conormal),reference)))
    print(dict(quadratic_touching_independent_relative_errors=errors),flush=True)
    assert errors[-1]<errors[0] and errors[-1]<2e-5


def test_quadratic_adaptive_separations_and_rejections():
    a=coefficient();kernel=StaticTensorGreen(a)
    pairs=[panels(a,gap) for gap in (.2,.015)]
    left=torch.cat([p[0] for p in pairs]);right=torch.cat([p[1] for p in pairs])
    v,k,receipt=kernel.adaptive_disjoint_quadratic_weak_layers(left,right,rtol=1e-6)
    reference=kernel.disjoint_quadratic_weak_layers(left,right,order=48)
    for got,want in zip((v,k),reference):torch.testing.assert_close(got,want,rtol=3e-5,atol=1e-11)
    assert bool((receipt['relative_change']<=1e-6).all())
    assert not receipt['certified_error_bound'] and not receipt['physical_qualification']
    print(dict(quadratic_adaptive_orders=receipt['orders'].tolist(),changes=receipt['relative_change'].tolist()),flush=True)
    with pytest.raises(ValueError,match='did not converge'):
        kernel.adaptive_disjoint_quadratic_weak_layers(left[1:],right[1:],orders=(2,4),rtol=1e-12)
    with pytest.raises(ValueError,match='positive gap'):kernel.disjoint_quadratic_weak_layers(left,left)
    with pytest.raises(ValueError,match='budget'):kernel.touching_quadratic_weak_layers(*pair(1),1,max_nodes=1)
    empty=left[:0]
    assert kernel.disjoint_quadratic_weak_layers(empty,empty)[0].shape==(0,6,6)
