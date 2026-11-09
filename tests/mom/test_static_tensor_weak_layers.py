"""Joint singular P1 static moments, separate from transmission qualification."""
import math
import pytest
import torch
from test_singular_convergence import pair
from test_static_tensor_green import coefficient
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_weak_self_closed_form_symmetry_and_positive_energy():
    a=torch.eye(3,device='cuda',dtype=torch.float64)
    kernel=StaticTensorGreen(a);geometry=pair(0)
    value,conormal=kernel.touching_linear_weak_layers(*geometry,0,order=12)
    exact=(2+math.sqrt(2))*math.asinh(1)/(12*math.pi)
    torch.testing.assert_close(value.sum(),a.new_tensor(exact),rtol=1e-9,atol=1e-11)
    torch.testing.assert_close(value,value.mT,rtol=1e-12,atol=1e-14)
    assert bool((torch.linalg.eigvalsh(value)>0).all())
    torch.testing.assert_close(conormal,torch.zeros_like(conormal),rtol=0,atol=0)
    left,right,li,ri=geometry
    reordered=kernel.touching_linear_weak_layers(left,right[:,[2,0,1]],li,ri[:,[2,0,1]],0,order=12)[0]
    torch.testing.assert_close(reordered,value[:,:,[2,0,1]],rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize('category',[1,2])
def test_weak_touching_order_convergence_reciprocity_and_scaling(category):
    a=coefficient();kernel=StaticTensorGreen(a);geometry=pair(category)
    reference=kernel.touching_linear_weak_layers(*geometry,category,order=16,max_nodes=400000)
    errors=[]
    for order in (4,8,12):
        actual=kernel.touching_linear_weak_layers(*geometry,category,order=order)
        errors.append(max(float((x-y).norm()/y.norm()) for x,y in zip(actual,reference)))
    print(dict(category=category,relative_errors=errors,physical_qualification=False),flush=True)
    assert errors[-1]<errors[1]<errors[0]
    assert errors[-1]<1e-4
    left,right,li,ri=geometry
    reverse=kernel.touching_linear_weak_layers(right,left,ri,li,category,order=16,max_nodes=400000)[0]
    torch.testing.assert_close(reverse.mT,reference[0],rtol=1e-8,atol=1e-11)
    scaled=StaticTensorGreen(17*a).touching_linear_weak_layers(*geometry,category,order=8)
    base=kernel.touching_linear_weak_layers(*geometry,category,order=8)
    torch.testing.assert_close(17*scaled[0],base[0],rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(scaled[1],base[1],rtol=1e-12,atol=1e-14)


def test_weak_native_spd_derivative_node_budget_and_topology():
    parameter=coefficient().requires_grad_();a=(parameter+parameter.T)/2
    geometry=pair(1);kernel=StaticTensorGreen(a)
    def objective(matrix):
        layers=StaticTensorGreen(matrix).touching_linear_weak_layers(*geometry,1,order=5)
        return sum(value.square().sum() for value in layers)
    result=objective(a);gradient,=torch.autograd.grad(result,parameter)
    direction=torch.ones_like(a)/10;step=1e-5
    expected=(objective(a.detach()+step*direction)-objective(a.detach()-step*direction))/(2*step)
    torch.testing.assert_close((gradient*direction).sum(),expected,rtol=1e-6,atol=1e-12)
    left,right,li,ri=geometry
    batched=kernel.touching_linear_weak_layers(left.expand(3,-1,-1),right.expand(3,-1,-1),li.expand(3,-1),ri.expand(3,-1),1,order=5,max_nodes=6*5**4)
    for got,want in zip(batched,kernel.touching_linear_weak_layers(*geometry,1,order=5)):
        torch.testing.assert_close(got,want.expand_as(got),rtol=1e-12,atol=1e-14)
    with pytest.raises(ValueError,match='budget'):kernel.touching_linear_weak_layers(*geometry,1,order=5,max_nodes=1)
    with pytest.raises(ValueError,match='topology'):kernel.touching_linear_weak_layers(*geometry,2)
    corrupt=right.clone();corrupt[:,0,0]+=.01
    with pytest.raises(ValueError,match='identical coordinates'):kernel.touching_linear_weak_layers(left,corrupt,li,ri,1)


@pytest.mark.parametrize('category',[1,2])
def test_weak_touching_matches_independent_analytic_source_integration(category):
    from witwin.maxwell.mom.quadrature import triangle_rule
    a=coefficient();kernel=StaticTensorGreen(a)
    left,right,li,ri=pair(category)
    reference=kernel.touching_linear_weak_layers(left,right,li,ri,category,order=16,max_nodes=400000)
    edge=torch.stack((left[:,1]-left[:,0],left[:,2]-left[:,0]),-1)
    cross=torch.linalg.cross(edge[:,:,0],edge[:,:,1]);normal=cross/cross.norm(dim=-1,keepdim=True)
    errors=[]
    for order in ((128,512) if category==1 else (32,64)):
        x,weight=triangle_rule(left,order)
        uv=torch.linalg.solve(edge.mT@edge,torch.einsum('pij,pqi->pjq',edge,x-left[:,:1])).mT
        basis=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)[0]
        observer=x[0].detach().requires_grad_()
        single=kernel.panel_linear_single_layer(observer,right.expand(len(observer),-1,-1))
        gradient=torch.stack([torch.autograd.grad(single[:,i].sum(),observer,retain_graph=True)[0]
                              for i in range(3)],1)
        conormal=(gradient@a.T*normal[:,None]).sum(-1)
        actual=(torch.einsum('q,qa,qb->ab',weight[0],basis,single)[None],
                torch.einsum('q,qa,qb->ab',weight[0],basis,conormal)[None])
        errors.append(max(float(((x-y).norm()/y.norm()).detach()) for x,y in zip(actual,reference)))
    print(dict(category=category,analytic_source_relative_errors=errors),flush=True)
    assert errors[-1]<errors[0]
    assert errors[-1]<2e-5


def test_edge_log_near_segment_and_projected_endpoint_derivatives():
    from witwin.maxwell.mom.fields import _static_edge_log
    point=coefficient().new_tensor([[.3,1e-12,0.],[0.,1e-7,0.],[1.,1e-7,0.],[-.2,1e-6,0.]],requires_grad=True)
    first=torch.zeros_like(point);last=first+point.new_tensor([1.,0.,0.])
    value=_static_edge_log(point,first,last)
    expected=torch.asinh((1-point[:,0])/point[:,1])-torch.asinh(-point[:,0]/point[:,1])
    torch.testing.assert_close(value,expected,rtol=1e-12,atol=1e-12)
    own,=torch.autograd.grad(value.sum(),point,retain_graph=True)
    oracle,=torch.autograd.grad(expected.sum(),point)
    torch.testing.assert_close(own,oracle,rtol=1e-9,atol=1e-8)
    collinear=point.new_tensor([[2.,0.,0.]],requires_grad=True)
    value=_static_edge_log(collinear,first[:1],last[:1])
    gradient,=torch.autograd.grad(value.sum(),collinear)
    torch.testing.assert_close(value,point.new_tensor([math.log(2.)]),rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(gradient,point.new_tensor([[-.5,0.,0.]]),rtol=1e-12,atol=1e-14)
