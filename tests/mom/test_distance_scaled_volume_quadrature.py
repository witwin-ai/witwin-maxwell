"""Distance-scaled signed fans and selected scalar-limit tensor volume controls."""
import pytest
import torch
from witwin.maxwell.mom.quadrature import distance_scaled_duffy_fan,duffy_fan

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_signed_measure_covariance_and_observer_derivative():
    triangles=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],device='cuda',dtype=torch.float64)
    obs=triangles.new_tensor([[.2,.3,1e-6],[1.2,-.2,.01],[.2,.3,0.]])
    nodes,weights=distance_scaled_duffy_fan(obs,triangles.expand(3,-1,-1),24)
    torch.testing.assert_close(weights.sum(-1),weights.new_full((3,),.5),rtol=1e-10,atol=1e-12)
    torch.testing.assert_close((nodes*weights[:,:,None]).sum(1),triangles.mean(1).expand(3,-1)*.5,rtol=1e-10,atol=1e-12)
    n,w=duffy_fan(obs[2:],triangles,24)
    torch.testing.assert_close(nodes[2:],n,rtol=2e-13,atol=1e-14)
    torch.testing.assert_close(weights[2:],w,rtol=2e-13,atol=1e-14)
    x=triangles.new_tensor([[.2,.3,.05]])
    nodes,weights=distance_scaled_duffy_fan(x,triangles,8)
    n,w=distance_scaled_duffy_fan(x*x.new_tensor([1.,1.,-1.]),triangles,8)
    torch.testing.assert_close(n,nodes,rtol=2e-13,atol=1e-13)
    torch.testing.assert_close(w,weights,rtol=2e-13,atol=1e-13)
    q=torch.linalg.qr(x.new_tensor([[1.,2.,3.],[3.,1.,2.],[2.,3.,1.]]))[0]
    shift=x.new_tensor([.7,-.3,.2])
    for scale in (1e-3,2.7,1e3):
        n,w=distance_scaled_duffy_fan(scale*x@q.T+shift,scale*triangles@q.T+shift,8)
        torch.testing.assert_close(n,scale*nodes@q.T+shift,rtol=2e-11,atol=2e-12)
        torch.testing.assert_close(w,scale**2*weights,rtol=2e-10,atol=1e-15)
    for scale in (1e100,1e-100):
        with pytest.raises(ValueError,match='finite positive face area'):
            distance_scaled_duffy_fan(scale*x,scale*triangles,8)
    x.requires_grad_()
    def integral(p):
        n,w=distance_scaled_duffy_fan(p,triangles,12)
        return (torch.exp(-(n-p[:,None]).square().sum(-1))*w).sum()
    gradient,=torch.autograd.grad(integral(x),x)
    finite=[]
    for axis in range(3):
        delta=torch.zeros_like(x);delta[0,axis]=1e-5
        finite.append((integral(x.detach()+delta)-integral(x.detach()-delta))/2e-5)
    torch.testing.assert_close(gradient,torch.stack(finite)[None],rtol=2e-7,atol=1e-9)


def test_selected_near_face_volume_reference_and_receipt():
    from test_static_tensor_volume import fixture
    from witwin.maxwell.mom.tensor_volume import cell_tensor_current_blocks
    from witwin.maxwell.mom.tensor_medium import TensorMedium,scalar_medium_if_isotropic
    from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
    from witwin.maxwell.mom.vie_operator import cell_current_field_blocks
    mesh,a=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    medium=TensorMedium(3e7,2*eye,1.3*eye);kernel=DyadicGreenKernel(medium)
    points=a.new_tensor([[.07,.11,0.],[.07,.11,1e-6],[1.3,-.2,.1]])
    e,h=cell_current_field_blocks(mesh,points,scalar_medium_if_isotropic(medium),order=40,principal_value=True)
    reference=(e.permute(0,2,1,3),h.permute(0,2,1,3))
    errors=[]
    for order in (6,12):
        value,receipt=cell_tensor_current_blocks(mesh,points,kernel,order=order,radial_order=6,
            principal_value=True,face_rule='distance_scaled_duffy',kernel_batch=8192)
        actual=(value[:,:,:3,:3],value[:,:,3:,:3])
        errors.append([((v-r).flatten(1).norm(dim=-1)/r.flatten(1).norm(dim=-1)).tolist()
                       for v,r in zip(actual,reference)])
        assert receipt['face_rule']=='distance_scaled_duffy'
        assert not receipt['volume_quadrature_converged'] and not receipt['physical_qualification']
    assert max(errors[0][1])<.005
    assert max(errors[1][1])<1e-6
    assert max(errors[1][0])<.005
    print({'distance_scaled_selected_scalar_errors':errors,'physical_qualification':False})


def test_distance_scaled_adaptive_source_failure_retains_policy():
    from contextlib import nullcontext
    from test_tensor_volume_source import setup
    from witwin.maxwell.mom.riesz import LocalRWGRiesz
    from witwin.maxwell.mom.volume_response_matrix import adaptive_volume_source_matrix,VolumeMatrixConvergenceError
    op,binding=setup()
    metric=LocalRWGRiesz(op.system,abs(op.exterior.eta))
    op.read_access=nullcontext
    op.riesz_inverse=metric.inverse
    op.refresh=op.validate_tensor_state
    with pytest.raises(VolumeMatrixConvergenceError) as info:
        adaptive_volume_source_matrix(op,(binding,),order=2,inner_order=2,max_order=3,max_inner_order=3,
            relative_tolerance=1e-13,absolute_tolerance=1e-20,tensor_face_rule='distance_scaled_duffy')
    receipt=info.value.receipt
    assert not receipt['converged'] and receipt['tensor_face_rule']=='distance_scaled_duffy'
    assert all(row['receipt']['face_rule']=='distance_scaled_duffy' for row in
               receipt['matrix']['bodies'][0]['tensor_source_block']['tensor_volume_evaluations'])
