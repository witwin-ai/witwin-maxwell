"""Complex physical-panel integration, branch, distribution and gradient controls."""
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.complex_static_tensor_green import (
    ComplexStaticTensorGreen,ComplexStaticPanelConvergenceError)
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def fixture():
    a=torch.tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]],device='cuda',dtype=torch.float64)
    panel=a.new_tensor([[[0.,0.,0.],[1.,0.,0.],[.1,.8,0.]]])
    return a,panel


def medium(tensor,*,reciprocal=True):
    return TensorMedium(3e7,tensor,torch.eye(3,device=tensor.device,dtype=tensor.dtype),reciprocal=reciprocal)


@pytest.mark.parametrize('factor',[2+.3j,-2+.3j,-2+0j])
def test_complex_scalar_times_spd_branch_face_pv_and_near_panel(factor):
    a,panel=fixture();x=a.new_tensor([[.2,.3,1.1],[.25,.2,0.],[.25,.2,1e-6]])
    panels=panel.expand(3,-1,-1);pv=torch.tensor([False,True,False],device=a.device)
    reference=StaticTensorGreen(EPSILON_0*a).panel_integrals(x,panels,principal_value=pv)
    kernel=ComplexStaticTensorGreen(medium(factor*a),relative_tolerance=1e-9)
    value,gradient,receipt=kernel.integrate_panels(x,panels,principal_value=pv)
    assert receipt['converged'] and not receipt['error_certified'] and not receipt['physical_qualification']
    for actual,expected in zip((value,gradient),reference):
        torch.testing.assert_close(actual,expected.to(actual.dtype)/factor,rtol=2e-8,atol=.02)


def test_nondiagonal_complex_distribution_face_jump_rotation_and_gyrotropy():
    a,panel=fixture();loss=a.new_tensor([[.4,.03,-.02],[.03,.2,.01],[-.02,.01,.3]])
    tensor=a+1j*loss;kernel=ComplexStaticTensorGreen(medium(tensor),relative_tolerance=1e-9)
    vertices=a.new_tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    panels=vertices[torch.tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],device=a.device)]
    normals=torch.linalg.cross(panels[:,1]-panels[:,0],panels[:,2]-panels[:,0]);normals/=normals.norm(dim=-1,keepdim=True)
    points=a.new_tensor([[.2,.2,.2],[1.4,.3,.4],[.2,.3,0.]])
    pairs=panels.repeat(3,1,1);queries=points.repeat_interleave(4,dim=0)
    pv=torch.zeros(12,device=a.device,dtype=torch.bool);pv[-1]=True
    _,grad,receipt=kernel.integrate_panels(queries,pairs,principal_value=pv)
    hessian=-torch.einsum('qfi,fj->qij',grad.reshape(3,4,3),normals.to(tensor.dtype))
    contact=torch.einsum('ij,qij->q',EPSILON_0*tensor,hessian)
    torch.testing.assert_close(contact,tensor.new_tensor([-1.,0.,-.5]),rtol=0,atol=2e-9)
    torch.testing.assert_close(hessian,hessian.transpose(-1,-2),rtol=1e-8,atol=.1)
    face=panel.mean(1);normal=a.new_tensor([[0.,0.,1.]])
    centre=kernel.panel_integrals(face,panel,principal_value=torch.ones(1,device=a.device,dtype=torch.bool))[1]
    plus=kernel.panel_integrals(face+1e-6*normal,panel)[1]
    minus=kernel.panel_integrals(face-1e-6*normal,panel)[1]
    torch.testing.assert_close((plus+minus)/2,centre,rtol=1e-7,atol=20.)
    expected_jump=-normal.to(tensor.dtype)/tensor[2,2]
    coarse_error=(EPSILON_0*(plus-minus)-expected_jump).norm()
    # A finite offset includes a regular O(h) term: verify its decay before
    # comparing the closer one-sided difference with the exact trace jump.
    plus=kernel.panel_integrals(face+1e-7*normal,panel)[1]
    minus=kernel.panel_integrals(face-1e-7*normal,panel)[1]
    fine_error=(EPSILON_0*(plus-minus)-expected_jump).norm()
    assert fine_error<.2*coarse_error
    torch.testing.assert_close(EPSILON_0*(plus-minus),expected_jump,rtol=1e-5,atol=1e-7)
    q,_=torch.linalg.qr(a.new_tensor([[1.,2.,3.],[0.,1.,2.],[2.,1.,1.]]))
    qc=q.to(tensor.dtype);rotated=qc@tensor@qc.T;rotated=(rotated+rotated.T)/2
    x=a.new_tensor([[.2,.3,1.1]])
    base=kernel.panel_integrals(x,panel)
    other=ComplexStaticTensorGreen(medium(rotated),relative_tolerance=1e-9).panel_integrals(x@q.T,panel@q.T)
    torch.testing.assert_close(other[0],base[0],rtol=2e-9,atol=.01)
    torch.testing.assert_close(other[1],base[1]@qc.T,rtol=2e-9,atol=.01)
    bias=1j*a.new_tensor([[0.,.1,0.],[-.1,0.,.05],[0.,-.05,0.]])
    for sign in (-1,1):
        biased=ComplexStaticTensorGreen(medium(tensor+sign*bias,reciprocal=False),relative_tolerance=1e-9)
        for actual,expected in zip(biased.panel_integrals(x,panel),base):
            torch.testing.assert_close(actual,expected,rtol=2e-14,atol=.01)
    # A separated physical triangle rule is independent of the signed fan.
    nodes,weights=triangle_rule(panel,32)
    values,gradients=kernel.evaluate((x[:,None]-nodes).reshape(-1,3))
    torch.testing.assert_close(base[0],(values.reshape_as(weights)*weights).sum(1),rtol=2e-10,atol=.01)
    torch.testing.assert_close(base[1],(gradients.reshape(1,-1,3)*weights[:,:,None]).sum(1),rtol=2e-10,atol=.01)


def test_native_query_material_gradients_and_magnetic_units():
    a,panel=fixture();x=a.new_tensor([[.2,.3,1.1]],requires_grad=True)
    strength=a.new_tensor(.2,requires_grad=True)
    def evaluate(t,p):
        tensor=a+1j*(t*torch.eye(3,device=a.device,dtype=a.dtype))
        kernel=ComplexStaticTensorGreen(medium(tensor),relative_tolerance=1e-9)
        value,gradient=kernel.panel_integrals(p,panel)
        return EPSILON_0*value,EPSILON_0*gradient
    value,gradient=evaluate(strength,x)
    dx=torch.autograd.grad(value.real.sum(),x,retain_graph=True)[0]
    torch.testing.assert_close(dx,gradient.real,rtol=2e-8,atol=1e-11)
    dt=torch.autograd.grad(value.real.sum(),strength)[0];step=1e-5
    finite=(evaluate(strength.detach()+step,x.detach())[0].real-evaluate(strength.detach()-step,x.detach())[0].real)/(2*step)
    torch.testing.assert_close(dt,finite.sum(),rtol=2e-7,atol=1e-11)
    tensor=a+1j*.2*torch.eye(3,device=a.device,dtype=a.dtype)
    sample=TensorMedium(3e7,tensor,tensor)
    magnetic=ComplexStaticTensorGreen(sample,constitutive='magnetic').panel_integrals(x.detach(),panel)
    electric=ComplexStaticTensorGreen(sample).panel_integrals(x.detach(),panel)
    for h,e in zip(magnetic,electric):torch.testing.assert_close(MU_0*h,EPSILON_0*e,rtol=2e-12,atol=1e-12)


def test_ellipticity_pv_owner_budget_and_nonfinite_admission():
    a,panel=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    for diagonal in ((1.,-1.,1.),(1.,0.,1.)):
        with pytest.raises(NotImplementedError,match='definite real or strictly lossy'):
            ComplexStaticTensorGreen(medium(torch.diag(eye.new_tensor(diagonal))))
    # Strict loss admits indefinite real parts, without sampling for cones.
    lossy=torch.diag(eye.new_tensor([1.,-1.,1.]))+.5j*eye
    value,_=ComplexStaticTensorGreen(medium(lossy)).panel_integrals(a.new_tensor([[.2,.3,1.1]]),panel)
    assert bool(torch.isfinite(value).all())
    kernel=ComplexStaticTensorGreen(medium(a.to(eye.dtype)))
    with pytest.raises(ValueError,match='principal'):
        kernel.panel_integrals(panel.mean(1),panel)
    with pytest.raises(ValueError,match='edges'):
        kernel.panel_integrals(panel[:,0],panel,principal_value=torch.ones(1,device=a.device,dtype=torch.bool))
    with pytest.raises(ValueError,match='representable'):
        kernel.evaluate(a.new_tensor([[1e200,0.,0.]]))
    strict=ComplexStaticTensorGreen(medium(a.to(eye.dtype)),relative_tolerance=1e-15,initial_order=2,max_order=4)
    with pytest.raises(ComplexStaticPanelConvergenceError) as error:
        strict.panel_integrals(a.new_tensor([[.2,.3,.01]]),panel)
    assert not error.value.receipt['converged'] and error.value.receipt['failure_reason']=='panel_order_budget_exhausted'
    kernel.medium.epsilon_r.add_(eye)
    with pytest.raises(ValueError,match='changed'):kernel.panel_integrals(a.new_tensor([[.2,.3,1.1]]),panel)
