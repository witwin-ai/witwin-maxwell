"""Actual tensor Maxwell Bloch response, not scalar composition."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0,MU_0
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def problem(parameter=None):
    a=torch.tensor([[.8,.1,0.],[0.,.9,.1],[.1,0.,1.]],device='cuda',dtype=torch.float64)
    lattice=LatticeSpec(a,a.new_tensor([.12,.08,.04]))
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    eps=eye.new_tensor([[2.,.2,.1],[.2,3.,.15],[.1,.15,4.]])+.05j*eye
    if parameter is not None:
        eps=eps+parameter*eye
    mu=eye.new_tensor([[1.1,.1,0.],[.1,1.3,.05],[0.,.05,1.6]])+.03j*eye
    return QuasiperiodicDyadicGreenKernel(lattice,TensorMedium(3e8,eps,mu))


def test_tensor_periodic_spectrum_maxwell_residual_and_material_gradient():
    parameter=torch.tensor(.1,device='cuda',dtype=torch.float64,requires_grad=True)
    kernel=problem(parameter)
    orders=torch.tensor([[0,0,0],[1,0,0],[-1,1,2]],device='cuda')
    modes=kernel.spectral_modes(orders)
    torch.manual_seed(17)
    source=torch.randn(3,6,device='cuda',dtype=torch.complex128)
    fields=torch.einsum('nij,nj->ni',modes['current_to_field'],source)
    e,h=fields[:,:3],fields[:,3:]
    q=modes['wavevectors'].to(fields.dtype)
    medium=kernel.medium
    torch.testing.assert_close(1j*torch.linalg.cross(q,e)-1j*medium.omega*(h@medium.mu.T),
        -source[:,3:]/kernel.lattice.cell_measure,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(1j*torch.linalg.cross(q,h)+1j*medium.omega*(e@medium.epsilon.T),
        source[:,:3]/kernel.lattice.cell_measure,rtol=1e-11,atol=1e-11)
    gradient,=torch.autograd.grad(fields.real.sum(),parameter)
    def objective(value):
        return torch.einsum('nij,nj->ni',problem(value).spectral_modes(orders)['current_to_field'],source).real.sum()
    numerical=(objective(parameter.detach()+1e-5)-objective(parameter.detach()-1e-5))/(2e-5)
    torch.testing.assert_close(gradient,numerical,rtol=1e-7,atol=1e-7)
    assert not modes['physical_qualification'] and not modes['global_resolvent_verified']


def test_heat_filtered_tensor_lattice_bloch_reciprocity_and_point_gradient():
    kernel=problem()
    sample=kernel.lattice.vectors
    points=sample.new_tensor([[.13,.17,.21]],requires_grad=True)
    value,receipt=kernel.evaluate_heat(points,shells=2,heat_time=.05)
    shifted,_=kernel.evaluate_heat(points+sample[0],shells=2,heat_time=.05)
    torch.testing.assert_close(shifted,value*torch.exp(1j*(sample[0]@kernel.lattice.bloch)),rtol=1e-11,atol=1e-11)
    opposite=QuasiperiodicDyadicGreenKernel(LatticeSpec(sample,-kernel.lattice.bloch),kernel.medium)
    reciprocal,_=opposite.evaluate_heat(-points,shells=2,heat_time=.05)
    signs=sample.new_tensor([1,1,1,-1,-1,-1])
    torch.testing.assert_close(value,signs[:,None]*reciprocal.transpose(-1,-2)*signs[None,:],rtol=1e-11,atol=1e-11)
    gradient,=torch.autograd.grad(value.real.sum(),points)
    step=1e-5
    shift=sample.new_tensor([[step,0,0]])
    plus,_=kernel.evaluate_heat(points.detach()+shift,shells=2,heat_time=.05)
    minus,_=kernel.evaluate_heat(points.detach()-shift,shells=2,heat_time=.05)
    torch.testing.assert_close(gradient[0,0],(plus-minus).real.sum()/(2*step),rtol=1e-7,atol=1e-7)
    assert receipt['quantity']=='finite_heat_filtered_full_distribution'
    assert not receipt['primary_singularity_separated'] and not receipt['physical_qualification']


def test_tensor_periodic_requested_pole_and_owned_material_rejection():
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    frequency=3e8
    k0=2*math.pi*frequency*math.sqrt(EPSILON_0*MU_0)
    vectors=eye.real*(2*math.pi/k0)
    lattice=LatticeSpec(vectors,vectors.new_zeros(3))
    kernel=QuasiperiodicDyadicGreenKernel(lattice,TensorMedium(frequency,eye,eye))
    orders=torch.tensor([[1,0,0]],device='cuda')
    with pytest.raises(ValueError,match='spectral pole'):
        kernel.spectral_modes(orders)
    kernel.medium.epsilon_r[0,0]=2
    with pytest.raises(ValueError,match='changed'):
        kernel.spectral_modes(orders)


def test_tensor_modal_adjoint_matches_inner_product_and_native_source_gradient():
    kernel=problem()
    orders=torch.tensor([[0,0,0],[1,-1,0]],device='cuda')
    torch.manual_seed(28)
    source=torch.randn(2,6,device='cuda',dtype=torch.complex128,requires_grad=True)
    covector=torch.randn_like(source)
    field=kernel.apply_modes(orders,source)
    adjoint=kernel.apply_modes(orders,covector,adjoint=True)
    torch.testing.assert_close(torch.vdot(covector.flatten(),field.flatten()),
        torch.vdot(adjoint.flatten(),source.flatten()),rtol=1e-12,atol=1e-12)
    gradient,=torch.autograd.grad(torch.vdot(covector.flatten(),field.flatten()).real,source)
    torch.testing.assert_close(gradient,adjoint,rtol=1e-12,atol=1e-12)


def test_open_tensor_kernel_scalar_ewald_hessian_reference():
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    from witwin.maxwell.mom.dyadic_green import cross_matrix
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    vectors=eye.real.new_tensor([[.7,0,0],[0,.8,0]])
    lattice=LatticeSpec(vectors,vectors.new_tensor([.2,.1,0.]))
    medium=TensorMedium(3e8,(1+.2j)*eye,(1+.1j)*eye)
    kernel=QuasiperiodicDyadicGreenKernel(lattice,medium)
    points=vectors.new_tensor([[.13,.17,.4],[.1,-.2,-.45]],requires_grad=True)
    actual,_=kernel.evaluate_open(points,shells=8)
    scalar=EwaldHelmholtzKernel(lattice,3.5,real_shells=3,reciprocal_shells=8,quadrature_order=128)
    k=medium.omega*math.sqrt(EPSILON_0*MU_0)*complex(1+.2j)**.5*complex(1+.1j)**.5
    green,gradient=scalar.evaluate(points,k)
    rows=[]
    for axis in range(3):
        real,=torch.autograd.grad(gradient[:,axis].real.sum(),points,retain_graph=True)
        imag,=torch.autograd.grad(gradient[:,axis].imag.sum(),points,retain_graph=True)
        rows.append(real+1j*imag)
    hessian=torch.stack(rows,1)
    potential=green[:,None,None]*eye+hessian/k**2
    curl=cross_matrix(gradient)
    expected=torch.cat((torch.cat((1j*medium.omega*MU_0*(1+.1j)*potential,-curl),-1),
        torch.cat((curl,1j*medium.omega*EPSILON_0*(1+.2j)*potential),-1)),-2)
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-7)


def test_open_anisotropic_kernel_rotated_frame_maxwell_and_bloch():
    base=problem()
    vectors=base.lattice.vectors[:2]
    beta=.1*vectors.sum(0)
    lattice=LatticeSpec(vectors,beta)
    kernel=QuasiperiodicDyadicGreenKernel(lattice,base.medium)
    normal=torch.linalg.cross(vectors[0],vectors[1])/lattice.cell_measure
    points=(.5*normal+.1*vectors[0])[None].requires_grad_()
    value,receipt=kernel.evaluate_open(points,shells=3)
    def curl(field):
        rows=[]
        for axis in range(3):
            real,=torch.autograd.grad(field[:,axis].real.sum(),points,retain_graph=True)
            imag,=torch.autograd.grad(field[:,axis].imag.sum(),points,retain_graph=True)
            rows.append(real+1j*imag)
        derivative=torch.stack(rows,1)
        return torch.stack((derivative[:,2,1]-derivative[:,1,2],
            derivative[:,0,2]-derivative[:,2,0],derivative[:,1,0]-derivative[:,0,1]),-1)
    for column in (0,3):
        e,h=value[:,:3,column],value[:,3:,column]
        torch.testing.assert_close(curl(e),1j*base.medium.omega*(h@base.medium.mu.T),rtol=1e-8,atol=1e-8)
        torch.testing.assert_close(curl(h),-1j*base.medium.omega*(e@base.medium.epsilon.T),rtol=1e-8,atol=1e-8)
    shifted,_=kernel.evaluate_open(points.detach()+vectors[0],shells=3)
    torch.testing.assert_close(shifted,value*torch.exp(1j*(vectors[0]@beta)),rtol=1e-10,atol=1e-10)
    assert not receipt['physical_qualification'] and not receipt['primary_singularity_separated']
    with pytest.raises(ValueError,match='source-plane'):
        kernel.evaluate_open(vectors[:1]*.1,shells=3)


def test_open_anisotropic_kernel_material_gradient_and_opposite_side_reciprocity():
    parameter=torch.tensor(.1,device='cuda',dtype=torch.float64,requires_grad=True)
    base=problem(parameter)
    vectors=base.lattice.vectors[:2]
    lattice=LatticeSpec(vectors,.1*vectors.sum(0))
    kernel=QuasiperiodicDyadicGreenKernel(lattice,base.medium)
    normal=torch.linalg.cross(vectors[0],vectors[1])/lattice.cell_measure
    points=(.5*normal+.1*vectors[0])[None]
    value,_=kernel.evaluate_open(points,shells=2)
    derivative,=torch.autograd.grad(value.real.sum(),parameter)
    def objective(p):
        medium=problem(p).medium
        result,_=QuasiperiodicDyadicGreenKernel(lattice,medium).evaluate_open(points,shells=2)
        return result.real.sum()
    step=1e-5
    numerical=(objective(parameter.detach()+step)-objective(parameter.detach()-step))/(2*step)
    torch.testing.assert_close(derivative,numerical,rtol=1e-6,atol=1e-6)
    opposite=QuasiperiodicDyadicGreenKernel(LatticeSpec(vectors,-lattice.bloch),base.medium)
    reciprocal,_=opposite.evaluate_open(-points,shells=2)
    signs=points.new_tensor([1,1,1,-1,-1,-1])
    torch.testing.assert_close(value,signs[:,None]*reciprocal.transpose(-1,-2)*signs[None,:],rtol=1e-9,atol=1e-9)


def test_lossy_dyadic_nonzero_images_match_regular_ewald_at_origin():
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
    from witwin.maxwell.mom.dyadic_green import cross_matrix
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    vectors=eye.real.new_tensor([[.7,0,0],[0,.8,0]])
    lattice=LatticeSpec(vectors,vectors.new_tensor([.2,.1,0.]))
    medium=TensorMedium(3e8,(1+2j)*eye,(1+2j)*eye)
    points=vectors.new_tensor([[0.,0.,0.],[.1,.12,0.]],requires_grad=True)
    actual,receipt=LossyDyadicImageReference(lattice,medium,3).evaluate(points,exclude_primary=True)
    k=medium.omega*math.sqrt(EPSILON_0*MU_0)*(1+2j)
    kernel=EwaldHelmholtzKernel(lattice,3.5,real_shells=3,reciprocal_shells=8,quadrature_order=128)
    green,gradient=kernel.regular_part(points,k)
    hessian=[]
    for axis in range(3):
        real,=torch.autograd.grad(gradient[:,axis].real.sum(),points,retain_graph=True)
        imag,=torch.autograd.grad(gradient[:,axis].imag.sum(),points,retain_graph=True)
        hessian.append(real+1j*imag)
    potential=green[:,None,None]*eye+torch.stack(hessian,1)/k**2
    curl=cross_matrix(gradient)
    expected=torch.cat((torch.cat((1j*medium.omega*MU_0*(1+2j)*potential,-curl),-1),
        torch.cat((curl,1j*medium.omega*EPSILON_0*(1+2j)*potential),-1)),-2)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-9)
    assert receipt['primary_image_excluded'] and not receipt['image_cutoff_converged']
    with pytest.raises(ValueError,match='retained dyadic image'):
        LossyDyadicImageReference(lattice,medium,1).evaluate(4*vectors[:1],exclude_primary=True)


def test_anisotropic_nonzero_image_reference_is_finite_on_primary_source():
    from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    base=problem()
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    medium=TensorMedium(base.medium.frequency,base.medium.epsilon_r+2j*eye,base.medium.mu_r+2j*eye)
    vectors=base.lattice.vectors[:1]
    lattice=LatticeSpec(vectors,.2*vectors[0])
    config=DyadicIntegrationConfig(64,16,128,32,relative_tolerance=1e-4,absolute_tolerance=1e-9)
    reference=LossyDyadicImageReference(lattice,medium,1,config)
    origin=vectors.new_zeros(1,3)
    value,receipt=reference.evaluate(origin,exclude_primary=True)
    assert torch.isfinite(value).all() and value.abs().max()>0
    assert all(row['converged'] for row in receipt['spatial_integration_receipts'])
    opposite=LossyDyadicImageReference(LatticeSpec(vectors,-lattice.bloch),medium,1,config)
    reverse,_=opposite.evaluate(origin,exclude_primary=True)
    signs=origin.new_tensor([1,1,1,-1,-1,-1])
    torch.testing.assert_close(value,signs[:,None]*reverse.transpose(-1,-2)*signs[None,:],rtol=2e-4,atol=1e-9)
    print({'anisotropic_regular_origin_max':float(value.abs().max()),'physical_qualification':False})


def test_dyadic_image_reference_rejects_lossless_limits_and_primary_singularities():
    from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
    base=problem()
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    lattice=LatticeSpec(base.lattice.vectors[:1],base.lattice.vectors.new_zeros(3))
    for eps,mu in ((eye,(1+1j)*eye),((1+1j)*eye,eye)):
        with pytest.raises(ValueError,match='strictly positive'):
            LossyDyadicImageReference(lattice,TensorMedium(3e8,eps,mu),1)
    reference=LossyDyadicImageReference(lattice,base.medium,1)
    with pytest.raises(ValueError,match='singular integration'):
        reference.evaluate(lattice.vectors.new_zeros(1,3))


def test_open_tensor_configured_pole_tolerance_rejects_near_grazing():
    from dataclasses import replace
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    frequency=1/(2*math.pi*math.sqrt(EPSILON_0*MU_0))
    vectors=eye.real.new_tensor([[2*math.pi/(1-1e-8),0,0],[0,1.,0]])
    lattice=LatticeSpec(vectors,vectors.new_zeros(3))
    kernel=QuasiperiodicDyadicGreenKernel(lattice,TensorMedium(frequency,eye,eye))
    points=vectors.new_tensor([[.1,.2,1.]])
    value,receipt=kernel.evaluate_open(points,shells=1)
    assert torch.isfinite(value).all() and not receipt['physical_qualification']
    with pytest.raises(ValueError,match='configured pole tolerance'):
        replace(kernel,pole_relative_tolerance=.01).evaluate_open(points,shells=1)


def test_open_tensor_cutoff_estimate_rejects_missing_near_plane_modes():
    from witwin.maxwell.mom.periodic_dyadic import PeriodicDyadicConvergenceError
    base=problem()
    vectors=base.lattice.vectors[:2]
    lattice=LatticeSpec(vectors,.1*vectors.sum(0))
    kernel=QuasiperiodicDyadicGreenKernel(lattice,base.medium)
    normal=torch.linalg.cross(vectors[0],vectors[1])/lattice.cell_measure
    far=(1.2*normal+.1*vectors[0])[None]
    value,receipt=kernel.estimate_open(far,shells=4,relative_tolerance=1e-5,absolute_tolerance=1e-9)
    assert receipt['cutoff_converged'] and not receipt['certified_error_bound']
    assert torch.isfinite(value).all()
    near=(.03*normal+.1*vectors[0])[None]
    with pytest.raises(PeriodicDyadicConvergenceError) as failure:
        kernel.estimate_open(near,shells=1,relative_tolerance=1e-5,absolute_tolerance=1e-9)
    assert not failure.value.receipt['cutoff_check']['passed']
    print({'open_near_cutoff_failure':failure.value.receipt['cutoff_check']})


def test_lossy_regular_image_tail_reuses_inner_images_and_keeps_failure(monkeypatch):
    from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference,PeriodicDyadicConvergenceError
    from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    vectors=eye.real.new_tensor([[.8,0,0]])
    lattice=LatticeSpec(vectors,vectors.new_tensor([.2,0,0]))
    medium=TensorMedium(3e8,(1+2j)*eye,(1+2j)*eye)
    reference=LossyDyadicImageReference(lattice,medium,1)
    points=vectors.new_tensor([[0.,0.,0.],[.1,.1,0.]])
    evaluated=[]
    original=DyadicGreenKernel.evaluate
    def observe(self,queries):
        evaluated.append(len(queries))
        return original(self,queries)
    monkeypatch.setattr(DyadicGreenKernel,'evaluate',observe)
    _,receipt=reference.estimate_regular(points,relative_tolerance=1e-4,absolute_tolerance=1e-9)
    assert sum(evaluated)==4*len(points)
    assert receipt['inner_images_reused'] and receipt['image_cutoff_converged']
    weak=TensorMedium(3e8,(1+.1j)*eye,(1+.1j)*eye)
    with pytest.raises(PeriodicDyadicConvergenceError) as failure:
        LossyDyadicImageReference(lattice,weak,1).estimate_regular(points,
            relative_tolerance=1e-4,absolute_tolerance=1e-9)
    assert not failure.value.receipt['image_cutoff_converged']
    print({'lossy_image_cutoff_failure':failure.value.receipt['cutoff_check']})
