"""Regular exterior tensor matrix ownership and scalar independent comparison."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
from witwin.maxwell.mom.periodic_dyadic_operator import assemble_periodic_dyadic_correction
from witwin.maxwell.mom.periodic_operator import _assemble_regular

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_regular_tensor_matrix_matches_scalar_ewald_and_has_no_interior_ownership():
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample=handle.system.bodies[0].mesh.vertices
    eye=torch.eye(3,device=sample.device,dtype=torch.complex128)
    exterior=compile_medium(mw.Material(sigma_e=.04,sigma_m=2000.),handle.exterior.frequency)
    tensor=TensorMedium(exterior.frequency,exterior.epsilon_r*eye,exterior.mu_r*eye)
    lattice=LatticeSpec(sample.new_tensor([[1.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    reference=LossyDyadicImageReference(lattice,tensor,6)
    config=IntegrationConfig(order=4,singular_order=4)
    actual,receipt=assemble_periodic_dyadic_correction(handle.system,reference=reference,config=config)
    kernel=EwaldHelmholtzKernel(lattice,3.,real_shells=3,reciprocal_shells=8,quadrature_order=128)
    expected,_=_assemble_regular(handle.system,exterior,config,kernel)
    eta=abs(exterior.eta)
    scale=actual.real.new_tensor([1.]*handle.system.total_edges+[eta]*handle.system.total_edges)
    error=((actual-expected)*scale[None,:]/scale[:,None]).norm()
    magnitude=(expected*scale[None,:]/scale[:,None]).norm()
    print({'balanced_relative_matrix_difference':float(error/magnitude)})
    assert error<1e-5*magnitude+1e-14
    changed=SurfaceSystem.create([(body.structure_index,body.name,body.basis,
        compile_medium(mw.Material(eps_r=7+i,mu_r=2+i),exterior.frequency))
        for i,body in enumerate(handle.system.bodies)])
    other,_=assemble_periodic_dyadic_correction(changed,reference=reference,config=config)
    torch.testing.assert_close(actual,other,rtol=0,atol=0)
    assert all(actual[a,b].norm()>0 for a in handle.system.J_slices for b in handle.system.M_slices)
    assert not receipt['primary_included'] and not receipt['interiors_periodic']
    assert not receipt['complete_operator'] and not receipt['physical_qualification']


def test_actual_anisotropic_periodic_regular_matrix_is_finite_and_nonzero():
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=handle.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    sample=body.mesh.vertices
    eye=torch.eye(3,device=sample.device,dtype=torch.complex128)
    epsilon=eye.new_tensor([[2.,.2,.1],[.2,3.,.15],[.1,.15,4.]])+2j*eye
    mu=eye.new_tensor([[1.1,.1,0.],[.1,1.3,.05],[0.,.05,1.6]])+2j*eye
    medium=TensorMedium(handle.exterior.frequency,epsilon,mu)
    lattice=LatticeSpec(sample.new_tensor([[.8,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    quadrature=DyadicIntegrationConfig(64,16,256,64,relative_tolerance=1e-4,absolute_tolerance=1e-9)
    reference=LossyDyadicImageReference(lattice,medium,1,quadrature)
    matrix,receipt=assemble_periodic_dyadic_correction(system,reference=reference,
        config=IntegrationConfig(order=1,singular_order=1))
    assert matrix.shape==(2*system.total_edges,2*system.total_edges)
    assert torch.isfinite(matrix).all()
    slices=(system.J_slices[0],system.M_slices[0])
    assert all(matrix[a,b].norm()>0 for a in slices for b in slices)
    for block in receipt['blocks']:
        for images in block['image_evaluations']:
            assert images['primary_image_excluded']
            assert all(row['converged'] for row in images['spatial_integration_receipts'])
    assert not receipt['complete_operator'] and not receipt['surface_quadrature_converged']
    print({'actual_tensor_regular_matrix_norm':float(matrix.norm()),'physical_qualification':False})


def test_complete_periodic_tensor_scalar_limit_matches_independent_ewald_system():
    from witwin.maxwell.mom.periodic_dyadic_operator import assemble_periodic_dyadic_system
    from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = handle.system.bodies[0].mesh.vertices
    eye = torch.eye(3, device=sample.device, dtype=torch.complex128)
    exterior = compile_medium(mw.Material(sigma_e=.04, sigma_m=2000.), handle.exterior.frequency)
    tensor = TensorMedium(exterior.frequency, exterior.epsilon_r*eye, exterior.mu_r*eye)
    # Include a tensor interior in addition to the tensor exterior; other bodies
    # retain scalar interiors, so every regional dispatch participates.
    entries = []
    for i, body in enumerate(handle.system.bodies):
        interior = body.interior
        if i == 0:
            interior = TensorMedium(interior.frequency, interior.epsilon_r*eye, interior.mu_r*eye)
        entries.append((body.structure_index, body.name, body.basis, interior))
    mixed = SurfaceSystem.create(entries)
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.2, 0., 0.]))
    config = IntegrationConfig(order=4, singular_order=4)
    actual, receipt = assemble_periodic_dyadic_system(mixed,
        reference=LossyDyadicImageReference(lattice, tensor, 6), config=config)
    kernel = EwaldHelmholtzKernel(lattice, 3., real_shells=3, reciprocal_shells=8, quadrature_order=128)
    expected, _ = assemble_periodic_system(handle.system, exterior, config=config, kernel=kernel)
    eta = abs(exterior.eta)
    scale = actual.real.new_tensor([1.]*mixed.total_edges+[eta]*mixed.total_edges)
    error = ((actual-expected)*scale[None, :]/scale[:, None]).norm()
    norm = (expected*scale[None, :]/scale[:, None]).norm()
    assert error < 1e-7*norm+1e-14
    assert receipt['complete_operator'] and receipt['primary_included']
    assert not receipt['interiors_periodic'] and not receipt['physical_qualification']
    print({'complete_matrix_balanced_relative_difference': float(error/norm)})
