"""Three-dimensional distributional Bloch spectra, distinct from interior fields."""
import math
import runpy
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_spectral import bloch_spectral_fields
from witwin.maxwell.mom.operator import local_geometry

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_bloch_spectra_maxwell_equations_zero_mode_and_current_gradient():
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample=handle.system.bodies[0].mesh.vertices
    # Both original bodies must fit the centered cell.
    lattice=LatticeSpec(sample.new_tensor([[1.,.1,0.],[0.,1.1,.1],[.1,0.,1.2]]),sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,2.,reciprocal_shells=3)
    orders=torch.tensor([[0,0,0],[1,0,0],[-1,2,0],[1,1,-2]],device=sample.device)
    torch.manual_seed(23)
    coefficients=torch.randn(2*handle.system.total_edges,device=sample.device,dtype=torch.complex128,requires_grad=True)
    result=bloch_spectral_fields(handle,coefficients,kernel=kernel,orders=orders)
    p=result['wavevectors'].to(coefficients.dtype)
    e,h=result['electric'],result['magnetic']
    medium=handle.exterior
    torch.testing.assert_close(1j*torch.linalg.cross(p,e)-1j*medium.omega*medium.mu*h,
        -result['magnetic_current_density'],rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(1j*torch.linalg.cross(p,h)+1j*medium.omega*medium.epsilon*e,
        result['electric_current_density'],rtol=1e-11,atol=1e-12)
    moments=[torch.zeros(3,device=sample.device,dtype=coefficients.dtype) for _ in range(2)]
    for i,body in enumerate(handle.system.bodies):
        edges,free,slopes=local_geometry(body.basis)
        centroid=body.mesh.triangles.mean(1)
        first_moment=body.mesh.areas[:,None,None]*(centroid[:,None]-free)
        for kind,selection in enumerate((handle.system.J_slices[i],handle.system.M_slices[i])):
            moments[kind]=moments[kind]+(coefficients[selection][edges][:,:,None]*slopes[:,:,None]*first_moment).sum((0,1))
    for key,expected in zip(('electric_current_density','magnetic_current_density'),moments):
        torch.testing.assert_close(result[key][0],expected/lattice.cell_measure,rtol=1e-12,atol=1e-13)
    derivative,=torch.autograd.grad(e.real.sum(),coefficients)
    delta=torch.randn_like(coefficients)
    step=1e-5
    plus=bloch_spectral_fields(handle,coefficients.detach()+step*delta,kernel=kernel,orders=orders)
    minus=bloch_spectral_fields(handle,coefficients.detach()-step*delta,kernel=kernel,orders=orders)
    numerical=(plus['electric']-minus['electric']).real.sum()/(2*step)
    torch.testing.assert_close((derivative.conj()*delta).real.sum(),numerical,rtol=1e-8,atol=1e-8)
    assert not result['physical_body_interior_fields'] and not result['physical_qualification']
    assert 'outward_power_W' not in result


def test_bloch_spectra_reject_unrequested_resolvent_pole():
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample=handle.system.bodies[0].mesh.vertices
    period=2*math.pi/handle.exterior.k.real
    lattice=LatticeSpec(torch.eye(3,device=sample.device,dtype=sample.dtype)*period,sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,2.,reciprocal_shells=3)
    zero=torch.zeros(2*handle.system.total_edges,device=sample.device,dtype=torch.complex128)
    with pytest.raises(ValueError,match='spectral pole'):
        bloch_spectral_fields(handle,zero,kernel=kernel,orders=torch.zeros(1,3,device=sample.device,dtype=torch.int64))
