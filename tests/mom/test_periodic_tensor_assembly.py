"""Periodic scalar exterior ownership with body-local tensor interiors."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
from witwin.maxwell.mom.anisotropic_operator import assemble_generalized_system
from test_generalized_system_blocks import basis_at

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dimension',(1,2,3))
def test_periodic_tensor_scalar_limit_and_interior_ownership(dimension,monkeypatch):
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    scalar=compile_medium(mw.Material(eps_r=2.,mu_r=1.3),3e8)
    tensor=TensorMedium(3e8,2.*eye,1.3*eye)
    exterior=compile_medium(mw.Material(sigma_e=.02),3e8)
    bases=(basis_at(0.),basis_at(.5))
    def system(first):
        return SurfaceSystem.create([(0,'a',bases[0],first),(2,'b',bases[1],scalar)])
    sample=bases[0].mesh.vertices
    lattice=LatticeSpec(2.*torch.eye(3,device=sample.device,dtype=sample.dtype)[:dimension],
                        sample.new_tensor([.2,0.,0.]))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=4,quadrature_order=48)
    config=IntegrationConfig(order=2,singular_order=3)
    reference_system=system(scalar)
    reference,_=assemble_periodic_system(reference_system,exterior,config,kernel=kernel)
    free_scalar,_=assemble_system(reference_system,exterior,config)
    mixed=system(tensor)
    actual,receipt=assemble_periodic_system(mixed,exterior,config,kernel=kernel,
                                           interior_kernels={1:DyadicGreenKernel(tensor)})
    torch.testing.assert_close(actual,reference,rtol=2e-12,atol=1e-13)
    assert not receipt['interiors_periodic'] and not receipt['physical_qualification']
    assert len(receipt['regular_blocks'])==4

    # Manufactured regular dyadic values isolate routing, not physical accuracy.
    def regular(bound,points):
        value=torch.arange(36,device=sample.device,dtype=sample.dtype).reshape(6,6).to(eye.dtype)/1000
        value=(value+1j*value.mT)*bound.medium.epsilon_r.trace()
        return value.expand(len(points),6,6),dict(converged=True,
            static_diagonal_restored=False,static_diagonal_singularity_subtracted=True)
    monkeypatch.setattr(DyadicGreenKernel,'evaluate_regular',regular)
    changed=system(TensorMedium(3e8,torch.diag(eye.new_tensor([2.,2.4,2.8]))+.1j*eye,
                               torch.diag(eye.new_tensor([1.3,1.6,1.9]))+.1j*eye))
    free_tensor,_=assemble_generalized_system(changed,exterior,config)
    periodic,_=assemble_periodic_system(changed,exterior,config,kernel=kernel)
    torch.testing.assert_close(periodic-free_tensor,reference-free_scalar,rtol=1e-9,atol=1e-13)
    for rows in (mixed.J_slices[0],mixed.M_slices[0]):
        for cols in (mixed.J_slices[1],mixed.M_slices[1]):
            torch.testing.assert_close(periodic[rows,cols],reference[rows,cols],rtol=0,atol=0)


def test_periodic_tensor_kernel_failure_propagates(monkeypatch):
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    tensor=TensorMedium(3e8,torch.diag(eye.new_tensor([2.,3.,4.])),eye)
    system=SurfaceSystem.create([(0,'tensor',basis_at(0.),tensor)])
    sample=system.bodies[0].mesh.vertices
    kernel=EwaldHelmholtzKernel(LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_zeros(3)),2.)
    class KernelFailure(RuntimeError):
        pass
    def failed(*args,**kwargs):
        raise KernelFailure('Regional dyadic evaluation failed')
    monkeypatch.setattr(DyadicGreenKernel,'evaluate_regular',failed)
    with pytest.raises(KernelFailure):
        assemble_periodic_system(system,compile_medium(mw.Material(),3e8),
            IntegrationConfig(order=2,singular_order=3),kernel=kernel)
