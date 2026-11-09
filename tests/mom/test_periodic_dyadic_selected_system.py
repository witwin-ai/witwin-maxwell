"""Complete selected tensor blocks against an independent scalar Ewald system."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
from witwin.maxwell.mom.operator import IntegrationConfig
import witwin.maxwell.mom.periodic_dyadic_operator as assembly

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_complete_selected_pairs_match_ewald_and_keep_interiors_local(monkeypatch):
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample=handle.system.bodies[0].mesh.vertices
    eye=torch.eye(3,device=sample.device,dtype=torch.complex128)
    scalar=compile_medium(mw.Material(sigma_e=.04,sigma_m=2000.),handle.exterior.frequency)
    medium=TensorMedium(scalar.frequency,scalar.epsilon_r*eye,scalar.mu_r*eye)
    entries=[]
    for i,body in enumerate(handle.system.bodies):
        interior=body.interior
        if i==0:interior=TensorMedium(interior.frequency,interior.epsilon_r*eye,interior.mu_r*eye)
        entries.append((body.structure_index,body.name,body.basis,interior))
    system=SurfaceSystem.create(entries)
    lattice=LatticeSpec(sample.new_tensor([[1.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    reference=LossyDyadicImageReference(lattice,medium,6)
    config=IntegrationConfig(order=4,singular_order=4)
    expected,_=assemble_periodic_system(handle.system,scalar,config=config,
        kernel=EwaldHelmholtzKernel(lattice,3.,real_shells=3,reciprocal_shells=8,quadrature_order=128))
    outside=DyadicGreenKernel(medium)
    interiors={system.bodies[0].region_id:DyadicGreenKernel(system.bodies[0].interior)}
    def forbidden(*args,**kwargs):raise AssertionError('Selected composition must not assemble a dense matrix')
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_system',forbidden)
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_correction',forbidden)
    re=torch.tensor([4,0],device=system.device);ce=torch.tensor([5,1],device=system.device)
    for i,j in ((0,0),(0,1),(1,0),(1,1)):
        value,receipt=assembly.periodic_dyadic_system_block(system,i,j,re,ce,reference=reference,
            exterior_kernel=outside,interior_kernels=interiors,config=config)
        rows=torch.cat((re+system.J_slices[i].start,re+system.M_slices[i].start))
        cols=torch.cat((ce+system.J_slices[j].start,ce+system.M_slices[j].start))
        target=expected[rows[:,None],cols]
        scale=sample.new_tensor([1.,1.,abs(scalar.eta),abs(scalar.eta)])
        error=((value-target)*scale[None,:]/scale[:,None]).norm()
        norm=(target*scale[None,:]/scale[:,None]).norm()
        assert error<1e-7*norm+1e-14
        assert (receipt['local_interior'] is not None)==(i==j)
        assert receipt['complete_operator'] and receipt['primary_included']
        assert not receipt['interiors_periodic'] and not receipt['physical_qualification']
    for kernels in ({},{1:outside},{1:interiors[1],2:outside}):
        with pytest.raises(ValueError,match='interior kernel'):
            assembly.periodic_dyadic_system_block(system,0,1,re,ce,reference=reference,
                exterior_kernel=outside,interior_kernels=kernels,config=config)
