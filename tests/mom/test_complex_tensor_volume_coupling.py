"""Actual lossy tensor AC volume blocks and weak-source scalar controls."""
import pytest
import torch
import witwin.maxwell as mw
from test_static_tensor_volume import fixture
from witwin.maxwell.mom.tensor_medium import TensorMedium,scalar_medium_if_isotropic
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel,DyadicIntegrationConfig
from witwin.maxwell.mom.tensor_volume import cell_tensor_current_blocks
from witwin.maxwell.mom.vie_operator import cell_current_field_blocks
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh,bind_volume_region
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.runtime import SystemDecodeHandle,MoMConfig
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_block

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def sample(a):
    eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    return TensorMedium(3e7,(2+.1j)*eye,(1.3+.03j)*eye)


def check_static_records(record):
    for kind,item in zip(('electric','magnetic'),record['static_cell_integrations']):
        assert item['constitutive']==kind
        r=item['receipt'];assert r['converged'] and not r['physical_qualification']
        assert r['integration_policy']==dict(version='complex_ac_static_cell_v1',
            relative_tolerance=1e-8,absolute_tolerance=0.,initial_order=8,max_order=128,pair_batch=8)
        assert r['estimator']=='successive_order_per_summed_cell_potential_gradient_hessian'


def test_lossy_tensor_electric_magnetic_volume_blocks_against_scalar():
    mesh,a=fixture();medium=sample(a);scalar=scalar_medium_if_isotropic(medium)
    points=a.new_tensor([[.2,.2,.2],[1.5,.7,1.1],[.07,.11,0.]])
    actual,receipt=cell_tensor_current_blocks(mesh,points,DyadicGreenKernel(medium),
        order=12,radial_order=8,principal_value=True,face_rule='projected_duffy',kernel_batch=8192)
    e,h=cell_current_field_blocks(mesh,points,scalar,order=40,principal_value=True)
    e,h=e.permute(0,2,1,3),h.permute(0,2,1,3)
    errors=[]
    for observed,expected in ((actual[:,:,:3,:3],e),(actual[:,:,3:,:3],h),
            (actual[:,:,:3,3:],-h),(actual[:,:,3:,3:],e*(scalar.epsilon/scalar.mu))):
        errors.append(float((observed-expected).norm()/expected.norm()))
    print({'lossy_tensor_volume_scalar_limit_errors':errors,'physical_qualification':False},flush=True)
    assert max(errors)<.005
    check_static_records(receipt)
    assert receipt['contact_distribution_included'] and not receipt['volume_quadrature_converged']


def test_lossy_tensor_weak_source_against_scalar_and_static_receipts():
    original,a=fixture()
    mesh=TetrahedralVolumeMesh(.07*original.vertices,original.cells,region_id=1)
    tensor=sample(a);scalar=scalar_medium_if_isotropic(tensor)
    def operator(inside):
        system=SurfaceSystem.create([(0,'volume',RWGBasis(mesh.boundary),inside)])
        config=MoMConfig(dyadic_integration=DyadicIntegrationConfig() if isinstance(inside,TensorMedium) else None)
        result=SystemDecodeHandle(system,compile_medium(mw.Material(),3e7),(),config,{})
        result.device=a.device;result.dtype=torch.complex128
        result.shape=(2*system.bodies[0].basis.n_edges,)*2
        return result,bind_volume_region(mesh,system)
    ids=mesh.cells.new_tensor([0]);op,binding=operator(tensor);reference,other=operator(scalar)
    expected,_=build_volume_source_block(reference,other,ids,ids,order=32,inner_order=24)
    actual,receipt=build_volume_source_block(op,binding,ids,ids,order=12,inner_order=8,
        tensor_face_rule='projected_duffy',tensor_kernel_batch=8192)
    errors=[float((actual[i]-expected[i]).norm()/expected[i].norm()) for i in range(2)]
    print({'lossy_tensor_weak_source_scalar_limit_errors':errors,'physical_qualification':False},flush=True)
    assert max(errors)<.005
    assert receipt['tensor_static_electric_testing']=='complete_rwg_divergence'
    assert receipt['tensor_volume_evaluations']
    for entry in receipt['tensor_volume_evaluations']:
        record=entry['receipt'];check_static_records(record)
        assert not record['static_diagonal_included']
        static=record['weak_static_electric']
        assert static['converged'] and static['integration_policy']['version']=='complex_ac_static_cell_v1'
    assert receipt['dyadic_integration']==op.body_handles[0].tensor_kernel.integration.payload()
