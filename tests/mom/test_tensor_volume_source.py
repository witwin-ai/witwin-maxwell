"""Tensor weak source rows: independent scalar weak form and ownership."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.runtime import SystemDecodeHandle,MoMConfig
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh,bind_volume_region
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_block,build_volume_source_matrix

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(tensor=True,anisotropic=False):
    v=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume=TetrahedralVolumeMesh(v,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    scalar=compile_medium(mw.Material(eps_r=2.,mu_r=1.3),3e7)
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    inside=TensorMedium(3e7,2*eye,1.3*eye) if tensor else scalar
    if anisotropic:
        a=torch.diag(v.new_tensor([1.2,1.1,1.05])).to(eye.dtype)
        inside=TensorMedium(3e7,a,a)
    system=SurfaceSystem.create([(0,'volume',RWGBasis(volume.boundary),inside)])
    binding=bind_volume_region(volume,system)
    spectral=DyadicIntegrationConfig(radial_map='matched_spd_branch_v1',relative_tolerance=2e-4,
        max_radial_order=1024,max_azimuthal_order=256) if anisotropic else DyadicIntegrationConfig()
    config=MoMConfig(dyadic_integration=spectral if tensor else None)
    handle=SystemDecodeHandle(system,compile_medium(mw.Material(),3e7),(),config,{})
    handle.device=v.device;handle.dtype=torch.complex128
    handle.shape=(2*system.bodies[0].basis.n_edges,)*2
    return handle,binding


def test_scalar_limit_weak_source_refinement_and_selected_ownership():
    tensor,binding=setup();scalar,reference_binding=setup(False)
    ids=binding.volume.cells.new_tensor([0]);cells=ids.clone()
    reference,_=build_volume_source_block(scalar,reference_binding,ids,cells,order=32,inner_order=24)
    errors=[]
    for order in (4,8,12):
        actual,receipt=build_volume_source_block(tensor,binding,ids,cells,order=order,inner_order=6)
        errors.append([float((actual[i]-reference[i]).norm()/reference[i].norm()) for i in range(2)])
    print({'tensor_weak_scalar_limit_errors':errors,'physical_qualification':False})
    assert max(errors[-1])<.005
    assert receipt['hessian_evaluated'] and len(receipt['tensor_volume_evaluations'])==2
    assert receipt['tensor_static_electric_testing']=='complete_rwg_divergence'
    assert all(not x['receipt']['static_diagonal_included'] for x in receipt['tensor_volume_evaluations'])
    assert receipt['dyadic_integration']==tensor.body_handles[0].tensor_kernel.integration.payload()
    selected,_=build_volume_source_block(tensor,binding,ids.repeat(2),cells.repeat(2),order=4,inner_order=3)
    single,_=build_volume_source_block(tensor,binding,ids,cells,order=4,inner_order=3)
    torch.testing.assert_close(selected,single.repeat_interleave(2,0).repeat(1,2),rtol=2e-12,atol=1e-13)
    # A selected block contains exactly the owning rows of the full weak map.
    full,record=build_volume_source_matrix(tensor,(binding,),order=4,inner_order=3)
    n=tensor.system.bodies[0].basis.n_edges
    torch.testing.assert_close(single,full[torch.cat((ids,ids+n))],rtol=2e-12,atol=1e-13)
    assert record['hessian_evaluated'] and 'tensor_source_block' in record['bodies'][0]
    with pytest.raises(ValueError,match='owning-system'):
        build_volume_source_block(tensor,reference_binding,ids,cells)
    kernel=tensor.body_handles[0].tensor_kernel
    tensor.body_handles[0].tensor_kernel=SimpleNamespace(medium=kernel.medium)
    with pytest.raises(ValueError,match='binding changed'):
        build_volume_source_block(tensor,binding,ids,cells)


def test_projected_rule_weak_selected_full_and_adaptive_forwarding():
    from contextlib import nullcontext
    from witwin.maxwell.mom.riesz import LocalRWGRiesz
    from witwin.maxwell.mom.volume_response_matrix import adaptive_volume_source_matrix,VolumeMatrixConvergenceError
    op,binding=setup();ids=binding.volume.cells.new_tensor([0])
    # Source assembly needs no system matrix. Supply the real regional RWG
    # metric for the adaptive column estimator on this fixed decode fixture.
    metric=LocalRWGRiesz(op.system,abs(op.exterior.eta))
    op.read_access=nullcontext
    op.riesz_inverse=metric.inverse
    op.refresh=op.validate_tensor_state
    for rule in ('triangle','projected_duffy','distance_scaled_duffy'):
        block,record=build_volume_source_block(op,binding,ids,ids,order=2,inner_order=2,tensor_face_rule=rule)
        full,receipt=build_volume_source_matrix(op,(binding,),order=2,inner_order=2,tensor_face_rule=rule)
        n=op.system.bodies[0].basis.n_edges
        torch.testing.assert_close(block,full[torch.cat((ids,ids+n))],rtol=2e-12,atol=1e-12)
        assert all(x['receipt']['face_rule']==rule for x in record['tensor_volume_evaluations'])
        assert all(x['receipt']['face_rule']==rule
                   for x in receipt['bodies'][0]['tensor_source_block']['tensor_volume_evaluations'])
    # A real order-exhaustion receipt retains the selected rule rather than
    # silently refining the default policy. This is not a convergence claim.
    with pytest.raises(VolumeMatrixConvergenceError) as info:
        adaptive_volume_source_matrix(op,(binding,),order=2,inner_order=2,max_order=3,max_inner_order=3,
            relative_tolerance=1e-13,absolute_tolerance=1e-20,tensor_face_rule='projected_duffy')
    assert info.value.receipt['tensor_face_rule']=='projected_duffy'
    assert not info.value.receipt['converged']
    assert all(x['receipt']['face_rule']=='projected_duffy' for x in
               info.value.receipt['matrix']['bodies'][0]['tensor_source_block']['tensor_volume_evaluations'])
    with pytest.raises(ValueError,match='face rule'):
        build_volume_source_matrix(op,(),tensor_face_rule='invalid')
