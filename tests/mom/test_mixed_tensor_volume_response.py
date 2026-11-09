"""Actual scalar-source volume response through a mixed tensor surface system."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh, bind_volume_region
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.volume_response_matrix import prepare_cell_field_response, build_volume_source_matrix

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_scalar_volume_current_in_actual_mixed_tensor_system():
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    first=TetrahedralVolumeMesh(vertices,cells,region_id=1)
    second=TetrahedralVolumeMesh(vertices+vertices.new_tensor([.2,0.,0.]),cells,region_id=2)
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    # Exact scalar-limit tensor retains the real generalized assembly/decoder;
    # this isolates mixed ownership without claiming anisotropic convergence.
    tensor=TensorMedium(3e8,2*eye,1.3*eye)
    scalar=compile_medium(mw.Material(eps_r=2.),3e8)
    exterior=compile_medium(mw.Material(),3e8)
    system=SurfaceSystem.create([(0,'linear_tensor',RWGBasis(first.boundary),tensor),
                                 (1,'scalar_volume',RWGBasis(second.boundary),scalar)])
    config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),
                        dyadic_integration=DyadicIntegrationConfig(),riesz_storage='compact_rwg_v1')
    operator=SystemOperator(system,exterior,[],config)
    binding=bind_volume_region(second,system)
    prepared=prepare_cell_field_response(operator,(binding,),
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    assert prepared['receipt']['converged'] and not prepared['receipt']['physical_qualification']
    assert torch.count_nonzero(prepared['background_surface'])==0
    assert torch.count_nonzero(prepared['background_electric'])==0
    source=prepared['receipt']['source_matrix']['matrix']
    rhs,_=build_volume_source_matrix(operator,(binding,),order=source['surface_testing_order'],inner_order=source['volume_order'])
    for rows in (system.J_slices[0],system.M_slices[0]):
        assert torch.count_nonzero(rhs[rows])==0
        assert bool(prepared['surface_response'][rows].abs().max()>0)
    current=eye.new_tensor([1.+.2j,-.3+.1j,.4-.1j])
    direct=torch.linalg.solve(operator.matrix,rhs@current)
    torch.testing.assert_close(prepared['surface_response']@current,direct,rtol=2e-11,atol=1e-12)
    from witwin.maxwell.mom.vie_operator import cell_current_field_blocks
    order=prepared['receipt']['final_order']
    from dataclasses import replace
    import copy
    local=copy.copy(operator.body_handles[1])
    local.integration=replace(local.integration,order=order,singular_order=order,
                              near_max_order=max(local.integration.near_max_order,order+2))
    from witwin.maxwell.mom.fields import decode_interior_columns
    coefficients=torch.cat((direct[system.J_slices[1]],direct[system.M_slices[1]]))[:,None]
    e,h=decode_interior_columns(local,coefficients,prepared['points'])
    ve,vh=cell_current_field_blocks(second,prepared['points'],scalar,order=order)
    for name,surface,volume in (('electric',e,ve),('magnetic',h,vh)):
        expected=surface.flatten()+volume.reshape(3,3)@current
        torch.testing.assert_close(prepared[name+'_response']@current,expected,rtol=2e-11,atol=1e-12)
