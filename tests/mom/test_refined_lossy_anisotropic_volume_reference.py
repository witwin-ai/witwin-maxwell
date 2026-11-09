"""Lossy non-scalar tensor source volume against a real-coordinate reference."""
import json
import math
import os
from pathlib import Path
import pytest
import torch
from witwin.maxwell.mom.tensor_medium import TensorMedium,scalar_medium_if_isotropic
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel,DyadicIntegrationConfig,isotropic_dyadic_green
from witwin.maxwell.mom.tensor_volume import cell_tensor_current_blocks
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.vie_operator import cell_current_field_blocks

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def errors(actual,expected):
    result=[]
    for row,column in ((slice(0,3),slice(0,3)),(slice(3,6),slice(0,3)),
                       (slice(0,3),slice(3,6)),(slice(3,6),slice(3,6))):
        value,target=actual[...,row,column],expected[...,row,column]
        value=value.reshape(-1,9);target=target.reshape(-1,9)
        scale=target.abs().amax(-1,keepdim=True)
        result.extend(float(error) for error in (((value-target)/scale).norm(dim=-1)/(target/scale).norm(dim=-1)))
    assert all(math.isfinite(value) for value in result)
    return result


def test_refined_lossy_anisotropic_volume_transform_reference(tmp_path):
    metric=torch.tensor([[1.2,.12,.05],[.12,1.1,.07],[.05,.07,1.05]],device='cuda',dtype=torch.float64)
    eye=torch.eye(3,device=metric.device,dtype=torch.complex128)
    electric,magnetic=2+.3j,1.3+.2j
    tensor=TensorMedium(3e7,electric*metric,magnetic*metric)
    isotropic=TensorMedium(3e7,electric*eye,magnetic*eye)
    transform=torch.linalg.cholesky(metric)/torch.linalg.det(metric).sqrt()
    inverse=torch.linalg.inv(transform)
    dual=torch.block_diag(inverse,inverse).to(eye.dtype)
    config=DyadicIntegrationConfig(relative_tolerance=2e-4,absolute_tolerance=1e-9,
        max_radial_order=1024,max_azimuthal_order=256)
    kernel=DyadicGreenKernel(tensor,config)
    record=dict(physical_qualification=False,reference='real_coordinate_transformed_lossy_scalar',
        scope='original_decimal_metric_lossy_tensor_volume_refinement',
        error_layout='block_then_query_cell',
        dyadic_integration=config.payload(),phase='volume_pending')
    report=Path(os.environ.get('MAGNETIC_CONTROL_REPORT',tmp_path/'reference.json'))
    def save():report.write_text(json.dumps(record,indent=2))
    save()
    try:
        record['point_reference_evidence']='/tmp/gsp-lossy-anisotropic-volume-v1/physical.json'
        record['phase']='volume_static_and_dynamic';save()
        vertices=.07*metric.new_tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        cells=torch.tensor([[0,1,2,3]],device=metric.device)
        mesh=TetrahedralVolumeMesh(vertices,cells,region_id=1)
        transformed=TetrahedralVolumeMesh(vertices@inverse.T,cells,region_id=1)
        queries=metric.new_tensor([[.10,.07,.11],[.015,.012,.018]])
        actual,receipt=cell_tensor_current_blocks(mesh,queries,kernel,order=8,radial_order=8,
            face_rule='distance_scaled_duffy',kernel_batch=8192)
        reference=scalar_medium_if_isotropic(isotropic)
        e,h=cell_current_field_blocks(transformed,queries@inverse.T,reference,order=40)
        e,h=e.permute(0,2,1,3),h.permute(0,2,1,3)
        scalar=torch.cat((torch.cat((e,-h),-1),torch.cat((h,e*reference.epsilon/reference.mu),-1)),-2)
        expected=torch.linalg.det(transform)*dual.T@scalar@dual
        torch.save({'actual':actual,'reference':expected,'epsilon_r':tensor.epsilon_r,'mu_r':tensor.mu_r,
                    'vertices':vertices,'cells':cells,'queries':queries},tmp_path/'volume-blocks.pt')
        record.update(volume_errors=errors(actual,expected),volume_receipt=receipt,phase='blocks_computed')
        save();assert max(record['volume_errors'])<.005
        assert all(item['receipt']['converged'] for item in receipt['static_cell_integrations'])
        assert all(item['converged'] for item in receipt['spectral_evaluations'])
        record['phase']='passed';save()
    except Exception as failure:
        record.update(failure_type=type(failure).__name__,failure_message=str(failure))
        if hasattr(failure,'receipt'):record['failure_receipt']=failure.receipt
        save()
        raise
