"""Explicit tensor kernel batches through source and complete response routes."""
from contextlib import nullcontext
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom import volume_response_matrix as response
from witwin.maxwell.mom.riesz import LocalRWGRiesz

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_source_kernel_batch_equivalence_forwarding_and_admission():
    op,binding=setup();ids=binding.volume.cells.new_tensor([0])
    values=[]
    for batch in (17,8192):
        value,record=response.build_volume_source_block(op,binding,ids,ids,order=2,inner_order=2,
            tensor_face_rule='projected_duffy',tensor_kernel_batch=batch)
        values.append(value)
        assert record['tensor_kernel_batch']==batch
        assert all(row['receipt']['kernel_batch']==batch for row in record['tensor_volume_evaluations'])
    torch.testing.assert_close(values[0],values[1],rtol=2e-11,atol=1e-12)
    full,receipt=response.build_volume_source_matrix(op,(binding,),order=2,inner_order=2,
        tensor_face_rule='projected_duffy',tensor_kernel_batch=8192)
    n=op.system.bodies[0].basis.n_edges
    torch.testing.assert_close(full[torch.cat((ids,ids+n))],values[1],rtol=2e-11,atol=1e-12)
    assert receipt['tensor_kernel_batch']==8192
    metric=LocalRWGRiesz(op.system,abs(op.exterior.eta));op.riesz_inverse=metric.inverse
    op.read_access=nullcontext;op.refresh=op.validate_tensor_state
    with pytest.raises(response.VolumeMatrixConvergenceError) as info:
        response.adaptive_volume_source_matrix(op,(binding,),max_order=3,max_inner_order=3,
            relative_tolerance=1e-13,absolute_tolerance=1e-20,tensor_kernel_batch=8192)
    receipt=info.value.receipt
    assert not receipt['converged'] and receipt['tensor_kernel_batch']==8192
    assert all(row['receipt']['kernel_batch']==8192 for row in
               receipt['matrix']['bodies'][0]['tensor_source_block']['tensor_volume_evaluations'])
    for bad in (True,0,-1,1.5,None):
        with pytest.raises(ValueError,match='kernel batch'):
            response.build_volume_source_block(op,binding,ids,ids,tensor_kernel_batch=bad)
        with pytest.raises(ValueError,match='kernel batch'):
            response.build_volume_source_matrix(op,(),tensor_kernel_batch=bad)
        with pytest.raises(ValueError,match='kernel batch'):
            response.adaptive_volume_source_matrix(op,(),tensor_kernel_batch=bad)


def test_complete_cell_forwards_batch_to_direct_volume(monkeypatch):
    from witwin.maxwell.mom import fields
    from witwin.maxwell.mom.operator import IntegrationConfig
    op,binding=setup()
    op.integration=IntegrationConfig(order=2,singular_order=2)
    op.body_handles[0].integration=op.integration
    op.matrix=torch.eye(op.shape[0],device=op.device,dtype=op.dtype)
    op.build_rhs=lambda:op.matrix.new_zeros(op.shape[0])
    op.read_access=nullcontext;op.refresh=op.validate_tensor_state
    metric=LocalRWGRiesz(op.system,abs(op.exterior.eta));op.riesz_inverse=metric.inverse
    # Isolate field policy forwarding with manufactured zero surface/source
    # maps. Actual tensor direct-volume integration remains active.
    calls=[]
    def source(*args,**kwargs):
        calls.append(kwargs['tensor_kernel_batch'])
        return op.matrix.new_zeros(op.shape[0],3),{'manufactured_zero_source':True}
    monkeypatch.setattr(response,'adaptive_volume_source_matrix',source)
    def columns(handle,coefficients,points):
        handle.last_column_field_receipt={'manufactured_zero_surface':True}
        value=coefficients.new_zeros(len(points),3,3)
        return value,value
    monkeypatch.setattr(fields,'decode_interior_columns',columns)
    with pytest.raises(response.VolumeMatrixConvergenceError) as info:
        response.prepare_cell_field_response(op,(binding,),tensor_kernel_batch=17,
            query_config=fields.FieldQueryConfig(relative_tolerance=1e-14,absolute_tolerance=1e-20,max_order=4))
    assert calls==[17]
    receipt=info.value.receipt
    assert receipt['failure_reason']=='complete_column_field_order_budget_exhausted'
    assert all(row['volume']['kernel_batch']==17 for row in receipt['tensor_field_evaluations'])
    assert len(receipt['tensor_field_evaluations'])==2
