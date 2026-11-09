"""Manufactured extreme-scale acceptance controls, not physical qualification."""
from contextlib import nullcontext
from dataclasses import replace
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom import volume_response_matrix as response
from witwin.maxwell.mom.volume_response_matrix import adaptive_volume_source_matrix
from witwin.maxwell.mom import fields,lifting_fields
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def metric_fixture():
    op,binding=setup(False)
    op.integration=IntegrationConfig(order=2,singular_order=2)
    op.body_handles[0].integration=op.integration
    op.matrix=torch.eye(op.shape[0],device=op.device,dtype=op.dtype)
    # A manufactured identity test metric isolates arithmetic from geometry.
    op.riesz_inverse=lambda value:value
    op.read_access=nullcontext
    op.refresh=op.system.assert_current
    op.build_rhs=lambda:op.matrix.new_zeros(op.shape[0])
    return op,binding


def test_source_dual_norm_scale_and_nonfinite_threshold(monkeypatch):
    op,binding=metric_fixture()
    matrix=op.matrix.new_full((op.shape[0],3),1e200)
    monkeypatch.setattr(response,'build_volume_source_matrix',lambda *a,**k:(matrix,{}))
    actual,receipt=response.adaptive_volume_source_matrix(op,(binding,),max_order=4,max_inner_order=4)
    assert receipt['converged'] and torch.equal(actual,matrix)
    with pytest.raises(response.VolumeMatrixConvergenceError) as info:
        response.adaptive_volume_source_matrix(op,(binding,),max_order=4,max_inner_order=4,
                                              relative_tolerance=1e200)
    assert info.value.receipt['failure_reason']=='nonfinite_source_error_metric'
    assert not info.value.receipt['converged']
    matrix.fill_(complex(float('nan'),0))
    with pytest.raises(response.VolumeMatrixConvergenceError,match='nonfinite_source_error_metric'):
        response.adaptive_volume_source_matrix(op,(binding,),max_order=4,max_inner_order=4)


def prepare_fixture(monkeypatch):
    op,binding=metric_fixture()
    source=op.matrix.new_zeros(op.shape[0],3)
    monkeypatch.setattr(response,'adaptive_volume_source_matrix',lambda *a,**k:(source,{'manufactured':True}))
    def columns(handle,coefficients,points):
        value=1e200 if handle.integration.order==2 else 1.1e200
        e=op.matrix.new_full((len(points),3,3),value)
        return e,torch.zeros_like(e)
    monkeypatch.setattr(fields,'decode_interior_columns',columns)
    monkeypatch.setattr(response,'cell_current_field_blocks',lambda *a,**k:
                        (source.new_zeros(1,3,1,3),source.new_zeros(1,3,1,3)))
    def background(*args,**kwargs):
        op.last_field_query_log={'manufactured':True}
        return source.new_zeros(1,3),source.new_zeros(1,3)
    monkeypatch.setattr(lifting_fields,'decode_system',background)
    return op,binding,source


def test_complete_column_stable_metric_and_threshold_rejection(monkeypatch):
    op,binding,_=prepare_fixture(monkeypatch)
    config=FieldQueryConfig(relative_tolerance=.01,absolute_tolerance=0,max_order=4)
    with pytest.raises(response.VolumeMatrixConvergenceError) as info:
        response.prepare_cell_field_response(op,(binding,),query_config=config)
    receipt=info.value.receipt
    assert receipt['failure_reason']=='complete_column_field_order_budget_exhausted'
    assert not receipt['checks'][0]['E']['passed']
    assert receipt['checks'][0]['E']['maximum_column_query_difference']==pytest.approx(3**.5*1e199)
    passed=response.prepare_cell_field_response(op,(binding,),query_config=replace(config,relative_tolerance=.2))
    assert passed['receipt']['converged'] and not passed['receipt']['physical_qualification']
    with pytest.raises(response.VolumeMatrixConvergenceError) as info:
        response.prepare_cell_field_response(op,(binding,),query_config=replace(config,relative_tolerance=1e200))
    assert info.value.receipt['failure_reason']=='nonfinite_complete_column_error_metric'
    assert info.value.receipt['failed_field']=='E'


def test_nonfinite_solve_residual_and_configured_threshold_rejected(monkeypatch):
    op,binding,source=prepare_fixture(monkeypatch)
    source.fill_(1e200)
    with pytest.raises(RuntimeError,match='configured dual residual'):
        response.prepare_cell_field_response(op,(binding,),stopping=StoppingRule(method='direct',rtol=1e200))
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',lambda matrix,rhs:torch.full_like(rhs,complex(float('nan'),0)))
        with pytest.raises(RuntimeError,match='configured dual residual'):
            response.prepare_cell_field_response(op,(binding,))


@pytest.mark.parametrize('inverse_value',[-float('inf'),-1e308])
def test_riesz_nonfinite_output_or_quadratic_rejected_before_clamp(monkeypatch,inverse_value):
    op,binding,source=prepare_fixture(monkeypatch)
    op.riesz_inverse=lambda value:torch.full_like(value,inverse_value)
    values=op.matrix.new_ones(op.shape[0],3)
    assert torch.isnan(response._column_dual_norm(op,values)).all()
    # Both nonfinite inverse output and overflow from finite inverse entries
    # must be rejected. Zero-valued differences cannot hide the magnitude.
    source.fill_(1)
    with pytest.raises(RuntimeError,match='configured dual residual'):
        response.prepare_cell_field_response(op,(binding,))
    # Restore the real adaptive entry after the preparation fixture patch.
    with monkeypatch.context() as patch:
        patch.setattr(response,'build_volume_source_matrix',lambda *a,**k:(values*k['inner_order'],{}))
        with pytest.raises(response.VolumeMatrixConvergenceError,match='nonfinite_source_error_metric'):
            adaptive_volume_source_matrix(op,(binding,),max_order=4,max_inner_order=4)
