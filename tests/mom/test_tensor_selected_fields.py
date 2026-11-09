"""Selected tensor RWG fields preserve supports, order and interior signs."""
from dataclasses import replace
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom.fields import selected_interior_field_blocks,decode_interior_columns

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_tensor_selected_support_columns_and_admission(monkeypatch):
    op,binding=setup();handle=op.body_handles[0];basis=handle.basis
    handle.integration=replace(handle.integration,order=3,singular_order=3)
    points=binding.volume.vertices.new_tensor([[.013,.017,.011],[.018,.012,.014]])
    ids=basis.face_edges.new_tensor([2,0,2]);n=basis.n_edges
    identity=torch.eye(2*n,device=points.device,dtype=torch.complex128)
    expected=decode_interior_columns(handle,identity,points)
    actual=selected_interior_field_blocks(handle,points,ids)
    columns=torch.cat((ids,ids+n))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b[...,columns],rtol=2e-12,atol=1e-10)
    assert handle.last_column_field_receipt['selected_edges']==3
    assert handle.last_column_field_receipt['unique_edges']==2
    calls=[];kernel=handle.tensor_kernel;original=type(kernel).evaluate_regular
    def measured(self,delta):
        if self is kernel:calls.append(len(delta))
        return original(self,delta)
    monkeypatch.setattr(type(kernel),'evaluate_regular',measured)
    one=ids.new_tensor([0,0])
    selected_interior_field_blocks(handle,points,one)
    selected_nodes=sum(calls);calls.clear()
    assert handle.last_column_field_receipt['support_panels']==2
    decode_interior_columns(handle,identity,points)
    assert 0<selected_nodes<sum(calls)
    for queries,edges,shape in ((points,ids[:0],(2,3,0)),(points[:0],ids,(0,3,6))):
        assert all(v.shape==shape for v in selected_interior_field_blocks(handle,queries,edges))
    with pytest.raises(ValueError,match='requested region'):
        selected_interior_field_blocks(handle,points.new_tensor([[.3,.2,.4]]),ids)
    with pytest.raises(ValueError):
        selected_interior_field_blocks(handle,basis.mesh.triangles[:1].mean(1),ids)
    with pytest.raises(ValueError,match='indices'):
        selected_interior_field_blocks(handle,points,ids.new_tensor([-1]))
    handle.tensor_kernel=None
    with pytest.raises(ValueError,match='owning regional'):
        selected_interior_field_blocks(handle,points,ids)
