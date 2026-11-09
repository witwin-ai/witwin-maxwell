"""Selected surface field columns retain complete supports and interior signs."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom import fields
from witwin.maxwell.mom.mesh import RWGBasis,sphere_mesh
from witwin.maxwell.mom.runtime import DecodeHandle
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def handle(dtype=torch.float64):
    basis=RWGBasis(sphere_mesh(.1,0,dtype=dtype))
    return DecodeHandle(basis,compile_medium(mw.Material(),3e8),compile_medium(mw.Material(eps_r=2.),3e8),(),IntegrationConfig(order=4,singular_order=4),{})


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_selected_columns_original_current_and_query_gradients(dtype,monkeypatch):
    h=handle(dtype);n=h.basis.n_edges
    ids=torch.tensor([5,1,5],device='cuda')
    points=torch.tensor([[.01,.017,.023],[-.019,.014,.007]],device='cuda',dtype=dtype,requires_grad=True)
    coefficients=torch.zeros(2*n,2*len(ids),device='cuda',dtype=torch.complex64 if dtype==torch.float32 else torch.complex128)
    coefficients[torch.cat((ids,ids+n)),torch.arange(2*len(ids),device='cuda')]=1
    expected=fields.decode_interior_columns(h,coefficients,points)
    actual=fields.selected_interior_field_blocks(h,points,ids)
    tolerance=3e-5 if dtype==torch.float32 else 2e-12
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
    gradients=[torch.autograd.grad(sum(v.real.square().sum() for v in result),points)[0] for result in (actual,expected)]
    torch.testing.assert_close(*gradients,rtol=2*tolerance,atol=2*tolerance)
    current=torch.randn(2*len(ids),device='cuda',dtype=coefficients.dtype,requires_grad=True)
    result=tuple(v@current for v in actual)
    reference=fields.decode_interior_columns(h,(coefficients@current)[:,None],points)
    for a,b in zip(result,reference):torch.testing.assert_close(a,b[...,0],rtol=2*tolerance,atol=2*tolerance)
    current_gradients=[torch.autograd.grad(sum(v.abs().square().sum() for v in values),current)[0]
                       for values in (result,reference)]
    torch.testing.assert_close(*current_gradients,rtol=3*tolerance,atol=3*tolerance)
    seen=[]
    for name in ('triangle_rule','duffy_fan'):
        original=getattr(fields,name)
        def capture(*args,_original=original,_name=name,**kwargs):
            triangles=args[0 if _name=='triangle_rule' else 1]
            seen.append(len(triangles));return _original(*args,**kwargs)
        monkeypatch.setattr(fields,name,capture)
    fields.selected_interior_field_blocks(h,points,ids[:1])
    assert sum(seen)==2*len(points)


def test_selected_field_empty_invalid_and_trace_guards():
    h=handle();ids=torch.tensor([0],device='cuda');points=h.basis.mesh.vertices.new_zeros(1,3)
    assert all(v.shape==(1,3,0) for v in fields.selected_interior_field_blocks(h,points,ids[:0]))
    assert all(v.shape==(0,3,2) for v in fields.selected_interior_field_blocks(h,points[:0],ids))
    for bad in (ids.cpu(),ids.float(),ids[:,None],ids-1,ids+h.basis.n_edges):
        with pytest.raises(ValueError,match='indices'):
            fields.selected_interior_field_blocks(h,points,bad)
    with pytest.raises(ValueError,match='region'):
        fields.selected_interior_field_blocks(h,points+1,ids)
    with pytest.raises(ValueError,match='trace'):
        fields.selected_interior_field_blocks(h,h.basis.mesh.triangles[:1].mean(1),ids)
