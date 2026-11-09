"""Unreduced-order separable translations and complete nested adjoint controls."""
import pytest
import torch
from witwin.maxwell.mom.fmm_shift import SeparableMultipoleShift,shift_storage_bytes
from witwin.maxwell.mom.fmm_expansion import shift_matrix

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
@pytest.mark.parametrize('order',(0,3,6,12))
def test_separable_shift_matches_unreduced_dense_and_native_adjoint(order,dtype):
    layout=SeparableMultipoleShift(order,device='cuda',dtype=dtype)
    d=torch.tensor([-.13,.27,.07],device='cuda',dtype=dtype)
    matrix=shift_matrix(d,order).to(layout.dtype)
    x=torch.randn((2,3,layout.count,4),device='cuda',dtype=layout.dtype,requires_grad=True)
    y=torch.randn_like(x)
    tolerance=64*torch.finfo(dtype).eps
    actual=layout.apply(d,x)
    reference=torch.einsum('ab,...bc->...ac',matrix,x)
    torch.testing.assert_close(actual,reference,rtol=tolerance,atol=tolerance)
    adjoint=layout.apply(d,y,adjoint=True)
    torch.testing.assert_close(adjoint,torch.einsum('ab,...ac->...bc',matrix.conj(),y),rtol=tolerance,atol=tolerance)
    gradient,=torch.autograd.grad(actual,x,y)
    torch.testing.assert_close(gradient,adjoint,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(layout.apply(d,x),actual,rtol=0,atol=0)
    assert sum(t.numel()*t.element_size() for t in layout.tensors())==6*layout.count*(order+1)*(16+d.element_size())
    for bad in (d.cpu(),d[0],d*float('nan')):
        with pytest.raises(ValueError):layout.apply(bad,x)
    layout.layouts[0][1].add_(1)
    with pytest.raises(ValueError,match='changed'):layout.apply(d,x)


def test_separable_complete_nested_maps_storage_and_current_gradient(monkeypatch):
    from test_fmm_moments import provider
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    blocks=provider();lower,upper=blocks.support_boxes()
    tree=build_support_octree(lower,upper,leaf_size=8)
    assert int(tree.depth.max())>=2
    dense=NestedRWGMultipoles(blocks,tree,order=6)
    compact=NestedRWGMultipoles(blocks,tree,order=6,shift_storage='separable_v1')
    x=torch.randn((2,tree.size),device=tree.device,dtype=torch.complex128,requires_grad=True)
    actual,reference=compact.upward(x),dense.upward(x)
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-13)
    coefficients=tuple(torch.randn_like(value) for value in actual)
    torch.testing.assert_close(compact.downward(coefficients),dense.downward(coefficients),rtol=2e-12,atol=1e-13)
    pairing=sum((coefficient.conj()*value).sum().real for coefficient,value in zip(coefficients,actual))
    gradient,=torch.autograd.grad(pairing,x)
    torch.testing.assert_close(gradient,compact.downward(coefficients),rtol=2e-12,atol=1e-13)
    receipt=compact.receipt()
    assert receipt['order']==dense.order and receipt['owned_retained_bytes']<dense.receipt()['owned_retained_bytes']
    assert receipt['owned_retained_bytes']==sum(t.numel()*t.element_size() for t in compact.tensors())
    print(dict(dense_owned_bytes=dense.receipt()['owned_retained_bytes'],
        separable_owned_bytes=receipt['owned_retained_bytes'],order=6,nodes=tree.node_count),flush=True)
    import witwin.maxwell.mom.fmm_moments as module
    def forbidden(*args,**kwargs):raise AssertionError('Capacity rejection must precede numerical setup')
    monkeypatch.setattr(module,'rwg_polynomial_moments',forbidden)
    monkeypatch.setattr(module,'SeparableMultipoleShift',forbidden)
    with pytest.raises(MemoryError,match='before moments'):
        NestedRWGMultipoles(blocks,tree,order=6,shift_storage='separable_v1',retained_budget_bytes=1)
    compact.transfers[next(iter(compact.transfers))].add_(1)
    with pytest.raises(ValueError,match='changed'):compact.upward(x)
