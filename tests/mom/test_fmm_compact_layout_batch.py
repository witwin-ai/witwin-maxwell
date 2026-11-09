"""Independent retained layout ownership with shared construction work."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
from witwin.maxwell.mom.fmm_capacity import storage_bytes
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
from witwin.maxwell.mom.physics import compile_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
def test_compact_layout_single_construction_actual_action_and_independent_mutation(dtype,monkeypatch):
    import witwin.maxwell.mom.fmm_compact as module
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,sigma_m=.002),3e8)
    d=torch.tensor([[1.+i*.01,.2,-.1] for i in range(65)],device='cuda',dtype=dtype)
    order=6;references={i:CompactElectromagneticM2L(d[i],medium,order,row_batch=11) for i in (0,31,32,64)}
    native=module._coefficient_layout;calls=[]
    def observed(*args,**kwargs):calls.append(args);return native(*args,**kwargs)
    monkeypatch.setattr(module,'_coefficient_layout',observed)
    objects=CompactElectromagneticM2L.create_many(d,medium,order,row_batch=11)
    assert len(calls)==1 and len(objects)==65
    assert storage_bytes(t for obj in objects for t in obj.tensors())==65*m2l_storage_bytes(order,d.element_size(),'compact')
    for channel in range(3):
        tensors=[obj.tensors()[channel] for obj in objects]
        assert len({t.untyped_storage().data_ptr() for t in tensors})==65
        assert all(t._base is None and t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
    tol=6e-5 if dtype==torch.float32 else 3e-13
    for index,ref in references.items():
        obj=objects[index]
        assert torch.equal(obj.indices,ref.indices) and torch.equal(obj.binomial,ref.binomial)
        torch.testing.assert_close(obj.coefficients,ref.coefficients,rtol=tol,atol=0)
        x=torch.randn((2,2,obj.count,4),device=d.device,dtype=obj.dtype,requires_grad=True);y=torch.randn_like(x)
        torch.testing.assert_close(obj.apply(x),ref.apply(x),rtol=tol,atol=tol)
        grad,=torch.autograd.grad(obj.apply(x),x,y)
        torch.testing.assert_close(grad,obj.adjoint(y),rtol=tol,atol=tol)
    for index,channel in enumerate((0,1,2)):
        objects[index].tensors()[channel].add_(1)
        with pytest.raises(ValueError,match='changed'):objects[index].assert_current()
    for obj in objects[3:]:obj.assert_current()
    # No hidden persistent shared cache: a second call builds its own layout.
    other=CompactElectromagneticM2L.create_many(d[:1],medium,order,row_batch=11)
    assert len(calls)==2
    assert all(a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr()
        for a,b in zip(objects[-1].tensors(),other[0].tensors(),strict=True))


def test_compact_layout_capacity_and_empty_gates_precede_shared_layout(monkeypatch):
    import witwin.maxwell.mom.fmm_compact as module
    medium=compile_medium(mw.Material(),3e8)
    d=torch.tensor([[1.,.2,-.1]]*65,device='cuda',dtype=torch.float64)
    def forbidden(*args,**kwargs):raise AssertionError('Rejected batch built shared layout or coefficients')
    monkeypatch.setattr(module,'_coefficient_layout',forbidden)
    monkeypatch.setattr(module,'helmholtz_taylor',forbidden)
    with pytest.raises(MemoryError,match='before coefficient'):
        CompactElectromagneticM2L.create_many(d,medium,6,retained_budget_bytes=1)
    assert CompactElectromagneticM2L.create_many(d[:0],medium,6)==()
    for params in ({'order':True},{'row_batch':0},{'retained_budget_bytes':False}):
        with pytest.raises(ValueError):CompactElectromagneticM2L.create_many(d,medium,**{'order':6,**params})
    for value in (d.cpu(),d*torch.nan,d.clone().requires_grad_()):
        with pytest.raises(ValueError):CompactElectromagneticM2L.create_many(value,medium,6)
